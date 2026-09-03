"""
TIM (Thinking Initiation Machine) — vocabulary-lens KV-cache warm priming.

Operates on an already-loaded HF `model` + `tokenizer`; there is deliberately
no backbone abstraction, because every experiment in this repo drives the same
two objects.

Stage 1: sample `noise_length` token IDs from the vocabulary. With a domain
         pool, 70% are drawn from it and 30% uniform random; without one,
         100% uniform random. IDs are used directly and never
         decode-then-re-tokenized — BPE does not round-trip losslessly, so
         doing so would silently change both content and length.
Stage 2: (`num_passes` > 1) iterative proto-thought accumulation. Each pass
         after the first samples `think_tokens` tokens one at a time from the
         model's own distribution given the current cache, appending each, so
         the cache genuinely grows across passes. `chain_mode="reseed"`
         injects a fresh noise chunk at the start of every pass;
         `"persistent"` just continues the chain.
Stage 3: (optional) prefill persona/domain text on top of the noise cache.

`prime()` returns a `past_key_values` object usable directly as
`model.generate(past_key_values=..., ...)`. Note that `generate()` mutates
that object in place, so callers must `copy.deepcopy()` it per task — see
`tim.generation.timed_generate`, which does this and times it.
"""

import time

import torch


class TIMPrimer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        noise_length: int = 64,
        num_passes: int = 1,
        think_tokens: int = 32,
        chain_mode: str = "persistent",   # "persistent" | "reseed"
        prime_temperature: float = 0.8,   # 0 => greedy proto-thoughts
        prime_top_k: int = 50,
        persona_max_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.noise_length = noise_length
        self.num_passes = num_passes
        self.think_tokens = think_tokens
        if chain_mode not in ("persistent", "reseed"):
            raise ValueError(f"chain_mode must be 'persistent' or 'reseed', got {chain_mode!r}")
        self.chain_mode = chain_mode
        self.prime_temperature = prime_temperature
        self.prime_top_k = prime_top_k
        self.persona_max_length = persona_max_length

        # Exclude real special tokens (chat-template markers, <think>,
        # <|im_end|>, etc.) from the noise pool rather than guessing a
        # fixed numeric offset — special tokens are typically appended
        # near the END of the vocab for Qwen3, not within a few IDs of 0.
        self._excluded_ids = set(tokenizer.all_special_ids or [])
        self.vocab_size = len(tokenizer)

    # ------------------------------------------------------------------ #
    # Vocabulary-lens noise
    # ------------------------------------------------------------------ #

    def _sample_excluding_special(self, n: int) -> torch.Tensor:
        """Sample n token IDs uniformly, rejecting special tokens."""
        out = []
        while len(out) < n:
            batch = torch.randint(0, self.vocab_size, (n - len(out),), device=self.device)
            for tid in batch.tolist():
                if tid not in self._excluded_ids:
                    out.append(tid)
        return torch.tensor(out, device=self.device, dtype=torch.long)

    def get_domain_token_ids(self, domain: str) -> list[int]:
        """Tokenize the domain string (and its whitespace-split words) into
        a small pool of real token IDs to seed domain-weighted noise."""
        words = [domain] + domain.split()
        ids: list[int] = []
        for w in words:
            ids.extend(self.tokenizer.encode(w, add_special_tokens=False))
        seen: set = set()
        unique = [x for x in ids if x not in self._excluded_ids and not (x in seen or seen.add(x))]
        return unique if unique else [self.tokenizer.eos_token_id]

    def get_domain_token_ids_from_words(self, words: list[str]) -> list[int]:
        """
        Build a domain token pool from a curated list of individual
        words/symbols (e.g. real Python keywords, builtins, dunder methods)
        rather than tokenizing one descriptive sentence. Each word is
        tokenized independently so short syntax tokens (def, self, ==,
        __init__) aren't diluted or merged away by surrounding context the
        way they would be if concatenated into a sentence first.
        """
        ids: list[int] = []
        for w in words:
            ids.extend(self.tokenizer.encode(w, add_special_tokens=False))
        seen: set = set()
        unique = [x for x in ids if x not in self._excluded_ids and not (x in seen or seen.add(x))]
        return unique if unique else [self.tokenizer.eos_token_id]

    def generate_vocabulary_noise(
        self,
        domain_tokens: list[int] | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Returns: [1, noise_length] LongTensor of raw token IDs (never
        decoded/re-encoded — kept as IDs the whole way through).
        """
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        else:
            gen = None

        n = self.noise_length

        if not domain_tokens:
            if gen is not None:
                ids = torch.randint(0, self.vocab_size, (n,), generator=gen, device=self.device)
                # simple rejection pass for special tokens under a fixed seed
                mask = torch.tensor([t.item() in self._excluded_ids for t in ids])
                while mask.any():
                    resample = torch.randint(0, self.vocab_size, (int(mask.sum()),), generator=gen, device=self.device)
                    ids[mask] = resample
                    mask = torch.tensor([t.item() in self._excluded_ids for t in ids])
            else:
                ids = self._sample_excluding_special(n)
        else:
            n_domain = int(n * 0.70)
            n_random = n - n_domain
            dt = torch.tensor(domain_tokens, device=self.device, dtype=torch.long)
            if gen is not None:
                domain_idx = torch.randint(0, len(dt), (n_domain,), generator=gen, device=self.device)
                domain_sample = dt[domain_idx]
                random_sample = torch.randint(0, self.vocab_size, (n_random,), generator=gen, device=self.device)
            else:
                domain_sample = dt[torch.randint(0, len(dt), (n_domain,), device=self.device)]
                random_sample = self._sample_excluding_special(n_random)
            all_ids = torch.cat([domain_sample, random_sample])
            perm = torch.randperm(n, generator=gen, device=self.device) if gen is not None \
                else torch.randperm(n, device=self.device)
            ids = all_ids[perm]

        return ids.long().unsqueeze(0)  # [1, noise_length]

    # ------------------------------------------------------------------ #
    # Forward pass helper
    # ------------------------------------------------------------------ #

    def _forward_with_kv(self, input_ids: torch.Tensor, past_key_values=None):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return out.logits, out.past_key_values

    # ------------------------------------------------------------------ #
    # Main public API
    # ------------------------------------------------------------------ #

    def _pick_next_token(
        self,
        logits: torch.Tensor,
        gen: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Choose the next token from the model's own predicted distribution.

        prime_temperature == 0 -> greedy (argmax), fully deterministic.
        prime_temperature  > 0 -> sample from the (optionally top-k truncated)
                                  softmax. This is the "proto-thought" step:
                                  the randomness comes from sampling the
                                  model's OWN output distribution given the
                                  current KV state, not from injecting fresh
                                  external noise.
        """
        logits = logits[:, -1, :].float()

        if self.prime_temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)  # [1, 1]

        logits = logits / self.prime_temperature

        if self.prime_top_k and self.prime_top_k > 0:
            k = min(self.prime_top_k, logits.shape[-1])
            topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=gen)
            return topk_idx.gather(-1, choice)  # [1, 1]

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=gen)

    def prime(
        self,
        domain_tokens: list[int] | None = None,
        seed: int | None = None,
        persona_text: str | None = None,
        collect_timing: bool = False,
    ):
        """
        Build and return a warm KV cache (past_key_values).

        Pass structure (genuinely iterative — each pass is a separate,
        observable step, unlike the previous implementation which collapsed
        every pass into one generate() call and made num_passes a no-op):

          Pass 1        : prefill the vocabulary-lens noise -> initial cache
          Pass 2..N     : sample `think_tokens` proto-thought tokens one at a
                          time from the CURRENT cache state, appending each to
                          the cache (so the cache genuinely accumulates across
                          passes). Under chain_mode="reseed" a fresh noise
                          chunk is injected at the start of every pass before
                          the thinking tokens, which is the "increasing noise"
                          variant; under "persistent" the chain just continues.

        Returns kv, or (kv, timing_dict) when collect_timing=True.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        timing = {"per_pass_ms": [], "per_pass_cache_len": []}

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ---- Pass 1: prefill the noise ---------------------------------
        noise_ids = self.generate_vocabulary_noise(domain_tokens, seed=seed)
        _sync()
        t0 = time.perf_counter()
        logits, kv = self._forward_with_kv(noise_ids)
        _sync()
        timing["per_pass_ms"].append((time.perf_counter() - t0) * 1000.0)
        timing["per_pass_cache_len"].append(self.get_cache_seq_len(kv))

        # ---- Passes 2..N: iterative proto-thought accumulation ---------
        for _pass in range(1, self.num_passes):
            _sync()
            t0 = time.perf_counter()

            if self.chain_mode == "reseed":
                # Inject a fresh noise chunk before this pass's thinking.
                pass_seed = None if seed is None else seed * 1000 + _pass
                fresh = self.generate_vocabulary_noise(domain_tokens, seed=pass_seed)
                logits, kv = self._forward_with_kv(fresh, past_key_values=kv)

            # Autoregressive thinking: one token at a time, cache grows.
            next_tok = self._pick_next_token(logits, gen=gen)
            for _ in range(self.think_tokens):
                logits, kv = self._forward_with_kv(next_tok, past_key_values=kv)
                next_tok = self._pick_next_token(logits, gen=gen)

            _sync()
            timing["per_pass_ms"].append((time.perf_counter() - t0) * 1000.0)
            timing["per_pass_cache_len"].append(self.get_cache_seq_len(kv))

        # ---- Stage 3: optional persona conditioning --------------------
        if persona_text is not None:
            persona_ids = self.tokenizer(
                persona_text, return_tensors="pt",
                truncation=True, max_length=self.persona_max_length,
            ).input_ids.to(self.device)
            _sync()
            t0 = time.perf_counter()
            _, kv = self._forward_with_kv(persona_ids, past_key_values=kv)
            _sync()
            timing["persona_ms"] = (time.perf_counter() - t0) * 1000.0

        timing["total_ms"] = sum(timing["per_pass_ms"]) + timing.get("persona_ms", 0.0)
        timing["final_cache_len"] = self.get_cache_seq_len(kv)
        timing["num_passes"] = self.num_passes
        timing["think_tokens"] = self.think_tokens
        timing["chain_mode"] = self.chain_mode

        if collect_timing:
            return kv, timing
        return kv

    def get_cache_seq_len(self, kv_cache) -> int:
        if hasattr(kv_cache, "get_seq_length"):
            return kv_cache.get_seq_length()
        # legacy tuple-of-tuples fallback
        return kv_cache[0][0].shape[-2]

    def timed_prime(self, **kwargs) -> tuple[object, float]:
        t0 = time.perf_counter()
        kv = self.prime(**kwargs)
        ms = (time.perf_counter() - t0) * 1_000.0
        return kv, ms
