"""
TIM (Thinking Initiation Machine) — vocabulary-lens KV-cache warm priming.

Works directly with a loaded HF `model` + `tokenizer` (as already used in
run_humaneval_evalplus.py) — no separate backbone abstraction required.

Stage 1: sample noise_length token IDs from the vocabulary. If domain_tokens
         is given: 70% sampled from the domain pool, 30% uniform random —
         otherwise 100% uniform random. IDs are used directly; never
         decode-then-re-tokenize (BPE does not round-trip losslessly, so
         doing so silently changes both content and length).
Stage 2: (optional) extend the noise via one autoregressive generate() call,
         then a single forward pass over the full sequence to obtain one
         unified KV cache.
Stage 3: (optional) prefill persona/domain-description text on top.

prime() returns a `past_key_values` object usable directly with
model.generate(past_key_values=..., ...).
"""

import time
from typing import List, Optional, Tuple

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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.noise_length = noise_length
        self.num_passes = num_passes
        self.think_tokens = think_tokens

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

    def get_domain_token_ids(self, domain: str) -> List[int]:
        """Tokenize the domain string (and its whitespace-split words) into
        a small pool of real token IDs to seed domain-weighted noise."""
        words = [domain] + domain.split()
        ids: List[int] = []
        for w in words:
            ids.extend(self.tokenizer.encode(w, add_special_tokens=False))
        seen: set = set()
        unique = [x for x in ids if x not in self._excluded_ids and not (x in seen or seen.add(x))]
        return unique if unique else [self.tokenizer.eos_token_id]

    def generate_vocabulary_noise(
        self,
        domain_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
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

    def prime(
        self,
        domain_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
        persona_text: Optional[str] = None,
    ):
        """Build and return a warm KV cache (past_key_values)."""
        noise_ids = self.generate_vocabulary_noise(domain_tokens, seed=seed)  # [1, noise_length]

        if self.num_passes > 1:
            think_total = self.think_tokens * (self.num_passes - 1)
            with torch.no_grad():
                full_seq = self.model.generate(
                    input_ids=noise_ids,
                    max_new_tokens=think_total,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            _, kv = self._forward_with_kv(full_seq)
        else:
            _, kv = self._forward_with_kv(noise_ids)

        if persona_text is not None:
            persona_ids = self.tokenizer(
                persona_text, return_tensors="pt", truncation=True, max_length=256,
            ).input_ids.to(self.device)
            _, kv = self._forward_with_kv(persona_ids, past_key_values=kv)

        return kv

    def get_cache_seq_len(self, kv_cache) -> int:
        if hasattr(kv_cache, "get_seq_length"):
            return kv_cache.get_seq_length()
        # legacy tuple-of-tuples fallback
        return kv_cache[0][0].shape[-2]

    def timed_prime(self, **kwargs) -> Tuple[object, float]:
        t0 = time.perf_counter()
        kv = self.prime(**kwargs)
        ms = (time.perf_counter() - t0) * 1_000.0
        return kv, ms