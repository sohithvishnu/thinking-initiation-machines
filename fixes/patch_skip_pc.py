with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

text = text.replace('ap.add_argument("--num_passes", type=int, default=1)',
                    'ap.add_argument("--num_passes", type=int, default=1)\n    ap.add_argument("--skip_prompt_control", action="store_true")')

text = text.replace('report["E"] = run_condition(model, tokenizer, device, "E_bf16_prompt_control"',
                    'if not args.skip_prompt_control:\n        report["E"] = run_condition(model, tokenizer, device, "E_bf16_prompt_control"')

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
