with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

text = text.replace('name = f"B_bf16_tim_domain_seed{seed}"', 'name = f"B_bf16_tim_domain_seed{seed}" + (f"_pass{args.num_passes}" if args.num_passes > 1 else "")')

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
