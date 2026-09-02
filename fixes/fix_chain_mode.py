import os
for fname in ["experiment-3/run_quant_gap.py", "experiment-3/run_int4_mechanism.py", "experiment-3/run_int4_tim_domain.py"]:
    with open(fname, "r") as f:
        text = f.read()
    
    text = text.replace('num_passes=args.num_passes)', 'num_passes=args.num_passes, chain_mode="reseed")')
    
    with open(fname, "w") as f:
        f.write(text)
