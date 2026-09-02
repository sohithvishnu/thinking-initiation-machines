import os

for fname in ["experiment-3/run_quant_gap.py", "experiment-3/run_int4_mechanism.py", "experiment-3/run_int4_tim_domain.py", "experiment-3/run_rung5_entropy.py"]:
    with open(fname, "r") as f:
        text = f.read()
    
    text = text.replace('LOGS_DIR / "mbpp_scale"', 'ROOT_DIR / "logs" / "mbpp_scale"')
    text = text.replace('LOGS / "mbpp_scale"', 'ROOT / "logs" / "mbpp_scale"')
    
    with open(fname, "w") as f:
        f.write(text)
