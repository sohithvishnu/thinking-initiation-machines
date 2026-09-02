with open("experiment-3/run_rung5_entropy.py", "r") as f:
    text = f.read()

text = text.replace('OUT_DIR = ROOT_DIR / "logs" / "int4_mechanism"',
                    'OUT_DIR = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "int4_mechanism"')

with open("experiment-3/run_rung5_entropy.py", "w") as f:
    f.write(text)
