with open("experiment-3/run_int4_mechanism.py", "r") as f:
    text = f.read()

text = text.replace('OUT_DIR_NEW = ROOT_DIR / "logs" / "int4_mechanism"',
                    'OUT_DIR_NEW = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "int4_mechanism"')

text = text.replace('OUT_DIR_TIM = ROOT_DIR / "logs" / "quality_quant"',
                    'OUT_DIR_TIM = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "quality_quant"')

with open("experiment-3/run_int4_mechanism.py", "w") as f:
    f.write(text)
