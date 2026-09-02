import re
with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

text = text.replace('ap.add_argument("--noise_length", type=int, default=64)',
                    'ap.add_argument("--noise_length", type=int, default=64)\n    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")\n    ap.add_argument("--num_passes", type=int, default=1)')

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
