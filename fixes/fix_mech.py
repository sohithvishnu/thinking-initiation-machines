import re
with open("experiment-3/run_int4_mechanism.py", "r") as f:
    text = f.read()

text = text.replace('ap.add_argument("--rungs", default="2,3a,3b,4",',
                    'ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")\n    ap.add_argument("--num_passes", type=int, default=1)\n    ap.add_argument("--rungs", default="2,3a,3b,4",')

with open("experiment-3/run_int4_mechanism.py", "w") as f:
    f.write(text)
