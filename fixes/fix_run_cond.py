import re
with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

text = text.replace('dataset=args.dataset, dataset="humaneval"', 'dataset="humaneval"')
text = text.replace('dataset=args.dataset,', 'dataset="humaneval",')

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
