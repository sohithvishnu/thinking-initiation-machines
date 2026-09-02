import re
with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

# Replace run_condition calls to include dataset
text = re.sub(r'run_condition\((.*?task_items),', r'run_condition(\1, dataset=args.dataset,', text)

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
