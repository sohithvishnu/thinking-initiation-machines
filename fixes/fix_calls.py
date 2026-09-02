with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

text = text.replace('dataset="humaneval",\n                                 prepend_text', 'dataset=args.dataset,\n                                 prepend_text')
text = text.replace('dataset="humaneval",\n            primed_kv', 'dataset=args.dataset,\n            primed_kv')

import re
text = re.sub(r'run_condition\((.*?), output_dir, task_items, dataset="humaneval",', r'run_condition(\1, output_dir, task_items, dataset=args.dataset,', text)

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
