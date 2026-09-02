import re
with open("experiment-3/run_int4_mechanism.py", "r") as f:
    text = f.read()

# Add get_mbpp_plus to imports
text = text.replace("from evalplus.data import get_human_eval_plus",
                    "from evalplus.data import get_human_eval_plus, get_mbpp_plus")

# Replace output_dir
text = text.replace('OUT_DIR = LOGS / "int4_mechanism"',
                    'OUT_DIR = LOGS / "mbpp_scale" if "mbpp" in sys.argv else LOGS / "int4_mechanism"')

# Arguments
text = text.replace('ap.add_argument("--num_seeds", type=int, default=3, help="Number of seeds for random arms.")',
                    'ap.add_argument("--num_seeds", type=int, default=3, help="Number of seeds for random arms.")\n    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval", help="Dataset to evaluate on")\n    ap.add_argument("--num_passes", type=int, default=1, help="Number of noise passes for TIMPrimer.")')

# get problems
def_get_problems = """
    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()
"""
text = text.replace('    problems = get_human_eval_plus()', def_get_problems)

# run_condition dataset passing
text = re.sub(r'run_condition\((.*?task_items),', r'run_condition(\1, dataset=args.dataset,', text)

# TIMPrimer instantiation
text = text.replace('num_passes=1', 'num_passes=args.num_passes')

with open("experiment-3/run_int4_mechanism.py", "w") as f:
    f.write(text)
