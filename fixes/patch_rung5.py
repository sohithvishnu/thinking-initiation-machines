import re
with open("experiment-3/run_rung5_entropy.py", "r") as f:
    text = f.read()

text = text.replace("from evalplus.data import get_human_eval_plus",
                    "from evalplus.data import get_human_eval_plus, get_mbpp_plus")

text = text.replace('OUT_DIR = LOGS / "int4_mechanism"',
                    'OUT_DIR = LOGS / "mbpp_scale" if "mbpp" in sys.argv else LOGS / "int4_mechanism"')

text = text.replace('ap.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (default: 3).")',
                    'ap.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (default: 3).")\n    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")')

def_get_problems = """
    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()
"""
text = text.replace('    problems = get_human_eval_plus()', def_get_problems)

text = re.sub(r'run_condition\((.*?task_items),', r'run_condition(\1, dataset=args.dataset,', text)

with open("experiment-3/run_rung5_entropy.py", "w") as f:
    f.write(text)
