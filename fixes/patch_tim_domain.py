import re
with open("experiment-3/run_int4_tim_domain.py", "r") as f:
    text = f.read()

text = text.replace("from evalplus.data import get_human_eval_plus",
                    "from evalplus.data import get_human_eval_plus, get_mbpp_plus\nimport argparse\nimport sys")

text = text.replace('OUT_DIR = LOGS_DIR / "quality_quant"',
                    'OUT_DIR = LOGS_DIR / "mbpp_scale" if "mbpp" in sys.argv else LOGS_DIR / "quality_quant"')

def_main = """
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_score", action="store_true")
    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_passes", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()

    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()
        
    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[:args.limit]
    score = not args.no_score
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
"""

text = re.sub(r'problems = get_human_eval_plus\(\)\n\s+task_items = list\(problems\.items\(\)\)\n\s+score = True', def_main, text)
text = text.replace('for seed in range(3):', 'for seed in seed_list:')
text = text.replace('num_passes=1', 'num_passes=args.num_passes')
text = text.replace('condition_name = f"int4_tim_domain_seed{seed}"', 'condition_name = f"int4_tim_domain_seed{seed}" + (f"_pass{args.num_passes}" if args.num_passes > 1 else "")')
text = re.sub(r'run_condition\((.*?task_items),', r'run_condition(\1, dataset=args.dataset,', text)

with open("experiment-3/run_int4_tim_domain.py", "w") as f:
    f.write(text)
