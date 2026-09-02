with open("experiment-3/run_int4_tim_domain.py", "r") as f:
    text = f.read()

text = text.replace('def main():\n    problems = get_human_eval_plus()\n    task_items = list(problems.items())',
'''def main():
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
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]''')

text = text.replace('OUT_DIR = ROOT_DIR / "logs" / "quality_quant"',
                    'OUT_DIR = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "quality_quant"')

text = text.replace('for seed in range(NUM_SEEDS):', 'for seed in seed_list:')
text = text.replace('name = f"int4_tim_domain_seed{seed}"', 'name = f"int4_tim_domain_seed{seed}" + (f"_pass{args.num_passes}" if args.num_passes > 1 else "")')
text = text.replace('score=True,', 'score=score, dataset=args.dataset,')
text = text.replace('num_passes=args.num_passes, chain_mode="reseed"', 'num_passes=args.num_passes, chain_mode="reseed"')

with open("experiment-3/run_int4_tim_domain.py", "w") as f:
    f.write(text)
