import re
with open("experiment-3/run_quant_gap.py", "r") as f:
    text = f.read()

# Add get_mbpp_plus to imports
text = text.replace("from evalplus.data import get_human_eval_plus, write_jsonl", 
                    "from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl")

# evaluate_with_evalplus dataset passing
text = text.replace('dataset="humaneval"', 'dataset=dataset')
text = text.replace('def evaluate_with_evalplus(sample_file: Path, dataset: str = "humaneval") -> dict:',
                    'def evaluate_with_evalplus(sample_file: Path, dataset: str) -> dict:')
text = text.replace('metrics.update(evaluate_with_evalplus(sanitized, dataset="humaneval"))',
                    'metrics.update(evaluate_with_evalplus(sanitized, dataset=dataset))')

# run_condition needs to accept dataset
text = text.replace('def run_condition(model, tokenizer, device, condition_name, output_dir, task_items,',
                    'def run_condition(model, tokenizer, device, condition_name, output_dir, task_items, dataset="humaneval",')

# Add arguments
text = text.replace('ap.add_argument("--noise_length", type=int, default=64, help="Number of tokens to prime.")',
                    'ap.add_argument("--noise_length", type=int, default=64, help="Number of tokens to prime.")\n    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval", help="Dataset to evaluate on")\n    ap.add_argument("--num_passes", type=int, default=1, help="Number of noise passes for TIMPrimer.")')

# get problems
def_get_problems = """
    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()
"""

# Replace problems = get_human_eval_plus() in all occurrences
text = text.replace('    problems = get_human_eval_plus()', def_get_problems)

# TIMPrimer instantiation
text = text.replace('num_passes=1', 'num_passes=args.num_passes')

# output_dir routing for mbpp
def replace_output_dir(section):
    return section.replace('output_dir = LOGS_DIR / safe /',
                           'output_dir = (LOGS_DIR / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe) /')
                           
text = replace_output_dir(text)
text = text.replace('output_dir = LOGS_DIR / safe / "gate"', 
                    'output_dir = LOGS_DIR / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe / "gate"')
text = text.replace('output_dir = LOGS_DIR / safe / "full"',
                    'output_dir = LOGS_DIR / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe / "full"')
text = text.replace('output_dir = LOGS_DIR / safe / "quality"',
                    'output_dir = LOGS_DIR / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe / "quality"')

with open("experiment-3/run_quant_gap.py", "w") as f:
    f.write(text)
