import json, re
def docstring_rate(path):
    n = d = 0
    for line in open(path):
        sol = json.loads(line)["solution"]
        n += 1
        # docstring immediately following a def
        if re.search(r'def\s+\w+\([^)]*\)[^:]*:\s*\n\s*(\'\'\'|""")', sol):
            d += 1
    return d, n, d/n
for f in ["prompt_control-sanitized.jsonl", "tim_domain_p12_seed0-sanitized.jsonl"]:
    print(f, docstring_rate(f))


import json
def passed(path):
    d = json.load(open(path))["eval"]
    return {t: e[0]["plus_status"] == "pass" for t, e in d.items()}

a = passed("prompt_control-sanitized.eval_results.json")
b = passed("tim_domain_p12_seed0-sanitized.eval_results.json")

only_a = [t for t in a if a[t] and not b[t]]
only_b = [t for t in b if b[t] and not a[t]]
print(f"prompt_control only: {len(only_a)} {only_a}")
print(f"tim_domain only    : {len(only_b)} {only_b}")
print(f"both pass: {sum(1 for t in a if a[t] and b[t])}  both fail: {sum(1 for t in a if not a[t] and not b[t])}")