from evalplus.data import get_mbpp
mbpp = get_mbpp()
first_id = list(mbpp.keys())[0]
print(f"Task ID: {first_id}")
print(f"Keys: {mbpp[first_id].keys()}")
print(f"Prompt preview:\n{mbpp[first_id]['prompt'][:200]}")
