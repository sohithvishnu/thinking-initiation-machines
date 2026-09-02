import re

with open("docs/int4_mechanism.md", "r") as f:
    text = f.read()

# Find the start of RUNG 5
r5_start = text.find("\n---\n\n## RUNG 5")
if r5_start != -1:
    r5_text = text[r5_start:]
    text = text[:r5_start]
    
    # Remove the sub-section "Files written" from Rung 5
    fw_start = r5_text.find("### Files written")
    if fw_start != -1:
        r5_content = r5_text[:fw_start].strip()
    else:
        r5_content = r5_text.strip()
        
    # Find the end of RUNG 4
    r4_end = text.find("\n---\n\n## Timing and memory")
    if r4_end != -1:
        # Insert Rung 5 before "Timing and memory"
        new_text = text[:r4_end] + "\n\n" + r5_content + "\n\n" + text[r4_end:].lstrip()
        
        # Ensure analysis_rung5 files are in the main Files written section
        if "answers_rung5.jsonl" not in new_text:
            new_text = new_text.replace(
                "- `logs/int4_mechanism/answers_rung{1,2,3,4}.jsonl`",
                "- `logs/int4_mechanism/answers_rung{1,2,3,4,5}.jsonl`"
            )
            new_text = new_text.replace(
                "- `scripts/analyze_int4_mechanism.py`\n- `logs/int4_mechanism/analysis_summary.json`",
                "- `scripts/analyze_int4_mechanism.py`\n- `scripts/analyze_rung5.py`\n- `logs/int4_mechanism/analysis_summary.json`\n- `logs/int4_mechanism/analysis_rung5.json`"
            )
            
        with open("docs/int4_mechanism.md", "w") as f:
            f.write(new_text)
        print("Document rewritten successfully.")
    else:
        print("Could not find Timing and memory section.")
else:
    print("Could not find RUNG 5 section.")
