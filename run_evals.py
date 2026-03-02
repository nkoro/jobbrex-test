"""
run_evals.py  — v2 (updated)
Runs agent, judges output, saves eval_results.json
Now includes:
  - KA2L routing metadata (from flight_agent_crew)
  - Rule-based pre-check (no LLM needed)
  - Safe per-field judge parsing
  - expected_verdict comparison
"""

import json
import hashlib
import uuid
import pandas as pd
import time
from datetime import datetime
from flight_agent_crew import run_flight_crew
from judge_api import evaluate_interaction

def load_policy():
    with open("flight_policy.md", "r") as f:
        return f.read()

# ── Rule-based pre-check (deterministic, no LLM) ─────────────────────────────
def rule_based_check(agent_output: str, expected_verdict: str) -> dict:
    r = agent_output.lower()
    is_reject  = any(w in r for w in ["reject", "cannot", "not allowed", "decline", "sorry"])
    is_approve = any(w in r for w in ["success", "confirmed", "changed", "approved"])

    if expected_verdict.upper() == "REJECT":
        correct = is_reject and not is_approve
    else:
        correct = is_approve and not is_reject

    return {
        "rule_based_pass":   int(correct),
        "expected_verdict":  expected_verdict,
        "output_direction":  "REJECT" if is_reject else "APPROVE" if is_approve else "UNCLEAR"
    }


# ── Main eval loop ────────────────────────────────────────────────────────────
def run_eval_loop():
    run_id     = str(uuid.uuid4())[:8]
    run_time   = datetime.now().isoformat()
    policy_text = load_policy()
    policy_hash = hashlib.md5(policy_text.encode()).hexdigest()[:8]

    print(f"── EvalOps v2 | run_id={run_id} | policy_hash={policy_hash} ──")

    df = pd.read_csv("test_cases.csv")

    # Support both column names
    if "input_prompt" not in df.columns and "user_input" in df.columns:
        df = df.rename(columns={"user_input": "input_prompt"})

    # Default expected_verdict to empty string if column missing
    if "expected_verdict" not in df.columns:
        df["expected_verdict"] = "UNKNOWN"

    results = []

    for _, row in df.iterrows():
        print(f"\nCase {row['id']}: {str(row['input_prompt'])[:50]}...")
        start = time.time()

        # 1. Run agent (now returns dict with routing metadata)
        try:
            agent_result  = run_flight_crew(row['input_prompt'])
            agent_output  = agent_result["output"]
            routing       = agent_result["routing"]
            agent_policy_hash = agent_result["policy_hash"]
        except Exception as e:
            agent_output  = f"CRASH: {e}"
            routing       = {"route": "unknown", "variance": -1, "model": "none"}
            agent_policy_hash = "unknown"

        expected = str(row.get("expected_verdict", "UNKNOWN"))

        # 2. Rule-based check (fast, no LLM)
        rule_check = rule_based_check(agent_output, expected)

        # 3. LLM judge
        metrics = evaluate_interaction(
            row['input_prompt'], agent_output, policy_text, expected
        )

        results.append({
            "run_id":          run_id,
            "run_time":        run_time,
            "policy_hash":     policy_hash,
            "id":              row['id'],
            "input":           row['input_prompt'],
            "output":          agent_output,
            "ka2l_destination": routing.get("route"),
            "ka2l_variance":   routing.get("variance"),
            "ka2l_model":      routing.get("model"),
            **rule_check,
            **metrics,
            "duration":        round(time.time() - start, 3)
        })
        print(f"  Route: {routing.get('route')} ({routing.get('model')}) | "
              f"Rule: {'PASS' if rule_check['rule_based_pass'] else 'FAIL'} | "
              f"Faithfulness: {metrics.get('faithfulness_score')}")
        time.sleep(8)  # respect Groq free tier TPM limit (6000 tokens/min)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Done. {len(results)} results saved → eval_results.json")

if __name__ == "__main__":
    run_eval_loop()
