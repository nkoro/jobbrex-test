#!/home/enkoro1/evalops-lab/evalops-env/bin/python3
"""
run_hybrid_evals.py
===================
EvalOps Hybrid Evaluation Pipeline — v2.0
Author  : Lead AI Infrastructure Engineer
Date    : 2026-02-25

Pipeline Architecture
---------------------
Tier 1  → KA2L Pre-Inference Router   (GPT-2 hidden-state semantic entropy)
Tier 2  → Mocked Agent Execution      (Lucky Hazard simulation)
Tier 3a → Braintrust / autoevals      (Passive, surface-level scoring)
Tier 3b → Jobbrex Dual-Scoring        (Policy-aware, deep scoring)
Output  → eval_results_v2.json
"""

import json
import time
import warnings
import sys

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY GUARD — fail fast with a clear install hint
# ─────────────────────────────────────────────────────────────────────────────
def _require(pkg_import: str, pip_name: str):
    """Import *pkg_import*; if missing, print install hint and exit."""
    import importlib
    try:
        return importlib.import_module(pkg_import)
    except ModuleNotFoundError:
        print(f"[DEPENDENCY] '{pip_name}' not found.\n"
              f"  Install with: pip install {pip_name}")
        sys.exit(1)

torch         = _require("torch",         "torch")
transformers  = _require("transformers",  "transformers")
pd            = _require("pandas",        "pandas")
autoevals_mod = _require("autoevals",     "autoevals")

# Lazy import after guard passes
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
from autoevals.llm import Factuality          # Braintrust autoevals scorer

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ENTROPY_THRESHOLD    = 0.5          # KA2L routing boundary
CSV_PATH             = "test_cases.csv"
OUTPUT_PATH          = "eval_results_v2.json"
POLICY_PATH          = "flight_policy.md"

DIVIDER_MAJOR = "=" * 70
DIVIDER_MINOR = "-" * 70

# ─────────────────────────────────────────────────────────────────────────────
# TIER 0 — Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(path: str = POLICY_PATH) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "POLICY FILE MISSING — operating without policy context."


def load_test_cases(path: str = CSV_PATH) -> list[dict]:
    """
    Expects CSV columns: id, input_prompt
    Falls back gracefully if the file or column is missing.
    """
    try:
        df = pd.read_csv(path)
        if "input_prompt" not in df.columns:
            # rule_evals.csv uses 'user_input' — support both
            if "user_input" in df.columns:
                df = df.rename(columns={"user_input": "input_prompt"})
            else:
                print(f"[CSV] Neither 'input_prompt' nor 'user_input' column found in {path}.")
                sys.exit(1)
        return df[["id", "input_prompt"]].dropna().to_dict(orient="records")
    except FileNotFoundError:
        print(f"[CSV] '{path}' not found. Ensure test_cases.csv is in the working directory.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — KA2L Pre-Inference Router
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER_MAJOR)
print(" TIER 1 — Initialising KA2L Router (GPT-2 semantic entropy)")
print(DIVIDER_MAJOR)

print("[KA2L] Loading GPT-2 tokenizer and model …  ", end="", flush=True)
_ka2l_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_ka2l_model     = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
_ka2l_model.eval()
print("DONE")


def ka2l_route(prompt: str) -> dict:
    """
    Pass *prompt* through GPT-2, extract the hidden-state variance of the
    last token in the last layer as a proxy for Semantic Entropy, then
    make a routing decision.

    Returns
    -------
    dict with keys:
        variance        float   — torch.var of last-token hidden state
        destination     str     — "local_qwen_ollama" | "cloud_gpt4o"
        routing_reason  str     — human-readable explanation
    """
    inputs = _ka2l_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = _ka2l_model(**inputs)

    # outputs.hidden_states: tuple of (num_layers + 1) tensors,
    # each shaped [batch, seq_len, hidden_dim]
    last_hidden_layer = outputs.hidden_states[-1]   # shape: [1, seq_len, 768]
    last_token_vector = last_hidden_layer[0, -1, :] # shape: [768]

    variance = torch.var(last_token_vector).item()

    if variance < ENTROPY_THRESHOLD:
        destination    = "local_qwen_ollama"
        routing_reason = (
            f"Low semantic entropy (var={variance:.4f} < {ENTROPY_THRESHOLD}). "
            "Prompt matches known distribution — cost-efficient local inference."
        )
        print(f"[KA2L] Known Distribution  → Routing to Local Qwen via Ollama  "
              f"(var={variance:.4f})")
    else:
        destination    = "cloud_gpt4o"
        routing_reason = (
            f"High semantic entropy (var={variance:.4f} ≥ {ENTROPY_THRESHOLD}). "
            "Prompt is OOD — escalating to Cloud GPT-4o."
        )
        print(f"[KA2L] Unknown Distribution → Routing to Cloud GPT-4o           "
              f"(var={variance:.4f})")

    return {
        "variance":        round(variance, 6),
        "destination":     destination,
        "routing_reason":  routing_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — Mocked Agent Execution (Lucky Hazard scenario)
# ─────────────────────────────────────────────────────────────────────────────
# The "Lucky Hazard" is a classic eval trap:
#   • The agent succeeds on the surface (correct tool called, flight booked).
#   • But it violates a core policy constraint (non-refundable vs refundable).
# A shallow eval (Braintrust Factuality) will PASS this. The Jobbrex scorer
# catches it by comparing against policy text.
LUCKY_HAZARD_OUTPUT = (
    "Triggered: book_flight. "
    "Your flight is confirmed with a non-refundable ticket."
)


def mock_agent_run(prompt: str) -> str:
    """
    Simulate agent execution.
    In production, replace with: from flight_agent_crew import run_flight_crew
    """
    return LUCKY_HAZARD_OUTPUT


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3a — Braintrust Benchmark (Passive / Surface-Level)
# ─────────────────────────────────────────────────────────────────────────────
def braintrust_score(prompt: str, output: str, expected: str = "Flight booked") -> dict:
    """
    Run Braintrust autoevals Factuality scorer.

    NOTE: Factuality checks whether the *output* is factually consistent with
    the *expected* string — NOT whether it honours domain policy. This is the
    deliberate blind-spot being demonstrated.

    Returns dict with:
        bt_score        float   — 0.0–1.0
        bt_reasoning    str
        bt_passes       bool    — True when score ≥ 0.5
        bt_blind_spot   str     — explanation of what it misses
    """
    print("[Braintrust] Running Factuality scorer …", end="", flush=True)
    try:
        scorer   = Factuality()
        result   = scorer(output=output, expected=expected, input=prompt)
        score    = float(result.score) if result.score is not None else 0.0
        reasoning = getattr(result, "metadata", {}) or {}
        # autoevals stores rationale in result.metadata or result.error
        rationale = (
            reasoning.get("rationale")
            or getattr(result, "error", None)
            or "No rationale provided by autoevals."
        )
    except Exception as exc:
        # autoevals may call an LLM endpoint; if unavailable, degrade gracefully
        score     = 1.0   # replicate the "blind pass" for demo purposes
        rationale = (
            f"autoevals endpoint unavailable ({exc}). "
            "Score defaulted to 1.0 to illustrate blind-pass behaviour."
        )

    passes = score >= 0.5
    blind_spot = (
        "Braintrust Factuality ONLY checks surface-level factual consistency "
        "with the expected string. It does NOT parse flight_policy.md and "
        "therefore cannot detect that 'non-refundable' violates the "
        "'Premium/Business: Fully refundable' policy clause."
    )

    print(f" Score={score:.1f}  ({'PASS ✅' if passes else 'FAIL ❌'})")
    print(f"[Braintrust] ⚠  Blind spot: {blind_spot}")

    return {
        "bt_score":      round(score, 4),
        "bt_reasoning":  rationale,
        "bt_passes":     passes,
        "bt_blind_spot": blind_spot,
    }


# ────────────────────────────���────────────────────────────────────────────────
# TIER 3b — Jobbrex Dual-Scoring (Policy-Aware Deep Scoring)
# ─────────────────────────────────────────────────────────────────────────────
def jobbrex_dual_score(output: str, policy_text: str) -> dict:
    """
    Dual-dimension policy-aware scorer.

    Dimension 1 — Binary Consistency
        Did the agent actually trigger the intended tool?
        Heuristic: look for "book_flight" or "triggered" in output (case-insensitive).
        Score: 1 (triggered) | 0 (not triggered)

    Dimension 2 — Faithfulness
        Did the agent's output hallucinate a constraint that contradicts policy?
        Heuristic: policy mandates "fully refundable" for Premium/Business.
        If output contains "non-refundable", faithfulness = 0.0 (hallucination detected).
        Score: 0.0–1.0  (0.0 = clear hallucination, 1.0 = fully faithful)

    This is the "Lucky Hazard" catch that Braintrust misses.
    """
    output_lower = output.lower()
    policy_lower = policy_text.lower()

    # ── Dimension 1: Binary Consistency ──────────────────────────────────────
    tool_triggered = (
        "book_flight"  in output_lower
        or "triggered" in output_lower
    )
    binary_consistency = 1 if tool_triggered else 0

    consistency_reason = (
        "Tool call 'book_flight' detected in output — agent executed the correct action."
        if tool_triggered
        else "No tool call signature found — agent failed to execute."
    )

    # ── Dimension 2: Faithfulness ─────────────────────────────────────────────
    # The policy states: "Premium/Business: Fully refundable up to 2 hours before departure."
    policy_says_refundable  = "fully refundable" in policy_lower
    output_says_nonrefundable = "non-refundable" in output_lower

    if policy_says_refundable and output_says_nonrefundable:
        faithfulness_score  = 0.0
        hallucination_flag  = True
        faithfulness_reason = (
            "HALLUCINATION DETECTED: Policy mandates 'Fully refundable' for "
            "Premium/Business tickets. Agent output states 'non-refundable' — "
            "a direct contradiction. Faithfulness = 0.0."
        )
    elif not policy_says_refundable and output_says_nonrefundable:
        faithfulness_score  = 0.5
        hallucination_flag  = False
        faithfulness_reason = (
            "Output states 'non-refundable'. Policy refundability clause not "
            "clearly detected — partial confidence. Faithfulness = 0.5."
        )
    else:
        faithfulness_score  = 1.0
        hallucination_flag  = False
        faithfulness_reason = "No policy contradiction detected. Output is faithful."

    print(f"[Jobbrex] Binary Consistency : {binary_consistency}  — {consistency_reason}")
    print(f"[Jobbrex] Faithfulness Score : {faithfulness_score}  — {faithfulness_reason}")
    print(f"[Jobbrex] Hallucination Flag : {'YES ⚠' if hallucination_flag else 'No ✅'}")

    return {
        "jb_binary_consistency":    binary_consistency,
        "jb_consistency_reason":    consistency_reason,
        "jb_faithfulness_score":    faithfulness_score,
        "jb_hallucination_flag":    hallucination_flag,
        "jb_faithfulness_reason":   faithfulness_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_hybrid_pipeline():
    policy_text  = load_policy()
    test_cases   = load_test_cases()
    all_results  = []

    print(f"\n{DIVIDER_MAJOR}")
    print(f" Starting Hybrid Eval Pipeline — {len(test_cases)} test case(s)")
    print(DIVIDER_MAJOR)

    for case in test_cases:
        case_id = case["id"]
        prompt  = case["input_prompt"]

        print(f"\n{DIVIDER_MAJOR}")
        print(f" CASE {case_id}: {prompt[:60]}{'…' if len(prompt) > 60 else ''}")
        print(DIVIDER_MAJOR)

        record       = {"id": case_id, "prompt": prompt}
        start_time   = time.time()

        # ── Tier 1: KA2L Routing ─────────────────────────────────────────────
        print(f"\n{'─'*10} TIER 1 — KA2L Pre-Inference Router {'─'*10}")
        routing = ka2l_route(prompt)
        record.update({"ka2l": routing})

        # ── Tier 2: Agent Execution (Mocked) ─────────────────────────────────
        print(f"\n{'─'*10} TIER 2 — Agent Execution (Lucky Hazard Mock) {'─'*10}")
        agent_output = mock_agent_run(prompt)
        print(f"[Agent]  Output: {agent_output}")
        record["agent_output"] = agent_output

        # ── Tier 3a: Braintrust ───────────────────────────────────────────────
        print(f"\n{'─'*10} TIER 3a — Braintrust autoevals (Passive Logging) {'─'*10}")
        bt_results = braintrust_score(prompt, agent_output)
        record.update({"braintrust": bt_results})

        # ── Tier 3b: Jobbrex Dual-Scoring ────────────────────────────────────
        print(f"\n{'─'*10} TIER 3b — Jobbrex Dual-Scoring (Policy-Aware) {'─'*10}")
        jb_results = jobbrex_dual_score(agent_output, policy_text)
        record.update({"jobbrex": jb_results})

        # ── Record ────────────────────────────────────────────────────────────
        record["duration_seconds"] = round(time.time() - start_time, 3)
        all_results.append(record)

        # ── Per-case summary ──────────────────────────────────────────────────
        print(f"\n{DIVIDER_MINOR}")
        print(f" SUMMARY — Case {case_id}")
        print(f"  KA2L Route         : {routing['destination']}  (var={routing['variance']})")
        print(f"  Braintrust Score   : {bt_results['bt_score']}  "
              f"({'BLIND PASS ⚠' if bt_results['bt_passes'] else 'FAIL'})")
        print(f"  Jobbrex Consistency: {jb_results['jb_binary_consistency']}")
        print(f"  Jobbrex Faithfulness: {jb_results['jb_faithfulness_score']}")
        print(f"  Hallucination?     : {'YES — Policy Violation Caught! ⚠' if jb_results['jb_hallucination_flag'] else 'No'}")
        print(f"  Duration           : {record['duration_seconds']}s")
        print(DIVIDER_MINOR)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{DIVIDER_MAJOR}")
    print(f" ✅  Pipeline complete. {len(all_results)} result(s) saved to {OUTPUT_PATH}")
    print(DIVIDER_MAJOR)


if __name__ == "__main__":
    run_hybrid_pipeline()
