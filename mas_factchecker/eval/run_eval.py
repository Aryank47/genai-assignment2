import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add root to path to import agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.graph import app

TEST_FILE = Path("eval/test_cases.json")
OUTPUT_FILE = Path("eval/results.csv")


def run_evaluation():
    print("Starting Expert Evaluation Harness...")

    # Load Test Cases
    with open(TEST_FILE, "r") as f:
        tests = json.load(f)

    results = []
    total_latency = 0
    correct_count = 0

    for t in tests:
        print(f"\nTest {t['id']}: {t['claim']}")
        start_time = time.time()

        try:
            # Invoke Graph
            output = app.invoke({"claim": t["claim"], "tool_calls": 0})
            latency = time.time() - start_time

            # Metrics
            prediction = output["verdict"]
            is_correct = prediction == t["expected"]
            if is_correct:
                correct_count += 1
            total_latency += latency

            # Constraint Checking
            valid_outputs = ["SUPPORTED", "REFUTED", "NEI"]
            constraint_violation = prediction not in valid_outputs

            results.append(
                {
                    "id": t["id"],
                    "claim": t["claim"],
                    "expected": t["expected"],
                    "predicted": prediction,
                    "correct": is_correct,
                    "latency_sec": round(latency, 2),
                    "tool_calls": output["tool_calls"],
                    "constraint_violation": constraint_violation,
                    "reason": output.get("reason", ""),
                }
            )

            print(f"   -> Verdict: {prediction} (Expected: {t['expected']})")
            print(f"   -> Latency: {latency:.2f}s | Tools Used: {output['tool_calls']}")

        except Exception as e:
            print(f"   Error: {e}")

    # Compute Aggregate Metrics
    df = pd.DataFrame(results)
    success_rate = (correct_count / len(tests)) * 100
    avg_latency = total_latency / len(tests)
    avg_tools = df["tool_calls"].mean()

    print("\n" + "=" * 40)
    print("FINAL EVALUATION REPORT")
    print("=" * 40)
    print(f"Success Rate:      {success_rate:.1f}%")
    print(f"Avg Latency:       {avg_latency:.2f} sec")
    print(f"Avg Tool Calls:    {avg_tools:.1f}")
    print(f"Constraint Errors: {df['constraint_violation'].sum()}")
    print("=" * 40)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Detailed results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_evaluation()
