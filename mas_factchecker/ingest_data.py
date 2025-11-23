import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def ingest_isot():
    print("🔨 Processing ISOT (Real vs Fake News)...")
    true_path = DATA_DIR / "True.csv"
    fake_path = DATA_DIR / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        print("Skipping ISOT: 'True.csv' or 'Fake.csv' not found.")
        return

    try:
        df_true = pd.read_csv(true_path)
        df_fake = pd.read_csv(fake_path)
        df_true["label"] = "REAL"
        df_fake["label"] = "FAKE"

        df_combined = (
            pd.concat([df_true, df_fake])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        output_path = DATA_DIR / "isot_index.csv"
        df_combined[["title", "subject", "label"]].to_csv(output_path, index=False)
        print(f"ISOT Index built: {len(df_combined)} records saved to {output_path}")
    except Exception as e:
        print(f"Error processing ISOT: {e}")


def ingest_fever_local():
    print("🔨 Processing Local FEVER File...")

    # Input: The big file you downloaded
    raw_path = DATA_DIR / "fever.jsonl"
    # Output: The optimized file for the agent
    output_path = DATA_DIR / "fever_index.jsonl"

    if not raw_path.exists():
        print(f"Error: Could not find {raw_path}")
        print("Make sure you moved the downloaded file to 'data/fever.jsonl'")
        return

    if output_path.exists():
        print("FEVER index already exists. Deleting old one to regenerate...")
        output_path.unlink()

    count = 0
    limit = 10000  # Process first 10k records (App becomes slow if we load all 100k+)

    print(f"Reading from {raw_path}...")

    with open(raw_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            try:
                row = json.loads(line)

                # Filter: We only want Verifiable facts
                if row["label"] in ["SUPPORTS", "REFUTES"]:

                    # FEVER raw evidence is [[WikiID, LineID], ...].
                    # We simplify this for the agent to just point to the Wiki
                    # source.
                    evidence_summary = "Evidence available in DB"
                    if row.get("evidence"):
                        # Try to grab the first Wiki page ID referenced
                        try:
                            wiki_page = row["evidence"][0][0][2]
                            evidence_summary = f"Reference: Wiki Page '{wiki_page}'"
                        except (IndexError, KeyError, TypeError):
                            # Skip if the nested structure is not as expected
                            pass

                    record = {
                        "claim": row["claim"],
                        "label": row["label"],
                        "evidence": evidence_summary,
                    }

                    f_out.write(json.dumps(record) + "\n")
                    count += 1

                    if count >= limit:
                        break
            except Exception:
                continue

    print(f"FEVER Index built: {count} records saved to {output_path}")


if __name__ == "__main__":
    ingest_isot()
    ingest_fever_local()
