#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-tool", type=Path, required=True)
    parser.add_argument("--new-tool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for tune_type in ("custom", "infer"):
        for name in ("bf16_NN.csv", "bf16_TN.csv"):
            old_path = args.previous_tool / "optimization_artifacts" / tune_type / "compare" / "csv" / name
            new_path = args.new_tool / "optimization_artifacts" / tune_type / "compare" / "csv" / name
            if not old_path.exists() or not new_path.exists():
                continue
            old = pd.read_csv(old_path)
            new = pd.read_csv(new_path)
            merged = new.merge(old, on=["M", "N", "K", "B"], suffixes=("_clean", "_old"))
            for _, r in merged.iterrows():
                rows.append({
                    "tune_type": tune_type,
                    "layout": name.removesuffix(".csv"),
                    "M": int(r["M"]), "N": int(r["N"]), "K": int(r["K"]), "B": int(r["B"]),
                    "old_tuned_gflops": float(r["new_gflops_old"]),
                    "clean_gflops": float(r["new_gflops_clean"]),
                    "ratio_clean_vs_old": float(r["new_gflops_clean"] / r["new_gflops_old"]),
                    "old_tuned_us": float(r["new_us_old"]),
                    "clean_us": float(r["new_us_clean"]),
                })
    frame = pd.DataFrame(rows).sort_values("ratio_clean_vs_old", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_suffix(".csv"), index=False)
    summary = {
        "count": len(frame),
        "improved_over_3pct": int((frame["ratio_clean_vs_old"] > 1.03).sum()),
        "regressed_over_3pct": int((frame["ratio_clean_vs_old"] < 0.97).sum()),
        "top_improvements": frame.head(15).to_dict("records"),
        "top_regressions": frame.tail(15).iloc[::-1].to_dict("records"),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
