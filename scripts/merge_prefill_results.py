#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-tool", type=Path, required=True)
    parser.add_argument("--new-tool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = args.previous_tool.resolve()
    new = args.new_tool.resolve()
    sys.path.insert(0, str(new))
    from src.settings.config import get_config
    get_config(str(new / "configs" / "tuning_args.txt"))
    from src.extractors.config import merge_and_deduplicate
    from src.generators import create_config

    summary = {"previous_tool": str(previous), "new_tool": str(new), "types": {}}
    for tune_type in ("custom", "infer"):
        old_dir = previous / "optimization_artifacts" / tune_type / "compare" / "csv"
        new_dir = new / "optimization_artifacts" / tune_type / "compare" / "csv"
        backup_dir = new / "optimization_artifacts" / tune_type / "compare" / "csv_targeted"
        backup_dir.mkdir(parents=True, exist_ok=True)
        names = sorted({p.name for p in old_dir.glob("*.csv")} |
                       {p.name for p in new_dir.glob("*.csv")})
        type_summary = {}
        for name in names:
            targeted_path = new_dir / name
            previous_path = old_dir / name
            if targeted_path.exists():
                shutil.copy2(targeted_path, backup_dir / name)
                targeted = pd.read_csv(targeted_path)
            else:
                targeted = pd.DataFrame()
            previous_rows = pd.read_csv(previous_path) if previous_path.exists() else pd.DataFrame()
            columns = previous_rows.columns.tolist() if not previous_rows.empty else targeted.columns.tolist()
            combined = pd.concat([targeted, previous_rows], ignore_index=True, sort=False)
            combined = combined.reindex(columns=columns)
            combined.drop_duplicates(subset=["M", "N", "K", "B"], keep="first", inplace=True)
            combined.to_csv(targeted_path, index=False)
            type_summary[name] = {"targeted_rows": len(targeted),
                                  "previous_rows": len(previous_rows),
                                  "merged_rows": len(combined)}
        old_csv = new / "optimization_configs" / "old" / "csv" / "library_gpu4" / tune_type
        merged_csv = new / "optimization_configs" / "new" / "csv" / "library_gpu4" / tune_type
        config_dir = new / "optimization_configs" / "new" / "config" / "library_gpu4"
        merge_and_deduplicate(str(old_csv), str(new_dir), str(merged_csv), tune_type)
        create_config(str(merged_csv), str(config_dir), tune_type, "ECO")
        summary["types"][tune_type] = type_summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
