#!/usr/bin/env bash
# 提取 CV-Bench 评测细节（终态分布 / 抽取兜底数 / 样例）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json, os, collections
B = "/root/private_data/workspace/robo/outputs/bench_public/cvbench"
s = json.load(open(os.path.join(B, "summary.json"), encoding="utf-8"))
print("planned:", s.get("n_planned"), " returned:", s.get("n_returned"),
      " fallback_extract:", s.get("n_fallback_extraction"))
print("terminal_states:", s.get("terminal_states"))
res = s.get("scores", {}).get("official_result", {})
print("official_result:", json.dumps(res, ensure_ascii=False))

ps = json.load(open(os.path.join(B, "per_sample.json"), encoding="utf-8"))
print("\n逐题样本数:", len(ps))
by = collections.Counter((p["source"], p["extracted_answer"], p["gt_answer"]) for p in ps)
wrong = [p for p in ps if p["extracted_answer"] != p["gt_answer"]]
print("提取答案与 GT 不一致:", len(wrong))
print("\n按 source 的错例数 / 总数:")
cnt = collections.Counter(p["source"] for p in ps)
wbad = collections.Counter(p["source"] for p in wrong)
for k in sorted(cnt):
    print("  %-8s %4d / %4d  = %.2f%%" % (k, wbad.get(k, 0), cnt[k],
                                          100.0 * (cnt[k] - wbad.get(k, 0)) / cnt[k]))
print("\n错例样例（前 8 条，截断）:")
for p in wrong[:8]:
    print("  id=%s src=%s gt=%s got=%r raw=%r" %
          (p["question_id"], p["source"], p["gt_answer"],
           p["extracted_answer"], (p["raw_output"] or "")[:90]))
print("\nraw_output 长度分布:")
ln = collections.Counter(min(len(p["raw_output"] or ""), 80) // 10 * 10
                         for p in ps)
for k in sorted(ln):
    print("  %3d-%3d 字符: %d" % (k, k + 9, ln[k]))
PY
