# `baselines/` —— 官方基线文件放置目录

本目录**故意为空**。按赛事规则，精度验收所需的官方材料由平台下发，
参赛者不得自建、不得修改：

| 文件 | 来源 | 说明 |
|---|---|---|
| `BAAI__RoboBrain2.5-8B-NV.json` | 平台 / 官方分题材料 | 官方 GPU 基线：冻结解码配置（greedy、思考开关、`max_tokens`、数据集与题量）与官方 GPU 分数 |
| `data/<subset>.jsonl` | 同上 | 各评测子集（题量与 `max_tokens` 由基线文件的 `metrics` 段指定） |
| `judge.py`（可选） | 同上 | 官方判分器；存在时 `verify.py` 会优先调用它（接口 `judge(response, reference) -> bool`） |

基线文件的结构（取自官方 PDF 给出的示例）：

```json
{
  "model": "BAAI/RoboBrain2.5-8B-NV",
  "extra_body": {},
  "metrics": {
    "<指标名>": {
      "subset": "data/<subset>.jsonl",
      "n": 500,
      "max_tokens": 2048,
      "ref": 78.0,
      "scale": 100
    }
  }
}
```

`verify.py` 的行为：

- 本目录没有 `BAAI__RoboBrain2.5-8B-NV.json` → `status=BLOCKED`，退出码 2，
  并额外记录一项**明确标注为内部指标**的「同卡不同后端逐坐标一致性」；
- 有基线但缺 `metrics` 段 / 缺子集 / 缺参考输出 / 判分方式无法确定 →
  对应指标 `BLOCKED`，整体**不判 PASS**，退出码 2；
- 全部材料齐备 → **真正调用服务**逐样本评测，按官方口径算劣化后逐项 PASS/FAIL。

**任何情况下都不会在没有评测证据时报 PASS。**
