# 同卡 · 首提基线 eager（赛事「首次提交」口径）

**配置**：`--enforce-eager` + stock rocBLAS 库 + 默认 attention。
**会话**：当前节点（同一张卡）
card_id=host:crdnotebook-2114695384420274177-scnjz2s4uc-41571:boot:62ce6c81-9a7b-44b6-b994-b5054faa7f6d

与 `../tuned/`（最优版）**必须来自同一张卡**才能相减；
逐档对比见 `../../reports/tuning_comparison.md` §2.1
与 `../../reports/25_tuning_delivery_20261031.md` §5.3。
