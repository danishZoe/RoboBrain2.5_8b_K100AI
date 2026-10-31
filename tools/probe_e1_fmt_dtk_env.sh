#!/usr/bin/env bash
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== A) E1 jsonl 原始结构（前 2 行，截断 1200 字符）====="
head -c 1200 outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/ab_4096_c1.jsonl; echo; echo "..."
echo "--- ab_4096_c8.jsonl ---"
head -c 1200 outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/ab_4096_c8.jsonl; echo
echo "--- 目录里还有什么 ---"
ls -la outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/smoke/ outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/probe/
echo "--- run_config.txt ---"
cat outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/run_config.txt

echo
echo "===== B) 找 DTK env.sh ====="
ls /opt/dtk/ 2>/dev/null | head
find /opt -maxdepth 4 -name 'env.sh' 2>/dev/null | head -n 10
grep -rn "galaxyhip" /opt/dtk/*/env.sh 2>/dev/null | head -n 5

echo
echo "===== C) source DTK env 后 flagevalmm 能否 import ====="
for E in $(find /opt -maxdepth 4 -name 'env.sh' 2>/dev/null | head -n 3); do
  echo "--- try: $E ---"
  ( set +u; source "$E" >/dev/null 2>&1;
    PYTHONPATH="$WS/third_party/upstream_eval/FlagEvalMM:$WS/envs/pylibs:${PYTHONPATH:-}" \
    python3 -c "
import flagevalmm.evaluator.point_utils as P
print('  point_utils OK')
import flagevalmm.evaluator.common_types as C
print('  common_types OK')
" 2>&1 | tail -n 3 )
done

echo
echo "===== D) pip open3d 日志 ====="
ls -la logs/ | grep -i open3d || echo "  无日志文件"
ps -ef | grep '[p]ip install' | head -n 3

echo
echo "===== E) 端口与服务状态 ====="
(ss -ltnp 2>/dev/null | grep -E ':(8000|30000)') || echo "  无 8000/30000 监听"
