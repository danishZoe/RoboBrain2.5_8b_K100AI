#!/usr/bin/env bash
echo "=== grep SCNET_USER_HOME in /root/.bashrc ==="
grep -n 'SCNET_USER_HOME' /root/.bashrc 2>&1 | head -5
echo "=== bashrc 是否存在 ==="
ls -l /root/.bashrc 2>&1
echo "=== 交互式 shell 里 SCNET_USER_HOME 有值吗 ==="
bash -ic 'echo "SCNET_USER_HOME=${SCNET_USER_HOME:-<empty>}"' 2>/dev/null | tail -2
echo "=== done ==="
