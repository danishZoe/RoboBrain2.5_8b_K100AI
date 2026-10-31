#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
for f in submission/verify.py submission/smoke.sh submission/env_check.sh \
         submission/run.sh submission/requirements-extra.txt submission/reproduce.sh; do
  echo "############ $f ############"
  cat "$f" 2>/dev/null
  echo
done
echo "############ samples dir ############"
ls -la samples 2>/dev/null
find samples -maxdepth 2 -type f 2>/dev/null | head -20
