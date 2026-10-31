#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "############ envs/vllm_shim/sitecustomize.py ############"
cat envs/vllm_shim/sitecustomize.py 2>/dev/null
echo
echo "############ scripts/serve_vllm_patched.py ############"
cat scripts/serve_vllm_patched.py 2>/dev/null
echo
echo "############ scripts/serve_vllm.sh ############"
cat scripts/serve_vllm.sh 2>/dev/null
echo
echo "############ patches/ ############"
ls -la patches 2>/dev/null
for p in patches/*.patch; do echo "--- $p"; cat "$p"; done 2>/dev/null
echo
echo "############ scripts/image_canonical.py ############"
cat scripts/image_canonical.py 2>/dev/null
