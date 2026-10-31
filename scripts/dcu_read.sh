#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "############ submission/README.md ############"
cat submission/README.md
echo
echo "############ reports/09_vllm_adaptation.md ############"
cat reports/09_vllm_adaptation.md 2>/dev/null | head -250
