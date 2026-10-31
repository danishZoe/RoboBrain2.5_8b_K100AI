#!/bin/bash
# Hygon DCU runtime environment (DTK 26.04-rc4 / driver 6.3.16-V1.1.0c)
# Usage:  source /root/private_data/workspace/robo/scripts/env_dcu.sh
#
# Why this file exists:
#   The container image ships the correct DTK/HYHAL env in PID 1's environment
#   and in /root/.bashrc (`source /opt/dtk/env.sh`), but non-interactive SSH
#   sessions do NOT read .bashrc. Without it:
#     - `hy-smi` / `hipcc` are not on PATH
#     - torch fails: libgalaxyhip.so.5 not found  (DCU runtime lib)
#     - torch fails: libomp.so not found          (OpenMP runtime in /opt/dtk/llvm/lib)
#   => `import torch` fails before any device check is possible.

export HYHAL_PATH=/opt/hyhal
export DTKROOT=/opt/dtk

# DTK sets ROCM_PATH / HIP_PATH / AMDGPU_TARGETS / PATH / LD_LIBRARY_PATH
if [ -f /opt/dtk/env.sh ]; then
    . /opt/dtk/env.sh
fi
if [ -f /opt/hyhal/env.sh ]; then
    . /opt/hyhal/env.sh
fi

# /opt/dtk/env.sh does not add /opt/dtk/llvm/lib on every path; torch's
# libtorch_cuda.so links libomp.so which lives only there.
export LD_LIBRARY_PATH="/opt/dtk/llvm/lib:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH}"

export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export DTK_VERSION=2604
export AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"

export PATH="/opt/hyhal/bin:/opt/dtk/bin:/opt/dtk/hip/bin:/opt/dtk/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# NOTE: keep /opt/conda/bin OUT of the front of PATH.
#   /usr/bin/python3 (3.10.12) carries the DTK-built torch / sglang / vllm.
#   /opt/conda/bin/python3 has NO torch installed.
# Use $DCU_PY (default /usr/bin/python3) to be explicit.
export DCU_PY=/usr/bin/python3
