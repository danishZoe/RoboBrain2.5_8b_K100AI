set -e
echo "=== dtk version file ==="
cat /opt/dtk/.dtk_version; echo
cat /opt/hyhal/.info/version 2>/dev/null; echo
echo "=== conda pythons ==="
ls /opt/conda/bin/python* 2>/dev/null || echo "no /opt/conda"
echo "=== hyhal bin ==="
ls /opt/hyhal/bin/
echo "=== hy-smi ==="
export PATH=/opt/hyhal/bin:/opt/dtk/bin:/opt/dtk/hip/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
(hy-smi 2>&1 | head -40) || echo "hy-smi failed"
echo "=== dtk env.sh head ==="
head -30 /opt/dtk/env.sh
