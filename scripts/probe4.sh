export PATH=/opt/hyhal/bin:/opt/dtk/bin:/opt/dtk/hip/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dcc/gcvm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip

echo "=== hy-smi showall ==="
hy-smi --showall 2>&1 | head -60
echo
echo "=== hy-smi showproductname ==="
hy-smi --showproductname 2>&1 | head -20
echo
echo "=== hy-smi showuniqueid / serial ==="
hy-smi --showserial 2>&1 | head -10
hy-smi --showdriverversion 2>&1 | head -10
hy-smi --showmeminfo vram 2>&1 | head -20
