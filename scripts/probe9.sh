echo "=== libomp candidates ==="
for p in /usr/lib/x86_64-linux-gnu/libomp.so* /usr/lib/libomp.so* /usr/local/lib/libomp.so* /usr/local/lib/python3.10/dist-packages/torch/lib/libomp* /opt/dtk/lib/libomp* /opt/dtk/llvm/lib/libomp* /usr/lib/llvm-*/lib/libomp.so*; do
  [ -e "$p" ] && ls -la $p
done
echo "--- gcc libgomp ---"
ls -la /usr/lib/gcc/x86_64-linux-gnu/*/libgomp.so* 2>/dev/null | head
echo
echo "=== ldd missing for torch libs ==="
for so in /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so; do
  echo "--- $so"
  ldd $so 2>&1 | grep -i "not found" | head -20
done
echo
echo "=== /proc/1 env (PATH/LD_LIBRARY) ==="
tr '\0' '\n' < /proc/1/environ | grep -E '^(PATH|LD_LIBRARY_PATH|ROCM_PATH|DTK_VERSION|HIP_PATH|LIBRARY_PATH)='
echo
echo "=== bashrc ==="
grep -inE 'dtk|hyhal|omp|LD_LIBRARY|source ' /root/.bashrc 2>/dev/null | head -20
echo "=== /etc/profile.d ==="
ls /etc/profile.d/ 2>/dev/null
echo "=== opt/dtk/env.sh key lines ==="
grep -nE 'LD_LIBRARY_PATH|^export|PATH=' /opt/dtk/env.sh | head -30
