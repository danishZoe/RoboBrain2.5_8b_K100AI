echo "=== find libomp ==="
find / -name 'libomp.so*' -not -path '/proc/*' 2>/dev/null | head -10
echo "=== find libgomp ==="
find / -name 'libgomp.so*' -not -path '/proc/*' 2>/dev/null | head -5
echo "=== PID1 full env ==="
tr '\0' '\n' < /proc/1/environ | sort
echo "=== /etc/profile.d ==="
ls /etc/profile.d/ 2>/dev/null
for f in /etc/profile.d/*.sh; do echo "--- $f"; head -40 "$f"; done 2>/dev/null | head -120
echo "=== root bashrc dtk refs ==="
grep -iE 'dtk|hyhal|libomp|LD_LIBRARY' /root/.bashrc /root/.profile /etc/profile 2>/dev/null | head -20
echo "=== ldd torch _C ==="
ldd /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so 2>/dev/null | grep -iE 'not found' | head -20
