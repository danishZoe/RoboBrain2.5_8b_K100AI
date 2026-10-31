set -x
echo "=== /opt/dtk ==="
ls -la /opt/dtk/
echo "=== /opt/hyhal ==="
ls -la /opt/hyhal/
echo "=== version files ==="
find /opt/dtk /opt/hyhal -maxdepth 2 -iname '*version*' 2>/dev/null | head -20
echo "=== libgalaxyhip ==="
find / -name 'libgalaxyhip*' -maxdepth 6 2>/dev/null | head -20
echo "=== env ==="
env | grep -iE 'dcu|dtk|hyhal|ld_library|cuda|rocm|hip|path=' | head -20
echo "=== image ==="
cat /proc/1/environ 2>/dev/null | tr '\0' '\n' | grep -iE 'image|container|dcu|dtk' | head
ls -la /.dockerenv
cat /etc/os-release | head -5
echo "=== pip list ==="
python3 -m pip list 2>/dev/null | head -60
echo "=== python paths ==="
python3 -c "import sys; print('\n'.join(sys.path))"
