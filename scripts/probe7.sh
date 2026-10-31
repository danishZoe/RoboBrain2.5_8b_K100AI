export PATH=/opt/hyhal/bin:/opt/dtk/bin:/opt/dtk/hip/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dcc/gcvm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip

for PY in /usr/bin/python3 /opt/conda/bin/python3; do
  echo "########## $PY ##########"
  $PY -V
  echo "--- torch location ---"
  $PY -c "
import importlib.util as u
s = u.find_spec('torch')
print('torch_spec', s.origin if s else None)
" 2>&1 | tail -3
  echo "--- import + device test ---"
  $PY -c "
import torch
print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None))
print('avail', torch.cuda.is_available(), 'count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('dev',i,'name',p.name,'mem_GB',round(p.total_memory/1024**3,2),'cc',f'{p.major}.{p.minor}','sm',p.multi_processor_count)
a=torch.randn(1024,1024,device='cuda'); b=a@a; torch.cuda.synchronize()
print('matmul_ok', bool(torch.isfinite(b).all().item()))
free,total=torch.cuda.mem_get_info(0)
print('free_MB',free//1024//1024,'total_MB',total//1024//1024)
" 2>&1 | tail -15
  echo
done

echo "=== sglang / transformers in system python ==="
/usr/bin/python3 -m pip list 2>/dev/null | grep -iE 'torch|transformers|sglang|vllm|accelerate|tokenizers|triton|xformers|decord|timm|flash|datasets'
