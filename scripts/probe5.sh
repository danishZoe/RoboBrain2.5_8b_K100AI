export PATH=/opt/hyhal/bin:/opt/dtk/bin:/opt/dtk/hip/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dcc/gcvm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip

echo "=== PCI device ==="
for d in /sys/bus/pci/devices/*; do
  v=$(cat $d/vendor 2>/dev/null); dd=$(cat $d/device 2>/dev/null)
  if [ "$v" = "0x1d94" ] || [ "$dd" = "0x1100" ]; then
    echo "dev $d vendor=$v device=$dd"
    cat $d/uevent 2>/dev/null | head -8
  fi
done
echo "--- all pci ids (vendor:device class) ---"
for d in /sys/bus/pci/devices/*; do
  printf "%s %s %s %s\n" "$(basename $d)" "$(cat $d/vendor)" "$(cat $d/device)" "$(cat $d/class)"
done

echo "=== hy-smi fan/temp/power ==="
hy-smi 2>&1 | head -20

echo "=== torch test (system python3) ==="
python3 -c "
import torch
print('torch', torch.__version__)
print('hip', getattr(torch.version,'hip',None))
print('avail', torch.cuda.is_available(), 'count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('dev',i,'name',p.name,'mem_GB',round(p.total_memory/1024**3,2),'cc',f'{p.major}.{p.minor}','sm',p.multi_processor_count)
a=torch.randn(1024,1024,device='cuda'); b=a@a; torch.cuda.synchronize()
print('matmul_ok', bool(torch.isfinite(b).all().item()))
free,total=torch.cuda.mem_get_info(0)
print('free_MB',free//1024//1024,'total_MB',total//1024//1024)
print('arch_list', torch.cuda.get_arch_list())
" 2>&1 | tail -25

echo "=== key pkgs ==="
python3 -m pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|transformers|sglang|vllm|accelerate|tokenizers|flash|triton|xformers|decord|av|qwen|timm|einops|sentencepiece|datasets) ' 
