import torch, os, platform

print("host        :", platform.node())
print("python      :", sys.version.split()[0] if (sys := __import__("sys")) else "?")
print("torch       :", torch.__version__)
print("torch.hip   :", getattr(torch.version, "hip", None))
print("cuda_avail  :", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("-" * 60)
    print("  index      :", i)
    print("  name       :", p.name)
    print("  gcnArchName:", getattr(p, "gcnArchName", "n/a"))
    print("  vram_GB    :", round(p.total_memory / 1024 ** 3, 2))
    print("  sm_count   :", p.multi_processor_count)
    print("  mem_clock  :", getattr(p, "memory_clock_rate", "n/a"))
    print("  mem_bus    :", getattr(p, "memory_bus_width", "n/a"))

# real compute on the device
torch.cuda.init()
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print("-" * 60)
print("fp16 matmul ok:", bool(torch.isfinite(c).all().item()),
      "sum=%.4f" % float(c.float().abs().sum().item()))
free, total = torch.cuda.mem_get_info(0)
print("vram free_MB :", free // 1024 // 1024, "/ total_MB", total // 1024 // 1024)
print("peak_MB      :", torch.cuda.max_memory_allocated(0) // 1024 // 1024)
