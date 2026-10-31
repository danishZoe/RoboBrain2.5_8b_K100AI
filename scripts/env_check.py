import sys, os, subprocess

print("=== python ===")
print(sys.version)

try:
    import torch
    print("=== torch ===")
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("torch.version.hip", getattr(torch.version, "hip", None))
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("device", i, "name", p.name,
              "mem_MB", p.total_memory // 1024 // 1024,
              "cc", "%d.%d" % (p.major, p.minor),
              "sm", getattr(p, "multi_processor_count", None))
    if torch.cuda.is_available():
        torch.cuda.init()
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print("matmul_ok", bool(torch.isfinite(c).all().item()),
              "out_sum", float(c.float().abs().sum().item()))
        free, total = torch.cuda.mem_get_info(0)
        print("mem_free_MB", free // 1024 // 1024, "mem_total_MB", total // 1024 // 1024)
except Exception as e:
    print("TORCH_CHECK_ERROR", type(e).__name__, e)

for mod in ["transformers", "sglang", "accelerate", "tokenizers", "PIL", "vllm", "flash_attn", "pynvml", "amdsmi"]:
    try:
        m = __import__(mod)
        print("pkg", mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print("pkg", mod, "MISSING", type(e).__name__)

print("=== dtk dirs ===")
for d in ["/opt/hyhal", "/opt/dtk", "/opt/dtk/dtk", "/usr/local/dtk"]:
    print(d, os.path.isdir(d))
for f in ["/opt/dtk/version", "/opt/hyhal/version", "/usr/local/dtk/version"]:
    if os.path.exists(f):
        print("--- ", f)
        print(open(f).read().strip()[:500])

print("=== devices ===")
for d in ["/dev/kfd", "/dev/dri"]:
    try:
        print(d, os.listdir(d))
    except Exception as e:
        print(d, "ERR", e)
