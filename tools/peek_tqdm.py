#!/usr/bin/env python3
# 读取官方 summarize_acc.py 子进程的 stderr 管道缓冲（tqdm 进度条），
# 不消耗 stdout（stdout 只有末尾的指标行，父进程还要用）。
# 用法: python3 peek_tqdm.py <parent_pid>
import os
import sys

pid = int(sys.argv[1])
pdir = "/proc/%d/fd" % pid
peers = {}
for fd in os.listdir(pdir):
    try:
        link = os.readlink(os.path.join(pdir, fd))
    except OSError:
        continue
    peers[fd] = link

# 找出所有 pipe
pipes = {fd: l for fd, l in peers.items() if l.startswith("pipe:")}
print("parent %d 的 pipe fd: %s" % (pid, pipes))

# 逐个 pipe 读一点，看哪个是 tqdm 进度
for fd in sorted(pipes):
    path = "/proc/%d/fd/%s" % (pid, fd)
    try:
        fd_ = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        print("  fd=%s open 失败 %s" % (fd, e))
        continue
    try:
        chunks = []
        while True:
            try:
                b = os.read(fd_, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not b:
                break
            chunks.append(b)
        data = b"".join(chunks)
    finally:
        os.close(fd_)
    if not data:
        continue
    txt = data.decode("utf-8", "replace")
    if "Evaluating" in txt or "it/s" in txt or "%|" in txt:
        lines = [x for x in txt.replace("\r", "\n").split("\n") if x.strip()]
        print("  fd=%s 疑似 tqdm 输出，末 5 条:" % fd)
        for l in lines[-5:]:
            print("     ", l[:160])
    else:
        print("  fd=%s 非 tqdm，前 200 字: %s" % (fd, txt[:200].replace("\n", " | ")))
