"""可重現性檢查：同一主種子跑兩次應逐位元相同；不同主種子應不同。"""
import hashlib, json, os, subprocess, sys, tempfile
here = os.path.dirname(__file__)
def run(seed):
    d = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(here, "benchmark.py"), "--seed", str(seed), "--sizes", "8", "--instances", "2",
                    "--inits", "2", "--out", d], check=True, capture_output=True)
    return hashlib.sha256(open(os.path.join(d, f"benchmark_seed{seed}.json"), "rb").read()).hexdigest()
a, b, c = run(5), run(5), run(6)
print("seed5 run1:", a[:16]); print("seed5 run2:", b[:16]); print("seed6     :", c[:16])
assert a == b, "同種子結果不一致（不可重現）"
assert a != c, "不同種子結果相同（種子未生效）"
print("OK：同種子可重現、不同種子不同")
