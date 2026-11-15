"""Run with:  python -m pytest -q   (or just: python test_count_partitions.py)"""
import io, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from count_partitions import parse, report

HERE = os.path.dirname(os.path.abspath(__file__))

def run(name):
    with open(os.path.join(HERE, "sample_logs", name), encoding="utf-8") as f:
        eps, cap = parse(f.read().splitlines())
    buf = io.StringIO(); code = report(eps, cap, out=buf)
    return eps, cap, code, buf.getvalue()

def test_cpu_only_real_log():
    eps, cap, code, text = run("cpu_only_real.log")
    assert list(eps) == ["CPUExecutionProvider"] and eps["CPUExecutionProvider"]["nodes"] == 4
    assert code == 0 and "everything runs on CPU" in text

def test_nnapi_split():
    eps, cap, code, text = run("nnapi_split_illustrative.log")
    assert eps["NnapiExecutionProvider"]["fused"] == 4
    assert eps["CPUExecutionProvider"]["nodes"] == 12
    assert cap["NnapiExecutionProvider"] == {"partitions": 4, "graph_nodes": 64, "supported_nodes": 52}
    assert code == 1 and "4 accelerator partitions" in text and "8 hand-offs" in text

def test_no_placements():
    eps, cap = parse(["just some text", "[I:onnxruntime:, x.cc:1 y] hello"])
    assert report(eps, cap, out=io.StringIO()) == 2

if __name__ == "__main__":
    for t in (test_cpu_only_real_log, test_nnapi_split, test_no_placements):
        t(); print("ok", t.__name__)
