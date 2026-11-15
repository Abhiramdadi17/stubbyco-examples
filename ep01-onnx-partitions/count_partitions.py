#!/usr/bin/env python3
"""
Count how ONNX Runtime split your model across execution providers.

Stubby & Co, episode 1: "Your AI model is 20x slower on phones. Here's why."

Two ways to use it:

  1. Read a verbose log you already have (Android logcat, a desktop run, anything):
       python count_partitions.py ort_verbose.log
       adb logcat -d | python count_partitions.py -

  2. Let the script create the session for you and read its own log (desktop Python):
       python count_partitions.py --model model.onnx --providers XnnpackExecutionProvider CPUExecutionProvider

How to get a verbose log:
  Android (Java):  opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
  Python:          so = ort.SessionOptions(); so.log_severity_level = 0

What it reads:
  - The "Node placements" section ONNX Runtime prints at session creation: which execution
    provider (EP) got how many nodes.
  - Fused nodes: compiling EPs such as NNAPI, Core ML or QNN replace each chunk of the graph they
    take with one fused node, so each fused node is one partition on that EP.
  - "number of partitions supported by ..." lines that some EPs print while claiming nodes.

What it can't see:
  The log doesn't record the order the partitions run in, so the number of hand-offs between
  EPs is an estimate (an upper bound on the accelerator side). Benchmark on the real phone to be sure.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict

# "All nodes placed on [CPUExecutionProvider]. Number of nodes: 4"
# " Node(s) placed on [NnapiExecutionProvider]. Number of nodes: 3"
PLACED = re.compile(r"(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)")
# node lines that follow a placement header: "  OpType (node_name)"
NODE = re.compile(r"\]\s{2,}(\S+) \(([^)]*)\)\s*$")
# "NnapiExecutionProvider::GetCapability, number of partitions supported by NNAPI: 3 ..."
CAPABILITY = re.compile(r"(\w+ExecutionProvider)::GetCapability.*?number of partitions supported by \w+: (\d+)"
                        r"(?:.*?number of nodes in the graph: (\d+))?(?:.*?number of nodes supported by \w+: (\d+))?")

# op-type prefixes that compiling EPs give their fused nodes
FUSED_PREFIXES = ("NNAPI_", "CoreML_", "QNN_", "TensorrtExecutionProvider_", "TRTKernel_", "OpenVINO-EP",
                  "DML_", "WebNN_", "VitisAI", "MIGraphX_", "CANN_", "SNPE_")
# EPs that run individual kernels, so every node it takes is its own unit of work
KERNEL_EPS = {"CPUExecutionProvider", "XnnpackExecutionProvider", "CUDAExecutionProvider",
              "ROCMExecutionProvider", "JsExecutionProvider", "WebGpuExecutionProvider"}

SHORT = {"CPUExecutionProvider": "CPU", "XnnpackExecutionProvider": "XNNPACK", "NnapiExecutionProvider": "NNAPI",
         "CoreMLExecutionProvider": "Core ML", "QNNExecutionProvider": "QNN", "CUDAExecutionProvider": "CUDA",
         "TensorrtExecutionProvider": "TensorRT", "DmlExecutionProvider": "DirectML"}


def parse(lines):
    eps: "OrderedDict[str, dict]" = OrderedDict()
    capability = {}
    current = None
    for raw in lines:
        line = raw.rstrip("\r\n")
        m = PLACED.search(line)
        if m:
            current = m.group(2)
            eps[current] = {"nodes": int(m.group(3)), "ops": [], "all": m.group(1) == "All nodes"}
            continue
        m = CAPABILITY.search(line)
        if m:
            capability[m.group(1)] = {k: (int(v) if v else None) for k, v in
                                      zip(("partitions", "graph_nodes", "supported_nodes"), m.groups()[1:])}
            continue
        if current:
            m = NODE.search(line)
            if m and "placed on" not in line:
                eps[current]["ops"].append(m.group(1))
                continue
            current = None  # placement block ended
    for name, info in eps.items():
        info["fused"] = sum(op.startswith(FUSED_PREFIXES) for op in info["ops"])
    return eps, capability


def report(eps, capability, out=sys.stdout):
    if not eps:
        print("No 'Node placements' section found.\n"
              "Make sure the log is VERBOSE (Java: ORT_LOGGING_LEVEL_VERBOSE, Python: log_severity_level = 0)\n"
              "and that it covers session creation, not just inference.", file=out)
        return 2
    total = sum(e["nodes"] for e in eps.values())
    print("Node placements", file=out)
    print("-" * 52, file=out)
    any_fused = any(e["fused"] for e in eps.values())
    for name, e in eps.items():
        label = SHORT.get(name, name)
        if e["fused"]:
            line = f"{e['fused']} partition{'s' if e['fused'] != 1 else ''} (each one is a fused chunk of your graph)"
        else:
            share = "" if any_fused else f"  {100 * e['nodes'] / total:5.1f}%"
            line = f"{e['nodes']} node{'s' if e['nodes'] != 1 else ''}{share}"
            if name in KERNEL_EPS and len(eps) > 1:
                line += "  (runs ops one by one)"
        print(f"  {label:<10} {line}", file=out)
    for name, c in capability.items():
        label = SHORT.get(name, name)
        parts = f"{c['partitions']} partition{'s' if c['partitions'] != 1 else ''}"
        cov = ""
        if c["graph_nodes"] and c["supported_nodes"] is not None:
            cov = f", supports {c['supported_nodes']}/{c['graph_nodes']} nodes"
        print(f"  {label} said it can take {parts}{cov}", file=out)
    print("-" * 52, file=out)

    accel = {n: e for n, e in eps.items() if e["fused"]}
    accel_parts = sum(e["fused"] for e in accel.values())
    for n, c in capability.items():
        if n not in accel and c["partitions"]:
            accel_parts = max(accel_parts, c["partitions"])
    fallback = [n for n in eps if n in KERNEL_EPS]

    if len(eps) == 1:
        only = SHORT.get(next(iter(eps)), next(iter(eps)))
        print(f"One provider, one piece: everything runs on {only}.", file=out)
        if only == "CPU":
            print("If you registered an accelerator, it didn't take anything. Check the provider order"
                  " and that the EP is actually in your build.", file=out)
        return 0
    if accel_parts <= 1:
        print("One accelerator partition. That's the goal: no tensors bouncing between chips.", file=out)
        return 0
    boundaries = 2 * accel_parts if fallback else accel_parts - 1
    print(f"{accel_parts} accelerator partitions, so up to {boundaries} hand-offs to "
          f"{', '.join(SHORT.get(f, f) for f in fallback) or 'other providers'} and back.", file=out)
    print("Every hand-off copies tensors between memory spaces. Benchmark this against running on\n"
          "XNNPACK (or CPU) alone before you ship: the accelerator isn't always faster.", file=out)
    return 1


def run_session(model, providers):
    """Create a session in a child process with verbose logging and return its log lines."""
    code = (
        "import onnxruntime as ort, sys\n"
        "so = ort.SessionOptions(); so.log_severity_level = 0\n"
        f"ort.InferenceSession({model!r}, so, providers={providers!r})\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ)
    if p.returncode != 0:
        sys.exit(f"session creation failed:\n{p.stderr[-2000:]}")
    return (p.stderr + p.stdout).splitlines()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count ONNX Runtime partitions per execution provider.")
    ap.add_argument("log", nargs="?", help="verbose log file, or - for stdin")
    ap.add_argument("--model", help="create a session for this .onnx file and read its log")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                    help="execution providers in priority order (with --model)")
    a = ap.parse_args(argv)
    if a.model:
        lines = run_session(a.model, a.providers)
    elif a.log == "-" or (a.log is None and not sys.stdin.isatty()):
        lines = sys.stdin.read().splitlines()
    elif a.log:
        with open(a.log, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    else:
        ap.error("give a log file, pipe one in, or use --model")
    eps, cap = parse(lines)
    return report(eps, cap)


if __name__ == "__main__":
    sys.exit(main())
