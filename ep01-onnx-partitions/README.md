# Count your partitions

From episode 1 of Stubby & Co: **Your AI model is 20× slower on phones. Here's why.**

When ONNX Runtime creates a session, it decides which execution provider (EP) runs which part of your
graph. If the accelerator (NNAPI on Android, Core ML on iOS) can't handle every operator, the graph is
split into partitions, and every boundary means copying tensors between chips. A model in many pieces
can end up slower on the accelerator than on plain CPU.

`count_partitions.py` reads ONNX Runtime's verbose log and tells you how your model was split.

## 1. Get a verbose log

**Android (Java/Kotlin)**, before creating the session:

```java
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
opts.addNnapi();
OrtSession session = env.createSession("model.onnx", opts);
```

Then save the log from the device:

```bash
adb logcat -d > ort_verbose.log
```

**Python (desktop):**

```python
so = ort.SessionOptions()
so.log_severity_level = 0  # verbose
```

## 2. Count

```bash
python count_partitions.py ort_verbose.log
# or straight from the phone:
adb logcat -d | python count_partitions.py -
# or let the script create the session itself (desktop):
python count_partitions.py --model model.onnx --providers XnnpackExecutionProvider CPUExecutionProvider
```

Example output for a model the accelerator split into four pieces
(from `sample_logs/nnapi_split_illustrative.log`):

```
Node placements
----------------------------------------------------
  NNAPI      4 partitions (each one is a fused chunk of your graph)
  CPU        12 nodes  (runs ops one by one)
  NNAPI said it can take 4 partitions, supports 52/64 nodes
----------------------------------------------------
4 accelerator partitions, so up to 8 hand-offs to CPU and back.
Every hand-off copies tensors between memory spaces. Benchmark this against running on
XNNPACK (or CPU) alone before you ship: the accelerator isn't always faster.
```

Exit codes: `0` one piece (good), `1` split across providers, `2` no placement info in the log.
So you can drop it into CI and fail the build when a model starts splitting.

## 3. What to do about a split

- **Find the operators the accelerator rejected.** They're the nodes listed under the CPU provider.
  Replace them in your model if you can (a different op, a different opset, or pre/post-processing
  moved out of the model).
- **Or stop fighting it.** Register only XNNPACK: a fast CPU provider that runs the whole graph
  in one piece. It's often faster end to end than a split accelerator.
- **Always benchmark on the real phones you ship to,** end to end, not just the model call.

## Files

| file | what it is |
|---|---|
| `count_partitions.py` | the script (Python 3.8+, no dependencies; `onnxruntime` only for `--model`) |
| `test_count_partitions.py` | tests: `python test_count_partitions.py` |
| `sample_logs/cpu_only_real.log` | a real verbose log from ONNX Runtime 1.29 on CPU |
| `sample_logs/nnapi_split_illustrative.log` | a hand-written logcat example of a split model; not from a real phone |
| `tiny.onnx` | a four-node test model for `--model` |

## Limits

The log says which provider got which nodes, not the order partitions run in, so the hand-off count
is an upper bound. The NNAPI/Core ML capability line only appears in builds and versions that print
it. If something doesn't parse, open an issue with your log.
