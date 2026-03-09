import onnxruntime
import numpy as np
import time
import os

def raw_benchmark():
    model_path = "./model/Fun-ASR-Nano-CTC.fp16.onnx"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 1. ORT Session Setup
    session_opts = onnxruntime.SessionOptions()
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.enable_profiling = True
    
    providers = ['DmlExecutionProvider']
    sess = onnxruntime.InferenceSession(model_path, sess_options=session_opts, providers=providers)

    
    input_name = sess.get_inputs()[0].name
    
    # 2. Prepare Data (30s = 510 frames)
    frames = 510
    
    # 自动推断输入数据类型
    in_type = sess.get_inputs()[0].type
    if 'float16' in in_type:
        input_dtype = np.float16
    else:
        input_dtype = np.float32
        
    print(f"Detected Input Type: {in_type} -> NumPy: {input_dtype}")
    fake_input = np.random.randn(1, frames, 512).astype(input_dtype)
    
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Input Name: {input_name}")
    print(f"Input Shape: {fake_input.shape}")
    print(f"Backend: {sess.get_providers()}")

    # 3. Warmup
    print("\nWarmup...")
    for _ in range(5):
        sess.run(None, {input_name: fake_input})
    
    # 4. Benchmark
    print("\nBenchmarking...")
    latencies = []
    for i in range(2):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: fake_input})
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        latencies.append(ms)
        print(f"  Round {i+1:2d}: {ms:.2f} ms")

    print("\n" + "="*30)
    print(f"  Min: {min(latencies):.2f} ms")
    print(f"  Max: {max(latencies):.2f} ms")
    print(f"  Avg: {np.mean(latencies):.2f} ms")
    print("="*30)

if __name__ == "__main__":
    raw_benchmark()
