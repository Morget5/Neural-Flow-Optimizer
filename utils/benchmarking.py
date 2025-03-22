import time
import torch
import numpy as np

def profile_inference(model, input_tensor, iterations=100):
    """
    Profiles the inference latency and memory footprint of a model.
    """
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            
    # Benchmark
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            latencies.append(time.perf_counter() - start_time)
            
    avg_latency = np.mean(latencies) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    return avg_latency, p95_latency
