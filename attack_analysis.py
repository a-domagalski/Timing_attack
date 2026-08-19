# code from https://blog.zksecurity.xyz/posts/timing-kocher/
import numpy as np
import random
import time
import gc
from custom_rsa import CustomRSA

def iqr_filter(data, factor=1):
    if len(data) < 4:
        return data
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    if iqr == 0:
        return data.tolist()
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = data[(data >= lower) & (data <= upper)]
    if len(filtered) < len(data) * 0.5:
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        filtered = data[(data >= lower) & (data <= upper)]
    return filtered.tolist()


def measure_timing_simple(operation, amplification=5000, num_runs=10):
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        times = []
        for _ in range(10):
            operation()
        for run in range(num_runs):
            if run > 0:
                time.sleep(0.001)
            start = time.perf_counter()
            for _ in range(amplification):
                operation()
            elapsed = time.perf_counter() - start
            times.append((elapsed * 1e6) / amplification)
        times_clean = iqr_filter(times)
        return np.median(times_clean)
    finally:
        if gc_was_enabled:
            gc.enable()

measure_timing_simple(CustomRSA().encrypt(""))