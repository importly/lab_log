import numpy as np
from lab_log import LabLog
import time
import math

print("Starting high-volume logging experiment...")

with LabLog(trial_name="scaling_test", experiment_id="optics_v1") as logger:
    logger.configure(sync_interval=5)
    logger.declare_channel("voltage_hf", dtype="f64", unit="V")
    logger.declare_channel("temperature", dtype="f64", unit="C")

    n_points = 500000
    
    print(f"Logging {n_points} points...")
    for i in range(n_points):
        v = math.sin(i * 0.1) * 2.5 + 2.5 + (np.random.randn() * 0.2)
        logger.log("voltage_hf", float(v))
        t = 20.0 + (i * 0.005) + (np.random.randn() * 0.05)
        logger.log("temperature", float(t))
        if i % 100 == 0:
            time.sleep(0.01)
            print(f"Logged {i}/{n_points}...")

    print(f"Run completed: {logger.run_id}")