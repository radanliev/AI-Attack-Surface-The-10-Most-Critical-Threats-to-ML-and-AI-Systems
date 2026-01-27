# demo5_redteam_pipeline.py
# Requirements: numpy, scikit-learn, torch (optional for FGSM)
import subprocess
import sys
import numpy as np

def run_script(path):
    print("Running:", path)
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

# Minimal orchestration: run the other demos in sequence to demonstrate identify->exploit->measure->mitigate
if __name__ == "__main__":
    scripts = [
        "demo1_adversarial_fgsm.py",
        "demo2_data_poisoning.py",
        "demo3_model_extraction_sim.py",
        "demo4_federated_byzantine.py"
    ]
    for s in scripts:
        try:
            run_script(s)
        except Exception as e:
            print("Failed to run", s, ":", str(e))

    print("Pipeline complete. Review outputs above for measurement and mitigation comparisons.")
