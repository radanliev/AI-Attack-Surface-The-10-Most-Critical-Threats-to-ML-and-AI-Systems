import os

files = [
    "01_adversarial_evasion.py",
    "02_data_supply_chain.py",
    "03_model_extraction.py",
    "04_rag_poisoning.py",
    "05_prompt_injection.py",
    "06_insecure_output.py",
    "07_excessive_agency.py",
    "08_prompt_leakage.py",
    "09_social_engineering.py",
    "10_denial_of_wallet.py"
]

for f in files:
    print(f"\n=== Running {f} ===")
    os.system(f"python {f}")
