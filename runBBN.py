import pandas as pd
from pathlib import Path
import subprocess

def run_alterbbn(eta=6.1e-10, is_Windows=False, print_output=True):
    check_file = Path("evolution.out")
    if(check_file.is_file()):
        print("Moving file evolution.out to evolution_old.out\n")
        subprocess.run(['mv', "evolution.out", "evolution_old.out"])
    d = subprocess.run(['make', '-C', "AlterBBN", "clean"], capture_output=True, text=True, shell=is_Windows)
    d = subprocess.run(['make', '-C', "AlterBBN", "alter_eta"], capture_output=True, text=True, shell=is_Windows)

    bbn = subprocess.run(["./AlterBBN/alter_eta.x", str(eta)], capture_output=True, text=True, shell=is_Windows)
    if print_output:
        print(bbn.stdout)

def read_bbn_output(filename="evolution.out"):
    res = pd.read_csv(filename, skipinitialspace=True)
    print(res.columns)
    return res