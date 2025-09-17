import pandas as pd
from pathlib import Path
import subprocess

def run_alterbbn(is_Windows=False, print_output=True):
    check_file = Path("evolution.out")
    if(check_file.is_file()):
        subprocess.run(['rm', "evolution.out"])
    d = subprocess.run(['make', '-C', "AlterBBN", "clean"], capture_output=True, text=True, shell=is_Windows)
    d = subprocess.run(['make', '-C', "AlterBBN", "alter_bbn"], capture_output=True, text=True, shell=is_Windows)

    bbn = subprocess.run(["./AlterBBN/alter_bbn.x", "32"], capture_output=True, text=True, shell=is_Windows)
    if print_output:
        print(bbn.stdout)

def read_bbn_output(filename="evolution.out"):
    res = pd.read_csv(filename, skipinitialspace=True)
    print(res.columns)
    return res