import os

scripts_to_run = ['main_blca.py', 'main_brca.py', 'main_gbmlgg.py', 'main_luad.py', 'main_ucec.py']

for script in scripts_to_run:
    os.system(f'python {script}')