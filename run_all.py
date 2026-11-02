"""Run the full four-part capstone in dependency order."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
commands=[[sys.executable,'generate_data.py'],[sys.executable,'src/part_a.py'],[sys.executable,'src/validate.py'],[sys.executable,'src/part_b.py'],[sys.executable,'src/part_c.py'],[sys.executable,'src/part_d.py']]
for cmd in commands:
    print('\n$',' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
print('\nALL FOUR PARTS COMPLETED')
