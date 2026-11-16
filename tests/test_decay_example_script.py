import runpy
from pathlib import Path

def test_script_runs_and_asserts():
    runpy.run_path(str(Path(__file__).parents[1]/'examples/decay_example.py'),run_name='__main__')
