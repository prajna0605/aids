"""
Verification script to run the code cells of ecg-heartbeat-categorization-using-cnn.ipynb
Executes the notebook workflow from cell 1 through cell 33 (with epochs=2 for fast test).
"""
import json
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless test

with open('ecg-heartbeat-categorization-using-cnn.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

global_scope = {}

print("Testing notebook cells execution...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        if not code.strip():
            continue
        # For testing, reduce epochs to 2 so execution is fast
        if "epochs = 10" in code:
            code = code.replace("epochs = 10", "epochs = 2")
        print(f"\n--- Running Cell {i} ---")
        try:
            exec(code, global_scope)
            print(f"Cell {i} passed.")
        except Exception as e:
            print(f"FAILED AT CELL {i}: {e}")
            raise e

print("\nALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY WITHOUT ANY ERRORS!")
