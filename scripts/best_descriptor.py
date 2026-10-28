"""Print the descriptor selected by the sanity gate (results/gate.json); exit 2 if the gate failed."""
import json
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "results" / "gate.json"
if not p.exists():
    print("cnn")
    sys.exit(0)
g = json.loads(p.read_text())
if not g.get("passed"):
    sys.exit(2)
print(g["best_descriptor"])
