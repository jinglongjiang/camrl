"""Re-run the Order 0 baseline and diff it. Any change is a STOP."""
import json, subprocess, sys
import sys
from pathlib import Path

# runnable from anywhere: the repo root is three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path("runs/refactor_baseline")
ref = json.loads((OUT / "baseline.json").read_text())
subprocess.run([sys.executable, str(OUT / "make_baseline.py")], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
got = json.loads((OUT / "baseline.json").read_text())
(OUT / "baseline.json").write_text(json.dumps(ref, indent=2, sort_keys=True))  # keep the reference
drift = {k: (ref[k], got.get(k)) for k in ref if got.get(k) != ref[k]}
if drift:
    print("BASELINE DRIFT -- STOP:")
    for k, (a, b) in drift.items():
        print(f"  {k}: {a} -> {b}")
    raise SystemExit(2)
print(f"baseline unchanged ({len(ref)} values)")
