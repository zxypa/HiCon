"""Create a small CSV so users can verify the end-to-end CLI with no external data."""
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicon.train import synthetic_demo

out = Path("data/demo.csv")
out.parent.mkdir(exist_ok=True, parents=True)
pd.DataFrame(synthetic_demo(540, 3), columns=["a", "b", "c"]).to_csv(out, index=False)
print(out)
