from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
from src.reproducibility import set_global_seed
from src.demo_run_pipeline import main
if __name__ == "__main__":
    set_global_seed(42)
    main()
