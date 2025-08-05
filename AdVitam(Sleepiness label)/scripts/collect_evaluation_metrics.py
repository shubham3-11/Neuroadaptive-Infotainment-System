import json
from pathlib import Path
import re
from typing import Any, Dict


def load_metrics(file_path: Path) -> Dict[str, Any]:
    """Load JSON metrics file safely."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return {}


def main() -> None:
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ 'results' directory not found in the current working directory.")
        return

    header = f"{'Folder':<70}  {'MSE':>10}  {'RMSE':>10}  {'MAE':>10}  {'R²':>10}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    # Track best runs
    best_mse: float = float("inf")
    best_mse_run: str | None = None
    best_r2: float = float("-inf")
    best_r2_run: str | None = None

    def _sort_key(path: Path) -> tuple[int, str]:
        """Sort by leading integer in the folder name, fallback to name itself."""
        match = re.match(r"(\d+)", path.name)
        if match:
            return int(match.group(1)), path.name
        return float("inf"), path.name

    for subdir in sorted(results_dir.iterdir(), key=_sort_key):
        if not subdir.is_dir():
            continue

        eval_file = subdir / "evaluation" / "evaluation_metrics.json"
        if not eval_file.exists():
            print(f"{subdir.name:<70}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
            continue

        metrics = load_metrics(eval_file)
        mse = metrics.get("mse", float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        mae = metrics.get("mae", float("nan"))
        r2 = metrics.get("r2", float("nan"))

        # Update best MSE (lower is better)
        try:
            mse_val = float(mse)
            if mse_val < best_mse:
                best_mse = mse_val
                best_mse_run = subdir.name
        except (TypeError, ValueError):
            pass  # Ignore non-numeric values

        # Update best R² (higher is better)
        try:
            r2_val = float(r2)
            if r2_val > best_r2:
                best_r2 = r2_val
                best_r2_run = subdir.name
        except (TypeError, ValueError):
            pass  # Ignore non-numeric values

        def fmt(value: Any) -> str:
            try:
                return f"{float(value):10.4f}"
            except (TypeError, ValueError):
                return f"{str(value):>10}"

        print(
            f"{subdir.name:<70}  {fmt(mse)}  {fmt(rmse)}  {fmt(mae)}  {fmt(r2)}"
        )

    # Print summary of best runs
    print("\nBest Runs:")
    if best_mse_run is not None:
        print(f"  • Lowest MSE : {best_mse_run}  (MSE={best_mse:.4f})")
    if best_r2_run is not None and best_r2_run != best_mse_run:
        print(f"  • Highest R² : {best_r2_run}  (R²={best_r2:.4f})")


if __name__ == "__main__":
    main() 