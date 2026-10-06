"""Hyperparameter tuning on the TUNE split only."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    """Grid search of retrieval hyperparameters on TUNE."""
    raise NotImplementedError("B")


if __name__ == "__main__":
    main()
