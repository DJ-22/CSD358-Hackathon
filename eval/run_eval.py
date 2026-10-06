"""Evaluation harness: run systems over EVAL/TUNE and write results."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    """Batch evaluation of retrieval systems."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
