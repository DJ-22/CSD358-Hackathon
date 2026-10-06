"""Build and pickle the positional zone index for a corpus (PLAN.md §2)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    """Index construction: --corpus dev|train [--no-stem]."""
    raise NotImplementedError("A2")


if __name__ == "__main__":
    main()
