"""Run S0 and S5 on the team's hand-judged own queries and report P@2 / P@10."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    """Precision@k on own judged queries."""
    raise NotImplementedError("A6")


if __name__ == "__main__":
    main()
