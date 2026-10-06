"""Download HotpotQA distractor Parquet and build corpora, splits and LTR questions (PLAN.md §D)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    """Build the document collection (corpus), qrels and EVAL/TUNE splits."""
    raise NotImplementedError("A1")


if __name__ == "__main__":
    main()
