"""BridgeHop live demo: prints the full agent trace for one question (PLAN.md Demo specification)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> None:
    """Render an inspectable multi-hop retrieval trace: --qid | --q | --replay."""
    raise NotImplementedError("A6")


if __name__ == "__main__":
    main()
