"""Install RAGent hooks into the selected coding agent."""

import argparse

from ragent.adapters import ADAPTER_REGISTRY
from ragent.utils import pause_if_frozen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install RAGent hooks into the selected coding agent."
    )
    parser.add_argument(
        "--adapter",
        default="claude_code",
        choices=sorted(ADAPTER_REGISTRY.keys()),
        help="Which agent adapter to install RAGent into (default: claude_code)",
    )
    args = parser.parse_args()

    try:
        adapter_cls = ADAPTER_REGISTRY[args.adapter]
        adapter_cls.install()
    except Exception:
        import traceback
        traceback.print_exc()

    pause_if_frozen()


if __name__ == "__main__":
    main()
