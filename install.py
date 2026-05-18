"""Install RAGent hooks into the selected coding agent."""

import argparse

from ragent.adapters import ADAPTER_REGISTRY


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
    adapter_cls = ADAPTER_REGISTRY[args.adapter]
    adapter_cls.install()


if __name__ == "__main__":
    main()
