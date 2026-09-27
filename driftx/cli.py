"""DRIFTX command-line surface.

The existing ``da3`` command remains available for compatibility. This small
product-facing entry point is intentionally dependency-light so ``driftx
--help`` works before model or GPU dependencies are loaded.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the DRIFTX product CLI parser."""
    return argparse.ArgumentParser(
        prog="driftx",
        description="DRIFTX geometry engine for single-pass UAV video reconstruction.",
    )


def main() -> int:
    """Run the DRIFTX product CLI."""
    build_parser().parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
