#!/usr/bin/env python
"""Reviewer-facing CLI for the claim-conditioned pairwise admission checker."""

from __future__ import annotations

import sys

from pairwise_admission import main


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"pairwise admission check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
