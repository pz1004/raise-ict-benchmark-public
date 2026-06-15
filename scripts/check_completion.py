#!/usr/bin/env python
"""Audit whether RAISE-ICT benchmark evidence satisfies completion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from raise_ict.validation import audit_completion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="manifests/completion/benchmark_completion_audit.json")
    parser.add_argument("--raw-results", default="results/tables/tier_p_core4/table_raw_results.csv")
    parser.add_argument("--summary-results", default="results/tables/tier_p_core4/table_main_results.csv")
    parser.add_argument("--split-manifest", default="manifests/splits/tier_p_core4_split_manifest.csv")
    parser.add_argument("--dataset-manifest", default="manifests/dataset_hashes/tier_p_core4_download_manifest.json")
    parser.add_argument("--feature-schema", default="manifests/feature_schemas/tier_p_core4_feature_schema.json")
    parser.add_argument("--hardware-audit", default="manifests/hardware/tier_e_hardware_audit.json")
    parser.add_argument("--profile-manifest", default="manifests/hardware/tier_e_profile_manifest.json")
    parser.add_argument("--manuscript", default="raise_ict_manuscript_scaffold.tex")
    parser.add_argument("--bibliography", default="references.bib")
    parser.add_argument("--require-tier-e", action="store_true")
    parser.add_argument("--require-full-scale-cse", action="store_true")
    parser.add_argument("--expected-raw-rows", type=int, default=240)
    parser.add_argument("--expected-summary-rows", type=int, default=48)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--expected-split-rows", type=int, default=20)
    parser.add_argument("--expected-feature-schema-records", type=int, default=20)
    parser.add_argument(
        "--expected-models",
        nargs="+",
        default=None,
        help="Expected model_id values. Defaults to the classical Core4 baselines.",
    )
    parser.add_argument("--require-timing", action="store_true")
    parser.add_argument("--timing-events", default="results/timing/timing_events.csv")
    parser.add_argument("--timing-summary", default="results/timing/timing_summary.csv")
    parser.add_argument("--command-timeline", default="results/timing/command_timeline.json")
    parser.add_argument("--expected-timing-stages", nargs="+", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required completion checks are incomplete.",
    )
    args = parser.parse_args()

    report = audit_completion(
        raw_results_path=args.raw_results,
        summary_results_path=args.summary_results,
        split_manifest_path=args.split_manifest,
        dataset_manifest_path=args.dataset_manifest,
        feature_schema_path=args.feature_schema,
        hardware_audit_path=args.hardware_audit,
        profile_manifest_path=args.profile_manifest,
        manuscript_path=args.manuscript,
        bibliography_path=args.bibliography,
        require_tier_e=args.require_tier_e,
        require_full_scale_cse=args.require_full_scale_cse,
        expected_raw_rows=args.expected_raw_rows,
        expected_summary_rows=args.expected_summary_rows,
        expected_models=args.expected_models,
        expected_seeds=args.expected_seeds,
        expected_split_rows=args.expected_split_rows,
        expected_feature_schema_records=args.expected_feature_schema_records,
        require_timing=args.require_timing,
        timing_events_path=args.timing_events,
        timing_summary_path=args.timing_summary,
        command_timeline_path=args.command_timeline,
        expected_timing_stages=args.expected_timing_stages,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)
    print(
        f"complete={report['complete']} "
        f"passed={report['summary']['passed']} "
        f"incomplete={report['summary']['incomplete']}"
    )
    if args.strict and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
