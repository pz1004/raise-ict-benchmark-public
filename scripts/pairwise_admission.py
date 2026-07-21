#!/usr/bin/env python
"""Implement claim-conditioned pairwise admission for IDS result rows.

The pairwise interface distinguishes scientific abstention from malformed input:
known invariant differences yield ``context_mismatch``; unresolved required
evidence yields ``insufficient_evidence``; otherwise the reported ordering is
``defined`` for the declared descriptive context.

The reviewer-facing CLI is ``scripts/check_pairwise_admission.py``. The
withdrawn single-row external screen is intentionally not implemented here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EVIDENCE_STATES = {"resolved", "unresolved", "not_applicable"}
FIELD_STATES = {"matched", "mismatched", "unresolved", "not_applicable"}
DECISIONS = {"defined", "context_mismatch", "insufficient_evidence"}

PAIR_COLUMNS = [
    "pair_id",
    "context_id",
    "row_a",
    "row_b",
    "metric",
    "claim",
    "decision",
    "mismatched_fields",
    "unresolved_fields",
    "not_applicable_fields",
    "hypothesized_decision",
    "hypothesis_match",
]

def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    items = _require_list(value, label)
    result = [str(item).strip() for item in items]
    if any(not item for item in result):
        raise ValueError(f"{label} contains an empty field name")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate field names")
    return result


def _nonempty(mapping: dict[str, Any], key: str, label: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} is missing required key {key!r}")
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, list):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key).casefold(), _canonical(item)) for key, item in value.items()))
    return value


def load_contexts(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], str]:
    payload = _require_mapping(_load_yaml(path), "context file")
    protocol_id = _nonempty(payload, "protocol_id", "context file")
    groups_raw = _require_mapping(payload.get("ablation_groups"), "ablation_groups")
    groups = {
        str(group_id): _string_list(fields, f"ablation group {group_id!r}")
        for group_id, fields in groups_raw.items()
    }
    field_to_group: dict[str, str] = {}
    for group_id, fields in groups.items():
        for field in fields:
            if field in field_to_group:
                raise ValueError(
                    f"field {field!r} occurs in both {field_to_group[field]!r} and {group_id!r}"
                )
            field_to_group[field] = group_id

    contexts: dict[str, dict[str, Any]] = {}
    for raw in _require_list(payload.get("contexts"), "contexts"):
        context = _require_mapping(raw, "context")
        context_id = _nonempty(context, "context_id", "context")
        if context_id in contexts:
            raise ValueError(f"duplicate context_id {context_id!r}")
        comparison_axis = _nonempty(context, "comparison_axis", context_id)
        outcome_field = _nonempty(context, "outcome_field", context_id)
        if comparison_axis == outcome_field:
            raise ValueError(f"{context_id} uses the same comparison and outcome field")

        required_invariants = _string_list(
            context.get("required_invariants"), f"{context_id}.required_invariants"
        )
        required_resolved = _string_list(
            context.get("required_resolved"), f"{context_id}.required_resolved"
        )
        optional_fields = _string_list(context.get("optional_fields", []), f"{context_id}.optional_fields")
        allowed_na = _string_list(
            context.get("allowed_not_applicable", []), f"{context_id}.allowed_not_applicable"
        )
        metric_identity = _string_list(
            context.get("metric_identity_fields"), f"{context_id}.metric_identity_fields"
        )
        allowed_metrics = _string_list(context.get("allowed_metrics"), f"{context_id}.allowed_metrics")
        if not metric_identity:
            raise ValueError(f"{context_id}.metric_identity_fields must not be empty")
        if "metric_name" not in metric_identity:
            raise ValueError(f"{context_id}.metric_identity_fields must include 'metric_name'")
        if not allowed_metrics:
            raise ValueError(f"{context_id}.allowed_metrics must not be empty")

        named_sets = {
            "required_invariants": set(required_invariants),
            "required_resolved": set(required_resolved),
            "optional_fields": set(optional_fields),
        }
        names = list(named_sets)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                overlap = named_sets[first] & named_sets[second]
                if overlap:
                    raise ValueError(f"{context_id} places {sorted(overlap)} in both {first} and {second}")
        if comparison_axis in set().union(*named_sets.values()):
            raise ValueError(f"{context_id} comparison axis {comparison_axis!r} is also a context field")
        if outcome_field in set().union(*named_sets.values()):
            raise ValueError(f"{context_id} outcome field {outcome_field!r} is also a context field")
        if not set(metric_identity).issubset(required_invariants):
            raise ValueError(f"{context_id} metric identity fields must be required invariants")
        if comparison_axis in allowed_na or outcome_field in allowed_na:
            raise ValueError(f"{context_id} cannot allow not_applicable for its axis or outcome")
        metric_na = set(metric_identity) & set(allowed_na)
        if metric_na:
            raise ValueError(
                f"{context_id} cannot allow not_applicable for metric identity fields "
                f"{sorted(metric_na)}"
            )

        normalized = dict(context)
        normalized.update(
            {
                "comparison_axis": comparison_axis,
                "outcome_field": outcome_field,
                "required_invariants": required_invariants,
                "required_resolved": required_resolved,
                "optional_fields": optional_fields,
                "allowed_not_applicable": allowed_na,
                "metric_identity_fields": metric_identity,
                "allowed_metrics": [_canonical(metric) for metric in allowed_metrics],
            }
        )
        contexts[context_id] = normalized
    if not contexts:
        raise ValueError("context file contains no contexts")
    return contexts, groups, protocol_id


def load_rows(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = _require_mapping(_load_yaml(path), "row file")
    protocol_id = _nonempty(payload, "protocol_id", "row file")
    completion_status = str(payload.get("completion_status", "complete")).strip()
    if completion_status != "complete":
        raise ValueError(f"row file must have completion_status=complete, found {completion_status!r}")
    rows: dict[str, dict[str, Any]] = {}
    for raw in _require_list(payload.get("rows"), "rows"):
        row = _require_mapping(raw, "row")
        row_id = _nonempty(row, "row_id", "row")
        if row_id in rows:
            raise ValueError(f"duplicate row_id {row_id!r}")
        for key in ["source_id", "source_url", "source_locator"]:
            _nonempty(row, key, row_id)
        _require_mapping(row.get("fields"), f"{row_id}.fields")
        rows[row_id] = row
    if not rows:
        raise ValueError("row file contains no rows")
    return rows, protocol_id


def load_pairs(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = _require_mapping(_load_yaml(path), "pair file")
    protocol_id = _nonempty(payload, "protocol_id", "pair file")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for raw in _require_list(payload.get("pairs"), "pairs"):
        pair = _require_mapping(raw, "pair")
        pair_id = _nonempty(pair, "pair_id", "pair")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        for key in ["context_id", "row_a", "row_b", "metric", "claim"]:
            _nonempty(pair, key, pair_id)
        requested_metric = _canonical(pair["metric"])
        hypothesis = str(pair.get("hypothesized_decision", "")).strip()
        if hypothesis and hypothesis not in DECISIONS:
            raise ValueError(f"{pair_id} has unknown hypothesized decision {hypothesis!r}")
        pairs.append({**pair, "metric": requested_metric})
    if not pairs:
        raise ValueError("pair file contains no pairs")
    return pairs, protocol_id


def _field_entry(row: dict[str, Any], field: str, allowed_na: set[str]) -> dict[str, Any]:
    row_id = str(row["row_id"])
    fields = _require_mapping(row.get("fields"), f"{row_id}.fields")
    entry = _require_mapping(fields.get(field), f"{row_id}.fields.{field}")
    status = str(entry.get("status", "")).strip().lower()
    if status not in EVIDENCE_STATES:
        raise ValueError(f"{row_id}.{field} has invalid evidence status {status!r}")
    evidence_anchor = str(entry.get("evidence_anchor", "")).strip()
    if not evidence_anchor:
        raise ValueError(f"{row_id}.{field} is missing an evidence anchor")
    value = entry.get("value")
    if status == "resolved" and (value is None or (isinstance(value, str) and not value.strip())):
        raise ValueError(f"{row_id}.{field} is resolved but has no value")
    if status == "not_applicable":
        if field not in allowed_na:
            raise ValueError(f"{row_id}.{field} uses unauthorized not_applicable")
        if value not in (None, ""):
            raise ValueError(f"{row_id}.{field} is not_applicable but still has a value")
    return {**entry, "status": status}


def _compare_field(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    field: str,
    *,
    require_equal: bool,
    allowed_na: set[str],
) -> str:
    entry_a = _field_entry(row_a, field, allowed_na)
    entry_b = _field_entry(row_b, field, allowed_na)
    statuses = {entry_a["status"], entry_b["status"]}
    if statuses == {"not_applicable"}:
        return "not_applicable"
    if "not_applicable" in statuses:
        return "mismatched"
    if "unresolved" in statuses:
        return "unresolved"
    if require_equal and _canonical(entry_a.get("value")) != _canonical(entry_b.get("value")):
        return "mismatched"
    return "matched"


def _evaluate_pair(
    pair: dict[str, Any],
    context: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_id = str(pair["pair_id"])
    row_a_id = str(pair["row_a"])
    row_b_id = str(pair["row_b"])
    if row_a_id == row_b_id:
        raise ValueError(f"{pair_id} compares a row with itself")
    if row_a_id not in rows or row_b_id not in rows:
        raise ValueError(f"{pair_id} references an unknown row")
    row_a = rows[row_a_id]
    row_b = rows[row_b_id]
    allowed_na = set(context["allowed_not_applicable"])
    field_states: dict[str, str] = {}

    requested_metric = _canonical(pair["metric"])
    if requested_metric not in set(context["allowed_metrics"]):
        raise ValueError(
            f"{pair_id} requests unsupported metric {pair['metric']!r} "
            f"for context {context['context_id']!r}"
        )

    axis = str(context["comparison_axis"])
    axis_a = _field_entry(row_a, axis, allowed_na)
    axis_b = _field_entry(row_b, axis, allowed_na)
    if "not_applicable" in {axis_a["status"], axis_b["status"]}:
        raise ValueError(f"{pair_id} uses not_applicable on comparison axis {axis!r}")
    if "unresolved" in {axis_a["status"], axis_b["status"]}:
        field_states[axis] = "unresolved"
    elif _canonical(axis_a.get("value")) == _canonical(axis_b.get("value")):
        raise ValueError(f"{pair_id} has identical comparison-axis values for {axis!r}")
    else:
        field_states[axis] = "matched"

    outcome = str(context["outcome_field"])
    outcome_a = _field_entry(row_a, outcome, allowed_na)
    outcome_b = _field_entry(row_b, outcome, allowed_na)
    if "not_applicable" in {outcome_a["status"], outcome_b["status"]}:
        raise ValueError(f"{pair_id} uses not_applicable on outcome field {outcome!r}")
    field_states[outcome] = (
        "unresolved"
        if "unresolved" in {outcome_a["status"], outcome_b["status"]}
        else "matched"
    )

    for field in context["required_invariants"]:
        field_states[field] = _compare_field(
            row_a, row_b, field, require_equal=True, allowed_na=allowed_na
        )
    for field in context["required_resolved"]:
        field_states[field] = _compare_field(
            row_a, row_b, field, require_equal=False, allowed_na=allowed_na
        )

    metric_field = "metric_name"
    metric_entries = [
        _field_entry(row, metric_field, allowed_na)
        for row in [row_a, row_b]
    ]
    if any(entry["status"] == "unresolved" for entry in metric_entries):
        field_states[metric_field] = "unresolved"
    elif any(_canonical(entry.get("value")) != requested_metric for entry in metric_entries):
        field_states[metric_field] = "mismatched"
    else:
        field_states[metric_field] = "matched"

    for state in field_states.values():
        if state not in FIELD_STATES:
            raise AssertionError(f"unexpected field state {state!r}")
    mismatched = sorted(field for field, state in field_states.items() if state == "mismatched")
    unresolved = sorted(field for field, state in field_states.items() if state == "unresolved")
    not_applicable = sorted(
        field for field, state in field_states.items() if state == "not_applicable"
    )
    if mismatched:
        decision = "context_mismatch"
    elif unresolved:
        decision = "insufficient_evidence"
    else:
        decision = "defined"
    hypothesis = str(pair.get("hypothesized_decision", "")).strip()
    return {
        "pair_id": pair_id,
        "context_id": str(context["context_id"]),
        "row_a": row_a_id,
        "row_b": row_b_id,
        "metric": requested_metric,
        "claim": str(pair["claim"]),
        "decision": decision,
        "field_states": field_states,
        "mismatched_fields": mismatched,
        "unresolved_fields": unresolved,
        "not_applicable_fields": not_applicable,
        "hypothesized_decision": hypothesis,
        "hypothesis_match": None if not hypothesis else hypothesis == decision,
    }


def evaluate_pairs(
    contexts: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pair in pairs:
        context_id = str(pair["context_id"])
        if context_id not in contexts:
            raise ValueError(f"{pair['pair_id']} references unknown context {context_id!r}")
        results.append(_evaluate_pair(pair, contexts[context_id], rows))
    return results


def _variant_contexts(
    contexts: dict[str, dict[str, Any]],
    fields_to_remove: set[str],
    *,
    metric_only: bool = False,
) -> dict[str, dict[str, Any]]:
    variants = copy.deepcopy(contexts)
    for context in variants.values():
        if metric_only:
            context["required_invariants"] = list(context["metric_identity_fields"])
            context["required_resolved"] = []
        else:
            context["required_invariants"] = [
                field for field in context["required_invariants"] if field not in fields_to_remove
            ]
            context["required_resolved"] = [
                field for field in context["required_resolved"] if field not in fields_to_remove
            ]
    return variants


def evaluate_ablation(
    contexts: dict[str, dict[str, Any]],
    groups: dict[str, list[str]],
    rows: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_by_pair = {row["pair_id"]: row["decision"] for row in full_results}
    variants: list[tuple[str, dict[str, dict[str, Any]]]] = [
        ("full", contexts),
        ("metric_only", _variant_contexts(contexts, set(), metric_only=True)),
    ]
    variants.extend(
        (f"without_{group_id}", _variant_contexts(contexts, set(fields)))
        for group_id, fields in groups.items()
    )
    output: list[dict[str, Any]] = []
    for variant_id, variant_contexts in variants:
        results = full_results if variant_id == "full" else evaluate_pairs(variant_contexts, rows, pairs)
        counts = Counter(result["decision"] for result in results)
        unsafe_flips = sorted(
            result["pair_id"]
            for result in results
            if full_by_pair[result["pair_id"]] != "defined" and result["decision"] == "defined"
        )
        output.append(
            {
                "variant": variant_id,
                "defined": counts["defined"],
                "context_mismatch": counts["context_mismatch"],
                "insufficient_evidence": counts["insufficient_evidence"],
                "unsafe_flip_count": len(unsafe_flips),
                "unsafe_flip_pair_ids": unsafe_flips,
            }
        )
    return output


def _pair_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ";".join(result[key]) if isinstance(result.get(key), list) else result.get(key, "")
        for key in PAIR_COLUMNS
    }


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_pairwise_outputs(
    results: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(result["decision"] for result in results)
    summary = {
        "pair_count": len(results),
        "decision_counts": {decision: counts[decision] for decision in sorted(DECISIONS)},
        "high_confidence_diversity_gate": all(counts[decision] > 0 for decision in DECISIONS),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "pair_decisions.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "pair_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(_pair_csv_row(result) for result in results)

    pair_lines = [
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Pair & Context & Metric & Decision & Blocking fields \\",
        r"\hline",
    ]
    for result in results:
        blocking = result["mismatched_fields"] + result["unresolved_fields"]
        pair_lines.append(
            " & ".join(
                [
                    _latex_escape(result["pair_id"]),
                    _latex_escape(result["context_id"]),
                    _latex_escape(result["metric"]),
                    _latex_escape(result["decision"]),
                    _latex_escape(", ".join(blocking) or "none"),
                ]
            )
            + r" \\"
        )
    pair_lines.extend([r"\hline", r"\end{tabular}", ""])
    (out_dir / "pair_decisions.tex").write_text("\n".join(pair_lines), encoding="utf-8")

    (out_dir / "ablation.json").write_text(
        json.dumps(ablation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ablation_columns = [
        "variant",
        "defined",
        "context_mismatch",
        "insufficient_evidence",
        "unsafe_flip_count",
        "unsafe_flip_pair_ids",
    ]
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ablation_columns)
        writer.writeheader()
        for row in ablation:
            writer.writerow(
                {
                    **row,
                    "unsafe_flip_pair_ids": ";".join(row["unsafe_flip_pair_ids"]),
                }
            )
    ablation_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Variant & Defined & Mismatch & Insufficient & Unsafe flips \\",
        r"\hline",
    ]
    for row in ablation:
        ablation_lines.append(
            " & ".join(
                [
                    _latex_escape(row["variant"]),
                    str(row["defined"]),
                    str(row["context_mismatch"]),
                    str(row["insufficient_evidence"]),
                    str(row["unsafe_flip_count"]),
                ]
            )
            + r" \\"
        )
    ablation_lines.extend([r"\hline", r"\end{tabular}", ""])
    (out_dir / "ablation.tex").write_text("\n".join(ablation_lines), encoding="utf-8")


def run_pairwise(args: argparse.Namespace) -> None:
    contexts, groups, context_protocol = load_contexts(Path(args.contexts))
    rows, row_protocol = load_rows(Path(args.rows))
    pairs, pair_protocol = load_pairs(Path(args.pairs))
    if len({context_protocol, row_protocol, pair_protocol}) != 1:
        raise ValueError(
            "protocol_id mismatch across contexts, rows, and pairs: "
            f"{context_protocol!r}, {row_protocol!r}, {pair_protocol!r}"
        )
    results = evaluate_pairs(contexts, rows, pairs)
    ablation = evaluate_ablation(contexts, groups, rows, pairs, results)
    out_dir = Path(args.out_dir)
    write_pairwise_outputs(results, ablation, out_dir)
    print(json.dumps(_load_yaml(out_dir / "summary.json"), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_pairwise(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"pairwise admission check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
