#!/usr/bin/env python
"""Download optional third-party mirror files used by RAISE-ICT experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402


DEFAULT_FILES = {
    "CICIDS2017": [
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Monday-WorkingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Monday-WorkingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Wednesday-workingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Wednesday-workingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
    ],
    "CSE-CIC-IDS2018": [
        {
            "mirror_url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
    ],
    "UNSW-NB15": [
        {
            "mirror_url": "https://huggingface.co/datasets/lacg030175/UNSW-NB15/resolve/main/temporal/train-00000-of-00001.parquet",
            "path": "data/raw/unsw_nb15/temporal/train-00000-of-00001.parquet",
            "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        },
        {
            "mirror_url": "https://huggingface.co/datasets/lacg030175/UNSW-NB15/resolve/main/temporal/test-00000-of-00001.parquet",
            "path": "data/raw/unsw_nb15/temporal/test-00000-of-00001.parquet",
            "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        },
    ],
    "TON_IoT": [
        {
            "mirror_url": "https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/main/train_test_network.csv",
            "path": "data/raw/ton_iot/train_test_network.csv",
            "official_source": "https://research.unsw.edu.au/projects/toniot-datasets",
        }
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data_registry.yaml")
    parser.add_argument("--datasets", nargs="*", default=["UNSW-NB15", "TON_IoT"])
    parser.add_argument("--manifest", default="manifests/dataset_hashes/download_manifest.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-third-party-mirrors",
        action="store_true",
        help="Download convenience mirror files after accepting the upstream dataset terms.",
    )
    args = parser.parse_args()

    registry = load_yaml(args.registry)
    registry_items = {item["dataset_id"]: item for item in registry.get("datasets", [])}
    known_ids = set(registry_items)
    if not args.allow_third_party_mirrors:
        source_lines = []
        for dataset_id in args.datasets:
            item = registry_items.get(dataset_id, {})
            if item.get("status") == "external":
                source_lines.append(f"- {dataset_id}: {item.get('official_url', 'official source not listed')}")
        if source_lines:
            raise SystemExit(
                "RAISE-ICT is official-source-first for public disclosure. "
                "Download raw datasets from the official pages and accept their terms before use. "
                "To use the convenience mirror URLs listed in this script, rerun with "
                "`--allow-third-party-mirrors`.\n"
                + "\n".join(source_lines)
            )
    records = []
    for dataset_id in args.datasets:
        if dataset_id not in known_ids:
            raise ValueError(f"{dataset_id} is not listed in {args.registry}")
        for item in DEFAULT_FILES.get(dataset_id, []):
            path = Path(item["path"])
            download_file(item["mirror_url"], path, args.force)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "download_url": item["mirror_url"],
                    "official_source": item["official_source"],
                    "source_policy": "third_party_mirror_opt_in",
                }
            )

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
