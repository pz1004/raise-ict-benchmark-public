#!/usr/bin/env python
"""Download public mirror files used by RAISE-ICT experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402


DEFAULT_FILES = {
    "CICIDS2017": [
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Monday-WorkingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Monday-WorkingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Wednesday-workingHours.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Wednesday-workingHours.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
        {
            "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
            "path": "data/raw/cicids2017/machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2017.html",
        },
    ],
    "CSE-CIC-IDS2018": [
        {
            "url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
        {
            "url": "https://huggingface.co/datasets/pcy12345BSU/CSE-CIC-IDS-2018/resolve/main/data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "path": "data/raw/cse_cic_ids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "official_source": "https://www.unb.ca/cic/datasets/ids-2018.html",
        },
    ],
    "UNSW-NB15": [
        {
            "url": "https://huggingface.co/datasets/lacg030175/UNSW-NB15/resolve/main/temporal/train-00000-of-00001.parquet",
            "path": "data/raw/unsw_nb15/temporal/train-00000-of-00001.parquet",
            "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        },
        {
            "url": "https://huggingface.co/datasets/lacg030175/UNSW-NB15/resolve/main/temporal/test-00000-of-00001.parquet",
            "path": "data/raw/unsw_nb15/temporal/test-00000-of-00001.parquet",
            "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        },
    ],
    "TON_IoT": [
        {
            "url": "https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/main/train_test_network.csv",
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
    parser.add_argument("--allow-third-party-mirrors", action="store_true")
    args = parser.parse_args()

    registry = load_yaml(args.registry)
    known_ids = {item["dataset_id"] for item in registry.get("datasets", [])}
    if not args.allow_third_party_mirrors:
        requested_sources = []
        for dataset_id in args.datasets:
            for item in DEFAULT_FILES.get(dataset_id, []):
                requested_sources.append(f"- {dataset_id}: {item['official_source']}")
        source_text = "\n".join(dict.fromkeys(requested_sources))
        print(
            "RAISE-ICT is official-source-first. Third-party mirror downloads "
            "require explicit opt-in with --allow-third-party-mirrors after "
            "checking upstream dataset terms.\n"
            f"Official source pages:\n{source_text}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    records = []
    for dataset_id in args.datasets:
        if dataset_id not in known_ids:
            raise ValueError(f"{dataset_id} is not listed in {args.registry}")
        for item in DEFAULT_FILES.get(dataset_id, []):
            path = Path(item["path"])
            download_file(item["url"], path, args.force)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "download_url": item["url"],
                    "official_source": item["official_source"],
                }
            )

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
