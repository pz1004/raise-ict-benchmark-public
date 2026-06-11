# Dataset Usage

RAISE-ICT does not redistribute raw IDS datasets. This public bundle includes code, result tables, manifests, and hardware evidence only. Reviewers who rerun the benchmark should obtain raw datasets from the official dataset providers, accept the applicable terms, and keep the downloaded files outside version control.

The MIT License in this repository applies to the RAISE-ICT code and documentation. It does not change the terms of CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, TON_IoT, NVIDIA software, or third-party mirrors.

## Official Sources

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets

The official CSE-CIC-IDS2018 page states that redistribution, republication, and mirroring are allowed when use or redistribution includes citation to CSE-CIC-IDS2018 and a link to the AWS page. The UNSW-NB15 and TON_IoT official pages point users to UNSW-hosted downloads. The TON_IoT page grants academic research use and asks users to cite the listed TON_IoT papers.

## Optional Mirror Downloads

`scripts/download_datasets.py` contains third-party mirror URLs only as a convenience path. These mirrors are not the authoritative source of truth for public disclosure.

Mirror downloads require explicit opt-in:

```bash
python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets UNSW-NB15 TON_IoT
```

Use this option only after confirming that the mirror, dataset terms, and your review or institutional policy allow it. Generated raw files should remain under `data/raw/`, which is ignored by `.gitignore`.

## Jetson Software Note

The included hardware evidence was generated with the recorded JetPack 6.2.1 / Jetson Linux 36.4.x software condition. As checked on June 11, 2026, NVIDIA's JetPack downloads page lists JetPack 7.2 with Jetson Linux 39.2 as the current release. Treat any JetPack 7.x rerun as a different software condition and record it separately.
