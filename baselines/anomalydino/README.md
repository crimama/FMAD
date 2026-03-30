# AnomalyDINO Baseline

Upstream repository: `https://github.com/dammsi/AnomalyDINO`

This workspace cannot resolve `github.com`, so the upstream repository could not be cloned during this session.

The expected runtime layout is:

```text
baselines/anomalydino/
├── README.md
├── requirements.txt
└── upstream/
    ├── run_anomalydino.py
    ├── requirements.txt
    └── src/
```

`scripts/run_anomalydino.sh` will use `baselines/anomalydino/upstream` if it already exists. If it does not exist and outbound network access is available at runtime, the script will attempt to clone the upstream repository automatically.
