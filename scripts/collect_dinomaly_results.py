#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


METRIC_PATTERNS = {
    "image_auroc": [
        r"(?:image|img)[-_ ]?(?:level[-_ ]?)?auroc[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bi[-_ ]?auroc[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "pixel_auroc": [
        r"(?:pixel|pix)[-_ ]?(?:level[-_ ]?)?auroc[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bp[-_ ]?auroc[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "pixel_ap": [
        r"\bp[-_ ]?ap[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?:pixel|pix)[-_ ]?ap[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "pixel_f1": [
        r"\bp[-_ ]?f1[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?:pixel|pix)[-_ ]?f1[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "pixel_aupro": [
        r"\bp[-_ ]?aupro[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?:pixel|pix)[-_ ]?aupro[:=]\s*([0-9]*\.?[0-9]+)",
    ],
}

MEAN_PATTERNS = [
    r"mean[:=]\s*([0-9]*\.?[0-9]+)",
    r"average[:=]\s*([0-9]*\.?[0-9]+)",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def extract_metric(patterns, content):
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def main():
    args = parse_args()
    log_path = Path(args.log)
    output_path = Path(args.output)
    content = log_path.read_text(encoding="utf-8", errors="ignore")

    metrics = {}
    for metric_name, patterns in METRIC_PATTERNS.items():
        value = extract_metric(patterns, content)
        if value is not None:
            metrics[metric_name] = value

    mean_value = extract_metric(MEAN_PATTERNS, content)
    if mean_value is not None:
        metrics["mean"] = mean_value

    payload = {
        "benchmark": args.benchmark,
        "entrypoint": args.entrypoint,
        "data_path": args.data_path,
        "metrics": metrics,
        "log_path": str(log_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

