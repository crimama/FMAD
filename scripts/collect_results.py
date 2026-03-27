import argparse
import glob
import json
import os


METRICS = ("I-AUROC", "P-AUROC", "AU-PRO")


def load_results(results_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = path
        records.append(payload)
    return records


def metric_value(payload, metric_name):
    value = payload.get("metrics", {}).get(metric_name)
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def render_markdown(records):
    datasets = sorted({record.get("dataset", "unknown") for record in records})
    methods = sorted({record.get("method", "unknown") for record in records})
    lookup = {
        (record.get("method", "unknown"), record.get("dataset", "unknown")): record
        for record in records
    }

    headers = ["Method"]
    for dataset in datasets:
        for metric in METRICS:
            headers.append(f"{dataset} {metric}")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for method in methods:
        row = [method]
        for dataset in datasets:
            payload = lookup.get((method, dataset), {})
            for metric in METRICS:
                row.append(metric_value(payload, metric))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Source files:")
    for record in records:
        status = record.get("status", "ok")
        lines.append(
            f"- `{os.path.basename(record['_path'])}`: "
            f"dataset={record.get('dataset', 'unknown')}, status={status}"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase1")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        raise FileNotFoundError(f"No JSON files found in {args.results_dir}")

    markdown = render_markdown(records)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
