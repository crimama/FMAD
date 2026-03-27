import json
import os
from pathlib import Path

import pandas as pd


DATASET_CANDIDATES = {
    "mvtec": [
        "/home/hun/Volume/DATA",
        "/home/hun/Volume/DATA/mvtec",
        "/home/hun/Volume/RESEARCH/VAD/Data/MVTecAD",
    ],
    "visa": [
        "/home/hun/Volume/DATA/visa",
        "/home/hun/Volume/DATA/VISA",
        "/home/hun/Volume/RESEARCH/VAD/Data/VISA",
    ],
    "mvtec_ad_2": [
        "/home/hun/Volume/DATA/mvtec_ad_2",
        "/home/hun/Volume/DATA/MVTecAD2",
        "/home/hun/Volume/DATA/mvtec2d-sam-b",
        "/home/hun/Volume/RESEARCH/VAD/Data/mvtec2d-sam-b",
    ],
    "robustad": [
        "/home/hun/Volume/DATA/robustad",
        "/home/hun/Volume/DATA/RobustAD",
    ],
}

PREPARED_ROOT = Path(__file__).resolve().parent / "_prepared_data"


def _list_dirs(path):
    return sorted(entry.name for entry in Path(path).iterdir() if entry.is_dir())


def resolve_dataset_root(dataset_name, data_path=None):
    candidates = []
    if data_path:
        candidates.append(data_path)
    candidates.extend(DATASET_CANDIDATES.get(dataset_name, []))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Dataset root for '{dataset_name}' was not found. Tried: {candidates}"
    )


def _prepared_dir(dataset_name):
    path = PREPARED_ROOT / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_meta(dataset_name, info):
    prepared_dir = _prepared_dir(dataset_name)
    meta_path = prepared_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
    return str(prepared_dir), str(meta_path)


def _looks_like_mvtec_layout(root):
    try:
        for entry in _list_dirs(root):
            cls_root = os.path.join(root, entry)
            if os.path.isdir(os.path.join(cls_root, "train")) and (
                os.path.isdir(os.path.join(cls_root, "test"))
                or os.path.isdir(os.path.join(cls_root, "test_public"))
            ):
                return True
    except FileNotFoundError:
        return False
    return False


def _guess_mask_path(root, cls_name, defect_name, image_name):
    # Try standard location, then test_public/ground_truth (MVTec AD 2)
    gt_dir = Path(root) / cls_name / "ground_truth" / defect_name
    if not gt_dir.is_dir():
        gt_dir = Path(root) / cls_name / "test_public" / "ground_truth" / defect_name
    if not gt_dir.is_dir():
        return ""

    image_path = Path(image_name)
    candidates = [
        gt_dir / image_path.name,
        gt_dir / f"{image_path.stem}_mask{image_path.suffix}",
        gt_dir / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    prefix_matches = sorted(gt_dir.glob(f"{image_path.stem}*"))
    if prefix_matches:
        return str(prefix_matches[0].resolve())
    return ""


def _resolve_test_dir(cls_root):
    """Return the test directory path, preferring 'test' but falling back to 'test_public'."""
    test_dir = os.path.join(cls_root, "test")
    if os.path.isdir(test_dir):
        return test_dir
    test_public = os.path.join(cls_root, "test_public")
    if os.path.isdir(test_public):
        return test_public
    return None


def build_meta_for_mvtec_style(dataset_name, root):
    info = {"train": {}, "test": {}}
    class_names = [
        cls_name
        for cls_name in _list_dirs(root)
        if os.path.isdir(os.path.join(root, cls_name, "train"))
        and _resolve_test_dir(os.path.join(root, cls_name)) is not None
    ]
    if not class_names:
        raise ValueError(f"No MVTec-style classes found under {root}")

    for cls_name in class_names:
        for phase in ("train", "test"):
            cls_info = []
            if phase == "test":
                phase_root = Path(_resolve_test_dir(os.path.join(root, cls_name)))
            else:
                phase_root = Path(root) / cls_name / phase
            for defect_dir in sorted(entry for entry in phase_root.iterdir() if entry.is_dir()):
                defect_name = defect_dir.name
                is_abnormal = defect_name != "good"
                for image_file in sorted(path for path in defect_dir.iterdir() if path.is_file()):
                    cls_info.append(
                        {
                            "img_path": str(image_file.resolve()),
                            "mask_path": _guess_mask_path(
                                root, cls_name, defect_name, image_file.name
                            )
                            if is_abnormal and phase == "test"
                            else "",
                            "cls_name": cls_name,
                            "specie_name": defect_name,
                            "anomaly": 1 if is_abnormal and phase == "test" else 0,
                        }
                    )
            info[phase][cls_name] = cls_info
    return _write_meta(dataset_name, info)


def build_meta_for_visa(root):
    split_csv = Path(root) / "split_csv" / "1cls.csv"
    if not split_csv.is_file():
        raise FileNotFoundError(f"VisA split CSV not found: {split_csv}")

    csv_data = pd.read_csv(split_csv, header=0)
    columns = list(csv_data.columns)
    info = {"train": {}, "test": {}}
    class_names = sorted(csv_data[columns[0]].unique().tolist())

    for cls_name in class_names:
        cls_data = csv_data[csv_data[columns[0]] == cls_name]
        for phase in ("train", "test"):
            cls_info = []
            cls_data_phase = cls_data[cls_data[columns[1]] == phase].reset_index(drop=True)
            for idx in range(cls_data_phase.shape[0]):
                row = cls_data_phase.loc[idx]
                is_abnormal = row[2] == "anomaly"
                image_path = str((Path(root) / row[3]).resolve())
                mask_path = str((Path(root) / row[4]).resolve()) if is_abnormal else ""
                cls_info.append(
                    {
                        "img_path": image_path,
                        "mask_path": mask_path,
                        "cls_name": cls_name,
                        "specie_name": "",
                        "anomaly": 1 if is_abnormal else 0,
                    }
                )
            info[phase][cls_name] = cls_info

    return _write_meta("visa", info)


def ensure_dataset_ready(dataset_name, data_path=None):
    root = resolve_dataset_root(dataset_name, data_path)
    meta_path = os.path.join(root, "meta.json")
    if os.path.isfile(meta_path):
        return root, meta_path

    if dataset_name == "visa":
        return build_meta_for_visa(root)

    if _looks_like_mvtec_layout(root):
        return build_meta_for_mvtec_style(dataset_name, root)

    raise FileNotFoundError(
        f"Unable to prepare dataset '{dataset_name}' at {root}: unsupported layout."
    )
