import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

def create_or_clean_folder(path: str):
    if not os.path.exists(path):
        print(f"Creating the folder: {path}")
        os.mkdir(path)
    else:
        print("Experiment folder already exists.")
        exp_files = os.listdir(path)
        if len(exp_files) == 0:
            print(f"No files to clean, the folder is empty")
        for file in exp_files:
            os.remove(
                os.path.join(path, file)
            )
            print(f"Removed: {file}")


def register_model(
        model,
        metadata,
        model_version_id,
        path_model_prod = os.path.join("models", "prod"),
        path_model_arch = os.path.join("models", "archive")
):
    
    os.makedirs(os.path.join(path_model_arch, model_version_id), exist_ok=True)

    for file in os.listdir(path_model_prod):
        
        src_path = os.path.join(path_model_prod, file)
        dst_path = os.path.join(path_model_arch, model_version_id, file)

        print(f"Archiving: {src_path} to {dst_path}")
        
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)

    print(f"Registering artifacts in: {path_model_prod}")
    with open(os.path.join(path_model_prod, "model.pkl"), "wb") as file:
        pickle.dump( model, file)

    with open(os.path.join(path_model_prod, "metadata.pkl"), "wb") as file:
        pickle.dump(metadata, file)




@dataclass(frozen=True)
class CheckResult:
    ok: bool
    missing_dirs: tuple[Path, ...]
    missing_files: tuple[Path, ...]


def check_project_artifacts(
    base_dir: Union[str, Path] = ".",
    date_tag: Optional[str] = None,
    *,
    required_dirs: Optional[Sequence[str]] = None,
    required_files: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> CheckResult:
    """
    Checks existence of required folders and files relative to base_dir.

    Supports `{date}` placeholder in paths. Example:
      - "data/processed/scoring_{date}.csv"
      - "models/archive/{date}/model.pkl"

    Args:
        base_dir: Project root directory.
        date_tag: Value to substitute into `{date}` placeholders (e.g. "202602").
        required_dirs: List of directory paths (relative to base_dir).
        required_files: List of file paths (relative to base_dir).
        strict: If True, raises FileNotFoundError when anything is missing.

    Example usage:
    result = check_project_artifacts(base_dir=".", date_tag="202602")
    print(result.ok)
    print("Missing dirs:", result.missing_dirs)
    print("Missing files:", result.missing_files)

    Returns:
        CheckResult with ok flag and missing paths.
    """


    base = Path(base_dir).resolve()
    fmt = {"date": date_tag}  # date_tag may be None; we handle that below

    def _render(p: str) -> str:
        if "{date}" in p and not date_tag:
            raise ValueError(f"Path '{p}' requires date_tag, but date_tag=None")
        return p.format(**fmt)

    # Defaults for your shown structure (edit as needed)
    if required_dirs is None:
        required_dirs = [
            "data/interim/exp01_hpt_nb",
            "data/processed",
            "data/raw",
            "models/archive/{date}",   # date-parametrized folder
            "models/prod",
            "outputs/sample",
        ]

    if required_files is None:
        required_files = [
            "data/interim/prep.csv",
            "data/interim/test.csv",
            "data/interim/train.csv",
            "data/processed/scoring_{date}.csv",  # date-parametrized file
            "data/raw/New%20Spanish%20Academic%20Dataset.csv",
            "models/archive/{date}/metadata.pkl",
            "models/archive/{date}/model.pkl",
            "models/prod/metadata.pkl",
            "models/prod/model.pkl",
            "outputs/bottom_least_freq_terms.png",
            "outputs/top_most_freq_terms.png",
        ]

    missing_dirs: list[Path] = []
    missing_files: list[Path] = []

    for d in required_dirs:
        path = base / _render(d)
        if not path.is_dir():
            missing_dirs.append(path)

    for f in required_files:
        path = base / _render(f)
        if not path.is_file():
            missing_files.append(path)

    ok = not missing_dirs and not missing_files

    if strict and not ok:
        parts = []
        if missing_dirs:
            parts.append("Missing dirs:\n  " + "\n  ".join(map(str, missing_dirs)))
        if missing_files:
            parts.append("Missing files:\n  " + "\n  ".join(map(str, missing_files)))
        raise FileNotFoundError("\n\n".join(parts))

    return CheckResult(ok=ok, missing_dirs=tuple(missing_dirs), missing_files=tuple(missing_files))


