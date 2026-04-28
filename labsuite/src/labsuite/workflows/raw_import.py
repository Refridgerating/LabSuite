"""Raw-file custody import helpers for ledger-aware workflows."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from labsuite.core.exceptions import WorkflowError

RawImportModality = Literal["vsm", "fmr", "esr"]


@dataclass(slots=True)
class RawImportResult:
    original_path: Path
    imported_path: Path
    copied: bool
    copied_sidecars: list[Path] = field(default_factory=list)
    status: str = "unchanged"
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "original_path": str(self.original_path),
            "imported_path": str(self.imported_path),
            "copied": self.copied,
            "copied_sidecars": [str(path) for path in self.copied_sidecars],
            "status": self.status,
            "message": self.message,
        }


def import_raw_for_ledger(
    source_path: Path,
    *,
    modality: RawImportModality,
    raw_import_root: Path,
) -> RawImportResult:
    """Copy an external raw file into the project raw tree for ledger custody."""

    original = source_path.resolve()
    root = raw_import_root.resolve()
    if _is_relative_to(original, root):
        return RawImportResult(
            original_path=original,
            imported_path=original,
            copied=False,
            status="unchanged",
            message="source already under raw import root",
        )
    destination_dir = root / modality.upper()
    destination_dir.mkdir(parents=True, exist_ok=True)
    sidecars = _sidecars_for(original, modality)
    destination = _unique_destination(destination_dir / original.name, sidecars)
    try:
        shutil.copy2(original, destination)
        copied_sidecars = []
        for sidecar in sidecars:
            copied_sidecar = destination.with_suffix(sidecar.suffix)
            shutil.copy2(sidecar, copied_sidecar)
            copied_sidecars.append(copied_sidecar)
    except OSError as exc:
        raise WorkflowError(f"Failed to import raw file {original}: {exc}") from exc
    return RawImportResult(
        original_path=original,
        imported_path=destination,
        copied=True,
        copied_sidecars=copied_sidecars,
        status="copied",
        message=None,
    )


def _sidecars_for(source_path: Path, modality: RawImportModality) -> list[Path]:
    if modality != "esr" or source_path.suffix.lower() != ".dsc":
        return []
    sidecars = []
    for suffix in (".DTA", ".dta"):
        candidate = source_path.with_suffix(suffix)
        if candidate.exists():
            sidecars.append(candidate)
            break
    return sidecars


def _unique_destination(target: Path, sidecars: list[Path]) -> Path:
    if not target.exists() and not any(
        target.with_suffix(sidecar.suffix).exists() for sidecar in sidecars
    ):
        return target
    for index in range(2, 10000):
        candidate = target.with_name(f"{target.stem}__{index}{target.suffix}")
        if candidate.exists():
            continue
        if any(candidate.with_suffix(sidecar.suffix).exists() for sidecar in sidecars):
            continue
        return candidate
    raise WorkflowError(f"Could not find an unused raw import filename for {target.name}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
