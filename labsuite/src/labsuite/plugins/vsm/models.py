"""Shared VSM-specific data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class VsmDataset:
    """Parsed VSM table preserved close to the raw source."""

    source_path: Path
    acquisition_index: IntArray
    field_oe: FloatArray
    field_mT: FloatArray
    moment_emu: FloatArray
    moment_std_err_emu: FloatArray
    temperature_k: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BranchSegment:
    """One monotonic field-sweep branch."""

    branch_id: int
    direction: str
    start_index: int
    end_index: int
    point_count: int
    field_start_mT: float
    field_end_mT: float


@dataclass(slots=True)
class CenteringResult:
    """Stored centering diagnostics and optional applied offsets."""

    field_offset_mT: float
    moment_offset_emu: float
    applied: bool
