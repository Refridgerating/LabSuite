from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from labsuite.core.recipes import load_esr_recipe
from labsuite.plugins.esr.batch_qc import (
    EsrBatchIdentity,
    EsrBatchQcRecord,
    compute_esr_qc_metrics,
    group_duplicate_runs,
    parse_esr_batch_identity,
    select_best_runs,
)
from labsuite.plugins.esr.service import analyze_esr_file


def test_parse_esr_batch_identity_groups_duplicate_runs_by_sample_replicate_angle_and_frequency() -> None:
    identity = parse_esr_batch_identity(
        Path("20260220_150301857_Temp2-MTJ-D-0,5nm-Pillars-65deg-R1-0,05mT-10mW-100kHz-1min.dsc"),
        {"bruker": {"frequency_GHz": 9.49889673634545}},
    )

    assert identity.sample_id == "Temp2-MTJ-D-0,5nm-Pillars"
    assert identity.replicate_id == "R1"
    assert identity.nominal_angle_deg == 65.0
    assert identity.frequency_bucket_GHz == 9.499


def test_parse_esr_batch_identity_separates_rounded_frequency_buckets() -> None:
    first = parse_esr_batch_identity(
        Path("sample-65deg-R1.dsc"),
        {"bruker": {"frequency_GHz": 9.49851}},
    )
    second = parse_esr_batch_identity(
        Path("sample-65deg-R1.dsc"),
        {"bruker": {"frequency_GHz": 9.49951}},
    )

    grouped = group_duplicate_runs(
        [
            EsrBatchQcRecord(
                source_path=Path("a.dsc"),
                source_stem="a",
                identity=first,
                fit_success=True,
                nrmse=0.01,
                r_squared=0.99,
                snr=10.0,
                edge_margin_peak1=5.0,
                edge_margin_peak2=None,
                uncertainty_score=0.02,
                acquisition_timestamp=None,
            ),
            EsrBatchQcRecord(
                source_path=Path("b.dsc"),
                source_stem="b",
                identity=second,
                fit_success=True,
                nrmse=0.01,
                r_squared=0.99,
                snr=10.0,
                edge_margin_peak1=5.0,
                edge_margin_peak2=None,
                uncertainty_score=0.02,
                acquisition_timestamp=None,
            ),
        ]
    )

    assert len(grouped) == 2


def test_compute_esr_qc_metrics_rejects_edge_truncated_run(
    tmp_path,
    project_root,
    write_bruker_esr_sample,
) -> None:
    source_file = write_bruker_esr_sample(
        tmp_path / "sample-65deg-R1.dsc",
        center_mT=343.7,
        gamma_mT=1.2,
        field_start_mT=340.0,
        field_end_mT=346.0,
    )
    analysis = analyze_esr_file(source_file, project_root / "recipes" / "esr" / "default.yaml")
    qc_record = compute_esr_qc_metrics(
        analysis,
        source_path=source_file,
        recipe=load_esr_recipe(project_root / "recipes" / "esr" / "default.yaml"),
    )

    assert qc_record.accepted_for_plot is False
    assert qc_record.reject_reason == "edge_truncated"


def test_compute_esr_qc_metrics_rejects_missing_second_peak(
    tmp_path,
    project_root,
    write_bruker_esr_sample,
) -> None:
    source_file = write_bruker_esr_sample(
        tmp_path / "sample-65deg-R1.dsc",
        components=[
            {"amplitude": 1.15, "center_mT": 335.0, "gamma_mT": 0.95, "offset": 0.0},
            {"amplitude": 0.9, "center_mT": 345.0, "gamma_mT": 1.1, "offset": 0.0},
        ],
    )
    analysis = analyze_esr_file(source_file, project_root / "recipes" / "esr" / "default.yaml", fit_mode="split")
    analysis.peak_fits = analysis.peak_fits[:1]
    recipe = load_esr_recipe(project_root / "recipes" / "esr" / "default.yaml")

    qc_record = compute_esr_qc_metrics(
        analysis,
        source_path=source_file,
        recipe=recipe,
    )

    assert qc_record.accepted_for_plot is False
    assert qc_record.reject_reason == "second_peak_missing"


def test_compute_esr_qc_metrics_rejects_high_nrmse(
    tmp_path,
    project_root,
    write_bruker_esr_sample,
) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "sample-65deg-R1.dsc")
    analysis = analyze_esr_file(source_file, project_root / "recipes" / "esr" / "default.yaml")
    analysis.selected_residual = np.asarray(analysis.processed.signal, dtype=float).copy()
    recipe = load_esr_recipe(project_root / "recipes" / "esr" / "default.yaml")

    qc_record = compute_esr_qc_metrics(
        analysis,
        source_path=source_file,
        recipe=recipe,
    )

    assert qc_record.nrmse is not None and qc_record.nrmse > recipe.batch_qc_nrmse_max
    assert qc_record.accepted_for_plot is False
    assert qc_record.reject_reason == "nrmse_exceeds_threshold"


def test_select_best_runs_marks_lower_quality_duplicates_as_superseded() -> None:
    identity = EsrBatchIdentity(
        sample_id="sample",
        replicate_id="R1",
        nominal_angle_deg=65.0,
        frequency_bucket_GHz=9.499,
    )
    earlier = EsrBatchQcRecord(
        source_path=Path("20260220_120000000_sample-65deg-R1.dsc"),
        source_stem="early",
        identity=identity,
        fit_success=True,
        nrmse=0.06,
        r_squared=0.99,
        snr=8.0,
        edge_margin_peak1=4.0,
        edge_margin_peak2=3.0,
        uncertainty_score=0.08,
        acquisition_timestamp=datetime(2026, 2, 20, 12, 0, 0),
    )
    later = EsrBatchQcRecord(
        source_path=Path("20260220_130000000_sample-65deg-R1.dsc"),
        source_stem="late",
        identity=identity,
        fit_success=True,
        nrmse=0.03,
        r_squared=0.995,
        snr=12.0,
        edge_margin_peak1=5.5,
        edge_margin_peak2=4.5,
        uncertainty_score=0.03,
        acquisition_timestamp=datetime(2026, 2, 20, 13, 0, 0),
    )

    select_best_runs([earlier, later])

    assert later.selected_as_best is True
    assert later.accepted_for_plot is True
    assert later.reject_reason is None
    assert earlier.selected_as_best is False
    assert earlier.accepted_for_plot is False
    assert earlier.reject_reason == "superseded_by_better_duplicate"


def test_select_best_runs_leaves_rejected_group_unplotted() -> None:
    identity = EsrBatchIdentity(
        sample_id="sample",
        replicate_id="R1",
        nominal_angle_deg=65.0,
        frequency_bucket_GHz=9.499,
    )
    rejected = EsrBatchQcRecord(
        source_path=Path("sample-65deg-R1.dsc"),
        source_stem="rejected",
        identity=identity,
        fit_success=False,
        nrmse=0.3,
        r_squared=0.2,
        snr=1.0,
        edge_margin_peak1=0.5,
        edge_margin_peak2=None,
        uncertainty_score=None,
        acquisition_timestamp=None,
        reject_reason="fit_failed",
    )

    select_best_runs([rejected])

    assert rejected.selected_as_best is False
    assert rejected.accepted_for_plot is False
