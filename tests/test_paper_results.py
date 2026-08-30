from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "reported" / "paper_results.json"
CONFIG_PATH = ROOT / "configs" / "paper_transcribed.json"


def _paper_results() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_reported_results_directory_is_paper_only() -> None:
    result_files = {path.name for path in RESULT_PATH.parent.iterdir() if path.is_file()}
    assert result_files == {"paper_results.json"}

    payload = _paper_results()
    assert payload["record_kind"] == "paper_reported_results_transcription"
    assert payload["status"] == "paper_reported_transcribed_not_reproduced"
    assert "not code-rerun" in payload["source"]["note"]


def test_configuration_directory_is_paper_transcription_only() -> None:
    config_files = {path.name for path in CONFIG_PATH.parent.iterdir() if path.is_file()}
    assert config_files == {"paper_transcribed.json"}

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "paper_configuration_transcription"
    assert payload["status"] == "paper_reported_transcribed_not_reproduced"
    assert payload["router"]["alpha"] is None
    assert payload["router"]["rho_min"] is None
    assert payload["router"]["lambda_head_structure_and_sigmoid_placement"] is None
    assert (
        payload["router"]["rho_head_structure_sigmoid_placement_and_range_mapping"]
        is None
    )
    assert payload["igsr"]["beta"] is None
    assert payload["igsr"]["subregion_partition_and_anchor_order"] is None
    assert payload["igsr"]["parallel_implementation_details"] is None
    assert "exact scorer is not specified" in payload["benchmark"]["metric"]


def test_paper_headline_rows_are_transcribed_verbatim() -> None:
    payload = _paper_results()

    main_row = payload["table_1"]["rows"][-1]
    assert main_row == [
        "GeoLLaVA-8K+DualComp",
        "42.4x",
        26.7,
        45.0,
        37.4,
        53.0,
        69.5,
        43.6,
        34.0,
        65.0,
        69.0,
        79.0,
        72.0,
        48.3,
        49.0,
        53.1,
    ]

    efficiency = {row[0]: row[1:] for row in payload["table_2"]["rows"]}
    assert efficiency["Compression Ratio"][-1] == 42.4
    assert efficiency["Inference Speed (s/image)"][-1] == 3.87
    assert efficiency["Avg. Score (%)"][-1] == 53.10

    transfer_row = payload["table_4"]["rows"][-1]
    assert transfer_row[0:2] == ["Qwen2.5-VL-7B+DualComp", "10.24x"]
    assert transfer_row[-1] == 47.9


def test_transcription_preserves_paper_internal_inconsistencies() -> None:
    payload = _paper_results()
    main_row = payload["table_1"]["rows"][-1]
    displayed_macro = sum(main_row[2:-1]) / 13
    assert displayed_macro == pytest.approx(53.1923076923)
    assert main_row[-1] == 53.1

    efficiency = {row[0]: row[1:] for row in payload["table_2"]["rows"]}
    dualcomp_visual = efficiency["Visual Encoding + Compression (s/image)"][-1]
    dualcomp_generation = efficiency["LLM Generation (s/image)"][-1]
    dualcomp_reported_e2e = efficiency["Inference Speed (s/image)"][-1]
    assert dualcomp_visual + dualcomp_generation == pytest.approx(3.77)
    assert dualcomp_reported_e2e == 3.87

    audit_locations = {item["location"] for item in payload["paper_internal_consistency_audit"]}
    assert {"Table 1 and Table 3", "Table 2", "Table 2 versus page-12 prose"} <= audit_locations
