import json
from pathlib import Path

import pytest

from scripts.build_recruitment_eval_index import build_recruitment_eval_index_dry_run
from scripts.generate_recruitment_eval_dataset import generate_dataset
from src.evaluation.dataset import (
    REQUIRED_ATTACK_TYPES,
    load_recruitment_eval_dataset,
    validate_recruitment_eval_dataset,
)


def test_phase13a_dataset_generator_creates_required_catalog(tmp_path):
    output_dir = tmp_path / "eval_v1"

    result = generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    validation = validate_recruitment_eval_dataset(dataset)

    assert result["status"] == "ok"
    assert validation.valid, validation.errors
    assert len(dataset.jobs) == 12
    assert len(dataset.candidates) == 40
    assert len([candidate for candidate in dataset.candidates if candidate.is_special_case]) >= 10
    assert (output_dir / "README.md").exists()
    assert (output_dir / "manifest.json").exists()


def test_phase13a_ids_are_unique_and_manifest_counts_match(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)

    job_ids = [job.job_id for job in dataset.jobs]
    candidate_ids = [candidate.candidate_id for candidate in dataset.candidates]

    assert len(job_ids) == len(set(job_ids))
    assert len(candidate_ids) == len(set(candidate_ids))
    assert all(candidate_id.startswith("candidate_") for candidate_id in candidate_ids)
    assert dataset.manifest["job_count"] == 12
    assert dataset.manifest["candidate_count"] == 40
    assert dataset.manifest["special_case_count"] == 10
    assert dataset.manifest["privacy_mode"] == "synthetic_anonymized"


def test_phase13a_relevance_labels_cover_every_candidate_and_level(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    candidate_ids = {candidate.candidate_id for candidate in dataset.candidates}

    for label in dataset.relevance_labels:
        assert set(label.candidate_relevance) == candidate_ids
        assert set(label.candidate_relevance.values()).issubset({0, 1, 2})
        assert sum(1 for value in label.candidate_relevance.values() if value == 2) >= 3
        assert sum(1 for value in label.candidate_relevance.values() if value == 1) >= 3


def test_phase13a_ideal_ranking_is_legal(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)

    for label in dataset.relevance_labels:
        ranked_relevance = [label.candidate_relevance[candidate_id] for candidate_id in label.ideal_ranking]
        assert all(value > 0 for value in ranked_relevance)
        assert set(label.ideal_ranking) == {
            candidate_id for candidate_id, relevance in label.candidate_relevance.items() if relevance > 0
        }
        assert ranked_relevance == sorted(ranked_relevance, reverse=True)


def test_phase13a_special_and_attack_cases_are_present(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    special_types = {candidate.special_case_type for candidate in dataset.candidates if candidate.is_special_case}
    attack_types = {case.attack_type for case in dataset.attack_cases}
    attack_candidate_ids = {case.candidate_id for case in dataset.attack_cases}
    candidate_ids = {candidate.candidate_id for candidate in dataset.candidates}

    assert REQUIRED_ATTACK_TYPES.issubset(special_types)
    assert REQUIRED_ATTACK_TYPES.issubset(attack_types)
    assert attack_candidate_ids.issubset(candidate_ids)
    assert "jd_as_resume" in attack_types
    assert "prompt_injection" in attack_types
    assert "duplicate_resume" in attack_types
    assert "filename_injection" in attack_types
    assert "missing_name" in attack_types
    assert "missing_education" in attack_types


def test_phase13a_same_name_and_source_file_safety(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    names = [candidate.display_name for candidate in dataset.candidates]

    assert any(names.count(name) > 1 for name in names if name)
    assert all("/" not in candidate.source_file_name for candidate in dataset.candidates)
    assert all("\\" not in candidate.source_file_name for candidate in dataset.candidates)
    assert all(not Path(candidate.source_file_name).is_absolute() for candidate in dataset.candidates)


def test_phase13a_privacy_patterns_are_absent(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    text = "\n".join(candidate.resume_text for candidate in dataset.candidates)

    assert "@" not in text
    assert "身份证" not in text
    assert "/Users/" not in text
    assert "13800000000" not in text


def test_phase13a_generator_same_seed_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_dataset(first, seed=2026, force=True)
    generate_dataset(second, seed=2026, force=True)

    for file_name in ["manifest.json", "jobs.json", "candidates.json", "relevance_labels.json", "attack_cases.json"]:
        assert (first / file_name).read_text(encoding="utf-8") == (second / file_name).read_text(encoding="utf-8")


def test_phase13a_generator_refuses_overwrite_without_force(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)

    with pytest.raises(FileExistsError):
        generate_dataset(output_dir, seed=2026, force=False)


def test_phase13a_generator_does_not_import_network_or_llm_clients():
    source = Path("scripts/generate_recruitment_eval_dataset.py").read_text(encoding="utf-8")

    forbidden = ["openai", "requests", "httpx", "urllib", "langchain", "llama_index", "chromadb"]
    assert not any(token in source for token in forbidden)


def test_phase13a_index_dry_run_is_summary_only_and_independent(tmp_path):
    output_dir = tmp_path / "eval_v1"
    index_dir = tmp_path / "evaluation_indexes" / "recruitment_eval_v1_chroma"
    generate_dataset(output_dir, seed=2026, force=True)

    result = build_recruitment_eval_index_dry_run(output_dir, index_dir)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["candidate_count"] == 40
    assert result["document_count"] == 40
    assert result["chunk_count_estimate"] >= 40
    assert result["index_dir"] == str(index_dir)
    assert result["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert result["summary_only"] is True
    assert not index_dir.exists()


def test_phase13a_index_dry_run_refuses_existing_chroma_db(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)

    with pytest.raises(ValueError):
        build_recruitment_eval_index_dry_run(output_dir, "chroma_db")


def test_phase13a_dataset_loader_round_trip_and_resume_files(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    summary = dataset.to_summary()

    assert summary["dataset_name"] == "recruitment_eval_v1"
    assert summary["job_count"] == 12
    assert summary["candidate_count"] == 40
    for candidate in dataset.candidates:
        resume_path = output_dir / "resumes" / f"{candidate.candidate_id}.txt"
        assert resume_path.exists()
        assert resume_path.read_text(encoding="utf-8") == candidate.resume_text


def test_phase13a_actual_dataset_files_if_present_are_valid():
    dataset_dir = Path("evaluation_data/v1")
    if not dataset_dir.exists():
        pytest.skip("Phase13A generated dataset is not present yet")

    dataset = load_recruitment_eval_dataset(dataset_dir)
    validation = validate_recruitment_eval_dataset(dataset)

    assert validation.valid, validation.errors
    assert len(dataset.jobs) == 12
    assert len(dataset.candidates) == 40
    assert len(dataset.attack_cases) >= 10


def test_phase13a_json_files_are_parseable_after_generation(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)

    for file_name in ["manifest.json", "jobs.json", "candidates.json", "relevance_labels.json", "attack_cases.json"]:
        payload = json.loads((output_dir / file_name).read_text(encoding="utf-8"))
        assert payload
