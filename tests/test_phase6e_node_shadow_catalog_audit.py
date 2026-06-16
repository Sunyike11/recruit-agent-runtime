import builtins
import json
from pathlib import Path

from src.integration import (
    NodeShadowCompareAuditExporter,
    NodeShadowCompareAuditor,
    NodeShadowCompareAuditReport,
    NodeShadowCompareFixtureCatalog,
    SingleNodeShadowCompareCase,
    SingleNodeShadowCompareHarness,
)


FIXTURE_PATH = Path("tests/fixtures/node_shadow_compare_cases.json")
SENSITIVE_TEXT = "FULL-RESUME-SECRET-API-KEY-MUST-NOT-ENTER-AUDIT"


def fake_catalog_data():
    return {
        "cases": [
            {
                "case_id": "refiner_aligned",
                "node_name": "query_refine",
                "node_type": "refiner",
                "input_data": {"query": "python"},
                "production_output": {"refined_query": "python remote"},
                "shadow_output": {"refined_query": "python remote"},
                "tags": ["aligned", "refiner"],
            },
            {
                "case_id": "matcher_aligned",
                "node_name": "candidate_match",
                "node_type": "matcher",
                "input_data": {"candidate_id": "candidate_1"},
                "production_output": {
                    "final_reports": [{"candidate_id": "candidate_1", "total_score": 90}]
                },
                "shadow_output": {
                    "match_report": {"candidate_id": "candidate_1", "total_score": 82},
                    "total_score": 82,
                },
                "tags": ["aligned", "matcher"],
            },
        ]
    }


def test_fixture_catalog_can_create_from_dict_and_list_get_filter_cases():
    catalog = NodeShadowCompareFixtureCatalog.from_dict(fake_catalog_data())

    assert [case.case_id for case in catalog.list_cases()] == [
        "refiner_aligned",
        "matcher_aligned",
    ]
    assert catalog.get_case("refiner_aligned").node_name == "query_refine"
    assert [case.case_id for case in catalog.filter_by_node_type("matcher")] == [
        "matcher_aligned"
    ]
    assert [case.case_id for case in catalog.filter_by_tag("aligned")] == [
        "refiner_aligned",
        "matcher_aligned",
    ]


def test_fixture_catalog_loads_json_fake_snapshot_cases():
    catalog = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH)

    assert len(catalog.list_cases()) == 4
    assert catalog.get_case("matcher_exact_score_mismatch").compare_exact_scores is True
    assert len(catalog.filter_by_node_type("refiner")) == 2


def test_catalog_aligned_refiner_snapshot_runs_as_match():
    catalog = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH)

    result = SingleNodeShadowCompareHarness().run_case(
        catalog.get_case("refiner_aligned_snapshot").to_compare_case()
    )

    assert result.decision.status == "match"
    assert result.metadata["real_production_graph_invoked"] is False


def test_catalog_aligned_matcher_does_not_compare_score_by_default():
    catalog = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH)

    result = SingleNodeShadowCompareHarness().run_case(
        catalog.get_case("matcher_aligned_score_presence").to_compare_case()
    )

    assert result.decision.status == "match"
    assert "match_report.score_presence" in result.parity_report.aligned_fields


def test_catalog_mismatch_fixture_emits_high_risk_mismatch():
    catalog = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH)

    result = SingleNodeShadowCompareHarness().run_case(
        catalog.get_case("refiner_query_mismatch").to_compare_case()
    )

    assert result.decision.status == "mismatch"
    assert result.decision.risk_level == "high"
    assert result.parity_report.mismatched_fields == ["refined_query"]


def test_compare_exact_scores_fixture_detects_matcher_score_mismatch():
    catalog = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH)

    result = SingleNodeShadowCompareHarness().run_case(
        catalog.get_case("matcher_exact_score_mismatch").to_compare_case()
    )

    assert result.decision.status == "mismatch"
    assert "match_report.total_score" in result.parity_report.mismatched_fields


def test_catalog_runs_cases_in_batch_through_existing_fake_harness():
    results = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH).run_cases()

    assert [result.decision.status for result in results] == [
        "match",
        "match",
        "mismatch",
        "mismatch",
    ]
    assert all(result.metadata["summary_only"] for result in results)


def test_audit_report_counts_match_mismatch_skipped_and_risk():
    results = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH).run_cases()
    warning = NodeShadowCompareFixtureCatalog.from_dict(fake_catalog_data()).run_cases()[0]
    warning.decision.status = "warning"

    def fail(input_data):
        raise RuntimeError("static injected failure")

    skipped = SingleNodeShadowCompareHarness().run_case(
        SingleNodeShadowCompareCase(
            case_id="skipped_injected_case",
            node_name="query_refine",
            node_type="refiner",
            input_data={},
            production_callable=fail,
            shadow_callable=lambda input_data: {"refined_query": "fake"},
        )
    )
    report = NodeShadowCompareAuditor().build_report(results + [warning, skipped])

    assert isinstance(report, NodeShadowCompareAuditReport)
    assert report.total_cases == 6
    assert report.match_count == 2
    assert report.warning_count == 1
    assert report.mismatch_count == 2
    assert report.skipped_count == 1
    assert report.high_risk_count == 2
    assert report.node_types == ["matcher", "refiner"]


def test_audit_json_export_is_parseable_and_round_trips_summary_fields():
    report = NodeShadowCompareAuditor().build_report(
        NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH).run_cases()
    )

    exported = NodeShadowCompareAuditExporter.export_json(report)
    restored = NodeShadowCompareAuditReport.from_dict(json.loads(exported))

    assert restored.total_cases == 4
    assert restored.mismatch_count == 2
    assert restored.metadata["summary_only"] is True


def test_audit_text_export_contains_decisions_but_not_sensitive_snapshot_payload():
    catalog = NodeShadowCompareFixtureCatalog.from_dict(
        {
            "cases": [
                {
                    "case_id": "safe_export",
                    "node_name": "query_refine",
                    "node_type": "refiner",
                    "input_data": {"context": SENSITIVE_TEXT},
                    "production_output": {"refined_query": SENSITIVE_TEXT},
                    "shadow_output": {"refined_query": SENSITIVE_TEXT},
                    "metadata": {"secret": SENSITIVE_TEXT},
                }
            ]
        }
    )
    report = NodeShadowCompareAuditor().build_report(catalog.run_cases())

    text = NodeShadowCompareAuditExporter.export_text(report)
    serialized = NodeShadowCompareAuditExporter.export_json(report)

    assert "case_id=safe_export" in text
    assert "status=match" in text
    assert SENSITIVE_TEXT not in text
    assert SENSITIVE_TEXT not in serialized
    assert "production_output" not in serialized
    assert "shadow_output" not in serialized


def test_phase6e_does_not_run_or_integrate_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    catalog_source = Path("src/integration/node_shadow_catalog.py").read_text(encoding="utf-8")
    audit_source = Path("src/integration/node_shadow_audit.py").read_text(encoding="utf-8")

    report = NodeShadowCompareAuditor().build_report(
        NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH).run_cases()
    )

    assert report.metadata["real_production_graph_invoked"] is False
    assert report.metadata["real_production_node_invoked"] is False
    assert "NodeShadowCompareFixtureCatalog" not in graph_source
    assert "src.core.graph" not in catalog_source
    assert "src.core.graph" not in audit_source
    assert "create_recruit_graph" not in catalog_source


def test_phase6e_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            ("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")
        ):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase6E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    results = NodeShadowCompareFixtureCatalog.from_json_file(FIXTURE_PATH).run_cases()
    report = NodeShadowCompareAuditor().build_report(results)

    assert report.total_cases == 4
    assert blocked == []
