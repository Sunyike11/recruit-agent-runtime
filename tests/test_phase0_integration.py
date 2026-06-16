import builtins
import importlib
import sys

from langchain_core.messages import AIMessage

from src.core.state import create_initial_state
from src.utils.structured_output import parse_json_object_or_default


def block_llama_index_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("llama_index"):
            raise ModuleNotFoundError("blocked llama_index import in fake-node test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FakePlanner:
    def __call__(self, state):
        return {
            "extracted_jd": {
                "tech_stack": ["PyTorch", "3D生成"],
                "education": "硕士",
                "must_have": ["3D生成经验"],
                "search_query": "PyTorch 3D生成",
            },
            "next_action": "retrieve_candidates",
            "messages": [AIMessage(content="fake planner done")],
        }


class FakeRetriever:
    def __call__(self, state):
        return {
            "candidate_pool": [
                {"text": "候选人A，硕士，PyTorch，3D生成项目", "metadata": {"file_name": "a.pdf"}}
            ],
            "next_action": "match_evaluation",
            "messages": [AIMessage(content="fake retriever done")],
        }


class FakeHighScoreMatcher:
    def __call__(self, state):
        return {
            "final_reports": [{"candidate_name": "A", "total_score": 90}],
            "next_action": "end",
            "messages": [AIMessage(content="fake matcher end")],
        }


class FakeLowScoreMatcher:
    def __call__(self, state):
        loop_count = state.get("loop_count", 0)
        if loop_count == 0:
            return {
                "final_reports": [{"candidate_name": "A", "total_score": 40}],
                "next_action": "refine",
                "refinement_advice": "扩大搜索词",
                "loop_count": 1,
                "messages": [AIMessage(content="fake matcher refine")],
            }
        return {
            "final_reports": [{"candidate_name": "A", "total_score": 75}],
            "next_action": "end",
            "messages": [AIMessage(content="fake matcher end after refine")],
        }


class FakeRefiner:
    def __call__(self, state):
        updated_jd = state["extracted_jd"].copy()
        updated_jd["search_query"] = "PyTorch 3D AIGC"
        return {
            "extracted_jd": updated_jd,
            "messages": [AIMessage(content="fake refiner done")],
        }


def build_fake_graph(matcher, interrupt_before=None):
    from src.core.graph import create_recruit_graph

    return create_recruit_graph(
        planner=FakePlanner(),
        retriever=FakeRetriever(),
        matcher=matcher,
        refiner=FakeRefiner(),
        interrupt_before=interrupt_before,
    )


def test_import_graph_does_not_load_real_retrieval_dependencies(monkeypatch):
    block_llama_index_imports(monkeypatch)
    sys.modules.pop("src.core.graph", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    graph_module = importlib.import_module("src.core.graph")

    assert hasattr(graph_module, "create_recruit_graph")
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_graph_reaches_end_without_refine(monkeypatch):
    block_llama_index_imports(monkeypatch)
    app = build_fake_graph(FakeHighScoreMatcher(), interrupt_before=[])
    config = {"configurable": {"thread_id": "phase0-end-test"}}
    events = list(app.stream(create_initial_state("测试JD"), config))

    assert events[-1]["matcher_node"]["next_action"] == "end"
    assert events[-1]["matcher_node"]["final_reports"][0]["total_score"] == 90


def test_graph_interrupts_before_refiner_and_resumes(monkeypatch):
    block_llama_index_imports(monkeypatch)
    app = build_fake_graph(FakeLowScoreMatcher())
    config = {"configurable": {"thread_id": "phase0-test"}}

    list(app.stream(create_initial_state("测试JD"), config))
    interrupted_state = app.get_state(config)

    assert interrupted_state.next == ("refiner_node",)
    assert interrupted_state.values["next_action"] == "refine"

    resumed_events = list(app.stream(None, config))

    assert resumed_events[-1]["matcher_node"]["next_action"] == "end"
    assert resumed_events[-1]["matcher_node"]["final_reports"][0]["total_score"] == 75


def test_default_graph_missing_runtime_dependencies_has_clear_error(monkeypatch):
    block_llama_index_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    from src.core.graph import create_recruit_graph

    try:
        create_recruit_graph()
    except RuntimeError as exc:
        assert "Unable to create default Recruit agents" in str(exc)
        assert "Fake-node tests should pass planner/retriever/matcher/refiner" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when real retrieval dependencies are unavailable")


def test_structured_output_parser_fills_defaults():
    parsed = parse_json_object_or_default(
        '```json\n{"tech_stack": ["PyTorch"], "search_query": "PyTorch"}\n```',
        {"tech_stack": [], "education": "", "must_have": [], "search_query": ""},
    )

    assert parsed["tech_stack"] == ["PyTorch"]
    assert parsed["education"] == ""
    assert parsed["search_query"] == "PyTorch"
