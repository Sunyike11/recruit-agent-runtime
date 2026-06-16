from langgraph.graph import StateGraph, END
from src.core.state import RecruitState
from langgraph.checkpoint.memory import MemorySaver


def create_default_agents():
    """Create the real production agents lazily.

    Fake-node tests can import this module and inject nodes without loading
    retrieval dependencies such as LlamaIndex, Chroma, or embedding models.
    """
    try:
        from src.agents.planner import PlannerAgent
        from src.agents.retriever import RetrieverAgent
        from src.agents.matcher import MatcherAgent
        from src.agents.refiner import RefinerAgent
    except ImportError as exc:
        raise RuntimeError(
            "Unable to create default Recruit agents. Install the runtime "
            "dependencies required by the real workflow, especially retrieval "
            "dependencies such as llama-index, Chroma, and embedding packages. "
            "Fake-node tests should pass planner/retriever/matcher/refiner "
            "explicitly to create_recruit_graph()."
        ) from exc

    return PlannerAgent(), RetrieverAgent(), MatcherAgent(), RefinerAgent()


def create_recruit_graph(
    planner=None,
    retriever=None,
    matcher=None,
    refiner=None,
    checkpointer=None,
    interrupt_before=None,
):
    # 实例化插件：内存检查点，用于保存状态以便中断后恢复
    memory = checkpointer or MemorySaver()
    workflow = StateGraph(RecruitState)

    # 定义节点
    if any(agent is None for agent in (planner, retriever, matcher, refiner)):
        default_planner, default_retriever, default_matcher, default_refiner = create_default_agents()
        planner = planner or default_planner
        retriever = retriever or default_retriever
        matcher = matcher or default_matcher
        refiner = refiner or default_refiner

    # 添加节点到图中
    workflow.add_node("planner_node", planner)
    workflow.add_node("retriever_node", retriever)
    workflow.add_node("matcher_node", matcher)
    workflow.add_node("refiner_node", refiner)  # 新节点

    # 设置起点
    workflow.set_entry_point("planner_node")

    # 定义逻辑边：Planner 跑完后，根据 next_action 决定去哪
    # 在 create_recruit_graph 函数里增加

    workflow.add_edge("planner_node", "retriever_node")
    workflow.add_edge("retriever_node", "matcher_node")
    workflow.add_edge("refiner_node", "retriever_node")

    # 定义条件边：根据 Matcher 的 next_action 决定去向
    workflow.add_conditional_edges(
        "matcher_node",
        lambda x: x["next_action"],
        {
            "refine": "refiner_node",  # 去优化
            "end": END  # 结束
        }
    )

    # workflow.add_edge("matcher_node", END)  # 暂时先结束，跑通

    # 在 refiner 之前中断，让 HR 决定是否允许 Agent 修改搜索词
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["refiner_node"] if interrupt_before is None else interrupt_before
    )
    return app

    # return workflow.compile()

