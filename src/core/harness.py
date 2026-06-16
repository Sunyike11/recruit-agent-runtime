import uuid

from src.core.graph import create_recruit_graph
from src.core.state import create_initial_state


DEFAULT_TEST_JD = """
招聘岗位：AI开发
职责：agent开发。
要求：计算机相关专业硕士，精通PyTorch。
"""


DEFAULT_INTERACTIVE_JD = """
招聘岗位：AI开发
职责：agent开发。
要求：计算机相关专业硕士，精通PyTorch。
"""


def run_test_flow(jd_text: str = DEFAULT_TEST_JD):
    app = create_recruit_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = create_initial_state(jd_text)

    print("--- 开始运行 Recruit-Graph 测试 ---")
    for output in app.stream(initial_state, config):
        for node_name, state_update in output.items():
            print(f"\n[节点触发]: {node_name}")
            if "extracted_jd" in state_update:
                print(f"解析到的结构化需求: {state_update['extracted_jd']}")
            if "messages" in state_update:
                print(f"Agent 回复: {state_update['messages'][-1].content}")
    print("\n--- 测试运行结束 ---")


def run_interactive_flow(jd_text: str = DEFAULT_INTERACTIVE_JD):
    app = create_recruit_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = create_initial_state(jd_text)

    for event in app.stream(initial_state, config):
        print(event)

    state = app.get_state(config)
    if state.next:
        last_message = state.values["messages"][-1].content
        print(f"\n[AI 建议]: {last_message}")
        print(f"[反思建议]: {state.values.get('refinement_advice')}")

        user_input = input("\n--- HR 审核环节 ---\n是否同意 Agent 修改搜索词并重试？(y/直接输入新的搜索词/n): ")
        if user_input.lower() == "n":
            print("HR 拒绝重试，流程结束。")
            return

        update = {}
        if user_input.lower() != "y":
            update["human_feedback"] = user_input
        if update:
            app.update_state(config, update, as_node="matcher_node")

        for event in app.stream(None, config):
            print(event)

        state = app.get_state(config)

    print(f"Agent 回复: {state.values['messages'][-1].content}")
