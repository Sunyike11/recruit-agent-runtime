class RuntimeResumeService:
    """Small convenience wrapper around RuntimeRunner resume APIs."""

    def __init__(self, runner):
        self.runner = runner

    def resume_task(self, task_id: str):
        return self.runner.resume_task(task_id)

    def add_human_feedback(self, task_id: str, feedback_type: str, payload):
        return self.runner.add_human_feedback(task_id, feedback_type, payload)

    def get_task_timeline(self, task_id: str):
        return self.runner.get_task_timeline(task_id)
