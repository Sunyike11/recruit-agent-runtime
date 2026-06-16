class OptionalCheckpointDependencyError(RuntimeError):
    """Raised when optional durable checkpoint dependencies are unavailable."""


def create_optional_sqlite_checkpointer(db_path):
    """Create a LangGraph SQLite checkpointer when the optional package exists."""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise OptionalCheckpointDependencyError(
            "LangGraph SQLite checkpointer is not installed. Install the optional "
            "langgraph-checkpoint-sqlite package to enable real durable graph "
            "checkpoint blobs. Phase1C still stores checkpoint metadata in "
            "SQLiteRuntimeStore without this optional dependency."
        ) from exc

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    # Keep the connection reachable for the lifetime of the saver. The
    # langgraph-checkpoint-sqlite from_conn_string API is a context manager in
    # current versions, while runtime callers need a concrete saver object.
    saver._runtime_sqlite_conn = conn
    return saver
