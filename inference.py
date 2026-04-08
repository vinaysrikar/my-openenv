def predict(input_data):
    env = EmailTriageEnv()
    task = input_data.get("task", "task_easy") if isinstance(input_data, dict) else "task_easy"
    obs = env.reset(task)
    return {
        "status": "ok",
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs
    }
