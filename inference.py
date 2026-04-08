

from env.environment import EmailTriageEnv

def predict(input_data):
    env = EmailTriageEnv()
    obs = env.reset("task_easy")

    return {
        "status": "ok",
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs
    }
