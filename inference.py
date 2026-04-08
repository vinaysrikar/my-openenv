import os
from openai import OpenAI
from env.environment import EmailTriageEnv

# Environment variables (defaults only for API_BASE_URL and MODEL_NAME, NOT HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "<your-actual-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-actual-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# OpenAI client configured via env variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

def predict(input_data):
    print("START")
    
    env = EmailTriageEnv()
    task = input_data.get("task", "task_easy") if isinstance(input_data, dict) else "task_easy"
    obs = env.reset(task)
    
    print("STEP: resetting environment")
    
    # LLM call using OpenAI client
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": str(obs)}
        ]
    )
    
    result = response.choices[0].message.content
    print("STEP: got LLM response")
    print("END")
    
    return {
        "status": "ok",
        "result": result
    }
