from autogen import AssistantAgent, UserProxyAgent

llm_config = {
    "config_list": [
        {
            "model": "llama3.1:8b",
            "api_type": "ollama",
            "base_url": "http://ollama:11434",
        }
    ]
}

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
)

user.initiate_chat(
    assistant,
    message="Reply with exactly: AUTO-GEN CONNECTED"
)

