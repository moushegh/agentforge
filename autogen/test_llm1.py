from autogen import AssistantAgent

llm_config = {
    "config_list": [
        {
            "model": "llama3.1:8b",
            "api_type": "ollama",
            "base_url": "http://ollama:11434",
        }
    ]
}

agent = AssistantAgent(
    name="sanity_check",
    llm_config=llm_config,
)

response = agent.initiate_chat(
    recipient=agent,
    message="Reply with exactly: AUTO-GEN CONNECTED"
)

print(response)
