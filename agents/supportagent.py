from langchain.agents import create_agent
from pydantic import BaseModel
from typing import Literal


class Ticket(BaseModel):
    categories: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    is_support_ticket: bool
    summary: str

def greeting() -> str:
    """Return a greeting message."""
    return "Hello! How can I assist you today?"

agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[greeting],
    system_prompt="Classify customer support messages, call the greeting tool if message is greeting",
    response_format=Ticket
)

query = input("Enter your query: ")
result = agent.invoke(
    input={"messages": [{"role": "user", "content": query}]}
)

t = result["structured_response"]
print(f"Priority: [{t.priority}],Category: {t.categories}, Summary: {t.summary}")
