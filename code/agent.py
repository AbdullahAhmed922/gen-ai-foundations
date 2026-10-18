# from langchain.agents import create_agent
# from datetime import datetime

# def get_time() -> str:
#     """Return the current local time."""
#     return datetime.now().strftime("%H:%M")

# agent = create_agent(
#     model="ollama:gemma4:31b-cloud",
#     tools=[get_time],
#     system_prompt="You are a helpful assistant."  # Or omit system_prompt entirely
# )

# query = input("Enter your query: ")
# result = agent.invoke(
#     input={"messages": [{"role": "user", "content": query}]}
# )

# print(result["messages"][-1].text)



# It is agentic logic not simple chat

from datetime import datetime
from typing import Optional # Using optional is old way we can use Union[str, None] instead of Optional[str]
from pydantic import BaseModel, Field
from langchain.agents import create_agent

# class AgentResponse(BaseModel):
#     is_greeting: bool = Field(
#         description="True if the user message is a greeting, otherwise False"
#     )
#     time: Optional[str] = Field(
#         description="Current time for greetings, otherwise None"
#     )
#     reply: str = Field(
#         description="A simple and helpful response to the user"
#     )

class AgentResponse(BaseModel):
    is_greeting: bool = Field(
        description="True if the user message is a greeting, otherwise False"
    )
    time: Optional[str] = Field(
        description="Current time for greetings, otherwise None"
    )
    reply: str = Field(
        description="A simple and helpful response to the user"
    )


def get_time() -> str:
    """Return the current local time in HH:MM format."""
    return datetime.now().strftime("%H:%M")


agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[get_time],
    response_format=AgentResponse,
    system_prompt=(
        "You are a helpful AI Agent. "
        "Decide whether the user's message is a greeting. "
        "Do not use a predefined list of greetings. "
        "If the user is greeting you, set is_greeting=True, "
        "call get_time, fill the time field with the tool result, "
        "and greet the user based on the time of day. "
        "Otherwise, set is_greeting=False, set time=None, "
        "and answer the user's question simply and clearly. "
        "Always fill in all fields in the response."
    ),
)

while True:
    query = input("\nYou: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    response = result["structured_response"]

    if response.is_greeting:
        print("Time:", response.time)
        print("Reply:", response.reply)
    else:
        print("Reply:", response.reply)
