# from urllib import response

# from langchain.chat_models import init_chat_model

# llm = init_chat_model(
#     model = "gemma4:31b-cloud",
#     model_provider = "ollama"
# )

# question = input("Ask Anything: ")
# messages= [
#     {
#         "role": "user",
#         "content": "You are helpful AI assistant. "
#     }

# ]

# response = llm.invoke(input= "Hi")
# print(response.text)


# # from langchain.chat_models import ChatOpenAI
# # from langchain.chat_models import ChatGoogleGenerativeAi
# # from langchain.chat_models import ChatOllama

# from langchain.chat_models import init_chat_model

# llm = init_chat_model(
#     model="ollama:gemma3:4b",
#     temperature=1.0
# )

# SYSTEM_PROMPT = "You are helpful Ai assistant"

# history =[
#     {"role": "system","content":SYSTEM_PROMPT}
# ]

# print("Welcome to cue chat")
# while True:
#     question = input("You: ")

#     if not question:
#         continue

#     if question.lower() in {"EXIT","QUIT","BYE"}:
#         print("\n Goodbye")
#         break

#     history.append({"role":"user","content":question})


#     # response = llm.invoke(input="Hi")
#     response = llm.invoke(history)
#     history.append({"role":"assistant","content":response.text})

#     print(f"Bot: {response.text}")


#     -----------------------------------


# from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import ChatGoogleGenerativeAi
# from langchain.chat_models import ChatOllama

# from urllib import response
import os
from langchain.chat_models import init_chat_model
from datetime import datetime
from langchain.tools import tool

model=os.getenv(key="MODEL_NAME")
if not model_name:
    raise SystemError("MODEL_NAME is not set in .env file (e.g openai:gpt-4)")

llm = init_chat_model(
    model=model_name,
    # temperature=1.0
    api_key = os.getenv(key="API_KEY")
)


SYSTEM_PROMPT = "You are a helpful AI assistant!"



@tool
def get_time()
    """Return get local time"""
    return datetime.now().strftime("%H:%M")
history = [
    { "role": "system", "content": SYSTEM_PROMPT}
]

print("Welcome to Cue chat!")
while True:
    question = input("You: ") # User Message

    if not question:
        continue

    if question.lower() in {"exit", "quit", "bye"}:
        print("\nGoodbye!")
        break

    history.append({"role": "user", "content": question})

    print(f"Bot: ", end="", flush=True)

    full = None
    completed = False
    try:
        for chunk in llm.stream(history):
            print(chunk.text, end="", flush=True)
            full = chunk if full is None else full + chunk
        completed = True
    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception as e:
        print(f"\n[error: {e}]")

    if completed and full is not None and full.text:
        history.append(full)
    else:
        history.pop()

print(history)
