from langchain_ollama import ChatOllama


qwen = ChatOllama(model="qwen2.5:latest", temperature=0)


def node_qwen(prompt: str):
    response = qwen.invoke(input=prompt)
    return response


def main():
    prompt = input(">>> ")
    response = node_qwen(prompt)
    print(response.content)


main()
