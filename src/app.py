import ollama
import chainlit as cl
from langchain.prompts import PromptTemplate
from utils import search_documents

# Define a prompt template with LangChain
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="You are an expert in economics. Use the following information to answer the question, answer as brief as possible. Don't over explain! It's not necessary to use the context information for very simple questionsn\n"
             "Context: {context}\n\n"
             "Question: {question}\n\n"
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])

@cl.on_message
async def generate_response(query: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    chat_history.append({"role": "user", "content": query.content})
    print(f"your prompt: {query.content}")

    # Retrieve relevant context from Elasticsearch
    retrieved_docs = search_documents(query.content)
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."
    print(f"relevant documents: {retrieved_docs}")

    # Format the prompt using LangChain
    full_prompt = prompt_template.format(context=context, question=query.content)
    print(f"full prompt: {full_prompt}")

    response = cl.Message(content="")  # Placeholder for streaming
    try:
        # Generate response with Llama2
        answer = ollama.chat(model="llama2", messages=[{"role": "user", "content": full_prompt}], stream=True)

        complete_answer = ""
        for token_dict in answer:
            token = token_dict.get("message", {}).get("content", "")
            complete_answer += token
            await response.stream_token(token)

        # Append response to history
        chat_history.append({"role": "assistant", "content": complete_answer})
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        response.content = f"❌ Error: {str(e)}"

    await response.send()
