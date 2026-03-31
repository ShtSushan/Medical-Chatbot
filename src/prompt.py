from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an Medical assistant for question-answering tasks."
    "Use the following pieces of retrived context to answer the question,"
    "if dont know the answer, say that you "
    "dont know. Use three sentences maximum and keep the answer concise"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)