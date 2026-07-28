from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from operator import itemgetter
import pandas as pd
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

try:
    doc=pd.read_csv("products01.csv")
    print("file loaded successfly")
except FileNotFoundError:
    print("Error: 'full_products_chatbot_data.csv' not found. Make sure it's in the same directory.")
    exit()

documents = []
for index, row in doc.iterrows():
    
    page_content = str(row["description"]) 

    documents.append(
         Document(
            page_content=page_content, # Using description as the main content for embedding
            metadata={
                "id": row.get("id", "N/A"),
                "name": row.get("nom", "N/A"), # Corrected: use "nom"
                "price": row.get("prix", "N/A"), # Corrected: use "prix"
                "brand": row.get("marque_name", "N/A"), # Corrected: use "marque_name"
                "category": row.get("category_name", "N/A") # Corrected: use "category_name"
            }  # Store other useful product info here
        )
    )

#text_splitter=RecursiveCharacterTextSplitter(
#    chunk_size=500,
#    chunk_overlap=20,
#    is_separator_regex=False
#)
#splitted_doc=text_splitter.split_documents([document])
embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore=FAISS.from_documents(documents,embeddings)
retriever=vectorstore.as_retriever()

llm=ChatOllama(model="gemma3:latest", temperature=0.3)


# 1. Define the System Message as an explicit PromptTemplate
system_template_string = (
    "You are a helpful assistant for answering questions about a cosmetics website products. "
    "You must use the provided data context to answer the user's question. "
    "If the answer is not in the context, politely state that you don't have enough information. "
    "Do not make up information."
)
system_message_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate.from_template(system_template_string)
)

# 2. Define the Human Message as an explicit PromptTemplate
human_template_string = (
    "Here is the data context:\n---\n{context}\n---\n\nQUESTION: {question}"
)
human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate.from_template(human_template_string)
)

# 3. Combine them into the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="history"),
    human_message_prompt
])

def debug_check(input_dict):
    print("\n--- DEBUGGING MESSAGES TO LLM ---")
    print(f"Received question: {input_dict.get('question', 'N/A')}")
    print(f"Received history count: {len(input_dict.get('history', []))}")

    context_preview = input_dict.get('context', 'CONTEXT MISSING OR EMPTY!')
    print(f"Received context (first 500 chars):\n{context_preview[:500]}...") # Print a preview of context
    print(f"Prompt's inferred input variables: {prompt.input_variables}")
    # Generate the actual messages that will be sent to the LLM
    messages_to_llm = prompt.invoke(input_dict).to_messages()
    print("\n--- Actual Messages Being Sent to LLM ---")
    for msg in messages_to_llm:
        # Print type and a preview of content (first 500 chars)
        print(f"{msg.type}:\n{msg.content[:500]}...") 
    print("--- END DEBUGGING ---")

    return input_dict

parser=StrOutputParser()


#Purpose: This function is crucial because your retriever
#  returns a list of Document objects, 
# but your Large Language Model's prompt expects a single,
#  coherent string for the {context} variable
def format_docs(docs):
    content_parts = []
    for doc in docs:
        if doc.page_content:
            content_parts.append(str(doc.page_content))
    return "\n\n".join(content_parts)

chain=({"question":itemgetter("question"),
        "history":itemgetter("history") }
        |RunnablePassthrough.assign(retrieved=itemgetter("question")|retriever)
        |RunnablePassthrough.assign(context=itemgetter("retrieved")|RunnableLambda(format_docs))
        |RunnableLambda(debug_check)
        |prompt|llm|parser)

#RunnableLambda allows to easily integrate any standard Python function directly into an LCEL chain.
#RunnablePassthrough takes the input it receives and passes that exact same input directly to the next step in the chain
#RunnablePassThrough.assign add the output of the actual function with the previous result

history_=[]
print("Chatbot is ready! You can ask questions about the cosmetics website database.")
print("Example questions: 'What tables are in the database?', 'Tell me about the product table schema.', 'List some categories of makeup.'")
print("Type 'exit' to end the conversation.")
while True:
    user=input("you:")
    if user.upper()==("EXIT"):
        print("bye")
        break
    response=chain.invoke({"question":user,"history":history_})
    print("chat:",response)
    history_.append(HumanMessage(content=user))
    history_.append(AIMessage(content=response))

