import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from openai import embeddings
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    pdf_path = "./2210.03629v3.pdf"

    loader = PyPDFLoader(file_path = pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30, separator = '\n')
    docs = text_splitter.split_documents(documents = documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    llm = ChatOpenAI(model_name="gpt-4o-mini")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    query = """Give me the gist of ReAct in 3 sentences"""

    result = retrival_chain.invoke(input={"input": query})
    print(result)
    print(result["answer"])