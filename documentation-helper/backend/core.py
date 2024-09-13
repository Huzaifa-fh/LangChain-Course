from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

from ingestion import embeddings

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore


from langchain_openai import ChatOpenAI, OpenAIEmbeddings

INDEX_NAME = "langchain-doc-index"

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name = INDEX_NAME, embedding = embeddings)
    llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0, verbose = True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(retriever = docsearch.as_retriever(),
                                combine_docs_chain = stuff_documents_chain)

    result = qa.invoke(input = {"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query = "What are LangChain runnables? How do I use them, show with code example.")
    print(res)
    print(res["answer"])