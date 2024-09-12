from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader(path = "./langchain-docs/api.python.langchain.com/en/latest", encoding = 'UTF-8')
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embedding = embeddings, index_name = "langchain-doc-index")

    print("Uploading to Pinecone done")

if __name__ == "__main__":
    ingest_docs()