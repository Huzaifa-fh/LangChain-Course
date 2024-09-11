import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingesting....")

    loader = TextLoader("./mediumblog1.txt", encoding = 'UTF-8')
    document = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Ingesting....")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Finish")