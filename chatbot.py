import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from tomodo.common.models import AtlasDeployment
from tomodo.functional import provision_atlas_instance
from pymongo.operations import SearchIndexModel
import chainlit as cl

# Set up local MongoDB
def setup_local_mongodb():
    deployment: AtlasDeployment = provision_atlas_instance(
        name="dazzling-mosquito",
        port=27017,
        version="7.0",
        username="foo",
        password="bar",
        image_repo="ghcr.io/yuvalherziger/tomodo",
        image_tag="1.1.4",
        network_name="mongo_network"
    )
    return deployment
    #return "mongodb://admin:admin@127.0.0.1:27017"

    # test if the deployment is avaialble
    # os.run("tomodo provision atlas -image-tag 1.1.4")


# Connect to local MongoDB instance
def connect_to_mongodb(deployment):
    client = MongoClient(f"mongodb://admin:admin@127.0.0.1:27017")
    db_name = "langchain_db"
    collection_name = "kb_base"
    collection = client[db_name][collection_name]
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 1536,
                    "path": "embedding",
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "page"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    result = collection.create_search_index(model=search_index_model)
    print(result)
    return client, collection

# Load and process the PDF
def load_and_process_pdf(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(data)
    return docs

# Create embeddings and store in MongoDB
def store_embeddings(docs, collection):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        collection.insert_one({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": vector
        })

# Create the vector store
def create_vector_store(collection):
    return MongoDBAtlasVectorSearch(
        collection,
        OpenAIEmbeddings(disallowed_special=()),
        index_name="vector_index"
    )

# Set up the RAG chain
def setup_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(streaming=True)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

@cl.on_chat_start
async def on_chat_start():
    # Set up local MongoDB
    deployment = setup_local_mongodb()
    client, collection = connect_to_mongodb(deployment)
    
    # Create vector store and RAG chain
    vector_store = create_vector_store(collection)
    rag_chain = setup_rag_chain(vector_store)
    
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("collection", collection)
    
    await cl.Message(content="Welcome! You can start by uploading a PDF or asking questions if PDFs have already been uploaded.").send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    response = await rag_chain.ainvoke(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    
    await cl.Message(content=response).send()

@cl.on_file_upload(accept=["application/pdf"])
async def on_file_upload(file: cl.File):
    collection = cl.user_session.get("collection")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.content)
        temp_file_path = temp_file.name
    
    docs = load_and_process_pdf(temp_file_path)
    store_embeddings(docs, collection)
    
    os.unlink(temp_file_path)
    
    await cl.Message(f"Processed and stored {len(docs)} document chunks from {file.name}").send()

if __name__ == "__main__":
    cl.run()