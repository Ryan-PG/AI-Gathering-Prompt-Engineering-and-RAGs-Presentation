from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load Ollama Model
def load_ollama_model(model_name="qwen2.5"):
    # Create an Ollama instance with the specified model
    return Ollama(model=model_name)

# Step 2: Build Knowledge Base
def build_knowledge_base(documents):
    # Use HuggingFaceEmbeddings to encode documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Split documents into chunks for better retrieval granularity
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        docs.extend(chunks)

    # Create FAISS vector store for semantic search
    vector_store = FAISS.from_texts(docs, embedding=embeddings)
    return vector_store

# Step 3: Set Up Retrieval-Augmented Generation (RAG)
def setup_rag_system(vector_store, llm):
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Step 4: Compare Baseline and RAG Responses
def compare_responses(query, qa_chain, llm):
    # Baseline: Generate response directly from the LLM
    baseline_response = llm(query)

    # RAG-based: Retrieve relevant documents and generate a response
    rag_response = qa_chain.run(query)

    # Print results
    print("\nQuery:", query)
    print("\nBaseline Response (No Context):")
    print(baseline_response)
    print("\nRAG-based Response (With Context):")
    print(rag_response)

# Main Execution
if __name__ == "__main__":
    # Step 1: Load Ollama model
    llm = load_ollama_model(model_name="llama")

    # Step 2: Define a small knowledge base
    documents = [
        "Natural language processing is a field of AI that focuses on the interaction between computers and humans using natural language.",
        "Machine learning is a subset of AI that uses statistical methods to enable machines to improve with experience.",
        "Deep learning is a type of machine learning that utilizes neural networks with many layers."
    ]
    vector_store = build_knowledge_base(documents)

    # Step 3: Set up RAG system
    qa_chain = setup_rag_system(vector_store, llm)

    # Step 4: Ask a query and compare responses
    query = "What is the difference between deep learning and machine learning?"
    compare_responses(query, qa_chain, llm)
