import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# Step 1: Load the Retriever Model
def load_retriever(device):
    # Load SentenceTransformer for embedding documents and queries
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return retriever_model

# Step 2: Load the Qwen-2.5 Model
def load_qwen_model(device):
    # model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Replace with the exact model name if needed
    # model_name = "Qwen/Qwen2.5-3B-Instruct"  # Replace with the exact model name if needed
    model_name = "Qwen/Qwen2.5-1.5B"  # Replace with the exact model name if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    return tokenizer, model

# Step 3: Create a Knowledge Base
def create_knowledge_base(documents, retriever, device):
    # Encode all documents into embeddings
    embeddings = retriever.encode(documents, convert_to_tensor=True, device=device)
    return embeddings

# Step 4: Query the Knowledge Base
def retrieve_relevant_documents(query, documents, retriever, embeddings, device, top_k=3):
    # Encode the query and compute similarity scores
    query_embedding = retriever.encode(query, convert_to_tensor=True, device=device)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    relevant_docs = [documents[i] for i in top_results.indices]

    # Debugging output for retrieved documents
    print("\n[DEBUG] Relevant Documents and Scores:")
    for i, (doc, score) in enumerate(zip(relevant_docs, top_results.values)):
        print(f"{i + 1}: {doc} (Score: {score:.4f})")
    return relevant_docs

# Step 5: Generate a Response
def generate_response(query, relevant_docs, tokenizer, model, device):
    # Combine query with relevant documents
    context = "\n".join(relevant_docs[:2])  # Use the top 2 relevant documents
    input_text = (
        f"Use the following context to answer the question:\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )

    # Tokenize input and generate response
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=5,
        temperature=0.7,
        top_p=0.9,
        early_stopping=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 6: Compare Baseline and RAG Responses
def compare_responses(query, documents, retriever, embeddings, tokenizer, model, device):
    # Baseline: Generate response from the query alone
    baseline_response = generate_response(query, [], tokenizer, model, device)

    # RAG-based: Retrieve relevant documents and generate a response
    relevant_docs = retrieve_relevant_documents(query, documents, retriever, embeddings, device)
    rag_response = generate_response(query, relevant_docs, tokenizer, model, device)

    # Print results
    print("\nQuery:", query)
    print("\nBaseline Response (No Context):", baseline_response)
    print("----------------------------------------------------\n")
    print("\nRelevant Documents:", relevant_docs)
    print("\nRAG-based Response (With Context):", rag_response)

# Main Execution
if __name__ == "__main__":
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    retriever = load_retriever(device)
    tokenizer, qwen_model = load_qwen_model(device)

    # Define a small knowledge base
    documents = [
        "Ryan Hieda is an AI mastery student.",
        "Deep learning is a type of machine learning that utilizes neural networks with many layers.",
        "Machine learning is a subset of AI that uses statistical methods to enable machines to improve with experience."
    ]
    embeddings = create_knowledge_base(documents, retriever, device)

    # Define a query
    query = "Who is Ryan Hieda?"

    # Compare responses
    compare_responses(query, documents, retriever, embeddings, tokenizer, qwen_model, device)
