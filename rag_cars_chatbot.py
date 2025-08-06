import os
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import argparse

def load_csv_to_documents(csv_path):
    """Load cars dataset from CSV and convert to LangChain documents."""
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        # Create a text representation of each car entry
        content = (
            f"Car Company: {row['Car_Company']}\n"
            f"Model: {row['Car_Model']}\n"
            f"Engine Type: {row['Engine_Type']}\n"
            f"CC/Battery Capacity: {row['CC_Battery_Capacity']}\n"
            f"Horsepower: {row['Horsepower_HP']} HP\n"
            f"Top Speed: {row['Top_Speed']}\n"
            f"0-100 km/h: {row['Zero_To_Hundred']}\n"
            f"Price: ${row['Price_USD']}\n"
            f"Fuel Type: {row['Fuel_Type']}\n"
            f"Seating Capacity: {row['Seating_Capacity']}\n"
            f"Torque: {row['Torque']}\n"
        )
        documents.append(Document(page_content=content, metadata={"source": f"{row['Car_Company']}_{row['Car_Model']}"}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """Create FAISS vector store from documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

def retrieve_documents(vector_store, query, k=3):
    """Retrieve top-k relevant documents for a given query."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

def main():
    parser = argparse.ArgumentParser(description="RAG-based Cars Dataset Retrieval Chatbot")
    parser.add_argument('--csv_path', type=str, default='data/cars_dataset.csv', help='Path to cars dataset CSV')
    args = parser.parse_args()

    # Load and process dataset
    print("Loading dataset...")
    documents = load_csv_to_documents(args.csv_path)
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(documents)
    
    # Interactive CLI
    print("Welcome to the Cars Dataset Retrieval Chatbot! Type 'exit' to quit.")
    while True:
        query = input("Ask a question about cars (e.g., 'Which cars have over 300 HP?'): ")
        if query.lower() == 'exit':
            break
        
        # Retrieve relevant documents
        results = retrieve_documents(vector_store, query)
        
        # Display results
        print("\nRetrieved Cars:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:\n{doc.page_content}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

if __name__ == "__main__":
    main()