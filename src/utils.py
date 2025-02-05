from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Elasticsearch client
es = Elasticsearch(
    os.environ["ELASTIC_SEARCH_LINK"],
    api_key= os.environ["ELASTIC_SEARCH_KEY"]
)
#es.delete_by_query(index="econ101", body={"query": {"match_all": {}}})
# Load embedding model (you can use another model if needed)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def index_document(text):
    # Generate embedding
    embedding = embedding_model.encode(text).tolist()
    
    # Store in Elasticsearch
    doc = {
        "text": text,
        "embedding": embedding
    }
    es.index(index="econ101", body=doc)
    print("✅ Document Indexed")

# function to search for relevant documents 
def search_documents(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()

    search_body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},  # Consider all docs
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }

    results = es.search(index="econ101", body=search_body)
    
    return [hit["_source"]["text"] for hit in results["hits"]["hits"]]

def populate_index():
    # Load JSON data
    with open("src/econ101.json", "r") as f:
        data = json.load(f)

    documents_to_index = []

    for topic_entry in data:
        topic = topic_entry["Topic"]
        for doc in topic_entry["Documents"]:
            full_text = f"{topic}, {doc}"  # Add topic at the beginning of the document
            embedding = embedding_model.encode(full_text).tolist()  # Generate embedding

            # Prepare document for indexing
            document = {
                "text": full_text,
                "embedding": embedding
            }
            documents_to_index.append(document)

    # Bulk index documents
    for i, doc in enumerate(documents_to_index):
        es.index(index="econ101", id=i, document=doc)

    print(f"Indexed {len(documents_to_index)} documents")
#populate_index()