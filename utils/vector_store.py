from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local("vector_db/stock_vector_db", embedding)

def retrieve_context(query):

    docs = db.similarity_search(query)

    context = "\n".join([doc.page_content for doc in docs])

    return context