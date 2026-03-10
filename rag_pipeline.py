from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi


# ---------------- LOAD EMBEDDINGS ----------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------- LOAD VECTOR DATABASE ----------------

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)


# ---------------- LOAD LLM ----------------

llm = Ollama(model="llama3")   # use the model you already have


# ---------------- BUILD BM25 INDEX ----------------

all_docs = db.similarity_search("test", k=100)

corpus = [doc.page_content.split() for doc in all_docs]

bm25 = BM25Okapi(corpus)


# ---------------- RESPONSE CACHE ----------------

response_cache = {}


# ---------------- QUESTION ANSWERING ----------------

def ask_question(query, chat_history):

    # Check cache
    if query in response_cache:
        return response_cache[query], []

    # Vector search
    vector_docs = db.similarity_search(query, k=3)

    # BM25 keyword search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top = sorted(
        zip(bm25_scores, all_docs),
        key=lambda x: x[0],
        reverse=True
    )[:2]

    bm25_docs = [doc for _, doc in bm25_top]

    # Merge results
    docs = vector_docs + bm25_docs

    context = "\n".join([doc.page_content for doc in docs])

    # Chat history
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['bot']}" for h in chat_history]
    )

    prompt = f"""
You are an AI assistant answering questions from documents.

Conversation History:
{history_text}

Context:
{context}

User Question:
{query}

Answer clearly and concisely.
"""

    response = llm.invoke(prompt)

    # Save to cache
    response_cache[query] = response

    return response, docs


# ---------------- QUIZ QUESTION GENERATION ----------------

def generate_question():

    docs = db.similarity_search("important concept", k=1)

    context = docs[0].page_content

    prompt = f"""
You are a technical interviewer.

Using the context below, generate ONE interview-style question.

Context:
{context}

Only output the question.
"""

    question = llm.invoke(prompt)

    return question, context


# ---------------- ANSWER EVALUATION ----------------

def evaluate_answer(question, user_answer, context):

    prompt = f"""
You are a technical interviewer.

Question:
{question}

Reference Context:
{context}

Candidate Answer:
{user_answer}

Evaluate the answer.

Respond in this format:

Evaluation: Correct / Partially Correct / Incorrect

Explanation:
Explain why the answer is correct or incorrect.

Ideal Answer:
Provide the correct answer.
"""

    result = llm.invoke(prompt)

    return result