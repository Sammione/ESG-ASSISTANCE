import os
import google.generativeai as genai
from retriever import create_vector_store, retrieve_docs
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Load Gemini API key
# ---------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file.")
    exit()
else:
    print("Gemini API Key loaded successfully.")

# Configure Gemini
genai.configure(api_key=api_key)

# ---------------------------
# Load ESG documents
# ---------------------------
docs_path = "esg_docs"
if not os.path.exists(docs_path):
    print("ERROR: Folder 'esg_docs' not found. Please create it and add some .txt files.")
    exit()

print(f"Loading ESG documents from '{docs_path}'...")
vector_store = create_vector_store(docs_path)
print(" ESG documents indexed successfully!")

# ---------------------------
# Initialize generation model
# ---------------------------
GEN_MODEL = "models/gemini-2.5-flash"
try:
    model = genai.GenerativeModel(GEN_MODEL)
    print(f"Generation model '{GEN_MODEL}' loaded successfully.")
except Exception as e:
    print("Error loading generation model:", e)
    exit()

# ---------------------------
# Start interactive loop
# ---------------------------
print("\nESG Assistant is ready!")
print("Type your question below (or type 'exit' to quit)\n")

while True:
    query = input("Your question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting ESG Assistant. Goodbye!")
        break
    if not query:
        continue

    # Retrieve top relevant docs
    context = retrieve_docs(vector_store, query)
    if not context.strip():
        print("No relevant context found for that query.\n")
        continue

   
    try:
        prompt = f"You are an ESG assistant. Use the following ESG context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = model.generate_content(prompt)
        print("\n Answer:", response.text.strip(), "\n")
    except Exception as e:
        print(" Gemini API Error:", e, "\n")
