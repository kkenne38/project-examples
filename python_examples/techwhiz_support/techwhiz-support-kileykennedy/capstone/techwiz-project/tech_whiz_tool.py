
# -----------------------------------------------------------------------------
#       CREATING A TECH WHIZ TOOL 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#   RAG PIPELINE (INTERNAL DOCUMENTS)
# -----------------------------------------------------------------------------
# %% --------------------------------------------------------------------------
# Imports for script 
# -----------------------------------------------------------------------------
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Paths and logging
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
PDF_DIR = Path(os.getenv("POLICIES_PATH", DATA_DIR / "policies"))

# %% --------------------------------------------------------------------------
# Chatbot setup 
# -----------------------------------------------------------------------------
# ============================================================
# CONFIG — override by setting USE_LOCAL_MODELS=1 in your .env
# ============================================================
USE_LOCAL_MODELS  = os.getenv("USE_LOCAL_MODELS", "0").strip().lower() in ("1", "true", "yes")
LOCAL_LLM_MODEL   = os.getenv("LOCAL_LLM_MODEL",   "google/gemma-4-e2b-it")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")

USE_LOCAL_MODELS  = os.getenv("USE_LOCAL_MODELS", "0").strip().lower() in ("1", "true", "yes")

# If local mode requested, ensure torch is available
if USE_LOCAL_MODELS:
    try:
        import torch
    except ImportError:
        logging.warning("Torch not installed; falling back to Azure OpenAI path.")
        USE_LOCAL_MODELS = False

if USE_LOCAL_MODELS:
    # ── LOCAL PATH (HuggingFace, no API key needed) ──────────────
    from transformers import pipeline as hf_pipeline
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    device = ("cuda"  if torch.cuda.is_available() else
              "mps"   if torch.backends.mps.is_available() else
              "cpu")
    print(f"Device: {device}  |  USE_LOCAL_MODELS: {USE_LOCAL_MODELS}")
    _pipe = hf_pipeline("text-generation", model=LOCAL_LLM_MODEL,
                        device=device, max_new_tokens=512)

    def generate(system_prompt, user_message):
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}]
        return _pipe(msgs)[0]["generated_text"][-1]["content"]

    embedding_function = SentenceTransformerEmbeddingFunction(LOCAL_EMBED_MODEL)
    print(f"Local models ready  |  LLM: {LOCAL_LLM_MODEL}  |  Embed: {LOCAL_EMBED_MODEL}")

else:
    # ── AZURE OPENAI PATH ────────────────────────────────
    from openai import AzureOpenAI

    _client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    def generate(system_prompt, user_message):
        response = _client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ]
        )
        return response.choices[0].message.content

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("OPENAI_API_VERSION"),
        deployment_id=os.getenv("EMBEDDING_MODEL")  # from .env, not hard-coded
    )
    print("Azure OpenAI setup complete.")

# Initialize the persistent ChromaDB database engine client
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))

# Now the previous code block can safely run:
collection = chroma_client.get_or_create_collection(
    name="company_policies",
    embedding_function=embedding_function  # Pass the dynamic instance here
)

collection = chroma_client.get_or_create_collection(
    name="company_policies",
    embedding_function=embedding_function
)

# Build policy index from PDFs if empty
if collection.count() == 0:
    logging.info(f"Building policy index from PDFs in {PDF_DIR}")
    loader = DirectoryLoader(path=str(PDF_DIR), glob="*.pdf", loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(raw_documents)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"policy_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    logging.info(f"Indexed {len(chunks)} policy chunks")

# Retrieval helper
def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Return top_k policy text chunks for a given query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]

# RAG prompt for contextual policy QA
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "If the answer is not in the context, say \"I don't have enough information to answer this.\""
)

def ask_rag(question: str) -> str:
    """Answer policy questions by retrieving context and querying the LLM."""
    chunks = retrieve(question, top_k=3)
    context = "\n\n".join(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    return generate(RAG_SYSTEM_PROMPT, user_message)


# %% --------------------------------------------------------------------------
# Test RAG 
# -----------------------------------------------------------------------------
q1 = "What is the return window for electronics?"
print(f"Question: {q1}")
print(f"Answer: {ask_rag(q1)}")




# -----------------------------------------------------------------------------
# Tools - Inventory and Product Details 
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# Imports and connection
# -----------------------------------------------------------------------------
import json
import duckdb

# Database connection (read-only)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "data" / "techwiz.db"))
conn = duckdb.connect(str(DB_PATH), read_only=True)


# %% --------------------------------------------------------------------------
# FUNCTION 1: Inventory Tool 
# -----------------------------------------------------------------------------
def check_inventory(product_name: str) -> str:
    """Check how many units of a product are currently in stock at TechWiz."""
    
    # 1. Handle global list requests first
    if product_name.lower() in ["all", "everything", "inventory", "different products", "products"]:
        query = """
            SELECT p.product_name, SUM(i.quantity_available) AS total
            FROM inventory i 
            JOIN products p ON i.product_id = p.product_id
            GROUP BY p.product_name
            HAVING SUM(i.quantity_available) > 0
        """
        results = conn.execute(query).fetchall()
        inventory_list = {row[0]: int(row[1]) for row in results}
        return json.dumps({"status": "global_catalog", "available_stock_list": inventory_list})

    # 2. AI CATEGORY TRANSLATION LAYER
    # Detect if the user input is a category instead of a specific item model
    category_keywords = ["gaming", "console", "consoles", "case", "cases", "accessory", "accessories", "headphones", "cables", "storage"]
    
    if any(kw in product_name.lower() for kw in category_keywords):
        # Fire a quick query to translate the generic category into specific brand/item filters
        translation_prompt = f"""You are a database query translation assistant. 
Look at this category inquiry: "{product_name}".
Based on our store items (Xbox, PlayStation, Nintendo, Spigen, Case, Cable, Sony, Bose, AirPods, Seagate), 
output a comma-separated list of the 2 or 3 exact keywords found in our database that fit this category.
Example Output for 'gaming consoles': Xbox, PlayStation, Nintendo
Example Output for 'cases': Case, Protector
Only output the keywords separated by commas, nothing else."""

        translation_response = _client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            temperature=0.0,
            messages=[{"role": "user", "content": translation_prompt}]
        )
        
        # Parse out keywords (e.g., ['Xbox', 'PlayStation', 'Nintendo'])
        raw_text = translation_response.choices[0].message.content
        keywords = [kw.strip() for kw in raw_text.split(",") if kw.strip()]
        
        # Build a dynamic multi-keyword SQL query using OR logic
        sql_clauses = " OR ".join(["LOWER(p.product_name) LIKE LOWER(?)" for _ in keywords])
        stock_query = f"""
            SELECT SUM(i.quantity_available) 
            FROM inventory i 
            JOIN products p ON i.product_id = p.product_id 
            WHERE {sql_clauses}
        """
        
        # Prepare wildcard search parameters (e.g., ['%Xbox%', '%PlayStation%', '%Nintendo%'])
        search_patterns = [f"%{kw}%" for kw in keywords]
        result = conn.execute(stock_query, search_patterns).fetchone()
        total = result[0] if (result and result[0] is not None) else 0
        
        return json.dumps({
            "product_name": product_name,
            "quantity_available": total,
            "status": "in_stock" if total > 0 else "out_of_stock",
            "translated_keywords_searched": keywords
        })

    # 3. STANDARD SEARCH: For specific single models (e.g., "Razer Blade", "iPhone 15")
    search_pattern = f"%{product_name}%"
    exist_query = "SELECT COUNT(*) FROM products WHERE LOWER(product_name) LIKE LOWER(?)"
    exists_result = conn.execute(exist_query, [search_pattern]).fetchone()
    
    if not exists_result or exists_result[0] == 0:
        return json.dumps({
            "product_name": product_name,
            "status": "not_found",
            "error": f"Product '{product_name}' does not exist in our catalog."
        })

    stock_query = """
        SELECT SUM(i.quantity_available) 
        FROM inventory i 
        JOIN products p ON i.product_id = p.product_id 
        WHERE LOWER(p.product_name) LIKE LOWER(?)
    """
    result = conn.execute(stock_query, [search_pattern]).fetchone()
    total = result[0] if (result and result[0] is not None) else 0
    
    return json.dumps({
        "product_name": product_name,
        "quantity_available": total,
        "status": "in_stock" if total > 0 else "out_of_stock"
    })


# %% --------------------------------------------------------------------------
# FUNCTION 2: Product Inventory 
# -----------------------------------------------------------------------------

def get_product_details(product_name: str) -> str:
    """Get the price, description, and warranty period for one or multiple matching products."""
    # If the user asks for generic options or options after a specific lookup
    if product_name.lower() in ["options", "more options", "different", "more"]:
        query = "SELECT product_name, price FROM products LIMIT 5"
        results = conn.execute(query).fetchall()
        options = {row[0]: f"${row[1]:.2f}" for row in results}
        return json.dumps({"status": "multiple_options_found", "catalog_samples": options})

    # Standard search logic, allowing up to 3 results instead of cutting off at 1
    query = """
        SELECT product_name, price, description, warranty_months 
        FROM products 
        WHERE LOWER(product_name) LIKE LOWER(?) 
        LIMIT 3
    """
    results = conn.execute(query, [f"%{product_name}%"]).fetchall()
    
    if not results:
        return json.dumps({"error": f"Product '{product_name}' not found."})
    
    # If multiple variations exist in the database, return them all to the chatbot
    items = []
    for row in results:
        items.append({
            "product_name": row[0],
            "price": f"${row[1]:.2f}",
            "description": row[2],
            "warranty_months": row[3]
        })
        
    return json.dumps({"status": "success", "matches": items})



# %% --------------------------------------------------------------------------
# TOOLS SCHEMA DEFINITIONS
# -----------------------------------------------------------------------------

tools = [
    {
        "type": "function", 
        "function": {
            "name": "check_inventory", 
            "description": "Check stock levels. Use this for specific queries like 'how many laptops' AND for broad queries like 'what do you have in stock' or 'list your inventory'.", 
            "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_product_details", 
            "description": "Get product specs, prices, and warranties. Use this when a user asks for alternative product options, price lists, or details about a single item.", 
            "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}
        }
    }
]



# %% --------------------------------------------------------------------------
# Connect these tools back to the chatbot 
# -----------------------------------------------------------------------------
available_functions = {
    "check_inventory": check_inventory,
    "get_product_details": get_product_details
}

def ask_agent(user_message: str) -> str:
    """
    An advanced agent loop that uses a while loop to sequentially execute multiple tools 
    if requested by the LLM, falling back to a validated Chroma RAG system for policy queries.
    """
    system_prompt = (
        "You are a helpful customer service assistant for TechWiz. "
        "You have access to tools to check live product data and inventory. "
        "For company policy or rule questions, use your internal knowledge space."
    )

    # Initialize message history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # === ITERATIVE AGENT LOOP (Handles chained tool execution) ===
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        response = _client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            temperature=0.3,
            messages=messages,
            tools=tools,               
            tool_choice="auto"
        )
        
        # Safe extraction for various SDK versions
        choice = response.choices[0] if isinstance(response.choices, list) else response.choices
        response_message = choice.message if hasattr(choice, 'message') else choice['message']
        tool_calls = getattr(response_message, 'tool_calls', None) if hasattr(response_message, 'tool_calls') else response_message.get('tool_calls')

        # If no tools are requested on this turn
        if not tool_calls:
            # If we already processed a tool call on a prior loop, return the tool-synthesized answer
            if len(messages) > 2: 
                return response_message.content if hasattr(response_message, 'content') else response_message['content']
            # Otherwise, it's a direct policy question from the start -> Break out to RAG Branch
            break

        # Process tool calls requested by the model
        messages.append(response_message)
        for tool_call in tool_calls:
            f_obj = tool_call.function if hasattr(tool_call, 'function') else tool_call['function']
            f_name = f_obj.name if hasattr(f_obj, 'name') else f_obj['name']
            f_args = json.loads(f_obj.arguments if hasattr(f_obj, 'arguments') else f_obj['arguments'])
            t_id = tool_call.id if hasattr(tool_call, 'id') else tool_call['id']
            
            logging.info(f"[Agent] Loop {iteration} - executing tool {f_name}({f_args})")
            
            function_to_call = available_functions[f_name]
            tool_output = function_to_call(**f_args)
            
            messages.append({
                "tool_call_id": t_id,
                "role": "tool",
                "name": f_name,
                "content": tool_output,
            })
            
        # The loop continues to see if the model wants to call *another* tool with the new data

    else:
        return "Agent error: Maximum tool execution loop depth reached."

    # === POLICY BRANCH (FALLBACK TO CHROMA RAG) ===
    logging.info("[Agent] No tools requested; falling back to policy RAG")
    return ask_rag(user_message)



# # %% --------------------------------------------------------------------------
# # Test combined agent 
# # -----------------------------------------------------------------------------
# # Test 1: Fallback back to standard RAG reasoning
# q1 = "What is the return window for electronics?"
# print(f"Question: {q1}")
# print(f"Answer: {ask_agent(q1)}\n")

# # Test 2: Triggering the live inventory database tool
# q2 = "Do you have any laptops in stock?"
# print(f"Question: {q2}")
# print(f"Answer: {ask_agent(q2)}\n")

# # Test 3: Triggers the non-existent tool criteria
# q3 = "Can I buy a flying carpet here?"
# print(f"Question: {q3}")
# print(f"Answer: {ask_agent(q3)}\n")

# # Test 3: Triggers the non-existent tool criteria
# q4 = "How much is a laptop and how long is the warrenty for?"
# print(f"Question: {q4}")
# print(f"Answer: {ask_agent(q4)}\n")

# # Test 3: Triggers the non-existent tool criteria
# q5 = "Can I return a laptop I bought 3 weeks ago?"
# print(f"Question: {q5}")
# print(f"Answer: {ask_agent(q5)}\n")

# # Test 3: Triggers the non-existent tool criteria
# q6 = "What’s your warranty on headphones, and do you have any in stock?"
# print(f"Question: {q6}")
# print(f"Answer: {ask_agent(q6)}\n")
# # %%
