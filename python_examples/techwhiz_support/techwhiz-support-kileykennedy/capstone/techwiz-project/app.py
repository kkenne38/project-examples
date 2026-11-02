import os
import json
import duckdb
import chromadb
import streamlit as st
from openai import AzureOpenAI
from openai._exceptions import APIConnectionError
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(r"C:\Users\KileyKennedy\Desktop\techwhiz_support_kileykennedy\capstone\techwiz-project\.env")


def chunk_text_response(text: str, chunk_size: int = 40):
    """Yield pieces of a response so Streamlit can display a streaming effect."""
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
# ============================================================
# 1. STREAMLIT PAGE CONFIG & INITIALIZATION
# ============================================================
st.set_page_config(page_title="TechWiz Support Agent", page_icon="🤖", layout="centered")
st.title("🤖 TechWiz Support Assistant")
st.caption("Ask me about company policies, product prices, or live inventory!")

# Sidebar: FAQs, usage tips, and chat controls
with st.sidebar:
    st.header("How to Use")
    st.write("Ask about policies, pricing, or stock. Click an FAQ to get started.")
    st.header("FAQs")
    faqs = {
        "What is the return policy?": None,
        "How long is the warranty?": None,
        "Do you ship internationally?": None,
        "Do you have laptops in stock?": None,
        "What are your shipping fees?": None
    }
    for question in faqs:
        if st.button(question, key=question):
            st.session_state.last_faq = question
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Initialize database connections once per session
@st.cache_resource
def init_connections():
    # FIXED: Added 'techwiz-project' back into the path structure
    conn = duckdb.connect(
        r"C:\Users\KileyKennedy\Desktop\techwhiz_support_kileykennedy\capstone\techwiz-project\data\techwiz.db", 
        read_only=True
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="company_policies")
    
    _client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )
    return conn, collection, _client

conn, collection, _client = init_connections()

# Paste your helper tools exactly as they are defined
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

tools = [
    {"type": "function", "function": {"name": "check_inventory", "description": "Check stock levels.", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}},
    {"type": "function", "function": {"name": "get_product_details", "description": "Get product specs, price, and warranty.", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}}, "required": ["product_name"]}}}
]
available_functions = {"check_inventory": check_inventory, "get_product_details": get_product_details}

# ============================================================
# 2. STREAMING AGENT LOOP
# ============================================================
def ask_agent_stream(user_message: str):
    """Executes tools and yields word-by-word text updates for Streamlit visualization."""
    system_prompt = "You are a helpful customer service assistant for TechWiz. Use tools for live product data."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    # Step 1: Structural intent evaluation
    try:
        response = _client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            temperature=0.3,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    except APIConnectionError as exc:
        yield f"*(Error: Unable to reach OpenAI services. {exc} )*"
        return
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # === Branch A: Tool Chaining execution ===
    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            f_name = tool_call.function.name
            f_args = json.loads(tool_call.function.arguments)
            
            yield f"*(Thinking: Checking database for {f_args.get('product_name')}...)*\n\n"
            tool_output = available_functions[f_name](**f_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": f_name,
                "content": tool_output,
            })
            
        # Call final completion with tool context
        try:
            second_response = _client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=messages
            )
        except APIConnectionError as exc:
            yield f"*(Error: Unable to reach OpenAI services. {exc} )*"
            return
        final_text = second_response.choices[0].message.content
        yield from chunk_text_response(final_text)
        return

    
    # === Branch B: Policy RAG or Intelligent Database Interception ===
    inventory_keywords = ["have", "stock", "inventory", "product", "buy", "sell", "carry", "tablet", "smartphone", "else", "total", "catalog", "items", "options", "anything", "laptop", "laptops"]
    policy_keywords = ["return", "returns", "policy", "warranty", "warranties", "refund", "refunds", "exchange", "exchanges", "fee"]

    # 1. PRIORITIZE POLICY OVER INVENTORY: If they ask about rules, skip the database block entirely
    if any(word in user_message.lower() for word in policy_keywords):
        pass # Let it fall through naturally to the standard Policy Manual RAG block below
    
    # 2. Otherwise, if it's strictly a stock question, proceed to database tables
    elif any(word in user_message.lower() for word in inventory_keywords):
        yield "*(Thinking: Querying inventory database tables...)*\n\n"
        
        words = user_message.lower().replace("?", "").split()
        specific_targets = [w for w in ["laptop", "laptops", "smartphone", "smartphones", "tablet", "tablets", "mouse", "keyboard", "monitors"] if w in words]
        
        if specific_targets:
            target_item = specific_targets[0].rstrip('s')
            tool_output = check_inventory(target_item)
        else:
            tool_output = check_inventory("all") 
        
        correction_messages = [
            {
                "role": "system", 
                "content": "You are a customer service representative summarizing database records. Read the provided JSON status. If it indicates stock is available, tell the user exactly how many units are available. Be clear and direct."
            },
            {"role": "user", "content": f"Database Live Status: {tool_output}. User Question: {user_message}"}
        ]
        
        try:
            rag_response = _client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                temperature=0.2,
                messages=correction_messages
            )
        except APIConnectionError as exc:
            yield f"*(Error: Unable to reach OpenAI services. {exc} )*"
            return
        final_text = rag_response.choices[0].message.content
        yield from chunk_text_response(final_text)
        return

    yield "*(Thinking: Searching policy manuals...)*\n\n"

    try:
        results = collection.query(query_texts=[user_message], n_results=3)
    except APIConnectionError as exc:
        yield f"*(Error: Unable to reach OpenAI services. {exc} )*"
        return
    context = "\n\n".join(results["documents"][0])
    
    rag_system_instructions = """You are a helpful customer support agent for TechWiz. Use ONLY the context below. 
If the answer is not in the context, say "I don't have that information." Stay concise."""

    rag_messages = [
        {"role": "system", "content": rag_system_instructions},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
    ]

    try:
        rag_response = _client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            temperature=0.1,
            messages=rag_messages
        )
    except APIConnectionError as exc:
        yield f"*(Error: Unable to reach OpenAI services. {exc} )*"
        return
    final_text = rag_response.choices[0].message.content
    yield from chunk_text_response(final_text)

# ============================================================
# 3. STREAMLIT SESSION STATE & CHAT UI
# ============================================================
# Initialize empty chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_faq" not in st.session_state:
    st.session_state.last_faq = None

# Redraw all historical messages from the current session on screen refresh
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept FAQ-triggered question (runs once) while always showing the chat input
faq_query = None
if st.session_state.last_faq:
    faq_query = st.session_state.last_faq
    st.session_state.last_faq = None

st.markdown("Type your message here…")
typed_query = st.chat_input(
    placeholder="Ask about policies, pricing, or stock…"
)

user_query = typed_query or faq_query

if user_query:
    # Render user message instantly
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Render bot response progressively using st.write_stream
    with st.chat_message("assistant"):
        response_generator = ask_agent_stream(user_query)
        full_response = st.write_stream(response_generator)
        
    # Commit final combined string back to state memory
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
