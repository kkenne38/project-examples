### **The Challenge: "The TechWiz Support Bot"**

**Scenario:**
You have been hired by "TechWiz," a fictional electronics store. They have some PDF/text documents containing detailed policies (shipping, returns, warranties) that the public doesn't know. They also have an internal database (a fake duckdb database) of current product stock.

**Your Goal:**
Build a Streamlit chatbot that can answer questions about company policies (using **RAG**) AND check if specific items are in stock (using **Tools**).

---

### **Project Requirements & Milestones**

#### **Part 1: The Knowledge Base (Vector Search & RAG)**

* **Data:** Text/PDF files containing fake policy info (e.g., _"Returns are accepted within 30 days. Shipping to the moon costs $500."_).
* **Task:**
    1. Write a script to **chunk** this text and generate **OpenAI Embeddings** for each chunk.
    2. Store these vectors in a simple list or a local vector store (like ChromaDB).
    3. Implement a `retrieve(query)` function that takes a user's question, embeds it, calculates **Cosine Similarity**, and returns the top 2 most relevant chunks of text.

#### **Part 2: The Tools (Agents)**

* **Task:** Define a Python function called `check_inventory(product_name)`.
  * It should look up the SQL database () and return the stock count.
* **Integration:** Configure the OpenAI Chat Completion API with a `tools` definition that describes this function so the model knows when to use it.

#### **Part 3: The Brain (The Agent Loop)**

* **System Prompt:** Write a prompt that instructs the model:
  * "You are a helpful support agent for TechWiz."
  * "ALWAYS use the provided context to answer policy questions."
  * "If asked about product availability, use the `check_inventory` tool."

* **Logic:**
    1. When a user sends a message, first run the **RAG retrieval** to get context.
    2. Send the **User Query + RAG Context** to the LLM.
    3. If the LLM decides to call `check_inventory`, execute the Python function and feed the result _back_ to the LLM for the final answer.

#### **Part 4: The Face (Streamlit UI)**

* **Task:** Build a web interface.
  * A simple Chat Input box.
  * A container to display the conversation history.
  * **Requirement:** The final response must **Stream** to the screen (typing effect) for the best user experience.

---

### **Why this is the perfect task:**

| Topic Covered      | How it is applied in this task                                                          |
| :----------------- | :-------------------------------------------------------------------------------------- |
| **OpenAI API**     | Used for Embeddings and Chat generation.                                                |
| **Vector Search**  | Used to find the relevant "Shipping/Return Policy" text.                                |
| **RAG**            | Combining the retrieved policy text with the user's question.                           |
| **Tools / Agents** | The model intelligently deciding to call `check_inventory` only when asked about stock. |
| **Streamlit**      | Wrapping the logic in a shareable web app.                                              |

### **Example Interaction to Aim For:**

> **User:** "Can I return a laptop I bought yesterday?"
> **Bot (Using RAG):** "Yes, according to our policy, returns are accepted within 30 days."
>
> **User:** "Great. Do you have any replacement laptops in stock?"
> **Bot (Using Tool):** _[Calls check_inventory("laptop")]_ ... "Yes! We currently have 5 laptops in stock."
