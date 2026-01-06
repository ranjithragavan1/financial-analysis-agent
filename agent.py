import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found! Please check your .env file.")

def run_agent():
    print("--- Initializing the Agent ---")
    
    # 1. Load Memory
    print("Loading vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Define the Tool Function
    def search_financials(query: str):
        print(f"   [Tool Called]: Searching PDF for '{query}'...")
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    # 3. Wrap it as a Tool
    finance_tool = Tool(
        name="search_apple_financials",
        func=search_financials,
        description="Use this tool to find financial numbers, sales, or revenue from the Apple 2023 report."
    )
    tools = [finance_tool]

    # 4. Connect to Brain (Gemini)
    print("Connecting to Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # 5. Create Agent
    agent_executor = create_react_agent(llm, tools)

    # 6. THE INTERACTIVE LOOP
    while True:
        try:
            # Get User Input
            user_input = input("\nUse > ")
            
            # Exit condition
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            # Run Agent
            response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
            
            # Clean Output Logic
            raw_content = response['messages'][-1].content
            
            if isinstance(raw_content, list):
                # If Gemini returns a list of blocks, extract just the text
                final_text = "".join([block['text'] for block in raw_content if 'text' in block])
            else:
                final_text = raw_content

            print(f" AI: {final_text}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_agent()