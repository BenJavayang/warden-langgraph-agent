import os
import time
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define State
class AgentState(TypedDict):
    messages: List[str]
    analysis_result: str

# Initialize DeepSeek Model
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

# Define Analyst Node
def market_analyst(state: AgentState):
    """Analyzes market sentiment using DeepSeek via LangGraph"""
    print("\n🤖 [LangGraph] Analyst is thinking via DeepSeek...")

    messages = [
        SystemMessage(content="You are a senior crypto analyst on Warden Protocol. Provide a brief, one-sentence market sentiment analysis."),
        HumanMessage(content="Analyze current ETH market sentiment.")
    ]

    response = llm.invoke(messages)
    return {"analysis_result": response.content}

# Build the Graph (Official Requirement)
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyst", market_analyst)
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", END)
    return workflow.compile()

# Main Execution Loop
def main():
    print("🚀 Initializing Warden LangGraph Agent...")
    app = build_graph()
    print("✅ LangGraph Structure Built Successfully.")

    while True:
        try:
            print(f"\n⏱️  [{time.strftime('%H:%M:%S')}] Starting Analysis Cycle...")
            result = app.invoke({"messages": []})
            print(f"💡 Agent Output: {result['analysis_result']}")
            print("💤 Sleeping for 5 minutes...")
            time.sleep(300)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()