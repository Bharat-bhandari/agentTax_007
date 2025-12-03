

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

from src.llms.openai_llm import OpenAILLM
from src.graphs.graph_builder import GraphBuilder

load_dotenv()
app = FastAPI(title="AgentTax 007 API")

try:
    print("🚀 Initializing AgentTax Brain...")
    llm = OpenAILLM().get_llm()
    graph_builder = GraphBuilder(llm)
    tax_graph = graph_builder.setup_graph()
    print("✅ Graph compiled successfully.")
except Exception as e:
    print(f"❌ Critical Error during startup: {e}")
    raise e

# Define Input Schema
class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest_user"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint to talk to AgentTax 007.
    Input: {"message": "Hello"}
    Output: {"response": "Hi! I am AgentTax..."}
    """
    try:
        # A. Prepare the state
        user_input = request.message

        config = {"configurable": {"thread_id": request.user_id}}

        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        # B. Run the Graph (The Thinking Process)
        # .invoke() runs the graph until it hits END
        result = tax_graph.invoke(initial_state, config=config)
        
        chat_history = []
        for msg in result["messages"]:
            # Map LangChain message types to standard API roles
            role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
            
            chat_history.append({
                "role": role,
                "content": msg.content,
                "type": msg.type 
            })
        
        return {
            "history": chat_history,
            "status": "success"
        }
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. Health Check (Good for deployment)
@app.get("/")
def health_check():
    return {"status": "running", "project": "AgentTax 007"}

if __name__ == '__main__':
    # Run with: python app.py
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)

