from langchain_core.messages import SystemMessage,HumanMessage
from src.states.tax_state import TaxState

class SupervisorNode:
    def __init__(self,llm):
        self.llm=llm

    def generate_response(self,state:TaxState):
        """
        Basic Chatbot logic: Takes history, adds system prompt, gets response.
        """

        # The Persona Definition
        system_prompt = """You are AgentTax 007, an AI Tax Filing Coach for Indian taxpayers. 
        Your goal is to make tax filing simple, conversational, and stress-free.
        
        Guidelines:
        1. Answer in plain English (or Hindi/Marathi if asked).
        2. Be proactive. If they say "I earn 12L", warn them about tax liability immediately.
        3. Do NOT give hallucinated legal advice. If unsure, say you need to check the database.
        
        Current context: You are in the 'Basic Chat' mode."""

        system_message = SystemMessage(content=system_prompt)

        messages = [system_message] + state["messages"]

        # Call LLM
        response = self.llm.invoke(messages)

        # Return the update to the state (LangGraph automatically appends this message)
        return {"messages": [response]}