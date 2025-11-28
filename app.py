from src.llms.groqllm import GroqLLM
from src.graphs.graph_builder import GraphBuilder
from langchain_core.messages import HumanMessage

llm = GroqLLM().get_llm()
graph = GraphBuilder(llm).setup_graph()

user_input = "Hi, I am a freelancer earning 15 Lakhs. Do I need to pay tax?"
initial_state = {"messages": [HumanMessage(content=user_input)]}

print("User:", user_input)
for event in graph.stream(initial_state):
    for value in event.values():
        print("AgentTax 007:", value["messages"][-1].content)

