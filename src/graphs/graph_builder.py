from langgraph.graph import StateGraph, START, END
from src.states.tax_state import TaxState
from src.nodes.supervisor import SupervisorNode

class GraphBuilder:
    def __init__(self,llm):
        self.llm=llm
        self.graph=StateGraph(TaxState)

    def build_graph(self):
        """
        Builds a simple Single-Node Graph for the Basic Chatbot
        """

        # Initialize Nodes
        self.supervisor_node = SupervisorNode(self.llm)

        # Add Nodes 
        self.graph.add_node("supervisor_node",self.supervisor_node.generate_response)

        # Add Edges
        self.graph.add_edge(START,"supervisor_node")
        self.graph.add_edge("supervisor_node",END)

        return self.graph.compile()
    
    def setup_graph(self):
        return self.build_graph()

