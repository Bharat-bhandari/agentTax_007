from typing import TypedDict,Annotated,List
from langgraph.graph.message import add_messages

class TaxState(TypedDict):
    messages: Annotated[List,add_messages]

    user_name: str
    income: float
