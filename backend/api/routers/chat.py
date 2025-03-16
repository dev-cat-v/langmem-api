from typing import Any, Dict, List
from langgraph.store.memory import InMemoryStore
from openai import OpenAI
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

class ChatRequest(BaseModel):
    user_id: str
    user_message: str


@router.post("/")
def chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.user_message

    memory_tools = [
        create_manage_memory_tool(namespace=("memories",user_id), store=store),
        create_search_memory_tool(namespace=("memories",user_id), store=store),
    ]

    result = run_agent(
        tools=memory_tools,
        user_input=user_message,
    )

    user_memories = store.search(("memories",user_id))
    serialized_memories = [memory.dict() for memory in user_memories]

    response = {
        "message": result,
        "memories": serialized_memories
    }

    return response

def execute_tool(tools_by_name: Dict[str, Any], tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result"""
    tool_name = tool_call["function"]["name"]

    if tool_name not in tools_by_name:
        return f"Error: Tool {tool_name} not found"

    tool = tools_by_name[tool_name]
    try:
        result = tool.invoke(tool_call["function"]["arguments"])
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent(tools: List[Any], user_input: str, max_steps: int = 5) -> str:
    """Run a simple agent loop that can use tools"""

    client = OpenAI()
    tools_by_name = {tool.name: tool for tool in tools}

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.tool_call_schema.model_json_schema(),
            },
        }
        for tool in tools
    ]

    messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools if step < max_steps - 1 else [],
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            return message.content

        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": tool_calls}
        )

        for tool_call in tool_calls:
            tool_result = execute_tool(tools_by_name, tool_call.model_dump())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return "Reached maximum number of steps"