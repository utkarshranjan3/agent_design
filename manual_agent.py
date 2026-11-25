import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_tool import search_hr_documents
from employee_tool import get_employee_details
from calculator_tool import calculate_leave_balance

load_dotenv()

client = OpenAI()



tools = [
    {
        "type": "function",
        "name": "search_hr_documents",
        "description": (
            "Search company HR policy documents "
            "to answer questions about leave, "
            "working hours, expenses and company policies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The HR question to search for."
                }
            },
            "required": ["question"],
            "additionalProperties": False
        },
        "strict": True
    },

    # TOOL 2: EMPLOYEE DATABASE
    {
        "type": "function",
        "name": "get_employee_details",
        "description": "Fetch employee details and leave balances using employee ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string"
                }
            },
            "required": ["employee_id"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "calculate_leave_balance",
        "description": "Calculate remaining annual leave after requested days.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_balance": {
                    "type": "number"
                },
                "days_requested": {
                    "type": "number"
                }
            },
            "required": [
                "current_balance",
                "days_requested"
            ],
            "additionalProperties": False
        },
        "strict": True
    }

]


def execute_tool(name, arguments):

    if name == "search_hr_documents":

        return search_hr_documents(
            arguments["question"]
        )

    elif name == "get_employee_details":

        return get_employee_details(
            arguments["employee_id"]
        )

    elif name == "calculate_leave_balance":

        return calculate_leave_balance(
            arguments["current_balance"],
            arguments["days_requested"]
        )

    return "Unknown tool."




from memory import ConversationMemory

# Create one conversation memory instance
memory = ConversationMemory()


def run_agent(user_question):

    # Get previous response ID
    previous_id = memory.get_previous_response_id()

    # Send question to OpenAI
    response = client.responses.create(
        model="gpt-4.1-mini",

        instructions="""
        You are an enterprise HR assistant.

        You have three tools:

        1. search_hr_documents:
           Search HR policies.

        2. get_employee_details:
           Retrieve employee information.
           If employee ID is missing, ask the user.

        3. calculate_leave_balance:
           Calculate remaining leave days.
           First retrieve the actual employee balance.

        IMPORTANT:
        - Remember information from previous messages.
        - Understand follow-up questions using conversation history.
        - Do not ask for information already provided.
        - Never invent employee details or leave balances.
        """,

        input=user_question,

        tools=tools,

        previous_response_id=previous_id,

        parallel_tool_calls=False
    )

    # Execute tools until OpenAI provides final answer
    while True:

        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not function_calls:
            break

        tool_outputs = []

        for item in function_calls:

            print(f"\nTool selected: {item.name}")

            arguments = json.loads(item.arguments)

            result = execute_tool(
                item.name,
                arguments
            )

            print("Tool result:", result)

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result
            })

        # Send tool results back to OpenAI
        response = client.responses.create(
            model="gpt-4.1-mini",

            previous_response_id=response.id,

            instructions="""
            You are an enterprise HR assistant.
            Use previous conversation context.
            Never invent employee information.
            """,

            input=tool_outputs,

            tools=tools,

            parallel_tool_calls=False
        )

    # Save the latest response ID
    memory.update(response.id)

    return response.output_text


if __name__ == "__main__":

    while True:

        question = input("\nAsk HR Agent (or exit): ")

        if question.lower() == "exit":
            break

        if question.lower() == "reset":
            memory.reset()
            print("Conversation memory cleared.")
            continue

        answer = run_agent(question)

        print("\nHR AGENT RESPONSE:")
        print(answer)
