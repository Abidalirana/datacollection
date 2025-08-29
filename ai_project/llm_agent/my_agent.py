import asyncio
from .config import client, MODEL_NAME
from ai_project.embedding.embedding_retrieval import query_embeddings

from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

# Disable tracing if not needed
set_tracing_disabled(disabled=True)

# Formatter for dot-style responses
class DotFormatter:
    @staticmethod
    def format_list(items: list[str]) -> str:
        return "\n".join([f". {item}" for item in items])

    @staticmethod
    def format_module(name: str, purpose: str, how_to_use: str, benefits: str) -> str:
        return DotFormatter.format_list([
            f"{name.title()}",
            f"Purpose: {purpose}",
            f"How to use: {how_to_use}",
            f"Benefits: {benefits}"
        ])

# Tools
@function_tool
def get_fundedflow_module_info(module_name: str) -> str:
    # Dynamically query modules from DB or config if needed
    modules = query_embeddings("modules")  # <-- This should return a dict like MODULES_DATA
    module = modules.get(module_name.lower())
    if not module:
        return DotFormatter.format_list([
            f"I only know these modules: {', '.join(modules.keys())}",
            "Pick one!"
        ])
    return DotFormatter.format_module(
        module_name,
        module["purpose"],
        module["how_to_use"],
        module["benefits"]
    )

@function_tool
def get_fundedflow_overview() -> str:
    return DotFormatter.format_list([
        "FundedFlow is your all-in-one trader dashboard",
        "Master your mindset",
        "Track your risk",
        "Reflect in your journal",
        "Recover with personalized plans",
        "Stay motivated with loyalty rewards",
        "Sharpen skills in the trading simulator",
        "Goal: Help traders get funded AND stay funded long term"
    ])

@function_tool
def list_fundedflow_modules() -> str:
    modules = query_embeddings("modules")  # Fetch all module names dynamically
    return DotFormatter.format_list(
        ["Modules available:"] + list(modules.keys())
    )

# Agent setup
agent = Agent(
    name="Floki AI Agent",
    instructions=(
        "Hey, I’m Floki! I’m your FundedFlow AI Assistant\n"
        "Core personality:\n"
        ". Super friendly, short, and encouraging\n"
        ". Tie answers back to FundedFlow modules\n"
        "Formatting: dot-style only\n"
        "Boundaries: Only trading/ FundedFlow questions"
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_fundedflow_module_info, get_fundedflow_overview, list_fundedflow_modules],
)

# Runner
async def run_my_agent(user_query: str) -> str:
    # Retrieve relevant embeddings first
    results = query_embeddings(user_query)  # returns relevant journals/trades
    # Optionally, you can feed `results` as context to the agent
    result = await Runner.run(agent, user_query)
    return result.final_output

# Terminal test
if __name__ == "__main__":
    async def main():
        print(". Welcome! I’m Floki, your FundedFlow AI Assistant")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print(". Bye!")
                break
            response = await run_my_agent(query)
            print(f"Floki:\n{response}\n")
    asyncio.run(main())
