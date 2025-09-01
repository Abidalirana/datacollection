import asyncio
from pydantic import BaseModel
from .config import client, MODEL_NAME
from ai_project.embedding.embedding_retrieval import query_embeddings

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

# -------------------
# Disable tracing
# -------------------
set_tracing_disabled(disabled=True)

# -------------------
# Formatter
# -------------------
class DotFormatter:
    @staticmethod
    def format_list(items: list[str]) -> str:
        return "\n".join([f". **{item}**" for item in items])

    @staticmethod
    def format_numbered(items: list[str]) -> str:
        return "\n".join([f"{i+1}. **{item}**" for i, item in enumerate(items)])

# -------------------
# Output Schema
# -------------------
class MessageOutput(BaseModel):
    response: str

# -------------------
# Core Website Content (Fallback)
# -------------------
CORE_CONTENT = {
    "modules": {
        "7-day reset challenge": {
            "purpose": "Reset mindset after tough trading patches",
            "how_to_use": "Follow daily prompts for 7 days",
            "benefits": "Builds mental strength & focus",
        },
        "risk tracker": {
            "purpose": "Track risk habits & trading patterns",
            "how_to_use": "Log trades, emotions & analyze",
            "benefits": "Improves discipline & consistency",
        },
        "trading journal": {
            "purpose": "Reflect on trades",
            "how_to_use": "Log trades & review patterns",
            "benefits": "Boosts decision-making & self-awareness",
        },
        "recovery plan generator": {
            "purpose": "Create personalized improvement plans",
            "how_to_use": "Generates PDF reports",
            "benefits": "Clear next steps & growth",
        },
        "loyalty program": {
            "purpose": "Rewards consistent discipline",
            "how_to_use": "Earn points & unlock perks",
            "benefits": "Keeps you motivated",
        },
        "trading simulator": {
            "purpose": "Practice strategies risk-free",
            "how_to_use": "Simulate trades & analyze results",
            "benefits": "Sharpen skills & confidence",
        },
    },
    "overview": [
        "FundedFlow is your all-in-one trader dashboard",
        "Master your mindset",
        "Track your risk",
        "Reflect in your journal",
        "Recover with personalized plans",
        "Stay motivated with loyalty rewards",
        "Sharpen skills in the trading simulator",
        "Goal: Help traders get funded AND stay funded long term"
    ],
    "faqs": {
        "what is fundedflow": "FundedFlow is a trader platform helping you get funded and stay disciplined in trading.",
        "how to use the journal": "The Trading Journal lets you log trades, emotions, and lessons to learn from every move.",
        "what is the simulator": "The Trading Simulator allows you to practice strategies risk-free and analyze results to sharpen skills.",
        "how does loyalty work": "The Loyalty Program rewards consistent discipline, helping you stay motivated with points and perks.",
    }
}

# -------------------
# Tools
# -------------------
@function_tool
def get_module_info(module_name: str) -> str:
    # Try RAG first
    modules = query_embeddings("modules") or {}
    module = modules.get(module_name.lower()) or CORE_CONTENT["modules"].get(module_name.lower())
    if not module:
        return DotFormatter.format_list([
            f"I only know these modules: {', '.join(CORE_CONTENT['modules'].keys())}",
            "Pick one!"
        ])
    return DotFormatter.format_list([
        f"{module_name.title()}",
        f"Purpose: {module['purpose']}",
        f"How to use: {module['how_to_use']}",
        f"Benefits: {module['benefits']}"
    ])

@function_tool
def get_overview() -> str:
    return DotFormatter.format_numbered(CORE_CONTENT["overview"])

@function_tool
def list_modules() -> str:
    modules = query_embeddings("modules") or CORE_CONTENT["modules"]
    return DotFormatter.format_list(["Modules available:"] + list(modules.keys()))

@function_tool
def answer_faq(question: str) -> str:
    faqs = CORE_CONTENT["faqs"]
    # Normalize question
    q = question.lower().strip()
    # Check RAG first
    rag_answer = query_embeddings(q)
    if rag_answer:
        return rag_answer
    # Fallback to core FAQ
    for k, v in faqs.items():
        if k in q:
            return DotFormatter.format_list([v])
    return DotFormatter.format_list([
        "I only answer questions about FundedFlow modules, features, and website content.",
        "Pick a module or ask about FundedFlow functionality."
    ])

# -------------------
# Privacy Guardrails
# -------------------
class PrivacyCheckOutput(BaseModel):
    reasoning: str
    has_personal_info: bool

privacy_guardrail_agent = Agent(
    name="Privacy Guardrail",
    instructions="Check if text contains personal info (emails, phones, etc.) → mark has_personal_info=True",
    output_type=PrivacyCheckOutput,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

@input_guardrail
async def privacy_input_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]):
    result = await Runner.run(privacy_guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.has_personal_info,
    )

@output_guardrail
async def privacy_output_guardrail(ctx: RunContextWrapper, agent: Agent, output: MessageOutput):
    result = await Runner.run(privacy_guardrail_agent, output.response, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.has_personal_info,
    )

# -------------------
# FundedFlow AI Agent
# -------------------
agent = Agent(
    name="FundedFlow General AI Agent",
    instructions=(
        "Hey! I’m your FundedFlow AI Assistant.\n"
        "Answer about FundedFlow modules, website features, and FAQs only.\n"
        "Never make up info—always stick to real content from the website.\n"
        "Use dot or numbered list style for clarity.\n"
        "If asked something outside FundedFlow, politely refuse."
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_module_info, get_overview, list_modules, answer_faq],
    input_guardrails=[privacy_input_guardrail],
    output_guardrails=[privacy_output_guardrail],
    output_type=MessageOutput,
)

# -------------------
# Runner
# -------------------
async def run_my_agent(query: str) -> str:
    try:
        result = await Runner.run(agent, query)
        response = result.final_output.response
        if not response.strip():
            return get_overview()
        return response
    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
        return ". **Sorry, I can’t share that information.**"
    except Exception:
        return get_overview()

# -------------------
# Terminal test
# -------------------
if __name__ == "__main__":
    async def main():
        print(". **Welcome! Ask me anything about FundedFlow modules or website features!**")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print(". **Bye!**")
                break
            response = await run_agent(query)
            print(response + "\n")

    asyncio.run(main())
