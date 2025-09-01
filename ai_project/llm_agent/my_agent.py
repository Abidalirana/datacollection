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

# Disable tracing if not needed
set_tracing_disabled(disabled=True)

# -------------------
# Formatter for dot & number style responses
# -------------------
class DotFormatter:
    @staticmethod
    def format_list(items: list[str]) -> str:
        return "\n".join([f". **{item}**" for item in items])

    @staticmethod
    def format_numbered(items: list[str]) -> str:
        return "\n".join([f"{i+1}. **{item}**" for i, item in enumerate(items)])

    @staticmethod
    def format_module(name: str, purpose: str, how_to_use: str, benefits: str) -> str:
        return DotFormatter.format_list([
            f"{name.title()}",
            f"Purpose: {purpose}",
            f"How to use: {how_to_use}",
            f"Benefits: {benefits}"
        ])


# -------------------
# Output Schema
# -------------------
class MessageOutput(BaseModel):
    response: str


# -------------------
# Tools (FundedFlow only)
# -------------------
@function_tool
def get_fundedflow_module_info(module_name: str) -> str:
    modules = query_embeddings("modules")  # dict of modules
    module = modules.get(module_name.lower())
    if not module:
        return DotFormatter.format_list([
            f"I only know these FundedFlow modules: {', '.join(modules.keys())}",
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
    return DotFormatter.format_numbered([
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
    modules = query_embeddings("modules")
    return DotFormatter.format_list(
        ["Modules available:"] + list(modules.keys())
    )


# -------------------
# Guardrails (Privacy only)
# -------------------
class PrivacyCheckOutput(BaseModel):
    reasoning: str
    has_personal_info: bool


# Sub-agent: Privacy Guardrail
privacy_guardrail_agent = Agent(
    name="Privacy Guardrail",
    instructions=(
        "Check if the text includes personal information like names, emails, "
        "phone numbers, addresses, or sensitive user data. "
        "If yes → mark has_personal_info=True."
    ),
    output_type=PrivacyCheckOutput,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


# Input guardrail: Privacy
@input_guardrail
async def privacy_input_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(privacy_guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.has_personal_info,
    )


# Output guardrail: Privacy
@output_guardrail
async def privacy_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(privacy_guardrail_agent, output.response, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.has_personal_info,
    )


# -------------------
# Floki AI Agent (FundedFlow only)
# -------------------
agent = Agent(
    name="FundedFlow AI Agent",
    instructions=(
        "Hey, I’m FundedFlow! I’m your FundedFlow AI Assistant.\n"
        "Core personality:\n"
        ". Super friendly, short, and encouraging\n"
        ". Tie answers back to FundedFlow modules\n"
        ". Always format responses in numbered or dotted bold list style\n"
        ". Only answer about FundedFlow.app (trading and modules)\n"
        ". Never share personal data, emails, phone numbers, or unrelated content\n"
        ". If asked something outside FundedFlow.app → politely refuse"
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_fundedflow_module_info, get_fundedflow_overview, list_fundedflow_modules],
    input_guardrails=[privacy_input_guardrail],
    output_guardrails=[privacy_output_guardrail],
    output_type=MessageOutput,
)


# -------------------
# Runner with Fallback
# -------------------
async def run_my_agent(user_query: str) -> str:
    try:
        # Try RAG pipeline
        results = query_embeddings(user_query)  # optional context
        result = await Runner.run(agent, user_query)
        return result.final_output.response

    except InputGuardrailTripwireTriggered:
        return ". **Sorry, I can’t help with that request.**"

    except OutputGuardrailTripwireTriggered:
        return ". **Sorry, I can’t share that information.**"

    except Exception as e:
        # Fallback if RAG or agent fails
        return DotFormatter.format_list([
            "Oops, something went wrong with my data lookup 🤖",
            "But don’t worry — I’m still here to help!",
            "You can explore FundedFlow modules like Journals, Risk, Simulator, and more.",
            "Which one would you like to dive into?"
        ])


# -------------------
# Terminal test
# -------------------
if __name__ == "__main__":
    async def main():
        print(". **Welcome! I’m FundedFlow AI Agent, your FundedFlow AI Assistant**")

        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print(". **Bye!**")
                break
            response = await run_my_agent(query)
            print(response + "\n")



    asyncio.run(main())
