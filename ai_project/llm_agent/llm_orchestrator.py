from .my_agent import run_my_agent

async def orchestrate(user_query: str, ml_output: dict = None) -> str:
    """
    Combines ML outputs with LLM explanations.
    ml_output: optional dict with ML results like tilt prediction, risk score, etc.
    """
    prompt = user_query
    if ml_output:
        # Prepend ML insights to user query
        insights = "\n".join([f"{k}: {v}" for k, v in ml_output.items()])
        prompt = f"{insights}\n\nUser question: {user_query}"
    
    response = await run_my_agent(prompt)
    return response

# Example usage
if __name__ == "__main__":
    import asyncio
    ml_dummy = {"risk_flag": "High", "suggested_action": "Take a break"}
    res = asyncio.run(orchestrate("What should I do next?", ml_dummy))
    print(res)
