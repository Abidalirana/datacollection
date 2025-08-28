# ml_models/recovery_agent.py
import random

def suggest_recovery_plan(trade_outcomes: list):
    """
    Suggest a recovery plan based on recent outcomes
    """
    losses = trade_outcomes.count(0)
    wins = trade_outcomes.count(1)

    if losses > wins:
        plans = [
            "Take a break and review your journal",
            "Switch to smaller lot sizes",
            "Pause trading after 2 consecutive losses"
        ]
    else:
        plans = [
            "Gradually scale risk with winning streak",
            "Document successful strategies",
            "Review confidence score trends"
        ]
    return random.choice(plans)


if __name__ == "__main__":
    print(suggest_recovery_plan([0, 0, 1, 0, 1]))
