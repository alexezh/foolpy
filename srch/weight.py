from goal import is_goal
from token import Token, TRef


def heuristic(tokens):
    # Heuristic: prefer fewer tokens, but account for goal structure
    # Goal structures with multiple variables are acceptable
    if is_goal(tokens):
        return 0
    # Simple heuristic: number of tokens left
    return len(tokens)
