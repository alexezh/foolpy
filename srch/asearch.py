import heapq
import re

from srch.actions import apply_mul, apply_parenthesis, apply_sum
from srch.goal import is_goal
from srch.parser import is_number, is_variable, parse_expression


ACTIONS = [apply_mul, apply_sum, apply_parenthesis]

def heuristic(tokens):
    # Simple heuristic: number of tokens left
    return len(tokens)

def a_star_search(start_tokens):
    heap = []
    heapq.heappush(heap, (heuristic(start_tokens), 0, start_tokens, []))
    visited = set()
    while heap:
        est_total, cost, tokens, path = heapq.heappop(heap)
        state_key = tuple(tokens)
        if state_key in visited:
            continue
        visited.add(state_key)
        if is_goal(tokens):
            return path + [tokens]
        for action in ACTIONS:
            new_tokens = action(tokens)
            if new_tokens and tuple(new_tokens) not in visited:
                heapq.heappush(heap, (cost+1+heuristic(new_tokens), cost+1, new_tokens, path + [tokens]))
    return None

# Example usage:
if __name__ == "__main__":
    expr_str = "x + y + x"
    expr = parse_expression(expr_str)
    result = a_star_search(expr)
    for step in result:
        print(step)