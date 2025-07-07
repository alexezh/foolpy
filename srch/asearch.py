import heapq
import re

from actions import apply_mul, apply_parenthesis, apply_sum, apply_sub, apply_div, apply_cancel, apply_cleanup, apply_sub_to_add
from goal import is_goal
from parser import is_number, is_variable, parse_expression
from weight import heuristic


ACTIONS = [apply_mul, apply_sum, apply_sub, apply_div, apply_cancel, apply_cleanup, apply_sub_to_add, apply_parenthesis]

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
    # expr_str = "-4 + 3 * 4 + x + y - 3"
    expr_str = "4 + 3 * 4"
    expr = parse_expression(expr_str)
    result = a_star_search(expr)
    for step in result:
        print(step)