import heapq
import re

def parse_expression(expr):
    # Tokenize numbers, variables (single letter), operators, and parentheses
    tokens = re.findall(r'\d+|[a-zA-Z]|[()+\-*/]', expr.replace(' ', ''))
    return tokens

def is_number(token):
    return token.isdigit()

def is_variable(token):
    return len(token) == 1 and token.isalpha()

# Example actions
def apply_mul(tokens):
    # Find and evaluate multiplication for numbers only
    for i in range(1, len(tokens)-1):
        if tokens[i] == '*':
            left, right = tokens[i-1], tokens[i+1]
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)*int(right))] + tokens[i+2:]
                return new_tokens
    return None

def apply_sum(tokens):
    # Find and evaluate addition for numbers only
    for i in range(1, len(tokens)-1):
        if tokens[i] == '+':
            left, right = tokens[i-1], tokens[i+1]
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)+int(right))] + tokens[i+2:]
                return new_tokens
    return None

def apply_parenthesis(tokens):
    # Evaluate expressions in parentheses if they reduce to a single number or variable
    for i in range(len(tokens)):
        if tokens[i] == '(':
            for j in range(i+2, len(tokens)):
                if tokens[j] == ')':
                    subexpr = tokens[i+1:j]
                    if len(subexpr) == 1 and (is_number(subexpr[0]) or is_variable(subexpr[0])):
                        new_tokens = tokens[:i] + subexpr + tokens[j+1:]
                        return new_tokens
    return None

ACTIONS = [apply_mul, apply_sum, apply_parenthesis]

def is_goal(tokens):
    # Goal: single token, number or variable
    return len(tokens) == 1 and (is_number(tokens[0]) or is_variable(tokens[0]))

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
    # expr_str = "(2 + x) * 4"
    expr_str = "2 + 5"

    expr = parse_expression(expr_str)
    result = a_star_search(expr)
    for step in result:
        print(step)