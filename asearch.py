import heapq
import re

def parse_expression(expr):
    # Tokenize numbers, variables (single letter), operators, and parentheses
    tokens = re.findall(r'\d+|[a-zA-Z]+|[()+\-*/^]', expr.replace(' ', ''))
    return tokens

def is_number(token):
    return token.isdigit()

def is_variable(token):
    return len(token) == 1 and token.isalpha()

def is_coeff_var(token):
    # Matches forms like '2x', '3y', or just 'x'
    return re.fullmatch(r'\d*[a-zA-Z]', token) is not None

def split_coeff_var(token):
    # Splits '2x' -> (2, 'x'), 'x' -> (1, 'x')
    m = re.fullmatch(r'(\d*)([a-zA-Z])', token)
    if m:
        coeff = int(m.group(1)) if m.group(1) else 1
        var = m.group(2)
        return coeff, var
    return None, None

# Actions
def apply_mul(tokens):
    # Handle x * x = x^2, 2 * x = 2x, x * 2 = 2x, 2x * 3 = 6x, etc.
    for i in range(1, len(tokens)-1):
        if tokens[i] == '*':
            left, right = tokens[i-1], tokens[i+1]
            # number * variable or variable * number
            if is_number(left) and is_variable(right):
                new_tokens = tokens[:i-1] + [str(int(left)) + right] + tokens[i+2:]
                return new_tokens
            if is_variable(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(right)) + left] + tokens[i+2:]
                return new_tokens
            # variable * variable
            if is_variable(left) and is_variable(right) and left == right:
                new_tokens = tokens[:i-1] + [left + '^2'] + tokens[i+2:]
                return new_tokens
            # coeffvar * number or number * coeffvar
            if is_coeff_var(left) and is_number(right):
                coeff, var = split_coeff_var(left)
                new_tokens = tokens[:i-1] + [str(coeff * int(right)) + var] + tokens[i+2:]
                return new_tokens
            if is_number(left) and is_coeff_var(right):
                coeff, var = split_coeff_var(right)
                new_tokens = tokens[:i-1] + [str(coeff * int(left)) + var] + tokens[i+2:]
                return new_tokens
            # coeffvar * coeffvar (same variable)
            if is_coeff_var(left) and is_coeff_var(right):
                cl, vl = split_coeff_var(left)
                cr, vr = split_coeff_var(right)
                if vl == vr:
                    new_tokens = tokens[:i-1] + [str(cl * cr) + vl + '^2'] + tokens[i+2:]
                    return new_tokens
            # number * number
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)*int(right))] + tokens[i+2:]
                return new_tokens
    return None

def apply_sum(tokens):
    # Handle x + x = 2x, 2x + 3x = 5x, 2 + 3 = 5, etc.
    for i in range(1, len(tokens)-1):
        if tokens[i] == '+':
            left, right = tokens[i-1], tokens[i+1]
            # number + number
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)+int(right))] + tokens[i+2:]
                return new_tokens
            # variable + variable
            if is_variable(left) and is_variable(right) and left == right:
                new_tokens = tokens[:i-1] + ['2' + left] + tokens[i+2:]
                return new_tokens
            # coeffvar + coeffvar (same variable)
            if is_coeff_var(left) and is_coeff_var(right):
                cl, vl = split_coeff_var(left)
                cr, vr = split_coeff_var(right)
                if vl == vr:
                    new_tokens = tokens[:i-1] + [str(cl + cr) + vl] + tokens[i+2:]
                    return new_tokens
            # variable + coeffvar or coeffvar + variable (same variable)
            if is_variable(left) and is_coeff_var(right):
                cr, vr = split_coeff_var(right)
                if left == vr:
                    new_tokens = tokens[:i-1] + [str(1 + cr) + vr] + tokens[i+2:]
                    return new_tokens
            if is_coeff_var(left) and is_variable(right):
                cl, vl = split_coeff_var(left)
                if right == vl:
                    new_tokens = tokens[:i-1] + [str(cl + 1) + vl] + tokens[i+2:]
                    return new_tokens
    return None

def apply_parenthesis(tokens):
    # Evaluate expressions in parentheses if they reduce to a single token
    for i in range(len(tokens)):
        if tokens[i] == '(':
            for j in range(i+2, len(tokens)):
                if tokens[j] == ')':
                    subexpr = tokens[i+1:j]
                    if len(subexpr) == 1:
                        new_tokens = tokens[:i] + subexpr + tokens[j+1:]
                        return new_tokens
    return None

ACTIONS = [apply_mul, apply_sum, apply_parenthesis]

def is_goal(tokens):
    # Goal: single token, number, variable, or coeffvar (e.g., 2x, x^2)
    return len(tokens) == 1 and (is_number(tokens[0]) or is_variable(tokens[0]) or is_coeff_var(tokens[0]) or re.fullmatch(r'\d*[a-zA-Z]\^2', tokens[0]))

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