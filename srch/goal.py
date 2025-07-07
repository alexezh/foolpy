from srch.parser import is_number, is_variable


def is_goal(tokens):
    # Goal: single token, number, variable, or coeffvar (e.g., 2x, x^2)
    return len(tokens) == 1 and (is_number(tokens[0]) or is_variable(tokens[0]) or re.fullmatch(r'\d*[a-zA-Z]\^2', tokens[0]))
