from sympy import re


def is_number(token):
    return token.isdigit()

def is_variable(token):
    return len(token) == 1 and token.isalpha()

def is_coeff_var(token):
    return re.fullmatch(r'\d+[a-zA-Z]', token) is not None

def parse_expression(expr: str) -> list[str]:
    tokens = re.findall(r'\d+|[a-zA-Z]+|[()+\-*/^]', expr.replace(' ', ''))
    processed = []
    for token in tokens:
        if is_coeff_var(token):
            num = ''.join(filter(str.isdigit, token))
            var = ''.join(filter(str.isalpha, token))
            processed.extend([num, '*', var])
        else:
            processed.append(token)
