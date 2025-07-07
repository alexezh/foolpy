

import re

def is_number(token):
    return token.isdigit() or (token.startswith('-') and token[1:].isdigit())

def is_variable(token):
    return len(token) == 1 and token.isalpha()


def parse_expression(expr: str) -> list[str]:
    """Standard tokenizer that processes character by character"""
    tokens = []
    i = 0
    expr = expr.replace(' ', '')  # Remove whitespace
    
    while i < len(expr):
        char = expr[i]
        
        # Handle numbers (including multi-digit)
        if char.isdigit():
            num_str = ''
            while i < len(expr) and expr[i].isdigit():
                num_str += expr[i]
                i += 1
            tokens.append(num_str)
            continue
        
        # Handle variables (single letters)
        elif char.isalpha():
            tokens.append(char)
            i += 1
            continue
        
        # Handle operators and parentheses
        elif char in '+-*/^()':
            tokens.append(char)
            i += 1
            continue
        
        else:
            # Skip unknown characters
            i += 1
    
    # Post-process to handle coefficient-variable combinations like "2x"
    processed = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens) and 
            is_number(tokens[i]) and 
            is_variable(tokens[i + 1])):
            # Check if they were adjacent in the original expression (no operator between)
            # This is a coefficient-variable pair like "2x"
            processed.extend([tokens[i], '*', tokens[i + 1]])
            i += 2
        else:
            processed.append(tokens[i])
            i += 1
    
    return processed
