

import re
from token import Token, TRef

def is_number(token):
    """Check if token (string, Token, or TRef) represents a number"""
    text = str(token)
    return text.isdigit() or (text.startswith('-') and text[1:].isdigit())

def is_variable(token):
    """Check if token (string, Token, or TRef) represents a variable"""
    text = str(token)
    return len(text) == 1 and text.isalpha()


def parse_expression(expr: str) -> list[TRef]:
    """Standard tokenizer that processes character by character and returns TRef objects"""
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
            tokens.append(Token(num_str))
            continue
        
        # Handle variables (single letters)
        elif char.isalpha():
            tokens.append(Token(char))
            i += 1
            continue
        
        # Handle operators and parentheses
        elif char in '+-*/^()':
            tokens.append(Token(char))
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
            processed.extend([
                TRef.from_token(tokens[i]), 
                TRef.from_text('*'), 
                TRef.from_token(tokens[i + 1])
            ])
            i += 2
        else:
            processed.append(TRef.from_token(tokens[i]))
            i += 1
    
    return processed
