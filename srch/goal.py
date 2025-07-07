from parser import is_number, is_variable
import re


def is_goal(tokens):
    # Goal: single token, number, variable, or coeffvar (e.g., 2x, x^2)
    # Also handle multiple tokens for powers and multiplication with any number of variables
    
    # Single token cases
    if len(tokens) == 1:
        token = tokens[0]
        return (is_number(token) or 
                is_variable(token) or 
                re.fullmatch(r'\d*[a-zA-Z](\^\d+)?', token))
    
    # For multiple tokens, check if it's a valid goal expression
    # Valid goals: number, variable, coefficient*variable(s), variable(s) with powers
    # Examples: [2], [x], [x, ^, 2], [2, *, x], [2, *, x, *, y], [x, *, y, ^, 3], etc.
    
    def is_valid_goal_sequence(tokens):
        """Check if tokens form a valid goal sequence"""
        if not tokens:
            return False
        
        # Check for simple fraction patterns like [x, /, y] or [2, /, 3]
        if len(tokens) == 3 and tokens[1] == '/':
            left, right = tokens[0], tokens[2]
            # number / number or variable / variable are valid goals
            if (is_number(left) and is_number(right)) or (is_variable(left) and is_variable(right)):
                return True
        
        # Check for addition patterns like [x, +, y] 
        if len(tokens) == 3 and tokens[1] == '+':
            left, right = tokens[0], tokens[2]
            # Different variables or numbers with addition can be valid goals
            if (is_number(left) and is_number(right)) or (is_variable(left) and is_variable(right) and left != right):
                return True
        
        # Check for subtraction patterns like [x, -, y] or [5, -, 3]
        if len(tokens) == 3 and tokens[1] == '-':
            left, right = tokens[0], tokens[2]
            # Different variables or numbers with subtraction can be valid goals
            if (is_number(left) and is_number(right)) or (is_variable(left) and is_variable(right) and left != right):
                return True
        
        i = 0
        # Optional leading coefficient or number
        if i < len(tokens) and is_number(tokens[i]):
            i += 1
            if i < len(tokens) and tokens[i] in ['*', '/', '-', '+']:
                i += 1
            else:
                # Just a number, valid goal
                return i == len(tokens)
        
        # Must have at least one variable
        has_variable = False
        
        while i < len(tokens):
            # Expect variable
            if i >= len(tokens) or not is_variable(tokens[i]):
                return False
            has_variable = True
            i += 1
            
            # Optional power
            if i + 1 < len(tokens) and tokens[i] == '^' and is_number(tokens[i + 1]):
                i += 2
            
            # Optional operation for next variable/number
            if i < len(tokens):
                if tokens[i] in ['*', '/', '-', '+']:
                    i += 1
                else:
                    # No more operations, should be end
                    return i == len(tokens)
        
        return has_variable
    
    return is_valid_goal_sequence(tokens)
