# Actions
from parser import is_number, is_variable


def apply_mul(tokens):
    # Handle x * x = x^2, 2 * x = 2x, x * 2 = 2x, 2x * 3 = 6x, x^2 * x^3 = x^5, etc.
    
    def get_variable_power(tokens, start_idx):
        """Get variable and power from tokens starting at start_idx. Returns (variable, power, end_idx)"""
        if start_idx >= len(tokens):
            return None, None, start_idx
        
        if is_variable(tokens[start_idx]):
            var = tokens[start_idx]
            # Check if next tokens are ^ and number
            if (start_idx + 2 < len(tokens) and 
                tokens[start_idx + 1] == '^' and 
                is_number(tokens[start_idx + 2])):
                power = int(tokens[start_idx + 2])
                return var, power, start_idx + 3
            else:
                return var, 1, start_idx + 1
        return None, None, start_idx
    
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
            
            # Check for variable * variable patterns (including powers)
            left_var, left_power, left_end = get_variable_power(tokens, i-1)
            right_var, right_power, right_end = get_variable_power(tokens, i+1)
            
            if left_var and right_var:
                if left_var == right_var:
                    # Same variable: x * x = x^2, x^2 * x^3 = x^5
                    total_power = left_power + right_power
                    if total_power == 1:
                        # x * x = x (when both have power 1)
                        new_tokens = tokens[:i-1] + [left_var] + tokens[right_end:]
                    else:
                        # x * x^2 = x^3 or x^2 * x^3 = x^5 (produce separate tokens)
                        new_tokens = tokens[:i-1] + [left_var, '^', str(total_power)] + tokens[right_end:]
                    return new_tokens
                else:
                    # Different variables: x * y = x*y (already in correct form, no change needed)
                    # This is already a valid goal state for multiple variables
                    return None
            
            # number * number
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)*int(right))] + tokens[i+2:]
                return new_tokens
    return None

def apply_sum(tokens):
    # Handle x + x = 2*x, 2x + 3x = 5x, 2 + 3 = 5, x^2 + x^2 = 2*x^2, etc.
    
    def get_variable_power(tokens, start_idx):
        """Get variable and power from tokens starting at start_idx. Returns (variable, power, end_idx)"""
        if start_idx >= len(tokens):
            return None, None, start_idx
        
        if is_variable(tokens[start_idx]):
            var = tokens[start_idx]
            # Check if next tokens are ^ and number
            if (start_idx + 2 < len(tokens) and 
                tokens[start_idx + 1] == '^' and 
                is_number(tokens[start_idx + 2])):
                power = int(tokens[start_idx + 2])
                return var, power, start_idx + 3
            else:
                return var, 1, start_idx + 1
        return None, None, start_idx
    
    for i in range(1, len(tokens)-1):
        if tokens[i] == '+':
            left, right = tokens[i-1], tokens[i+1]
            # number + number
            if is_number(left) and is_number(right):
                new_tokens = tokens[:i-1] + [str(int(left)+int(right))] + tokens[i+2:]
                return new_tokens
            
            # Handle pure variables with powers (produce separate tokens)
            left_var, left_power, left_end = get_variable_power(tokens, i-1)
            right_var, right_power, right_end = get_variable_power(tokens, i+1)
            
            if left_var and right_var and left_var == right_var and left_power == right_power:
                if left_power == 1:
                    # x + x = 2*x (produce separate multiplication tokens)
                    new_tokens = tokens[:i-1] + ['2', '*', left_var] + tokens[right_end:]
                else:
                    # x^2 + x^2 = 2*x^2 (produce separate multiplication tokens)
                    new_tokens = tokens[:i-1] + ['2', '*', left_var, '^', str(left_power)] + tokens[right_end:]
                return new_tokens
            
            # variable + number or number + variable (no simplification, just reorder)
            if is_variable(left) and is_number(right):
                new_tokens = tokens[:i-1] + [right + '+' + left] + tokens[i+2:]
                return new_tokens
            if is_number(left) and is_variable(right):
                new_tokens = tokens[:i-1] + [left + '+' + right] + tokens[i+2:]
                return new_tokens
    return None

def apply_sub(tokens):
    # Handle x - x = 0, 2x - 3x = -x, 5 - 3 = 2, x^2 - x^2 = 0, etc.
    
    def get_variable_power(tokens, start_idx):
        """Get variable and power from tokens starting at start_idx. Returns (variable, power, end_idx)"""
        if start_idx >= len(tokens):
            return None, None, start_idx
        
        if is_variable(tokens[start_idx]):
            var = tokens[start_idx]
            # Check if next tokens are ^ and number
            if (start_idx + 2 < len(tokens) and 
                tokens[start_idx + 1] == '^' and 
                is_number(tokens[start_idx + 2])):
                power = int(tokens[start_idx + 2])
                return var, power, start_idx + 3
            else:
                return var, 1, start_idx + 1
        return None, None, start_idx
    
    for i in range(1, len(tokens)-1):
        if tokens[i] == '-':
            left, right = tokens[i-1], tokens[i+1]
            # number - number
            if is_number(left) and is_number(right):
                result = int(left) - int(right)
                new_tokens = tokens[:i-1] + [str(result)] + tokens[i+2:]
                return new_tokens
            
            # Handle pure variables with powers (produce separate tokens)
            left_var, left_power, left_end = get_variable_power(tokens, i-1)
            right_var, right_power, right_end = get_variable_power(tokens, i+1)
            
            if left_var and right_var and left_var == right_var and left_power == right_power:
                # x - x = 0, x^2 - x^2 = 0
                new_tokens = tokens[:i-1] + ['0'] + tokens[right_end:]
                return new_tokens
    return None

def apply_div(tokens):
    # Handle x / x = 1, 6 / 2 = 3, x^3 / x^2 = x, etc.
    
    def get_variable_power(tokens, start_idx):
        """Get variable and power from tokens starting at start_idx. Returns (variable, power, end_idx)"""
        if start_idx >= len(tokens):
            return None, None, start_idx
        
        if is_variable(tokens[start_idx]):
            var = tokens[start_idx]
            # Check if next tokens are ^ and number
            if (start_idx + 2 < len(tokens) and 
                tokens[start_idx + 1] == '^' and 
                is_number(tokens[start_idx + 2])):
                power = int(tokens[start_idx + 2])
                return var, power, start_idx + 3
            else:
                return var, 1, start_idx + 1
        return None, None, start_idx
    
    for i in range(1, len(tokens)-1):
        if tokens[i] == '/':
            left, right = tokens[i-1], tokens[i+1]
            # number / number
            if is_number(left) and is_number(right):
                if int(right) != 0 and int(left) % int(right) == 0:
                    result = int(left) // int(right)
                    new_tokens = tokens[:i-1] + [str(result)] + tokens[i+2:]
                    return new_tokens
            
            # Check for variable / variable patterns (including powers)
            left_var, left_power, left_end = get_variable_power(tokens, i-1)
            right_var, right_power, right_end = get_variable_power(tokens, i+1)
            
            if left_var and right_var and left_var == right_var:
                power_diff = left_power - right_power
                if power_diff == 0:
                    # x / x = 1, x^2 / x^2 = 1
                    new_tokens = tokens[:i-1] + ['1'] + tokens[right_end:]
                elif power_diff == 1:
                    # x^3 / x^2 = x
                    new_tokens = tokens[:i-1] + [left_var] + tokens[right_end:]
                else:
                    # x^4 / x^2 = x^2 (produce separate tokens)
                    new_tokens = tokens[:i-1] + [left_var, '^', str(power_diff)] + tokens[right_end:]
                return new_tokens
    return None

def apply_cancel(tokens):
    # Cancel out same numbers/variables with opposite signs: x + 3 - 3 = x, 5 + x - x = 5
    # But avoid canceling terms that are part of multiplication operations
    
    def is_part_of_multiplication(tokens, index):
        """Check if a term at given index is part of a multiplication"""
        # Check if previous token is *
        if index > 0 and tokens[index - 1] == '*':
            return True
        # Check if next token is *
        if index + 1 < len(tokens) and tokens[index + 1] == '*':
            return True
        return False
    
    # Look for addition and subtraction of the same term to cancel out
    for i in range(len(tokens)):
        if tokens[i] == '+' and i + 1 < len(tokens):
            # Found a + term, check if it's not part of multiplication
            add_term = tokens[i + 1]
            if is_part_of_multiplication(tokens, i + 1):
                continue
            
            # Search for corresponding subtraction
            for j in range(i + 2, len(tokens)):
                if (tokens[j] == '-' and 
                    j + 1 < len(tokens) and 
                    tokens[j + 1] == add_term and
                    not is_part_of_multiplication(tokens, j + 1)):
                    
                    # Found matching + term and - term, cancel them out
                    # Remove the second occurrence first (higher index)
                    new_tokens = tokens[:j] + tokens[j+2:]
                    # Then remove the first occurrence (lower index)  
                    new_tokens = new_tokens[:i] + new_tokens[i+2:]
                    return new_tokens
    
    # Also look for subtraction followed by addition of the same term
    for i in range(len(tokens)):
        if tokens[i] == '-' and i + 1 < len(tokens):
            # Found a - term, check if it's not part of multiplication
            sub_term = tokens[i + 1]
            if is_part_of_multiplication(tokens, i + 1):
                continue
            
            # Search for corresponding addition
            for j in range(i + 2, len(tokens)):
                if (tokens[j] == '+' and 
                    j + 1 < len(tokens) and 
                    tokens[j + 1] == sub_term and
                    not is_part_of_multiplication(tokens, j + 1)):
                    
                    # Found matching - term and + term, cancel them out
                    # Remove the second occurrence first (higher index)
                    new_tokens = tokens[:j] + tokens[j+2:]
                    # Then remove the first occurrence (lower index)
                    new_tokens = new_tokens[:i] + new_tokens[i+2:]
                    return new_tokens
    
    # Handle case where first term is implicitly positive
    if len(tokens) >= 3 and not tokens[0] in ['+', '-']:
        first_term = tokens[0]
        if not is_part_of_multiplication(tokens, 0):
            # Look for the same term later with a minus sign
            for i in range(1, len(tokens)):
                if (i + 1 < len(tokens) and 
                    tokens[i] == '-' and 
                    tokens[i + 1] == first_term and
                    not is_part_of_multiplication(tokens, i + 1)):
                    # Cancel out: remove first term and the "- term" pair
                    new_tokens = tokens[1:i] + tokens[i+2:]
                    return new_tokens
    
    return None

def apply_sub_to_add(tokens):
    # Convert subtraction to addition with negative numbers: x - 3 becomes x + (-3)
    for i in range(1, len(tokens)):
        if tokens[i] == '-' and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if is_number(next_token):
                # Convert "- 3" to "+ -3"
                new_tokens = tokens[:i] + ['+', '-' + next_token] + tokens[i+2:]
                return new_tokens
    return None

def apply_cleanup(tokens):
    # Clean up leading operators and other formatting issues
    if not tokens:
        return None
    
    # Remove leading + operator
    if tokens[0] == '+':
        return tokens[1:]
    
    # Remove leading - operator and negate the first term if it's a number
    if tokens[0] == '-' and len(tokens) > 1:
        if is_number(tokens[1]):
            # Convert -3 to negative number representation
            return ['-' + tokens[1]] + tokens[2:]
        else:
            # For variables, keep the minus but this might need special handling
            return tokens
    
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
