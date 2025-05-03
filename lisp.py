from collections import ChainMap
import operator as op

def standard_env():
    env = {
        '+': op.add,
        '-': op.sub,
        '*': op.mul,
        '/': op.truediv,
        '>': op.gt,
        '<': op.lt,
        '>=': op.ge,
        '<=': op.le,
        '=': op.eq,
    }
    return env

# Basic environment with arithmetic ops
global_env = standard_env()

def tokenize(s):
    """Split string into tokens."""
    return s.replace('(', ' ( ').replace(')', ' ) ').split()

def add_external(name, func):
    """Add external Python function to the environment."""
    global_env[name] = func

def parse(tokens):
    """Read tokens and create nested list structure."""
    if len(tokens) == 0:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(parse(tokens))
        tokens.pop(0)  # pop ')'
        return L
    elif token == ')':
        raise SyntaxError("Unexpected )")
    else:
        return atom(token)

def atom(token):
    """Convert token to int/float or leave as string."""
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return str(token)

def eval_(x, env):
    if isinstance(x, str):
        return env[x]
    elif not isinstance(x, list):
        return x
    elif x[0] == 'define':
        (_, var, expr) = x
        env[var] = eval_(expr, env)
    elif x[0] == 'lambda':
        (_, params, body) = x
        def func(*args):
            local_env = ChainMap(dict(zip(params, args)), env)
            return eval_(body, local_env)
        return func
    else:
        proc = eval_(x[0], env)
        args = [eval_(arg, env) for arg in x[1:]]
        return proc(*args)

add_external("match", meta_match)

# returns function 
def meta_match():
    # 

def meta_action():
    # 
    
# Example usage
# if __name__ == '__main__':
#     while True:
#         try:
#             code = input('lisp> ')
#             if code == 'exit':
#                 break
#             tokens = tokenize(code)
#             ast = parse(tokens)
#             result = eval_(ast)
#             if result is not None:
#                 print(result)
#         except Exception as e:
#             print(f"Error: {e}")
