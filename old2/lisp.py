from collections import ChainMap
import operator as op
from typing import Any, List
from collections import defaultdict

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
      if x.startswith('"') and x.endswith('"'):
        return x
      else:
        return env[x]
    elif not isinstance(x, list):
        return x
    elif x[0] == 'define':
        (_, var, expr) = x
        env[var] = eval_(expr, env)
    elif x[0] == 'lambda':
        (_, params, body) = x
        def func(env, *args):
            local_env = ChainMap(dict(zip(params, args)), env)
            return eval_(body, local_env)
        return func
    else:
        proc = eval_(x[0], env)
        args = [eval_(arg, env) for arg in x[1:]]
        return proc(env, args)

def exec_(code):
  tokens = tokenize(code)
  ast = parse(tokens)
  eval_(ast, global_env)

def if_(env, args: List[Any]):
  if len(args) == 3:
      _, test, conseq = args
      test_result = eval_(test, env)
      if test_result:
          return eval_(conseq, env)
      else:
          return None  # no 'else', return None
  elif len(args) == 4:
      _, test, conseq, alt = args
      test_result = eval_(test, env)
      if test_result:
          return eval_(conseq, env)
      else:
          return eval_(alt, env)

# returns function 
def mmatch(env, args: List[Any]):
  # now we want to select match function
  # and we have multiple options, so we want to run all of them forking output
  # to do this, we can change if(x, t, e) monad ? 
  #
  # if should return enumerator of options, function which can be called many time
  # and if will propagate to for-each which will fork execution
  return True

def maction(env, args: List[Any]):
  print("Hello World")

def for_each(env, args: List[Any]):
  func, lst = args
  for arg in lst(env):
      func(env, arg)

def objects(env):
  return [1, 2]

# run criteris function
def msolve(env, args):
    func, crit = args
    return crit(env)

def more(env, args):
    a1, a2 = args
    return a1(env) > a2(env)

def reduce(env, args):
    lst, func = args
    for elem in lst:
        func(env, elem)

def group(env, args):
    lst, func = args
    result = defaultdict(list)

    for item in lst(env):
        key = func(env, item)
        result[key].append(item)
    return dict(result)    

def isnumber(env, args):
    val = eval_(args[0], env)
    return isinstance(val, (int, float))

def len_(env, args):
    return len(args[0])

add_external("if", if_)
add_external("for-each", for_each)
add_external("mmatch", mmatch)
add_external("maction", maction)
add_external("msolve", msolve)
add_external("more", more)
add_external("group", group)
add_external("reduce", reduce)
add_external("isnumber", isnumber)
add_external("len", len_)

add_external("objects", objects)
# add_external("count", meta_action)

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
