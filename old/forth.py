class ForthInterpreter:
    def __init__(self):
        self.stack = []
        self.words = {
            '+': self._add,
            '-': self._sub,
            '*': self._mul,
            '/': self._div,
            'dup': self._dup,
            'drop': self._drop,
            'swap': self._swap,
            'over': self._over,
            '.s': self._print_stack,
        }

    def _add(self): self._binary_op(lambda a, b: a + b)
    def _sub(self): self._binary_op(lambda a, b: a - b)
    def _mul(self): self._binary_op(lambda a, b: a * b)
    def _div(self): self._binary_op(lambda a, b: a / b)

    def _dup(self): self.stack.append(self.stack[-1])
    def _drop(self): self.stack.pop()
    def _swap(self): self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
    def _over(self): self.stack.append(self.stack[-2])

    def _print_stack(self):
        print("Stack:", self.stack)

    def _binary_op(self, func):
        b = self.stack.pop()
        a = self.stack.pop()
        self.stack.append(func(a, b))

    def execute(self, code):
        tokens = code.strip().split()
        for token in tokens:
            if token in self.words:
                self.words[token]()
            else:
                try:
                    num = int(token)
                except ValueError:
                    try:
                        num = float(token)
                    except ValueError:
                        raise ValueError(f"Unknown word: {token}")
                self.stack.append(num)

    def register_word(self, name, func):
        """Register an external method as a new word."""
        def wrapped():
            # Pass the interpreter itself in case external wants access to stack
            func(self)
        self.words[name] = wrapped
