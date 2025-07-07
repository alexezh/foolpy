"""
Token and reference classes for tracking tokens during computation
"""

class Token:
    """A token with content and unique ID"""
    _next_id = 0
    
    def __init__(self, text: str):
        self.text = text
        self.id = Token._next_id
        Token._next_id += 1
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"Token('{self.text}', id={self.id})"
    
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.text == other.text
        elif isinstance(other, str):
            return self.text == other
        return False
    
    def __hash__(self):
        return hash((self.text, self.id))


class TRef:
    """A reference to a token with additional text context"""
    
    def __init__(self, token: Token, text: str = None):
        self.token = token
        self.text = text if text is not None else token.text
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"TRef({self.token!r}, text='{self.text}')"
    
    def __eq__(self, other):
        if isinstance(other, TRef):
            return self.text == other.text
        elif isinstance(other, str):
            return self.text == other
        elif isinstance(other, Token):
            return self.text == other.text
        return False
    
    def __hash__(self):
        return hash(self.text)
    
    @property
    def token_id(self):
        return self.token.id
    
    @classmethod
    def from_token(cls, token: Token):
        """Create a TRef directly from a Token"""
        return cls(token, token.text)
    
    @classmethod
    def from_text(cls, text: str, source_token: Token = None):
        """Create a TRef with new text, optionally linked to a source token"""
        if source_token is None:
            source_token = Token(text)
        return cls(source_token, text)