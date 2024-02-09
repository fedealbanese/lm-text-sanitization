import re
from asapp.commons import Unigram

class Redactor:
    """
    Redacts terms in a document that don't meet a privacy threshold.

    Attributes:
    context(str): the sanetized previous utterance.
    """
    _alnum_regex = re.compile(r'\w')

    def __init__(self, model, p = 0.0):
        """
        Constructor for a Redactor.

        Args:
        model: language model to query for word frequencies.
        p(float): privacy threshold to use as cutoff.
        """
        self.model = model
        self.p_threshold = p
        self.context = ""

    def redact_term(self, term: str) -> str:
        """
        Returns a redaction token for the given term.

        Args:
        term(str): term to redact.

        Returns: 
        str: replacement value for the term.
        """
        return '*'
    
    def is_safe(self, token: Unigram, document: str) -> bool:
        """
        Returns whether a token is safe based on its probability and its similarity with alternative words.

        Args:
        token(Unigram): token to process.
        document(str): document to redact.

        Returns: 
        bool: rwhether a token is safe.
        """
        return token.p >= self.p_threshold

    def redact(self, document: str, tokens: list = None) -> str:
        """
        Redacts a given document (replacing term not meeting the p-threshold
        with a redaction token).

        Args:
        document(str): document to redact.
        tokens(list, optional): list of token representation of words in document. If None, it calculates the 
        doc representation.

        Returns: 
        str: redacted document.
        """
        previous = 0
        redacted_segments = []
        if tokens is None:
            tokens = self.model.get_doc_repr(document, self.context)
        for token in tokens:
            start = token.start
            end = token.end
            if previous < start:
                redacted_segments.append(document[previous:start])
            previous = end
            original = document[start:end]
            if self.is_safe(token, document):
                redacted_segments.append(original)
            else:
                redacted_segments.append(self.redact_term(original))
        redacted_segments.append(document[previous:])
        return ''.join(redacted_segments)

    def redact_document_with_redactions(self, document:str, redactions:list) -> str:
        """
        Redacts a given document with a list of the redactions.

        Args:
        document(str): document to redact.
        redactions(list): list of words to redact.

        Returns:
        redacted_document(str): the original document without the words in redactions.
        """
        redaction_prefix = 10 #"PFILTERED:" in "PFILTERED:word"
        redacted_document = ""
        index = 0
        for redacted_word in redactions:
            redacted_document += document[index:redacted_word[1]] 
            redacted_document += self.redact_term(redacted_word[0][redaction_prefix:])
            index = redacted_word[1] + redacted_word[2]
        redacted_document += document[index:] 
        return redacted_document

    def find_redactions(self, document):
        """
        Finds substrings to remove to sanitize a given document.

        Args:
        document(str): document to redact.

        Returns:
        tuple: (safe, redactions) where:
        - safe is a set of safe terms for each p threshold; and
        - redactions is a list of (indexed) redactions for each p threshold.
        """
        safe = set()
        redactions = []
        tokens = self.model.get_doc_repr(document, self.context)
        for token in tokens:
          if self.is_safe(token, document):
              safe.add(token.term)
              continue
          start = token.start
          end = token.end
          term = document[start:end]
          indexes = [
              match.start()
              for match in Redactor._alnum_regex.finditer(term)
          ]
          first = indexes[0]
          last = indexes[-1]
          if first >= 0 and last >= 0:
              tag = f"PFILTERED:{term[first:last+1]}"
              redactions.append([tag, start + first, last - first + 1])
        self.context = self.redact_document_with_redactions(document, redactions)
        return safe, redactions
