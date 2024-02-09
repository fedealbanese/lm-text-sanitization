import re


class Unigram:
    """
    This class represents a processed token.
    """

    _token_regex = re.compile(
        r"({[^}]+}|" +                # Redacted token (to skip)
        r"((\w|\d+)[^{\w'‘’`]+)+|" +   # Acronyms and number sequences
        r"(\d+[^\s\w]*)+\w+|" +       # Alphanumeric with digit prefix
        r"\w+(['‘’`]\w+)*)",          # Alphanumeric with apostrophes
        re.IGNORECASE
    )

    _drop_regex = re.compile(r"\W")
    _prefix_regex = re.compile(r"^\W*")
    _suffix_regex = re.compile(r"\W*$")

    @staticmethod
    def tokenize(document) -> iter:
        """
        Returns an iterator of tokens from a given document.

        Args:
        document(str): document to iterate.

        Returns:
        iter(Token): token iterator.
        """
        for match in Unigram._token_regex.finditer(document + ' '):
            yield Unigram(match)

    @staticmethod
    def get_label(term: str) -> str:
        """
        Returns the processed label for a given match.

        Args:
        term(str): term to process.

        Returns:
        str: normalized term.
        """
        return Unigram._drop_regex.sub('', term).lower()

    def __init__(self, match: re.Match):
        """
        Constructor for a Token.

        Args:
        match(re.Match): token regular expression match.
        """
        group = match.group()
        prefix_end = Unigram._prefix_regex.search(group).end()
        suffix_start = Unigram._suffix_regex.search(group).start()
        self.start = match.start() + prefix_end
        self.end = match.start() + suffix_start
        self.match = group[prefix_end:suffix_start]
        self.term = Unigram.get_label(self.match)
        self.p = None

    def __str__(self):
        """
        Returns a string representation of this token.

        Returns:
        str: string representation of this token.
        """
        return f"{self.term}[{self.p}]"

    def __lt__(self, other):
        """
        Compares two Unigram.

        Args:
        other(Unigram): to compare this against.

        Returns:
        bool: true if this is less than other (considering token counts).
        """
        return self.p < other.p or (
            self.p == other.p and self.term < other.term
        )