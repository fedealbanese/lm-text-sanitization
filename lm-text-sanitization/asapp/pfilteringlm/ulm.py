
import logging
import math
import msgpack

from asapp.commons import Unigram


class ULM:
    """
    Unigram Language Model, keeps the count of the number of occurrences of
    each word in the vocabulary, and the total number of occurrences to
    compute their frequency.
    """

    _total_key = '__total__'

    def __init__(
            self,
            model_filename: str = None,
            load_init: bool = True
    ):
        """
        Constructor for an ULM.

        Args:
        model_filename(str): filename where the model is persisted.
        total_key(str): key to use to keep track of total tokens.
        load_init(bool): indicates if load the model on construction.
        """
        self.model_filename = model_filename
        self.unigram_counter = {}
        if load_init:
            self.load()

    def add_doc(self, document: str):
        """
        Adds a document to the model.

        Args:
        document(str): document to process.
        """
        tokens = Unigram.tokenize(document)
        for token in tokens:
            token_count = self.unigram_counter.get(token.term, 0)
            self.unigram_counter[token.term] = token_count + 1
            self.unigram_counter[ULM._total_key] = self.get_token_bound() + 1

    def _get_token_freq(self, token: Unigram) -> Unigram:
        """
        Returns a token enriched with its corresponding term count.

        Args:
        token(Token): token to enrich.

        Returns:
        the token enriched with its term count and p.
        """
        token_count = self.unigram_counter.get(token.term, 0)
        token.p = token_count / self.get_token_bound()
        return token

    def get_token_bound(self) -> int:
        """
        Returns the term count bound.

        Returns:
        int: total term count.
        """
        return self.unigram_counter.get(ULM._total_key, 0)

    def get_doc_repr(self, document: str, *args) -> list:
        """
        Returns the representation of a document.

        Args:
        document(str): document to process.

        Returns:
        list(Unigram): representation of the document.
        """
        return [
            self._get_token_freq(token)
            for token in Unigram.tokenize(document)
        ]

    def clean_up(self, p: float):
        """
        Removes terms that do not meet a given frequency threshold.

        Args:
        p(float): frequency threshold.
        """
        cutoff = math.ceil(p * self.get_token_bound())
        to_remove = [
            term for term, count in self.unigram_counter.items()
            if count < cutoff
        ]
        for term in to_remove:
            self.unigram_counter.pop(term)

    def load(self) -> bool:
        """
        Loads model values from file.
        """
        if not self.model_filename:
            logging.warning("Cannot load from invalid model filename")
            return False
        try:
            with open(self.model_filename, 'rb') as model_file:
                self.unigram_counter = msgpack.load(model_file)
                return True
        except FileNotFoundError:
            logging.warning(f"Model file not found: {self.model_filename}")
            return False

    def dump(self) -> bool:
        """
        Dumps model values to file.
        """
        if not self.model_filename:
            logging.warning("Cannot store to invalid model filename")
            return False
        with open(self.model_filename, 'wb') as model_file:
            msgpack.dump(self.unigram_counter, model_file)
            return True
