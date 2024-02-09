
from collections import Counter
from typing import Iterator

from asapp.kanonymity import LabelSet


class InvertedIndex:
    """
    A data structure mapping tokens to the documents that contain them.
    """

    _empty_set = frozenset()

    def __init__(self):
        """
        Constructor for an InvertedIndex.
        """
        self._next_did = 0
        self._words_to_docs = {}
        self._docs_to_words = {}
        self._cache = {}

    def add(self, words: set):
        """
        Adds a document (represented as a list of words) to the index.

        Args:
        words(list of str): list of words representing the document.
        """
        self._docs_to_words[self._next_did] = set(words)
        for token in words:
            if token not in self._words_to_docs:
                self._words_to_docs[token] = {self._next_did}
            else:
                self._words_to_docs[token].add(self._next_did)
        self._next_did += 1  # Generates sequential document IDs.

    def get_documents(self, word: str) -> set:
        """
        Returns the set of document IDs containing a given word.

        Args:
        word(str): word to search for.

        Returns:
        Set of document IDs containing the word.
        """
        return self._words_to_docs.get(word, InvertedIndex._empty_set)

    def get_words(self, did: int) -> set:
        """
        Returns tokens associated with a given document ID.

        Args:
        did(int): document ID to search for.

        Returns:
        set of str: set of words in the document.
        """
        return self._docs_to_words.get(did, InvertedIndex._empty_set)

    def clear_cache(self):
        """
        Clears the inverted index similarity cache.
        """
        self._cache.clear()

    def get_similar(self, words: LabelSet) -> Iterator:
        """
        Returns an iterator of documents sorted by similarity.

        Args:
        words(LabelSet): set of words to search for similar docs.

        Returns:
        Iterator of int: the document IDs similar to the given words.
        """
        buckets = self._cache.get(words)
        if buckets is None:
            document_counts = Counter()
            for word, count in words:
                for did in self.get_documents(word):
                    document_counts[did] += count
            buckets = {i: set() for i in range(len(words), 0, -1)}
            for did, count in document_counts.items():
                buckets[count].add(did)
            self._cache[words] = buckets
        for count, dids in buckets.items():
            yield from dids
