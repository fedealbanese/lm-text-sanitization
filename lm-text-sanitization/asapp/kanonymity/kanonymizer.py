from __future__ import annotations

import pickle
import logging
from asapp.kanonymity import PQueue, InvertedIndex, LabelSet
from asapp.commons import Unigram


class KAnonymizer:
    """
    This class implements text sanitization via k-anonymity.
    """

    def __init__(
            self, 
            k: int,
            model_filename: str = None):
        """
        Constructor for a KAnonymizer.

        Args:
        k(int): number of documents to make it indistinguishable.
        model_filename(str): filename where the model is persisted.
        """
        self.k = k
        self.model_filename = model_filename
        self.iindex = InvertedIndex()

    def add_doc(self, document: str):
        """
        Adds a document to the KAnonymizer's inverted index.

        Args:
        document(str): text to add.
        """
        self.iindex.add({token.label for token in Unigram.tokenize(document)})

    def get_doc_repr(self, document: str) -> list:
        """
        Returns the representation of a document.

        Args:
        document(str): document to process.

        Returns:
        list(Unigram): representation of the document.
        """
        tokens = list(Unigram.tokenize(document))
        safe_labels = self._get_safe(LabelSet.build(tokens))
        result = []
        for token in tokens:
            token.p = 1.0 if token.label in safe_labels else 0.0
            result.append(token)
        return result

    def _get_safe(self, tokens: LabelSet) -> LabelSet:
        """
        Returns the set of safe unigram labels.

        Args:
        tokens(list of Unigram): tokens to filter.

        Returns:
        LabelSet: set of safe unigram labels.
        """
        self.iindex.clear_cache()
        queue = PQueue()
        queue.push(SearchState(self.iindex, tokens))
        while queue:
            current = queue.pop()
            if len(current.dids) == self.k:
                return current.safe
            following = current.following()
            if following is None:
                continue
            queue.push(following)
            queue.push(current)
        return LabelSet.empty()
    
    def dump(self):
        """
        Dumps model values to file.
        """        
        if not self.model_filename:
            logging.warning("Cannot store to invalid model filename")
        with open(self.model_filename, 'wb') as model_file:
            pickle.dump(self.iindex, model_file)

    def load(self):
        """
        Loads model values from file.
        """ 
        if not self.model_filename:
            logging.warning("Cannot load from invalid model filename")

        try:
            with open(self.model_filename, 'rb') as model_file:
                self.iindex = pickle.load(model_file)
        except FileNotFoundError:
            logging.warning(f"Model file not found: {self.model_filename}")


class SearchState:
    """
    This class represents a k-anonymity search state.
    """

    _empty_set = frozenset()

    def __init__(
            self,
            iindex: InvertedIndex,
            original: LabelSet,
            safe = None,
            dids = None):
        """
        Constructor for a SearchState.

        Args:
        iindex(InvertedIndex): KAnonymizer's inverted index.
        original(LabelSet): original set of labels to redact.
        safe(LabelSet): set of labels considered safe so far.
        dids(frozenset of int): document IDs used as indistinguishable.
        """
        self.iindex = iindex
        self.original = original
        self.safe = safe if safe is not None else original
        self.redacted = len(self.original) - len(self.safe)
        self.dids = dids if dids is not None else SearchState._empty_set
        self.siblings = iindex.get_similar(self.safe)

    def __lt__(self, other: SearchState) -> bool:
        """
        Returns whether this state is less than other.

        Args:
        other(SearchState): other search state to compare against.
        """
        return self.redacted < other.redacted or (
            self.redacted == other.redacted and len(self.dids) > len(other.dids)
        )

    def following(self):
        """
        Returns the following search state.

        Returns:
        SearchState: the next search state to consider.
        """
        for did in self.siblings:
            if did not in self.dids:
                break
        else:
            return None
        safe = self.safe.intersection(self.iindex.get_words(did))
        self.redacted = len(self.original) - len(safe)
        if not safe:
            return None
        return SearchState(
            self.iindex, self.original, safe, self.dids.union({did})
        )
