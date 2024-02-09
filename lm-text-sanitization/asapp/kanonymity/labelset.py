from __future__ import annotations

from collections import Counter
from typing import Iterator

from asapp.commons import Unigram


class LabelSet:
    """
    This class represents a set of labels.
    """

    @staticmethod
    def empty() -> LabelSet:
        """
        Builds an empty label set.

        Returns:
        LabelSet: an empty set of labels.
        """
        return LabelSet(Counter(), 0)

    @staticmethod
    def build(unigrams: list) -> LabelSet:
        """
        Builds a LabelSet from a list of unigrams.

        Args:
        unigrams(list of Unigram): unigrams to be contained in the set.

        Returns:
        LabelSet: set of labels with count information.
        """
        counter = Counter()
        length = 0
        for unigram in unigrams:
            counter[unigram.label] += 1
            length += 1
        return LabelSet(counter, length)

    def __init__(self, counter: Counter, length: int):
        """
        Constructor for a LabelSet.

        Args:
        counter(Counter[str, int]): label counter.
        length(int): total number of elements in the label set.
        """
        self.counter = counter
        self.length = length
        self._hash = hash(frozenset(self.counter.keys()))

    def __hash__(self) -> int:
        """
        Returns a hash value for this LabelSet.

        Returns:
        int: hash value.
        """
        return self._hash

    def __eq__(self, other: LabelSet) -> bool:
        """
        Compares two LabelSets for equality.

        Args:
        other(LabelSet): other label set to compare with.

        Returns:
        bool: whether the label sets match or not.
        """
        return self.counter == other.counter

    def __len__(self) -> int:
        """
        Returns the number of labels in the set.

        Returns:
        int: The number of labels.
        """
        return self.length

    def __contains__(self, label: str) -> bool:
        """
        Indicates whether a label is contained in the set.

        Args:
        label(str): label to look for.

        Returns:
        bool: whether the label is contained or not.
        """
        return label in self.counter

    def __iter__(self) -> Iterator:
        """
        Returns an Iterator for the label set.

        Returns:
        Iterator of tuples (label, count): covering each label and its count.
        """
        yield from self.counter.items()

    def intersection(self, other: set) -> LabelSet:
        """
        Returns a label set with the intersection of this and other.

        Args:
        other(set[str]): other set of labels to intersect with.

        Returns:
        LabelSet result of the intersection.
        """
        counter = Counter()
        length = 0
        if len(other) < len(self.counter):
            for label in other:
                count = self.counter[label]
                if count > 0:
                    counter[label] = count
                    length += count
        else:
            for label, count in self.counter.items():
                if label in other:
                    counter[label] = count
                    length += count
        return LabelSet(counter, length)
