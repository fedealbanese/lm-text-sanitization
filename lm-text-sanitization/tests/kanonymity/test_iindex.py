
from asapp.kanonymity import InvertedIndex


def test_empty():
    iindex = InvertedIndex()
    docs = iindex.get_documents("any")
    assert not docs
    words = iindex.get_words(0)
    assert not words
