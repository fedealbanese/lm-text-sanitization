
from heapq import *


class PQueue:
    """
    This class implements a priority queue.
    """

    def __init__(self, seq=()):
        """
        Constructor for a pqueue.
        :param seq: iterable, initial collection of elements.
        """
        self._heap = list(seq)
        heapify(self._heap)

    def push(self, item):
        """Adds an element to the queue."""
        heappush(self._heap, item)

    def pop(self):
        """Removes and returns the smallest element in the queue."""
        return heappop(self._heap)

    def peek(self):
        """Returns the smallest element in the queue (without removing it)."""
        return self._heap[0]

    def clear(self):
        """Removes all elements in this queue."""
        self._heap.clear()

    def pop_push(self, item):
        """Removes the smallest element in the queue and then adds a new one."""
        return heapreplace(self._heap, item)

    def push_pop(self, item):
        """Adds an element to the queue and then removes and returns the smallest one."""
        return heappushpop(self._heap, item)

    def __len__(self):
        """Returns the number of elements in the queue."""
        return len(self._heap)

    def __bool__(self):
        """Returns whether this queue is empty."""
        return len(self) > 0

    def __eq__(self, other):
        """Returns whether two queues are equal."""
        return self._heap == other._heap

    def __hash__(self):
        """Raises a TypeError since pqueues are mutable and should not be hashed."""
        raise TypeError("unhashable type: 'pqueue'")
