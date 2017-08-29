# -*- coding: utf-8 -*-
import collections
import heapq


class LRUCache(object):
    """
    146. LRU Cache
    Design and implement a data structure for Least Recently Used (LRU)
    cache. It should support the following operations: get and put.

    get(key) - Get the value (will always be positive) of the key
    if the key exists in the cache, otherwise return -1.
    put(key, value) - Set or insert the value if the key is not
    already present. When the cache reached its capacity,
    it should invalidate the least recently used item before
    inserting a new item.

    Follow up:
    Could you do both operations in O(1) time complexity?
    """

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            v = self.cache[key]
            del self.cache[key]
            self.cache[key] = v
            return v
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value

        if self.capacity < len(self.cache):
            self.cache.popitem(last=False)


class MedianFinder(object):
    """
    295. Find Median from Data Stream
    Median is the middle value in an ordered integer list.
    If the size of the list is even, there is no middle value.
    So the median is the mean of the two middle value.

    Examples:
    [2,3,4] , the median is 3

    [2,3], the median is (2 + 3) / 2 = 2.5

    Design a data structure that supports the following two operations:

    void addNum(int num) - Add a integer number from the data stream to the data structure.
    double findMedian() - Return the median of all elements so far.

    For example:

        addNum(1)
        addNum(2)
        findMedian() -> 1.5
        addNum(3)
        findMedian() -> 2

    """

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = []
        self.large = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        if len(self.small) > len(self.large):
            heapq.heappush(self.large, -heapq.heappop(self.small))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0]) / 2.0
        else:
            return self.large[0] * 1.0


class WordDistance(object):
    """
    244. Shortest Word Distance II
    This is a follow up of Shortest Word Distance.
    The only difference is now you are given the list of words and your
    method will be called repeatedly many times with different parameters. How would you optimize it?

    Design a class which receives a list of words in the constructor,
    and implements a method that takes two words word1 and word2 and return
    the shortest distance between these two words in the list.

    For example,
    Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

    Given word1 = “coding”, word2 = “practice”, return 3.
    Given word1 = "makes", word2 = "coding", return 1.

    """

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.d = {}
        for i, w in enumerate(words):
            self.d[w] = self.d.get(w, []) + [i]

    def shortest(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        return min(abs(i - j) for i in self.d[word1] for j in self.d[word2])


class ZigzagIterator(object):
    """
    281. Zigzag Iterator
    Given two 1d vectors, implement an iterator to return their elements alternately.

    For example, given two 1d vectors:

    v1 = [1, 2]
    v2 = [3, 4, 5, 6]
    By calling next repeatedly until hasNext returns false, the order of elements
    returned by next should be: [1, 3, 2, 4, 5, 6].

    Follow up: What if you are given k 1d vectors? How well can your code be extended to such cases?

    """
    '''不要将值都存储出来。这不是iterator应该做的'''

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self._queue = [(len(v), iter(v)) for v in [v1, v2] if v]

    def next(self):
        """
        :rtype: int
        """
        l, i = self._queue.pop(0)
        if l > 1:
            self._queue.append((l - 1, i))
        return i.next()

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self._queue) > 0
