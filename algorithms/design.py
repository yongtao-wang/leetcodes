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
