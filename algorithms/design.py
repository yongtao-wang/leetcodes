# -*- coding: utf-8 -*-
import collections


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
