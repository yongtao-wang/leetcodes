# -*- coding: utf-8 -*-
class SubstringAndPalindrome(object):

    def lengthOfLongestSubstring(self, s):
        """
        # 3 Longest Substring Without Repeating Characters
        Given a string, find the length of the longest substring without repeating characters.

        :type s: str
        :rtype: int
        """
        '''NOTE: 1.使用max()函数代替if比较；2.若需要使用下标，就别用set'''
        p = 0
        max_length = 0
        used = {}
        for i in xrange(len(s)):
            if s[i] in used and p <= used[s[i]]:
                p = used[s[i]] + 1
            else:
                max_length = max(max_length, i - p + 1)
            used[s[i]] = i
        return max_length

    def longestPalindrome(self, s):
        """
        # 5 Given a string s, find the longest palindromic substring in s.
        You may assume that the maximum length of s is 1000.
        :type s: str
        :rtype: str
        """
        if len(s) == 0:
            return 0
        max_length = 1
        start = 0
        for i in xrange(len(s)):
            if i - max_length >= 1 and s[i - max_length - 1:i + 1] == s[i - max_length - 1:i + 1][::-1]:
                start = i - max_length - 1
                max_length += 2
                continue

            if i - max_length >= 0 and s[i - max_length:i + 1] == s[i - max_length:i + 1][::-1]:
                start = i - max_length
                max_length += 1
        return s[start:start + max_length]
