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
        '''NOTE: 通常处理palindrome都是两个指针，其中一个用于遍历，另一个用于定位起始'''
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

    def shortestPalindrome(self, s):
        """
        214. Shortest Palindrome
        Given a string S, you are allowed to convert it to a palindrome by
        adding characters in front of it. Find and return the shortest palindrome
        you can find by performing this transformation.

        For example:

        Given "aacecaaa", return "aaacecaaa".

        Given "abcd", return "dcbabcd".

        :type s: str
        :rtype: str
        """
        ll = len(s)
        if ll <= 1:
            return s
        min_start, max_len, i = 0, 1, 0
        while i < ll:
            if ll - i < max_len / 2:
                break
            j, k = i, i
            while k < ll - 1 and s[k] == s[k + 1]:
                k += 1
            i = k + 1
            while k < ll - 1 and j > 0 and s[k + 1] == s[j - 1]:
                k, j = k + 1, j - 1
            if k - j + 1 >= max_len and j == 0:
                min_start, max_len = j, k - j + 1
        prefix = s[min_start + max_len: ll][::-1]
        return prefix + s
