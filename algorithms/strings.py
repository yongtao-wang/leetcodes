# -*- coding: utf-8 -*-
import sys


class Strings(object):
    def myAtoi(self, str):
        """
        # 8 String to Integer (atoi)
        Implement atoi to convert a string to an integer.
        :type str: str
        :rtype: int
        """
        '''
        test cases including:
            "++1"
            "    +4500"
            " +-1"
            " --2"
            "  +123ab33"
            "-2147483648"
        anyway, a shitty question as commented
        '''
        if not str:
            return 0
        index = 0
        base = 0
        while str[index] == ' ':
            index += 1
        s = str[index:]
        if len(s) == 1 and ord(s[0]) < ord('0') or ord(s[0]) > ord('9'):
            return 0
        sign = 1
        if s.startswith('-'):
            sign = -1
        s = s[1:]

        for c in s:
            if ord(c) < ord('0') or ord(c) > ord('9'):
                break
            base = base * 10 + int(c)
            if base * sign > 0x7fffffff:
                return 0x7fffffff
            if base * sign < -0x80000000:
                return -0x80000000

        return base * sign

    def isMatch(self, s, p):
        """
        # 10 regular expression matching

        Implement regular expression matching with support for '.' and '*'.

        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.

        The matching should cover the entire input string (not partial).

        The function prototype should be:
        bool isMatch(const char *s, const char *p)

        Some examples:
        isMatch("aa","a") → false
        isMatch("aa","aa") → true
        isMatch("aaa","aa") → false
        isMatch("aa", "a*") → true
        isMatch("aa", ".*") → true
        isMatch("ab", ".*") → true
        isMatch("aab", "c*a*b") → true
        :type s: str
        :type p: str
        :rtype: bool
        """
        '''
        当j未溢出时，循环：
            如果是最后一个字符
                如果对应s[i]是最后一个字符：
                    如果p[j]是'.'
                        return True
                    如果p[j]不是'.'
                        return s[i] == p[j]
                如果对应s[i]不是最后一个字符：
                    return False
            如果不是最后一个字符
                如果p[j]是'.'
                    如果紧跟着'*'
                        如果这个'*'是最后一个字符
                            True
                        如果这个'*'不是最后一个字符
                            # 这tm就麻烦了……得反向匹配了？
                    如果没有跟着'*'
                        匹配任意一个i（等于什么都不做，i += 1）
                        j += 1
                如果p[j]不是'.'
                    如果紧跟着'*'
                        循环直至不匹配i
                        j += 2
                    如果没有跟着'*'
                        匹配一个i (if not s[i] == p[j] return False)
                        j += 1
                        
        ----  leetcode讲解  ----
        leet上大多为dp解法。此处暂时先写上。
        
        ----  此处为DFS解法 速度会明显慢一些  ----
        
        def isMatch(self, s, p):
            cache = {}
            if (s, p) in self.cache:
                return self.cache[(s, p)]
            if not p:
                return not s
            if p[-1] == '*':
                if self.isMatch(s, p[:-2]):
                    self.cache[(s, p)] = True
                    return True
                if s and (s[-1] == p[-2] or p[-2] == '.') and self.isMatch(s[:-1], p):
                    self.cache[(s, p)] = True
                    return True
            if s and (p[-1] == s[-1] or p[-1] == '.') and self.isMatch(s[:-1], p[:-1]):
                self.cache[(s, p)] = True
                return True
            self.cache[(s, p)] = False
            return False
            
        -----------------------------------------
        
        '''
        dp = [[False] * (len(s) + 1) for _ in range(len(p) + 1)]
        dp[0][0] = True
        for i in range(1, len(p)):
            dp[i + 1][0] = dp[i - 1][0] and p[i] == '*'
        for i in range(len(p)):
            for j in range(len(s)):
                if p[i] == '*':
                    dp[i + 1][j + 1] = dp[i - 1][j + 1] or dp[i][j + 1]
                    if p[i - 1] == s[j] or p[i - 1] == '.':
                        dp[i + 1][j + 1] |= dp[i + 1][j]
                else:
                    dp[i + 1][j + 1] = dp[i][j] and (p[i] == s[j] or p[i] == '.')
        return dp[-1][-1]

    def romanToInt(self, s):
        """
        # 13 Roman to Integer
        Given a roman numeral, convert it to an integer.

        Input is guaranteed to be within the range from 1 to 3999.
        :type s: str
        :rtype: int
        """
        '''使用tuple。也可以再设一个变量。只是这样比较Pythonic'''
        d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res, p = 0, 'I'
        for c in s[::-1]:
            res, p = res - d[c] if d[c] < d[p] else res + d[c], c
        return res

    def intToRoman(self, num):
        """
        # 12 Integer to Roman
        Given an integer, convert it to a roman numeral.

        Input is guaranteed to be within the range from 1 to 3999.
        :type num: int
        :rtype: str
        """
        m = ["", "M", "MM", "MMM"]
        c = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        x = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        i = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return m[num / 1000] + c[(num % 1000) / 100] + x[(num % 100) / 10] + i[num % 10]

    def longestCommonPrefix(self, strs):
        """
        # 14 Longest Common Prefix
        Write a function to find the longest common prefix string amongst an array of strings.

        :type strs: List[str]
        :rtype: str
        """
        count = 0
        for character in zip(*strs):
            if len(set(character)) > 1:
                break
            count += 1
        return '' if not strs else strs[0][:count]

    def isValidParentheses(self, s):
        """
        # 20. Valid Parentheses
        Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
        determine if the input string is valid.
        :type s: str
        :rtype: bool
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for p in s:
            if p in ['(', '{', '[']:
                stack.append(p)
            elif p in [')', '}', ']']:
                if not stack:
                    return False
                if stack[-1] == mapping[p]:
                    stack.pop()  # much faster than del stack[-1]
                else:
                    return False
        if not stack:
            return True
        else:
            return False

    def generateParenthesis(self, n):
        """
        # 22. Generate Parentheses
        Given n pairs of parentheses, write a function to generate
        all combinations of well-formed parentheses.

        :type n: int
        :rtype: List[str]
        """
        self._parentheses = []
        '''本题和lists中的#17颇有不同。不能用笛卡尔积'''

        def generate(p, left, right):
            if left:
                generate(p + '(', left - 1, right)
            if right > left:
                generate(p + ')', left, right - 1)
            if not right:
                self._parentheses += p,
            return self._parentheses

        return generate('', n, n)

    def countAndSay(self, n):
        """
        # 38. Count and Say
        Given an integer n, generate the nth term of the count-and-say sequence.
        Google count-and-say if you don't know what it is.

        :type n: int
        :rtype: str
        """
        base = '1'
        if n == 1:
            return base
        for _ in range(n - 1):
            count = 1
            temp = []
            for i in range(1, len(base)):
                if base[i] == base[i - 1]:
                    count += 1
                else:
                    temp.append(str(count))
                    temp.append(base[i - 1])
                    count = 1
            temp.append(str(count))
            temp.append(base[-1])
            base = ''.join(temp)
        return base

    def lengthOfLastWord(self, s):
        """
        58. Length of Last Word
        Given a string s consists of upper/lower-case alphabets
        and empty space characters ' ', return the length of last word in the string.

        If the last word does not exist, return 0.

        :type s: str
        :rtype: int
        """
        sp = s.split()
        return 0 if not sp else len(sp[-1])

    def addBinary(self, a, b):
        """
        67. Add Binary
        Given two binary strings, return their sum (also a binary string).

        For example,
        a = "11"
        b = "1"
        Return "100".

        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a, 2) + int(b, 2))[2:]

    def simplifyPath(self, path):
        """
        71. Simplify Path

        Given an absolute path for a file (Unix-style), simplify it.

        For example,
        path = "/home/", => "/home"
        path = "/a/./b/../../c/", => "/c"

        :type path: str
        :rtype: str
        """
        '''split应该是第一反应'''
        stack = []
        split_path = [p for p in path.split('/') if p != '.' and p != '']
        for p in split_path:
            if p == '..':
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(p)
        return '/' + '/'.join(stack)

    def minWindow(self, s, t):
        """
        76. Minimum Window Substring
        Given a string S and a string T, find the minimum window in S
        which will contain all the characters in T in complexity O(n).

        For example,
        S = "ADOBECODEBANC"
        T = "ABC"
        Minimum window is "BANC".
        :type s: str
        :type t: str
        :rtype: str
        """
        h = {}
        start, end = 0, 0
        count = len(t)
        w_len = sys.maxint
        min_start = 0

        # prepare hash map
        for i in t:
            if i in h:
                h[i] += 1
            else:
                h[i] = 1

        while end < len(s):
            # find a window
            while count > 0 and end < len(s):
                if s[end] in h:
                    h[s[end]] -= 1
                    if h[s[end]] >= 0:
                        count -= 1
                end += 1
            # optimize current window
            while count == 0:
                if s[start] in h:
                    h[s[start]] += 1
                    if h[s[start]] > 0:
                        count += 1
                        if end - start < w_len:
                            w_len = end - start
                            min_start = start
                start += 1
        return s[min_start: min_start + w_len] if w_len != sys.maxint else ''

    def numberToWords(self, num):
        """
        273. Integer to English Words
        Convert a non-negative integer to its english words representation.
        Given input is guaranteed to be less than 2^31 - 1.

        For example,
        123 -> "One Hundred Twenty Three"
        12345 -> "Twelve Thousand Three Hundred Forty Five"
        1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
        :type num: int
        :rtype: str
        """
        if num == 0:
            return 'Zero'

        to_19 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
                 "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
                 "Nineteen"]
        tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        thousands = ["Thousand", "Million", "Billion"]

        def build(n):
            if n == 0:
                return []
            if n < 20:
                return [to_19[n]]
            if n < 100:
                return [tens[n / 10]] + build(n % 10)
            if n < 1000:
                return [to_19[n / 100]] + ['Hundred'] + build(n % 100)

            for i, v in enumerate(thousands, 1):
                if n < 1000 ** (i + 1):
                    return build(n / (1000 ** i)) + [v] + build(n % (1000 ** i))

        return ' '.join(build(num))

    def removeInvalidParentheses(self, s):
        """
        301. Remove Invalid Parentheses
        Remove the minimum number of invalid parentheses in order to
        make the input string valid. Return all possible results.

        Note: The input string may contain letters other than the parentheses ( and ).

        Examples:
            "()())()" -> ["()()()", "(())()"]
            "(a)())()" -> ["(a)()()", "(a())()"]
            ")(" -> [""]

        :type s: str
        :rtype: List[str]
        """
        '''又是一个BFS。参考126 - word ladder II'''

        def _is_valid(s):
            stack = 0
            for p in s:
                if p == '(':
                    stack += 1
                elif p == ')':
                    stack -= 1
                    if stack < 0:
                        return False
            return stack == 0

        level = {s}
        while level:
            valid = filter(_is_valid, level)
            if valid:
                return valid
            level = {s[:i] + s[i + 1:] for s in level for i in xrange(len(s))}


if __name__ == '__main__':
    ss = Strings()
    print ss.numberToWords(1234567)
