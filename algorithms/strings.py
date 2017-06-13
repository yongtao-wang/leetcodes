# -*- coding: utf-8 -*-
class Strings(object):
    def myAtoi(str):
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
        :type s: str
        :rtype: int
        """
        d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res, p = 0, 'I'
        for c in s[::-1]:
            res, p = res - d[c] if d[c] < d[p] else res + d[c], c
        return res
