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
