# -*- coding: utf-8 -*-
class Numbers(object):

    def reverse(self, x):
        """
        # 7
        Reverse digits of an integer.

        Example1: x = 123, return 321
        Example2: x = -123, return -321
        :type x: int
        :rtype: int
        """
        neg = False if x >= 0 else True
        x = int(str(abs(x))[::-1])
        if x >= 0x7fffffff:  # pow(2, 31) - 1
            return 0
        return x if not neg else x * -1
