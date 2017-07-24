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

    def myPow(self, x, n):
        """
        # 50. Pow(x, n)
        Implement pow(x, n).
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1.0
        k = abs(n)
        ans = x
        index = 1
        while index * 2 < k:
            index *= 2
            ans *= ans
        ans *= self.myPow(x, k - index)
        return ans if n > 0 else (1.0 / ans if ans != 0 else 1.0)

    def mySqrt(self, x):
        """
        69. Sqrt(x)
        Implement int sqrt(int x).

        Compute and return the square root of x.
        :type x: int
        :rtype: int
        """
        l = 0
        h = x
        while l <= h:
            mid = (l + h) / 2
            if mid ** 2 <= x < (mid + 1) ** 2:
                return mid
            elif mid ** 2 > x:
                h = mid
            else:
                l = mid + 1


if __name__ == '__main__':
    # debug template
    n = Numbers()
    print n.myPow(2.0, -2147483648)
