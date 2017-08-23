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

    def convertToTitle(self, n):
        """
        168. Excel Sheet Column Title
        Given a positive integer, return its corresponding column title as appear in an Excel sheet.

        For example:

            1 -> A
            2 -> B
            3 -> C
            ...
            26 -> Z
            27 -> AA
            28 -> AB
        :type n: int
        :rtype: str
        """
        '''坑爹的是，并非从0开始。所以每轮要先减一'''
        build = ''
        while n > 0:
            n -= 1
            build = chr(n % 26 + 65) + build
            n /= 26
        return build

    def isHappy(self, n):
        """
        202. Happy Number
        Write an algorithm to determine if a number is "happy".

        A happy number is a number defined by the following process:
        Starting with any positive integer, replace the number by the sum of
        the squares of its digits, and repeat the process until the number equals 1
        (where it will stay), or it loops endlessly in a cycle which does not include 1.
        Those numbers for which this process ends in 1 are happy numbers.

        :type n: int
        :rtype: bool
        """
        nums = [i for i in str(n)]
        dict = set()
        while True:
            next_sum = 0
            for d in nums:
                next_sum += int(d) ** 2
            if next_sum == 1:
                return True
            elif next_sum in dict:
                return False
            else:
                dict.add(next_sum)
                nums = [i for i in str(next_sum)]

    def isPowerOfThree(self, n):
        """
        326. Power of Three
        Given an integer, write a function to determine if it is a power of three.

        Follow up:
        Could you do it without using any loop / recursion?

        :type n: int
        :rtype: bool
        """
        return n > 0 == 3 ** 19 % n


if __name__ == '__main__':
    # debug template
    n = Numbers()
    print n.isHappy(4)
