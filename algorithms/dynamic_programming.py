# -*- coding: utf-8 -*-

class DynamicProgramming(object):
    def canJump(self, nums):
        """
        55. Jump Game
        Given an array of non-negative integers, you are initially positioned at the first index of the array.

        Each element in the array represents your maximum jump length at that position.

        Determine if you are able to reach the last index.

        For example:
        A = [2,3,1,1,4], return true.

        A = [3,2,1,0,4], return false.

        :type nums: List[int]
        :rtype: bool
        """
        '''严格来说这不算DP解法。但思路是基于DP的。另外从后往前往往效率要高一些'''

        if not nums:
            return True
        jump_index = 0
        for i, value in enumerate(nums):
            if i > jump_index:
                return False
            jump_index = max(jump_index, i + value)
        return True

    def coinChange(self, coins, amount):
        """
        # 322. Coin Change
        You are given coins of different denominations and a total
        amount of money amount. Write a function to compute the fewest
        number of coins that you need to make up that amount. If that
        amount of money cannot be made up by any combination of
        the coins, return -1.

        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        memo = {0: 0}
        for c in coins:
            memo[c] = 1
        for i in xrange(1, amount + 1):
            if i not in memo:
                checklist = []
                for c in coins:
                    if i > c and i - c in memo:
                        checklist.append(memo[i - c])
                if not checklist:
                    continue
                memo[i] = min(checklist) + 1
        return memo[amount] if amount in memo else -1

    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        464. Can I Win
        In the "100 game," two players take turns adding, to a running total,
        any integer from 1..10. The player who first causes the running total to reach or exceed 100 wins.

        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        if (1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal:
            return False
        return self._solve(range(1, maxChoosableInteger + 1), desiredTotal, {})

    def _solve(self, nums, target, d):
        h = str(nums)
        if h in d:
            return d[h]

        if nums[-1] >= target:
            d[h] = True

        elif len(nums) == 1 and nums[0] < target:
            d[h] = False

        else:
            d[h] = False
            for i in xrange(len(nums)):
                if not self._solve(nums[:i] + nums[i + 1:], target - nums[i], d):
                    d[h] = True
                    return True
        return d[h]


if __name__ == '__main__':
    dp = DynamicProgramming()
    print dp.canIWin(10, 11)
