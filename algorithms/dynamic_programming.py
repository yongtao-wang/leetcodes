# -*- coding: utf-8 -*-

class DynamicProgramming(object):
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
