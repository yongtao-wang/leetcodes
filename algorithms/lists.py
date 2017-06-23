# -*- coding: utf-8 -*-
class Lists(object):

    def twoSum(self, nums, target):
        """
        # 1
        Given an array of integers, return indices of the two numbers such that they add up to a specific target.

        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        :param nums:
        :param target:
        :return:
        """
        '''
        此类题目用#15那样的左右逼近是最好解法，但必须是sorted list
        又由于此处求的是下标，所以若遇到重复数字比如nums=[3, 3], target=6就很混乱
        '''
        rev = {}
        for index, n in enumerate(nums, 0):
            minus = target - n
            if n not in rev:
                rev[minus] = index
            else:
                return [rev[n], index]

    def threeSum(self, nums):
        """
        # 15 3Sum
        Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0?
        Find all unique triplets in the array which gives the sum of zero.
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''使用左右逼近能达到logN。总效率N * logN'''
        res = []
        nums.sort()  # 如果不想改变list顺序，可以使用sorted(nums)
        for i in xrange(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i + 1
            r = len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > 0:
                    l += 1
                elif s < 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res

    def threeSumClosest(self, nums, target):
        """
        # 16 3-Sum Closest
        Given an array S of n integers, find three integers in S
        such that the sum is closest to a given number, target.
        Return the sum of the three integers.
        You may assume that each input would have exactly one solution.
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        ss = sorted(nums)
        closest_sum = ss[0] + ss[1] + ss[2]
        for i in xrange(len(ss) - 2):
            l = i + 1
            r = len(ss) - 1
            while l < r:
                sum3 = ss[l] + ss[i] + ss[r]
                if sum3 == target:
                    return sum3
                elif abs(sum3 - target) < abs(closest_sum - target):
                    closest_sum = sum3

                if sum3 < target:
                    l += 1
                else:
                    r -= 1
        return closest_sum


