# -*- coding: utf-8 -*-

import numpy


class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


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

    def findMedianSortedArrays(self, nums1, nums2):
        """
        4. Median of Two Sorted Arrays
        There are two sorted arrays nums1 and nums2 of size m and n respectively.

        Find the median of the two sorted arrays.
        The overall run time complexity should be O(log (m+n)).
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l = len(nums1) + len(nums2)
        if l % 2 == 1:
            return self._find_kth(nums1, nums2, l / 2)
        else:
            return (self._find_kth(nums1, nums2, l / 2 - 1) + self._find_kth(nums1, nums2, l / 2)) / 2.0

    def _find_kth(self, a, b, k):
        """
        :type a: List[int]
        :type b: List[int]
        :type k: int
        """
        if len(a) > len(b):
            a, b = b, a
        if not a:
            return b[k]
        if k == len(a) + len(b) - 1:
            return max(a[-1], b[-1])
        i = min(len(a) - 1, k / 2)
        j = min(len(b) - 1, k - i)
        if a[i] > b[j]:
            return self._find_kth(a[:i], b[j:], i)
        else:
            return self._find_kth(a[i:], b[:j], j)

    def threeSum(self, nums):
        """
        # 15 3Sum
        Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0?
        Find all unique triplets in the array which gives the sum of zero.
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''使用左右逼近能达到logN。总效率N * logN'''
        '''优化后的速度能达到98.04%，未优化时为68.44%'''
        res = []
        nums.sort()  # 如果不想改变list顺序，可以使用sorted(nums)
        for i in xrange(len(nums) - 2):

            # some optimization
            if nums[i] > 0:
                break

            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l = i + 1
            r = len(nums) - 1

            # some optimization, too
            if nums[l] > -nums[i] / 2.0:
                continue
            if nums[r] < -nums[i] / 2.0:
                continue

            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > 0:
                    l += 1
                elif s < 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
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
        '''此题只能做到N^2。比较大小时注意需要使用abs()'''
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

    def letterCombinations(self, digits):
        """
        # 17. Letter Combinations of a Phone Number
        Given a digit string, return all possible letter combinations that the number could represent.
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        import itertools
        pad = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        res = []
        out = []
        for d in digits:
            if d not in pad:
                return None
            res.append(pad[d])
        cartesian = list(itertools.product(*res))
        for c in cartesian:
            out.append(''.join(c))
        return out

    def fourSum(self, nums, target):
        """
        # 18 4-Sum
        Given an array S of n integers, are there elements a, b, c, and d in S
        such that a + b + c + d = target? Find all unique quadruplets in the array
        which gives the sum of target.
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        '''
        这是一个较为优化的算法，将4-sum转变成两个2-sum。精益求精。
        
        def fourSum(self, nums, target):
            def findNsum(nums, target, N, result, results):
                if len(nums) < N or N < 2 or target < nums[0]*N or target > nums[-1]*N:  # early termination
                    return
                if N == 2: # two pointers solve sorted 2-sum problem
                    l,r = 0,len(nums)-1
                    while l < r:
                        s = nums[l] + nums[r]
                        if s == target:
                            results.append(result + [nums[l], nums[r]])
                            l += 1
                            while l < r and nums[l] == nums[l-1]:
                                l += 1
                        elif s < target:
                            l += 1
                        else:
                            r -= 1
                else: # recursively reduce N
                    for i in range(len(nums)-N+1):
                        if i == 0 or (i > 0 and nums[i-1] != nums[i]):
                            findNsum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)
        
            results = []
            findNsum(sorted(nums), target, 4, [], results)
            return results
        '''
        nums.sort()
        result = []
        for i in xrange(len(nums) - 3):
            if nums[i] > target / 4.0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            target2 = target - nums[i]
            for j in xrange(i + 1, len(nums) - 2):
                if nums[j] > target2 / 3.0:
                    break
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                l = j + 1
                r = len(nums) - 1
                target3 = target2 - nums[j]

                if nums[l] > target3 / 2.0:
                    continue
                if nums[r] < target3 / 2.0:
                    continue
                while l < r:
                    sum_lr = nums[l] + nums[r]
                    if sum_lr == target3:
                        result.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif sum_lr < target3:
                        l += 1
                    else:
                        r -= 1
        return result

    def removeDuplicates(self, nums):
        """
        # 26. Remove Duplicates from Sorted Array

        Given a sorted array, remove the duplicates in place
        such that each element appear only once and return the new length.

        Do not allocate extra space for another array,
        you must do this in place with constant memory.
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        p = 0
        q = 0
        while q < len(nums):
            if nums[p] == nums[q]:
                q += 1
                continue
            p += 1
            # or simply nums[p] = nums[q]. speed up from 58% to 83%
            nums[p], nums[q] = nums[q], nums[p]
            q += 1
        return p + 1

    def removeElement(self, nums, val):
        """
        27. Remove Element

        Given an array and a value, remove all instances of that value in place
        and return the new length.

        Do not allocate extra space for another array,
        you must do this in place with constant memory.

        The order of elements can be changed.
        It doesn't matter what you leave beyond the new length.

        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if not nums:
            return 0
        p = 0
        q = len(nums) - 1
        while p < q:
            if nums[q] == val:
                q -= 1
                continue
            if nums[p] == val:
                nums[p], nums[q] = nums[q], nums[p]
                q -= 1
            p += 1
        if nums[p] is not val:
            p += 1
        return p

    def searchInsert(self, nums, target):
        """
        35. Search Insert Position
        Given a sorted array and a target value, return the index if the target is found.
        If not, return the index where it would be if it were inserted in order.

        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        def search(list_num, start, end):
            if not nums:
                return start
            mid = (start + end) / 2
            if start == end:
                return start + 1 if nums[start] < target else start
            if start + 1 == end:
                if target <= nums[start]:
                    return start
                elif target > nums[end]:
                    return end + 1
                else:
                    return end
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                return search(list_num[mid + 1:end + 1], mid, end)
            if nums[mid] > target:
                return search(list_num[start:mid], start, mid)

        return search(nums, 0, len(nums) - 1)

    def nextPermutation(self, nums):
        """
        #31. Next Permutation
        Implement next permutation, which rearranges numbers into the lexicographically
        next greater permutation of numbers.

        If such arrangement is not possible, it must rearrange it as the lowest
        possible order (ie, sorted in ascending order).

        The replacement must be in-place, do not allocate extra memory.

        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in xrange(len(nums) - 1, -1, -1):
            if i == 0:  # meaning the list is at maximum permutation
                nums[:] = nums[::-1]
                return
            if nums[i - 1] < nums[i]:
                # search from rightmost to find the first one larger than nums[i-1]
                for k in xrange(len(nums) - 1, i - 1, -1):
                    if nums[k] > nums[i - 1]:
                        nums[i - 1], nums[k] = nums[k], nums[i - 1]
                        break
                # reverse from i
                nums[i:] = nums[i:][::-1]
                return

    def longestValidParentheses(self, s):
        """
        32. Longest Valid Parentheses
        Given a string containing just the characters '(' and ')',
        find the length of the longest valid (well-formed) parentheses substring.

        For "(()", the longest valid parentheses substring is "()",
        which has length = 2.

        Another example is ")()())", where the longest valid parentheses substring is "()()",
        which has length = 4.
        :type s: str
        :rtype: int
        """
        stack = []
        matched = [0] * len(s)
        longest = 0
        for i in xrange(len(s)):
            if s[i] == '(':
                stack.append(i)
            elif stack:
                j = stack.pop(-1)
                matched[i], matched[j] = 1, 1
        count = 0
        for k in xrange(len(matched)):
            if matched[k] == 1:
                count += 1
            else:
                longest = max(longest, count)
                count = 0
        longest = max(longest, count)
        return longest

    def search(self, nums, target):
        """
        33. Search in Rotated Sorted Array
        Suppose an array sorted in ascending order is rotated
        at some pivot unknown to you beforehand.

        (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

        You are given a target value to search.
        If found in the array return its index, otherwise return -1.

        You may assume no duplicate exists in the array.

        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) / 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    def searchRange(self, nums, target):
        """
        # 34. Search for a Range
        Given an array of integers sorted in ascending order,
        find the starting and ending position of a given target value.

        Your algorithm's runtime complexity must be in the order of O(log n).

        If the target is not found in the array, return [-1, -1].

        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        另一个concise solution. 分开算左边和右边。效率略高一些。
        ---------------------------------------
        def searchRange(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            
            left, right = self.search_left(nums, target), self.search_right(nums, target)
            return [left, right] if left <= right else [-1, -1]
        
        def search_left(self, nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) / 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
        
        def search_right(self, nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) / 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            return right
        '''
        i = 0
        j = len(nums) - 1
        ans = [-1, -1]
        if not nums:
            return ans
        while i < j:
            mid = (i + j) / 2
            if nums[mid] < target:
                i = mid + 1
            else:
                j = mid
        if nums[i] != target:
            return ans
        else:
            ans[0] = i

        j = len(nums) - 1
        while i < j:
            mid = (i + j) / 2 + 1
            if nums[mid] > target:
                j = mid - 1
            else:
                i = mid
        ans[1] = j
        return ans

    def isValidSudoku(self, board):
        """
        # 36. Valid Sudoku
        Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.

        The Sudoku board could be partially filled,
        where empty cells are filled with the character '.'.

        Note:
        A valid Sudoku board (partially filled) is not necessarily solvable.
        Only the filled cells need to be validated.
        :type board: List[List[str]]
        :rtype: bool
        """
        row_dict = {}
        col_dict = {}
        cube_dict = {}
        if len(board) != 9:
            return False
        for i in xrange(len(board)):
            if len(board[i]) != 9:
                return False

        for i in xrange(len(board)):
            if i not in row_dict:
                row_dict[i] = set()
            for j in xrange(len(board[0])):
                if j not in col_dict:
                    col_dict[j] = set()
                if (i / 3, j / 3) not in cube_dict:
                    cube_dict[(i / 3, j / 3)] = set()

                if board[i][j] == '.':
                    continue
                if board[i][j] in row_dict[i] or board[i][j] in col_dict[j] \
                        or board[i][j] in cube_dict[(i / 3, j / 3)]:
                    return False
                row_dict[i].add(board[i][j])
                col_dict[j].add(board[i][j])
                cube_dict[(i / 3, j / 3)].add(board[i][j])
        return True

    def combinationSum(self, candidates, target):
        """
        # 39. Combination Sum
        Given a set of candidate numbers (C) (without duplicates) and a target number (T),
        find all unique combinations in C where the candidate numbers sums to T.

        The same repeated number may be chosen from C unlimited number of times.
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(nums, target, index, path, res):
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return
            for i in xrange(index, len(nums)):
                dfs(nums, target - nums[i], i, path + [nums[i]], res)

        res = []
        candidates.sort()
        dfs(candidates, target, 0, [], res)
        return res

    def combinationSum2(self, candidates, target):
        """
        # 40. Combination Sum II
        Given a collection of candidate numbers (C) and a target number (T),
        find all unique combinations in C where the candidate numbers sums to T.

        Each number in C may only be used once in the combination.
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(nums, target, index, path, res):
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return
            for i in xrange(index, len(nums)):
                if i != index and nums[i] == nums[i - 1]:
                    continue
                dfs(nums, target - nums[i], i + 1, path + [nums[i]], res)

        res = []
        candidates.sort()
        dfs(candidates, target, 0, [], res)
        return res

    def trap(self, height):
        """
        42. Trapping Rain Water
        Given n non-negative integers representing an elevation map
        where the width of each bar is 1, compute how much water it is able to trap after raining.

        For example,
        Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.

        :type height: List[int]
        :rtype: int
        """
        h = 0
        d = {}
        water = 0
        for i in xrange(len(height)):
            h = max(h, height[i])
            d[i] = h
        h = 0
        for i in xrange(len(height) - 1, -1, -1):
            h = max(h, height[i])
            water += min(d[i], h) - height[i]
        return water

    def permute(self, nums):
        """
        # 46. Permutations
        Given a collection of distinct numbers, return all possible permutations.

        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def dfs(n, path, ans):
            if not n:
                ans.append(path)
                return
            for i in xrange(len(n)):
                dfs(n[:i] + n[i + 1:], path + [n[i]], ans)

        res = []
        dfs(nums, [], res)
        return res

    def permuteUnique(self, nums):
        """
        # 47. Permutations II
        Given a collection of numbers that might contain duplicates,
        return all possible unique permutations.

        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def dfs(n, path, ans):
            if not n:
                ans.append(path)
                return
            for i in xrange(len(n)):
                if i != 0 and n[i] == n[i - 1]:
                    continue
                dfs(n[:i] + n[i + 1:], path + [n[i]], ans)

        nums.sort()
        res = []
        dfs(nums, [], res)
        return res

    def rotate(self, matrix):
        """
        # 48. Rotate Image
        You are given an n x n 2D matrix representing an image.
        Rotate the image by 90 degrees (clockwise).

        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        '''
        取巧的方法是:
                matrix[::] = zip(*matrix[::-1])
        1-line solution
        '''
        if not matrix:
            return matrix
        if len(matrix) != len(matrix[0]):
            return None
        n = len(matrix)
        matrix.reverse()
        for i in xrange(n):
            for j in xrange(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix

    def groupAnagrams(self, strs):
        """
        # 49. Group Anagrams
        Given an array of strings, group anagrams together.

        For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
        Return:

        [
          ["ate", "eat","tea"],
          ["nat","tan"],
          ["bat"]
        ]
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dictionary = {}
        for s in strs:
            key = ''.join(sorted(s))
            if key in dictionary:
                dictionary[key].append(s)
            else:
                dictionary[key] = [s]
        return [i for i in dictionary.itervalues()]

    def solveNQueens(self, n):
        """
        51. N-Queens
        The n-queens puzzle is the problem of placing n queens on an n×n chessboard
        such that no two queens attack each other.

        Given an integer n, return all distinct solutions to the n-queens puzzle.

        Each solution contains a distinct board configuration of the n-queens' placement,
        where 'Q' and '.' both indicate a queen and an empty space respectively.

        For example,
        There exist two distinct solutions to the 4-queens puzzle:

        [
         [".Q..",  // Solution 1
          "...Q",
          "Q...",
          "..Q."],

         ["..Q.",  // Solution 2
          "Q...",
          "...Q",
          ".Q.."]
        ]

        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        self._dfs_leet51([], n, [], [], res)
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in r] for r in res]

    def _dfs_leet51(self, queens, n, xy_diff, xy_sum, res):
        p = len(queens)
        if p == n:  # check width
            res.append(queens)
            return None
        for q in xrange(n):  # loop through all heights
            if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                self._dfs_leet51(queens + [q], n, xy_diff + [p - q], xy_sum + [p + q], res)

    def totalNQueens(self, n):
        """
        52. N-Queens II
        Follow up for N-Queens problem.

        Now, instead outputting board configurations,
        return the total number of distinct solutions
        :type n: int
        :rtype: int
        """
        self.res = 0
        self._dfs_leet52([], n, [], [])
        return self.res

    def _dfs_leet52(self, queens, n, xy_diff, xy_sum):
        p = len(queens)
        if p == n:
            self.res += 1
            return None
        for q in xrange(n):
            if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                self._dfs_leet52(queens + [q], n, xy_diff + [p - q], xy_sum + [p + q])

    def maxSubArray(self, nums):
        """
        # 53. Maximum Subarray
        Find the contiguous subarray within an array (containing at least one number)
        which has the largest sum.

        For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
        the contiguous subarray [4,-1,2,1] has the largest sum = 6.
        :type nums: List[int]
        :rtype: int
        """
        for i in xrange(1, len(nums)):
            nums[i] = max(nums[i - 1] + nums[i], nums[i])
        return max(nums)

    def spiralOrder(self, matrix):
        """
        # 54. Spiral Matrix

        Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

        For example,
        Given the following matrix:

        [
         [ 1, 2, 3 ],
         [ 4, 5, 6 ],
         [ 7, 8, 9 ]
        ]
        You should return [1,2,3,6,9,8,7,4,5].

        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        '''处理graph的要诀就是颠倒变换至便于操作的位置。参考#48的"取巧"解法'''
        return matrix and list(matrix.pop(0)) + self.spiralOrder(zip(*matrix)[::-1])

    def merge(self, intervals):
        """
        56. Merge Intervals

        Given a collection of intervals, merge all overlapping intervals.

        For example,
        Given [1,3],[2,6],[8,10],[15,18],
        return [1,6],[8,10],[15,18].

        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        '''对于sorted有了新的认识'''

        merged = []
        for i in sorted(intervals, key=lambda k: k.start):
            if merged and i.start <= merged[-1].end:
                merged[-1].end = max(merged[-1].end, i.end)
            else:
                merged.append(i)
        return merged

    def insert(self, intervals, newInterval):
        """
        57. Insert Interval
        Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

        You may assume that the intervals were initially sorted according to their start times.

        Example 1:
        Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

        Example 2:
        Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].

        This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        '''
        OR we can make a little use of # 56
        ------------------------------------------------------
        def insert(self, intervals, newInterval):
            intervals += [newInterval]
            res = []
            for i in sorted(intervals, key=lambda k: k.start):
                if res and i.start <= res[-1].end:
                    res[-1].end = max(res[-1].end, i.end)
                else:
                    res.append(i)
            return res
        ------------------------------------------------------
        '''
        start = newInterval.start
        end = newInterval.end
        i = 0
        res = []
        while i < len(intervals):
            if start <= intervals[i].end:
                if end < intervals[i].start:
                    break
                start = min(start, intervals[i].start)
                end = max(end, intervals[i].end)
            else:
                res.append(intervals[i])
            i += 1
        res.append(Interval(start, end))
        res += intervals[i:]
        return res

    # noinspection PyTypeChecker
    def generateMatrix(self, n):
        """
        59. Spiral Matrix II
        Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

        For example,
        Given n = 3,
        You should return the following matrix:
        [
         [ 1, 2, 3 ],
         [ 8, 9, 4 ],
         [ 7, 6, 5 ]
        ]

        :type n: int
        :rtype: List[List[int]]
        """
        '''
        临场能写出
        def generateMatrix(self, n):
            A = [[n*n]]
            while A[0][0] > 1:
                A = [range(A[0][0] - len(A), A[0][0])] + zip(*A[::-1])
            return A * (n>0)
        就相当了不起了
        '''
        matrix = []
        s = n * n + 1  # make it starting from 1 rather than 0
        while s > 1:
            s, e = s - len(matrix), s
            matrix = [range(s, e)] + zip(*matrix[::-1])
        return matrix  # spiral counter clockwise return zip(*matrix)

    def getPermutation(self, n, k):
        """
        60. Permutation Sequence

        The set [1,2,3,…,n] contains a total of n! unique permutations.

        By listing and labeling all of the permutations in order,
        We get the following sequence (ie, for n = 3):

        "123"
        "132"
        "213"
        "231"
        "312"
        "321"
        Given n and k, return the kth permutation sequence.

        Note: Given n will be between 1 and 9 inclusive.
        :type n: int
        :type k: int
        :rtype: str
        """
        '''注意，以下解法是在一定有解的前提下。
        如果不可解（比如n=1, k=2）则在判断nums[c]时加入些条件即可'''
        res = []
        nums = [i for i in xrange(1, n + 1)]
        f = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        while n > 0:
            c, k = k / f[n - 1], k % f[n - 1]
            if k > 0:
                res.append(str(nums[c]))
                nums.remove(nums[c])
            else:
                res.append(str(nums[c - 1]))
                nums.remove(nums[c - 1])
            n -= 1
        return ''.join(res)

    def uniquePaths(self, m, n):
        """
        62. Unique Paths
        A robot is located at the top-left corner of a m x n grid
        (marked 'Start' in the diagram below).

        The robot can only move either down or right at any point in time.
        The robot is trying to reach the bottom-right corner of the grid
        (marked 'Finish' in the diagram below).

        How many possible unique paths are there?

        :type m: int
        :type n: int
        :rtype: int
        """
        matrix = [[0] * m for _ in xrange(n)]
        for i in xrange(n):
            for j in xrange(m):
                if j == 0 or i == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1]
        return matrix[n - 1][m - 1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        63. Unique Paths II
        Follow up for "Unique Paths":

        Now consider if some obstacles are added to the grids. How many unique paths would there be?

        An obstacle and empty space is marked as 1 and 0 respectively in the grid.

        For example,
        There is one obstacle in the middle of a 3x3 grid as illustrated below.

        [
          [0,0,0],
          [0,1,0],
          [0,0,0]
        ]
        The total number of unique paths is 2.

        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = 0
        if m > 0:
            n = len(obstacleGrid[0])
        matrix = [[0] * n for _ in range(m)]
        for i in xrange(m):
            for j in xrange(n):
                if i == 0 and j == 0:
                    matrix[i][j] = 1 if obstacleGrid[i][j] == 0 else 0
                if obstacleGrid[i][j] == 1:
                    continue
                if i == 0 and j > 0 and obstacleGrid[i][j - 1] == 0:
                    matrix[i][j] = matrix[i][j - 1]
                    continue
                elif j == 0 and i > 0 and obstacleGrid[i - 1][j] == 0:
                    matrix[i][j] = matrix[i - 1][j]
                    continue
                if i > 0 and j > 0 and obstacleGrid[i - 1][j] == 0:
                    matrix[i][j] += matrix[i - 1][j]
                if i > 0 and j > 0 and obstacleGrid[i][j - 1] == 0:
                    matrix[i][j] += matrix[i][j - 1]
        return matrix[m - 1][n - 1]

    def minPathSum(self, grid):
        """
        64. Minimum Path Sum

        Given a m x n grid filled with non-negative numbers,
        find a path from top left to bottom right which minimizes
        the sum of all numbers along its path.

        Note: You can only move either down or right at any point in time.
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = 0
        if m > 0:
            n = len(grid[0])
        matrix = [[0] * n for _ in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if i == j == 0:
                    matrix[i][j] = grid[i][j]
                    continue
                if i == 0 and j > 0:
                    matrix[i][j] = grid[i][j] + matrix[i][j - 1]
                    continue
                elif j == 0 and i > 0:
                    matrix[i][j] = grid[i][j] + matrix[i - 1][j]
                    continue
                matrix[i][j] = grid[i][j] + min(matrix[i - 1][j], matrix[i][j - 1])
        return matrix[m - 1][n - 1]

    def plusOne(self, digits):
        """
        66. Plus One

        Given a non-negative integer represented as a non-empty array of digits,
        plus one to the integer.

        You may assume the integer do not contain any leading zero,
        except the number 0 itself.

        The digits are stored such that the most significant digit is
        at the head of the list.
        :type digits: List[int]
        :rtype: List[int]
        """
        '''没有转换为integer是为避免溢出'''
        reverse_digits = digits[::-1]
        reverse_digits[0] += 1
        for i in xrange(len(digits) - 1):
            if reverse_digits[i] > 9:
                reverse_digits[i] -= 10
                reverse_digits[i + 1] += 1
        if reverse_digits[-1] > 9:
            reverse_digits[-1] -= 10
            reverse_digits.append(1)
        return reverse_digits[::-1]

    def fullJustify(self, words, maxWidth):
        """
        68. Text Justification
        Given an array of words and a length L, format the text such that
        each line has exactly L characters and is fully (left and right) justified.

        You should pack your words in a greedy approach; that is,
        pack as many words as you can in each line. Pad extra spaces ' ' when necessary
        so that each line has exactly L characters.

        Extra spaces between words should be distributed as evenly as possible.
        If the number of spaces on a line do not divide evenly between words,
        the empty slots on the left will be assigned more spaces than the slots
        on the right.

        For the last line of text, it should be left justified and no extra space
        is inserted between words.

        For example,
        words: ["This", "is", "an", "example", "of", "text", "justification."]
        L: 16.

        Return the formatted lines as:
        [
           "This    is    an",
           "example  of text",
           "justification.  "
        ]
        Note: Each word is guaranteed not to exceed L in length.
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        text = ' '.join(words) + ' '
        if text == ' ':
            return [' ' * maxWidth]
        res = []
        while text:
            index = text.rfind(' ', 0, maxWidth + 1)
            line = text[:index].split()
            c_length = sum([len(w) for w in line])
            word_count = len(line)
            if word_count == 1:
                res.append(line[0] + ' ' * (maxWidth - c_length))
            else:
                space_each = (maxWidth - c_length) / (word_count - 1)
                space_remain = (maxWidth - c_length) % (word_count - 1)
                line[:-1] = [w + ' ' * space_each for w in line[:-1]]
                line[:space_remain] = [w + ' ' for w in line[:space_remain]]
                res.append(''.join(line))
            text = text[index + 1:]
        res[-1] = ' '.join(res[-1].split()).ljust(maxWidth)
        return res

    def climbStairs(self, n):
        """
        70. Climbing Stairs
        You are climbing a stair case. It takes n steps to reach to the top.

        Each time you can either climb 1 or 2 steps.
        In how many distinct ways can you climb to the top?
        :type n: int
        :rtype: int
        """
        climb = {}
        climb[1] = 1
        climb[2] = 2
        for i in xrange(3, n + 1):
            climb[i] = climb[i - 1] + climb[i - 2]
        return climb[n]

    def setZeroes(self, matrix):
        """
        73. Set Matrix Zeroes
        Given a m x n matrix, if an element is 0,
        set its entire row and column to 0. Do it in place.

        Follow up:
        Did you use extra space?
        A straight forward solution using O(mn) space is probably a bad idea.
        A simple improvement uses O(m + n) space, but still not the best solution.
        Could you devise a constant space solution?
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        if m == 0:
            return
        n = len(matrix[0])
        n_zero = False
        m_zero = False

        if 0 in matrix[0]:
            n_zero = True
        for line in matrix:
            if line[0] == 0:
                m_zero = True
                break

        for i in xrange(1, m):
            for j in xrange(1, n):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
        for i in xrange(1, m):
            if matrix[i][0] == 0:
                matrix[i] = [0] * n
        for j in xrange(1, n):
            if matrix[0][j] == 0:
                for k in xrange(m):
                    matrix[k][j] = 0
        if n_zero:
            matrix[0] = [0] * n
        if m_zero:
            for line in matrix:
                line[0] = 0

    def searchMatrix(self, matrix, target):
        """
        74. Search a 2D Matrix
        Write an efficient algorithm that searches for a value in an m x n matrix.
        This matrix has the following properties:

        Integers in each row are sorted from left to right.
        The first integer of each row is greater than the last integer of the previous row.
        For example,

        Consider the following matrix:

        [
          [1,   3,  5,  7],
          [10, 11, 16, 20],
          [23, 30, 34, 50]
        ]
        Given target = 3, return true.
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        line = None
        l = 0
        r = len(matrix) - 1
        while l <= r:
            mid = (l + r) / 2
            if target < matrix[l][0] or target > matrix[r][-1]:
                return False
            if matrix[mid][0] <= target <= matrix[mid][-1]:
                line = matrix[mid]
                break
            elif target > matrix[mid][-1]:
                l = mid + 1
            else:
                r = mid - 1
        return target in line

    def sortColors(self, nums):
        """
        75. Sort Colors
        Given an array with n objects colored red, white or blue,
        sort them so that objects of the same color are adjacent,
        with the colors in the order red, white and blue.

        Here, we will use the integers 0, 1, and 2 to represent the color red,
        white, and blue respectively.

        Note:
        You are not suppose to use the library's sort function for this problem.

        click to show follow up.

        Follow up:
        A rather straight forward solution is a two-pass algorithm using counting sort.
        First, iterate the array counting number of 0's, 1's, and 2's,
        then overwrite array with total number of 0's, then 1's and followed by 2's.

        Could you come up with an one-pass algorithm using only constant space?

        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        p1, p2 = 0, 0
        for i in xrange(len(nums)):
            if nums[i] == 0:
                nums[i], nums[p2] = nums[p2], nums[p1]
                nums[p1] = 0
                p1 += 1
                p2 += 1
            elif nums[i] == 1:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 += 1

    def combine(self, n, k):
        """
        77. Combinations
        Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

        For example,
        If n = 4 and k = 2, a solution is:

        [
          [2,4],
          [3,4],
          [2,3],
          [1,2],
          [1,3],
          [1,4],
        ]
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        '''
        THERE IS A FORMULA OF
            C(n, k) = C(n - 1, k - 1) + C(n - 1, k)
        the following solution takes its idea.
        time for some high school maths
        -------------------------------------------
        if k == 1:
            return [[i] for i in range(1, n + 1)]
        elif k == n:
            return [[i for i in range(1, n + 1)]]
        else:
            rs = []
            rs += self.combine(n - 1, k)
            part = self.combine(n - 1, k - 1)
            for ls in part:
                ls.append(n)
            rs += part
            return rs
        ------------------------------------------
        DFS itself is a little bit slow for this question
        '''

        def dfs(nums, c, index, path, res):
            if c == 0:
                res.append(path)
                return
            if len(nums[index:]) < c:
                return
            for i in xrange(index, len(nums)):
                dfs(nums, c - 1, i + 1, path + [nums[i]], res)

        result = []
        dfs(list(xrange(1, n + 1)), k, 0, [], result)
        return result

    def subsets(self, nums):
        """
        78. Subsets
        Given a set of distinct integers, nums, return all possible subsets.

        Note: The solution set must not contain duplicate subsets.

        For example,
        If nums = [1,2,3], a solution is:

        [
          [3],
          [1],
          [2],
          [1,2,3],
          [1,3],
          [2,3],
          [1,2],
          []
        ]
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''
        def dfs(numbers, start, path, res):
            res.append(path)
            for i in xrange(start, len(numbers)):
                dfs(numbers, i + 1, path + [numbers[i]], res)
        result = []
        dfs(sorted(nums), 0, [], result)
        return result
        '''
        res = [[]]
        for n in sorted(nums):
            res += [r + [n] for r in res]
        return res

    def exist(self, board, word):
        """
        79. Word Search
        Given a 2D board and a word, find if the word exists in the grid.

        The word can be constructed from letters of sequentially adjacent cell,
        where "adjacent" cells are those horizontally or vertically neighboring.
        The same letter cell may not be used more than once.

        For example,
        Given board =

        [
          ['A','B','C','E'],
          ['S','F','C','S'],
          ['A','D','E','E']
        ]
        word = "ABCCED", -> returns true,
        word = "SEE", -> returns true,
        word = "ABCB", -> returns false.

        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board:
            return False
        visited = {}
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                if self._dfs_leet79(board, i, j, visited, word):
                    return True
        return False

    def _dfs_leet79(self, board, i, j, visited, word):
        if len(word) == 0:
            return True
        if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or visited.get((i, j)):
            return False
        if board[i][j] != word[0]:
            return False
        visited[(i, j)] = True
        # DO NOT RUN SEPARATELY AS WE DON'T NEED TO CALCULATE ALL OF FOUR
        res = self._dfs_leet79(board, i - 1, j, visited, word[1:]) or \
              self._dfs_leet79(board, i + 1, j, visited, word[1:]) or \
              self._dfs_leet79(board, i, j - 1, visited, word[1:]) or \
              self._dfs_leet79(board, i, j + 1, visited, word[1:])
        visited[(i, j)] = False
        return res

    def largestRectangleArea(self, height):
        """
        84. Largest Rectangle in Histogram
        Given n non-negative integers representing the histogram's bar height
        where the width of each bar is 1, find the area of largest rectangle in the histogram.

        :type heights: List[int]
        :rtype: int
        """
        n = len(height)
        if n < 2:
            return height[0] if n else 0

        height.append(0)  # guard
        max_area = 0
        for i in xrange(0, n):
            if height[i] > height[i + 1]:
                bar = height[i]
                k = i
                while k >= 0 and height[k] > height[i + 1]:
                    bar = min(bar, height[k])
                    max_area = max(max_area, bar * (i - k + 1))
                    k -= 1
        return max_area

    def maximalRectangle(self, matrix):
        """
        85. Maximal Rectangle
        Given a 2D binary matrix filled with 0's and 1's,
        find the largest rectangle containing only 1's and return its area.
        For example, given the following matrix:

        1 0 1 0 0
        1 0 1 1 1
        1 1 1 1 1
        1 0 0 1 0

        Return 6.
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        w = len(matrix[0])
        max_area = 0
        height = [0] * (w + 1)
        for row in matrix:
            for i in xrange(w):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            for i in xrange(w):
                if height[i] > height[i + 1]:
                    bar = height[i]
                    j = i
                    while j >= 0 and height[j] > height[i + 1]:
                        bar = min(bar, height[j])
                        max_area = max((i - j + 1) * bar, max_area)
                        j -= 1
        return max_area

    def numDecodings(self, s):
        """
        91. Decode Ways

        :type s: str
        :rtype: int
        A message containing letters from A-Z is being encoded to numbers
        using the following mapping:

        'A' -> 1
        'B' -> 2
        ...
        'Z' -> 26
        Given an encoded message containing digits, determine
        the total number of ways to decode it.

        For example,
        Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

        The number of ways decoding "12" is 2.
        """

        if not s:
            return 0

        d = [0] * (len(s) + 1)
        d[0] = 1
        d[1] = 1 if s[0] != '0' else 0
        for i in xrange(2, len(s) + 1):
            if 0 < int(s[i - 1:i]) <= 9:
                d[i] += d[i - 1]
            if s[i - 2:i][0] != '0' and int(s[i - 2:i]) < 27:
                d[i] += d[i - 2]
        return d[len(s)]

    def maxProfit(self, prices):
        """
        121. Best Time to Buy and Sell Stock
        Say you have an array for which the ith element is the price of a given stock on day i.

        If you were only permitted to complete at most one transaction
        (ie, buy one and sell one share of the stock), design an algorithm to find
        the maximum profit.
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        min_price = prices[0]
        max_profit = 0
        for i in xrange(1, len(prices)):
            current_profit = prices[i] - min_price
            max_profit = max(current_profit, max_profit)
            if prices[i] < min_price:
                min_price = prices[i]
        return max_profit

    def findLadders(self, beginWord, endWord, wordList):
        """
        126. Word Ladder II
        (same as 127, return a list or visited words rather than the count of steps)
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        if endWord not in wordList:
            return []
        res = []
        wordList = set(wordList)
        queue = {beginWord: [[beginWord]]}

        # 将当前queue全部取出，全部处理完成后再放进queue
        # 考虑做为BFS求全部解的模板
        # 127不需要全部取出是因为只需要找到一任意一个解即可结束
        while queue:
            build_dict = {}
            for word in queue:
                if word == endWord:
                    res.extend([k for k in queue[word]])
                else:
                    for i in xrange(len(word)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            build = word[:i] + c + word[i + 1:]
                            if build in wordList:
                                if build in build_dict:
                                    build_dict[build] += [seq + [build] for seq in queue[word]]
                                else:
                                    build_dict[build] = [seq + [build] for seq in queue[word]]
            wordList -= set(build_dict.keys())
            queue = build_dict
        return res

    def ladderLength(self, beginWord, endWord, wordList):
        """
        127. Word Ladder
        Given two words (beginWord and endWord), and a dictionary's word list,
        find the length of shortest transformation sequence from beginWord to endWord, such that:

        Only one letter can be changed at a time.
        Each transformed word must exist in the word list.
        Note that beginWord is not a transformed word.

        Note:
        Return 0 if there is no such transformation sequence.
        All words have the same length.
        All words contain only lowercase alphabetic characters.
        You may assume no duplicates in the word list.
        You may assume beginWord and endWord are non-empty and are not the same.
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if endWord not in wordList:
            return 0
        queue = [(beginWord, 1)]
        wordList = set(wordList)
        visited = set()
        while queue:
            word, index = queue.pop(0)
            for i in xrange(len(word)):
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    build = word[:i] + j + word[i + 1:]
                    if build == endWord:
                        return index + 1
                    if build not in visited and build in wordList:
                        visited.add(build)
                        queue.append((build, index + 1))
        return 0

    def longestConsecutive(self, nums):
        """
        128. Longest Consecutive Sequence

        Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

        For example,
        Given [100, 4, 200, 1, 3, 2],
        The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

        Your algorithm should run in O(n) complexity.
        :type nums: List[int]
        :rtype: int
        """
        s = set(nums)
        longest = 0
        for n in nums:
            if n - 1 not in s:
                end = n + 1
                while end in s:
                    end += 1
                longest = max(longest, end - n)
        return longest

    def wordBreak(self, s, wordDict):
        """
        139. Word Break
        Given a non-empty string s and a dictionary wordDict containing
        a list of non-empty words, determine if s can be segmented into
        a space-separated sequence of one or more dictionary words.
        You may assume the dictionary does not contain duplicate words.

        For example, given
        s = "leetcode",
        dict = ["leet", "code"].

        Return true because "leetcode" can be segmented as "leet code".

        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        wordDict = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in xrange(0, len(s)):
            for j in xrange(i, len(s)):
                if dp[i] and s[i:j + 1] in wordDict:
                    dp[j + 1] = True
        return dp[len(s)]

    def wordBreakII(self, s, wordDict):
        """
        140. Word Break II
        Given a non-empty string s and a dictionary wordDict containing
        a list of non-empty words, add spaces in s to construct a sentence
        where each word is a valid dictionary word. You may assume
        the dictionary does not contain duplicate words.

        Return all such possible sentences.

        For example, given
        s = "catsanddog",
        dict = ["cat", "cats", "and", "sand", "dog"].

        A solution is ["cats and dog", "cat sand dog"].

        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        '''DP + DFS'''
        d = {}
        return self._dfs_leet140(s, d, wordDict)

    def _dfs_leet140(self, s, d, wordDict):
        if not s:
            return [None]
        if s in d:
            return d[s]
        res = []
        for word in wordDict:
            n = len(word)
            if word == s[:n]:
                for each in self._dfs_leet140(s[n:], d, wordDict):
                    if each:
                        res.append(word + ' ' + each)
                    else:
                        res.append(word)
            d[s] = res
        return res

    def maxPoints(self, points):
        """
        149. Max Points on a Line
        Given n points on a 2D plane, find the maximum number
        of points that lie on the same straight line.

        :type points: List[Point]
        :rtype: int
        """

        l = len(points)
        max_count = 0
        for i in xrange(l):
            d = {'vertical': 1}
            count_lines = 0
            ix, iy = points[i].x, points[i].y
            for j in xrange(i + 1, l):
                px, py = points[j].x, points[j].y
                if px == ix and py == iy:
                    count_lines += 1
                    continue
                if px == ix:
                    slope = 'vertical'
                else:
                    slope = numpy.longdouble(1) * (py - iy) / (px - ix)
                if slope not in d:
                    d[slope] = 1
                d[slope] += 1

            max_count = max(max_count, max(d.values()) + count_lines)
        return max_count

    def rob(self, nums):
        """
        198. House Robber
        You are a professional robber planning to rob houses along a street.
        Each house has a certain amount of money stashed, the only constraint
        stopping you from robbing each of them is that adjacent houses have security
        system connected and it will automatically contact the police if two adjacent
        houses were broken into on the same night.

        Given a list of non-negative integers representing the amount of money of each house,
        determine the maximum amount of money you can rob tonight without alerting the police.

        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        pre, now = 0, 0
        for i in nums:
            pre, now = now, max(pre + i, now)
        return now

    def numIslands(self, grid):
        """
        200. Number of Islands
        Given a 2d grid map of '1's (land) and '0's (water),
        count the number of islands. An island is surrounded by
        water and is formed by connecting adjacent lands horizontally or vertically.
        You may assume all four edges of the grid are all surrounded by water.

        Example 1:

        11110
        11010
        11000
        00000
        Answer: 1

        Example 2:

        11000
        11000
        00100
        00011
        Answer: 3

        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0

        count = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j] == '1':
                    self._dfs_leet200(grid, i, j)
                    count += 1
        return count

    def _dfs_leet200(self, grid, m, n):
        if m < 0 or n < 0 or m >= len(grid) or n >= len(grid[0]) or grid[m][n] != '1':
            return
        grid[m] = grid[m][:n] + '$' + grid[m][n + 1:]
        self._dfs_leet200(grid, m + 1, n)
        self._dfs_leet200(grid, m - 1, n)
        self._dfs_leet200(grid, m, n + 1)
        self._dfs_leet200(grid, m, n - 1)

    def rob_2(self, nums):
        """
        213. House Robber II
        Note: This is an extension of House Robber.

        After robbing those houses on that street, the thief has found himself
        a new place for his thievery so that he will not get too much attention.
        This time, all houses at this place are arranged in a circle. That means
        the first house is the neighbor of the last one. Meanwhile, the security
        system for these houses remain the same as for those in the previous street.

        Given a list of non-negative integers representing the amount of money of each house,
        determine the maximum amount of money you can rob tonight without alerting the police.

        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        pre, now = 0, 0
        if len(nums) == 1:
            return nums[0]
        for i in nums[:-1]:
            pre, now = now, max(pre + i, now)
        highest = now
        pre, now = 0, 0
        for i in nums[1:]:
            pre, now = now, max(pre + i, now)
        return max(highest, now)

    def findKthLargest(self, nums, k):
        """
        215. Kth Largest Element in an Array
        Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

        For example,
        Given [3,2,1,5,6,4] and k = 2, return 5.

        Note:
        You may assume k is always valid, 1 ? k ? array's length.
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        def partition(numbers, low, high):
            check, pivot = low, numbers[high]
            for i in xrange(low, high):
                if numbers[i] <= pivot:
                    numbers[check], numbers[i] = numbers[i], numbers[check]
                    check += 1
            numbers[check], numbers[high] = numbers[high], numbers[check]
            return check

        l, h = 0, len(nums) - 1
        k = len(nums) - k
        while True:
            index = partition(nums, l, h)
            if index == k:
                return nums[index]
            elif index > k:
                h = index - 1
            else:
                l = index + 1

    def getSkyline(self, buildings):
        """
        218. The Skyline Problem
        (see description: https://leetcode.com/problems/the-skyline-problem/description/ )
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        import heapq
        h = []
        heights = [0]
        max_height = 0
        heapq.heapify(h)
        skylines = []

        for b in buildings:
            heapq.heappush(h, (b[0], -b[2]))
            heapq.heappush(h, (b[1], b[2]))
        while h:
            index, height = heapq.heappop(h)
            if height < 0:
                # found a starting point
                height *= -1
                if height > max_height:
                    max_height = height
                    heights.append(height)
                    skylines.append([index, height])
                else:
                    heights.append(height)
            else:
                # found an ending point
                if height < max_height or heights.count(max_height) > 1:
                    heights.remove(height)
                else:
                    heights.remove(max_height)
                    max_height = sorted(list(heights))[-1]
                    skylines.append([index, max_height])

        return skylines

    def summaryRanges(self, nums):
        """
        228. Summary Ranges
        Given a sorted integer array without duplicates, return the summary of its ranges.

        Example 1:
            Input: [0,1,2,4,5,7]
            Output: ["0->2","4->5","7"]
        Example 2:
            Input: [0,2,3,4,6,8,9]
            Output: ["0","2->4","6","8->9"]

        :type nums: List[int]
        :rtype: List[str]
        """
        if not nums:
            return []
        nums.append(nums[0] - 1)
        res = []
        i = 0
        while i < len(nums) - 1:
            j = i
            while j < len(nums) - 1:
                if nums[j] + 1 == nums[j + 1]:
                    j += 1
                else:
                    if i == j:
                        s = str(nums[i])
                    else:
                        s = str(nums[i]) + '->' + str(nums[j])
                    res.append(s)
                    j += 1
                    i = j
                    break
        return res

    def productExceptSelf(self, nums):
        """
        238. Product of Array Except Self
        Given an array of n integers where n > 1, nums,
        return an array output such that output[i] is equal to
        the product of all the elements of nums except nums[i].

        Solve it without division and in O(n).

        For example, given [1,2,3,4], return [24,12,8,6].

        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return 0
        left = [1]
        right = [1]
        res = []

        for i in xrange(len(nums) - 1, -1, -1):
            right.append(nums[i] * right[-1])
        for i in xrange(len(nums)):
            left.append(nums[i] * left[-1])
            res.append(left[i] * right[len(nums) - i - 1])
        return res

    def searchMatrixII(self, matrix, target):
        """
        240. Search a 2D Matrix II
        Write an efficient algorithm that searches for a value in an m x n matrix.
        This matrix has the following properties:

        Integers in each row are sorted in ascending from left to right.
        Integers in each column are sorted in ascending from top to bottom.
        For example,

        Consider the following matrix:

        [
          [1,   4,  7, 11, 15],
          [2,   5,  8, 12, 19],
          [3,   6,  9, 16, 22],
          [10, 13, 14, 17, 24],
          [18, 21, 23, 26, 30]
        ]
        Given target = 5, return true.

        Given target = 20, return false.

        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        for row in matrix:
            if row[0] > target:
                break
            l, r = 0, len(row) - 1
            while l < r:
                mid = (l + r) / 2
                if row[mid] == target:
                    return True
                elif row[mid] < target:
                    l = mid + 1
                else:
                    r = mid
            if row[l] == target or row[r] == target:
                return True
        return False

    def shortestDistance(self, words, word1, word2):
        """
        243. Shortest Word Distance
        Given a list of words and two words word1 and word2,
        return the shortest distance between these two words in the list.

        For example,
        Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

        Given word1 = “coding”, word2 = “practice”, return 3.
        Given word1 = "makes", word2 = "coding", return 1.

        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l = len(words)
        m, n = l, l
        res = l
        for i, w in enumerate(words):
            if w == word1:
                m = i
                res = min(res, abs(m - n))
            elif w == word2:
                n = i
                res = min(res, abs(m - n))
        return res

    def shortestWordDistance(self, words, word1, word2):
        """
        245. Shortest Word Distance III
        This is a follow up of Shortest Word Distance.
        The only difference is now word1 could be the same as word2.

        Given a list of words and two words word1 and word2,
        return the shortest distance between these two words in the list.

        word1 and word2 may be the same and they represent two individual words in the list.

        For example,
        Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

        Given word1 = “makes”, word2 = “coding”, return 1.
        Given word1 = "makes", word2 = "makes", return 3.

        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if word1 == word2:
            d = [i for i, w in enumerate(words) if w == word1]
            res = len(words)
            for n in xrange(1, len(d)):
                res = min(res, d[n] - d[n - 1])
            return res
        res, d = len(words), {}
        for i, w in enumerate(words):
            if w == word1 and word2 in d:
                res = min(res, i - d[word2])
            if w == word2 and word1 in d:
                res = min(res, i - d[word1])
            if w == word1 or w == word2:
                d[w] = i
        return res

    def minCost(self, costs):
        """
        256. Paint House
        There are a row of n houses, each house can be painted
        with one of the three colors: red, blue or green.
        The cost of painting each house with a certain color is different.
        You have to paint all the houses such that no two adjacent houses have the same color.

        The cost of painting each house with a certain color is represented
        by a n x 3 cost matrix. For example, costs[0][0] is the cost of painting
        house 0 with color red; costs[1][2] is the cost of painting house 1 with color green,
        and so on... Find the minimum cost to paint all houses.

        Note:
        All costs are positive integers.

        :type costs: List[List[int]]
        :rtype: int
        """
        if len(costs) == 0:
            return 0
        dp = costs[0]

        for i in xrange(1, len(costs)):
            cur = dp[:]
            dp[0] = costs[i][0] + min(cur[1:3])
            dp[1] = costs[i][1] + min(cur[0], dp[2])
            dp[2] = costs[i][2] + min(cur[:2])
        return min(dp)

    def binaryTreePaths(self, root):
        """
        257. Binary Tree Paths
        Given a binary tree, return all root-to-leaf paths.
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        res = []
        paths = []
        self._dfs_leet257(root, [], res)
        for r in res:
            paths.append('->'.join(r))
        return paths

    def _dfs_leet257(self, node, path, res):
        if not node.left and not node.right:
            res.append(path + [str(node.val)])
            return
        if node.left:
            self._dfs_leet257(node.left, path + [str(node.val)], res)
        if node.right:
            self._dfs_leet257(node.right, path + [str(node.val)], res)

    def addOperators(self, num, target):
        """
        282. Expression Add Operators
        Given a string that contains only digits 0-9 and a target value,
        return all possibilities to add binary operators (not unary) +, -, or *
        between the digits so they evaluate to the target value.

        Examples:
        "123", 6 -> ["1+2+3", "1*2*3"]
        "232", 8 -> ["2*3+2", "2+3*2"]
        "105", 5 -> ["1*0+5","10-5"]
        "00", 0 -> ["0+0", "0-0", "0*0"]
        "3456237490", 9191 -> []
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        res = []
        for i in range(1, len(num) + 1):
            if i == 1 or (i > 1 and num[0] != "0"):  # prevent "00*" as a number
                self._dfs_leet282(num[i:], num[:i], int(num[:i]), int(num[:i]), target,
                                  res)  # this step put first number in the string
        return res

    def _dfs_leet282(self, num, temp, cur, last, target, res):
        if not num:
            if cur == target:
                res.append(temp)
            return
        for i in range(1, len(num) + 1):
            val = num[:i]
            if i == 1 or (i > 1 and num[0] != "0"):  # prevent "00*" as a number
                self._dfs_leet282(num[i:], temp + "+" + val, cur + int(val), int(val), target, res)
                self._dfs_leet282(num[i:], temp + "-" + val, cur - int(val), -int(val), target, res)
                self._dfs_leet282(num[i:], temp + "*" + val, cur - last + last * int(val), last * int(val), target, res)

    def moveZeroes(self, nums):
        """
        283. Move Zeroes
        Given an array nums, write a function to move all 0's to the end of it
        while maintaining the relative order of the non-zero elements.

        For example, given nums = [0, 1, 0, 3, 12], after calling your function,
        nums should be [1, 3, 12, 0, 0].

        Note:
        You must do this in-place without making a copy of the array.
        Minimize the total number of operations.

        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        p, q = 0, 0
        while q < len(nums):
            if nums[q] != 0:
                nums[p], nums[q] = nums[q], nums[p]
                p += 1
            q += 1

    def maxCoins(self, nums):
        """
        312. Burst Balloons
        Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it
        represented by array nums. You are asked to burst all the balloons.
        If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins.
        Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

        Find the maximum coins you can collect by bursting the balloons wisely.

        Note:
        (1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
        (2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

        Example:

        Given [3, 1, 5, 8]

        Return 167

            nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
           coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167

        :type nums: List[int]
        :rtype: int
        """
        nums = [1] + nums + [1]
        n = len(nums)
        matrix = [[0 for _ in xrange(n)] for _ in xrange(n)]
        k = 0
        while k + 2 < n:
            left, right = k, k + 2
            matrix[left][right] = nums[k] * nums[k + 1] * nums[k + 2]
            k += 1
        for i in xrange(3, n):
            k = 0
            while k + i < n:
                left, right = k, k + i
                solutions = []
                for j in xrange(left + 1, right):
                    ans = matrix[left][j] + nums[left] * nums[j] * nums[right] + matrix[j][right]
                    solutions.append(ans)
                sol = max(solutions)
                matrix[left][right] = sol
                k += 1
            i += 1
        return matrix[0][-1]

    def countSmaller(self, nums):
        """
        315. Count of Smaller Numbers After Self
        You are given an integer array nums and you have to return a new counts array.
        The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

        Example:

        Given nums = [5, 2, 6, 1]

        To the right of 5 there are 2 smaller elements (2 and 1).
        To the right of 2 there is only 1 smaller element (1).
        To the right of 6 there is 1 smaller element (1).
        To the right of 1 there is 0 smaller element.
        Return the array [2, 1, 1, 0].

        :type nums: List[int]
        :rtype: List[int]
        """
        rank = {val: i + 1 for i, val in enumerate(sorted(nums))}
        N, res = len(nums), []
        BITree = [0] * (N + 1)

        def update(i):
            while i <= N:
                BITree[i] += 1
                i += (i & -i)

        def getSum(i):
            s = 0
            while i:
                s += BITree[i]
                i -= (i & -i)
            return s

        for x in reversed(nums):
            res += getSum(rank[x] - 1),
            update(rank[x])
        return res[::-1]

    def palindromePairs(self, words):
        """
        336. Palindrome Pairs
        Given a list of unique words, find all pairs of distinct indices (i, j) in the given list,
        so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.

        Example 1:
        Given words = ["bat", "tab", "cat"]
        Return [[0, 1], [1, 0]]
        The palindromes are ["battab", "tabbat"]
        Example 2:
        Given words = ["abcd", "dcba", "lls", "s", "sssll"]
        Return [[0, 1], [1, 0], [3, 2], [2, 4]]
        The palindromes are ["dcbaabcd", "abcddcba", "slls", "llssssll"]

        :type words: List[str]
        :rtype: List[List[int]]
        """
        d = {w[::-1]: i for i, w in enumerate(words)}
        res = []
        for i, w in enumerate(words):
            for j in xrange(len(w) + 1):
                pre, post = w[:j], w[j:]
                if pre in d and i != d[pre] and post == post[::-1]:
                    res.append([i, d[pre]])
                # check j > 0 to avoid calculating itself twice
                if j > 0 and post in d and i != d[post] and pre == pre[::-1]:
                    res.append([d[post], i])
        return res

    def rob_3(self, root):
        """
        337. House Robber III
        The thief has found himself a new place for his thievery again.
        There is only one entrance to this area, called the "root." Besides the root,
        each house has one and only one parent house. After a tour, the smart thief
        realized that "all houses in this place forms a binary tree". It will automatically
        contact the police if two directly-linked houses were broken into on the same night.

        :type root: TreeNode
        :rtype: int
        """
        return max(self._dfs_leet337(root))

    def _dfs_leet337(self, root):
        if not root:
            return (0, 0)
        left, right = self._dfs_leet337(root.left), self._dfs_leet337(root.right)
        return (root.val + left[1] + right[1]), max(left) + max(right)

    def reverseVowels(self, s):
        """
        345. Reverse Vowels of a String
        Write a function that takes a string as input and reverse only the vowels of a string.
        :type s: str
        :rtype: str
        """
        d = 'aeiouAEIOU'
        l, r = 0, len(s) - 1
        s = list(s)
        while l <= r:
            if s[l] in d and s[r] in d:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
            elif s[l] not in d:
                l += 1
            elif s[r] not in d:
                r -= 1
            else:
                l += 1
                r -= 1
        return ''.join(s)

    def numberOfBoomerangs(self, points):
        """
        447. Number of Boomerangs
        Given n points in the plane that are all pairwise distinct,
        a "boomerang" is a tuple of points (i, j, k) such that
        the distance between i and j equals the distance between i and k
        (the order of the tuple matters).

        Find the number of boomerangs. You may assume that n will be
        at most 500 and coordinates of points are all in the range [-10000, 10000] (inclusive).

        :type points: List[List[int]]
        :rtype: int
        """
        res = 0
        for p in points:
            h = {}
            for q in points:
                if p != q:
                    dis = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                    h[dis] = 1 + h.get(dis, 0)
            for k in h:
                res += h[k] * (h[k] - 1)
        return res

    def findDisappearedNumbers(self, nums):
        """
        448. Find All Numbers Disappeared in an Array
        Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array),
        some elements appear twice and others appear once.

        Find all the elements of [1, n] inclusive that do not appear in this array.

        Could you do it without extra space and in O(n) runtime? You may assume
        the returned list does not count as extra space.

        Example:

        Input:
        [4,3,2,7,8,2,3,1]

        Output:
        [5,6]

        :type nums: List[int]
        :rtype: List[int]
        """
        return list(set(i for i in xrange(1, len(nums) + 1)) - set(nums))


if __name__ == '__main__':
    # debug template
    l = Lists()
    print l.summaryRanges([0, 1, 2, 4, 5, 7])
