# -*- coding: utf-8 -*-
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
        Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

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


if __name__ == '__main__':
    # debug template
    l = Lists()
    print l.longestValidParentheses('()')
