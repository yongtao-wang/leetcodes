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
        '''优化后的速度能达到98.04%，未优化时为68.44%'''
        res = []
        nums.sort()  # 如果不想改变list顺序，可以使用sorted(nums)
        for i in xrange(len(nums) - 2):

            # some optimization
            if nums[i] > 0:
                break

            if i > 0 and nums[i] == nums[i-1]:
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
        for i in xrange(len(nums)-3):
            if nums[i] > target / 4.0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            target2 = target - nums[i]
            for j in xrange(i+1, len(nums)-2):
                if nums[j] > target2 / 3.0:
                    break
                if j > i+1 and nums[j] == nums[j-1]:
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
                        while l < r and nums[l] == nums[l+1]:
                            l += 1
                        while l < r and nums[r] == nums[r-1]:
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
                return start+1 if nums[start] < target else start
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
                return search(list_num[mid+1:end+1], mid, end)
            if nums[mid] > target:
                return search(list_num[start:mid], start, mid)
        return search(nums, 0, len(nums)-1)

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
        for i in xrange(len(nums)-1, -1, -1):
            if i == 0:  # meaning the list is at maximum permutation
                nums[:] = nums[::-1]
                return
            if nums[i-1] < nums[i]:
                # search from rightmost to find the first one larger than nums[i-1]
                for k in xrange(len(nums)-1, i-1, -1):
                    if nums[k] > nums[i-1]:
                        nums[i-1], nums[k] = nums[k], nums[i-1]
                        break
                # reverse from i
                nums[i:] = nums[i:][::-1]
                return

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
                    right = mid -1
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
                if (i/3, j/3) not in cube_dict:
                    cube_dict[(i/3, j/3)] = set()

                if board[i][j] == '.':
                    continue
                if board[i][j] in row_dict[i] or board[i][j] in col_dict[j] \
                        or board[i][j] in cube_dict[(i/3, j/3)]:
                    return False
                row_dict[i].add(board[i][j])
                col_dict[j].add(board[i][j])
                cube_dict[(i/3, j/3)].add(board[i][j])
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
                dfs(nums, target-nums[i], i, path+[nums[i]], res)
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
                if i != 0 and n[i] == n[i-1]:
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


if __name__ == '__main__':
    # debug template
    l = Lists()
    print l.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
