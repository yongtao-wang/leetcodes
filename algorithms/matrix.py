class Matrix(object):
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
        row, col = 0, len(matrix[0]) - 1
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False

    def longestIncreasingPath(self, matrix):
        """
        329. Longest Increasing Path in a Matrix
        Given an integer matrix, find the length of the longest increasing path.

        From each cell, you can either move to four directions: left, right, up or down.
        You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

        Example 1:

        nums = [
          [9,9,4],
          [6,6,8],
          [2,1,1]
        ]
        Return 4
        The longest increasing path is [1, 2, 6, 9].

        Example 2:

        nums = [
          [3,4,5],
          [3,2,6],
          [2,2,1]
        ]
        Return 4
        The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.

        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in xrange(n)] for _ in xrange(m)]

        def dfs(grid, i, j):
            if dp[i][j] == 0:
                val = grid[i][j]
                dp[i][j] = 1 + max(
                    dfs(grid, i - 1, j) if i > 0 and val < grid[i - 1][j] else 0,
                    dfs(grid, i + 1, j) if i < m - 1 and val < grid[i + 1][j] else 0,
                    dfs(grid, i, j - 1) if j > 0 and val < grid[i][j - 1] else 0,
                    dfs(grid, i, j + 1) if j < n - 1 and val < grid[i][j + 1] else 0
                )
            return dp[i][j]

        distances = []
        for i in xrange(m):
            for j in xrange(n):
                distances.append(dfs(matrix, i, j))
        return max(distances)
