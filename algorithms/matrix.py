class Matrix(object):
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

    def islandPerimeter(self, grid):
        """
        463. Island Perimeter

        You are given a map in form of a two-dimensional integer grid where 1 represents land
        and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally).
        The grid is completely surrounded by water, and there is exactly one island
        (i.e., one or more connected land cells). The island doesn't have "lakes"
        (water inside that isn't connected to the water around the island).
        One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100.
        Determine the perimeter of the island.

        Example:

        [[0,1,0,0],
         [1,1,1,0],
         [0,1,0,0],
         [1,1,0,0]]

        Answer: 16

        :type grid: List[List[int]]
        :rtype: int
        """
        res = 0
        if not grid or not grid[0]:
            return res
        m, n = len(grid), len(grid[0])
        for i in xrange(m):
            for j in xrange(n):
                if grid[i][j] == 1:
                    peri = 4
                    if i > 0 and grid[i - 1][j] == 1:
                        peri -= 1
                    if i + 1 < m and grid[i + 1][j] == 1:
                        peri -= 1
                    if j > 0 and grid[i][j - 1] == 1:
                        peri -= 1
                    if j + 1 < n and grid[i][j + 1] == 1:
                        peri -= 1
                    res += peri
        return res

    def hasPath(self, maze, start, destination):
        """
        490. The Maze

        There is a ball in a maze with empty spaces and walls.
        The ball can go through empty spaces by rolling up, down,
        left or right, but it won't stop rolling until hitting a wall.
        When the ball stops, it could choose the next direction.

        Given the ball's start position, the destination and the maze,
        determine whether the ball could stop at the destination.

        The maze is represented by a binary 2D array. 1 means the wall
        and 0 means the empty space. You may assume that the borders of
        the maze are all walls. The start and destination coordinates are
        represented by row and column indexes.

        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        if not maze or not start or not destination:
            return False
        if start == destination:
            return True
        visited = set()
        self.found = False

        def dfs(cur):
            i, j = cur
            if cur in visited:
                return
            if cur == tuple(destination):
                self.found = True
                return
            visited.add(cur)

            left = j
            while left > 0 and maze[i][left - 1] == 0:
                left -= 1
            right = j
            while right < len(maze[0]) - 1 and maze[i][right + 1] == 0:
                right += 1
            top = i
            while top > 0 and maze[top - 1][j] == 0:
                top -= 1
            bottom = i
            while bottom < len(maze) - 1 and maze[bottom + 1][j] == 0:
                bottom += 1

            dfs((i, left))
            dfs((i, right))
            dfs((top, j))
            dfs((bottom, j))

        dfs(tuple(start))
        return self.found


if __name__ == '__main__':
    m = Matrix()
    matrix = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    print m.hasPath(matrix, [0, 4], [4, 4])
