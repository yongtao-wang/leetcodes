# -*- coding: utf-8 -*-


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Tree(object):
    def lowestCommonAncestorBST(self, root, p, q):
        """
        235. Lowest Common Ancestor of a Binary Search Tree
        Given a binary search tree (BST), find the lowest common
        ancestor (LCA) of two given nodes in the BST.

        According to the definition of LCA on Wikipedia:
        “The lowest common ancestor is defined between two nodes v
        and w as the lowest node in T that has both v and w as descendants
        (where we allow a node to be a descendant of itself).”
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        while root:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root

    def lowestCommonAncestor(self, root, p, q):
        """
        236. Lowest Common Ancestor of a Binary Tree
        Given a binary tree, find the lowest common ancestor (LCA)
        of two given nodes in the tree.
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if (not root) or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        return left or right


# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class BSTIterator(object):
    """
    173. Binary Search Tree Iterator
    Implement an iterator over a binary search tree (BST).
    Your iterator will be initialized with the root node of a BST.

    Calling next() will return the next smallest number in the BST.

    Note: next() and hasNext() should run in average O(1) time and
    uses O(h) memory, where h is the height of the tree.
    """

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0

    def next(self):
        """
        :rtype: int
        """
        current = self.stack.pop()
        n = current.right
        while n:
            self.stack.append(n)
            n = n.left
        return current.val


        # Your BSTIterator will be called like this:
        # i, v = BSTIterator(root), []
        # while i.hasNext(): v.append(i.next())
