# -*- coding: utf-8 -*-


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Tree(object):
    def recoverTree(self, root):
        """
        99. Recover Binary Search Tree
        Two elements of a binary search tree (BST) are swapped by mistake.

        Recover the tree without changing its structure.
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        self.pre = None
        self.n1 = None
        self.n2 = None
        self._traversal(root)
        if self.n1 and self.n2:
            self.n1.val, self.n2.val = self.n2.val, self.n1.val

    def _traversal(self, node):
        if not node:
            return
        self._traversal(node.left)
        if self.pre:
            if self.pre.val > node.val:
                if not self.n1:
                    self.n1 = self.pre
                    self.n2 = node
                else:
                    self.n2 = node
        self.pre = node
        self._traversal(node.right)

    def isSymmetric(self, root):
        """
        101. Symmetric Tree
        Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

        :type root: TreeNode
        :rtype: bool
        """
        '''
        another solution using recursion:
        
        def isSymmetric(self, root):
            """
            :type root: TreeNode
            :rtype: bool
            """
            def sym(left, right):
                if left and right:
                    return left.val == right.val and sym(left.left, right.right) and sym(left.right, right.left)
                else:
                    return left == right
            if not root:
                return True
            return sym(root.left, root.right)
        '''
        if not root:
            return True
        queue = [[root.left, root.right]]

        while queue:
            left, right = queue.pop(0)
            if not left and not right:
                continue
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            queue.append([left.right, right.left])
            queue.append([left.left, right.right])
        return True

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

    def longestConsecutive(self, root):
        """
        298. Binary Tree Longest Consecutive Sequence
        Given a binary tree, find the length of the longest consecutive sequence path.

        The path refers to any sequence of nodes from some starting node to any node
        in the tree along the parent-child connections. The longest consecutive path
        need to be from parent to child (cannot be the reverse).

        :type root: TreeNode
        :rtype: int
        """
        '''
        stack solution provided
        --------------------------------------------
        def longestConsecutive(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        max_len = 0
        if not root:
            return 0
        stack = [(root, 1)]
        while stack:
            node, c = stack.pop()
            if node.left:
                stack.append((node.left, c + 1 if node.val + 1 == node.left.val else 1))
            if node.right:
                stack.append((node.right, c + 1 if node.val + 1 == node.right.val else 1))
            max_len = max(max_len, c)
        return max_len
        --------------------------------------------
        '''
        if not root:
            return 0
        return max(self._find(root.left, 1, root.val), self._find(root.right, 1, root.val))

    def _find(self, node, n, val):
        if not node:
            return n
        n = n + 1 if val + 1 == node.val else 1
        left = self._find(node.left, n, node.val)
        right = self._find(node.right, n, node.val)
        return max(left, right, n)


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


class Codec:
    """
    297. Serialize and Deserialize Binary Tree
    Serialization is the process of converting a data structure or object into
    a sequence of bits so that it can be stored in a file or memory buffer,
    or transmitted across a network connection link to be reconstructed later in
    the same or another computer environment.

    Design an algorithm to serialize and deserialize a binary tree.
    There is no restriction on how your serialization/deserialization algorithm should work.
    You just need to ensure that a binary tree can be serialized to a string and this string
    can be deserialized to the original tree structure.

    """
    '''Using pre-order traversal'''

    # Your Codec object will be instantiated and called as such:
    # codec = Codec()
    # codec.deserialize(codec.serialize(root))

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        path = []
        self._traversal(root, path)
        return ','.join(path)

    def _traversal(self, current_node, path):
        if current_node:
            path.append(str(current_node.val))
            self._traversal(current_node.left, path)
            self._traversal(current_node.right, path)
        else:
            path.append('#')

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        return self._rebuild(data.split(','))

    def _rebuild(self, path):
        if not path:
            return None
        v = path.pop(0).strip()
        if v == '#':
            return None
        n = TreeNode(int(v))
        n.left = self._rebuild(path)
        n.right = self._rebuild(path)
        return n


class TrieNode(object):
    def __init__(self):
        self.is_word = False
        self.children = {}


class Trie(object):
    """
    208. Implement Trie (Prefix Tree)
    Implement a trie with insert, search, and startsWith methods.

    Note:
    You may assume that all inputs are consist of lowercase letters a-z.

    """
    '''Trie的相关知识可以参考http://blog.csdn.net/v_july_v/article/details/6897097'''

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True if node else False

    def isSubtree(self, s, t):
        """
        572. Subtree of Another Tree
        Given two non-empty binary trees s and t, check whether tree t has exactly
        the same structure and node values with a subtree of s. A subtree of s is a
        tree consists of a node in s and all of this node's descendants.
        The tree s could also be considered as a subtree of itself.

        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        return self._serialize(t) in self._serialize(s)

    def _serialize(self, root):
        if not root:
            return 'N'
        return '(%s),(%s),(%s)' % (root.val, self._serialize(root.left), self._serialize(root.right))

if __name__ == '__main__':
    c = Codec()
    node = c.deserialize('1, #, #')
    print node.val
