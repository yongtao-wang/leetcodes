# -*- coding: utf-8 -*-

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class LinkedList(object):

    def addTwoNumbers(self, l1, l2):
        """
        # 2
        You are given two non-empty linked lists representing two non-negative integers.
        The digits are stored in reverse order and each of their nodes contain a single digit.
        Add the two numbers and return it as a linked list.

        You may assume the two numbers do not contain any leading zero, except the number 0 itself.
        :param l1:
        :param l2:
        :return:
        """
        if not l1:
            return l2
        if not l2:
            return l1

        head = ListNode(0)
        head.next = l1
        res = head

        addition = False
        while l1 and l2:
            head.next.val = l1.val + l2.val
            if addition:
                head.next.val += 1
            if head.next.val > 9:
                head.next.val -= 10
                addition = True
            else:
                addition = False

            head = head.next
            l1 = l1.next
            l2 = l2.next

        if l2:
            head.next = l2
            l1 = l2

        while l1:
            head.next.val = l1.val
            if addition:
                head.next.val += 1
            if head.next.val > 9:
                head.next.val -= 10
                addition = True
            else:
                addition = False
            head = head.next
            l1 = l1.next

        if addition:
            head.next = ListNode(1)

        return res.next

    def findMedianSortedArrays(self, nums1, nums2):
        """
        # 4 Median of two sorted arrays
        There are two sorted arrays nums1 and nums2 of size m and n respectively.

        Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        pass

    def removeNthFromEnd(self, head, n):
        """
        # 19. Remove Nth Node From End of List
        Given a linked list, remove the nth node from the end of list and return its head.
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return None
        p1 = head
        p2 = head
        for _ in xrange(n):
            if not p2:
                return head
            p2 = p2.next
        # meaning head is to be removed
        if not p2:
            return head.next
        while p2.next:
            p2 = p2.next
            p1 = p1.next
        p1.next = p1.next.next
        return head

    def mergeTwoLists(self, l1, l2):
        """
        # 21. Merge Two Sorted Lists

        Merge two sorted linked lists and return it as a new list.
        The new list should be made by splicing together the nodes of the first two lists.

        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1:
            return l2
        if not l2:
            return l1
        head = ListNode(0)
        res = head
        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next

        head.next = l1 or l2  # PYTHONIC!
        return res.next

    def swapPairs(self, head):
        """
        #24. Swap Nodes in Pairs

        Given a linked list, swap every two adjacent nodes and return its head.

        For example,
        Given 1->2->3->4, you should return the list as 2->1->4->3.

        Your algorithm should use only constant space.
        You may not modify the values in the list, only nodes itself can be changed.

        :type head: ListNode, None
        :rtype: ListNode
        """
        '''一系列需要用到pre-node的样板'''
        res = ListNode(0)
        res.next = head
        pre = res
        while head:
            if not head.next:
                break
            current = head
            follow = head.next
            rest = follow.next
            head = follow
            head.next = current
            head.next.next = rest
            pre.next = head
            pre = head.next
            head = head.next.next
        return res.next


if __name__ == '__main__':
    # debug template
    link = LinkedList()
    n1 = ListNode(1)
    n2 = ListNode(2)
    n3 = ListNode(3)
    n4 = ListNode(4)
    n1.next = n2
    n2.next = n3
    n3.next = n4
