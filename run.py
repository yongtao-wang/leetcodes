# -*- coding: utf-8 -*-
import unittest

from algorithms import linked_list as lk
from algorithms import lists as lp
from algorithms import numbers
from algorithms import strings
from algorithms import substring_and_palindrome as sp


class Run(unittest.TestCase):
    def setUp(self):
        self.lp = lp.Lists()
        self.sp = sp.SubstringAndPalindrome()
        self.num = numbers.Numbers()
        self.s_op = strings.Strings()
        self.link = lk.LinkedList()

    def test_two_sum_leet1(self):
        self.assertEqual(self.lp.twoSum([3, 2, 4], 6), [1, 2])
        self.assertEqual(self.lp.twoSum([1, 2, 2, 3, 4], 4), [1, 2])

    def test_length_of_longest_substring_leet3(self):
        self.assertEqual(self.sp.lengthOfLongestSubstring('accadddfghdkg'), 5)

    def test_longest_palindrome_leet5(self):
        self.assertEqual(self.sp.longestPalindrome('daabcbaddd'), 'abcba')
        self.assertEqual(self.sp.longestPalindrome('aaaab'), 'aaaa')

    def test_reverse_integer_leet7(self):
        self.assertEqual(self.num.reverse(-3154), -4513)
        self.assertEqual(self.num.reverse(1563847412), 0)

    def test_array_to_integer_leet8(self):
        self.assertEqual(self.s_op.myAtoi('++1'), 0)
        self.assertEqual(self.s_op.myAtoi('    +4500'), 4500)
        self.assertEqual(self.s_op.myAtoi(' +-3'), 0)
        self.assertEqual(self.s_op.myAtoi(' --2  '), 0)
        self.assertEqual(self.s_op.myAtoi('  +123ab3'), 123)
        self.assertEqual(self.s_op.myAtoi('-2147483648'), -0x80000000)

    def test_regular_expression_leet10(self):
        self.assertTrue(self.s_op.isMatch("aa", "aa"))
        self.assertTrue(self.s_op.isMatch("aa", "a*"))
        self.assertTrue(self.s_op.isMatch("ab", ".*"))
        self.assertTrue(self.s_op.isMatch("aab", "c*a*b"))
        self.assertFalse(self.s_op.isMatch("aaa", "aa"))
        self.assertFalse(self.s_op.isMatch("aa", "a"))

    def test_integer_to_roman_leet12(self):
        self.assertEqual(self.s_op.intToRoman(3154), "MMMCLIV")
        self.assertEqual(self.s_op.intToRoman(322), "CCCXXII")
        self.assertEqual(self.s_op.intToRoman(59), "LIX")

    def test_roman_to_integer_leet13(self):
        self.assertEqual(self.s_op.romanToInt('XXXII'), 32)
        self.assertEqual(self.s_op.romanToInt('CCXXXIV'), 234)
        self.assertEqual(self.s_op.romanToInt('XL'), 40)

    def test_longest_common_prefix_leet14(self):
        l1 = ['abcfff', 'abcd', 'abcqwerabc', 'abc']
        l2 = ['qweraaaa', 'qwert', 'qtellars']
        self.assertEqual(self.s_op.longestCommonPrefix(l1), 'abc')
        self.assertEqual(self.s_op.longestCommonPrefix(l2), 'q')

    def test_3_sum_leet15(self):
        nums = [-1, 0, 1, 2, -1, -4]
        solution = [[-1, 0, 1], [-1, -1, 2]]
        self.assertEqual(sorted(self.lp.threeSum(nums)), sorted(solution))

    def test_digit_letter_combination_leet17(self):
        digits = '233'
        expected = ['add', 'ade', 'adf', 'aed', 'aee', 'aef', 'afd', 'afe', 'aff',
                    'bdd', 'bde', 'bdf', 'bed', 'bee', 'bef', 'bfd', 'bfe', 'bff',
                    'cdd', 'cde', 'cdf', 'ced', 'cee', 'cef', 'cfd', 'cfe', 'cff']
        self.assertEqual(self.lp.letterCombinations(digits), expected)
        self.assertIsNone(self.lp.letterCombinations('419'))

    def test_4_sum_leet18(self):
        array = [1, 0, -1, 0, -2, 2]
        target = 0
        solution = [[-1, 0, 0, 1], [-2, -1, 1, 2], [-2, 0, 0, 2]]
        self.assertEqual(sorted(self.lp.fourSum(array, target)), sorted(solution))

    def test_remove_nth_node_leet19(self):
        n1 = lk.ListNode(1)
        n2 = lk.ListNode(2)
        n3 = lk.ListNode(3)
        n4 = lk.ListNode(4)
        n5 = lk.ListNode(5)
        self.assertEqual(self.link.removeNthFromEnd(n1, 1), None)
        n1.next = n2
        self.assertEqual(self.link.removeNthFromEnd(n1, 2), n2)
        n1.next = n2
        n2.next = n3
        n3.next = n4
        n4.next = n5
        self.link.removeNthFromEnd(n1, 3)
        values = ''
        h = n1
        while h:
            values += str(h.val)
            h = h.next
        self.assertEqual(values, '1245')

    def test_valid_parentheses_leet20(self):
        p1 = '(({}[{}][()])[])'
        p2 = ']]]'
        p3 = '[[]}'
        p4 = '([)]'
        self.assertTrue(self.s_op.isValidParentheses(p1))
        self.assertFalse(self.s_op.isValidParentheses(p2))
        self.assertFalse(self.s_op.isValidParentheses(p3))
        self.assertFalse(self.s_op.isValidParentheses(p4))

    def test_generate_parentheses_leet22(self):
        answer = ['((()))', '(()())', '(())()', '()(())', '()()()']
        self.assertEqual(sorted(self.s_op.generateParenthesis(3)), sorted(answer))

    def test_swap_nodes_in_pairs_leet24(self):
        n1 = lk.ListNode(1)
        n2 = lk.ListNode(2)
        n3 = lk.ListNode(3)
        n4 = lk.ListNode(4)
        self.assertEqual(self.link.swapPairs(None), None)
        self.assertEqual(self.link.swapPairs(n1), n1)
        n1.next = n2
        n2.next = n3
        n3.next = n4
        head = self.link.swapPairs(n1)
        answer = ''
        while head:
            answer += str(head.val)
        self.assertEqual(answer, '2143')


if __name__ == '__main__':
    unittest.main()
