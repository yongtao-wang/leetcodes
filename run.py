import unittest

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

    def test_two_sum_leet1(self):
        self.assertEqual(self.lp.twoSum([3, 2, 4], 6), [1, 2])

    def test_length_of_longest_substring_leet3(self):
        self.assertEqual(self.sp.lengthOfLongestSubstring('accadddfghdkg'), 5)

    def test_longest_palindrome_leet5(self):
        self.assertEqual(self.sp.longestPalindrome('daabcbaddd'), 'abcba')
        self.assertEqual(self.sp.longestPalindrome('aaaab'), 'aaaa')

    def test_reverse_integer_leet7(self):
        self.assertEqual(self.num.reverse(-3154), -4513)
        self.assertEqual(self.num.reverse(1563847412), 0)

    def test_regular_expression_leet8(self):
        self.assertTrue(self.s_op.isMatch("aa", "aa"))
        self.assertTrue(self.s_op.isMatch("aa", "a*"))
        self.assertTrue(self.s_op.isMatch("ab", ".*"))
        self.assertTrue(self.s_op.isMatch("aab", "c*a*b"))
        self.assertFalse(self.s_op.isMatch("aaa","aa"))
        self.assertFalse(self.s_op.isMatch("aa","a"))


if __name__ == '__main__':
    unittest.main()
