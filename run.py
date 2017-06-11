import unittest

from algorithms import lists as lp
from algorithms import numbers
from algorithms import substring_and_palindrome as sp


class Run(unittest.TestCase):
    def setUp(self):
        self.lp = lp.Lists()
        self.sp = sp.SubstringAndPalindrome()
        self.num = numbers.Numbers()

    def test_two_sum_leet1(self):
        self.assertEqual(self.lp.twoSum([3, 2, 4], 6), [1, 2])

    def test_length_of_longest_substring_leet3(self):
        self.assertEqual(self.sp.lengthOfLongestSubstring('accadddfghdkg'), 5)

    def test_longest_palindrome_leet5(self):
        self.assertEqual(self.sp.longestPalindrome('daabcbaddd'), 'abcba')

    def test_reverse_integer_leet7(self):
        self.assertEqual(self.num.reverse(-3154), -4513)

    def test_reverse_integer_overflow_leet7(self):
        self.assertEqual(self.num.reverse(1563847412), 0)


if __name__ == '__main__':
    unittest.main()
