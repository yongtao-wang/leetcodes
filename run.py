import unittest

from algorithms import list_processing as lp
from algorithms import substring_and_palindrome as sp


class Run(unittest.TestCase):
    def setUp(self):
        self.lp = lp.ListProcessing()
        self.sp = sp.SubstringAndPalindrome()

    def test_two_sum_leet1_leet3(self):
        self.assertEqual(self.lp.twoSum([3, 2, 4], 6), [1, 2])

    def test_length_of_longest_substring(self):
        self.assertEqual(self.sp.lengthOfLongestSubstring('accadddfghdkg'), 5)

    def test_longest_palindrome_leet5(self):
        self.assertEqual(self.sp.longestPalindrome('daabcbaddd'), 'abcba')


if __name__ == '__main__':
    unittest.main()
