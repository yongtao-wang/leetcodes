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

if __name__ == '__main__':
    unittest.main()
