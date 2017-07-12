# -*- coding: utf-8 -*-
import unittest

from algorithms import linked_list as lk
from algorithms import lists as lp
from algorithms import numbers
from algorithms import strings
from algorithms import substring_and_palindrome as sp
from algorithms import dynamic_programming as dp


class Run(unittest.TestCase):
    def setUp(self):
        self.lp = lp.Lists()
        self.sp = sp.SubstringAndPalindrome()
        self.num = numbers.Numbers()
        self.s_op = strings.Strings()
        self.link = lk.LinkedList()
        self.dp = dp.DynamicProgramming()

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
            head = head.next
        self.assertEqual(answer, '2143')

    def test_remove_duplicates_from_sorted_array_leet26(self):
        sorted_1 = [1, 1, 2, 3, 4, 4, 5, 5, 5]
        sorted_2 = []
        sorted_3 = [1, 1, 1, 1]
        self.assertEqual(self.lp.removeDuplicates(sorted_1), 5)
        self.assertEqual(self.lp.removeDuplicates(sorted_2), 0)
        self.assertEqual(self.lp.removeDuplicates(sorted_3), 1)

    def test_remove_element_leet27(self):
        l1 = [3, 2, 2, 3]
        l2 = [5, 5, 6, 5]
        self.assertEqual(sorted(l1[:self.lp.removeElement(l1, 3)]), [2, 2])
        self.assertEqual(sorted(l2[:self.lp.removeElement(l2, 3)]), sorted(l2))
        self.assertEqual(self.lp.removeElement([], 8), 0)

    def test_next_permutation_leet31(self):
        l1 = [1, 2]
        l2 = [1, 5, 3]
        l3 = [5, 3, 1]
        l4 = [2, 5, 3, 1]
        self.lp.nextPermutation(l1)
        self.lp.nextPermutation(l2)
        self.lp.nextPermutation(l3)
        self.lp.nextPermutation(l4)
        self.assertEqual(l1, [2, 1])
        self.assertEqual(l2, [3, 1, 5])
        self.assertEqual(l3, [1, 3, 5])
        self.assertEqual(l4, [3, 1, 2, 5])

    def test_search_in_rotated_sorted_array_leet33(self):
        l1 = []
        l2 = [4, 5, 6, 7, 0, 1, 2]
        l3 = [5, 1, 2, 3, 4]
        self.assertEqual(self.lp.search(l1, 3), -1)
        self.assertEqual(self.lp.search(l2, 6), 2)
        self.assertEqual(self.lp.search(l2, 9), -1)
        self.assertEqual(self.lp.search(l3, 4), 4)

    def test_search_for_a_range_leet34(self):
        l1 = [5, 7, 7, 8, 8, 10]
        self.assertEqual(self.lp.searchRange(l1, 8), [3, 4])
        self.assertEqual(self.lp.searchRange(l1, 6), [-1, -1])

    def test_search_insert_pos_leet35(self):
        self.assertEqual(self.lp.searchInsert([1, 3, 5, 6], 5), 2)
        self.assertEqual(self.lp.searchInsert([1, 3, 5, 6], 2), 1)
        self.assertEqual(self.lp.searchInsert([1, 3, 5, 6], 7), 4)
        self.assertEqual(self.lp.searchInsert([1, 3, 5, 6], 0), 0)
        self.assertEqual(self.lp.searchInsert([1, 3, 5, 6, 9], 6), 3)
        self.assertEqual(self.lp.searchInsert([], 88), 0)

    def test_is_valid_sudoku_leet36(self):
        board = [".87654321", "2........", "3........",
                 "4........", "5........", "6........",
                 "7........", "8........", "9........"]
        self.assertTrue(self.lp.isValidSudoku(board=board))
        self.assertFalse(self.lp.isValidSudoku(board=[]))

    def test_count_and_say_leet38(self):
        self.assertEqual(self.s_op.countAndSay(1), '1')
        self.assertEqual(self.s_op.countAndSay(4), '1211')
        self.assertEqual(self.s_op.countAndSay(6), '312211')
        self.assertEqual(self.s_op.countAndSay(10), '13211311123113112211')
        self.assertEqual(self.s_op.countAndSay(12), '3113112221232112111312211312113211')

    def test_combination_sum_leet39(self):
        self.assertEqual(sorted(self.lp.combinationSum([2, 3, 6, 7], 7)),
                         sorted([[2, 2, 3], [7]]))

    def test_combination_sum_2_leet40(self):
        self.assertEqual(sorted(self.lp.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8)),
                         sorted([[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]))

    def test_permutation_leet46(self):
        self.assertEqual(sorted(self.lp.permute([1, 2, 3])),
                         sorted([[1, 2, 3], [1, 3, 2], [2, 1, 3],
                                 [2, 3, 1], [3, 1, 2], [3, 2, 1]]))

    def test_permutation_2_leet47(self):
        self.assertEqual(sorted(self.lp.permuteUnique([1, 1, 2])),
                         sorted([[1, 1, 2], [1, 2, 1], [2, 1, 1]]))

    def test_rotate_image_leet48(self):
        img_1 = [[1, 2], [3, 4]]
        img_2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(self.lp.rotate(img_1), [[3, 1], [4, 2]])
        self.assertEqual(self.lp.rotate(img_2), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    def test_group_anagrams_leet49(self):
        anagrams = ["eat", "tea", "tan", "ate", "nat", "bat"]
        self.assertEqual(sorted(self.lp.groupAnagrams(anagrams)),
                         sorted([["tan", "nat"], ["bat"], ["eat", "tea", "ate"]]))

    def test_pow_leet50(self):
        self.assertEqual("{:.5f}".format(self.num.myPow(8.88023, 3)), str(700.28148))
        self.assertEqual("{:.5f}".format(self.num.myPow(8.88023, -3)), str(0.00143))

    def test_maximum_subarray_leet51(self):
        self.assertEqual(self.lp.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)

    def test_spiral_order_leet54(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        self.assertEqual(self.lp.spiralOrder(matrix), [1, 2, 3, 6, 9, 8, 7, 4, 5])

    def test_jump_game(self):
        self.assertTrue(self.dp.canJump([0]))
        self.assertTrue(self.dp.canJump([3, 2, 1, 1, 4]))
        self.assertFalse(self.dp.canJump([3, 2, 1, 0, 4]))

    def test_coin_change_leet322(self):
        self.assertEqual(self.dp.coinChange([1, 2, 5], 11), 3)
        self.assertEqual(self.dp.coinChange([2], 3), -1)


if __name__ == '__main__':
    unittest.main()
