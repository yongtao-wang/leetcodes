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
        self.lst = lp.Lists()
        self.sp = sp.SubstringAndPalindrome()
        self.num = numbers.Numbers()
        self.str = strings.Strings()
        self.link = lk.LinkedList()
        self.dp = dp.DynamicProgramming()

    def test_two_sum_leet1(self):
        self.assertEqual(self.lst.twoSum([3, 2, 4], 6), [1, 2])
        self.assertEqual(self.lst.twoSum([1, 2, 2, 3, 4], 4), [1, 2])

    def test_length_of_longest_substring_leet3(self):
        self.assertEqual(self.sp.lengthOfLongestSubstring('accadddfghdkg'), 5)

    def test_longest_palindrome_leet5(self):
        self.assertEqual(self.sp.longestPalindrome('daabcbaddd'), 'abcba')
        self.assertEqual(self.sp.longestPalindrome('aaaab'), 'aaaa')

    def test_reverse_integer_leet7(self):
        self.assertEqual(self.num.reverse(-3154), -4513)
        self.assertEqual(self.num.reverse(1563847412), 0)

    def test_array_to_integer_leet8(self):
        self.assertEqual(self.str.myAtoi('++1'), 0)
        self.assertEqual(self.str.myAtoi('    +4500'), 4500)
        self.assertEqual(self.str.myAtoi(' +-3'), 0)
        self.assertEqual(self.str.myAtoi(' --2  '), 0)
        self.assertEqual(self.str.myAtoi('  +123ab3'), 123)
        self.assertEqual(self.str.myAtoi('-2147483648'), -0x80000000)

    def test_regular_expression_leet10(self):
        self.assertTrue(self.str.isMatch("aa", "aa"))
        self.assertTrue(self.str.isMatch("aa", "a*"))
        self.assertTrue(self.str.isMatch("ab", ".*"))
        self.assertTrue(self.str.isMatch("aab", "c*a*b"))
        self.assertFalse(self.str.isMatch("aaa", "aa"))
        self.assertFalse(self.str.isMatch("aa", "a"))

    def test_integer_to_roman_leet12(self):
        self.assertEqual(self.str.intToRoman(3154), "MMMCLIV")
        self.assertEqual(self.str.intToRoman(322), "CCCXXII")
        self.assertEqual(self.str.intToRoman(59), "LIX")

    def test_roman_to_integer_leet13(self):
        self.assertEqual(self.str.romanToInt('XXXII'), 32)
        self.assertEqual(self.str.romanToInt('CCXXXIV'), 234)
        self.assertEqual(self.str.romanToInt('XL'), 40)

    def test_longest_common_prefix_leet14(self):
        l1 = ['abcfff', 'abcd', 'abcqwerabc', 'abc']
        l2 = ['qweraaaa', 'qwert', 'qtellars']
        self.assertEqual(self.str.longestCommonPrefix(l1), 'abc')
        self.assertEqual(self.str.longestCommonPrefix(l2), 'q')

    def test_3_sum_leet15(self):
        nums = [-1, 0, 1, 2, -1, -4]
        solution = [[-1, 0, 1], [-1, -1, 2]]
        self.assertEqual(sorted(self.lst.threeSum(nums)), sorted(solution))

    def test_digit_letter_combination_leet17(self):
        digits = '233'
        expected = ['add', 'ade', 'adf', 'aed', 'aee', 'aef', 'afd', 'afe', 'aff',
                    'bdd', 'bde', 'bdf', 'bed', 'bee', 'bef', 'bfd', 'bfe', 'bff',
                    'cdd', 'cde', 'cdf', 'ced', 'cee', 'cef', 'cfd', 'cfe', 'cff']
        self.assertEqual(self.lst.letterCombinations(digits), expected)
        self.assertIsNone(self.lst.letterCombinations('419'))

    def test_4_sum_leet18(self):
        array = [1, 0, -1, 0, -2, 2]
        target = 0
        solution = [[-1, 0, 0, 1], [-2, -1, 1, 2], [-2, 0, 0, 2]]
        self.assertEqual(sorted(self.lst.fourSum(array, target)), sorted(solution))

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
        self.assertTrue(self.str.isValidParentheses(p1))
        self.assertFalse(self.str.isValidParentheses(p2))
        self.assertFalse(self.str.isValidParentheses(p3))
        self.assertFalse(self.str.isValidParentheses(p4))

    def test_generate_parentheses_leet22(self):
        answer = ['((()))', '(()())', '(())()', '()(())', '()()()']
        self.assertEqual(sorted(self.str.generateParenthesis(3)), sorted(answer))

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
        self.assertEqual(self.lst.removeDuplicates(sorted_1), 5)
        self.assertEqual(self.lst.removeDuplicates(sorted_2), 0)
        self.assertEqual(self.lst.removeDuplicates(sorted_3), 1)

    def test_remove_element_leet27(self):
        l1 = [3, 2, 2, 3]
        l2 = [5, 5, 6, 5]
        self.assertEqual(sorted(l1[:self.lst.removeElement(l1, 3)]), [2, 2])
        self.assertEqual(sorted(l2[:self.lst.removeElement(l2, 3)]), sorted(l2))
        self.assertEqual(self.lst.removeElement([], 8), 0)

    def test_next_permutation_leet31(self):
        l1 = [1, 2]
        l2 = [1, 5, 3]
        l3 = [5, 3, 1]
        l4 = [2, 5, 3, 1]
        self.lst.nextPermutation(l1)
        self.lst.nextPermutation(l2)
        self.lst.nextPermutation(l3)
        self.lst.nextPermutation(l4)
        self.assertEqual(l1, [2, 1])
        self.assertEqual(l2, [3, 1, 5])
        self.assertEqual(l3, [1, 3, 5])
        self.assertEqual(l4, [3, 1, 2, 5])

    def test_longest_valid_parenthesis_leet32(self):
        self.assertEqual(self.lst.longestValidParentheses('()'), 2)
        self.assertEqual(self.lst.longestValidParentheses('()(('), 2)
        self.assertEqual(self.lst.longestValidParentheses('()(()'), 2)
        self.assertEqual(self.lst.longestValidParentheses('()(())()'), 8)

    def test_search_in_rotated_sorted_array_leet33(self):
        l1 = []
        l2 = [4, 5, 6, 7, 0, 1, 2]
        l3 = [5, 1, 2, 3, 4]
        self.assertEqual(self.lst.search(l1, 3), -1)
        self.assertEqual(self.lst.search(l2, 6), 2)
        self.assertEqual(self.lst.search(l2, 9), -1)
        self.assertEqual(self.lst.search(l3, 4), 4)

    def test_search_for_a_range_leet34(self):
        l1 = [5, 7, 7, 8, 8, 10]
        self.assertEqual(self.lst.searchRange(l1, 8), [3, 4])
        self.assertEqual(self.lst.searchRange(l1, 6), [-1, -1])

    def test_search_insert_pos_leet35(self):
        self.assertEqual(self.lst.searchInsert([1, 3, 5, 6], 5), 2)
        self.assertEqual(self.lst.searchInsert([1, 3, 5, 6], 2), 1)
        self.assertEqual(self.lst.searchInsert([1, 3, 5, 6], 7), 4)
        self.assertEqual(self.lst.searchInsert([1, 3, 5, 6], 0), 0)
        self.assertEqual(self.lst.searchInsert([1, 3, 5, 6, 9], 6), 3)
        self.assertEqual(self.lst.searchInsert([], 88), 0)

    def test_is_valid_sudoku_leet36(self):
        board = [".87654321", "2........", "3........",
                 "4........", "5........", "6........",
                 "7........", "8........", "9........"]
        self.assertTrue(self.lst.isValidSudoku(board=board))
        self.assertFalse(self.lst.isValidSudoku(board=[]))

    def test_count_and_say_leet38(self):
        self.assertEqual(self.str.countAndSay(1), '1')
        self.assertEqual(self.str.countAndSay(4), '1211')
        self.assertEqual(self.str.countAndSay(6), '312211')
        self.assertEqual(self.str.countAndSay(10), '13211311123113112211')
        self.assertEqual(self.str.countAndSay(12), '3113112221232112111312211312113211')

    def test_combination_sum_leet39(self):
        self.assertEqual(sorted(self.lst.combinationSum([2, 3, 6, 7], 7)),
                         sorted([[2, 2, 3], [7]]))

    def test_combination_sum_2_leet40(self):
        self.assertEqual(sorted(self.lst.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8)),
                         sorted([[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]))

    def test_permutation_leet46(self):
        self.assertEqual(sorted(self.lst.permute([1, 2, 3])),
                         sorted([[1, 2, 3], [1, 3, 2], [2, 1, 3],
                                 [2, 3, 1], [3, 1, 2], [3, 2, 1]]))

    def test_permutation_2_leet47(self):
        self.assertEqual(sorted(self.lst.permuteUnique([1, 1, 2])),
                         sorted([[1, 1, 2], [1, 2, 1], [2, 1, 1]]))

    def test_rotate_image_leet48(self):
        img_1 = [[1, 2], [3, 4]]
        img_2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(self.lst.rotate(img_1), [[3, 1], [4, 2]])
        self.assertEqual(self.lst.rotate(img_2), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    def test_group_anagrams_leet49(self):
        anagrams = ["eat", "tea", "tan", "ate", "nat", "bat"]
        self.assertEqual(sorted(self.lst.groupAnagrams(anagrams)),
                         sorted([["tan", "nat"], ["bat"], ["eat", "tea", "ate"]]))

    def test_pow_leet50(self):
        self.assertEqual("{:.5f}".format(self.num.myPow(8.88023, 3)), str(700.28148))
        self.assertEqual("{:.5f}".format(self.num.myPow(8.88023, -3)), str(0.00143))

    def test_maximum_subarray_leet51(self):
        self.assertEqual(self.lst.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)

    def test_spiral_order_leet54(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        self.assertEqual(self.lst.spiralOrder(matrix), [1, 2, 3, 6, 9, 8, 7, 4, 5])

    def test_jump_game_leet55(self):
        self.assertTrue(self.dp.canJump([0]))
        self.assertTrue(self.dp.canJump([3, 2, 1, 1, 4]))
        self.assertFalse(self.dp.canJump([3, 2, 1, 0, 4]))

    def test_merge_intervals_leet56(self):
        intervals = [lp.Interval(1, 3), lp.Interval(2, 6), lp.Interval(9, 10),
                     lp.Interval(8, 10), lp.Interval(15, 18), lp.Interval(3, 5)]
        merged = self.lst.merge(intervals)
        answer = [lp.Interval(1, 6), lp.Interval(8, 10), lp.Interval(15, 18)]
        self.assertEqual(sorted([i.start for i in merged]), sorted([i.start for i in answer]))
        self.assertEqual(sorted([i.end for i in merged]), sorted([i.end for i in answer]))

    def test_length_of_last_word_leet58(self):
        self.assertEqual(self.str.lengthOfLastWord('abc'), 3)
        self.assertEqual(self.str.lengthOfLastWord('abc ss '), 2)

    def test_spiral_matrix_leet59(self):
        matrix = [
            [1, 2, 3],
            [8, 9, 4],
            [7, 6, 5]
        ]
        self.assertEqual([list(i) for i in self.lst.generateMatrix(3)], matrix)

    def test_permutation_sequence_leet60(self):
        self.assertEqual(self.lst.getPermutation(3, 5), '312')
        self.assertEqual(self.lst.getPermutation(4, 1), '1234')

    def test_rotate_list_leet61(self):
        n1 = lk.ListNode(1)
        n2 = lk.ListNode(2)
        n3 = lk.ListNode(3)
        n4 = lk.ListNode(4)
        n5 = lk.ListNode(5)
        n1.next = n2
        n2.next = n3
        n3.next = n4
        n4.next = n5
        head = self.link.rotateRight(n1, 3)
        val = []
        while head:
            val.append(str(head.val))
            head = head.next
        self.assertEqual(''.join(val), '34512')

    def test_unique_paths_leet62(self):
        self.assertEqual(self.lst.uniquePaths(3, 2), 3)
        self.assertEqual(self.lst.uniquePaths(7, 11), 8008)

    def test_unique_paths_II_leet63(self):
        obs = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(self.lst.uniquePathsWithObstacles(obs), 2)

    def test_minimum_path_sum_leet64(self):
        self.assertEqual(self.lst.minPathSum([[1, 2], [5, 6], [1, 1]]), 8)

    def test_plus_one_leet66(self):
        self.assertEqual(self.lst.plusOne([7]), [8])
        self.assertEqual(self.lst.plusOne([1, 9]), [2, 0])
        self.assertEqual(self.lst.plusOne([9, 9]), [1, 0, 0])

    def test_add_binary_leet67(self):
        self.assertEqual(self.str.addBinary('10010', '11011'), '101101')
        self.assertEqual(self.str.addBinary('111', '1'), '1000')

    def test_text_justification_leet68(self):
        words = ["This", "is", "an", "example", "of", "text", "justification."]
        justified = ['This    is    an', 'example  of text', 'justification.  ']
        self.assertEqual(self.lst.fullJustify(words, 16), justified)

    def test_sqrt_leet69(self):
        self.assertEqual(self.num.mySqrt(33), 5)

    def test_climbing_stairs_leet70(self):
        self.assertEqual(self.lst.climbStairs(15), 987)
        self.assertEqual(self.lst.climbStairs(77), 8944394323791464)

    def test_simplify_path_leet71(self):
        self.assertEqual(self.str.simplifyPath('/a/./b/../../c/'), '/c')
        self.assertEqual(self.str.simplifyPath('/home/'), '/home')

    def test_set_matrix_zeroes_leet73(self):
        matrix = [[0, 0, 0, 5], [4, 3, 1, 4], [0, 1, 1, 4], [1, 2, 1, 3], [0, 0, 1, 1]]
        answer = [[0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0]]
        self.lst.setZeroes(matrix)
        self.assertEqual(matrix, answer)

    def test_search_2d_matrix_leet74(self):
        matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 50]]
        self.assertTrue(self.lst.searchMatrix(matrix, 1))
        self.assertTrue(self.lst.searchMatrix(matrix, 3))
        self.assertTrue(self.lst.searchMatrix(matrix, 50))
        self.assertFalse(self.lst.searchMatrix(matrix, 21))

    def test_sort_colors_leet75(self):
        colors = [2, 1, 2, 0, 0]
        self.lst.sortColors(colors)
        self.assertEqual(colors, [0, 0, 1, 2, 2])

    def test_min_window_substring_leet76(self):
        self.assertEqual(self.str.minWindow('ADOBECODEBANC', 'ABC'), 'BANC')
        self.assertEqual(self.str.minWindow('ab', 'a'), 'a')
        self.assertEqual(self.str.minWindow('ab', 'b'), 'b')
        self.assertEqual(self.str.minWindow('ab', 'bb'), '')

    def test_combinations_leet77(self):
        ans = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
               [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        self.assertEqual(sorted(self.lst.combine(5, 3)), sorted(ans))

    def test_subset_leet78(self):
        ans = [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
        self.assertEqual(sorted(ans), sorted(self.lst.subsets([1, 2, 3])))

    def test_word_search_leet79(self):
        self.assertTrue(self.lst.exist(["ABCE", "SFES", "ADEE"], "ABCESEEEFS"))
        self.assertTrue(self.lst.exist(["ABCE", "SFCS", "ADEE"], "ABCCED"))
        self.assertFalse(self.lst.exist(["ABCE", "SFES", "ADEE"], "ABCCED"))

    def test_largest_rectangle_area_leet84(self):
        self.assertEqual(self.lst.largestRectangleArea([5, 4, 3, 5, 4]), 15)
        self.assertEqual(self.lst.largestRectangleArea([2, 1, 5, 6, 2, 3]), 10)

    def test_best_time_to_buy_and_sell_stocks_leet121(self):
        self.assertEqual(self.lst.maxProfit([7, 1, 5, 3, 6, 4]), 5)
        self.assertEqual(self.lst.maxProfit([7, 6, 4, 3, 1]), 0)

    def test_longest_consecutive_leet128(self):
        self.assertEqual(self.lst.longestConsecutive([100, 4, 200, 1, 3, 2]), 4)

    def test_excel_sheet_column_title_leet168(self):
        self.assertEqual(self.num.convertToTitle(1), 'A')
        self.assertEqual(self.num.convertToTitle(26), 'Z')
        self.assertEqual(self.num.convertToTitle(27), 'AA')
        self.assertEqual(self.num.convertToTitle(52), 'AZ')
        self.assertEqual(self.num.convertToTitle(231894), 'MDZZ')

    def test_number_of_islands_leet200(self):
        map1 = ["11110", "11010", "11000", "00000"]
        self.assertEqual(self.lst.numIslands(map1), 1)

    def test_happy_number_leet202(self):
        self.assertTrue(self.num.isHappy(7))
        self.assertFalse(self.num.isHappy(4))

    def test_kth_largest_element_leet215(self):
        self.assertEqual(self.lst.findKthLargest([3, 2, 1, 5, 6, 4], 2), 5)

    def test_add_operators_leet282(self):
        self.assertEqual(sorted(self.lst.addOperators('123', 6)), sorted(["1+2+3", "1*2*3"]))
        self.assertEqual(sorted(self.lst.addOperators('232', 8)), sorted(["2*3+2", "2+3*2"]))
        self.assertEqual(sorted(self.lst.addOperators('105', 5)), sorted(["1*0+5", "10-5"]))

    def test_move_zero_leet283(self):
        l = [0, 1, 0, 3, 12]
        self.lst.moveZeroes(l)
        self.assertEqual(l, [1, 3, 12, 0, 0])

    def test_coin_change_leet322(self):
        self.assertEqual(self.dp.coinChange([1, 2, 5], 11), 3)
        self.assertEqual(self.dp.coinChange([2], 3), -1)

    def test_number_of_boomerangs_leet447(self):
        self.assertEqual(self.lst.numberOfBoomerangs([[0, 0], [1, 0], [2, 0]]), 2)


if __name__ == '__main__':
    unittest.main()
