from algorithms import list_processing as lp
from algorithms import substring_and_palindrome as sp


def try_1():
    p = lp.ListProcessing()
    print p.twoSum([3, 2, 4], 6)


def try_3():
    p = sp.SubstringAndPalindrome()
    print p.lengthOfLongestSubstring('accaddddfghdkg')

if __name__ == '__main__':
    # try_1()
    try_3()
