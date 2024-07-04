# coding:utf-8
"""
    Author: apple
    Date: 26/11/2023
    File: algorithm_review.py
    ProjectName: Algorithm
    Time: 16:10
"""
from collections import Counter


# LeetCode699 掉落的方块
class FallingSquares:
    def fallingSquares(self, positions):
        # 离散化处理
        index_memory = {}
        index_record = []
        for position in positions:
            index_record.append(position[0])
            index_record.append(position[0] + position[1])
        index_record.sort()
        current = 0
        for index in index_record:
            if index not in index_memory:
                index_memory[index] = current
                current += 1
        api = SegmentTree(len(index_memory))
        ans = []
        cur_ans = 0
        for position in positions:
            index1, index2 = index_memory[position[0]], index_memory[position[0] + position[1]]
            previous = api.query(1, index1, index2 - 1)
            target_high = previous + position[1]
            cur_ans = max(cur_ans, target_high)
            ans.append(cur_ans)
            api.change_range(1, index1, index2 - 1, target_high)
        return ans


class Node:
    def __init__(self):
        self.mark = self.maxV = self.l = self.r = 0


class SegmentTree:
    def __init__(self, n):
        self.segment_tree = [Node() for _ in range(4 * n)]
        self.build_tree(1, 0, n - 1)

    def build_tree(self, index, l, r):
        self.segment_tree[index].l, self.segment_tree[index].r = l, r
        if l == r:
            return
        mid = (l + r) >> 1
        self.build_tree(index * 2, l, mid)
        self.build_tree(index * 2 + 1, mid + 1, r)

    def query(self, index, l, r):
        max1, max2 = 0, 0
        if l <= self.segment_tree[index].l and r >= self.segment_tree[index].r:
            return self.segment_tree[index].maxV
        self.spread(index)
        mid = (self.segment_tree[index].l + self.segment_tree[index].r) >> 1
        if l <= mid:
            max1 = self.query(index * 2, l, r)
        if r > mid:
            max2 = self.query(index * 2 + 1, l, r)
        return max(max1, max2)

    def change_range(self, index, l, r, target_high):
        if l <= self.segment_tree[index].l and r >= self.segment_tree[index].r:
            self.segment_tree[index].maxV = target_high
            self.segment_tree[index].mark = target_high
            return
        self.spread(index)
        mid = (self.segment_tree[index].l + self.segment_tree[index].r) >> 1
        if l <= mid:
            self.change_range(index * 2, l, r, target_high)
        if r > mid:
            self.change_range(index * 2 + 1, l, r, target_high)
        self.segment_tree[index].maxV = max(self.segment_tree[2 * index].maxV, self.segment_tree[2 * index + 1].maxV)

    def spread(self, index):
        if self.segment_tree[index].mark != 0:
            self.segment_tree[2 * index].maxV = self.segment_tree[index].mark
            self.segment_tree[2 * index].mark = self.segment_tree[index].mark
            self.segment_tree[2 * index + 1].maxV = self.segment_tree[index].mark
            self.segment_tree[2 * index + 1].mark = self.segment_tree[index].mark
            self.segment_tree[index].mark = 0


class BeautifulSubstrings:
    def beautifulSubstrings(self, s, k):
        v = {'a', 'e', 'i', 'o', 'u'}
        pre_sum = [0] * (len(s) + 1)
        for i in range(len(s)):
            if s[i] in v:
                pre_sum[i + 1] = pre_sum[i] + 1
            else:
                pre_sum[i + 1] = pre_sum[i]
        ans = 0
        for i in range(1, len(s)):
            for j in range(i):
                count_v = pre_sum[i + 1] - pre_sum[j]
                if not (count_v * (i - j + 1 - count_v)) % k and count_v == (i - j + 1 - count_v):
                    ans += 1
        return ans


class LexicographicallySmallestArray:
    def lexicographicallySmallestArray(self, nums, limit):
        ans = [0] * len(nums)
        temp = [0] * len(nums)
        for index, value in enumerate(nums):
            temp[index] = [index, value]
        temp.sort(key=lambda x: x[1])
        current, final = [[temp[0][0]], [temp[0][1]]], []
        for i in range(1, len(temp)):
            if temp[i][1] - temp[i - 1][1] <= limit:
                current[0].append(temp[i][0])
                current[1].append(temp[i][1])
            else:
                final.append(current)
                current = [[temp[i][0]], [temp[i][1]]]
        final.append(current)
        for cur_range in final:
            cur_range[0].sort()
        for each in final:
            index, value = each[0], each[1]
            i = 0
            while i < len(value):
                ans[index[i]] = value[i]
                i += 1
        return ans


class CloseStrings:
    def closeStrings(self, word1, word2):
        if len(word1) != len(word2):
            return False
        hash1, hash2 = {}, {}
        for i in range(len(word1)):
            if word1[i] not in hash1:
                hash1[word1[i]] = 1
            else:
                hash1[word1[i]] += 1
            if word2[i] not in hash2:
                hash2[word2[i]] = 1
            else:
                hash2[word2[i]] += 1
        if len(hash1) != len(hash2):
            return False
        record1, record2 = {}, {}
        for key, value in hash1.items():
            if key not in hash2:
                return False
            if value not in record1:
                record1[value] = 1
            else:
                record1[value] += 1
        for key, value in hash2.items():
            if value not in record2:
                record2[value] = 1
            else:
                record2[value] += 1
        for value in record1:
            if value not in record2 or record2[value] != record1[value]:
                return False

        return True


class FindLongestSubarray:
    def findLongestSubarray(self, array):
        record, pre_sum = {}, [0] * (len(array) + 1)
        temp = [0] * (len(array))
        for i in range(len(array)):
            if array[i].isalpha():
                temp[i] = -1
            else:
                temp[i] = 1
        for i in range(len(array)):
            pre_sum[i + 1] = pre_sum[i] + temp[i]
        value, ans = 0, []
        record[0] = 0
        for i in range(len(array)):
            if pre_sum[i + 1] in record:
                if i + 1 - record[pre_sum[i + 1]] > value:
                    value = i + 1 - record[pre_sum[i + 1]]
                    ans = [record[pre_sum[i + 1]], i]
            else:
                record[pre_sum[i + 1]] = i + 1
        return ans if not ans else array[ans[0]:ans[1] + 1]


class FirstCompleteIndex:
    def firstCompleteIndex(self, arr, mat):
        m, n = len(mat), len(mat[0])
        lines = [n] * m
        cols = [m] * n
        index = {}
        for i in range(m):
            for j in range(n):
                index[mat[i][j]] = (i, j)
        for x, num in enumerate(arr):
            i, j = index[num]
            lines[i] -= 1
            cols[j] -= 1
            if not lines[i] or not cols[j]:
                return x


def find_num(n):
    result = [True] * (n + 1)
    ans = []
    maxV = int(n ** 0.5) + 1
    for i in range(2, maxV):
        if result[i]:
            j = i
            while j * i <= n:
                result[j * i] = False
                j += 1
    for i, v in enumerate(result):
        if v:
            ans.append(i)
    return ans[2:]


class MinSubArrayLen:
    def minSubArrayLen(self, target, nums):
        l, current, ans = 0, 0, float('inf')
        for r in range(len(nums)):
            current += nums[r]
            if current < target:
                continue
            while current >= target:
                current -= nums[l]
                l += 1
            ans = min(ans, r - l + 2)
        return ans


class ProductExceptSelf:
    def productExceptSelf(self, nums):
        n = len(nums)
        pre_product, suf_product = [1] * (len(nums) + 1), [1] * (len(nums) + 1)
        for i in range(n):
            pre, suf = i, n - i - 1
            pre_product[pre + 1] = pre_product[pre] * nums[pre]
            suf_product[suf] = suf_product[suf + 1] * nums[suf]
        ans = [0] * n
        for i in range(n):
            ans[i] = pre_product[i] * suf_product[i + 1]
        return ans


class DiffWaysToCompute:
    def diffWaysToCompute(self, expression):
        self.record = []
        self.memory = {}
        symbol = {'+', '-', '*'}
        i = 0
        cur = ''
        while i < len(expression):
            if expression[i] in symbol:
                self.record.append(int(cur))
                self.record.append(expression[i])
                cur = ''
            else:
                cur += expression[i]
            i += 1
        self.record.append(int(cur))
        print(self.record)
        return self.sub_ques(0, len(self.record) - 1)

    def sub_ques(self, l, r):
        if (l, r) in self.memory:
            return self.memory[(l, r)]
        if l == r:
            return [self.record[l]]
        current = []
        for symbol in range(l + 1, r, 2):
            cur = []
            ans1 = self.sub_ques(l, symbol - 1)
            ans2 = self.sub_ques(symbol + 1, r)
            for i in ans1:
                for j in ans2:
                    if self.record[symbol] == '+':
                        cur.append(i + j)
                    elif self.record[symbol] == '-':
                        cur.append(i - j)
                    else:
                        cur.append(i * j)
            current += cur
        self.memory[(l, r)] = current
        return current


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ConstructMaximumBinaryTree:
    def constructMaximumBinaryTree(self, nums):
        return self.build(nums, 0, len(nums) - 1)

    def build(self, nums, l, r):
        if l > r:
            return
        cur = -1
        index = l
        for i in range(l, r + 1):
            if nums[i] > cur:
                cur = nums[i]
                index = i
        node = TreeNode(nums[index])
        node.left = self.build(nums, l, index - 1)
        node.right = self.build(nums, index + 1, r)
        return node
