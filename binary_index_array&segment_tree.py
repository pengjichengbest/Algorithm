# coding:utf-8
"""
    Author: apple
    Date: 24/11/2023
    File: binary_index_array&segment_tree.py
    ProjectName: Algorithm
    Time: 20:14
"""
import math


# 树状数组类，支持单点修改、区间和查询
class BinaryIndexArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.binary_index_array = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.update(i, nums[i - 1])

    def update(self, index, delta):
        while index <= self.n:
            self.binary_index_array[index] += delta
            index += index & (-index)

    def query(self, index):
        ans = 0
        while index > 0:
            ans += self.binary_index_array[index]
            index -= index & (-index)
        return ans


# 树状数组调用接口
class BinaryArrayAPI:
    def __init__(self, nums):
        self.binary_array = BinaryIndexArray(nums)
        self.nums = nums

    def update(self, index, val):
        delta = val - self.nums[index]
        self.nums[index] = val
        self.binary_array.update(index + 1, delta)

    def sumRange(self, left, right):
        return self.binary_array.query(right + 1) - self.binary_array.query(left)


# 线段树类 支持单点修改、区间修改、查询区间和、区间最值
class Node:
    def __init__(self):
        # mark表示该节点之前已经被记录过的偏移值，第一次遍历到对子树不做修改，下次如果进入到子节点，修复bug，把该bug转移到下一层节点
        self.l = self.r = self.sum_v = self.mark = self.max_v = self.min_v = 0


class SegmentTree:
    def __init__(self, nums):
        n = len(nums)
        self.segment_tree = [Node() for _ in range(4 * n)]
        # 建树的时候注意索引要从1开始
        self.build_tree(1, 0, n - 1, nums)

    # 递归建树
    def build_tree(self, index, l, r, nums):
        self.segment_tree[index].l, self.segment_tree[index].r = l, r
        if l == r:
            self.segment_tree[index].sum_v = nums[l]
            self.segment_tree[index].max_v = nums[l]
            self.segment_tree[index].min_v = nums[l]
            return
        mid = (l + r) >> 1
        self.build_tree(index * 2, l, mid, nums)
        self.build_tree(index * 2 + 1, mid + 1, r, nums)
        self.segment_tree[index].sum_v = self.segment_tree[2 * index].sum_v + self.segment_tree[2 * index + 1].sum_v
        self.segment_tree[index].max_v = max(self.segment_tree[2 * index].max_v, self.segment_tree[2 * index + 1].max_v)
        self.segment_tree[index].min_v = min(self.segment_tree[2 * index].min_v, self.segment_tree[2 * index + 1].min_v)

    # 单点修改
    def change(self, index, target, nums, val):
        l, r = self.segment_tree[index].l, self.segment_tree[index].r
        if l == r:
            self.segment_tree[index].sum_v = val
            self.segment_tree[index].min_v = val
            self.segment_tree[index].max_v = val
            return
        self.spread(index)
        mid = (l + r) >> 1
        if mid < target:
            self.change(2 * index + 1, target, nums, val)
        else:
            self.change(2 * index, target, nums, val)
        self.segment_tree[index].sum_v = self.segment_tree[2 * index].sum_v + self.segment_tree[2 * index + 1].sum_v
        self.segment_tree[index].max_v = max(self.segment_tree[2 * index].max_v, self.segment_tree[2 * index + 1].max_v)
        self.segment_tree[index].min_v = min(self.segment_tree[2 * index].min_v, self.segment_tree[2 * index + 1].min_v)

    # 区间查询
    def query(self, index, l, r):
        sum1, sum2 = 0, 0
        max1, max2 = float('-inf'), float('-inf')
        min1, min2 = float('inf'), float('inf')
        if l <= self.segment_tree[index].l and r >= self.segment_tree[index].r:
            return self.segment_tree[index].sum_v, self.segment_tree[index].max_v, self.segment_tree[index].min_v
        self.spread(index)
        mid = (self.segment_tree[index].l + self.segment_tree[index].r) >> 1
        if l <= mid:
            sum1, max1, min1 = self.query(index * 2, l, r)
        if r > mid:
            sum2, max2, min2 = self.query(index * 2 + 1, l, r)
        return sum1 + sum2, max(max1, max2), min(min1, min2)

    # 区间修改-懒惰标记
    def change_range(self, index, l, r, delta):
        if l <= self.segment_tree[index].l and r >= self.segment_tree[index].r:
            self.segment_tree[index].sum_v += delta * (self.segment_tree[index].r - self.segment_tree[index].l + 1)
            self.segment_tree[index].min_v += delta
            self.segment_tree[index].max_v += delta
            self.segment_tree[index].mark += delta
            return
        self.spread(index)
        mid = (self.segment_tree[index].l + self.segment_tree[index].r) >> 1
        if l <= mid:
            self.change_range(index * 2, l, r, delta)
        if r > mid:
            self.change_range(index * 2 + 1, l, r, delta)
        self.segment_tree[index].sum_v = self.segment_tree[index * 2].sum_v + self.segment_tree[index * 2 + 1].sum_v
        self.segment_tree[index].max_v = max(self.segment_tree[index * 2].max_v, self.segment_tree[index * 2 + 1].max_v)
        self.segment_tree[index].min_v = min(self.segment_tree[index * 2].min_v, self.segment_tree[index * 2 + 1].min_v)

    # 修复bug，消除本节点标记，传递给下面一层
    def spread(self, index):
        if self.segment_tree[index].mark != 0:
            delta = self.segment_tree[index].mark
            self.segment_tree[2 * index].sum_v += delta * (
                    self.segment_tree[2 * index].r - self.segment_tree[2 * index].l + 1)
            self.segment_tree[2 * index].max_v += delta
            self.segment_tree[2 * index].min_v += delta
            self.segment_tree[2 * index].mark += delta
            self.segment_tree[2 * index + 1].sum_v += delta * (
                    self.segment_tree[2 * index + 1].r - self.segment_tree[2 * index + 1].l + 1)
            self.segment_tree[2 * index + 1].max_v += delta
            self.segment_tree[2 * index + 1].min_v += delta
            self.segment_tree[2 * index + 1].mark += delta
            self.segment_tree[index].mark = 0


# 线段树调用接口
class SegmentTreeAPI:
    def __init__(self, nums):
        self.nums = nums
        self.segment_tree_obj = SegmentTree(nums)

    def change_single_value(self, target_index, val):
        self.nums[target_index] = val
        # 注意这里调用线段树索引要从1开始
        self.segment_tree_obj.change(1, target_index, self.nums, val)

    def query_range(self, l, r):
        # 依次返回区间和、区间最大值、区间最小值
        return self.segment_tree_obj.query(1, l, r)

    # 区间修改，代表在数组nums中从l到r的元素分别加delta
    def change_range(self, l, r, delta):
        self.segment_tree_obj.change_range(1, l, r, delta)


