# coding:utf-8
"""
    Author: apple
    Date: 20/11/2024
    File: new_problem.py
    ProjectName: Algorithm
    Time: 20:08
"""
from collections import Counter, defaultdict
from itertools import accumulate


class MinimumSteps:
    def minimumSteps(self, s):
        ans = 0
        last = (len(s) - 1)
        i = len(s) - 1
        while i >= 0:
            if s[i] == '1':
                i -= 1
                last -= 1
            else:
                break
        while i >= 0:
            if s[i] == '0':
                i -= 1
                continue
            ans += last - i
            last -= 1
            i -= 1
        return ans


class MaximumXorProduct:
    def maximumXorProduct(self, a, b, n):
        big = 1000000007
        flag = 'equal'
        currentA, currentB = 0, 0
        for step in range(49, n - 1, -1):
            current = 1 << step
            if current & a == current & b:
                continue
            if current & a:
                flag = 'A'
                break
            else:
                flag = 'B'
                break
        for step in range(49, -1, -1):
            current = 1 << step
            if step >= n:
                if current & a:
                    currentA = (currentA % big + current % big) % big
                if current & b:
                    currentB = (currentB % big + current % big) % big
            else:
                if current & a == current & b:
                    currentA = (currentA % big + current % big) % big
                    currentB = (currentB % big + current % big) % big
                elif flag == 'A':
                    currentB = (currentB % big + current % big) % big
                elif flag == 'B':
                    currentA = (currentA % big + current % big) % big
                else:
                    currentA = (currentA % big + current % big) % big
                    flag = 'A'

        return (currentA * currentB) % big


class MinDeletion:
    def minDeletion(self, nums):
        cnt = 0
        i, j = 0, 1
        while i < len(nums):
            while j < len(nums) and nums[i] == nums[j]:
                cnt += 1
                j += 1
            if j == len(nums):
                break
            i, j = j + 1, j + 2
        if (len(nums) - cnt) % 2 == 1:
            return cnt + 1
        return cnt


class MinPathCost:
    def minPathCost(self, grid, moveCost):
        memory = {}
        ans = float('inf')
        for node in grid[0]:
            memory[node] = node
        for i in range(1, len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] not in memory:
                    memory[grid[i][j]] = float('inf')
                for k in range(len(grid[i - 1])):
                    memory[grid[i][j]] = min(memory[grid[i][j]], memory[grid[i - 1][k]] + moveCost[grid[i - 1][k]][j])
                memory[grid[i][j]] += grid[i][j]
        for node in grid[-1]:
            ans = min(ans, memory[node])
        return ans


class MaximizeSquareHoleArea:
    def maximizeSquareHoleArea(self, n, m, hBars, vBars):
        hBars.sort()
        vBars.sort()
        cur_line, max_line = 1, 0
        for i in range(1, len(hBars)):
            if hBars[i] - hBars[i - 1] == 1:
                cur_line += 1
            else:
                max_line = max(max_line, cur_line)
                cur_line = 1
        max_line = max(max_line, cur_line)
        cur_col, max_col = 1, 0
        for j in range(1, len(vBars)):
            if vBars[j] - vBars[j - 1] == 1:
                cur_col += 1
            else:
                max_col = max(max_col, cur_col)
                cur_col = 1
        max_col = max(max_col, cur_col)
        ans = min(max_line, max_col)
        return (ans + 1) * (ans + 1)


class CarPooling:
    def carPooling(self, trips, capacity):
        n = len(trips)
        down = []
        cnt = Counter()
        for i in range(n):
            cnt[trips[i][2]] += trips[i][0]
        for key, value in cnt.items():
            down.append((key, value))
        down.sort(key=lambda x: x[0])
        trips.sort(key=lambda x: x[1])
        l = 0
        for r in range(n):
            while down[l][0] <= trips[r][1]:
                capacity += down[l][1]
                l += 1
            capacity -= trips[r][0]
            if capacity < 0:
                return False
        return True


class MaxScore:
    def maxScore(self, cardPoints, k):
        if k == len(cardPoints):
            return sum(cardPoints)
        n, rest = len(cardPoints), len(cardPoints) - k

        cur, cur_sum, ans = 0, 0, float('inf')
        for i in range(n):
            cur += 1
            cur_sum += cardPoints[i]
            if cur < rest:
                continue
            elif cur == rest:
                ans = min(ans, cur_sum)
                cur -= 1
                cur_sum -= cardPoints[i - rest + 1]
        return sum(cardPoints) - ans


class MinimumAddedCoins:
    def minimumAddedCoins(self, coins, target):
        cur_sum = 0
        ans = 0
        cnt = Counter()
        for coin in coins:
            cnt[coin] += 1
        for i in range(1, target + 1):
            if i in cnt:
                cur_sum += i * cnt[i]
            elif cur_sum < i:
                cur_sum += i
                ans += 1
        return ans


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BstToGst:
    def bstToGst(self, root):
        self.values = []
        self.memory = {}
        self.add_val(root)
        self.memory[self.values[-1]] = self.values[-1]
        for i in range(len(self.values) - 2, -1, -1):
            self.memory[self.values[i]] = self.values[i] + self.memory[self.values[i + 1]]
        self.change_val(root)
        return root

    def add_val(self, node):
        if not node:
            return
        self.add_val(node.left)
        self.values.append(node.val)
        self.add_val(node.right)

    def change_val(self, node):
        if not node:
            return
        self.change_val(node.left)
        node.val = self.memory[node.val]
        self.change_val(node.right)


class PlatesBetweenCandles:
    def platesBetweenCandles(self, s, queries):
        candles = []
        for i in range(len(s)):
            if s[i] == '|':
                candles.append(i)
        pre_sum = [0] * (len(s) + 1)
        for i in range(len(s)):
            if s[i] == '*':
                pre_sum[i + 1] = pre_sum[i] + 1
            else:
                pre_sum[i + 1] = pre_sum[i]
        ans = [0] * len(queries)
        for i, query in enumerate(queries):
            l, r = query[0], query[1]
            l = self.find_first_min(l, candles)
            r = self.find_last_max(r, candles)
            if l >= r or l == -1 or r == len(candles):
                continue
            l, r = candles[l], candles[r]
            ans[i] = pre_sum[r + 1] - pre_sum[l]
        return ans

    def find_last_max(self, target, candles):
        left, right = -1, len(candles) - 1
        while left < right:
            mid = (left + right + 1) >> 1
            if candles[mid] > target:
                right = mid - 1
            else:
                left = mid
        return left

    def find_first_min(self, target, candles):
        left, right = 0, len(candles)
        while left < right:
            mid = (left + right) >> 1
            if candles[mid] >= target:
                right = mid
            else:
                left = mid + 1
        return left


# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Construct:
    def construct(self, grid):
        pre_sum = [[0] * (len(grid[0]) + 1) for _ in range(len(grid) + 1)]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                pre_sum[i + 1][j + 1] = grid[i][j] + pre_sum[i][j + 1] + pre_sum[i + 1][j] - pre_sum[i][j]
        return self.build((0, 0), (len(grid), len(grid)), pre_sum)

    def build(self, index1, index2, pre_sum):
        node = Node(None, None, None, None, None, None)
        x1, y1, x2, y2 = index1[0], index1[1], index2[0], index2[1]
        width = x2 - x1
        height = y2 - y1
        cur_sum = pre_sum[x2][y2] - pre_sum[x1][y2] - pre_sum[x2][y1] + pre_sum[x1][y1]
        if cur_sum == 0 or cur_sum == height * width:
            node.val = 0 if cur_sum == 0 else 1
            node.isLeaf = True
            return node
        child1 = [(x1, y1), ((x1 + x2) >> 1, (y1 + y2) >> 1)]
        child2 = [(x1, (y1 + y2) >> 1), ((x1 + x2) >> 1, y2)]
        child3 = [((x1 + x2) >> 1, y1), (x2, (y1 + y2) >> 1)]
        child4 = [((x1 + x2) >> 1, (y1 + y2) >> 1), (x2, y2)]
        node.val = cur_sum
        node.isLeaf = False
        node.topLeft = self.build(child1[0], child1[1], pre_sum)
        node.topRight = self.build(child2[0], child2[1], pre_sum)
        node.bottomLeft = self.build(child3[0], child3[1], pre_sum)
        node.bottomRight = self.build(child4[0], child4[1], pre_sum)
        return node


class MaxIceCream:
    def maxIceCream(self, costs, coins):
        maxV = 100000
        count_sort = [0] * 100000
        for i in range(len(costs)):
            count_sort[costs[i] - 1] += 1
        cur_sum = 0
        cur_pointer = 0
        for i in range(maxV):
            if count_sort[i] > 0:
                cur_pointer = i
                break
        ans = 0
        while cur_pointer < maxV and cur_sum + cur_pointer + 1 <= coins:
            ans += 1
            cur_sum += cur_pointer + 1
            count_sort[cur_pointer] -= 1
            while cur_pointer < maxV and not count_sort[cur_pointer]:
                cur_pointer += 1
        return ans


from collections import deque

import math


class MinimumFuelCost:
    def minimumFuelCost(self, roads, seats):
        n = len(roads) + 1
        out_edges = [[] for _ in range(n)]
        for road in roads:
            out_edges[road[0]].append(road[1])
            out_edges[road[1]].append(road[0])
        in_edges = [0] * n
        q = deque()
        memory = set()
        q.append(0)
        while q:
            top = q.popleft()
            memory.add(top)
            for node in out_edges[top]:
                if node not in memory:
                    in_edges[top] += 1
                    q.append(node)
        return self.top_sort(in_edges, out_edges, seats)

    def top_sort(self, in_edges, out_edges, seats):
        n = len(out_edges)
        record = [[0] * 3 for _ in range(n)]
        memory = set()
        q = deque()
        for i, v in enumerate(in_edges):
            if not v:
                q.append(i)
                record[i] = [0, 1, 1]
        while q:
            child = q.popleft()
            memory.add(child)
            for fa in out_edges[child]:
                if fa not in memory:
                    record[fa][0] += record[child][0] + record[child][2]
                    record[fa][1] += record[child][1]
                    in_edges[fa] -= 1
                    if not in_edges[fa]:
                        record[fa][1] += 1
                        record[fa][2] = math.ceil(record[fa][1] / seats)
                        q.append(fa)
        return record[0][0]


from collections import deque


class MinReorder:
    def minReorder(self, n, connections):
        out_edges = [[] for _ in range(n)]
        for road in connections:
            out_edges[road[0]].append((road[1], False))
            out_edges[road[1]].append((road[0], True))
        q = deque()
        ans = 0
        memory = set()
        q.append(0)
        while q:
            top = q.popleft()
            memory.add(top)
            for child in out_edges[top]:
                if child[0] not in memory:
                    q.append(child[0])
                    if not child[1]:
                        ans += 1
        return ans


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class IsEvenOddTree:
    def isEvenOddTree(self, root):
        record = {}
        q = deque()
        q.append((root, 0))
        record[0] = [root.val]
        while q:
            top = q.popleft()
            if top[0].left:
                cur_layer = top[1] + 1
                if cur_layer not in record:
                    record[cur_layer] = [top[0].left.val]
                else:
                    record[cur_layer].append(top[0].left.val)
                q.append((top[0].left, cur_layer))
            if top[0].right:
                cur_layer = top[1] + 1
                if cur_layer not in record:
                    record[cur_layer] = [top[0].right.val]
                else:
                    record[cur_layer].append(top[0].right.val)
                q.append((top[0].right, cur_layer))
        for k, v in record.items():
            if not k % 2:
                if not v[0] % 2:
                    return False
                i = 1
                while i < len(v):
                    if not v[i] % 2:
                        return False
                    if v[i] <= v[i - 1]:
                        return False
                    i += 1
            else:
                if v[0] % 2:
                    return False
                i = 1
                while i < len(v):
                    if v[i] % 2:
                        return False
                    if v[i] >= v[i - 1]:
                        return False
                    i += 1
        return True


from queue import PriorityQueue as PQ


class KSmallestPairs:
    def kSmallestPairs(self, nums1, nums2, k):
        q = PQ()
        ans = []
        m, n = len(nums1), len(nums2)
        for i in range(m):
            q.put((nums1[i] + nums2[0], (i, 0)))
        for i in range(k):
            if q.empty():
                return ans
            top = q.get()
            value, a, b = top[0], top[1][0], top[1][1]
            ans.append([nums1[a], nums2[b]])
            if b + 1 < n:
                q.put((nums1[a] + nums2[b + 1], (a, b + 1)))
        return ans


class RemoveAlmostEqualCharacters:
    def removeAlmostEqualCharacters(self, word):
        ans = 0
        i = 1
        while i < len(word):
            if abs(ord(word[i]) - ord(word[i - 1])) <= 1:
                ans += 1
                i += 2
            else:
                i += 1
        return ans


class MaxSubarrayLength:
    def maxSubarrayLength(self, nums, k):
        cnt = Counter()
        ans = 0
        l = 0
        for i in range(len(nums)):
            cnt[nums[i]] += 1
            while cnt[nums[i]] > k:
                cnt[nums[l]] -= 1
                l += 1
            if cnt[nums[i]] <= k:
                ans = max(ans, i - l + 1)
        return ans


import heapq


class MinimumEffortPath:
    def minimumEffortPath(self, heights):
        m, n = len(heights), len(heights[0])
        dist = {(i, j): float('inf') for i in range(m) for j in range(n)}
        dist[(0, 0)] = 0
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        while heap:
            top = heapq.heappop(heap)
            dis, x, y = top[0], top[1], top[2]
            if x == m - 1 and y == n - 1:
                return dist[(m - 1, n - 1)]
            for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= a < m and 0 <= b < n and dist[(a, b)] > max(dist[(x, y)], abs(heights[x][y] - heights[a][b])):
                    dist[(a, b)] = max(dist[(x, y)], abs(heights[x][y] - heights[a][b]))
                    heapq.heappush(heap, (dist[(a, b)], a, b))


class SecondGreaterElement:
    def secondGreaterElement(self, nums):
        neareast = [-1] * len(nums)
        stack1, stack2 = [], []
        for i in range(len(nums)):
            temp = []
            while stack1 and nums[stack1[-1]] < nums[i]:
                top = stack1.pop()
                temp.append(top)
            stack1.append(i)
            while stack2 and nums[stack2[-1]] < nums[i]:
                top = stack2.pop()
                neareast[top] = nums[i]
            stack2 += temp[::-1]
        return neareast


class NumTrees:
    def numTrees(self, n):
        self.memory = {}
        self.ans = 0
        for i in range(1, n + 1):
            self.ans += self.find_solutions(i, 1, n)
        return self.ans

    def find_solutions(self, root, l, r):
        if (root, l, r) in self.memory:
            return self.memory[(root, l, r)]
        ans1, ans2 = 1, 1
        for left in range(l, root):
            ans1 += self.find_solutions(left, l, root - 1)
        if ans1 > 1:
            ans1 -= 1
        for right in range(root + 1, r + 1):
            ans2 += self.find_solutions(right, root + 1, r)
        if ans2 > 1:
            ans2 -= 1
        self.memory[(root, l, r)] = ans1 * ans2
        return ans1 * ans2


class SortedListToBST:
    def sortedListToBST(self, head):
        record = []
        while head:
            record.append(head.val)
            head = head.next
        return self.build(0, len(record) - 1, record)

    def build(self, l, r, record):
        if l > r:
            return
        root = (l + r) >> 1
        node = TreeNode(record[root])
        node.left = self.build(l, root - 1, record)
        node.right = self.build(root + 1, r, record)
        return node


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class RecoverTree:
    def recoverTree(self, root):
        """
        Do not return anything, modify root in-place instead.
        """
        self.record = []
        self.traverse(root)
        big = 0
        small = 0
        for i in range(len(self.record) - 1):
            if self.record[i].val > self.record[i + 1].val:
                big = i
                break
        for j in range(big + 2, len(self.record)):
            if self.record[j].val < self.record[j - 1].val:
                small = j
        if not small:
            small = big + 1
        self.record[big].val, self.record[small].val = self.record[small].val, self.record[big].val
        return root

    def traverse(self, root):
        if not root:
            return
        self.traverse(root.left)
        self.record.append(root)
        self.traverse(root.right)


class MaxPoints:
    def maxPoints(self, points):
        memory = {}
        ans = 0
        memory[(points[0][0],), (points[0][0], points[0][1])] = 1
        for i in range(1, len(points)):
            for j in range(i):
                x1, y1, x2, y2 = points[j][0], points[j][1], points[i][0], points[i][1]
                cur = self.find_k_b(x1, y1, x2, y2)
                if cur in memory:
                    memory[cur] += 1
                else:
                    memory[cur] = 2
        print(memory)
        for res in memory:
            ans = max(ans, memory[res])
        return ans

    def find_k_b(self, x1, y1, x2, y2):
        if x1 == x2:
            return (x1,), (x2, y2)
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        k, b = round(k, 4), round(b, 4)
        return (k, b), (x2, y2)


class DivideArray:
    def divideArray(self, nums, k):
        nums.sort()
        n = len(nums)
        result = [[0] * 3 for each in range(n // 3)]
        result[0][0] = nums[0]
        for i in range(1, n):
            result[i // 3][i % 3] = nums[i]
            if i % 3 == 2 and nums[i] - nums[i - 2] > k:
                return []
        return result


class MinimumCost:
    def minimumCost(self, nums):
        nums.sort()
        print(nums)
        n = len(nums)
        if n % 2:
            mid = nums[n // 2]
            first_big = self.find_min_bigger(mid)
            last_min = self.find_max_smaller(mid)
            if 10 == mid:
                last_min = 9
            print(first_big)
            print(last_min)
            ans1, ans2 = 0, 0
            for i in range(n):
                ans1 += abs(first_big - nums[i])
                ans2 += abs(last_min - nums[i])
            return min(ans1, ans2)
        else:
            left, right = nums[n // 2 - 1], nums[n // 2]
            first_big = self.find_min_bigger(left)
            last_min = self.find_max_smaller(right)
            ans1, ans2 = 0, 0
            for i in range(n):
                ans1 += abs(first_big - nums[i])
                ans2 += abs(last_min - nums[i])
            return min(ans1, ans2)

    def find_min_bigger(self, lower):
        lower = str(lower)
        length = len(lower)
        if length % 2:
            if int(lower[length // 2:]) <= int(lower[:length // 2 + 1][::-1]):
                return int(lower[:length // 2 + 1] + lower[:length // 2][::-1])
            change = str(int(lower[:length // 2 + 1]) + 1)
            return int(change + change[:-1][::-1])
        else:
            if int(lower[length // 2:]) <= int(lower[:length // 2][::-1]):
                return int(lower[:length // 2] + lower[:length // 2][::-1])
            change = str(int(lower[:length // 2]) + 1)
            return int(change + change[::-1])

    def find_max_smaller(self, upper):
        upper = str(upper)
        length = len(upper)
        if length % 2:
            if int(upper[length // 2:]) >= int(upper[:length // 2 + 1][::-1]):
                return int(upper[:length // 2 + 1] + upper[:length // 2][::-1])
            change = str(int(upper[:length // 2 + 1]) - 1)
            if len(change) < len(upper[:length // 2 + 1]):
                return int(change + change)
            return int(change + change[:-1][::-1])
        else:
            if int(upper[length // 2:]) >= int(upper[:length // 2][::-1]):
                return int(upper[:length // 2] + upper[:length // 2][::-1])
            change = str(int(upper[:length // 2]) - 1)
            if len(change) < len(str(upper[:length // 2])):
                return int(change + change + change[-1])
            return int(change + change[::-1])


class MaximizeSquareArea:
    def maximizeSquareArea(self, m, n, hFences, vFences):
        hFences.append(1)
        hFences.append(m)
        vFences.append(1)
        vFences.append(n)
        hFences.sort()
        vFences.sort()
        memory = set()
        for i in range(1, len(hFences)):
            for j in range(i):
                memory.add(hFences[i] - hFences[j])
        ans = -1
        for i in range(1, len(vFences)):
            for j in range(i):
                if vFences[i] - vFences[j] in memory:
                    ans = max(ans, vFences[i] - vFences[j])
        return ans


import heapq


class MinimumCost:
    def minimumCost(self, source, target, original, changed, cost):
        memory = {}
        for i in range(len(original)):
            start, end = ord(original[i]) - 97, ord(changed[i]) - 97
            if (start, end) not in memory:
                memory[start, end] = cost[i]
            else:
                if memory[start, end] > cost[i]:
                    memory[start, end] = cost[i]
        outEdges = [[] for each in range(26)]
        for edge in memory:
            start, end, weight = edge[0], edge[1], memory[edge]
            outEdges[start].append((end, weight))
        dist = {}
        for i in range(26):
            dist.update(self.find_shortes_path(i, outEdges))
        ans = 0
        for i in range(len(source)):
            start, end = ord(source[i]) - 97, ord(target[i]) - 97
            if dist[start, end] == float('inf'):
                return -1
            ans += dist[start, end]
        return ans


class MinimumCostII:
    def minimumCost(self, source, target, original, changed, cost):
        path = defaultdict(dict)
        for x, y, z in zip(original, changed, cost):
            if y not in path[x]:
                path[x][y] = z
            else:
                path[x][y] = min(path[x][y], z)
        dist = {}
        for node in path:
            dist.update(self.find_chortest_path(node, path))
        dp = [float('inf')] * (len(source) + 1)
        dp[0] = 0
        for i in range(1, len(dp)):
            if source[i - 1] == target[i - 1]:
                dp[i] = dp[i - 1]
            visited = set()
            for v in dist:
                length = len(v)
                if length > i or length in visited:
                    continue
                if source[i - length:i] in dist and target[i - length:i] in dist[source[i - length:i]]:
                    dp[i] = min(dp[i], dp[i - length] + dist[source[i - length:i]][target[i - length:i]])
                    visited.add(length)
        return dp[-1] if dp[-1] != float('inf') else -1

    def find_chortest_path(self, node, path):
        dist = {}
        dist[node] = {}
        dist[node][node] = 0
        q = []
        q.append((0, node))
        while q:
            top = heapq.heappop(q)
            if top[1] not in path:
                continue
            for child in path[top[1]]:
                if child not in dist[node] or dist[node][child] > dist[node][top[1]] + path[top[1]][child]:
                    dist[node][child] = dist[node][top[1]] + path[top[1]][child]
                    heapq.heappush(q, (dist[node][child], child))
        return dist


class LargestPerimeter:
    def largestPerimeter(self, nums):
        nums.sort()
        sum_l = sum(nums)
        for i in range(len(nums) - 1, -1, -1):
            if sum_l - nums[i] > nums[i]:
                return sum_l
            sum_l -= nums[i]
        return -1


class IncremovableSubarrayCount:
    def incremovableSubarrayCount(self, nums):
        nums = [0] + nums + [1000000001]
        l, r = 0, len(nums) - 1
        while l + 1 < len(nums) and nums[l + 1] > nums[l]:
            l += 1
        while r - 1 >= 0 and nums[r - 1] < nums[r]:
            r -= 1
        n = len(nums) - 2
        if l >= r:
            return n * (n - 1) // 2 + n
        record = [-1] * (len(nums) - r)
        record[-1] = l
        start = len(nums) - 2
        for j in range(len(record) - 2, -1, -1):
            i = record[j + 1]
            if i == -1:
                break
            while i >= 0 and nums[i] >= nums[start]:
                i -= 1
            if i >= 0:
                record[j] = i
            start -= 1
        ans = 0
        for index in record:
            if index >= 0:
                ans += index + 1
        return ans


def cal_min_persons(requirments, M):
    requirments.sort()

    maxV, minV = requirments[-1] + requirments[-2], requirments[-1]
    while minV < maxV:
        mid = (minV + maxV) >> 1
        if judge(requirments, mid, M):
            maxV = mid
        else:
            minV = mid + 1
    return minV


def judge(requirments, target, M):
    r = len(requirments) - 1
    pair = 0
    for l in range(len(requirments)):
        while l < r and requirments[r] + requirments[l] > target:
            r -= 1
        if l >= r:
            break
        pair += 1
    ans = len(requirments) - pair * 2 + pair
    return True if ans <= M else False


class PlacedCoins:
    def placedCoins(self, edges, cost):
        self.cost = cost
        self.coin = [0] * len(cost)
        self.out_edges = [[] for each in range(len(cost))]
        for edge in edges:
            self.out_edges[edge[0]].append(edge[1])
            self.out_edges[edge[1]].append(edge[0])
        self.dfs(-1, 0)
        return self.coin

    def dfs(self, fa, index):
        n_sum, p_sum = [], []
        child_sum = 1
        for child in self.out_edges[index]:
            if child == fa:
                continue
            child_num, negtive, postive = self.dfs(index, child)

            child_sum += child_num
            n_sum += negtive
            p_sum += postive
        if self.cost[index] >= 0:
            p_sum += [self.cost[index]]
        else:
            n_sum += [-1 * self.cost[index]]
        heapq.heapify(p_sum)
        heapq.heapify(n_sum)
        while len(p_sum) > 3:
            heapq.heappop(p_sum)
        while len(n_sum) > 2:
            heapq.heappop(n_sum)
        if child_sum < 3:
            self.coin[index] = 1
        else:
            max1, max2 = 0, 0
            if len(n_sum) == 2 and p_sum:
                max1 = n_sum[0] * n_sum[1] * max(p_sum)
            if len(p_sum) == 3:
                max2 = p_sum[0] * p_sum[1] * p_sum[2]
            self.coin[index] = max(max1, max2)
        return child_sum, n_sum, p_sum


class MaximumLength:
    def maximumLength(self, s):
        record = [[] for _ in range(26)]
        current_length = 1
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                record[ord(s[i - 1]) - 97].append(current_length)
                current_length = 1
            else:
                current_length += 1
        record[ord(s[-1]) - 97].append(current_length)
        memory = {}
        for i, rec in enumerate(record):
            for length in rec:
                if (i, length) in memory:
                    memory[i, length] += 1
                else:
                    memory[i, length] = 1
                if length >= 3:
                    if (i, length - 1) in memory:
                        memory[i, length - 1] += 2
                    else:
                        memory[i, length - 1] = 2
                    if (i, length - 2) in memory:
                        memory[i, length - 2] += 3
                    else:
                        memory[i, length - 2] = 3
                elif length == 2:
                    if (i, length - 1) in memory:
                        memory[i, length - 1] += 2
                    else:
                        memory[i, length - 1] = 2
        ans = -1
        for key, value in memory.items():
            if value >= 3:
                ans = max(ans, key[1])
        return ans


class Solution:
    def canMakePalindromeQueries(self, s: str, queries):
        # 分成左右两半，右半反转
        n = len(s) // 2
        t = s[n:][::-1]
        s = s[:n]

        # 预处理三种前缀和
        sum_s = [[0] * 26 for _ in range(n + 1)]
        for i, b in enumerate(s):
            sum_s[i + 1] = sum_s[i][:]
            sum_s[i + 1][ord(b) - ord('a')] += 1

        sum_t = [[0] * 26 for _ in range(n + 1)]
        for i, b in enumerate(t):
            sum_t[i + 1] = sum_t[i][:]
            sum_t[i + 1][ord(b) - ord('a')] += 1

        sum_ne = list(accumulate((x != y for x, y in zip(s, t)), initial=0))

        # 计算子串中各个字符的出现次数，闭区间 [l,r]
        def count(sum, l, r):
            return [x - y for x, y in zip(sum[r + 1], sum[l])]

        def subtract(s1, s2):
            for i, s in enumerate(s2):
                s1[i] -= s
                if s1[i] < 0:
                    return False
            return s1

        def check(l1: int, r1: int, l2: int, r2: int, sumS, sumT):
            # [0,l1-1] 有 s[i] != t[i] 或者 [max(r1,r2)+1,n-1] 有 s[i] != t[i]
            if sum_ne[l1] > 0 or sum_ne[n] - sum_ne[max(r1, r2) + 1] > 0:
                return False
            if r2 <= r1:  # 区间包含
                return count(sumS, l1, r1) == count(sumT, l1, r1)
            if r1 < l2:  # 区间不相交
                # [r1+1,l2-1] 都满足 s[i] == t[i]
                return sum_ne[l2] - sum_ne[r1 + 1] == 0 and \
                    count(sumS, l1, r1) == count(sumT, l1, r1) and \
                    count(sumS, l2, r2) == count(sumT, l2, r2)
            # 区间相交但不包含
            s1 = subtract(count(sumS, l1, r1), count(sumT, l1, l2 - 1))
            s2 = subtract(count(sumT, l2, r2), count(sumS, r1 + 1, r2))
            return s1 and s2 and s1 == s2

        ans = [False] * len(queries)
        for i, (l1, r1, c, d) in enumerate(queries):
            l2, r2 = n * 2 - 1 - d, n * 2 - 1 - c
            ans[i] = check(l1, r1, l2, r2, sum_s, sum_t) if l1 <= l2 else \
                check(l2, r2, l1, r1, sum_t, sum_s)
        return ans


class MinOperationsMaxProfit:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        rest = 0
        profit = 0
        record = []
        for i in range(len(customers)):
            if rest + customers[i] > 4:
                profit += 4 * boardingCost - runningCost
                rest = rest + customers[i] - 4
            else:
                profit += (rest + customers[i]) * boardingCost - runningCost
                rest = 0
            record.append(profit)
        while rest > 0:
            if rest > 4:
                profit += 4 * boardingCost - runningCost
                rest -= 4
            else:
                profit += rest * boardingCost - runningCost
                rest = 0
            record.append(profit)
        times = -1
        maxV = -1
        for i in range(len(record)):
            if record[i] > maxV:
                maxV = record[i]
                times = i + 1
        return times if maxV > 0 else -1


class GetMaxRepetitions:
    def getMaxRepetitions(self, s1, n1, s2, n2):
        m, n = len(s1), len(s2)
        record = set(s1)
        for i in range(n):
            if s2[i] not in record:
                return 0

        def check(end):
            length = 1
            for i in range(n):
                while s1[end] != s2[i]:
                    end = (end + 1) % m
                    length += 1
                if i == n - 1:
                    return end, length
                end, length = (end + 1) % m, length + 1

        start, start_length = check(0)
        if m * n1 < start_length:
            return 0
        record = [0]
        memory_index = set()
        memory_index.add(start)
        end = (start + 1) % m
        length_sum = 0
        period_s2 = 0
        while True:
            end, length = check(end)
            period_s2 += 1
            length_sum += length
            record.append(record[-1] + length)
            if end not in memory_index:
                memory_index.add(end)
                end = (end + 1) % m
            else:
                break
        div, mod = (m * n1 - start_length) // length_sum, (m * n1 - start_length) % length_sum
        while mod < record[-1]:
            record.pop()
        maxV = div * period_s2 + len(record)
        ans = maxV // n2
        return ans


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class RemoveNodes:
    def removeNodes(self, head):
        stack = []
        while head:
            while stack and stack[-1].val < head.val:
                stack.pop()
            stack.append(head)
            head = head.next
        stack.append(None)
        for i in range(len(stack) - 1):
            stack[i].next = stack[i + 1]
        return stack[0]


# 最笨的写法
class MaximumRows:
    def maximumRows(self, matrix, numSelect):
        m, n = len(matrix), len(matrix[0])
        lines = [0] * m
        for i in range(m):
            for j in range(n):
                if matrix[i][j]:
                    lines[i] += 1
        final_ans = []

        def dfs(index, rest):
            if n - index < rest:
                return
            if not rest:
                ans = 0
                for i in range(m):
                    if not lines[i]:
                        ans += 1
                final_ans.append(ans)
                return
            for i in range(m):
                if matrix[i][index]:
                    lines[i] -= 1
            dfs(index + 1, rest - 1)
            for i in range(m):
                if matrix[i][index]:
                    lines[i] += 1
            dfs(index + 1, rest)

        dfs(0, numSelect)
        return max(final_ans)


# 自创解法，稍微复杂于官解
class CanSeePersonsCount:
    def canSeePersonsCount(self, heights):
        ans = [0] * len(heights)
        n = len(heights)
        stack = []
        record_right_higher = [-1] * n
        for i in range(n):
            while stack and heights[stack[-1]] <= heights[i]:
                record_right_higher[stack[-1]] = i
                stack.pop()
            stack.append(i)
        record_increase = [1] * n
        for i in range(n - 2, -1, -1):
            right_index = record_right_higher[i]
            if right_index > 0 and heights[record_right_higher[i]] > heights[i]:
                record_increase[i] = record_increase[record_right_higher[i]] + 1
        for i in range(n - 1):
            right_index = record_right_higher[i]
            ans[i] = record_increase[i + 1] - record_increase[right_index] + 1
        return ans


# 官方思路
class CanSeePersonsCount2:
    def canSeePersonsCount(self, heights):
        stack = []
        ans = [0] * len(heights)
        for i in range(len(heights) - 1, -1, -1):
            res = 0
            while stack and heights[stack[-1]] < heights[i]:
                res += 1
                stack.pop()
            if stack:
                res += 1
            ans[i] = res
            stack.append(i)
        return ans


class AreaOfMaxDiagonal:
    def areaOfMaxDiagonal(self, dimensions):
        length = 0
        ans = 0
        record = []
        for tancular in dimensions:
            cur = tancular[0] ** 2 + tancular[1] ** 2
            record.append(cur)
            if cur > length:
                length = cur
                ans = tancular[0] * tancular[1]
            elif cur == length:
                ans = max(tancular[0] * tancular[1], ans)
        return ans


class Solution:
    def maxFrequencyElements(self, nums):
        cnt = Counter()
        for num in nums:
            cnt[num] += 1
        maxfreq = -1
        ans = 0
        for key, val in cnt.items():
            if val > maxfreq:
                maxfreq = val
                ans = maxfreq
            elif val == maxfreq:
                ans += maxfreq
        return ans


class BeautifulIndices:
    def beautifulIndices(self, s, a, b, k):
        record1, record2 = [], []
        if len(s) < len(a) or len(s) < len(b):
            return []
        m, n = len(a), len(b)
        start = 0
        for i in range(m - 1, len(s)):
            if s[start: i + 1] == a:
                record1.append(start)
            start += 1
        other = 0
        for j in range(n - 1, len(s)):
            if s[other: j + 1] == b:
                record2.append(other)
            other += 1
        ans = []
        j = 0
        for i in range(len(record1)):
            left, right = record1[i] - k if record1[i] - k >= 0 else 0, record1[i] + k if record1[i] + k < len(
                s) else len(s) - 1
            print(left, right)
            while j < len(record2) and record2[j] < left:
                j += 1
            if j == len(record2):
                return ans
            if record2[j] <= right:
                ans.append(record1[i])
        return ans


class MaximumLength:
    def maximumLength(self, nums):
        record = Counter()
        memory = {}
        for num in nums:
            record[num] += 1
            memory[num] = False
        nums.sort()
        ans = 0
        for i in range(len(nums) - 1, -1, -1):
            if memory[nums[i]]:
                continue
            memory[nums[i]] = True
            current = 1
            record[nums[i]] -= 1
            cur = int(nums[i] ** 0.5)

            while cur * cur == nums[i] and record[cur] > 1:
                current += 2
                record[cur] -= 2
                memory[cur] = True
                nums[i] = cur
                cur = int(nums[i] ** 0.5)
            ans = max(ans, current)
        return ans


class FlowerGame:
    def flowerGame(self, n, m):
        if n == m == 1:
            return 0
        if not n % 2:
            odd_left = even_left = n // 2
        else:
            even_left, odd_left = n // 2, n // 2 + 1
        if not m % 2:
            odd_right = even_right = m // 2
        else:
            even_right, odd_right = m // 2, m // 2 + 1
        return odd_left * even_right + even_left * odd_right


class MinOrAfterOperations:
    def minOrAfterOperations(self, nums, k):
        ans = mask = 0
        for b in range(29, -1, -1):
            mask |= 1 << b
            res = -1
            cnt = 0
            for num in nums:
                res &= mask & num
                if not res:
                    res = -1
                else:
                    cnt += 1
            if cnt > k:
                mask ^= 1 << b
                ans |= 1 << b
        return ans


import heapq


class NumsGame:
    def numsGame(self, nums):
        record1, record2 = [], []
        sum_left, sum_right = 0, 0
        ans = []
        for i in range(len(nums)):
            if (i + 1) % 2:
                heapq.heappush(record1, (nums[i] - i) * (-1))
                top = heapq.heappop(record1)
                heapq.heappush(record2, -1 * top)
                sum_left = sum_left + (nums[i] - i) + top
                sum_right = sum_right - top
                ans.append(sum_right - sum_left - record2[0])
            else:
                heapq.heappush(record2, nums[i] - i)
                top = heapq.heappop(record2)
                heapq.heappush(record1, top * (-1))
                sum_right = sum_right + (nums[i] - i) - top
                sum_left += top
                ans.append(sum_right - sum_left)
        return ans


class StoneGameVI:
    def stoneGameVI(self, aliceValues, bobValues):
        sum_record = [(aliceValues[i] + bobValues[i], i) for i in range(len(aliceValues))]
        sum_record.sort(key=lambda x: -x[0])
        sum_A, sum_B = 0, 0
        i = 0
        while i < len(sum_record):
            if not i % 2:
                sum_A += aliceValues[sum_record[i][1]]
            else:
                sum_B += bobValues[sum_record[i][1]]
            i += 1
        if sum_A == sum_B:
            return 0
        return 1 if sum_A > sum_B else -1


class SostoneGameVII:
    def stoneGameVII(self, stones):
        pre_sum = [0] * (len(stones) + 1)
        for i in range(len(stones)):
            pre_sum[i + 1] = pre_sum[i] + stones[i]
        dp_A = [[[0, 0] for each in range(len(stones))] for _ in range(len(stones) + 1)]
        dp_B = [[[0, 0] for each in range(len(stones))] for _ in range(len(stones) + 1)]
        n = len(stones)
        for i in range(2, len(stones) + 1):
            for j in range(i - 1, len(stones)):
                left = pre_sum[j + 1] - pre_sum[j - i + 2]
                right = pre_sum[j] - pre_sum[j - i + 1]
                # for A
                result1 = left + dp_B[i - 1][j][1] - dp_B[i - 1][j][0]
                result2 = right + dp_B[i - 1][j - 1][1] - dp_B[i - 1][j - 1][0]
                if result1 >= result2:
                    dp_A[i][j][0] = left + dp_B[i - 1][j][1]
                    dp_A[i][j][1] = dp_B[i - 1][j][0]
                else:
                    dp_A[i][j][0] = right + dp_B[i - 1][j - 1][1]
                    dp_A[i][j][1] = dp_B[i - 1][j - 1][0]
                # for B
                result3 = left + dp_A[i - 1][j][1] - dp_A[i - 1][j][0]
                result4 = right + dp_A[i - 1][j - 1][1] - dp_A[i - 1][j - 1][0]
                if result3 >= result4:
                    dp_B[i][j][0] = left + dp_A[i - 1][j][1]
                    dp_B[i][j][1] = dp_A[i - 1][j][0]
                else:
                    dp_B[i][j][0] = right + dp_A[i - 1][j - 1][1]
                    dp_B[i][j][1] = dp_A[i - 1][j - 1][0]
        print(dp_A)
        return dp_A[-1][len(stones) - 1][0] - dp_A[-1][len(stones) - 1][1]


class SostoneGameVIII:
    def stoneGameVII(self, stones):
        pre_sum = [0] * (len(stones) + 1)
        for i in range(len(stones)):
            pre_sum[i + 1] = pre_sum[i] + stones[i]
        dp = [[[0, 0] for each in range(len(stones))] for _ in range(len(stones) + 1)]
        for i in range(2, len(stones) + 1):
            for j in range(i - 1, len(stones)):
                left = pre_sum[j + 1] - pre_sum[j - i + 2]
                right = pre_sum[j] - pre_sum[j - i + 1]
                result1 = left + dp[i - 1][j][1] - dp[i - 1][j][0]
                result2 = right + dp[i - 1][j - 1][1] - dp[i - 1][j - 1][0]
                if result1 >= result2:
                    dp[i][j][0] = left + dp[i - 1][j][1]
                    dp[i][j][1] = dp[i - 1][j][0]
                else:
                    dp[i][j][0] = right + dp[i - 1][j - 1][1]
                    dp[i][j][1] = dp[i - 1][j - 1][0]
        return dp[-1][-1][0] - dp[-1][-1][1]


from collections import deque


class MaxResult:
    def maxResult(self, nums, k):
        q = deque()
        dp = [0] * len(nums)
        for i, value in enumerate(nums):
            if not i:
                q.append((i, value))
                dp[0] = value
                continue
            if q and i - q[0][0] > k:
                q.popleft()
            dp[i] = q[0][1] + value
            if i <= k:
                dp[i] = max(value, dp[i])
            while q and q[-1][1] <= dp[i]:
                q.pop()
            q.append((i, dp[i]))
        return dp[-1]


class MinimumTimeToInitialState:
    def minimumTimeToInitialState(self, word, k):
        n = len(word)
        start = 1
        while start * k <= n:
            rest = n - start * k
            right = word[start * k - 1:]
            if word[:rest] == right:
                return start
            start += 1
        return start


class MaximumSubarraySum:
    def maximumSubarraySum(self, nums, k):
        ans = float('-inf')
        pre_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            pre_sum[i + 1] = pre_sum[i] + nums[i]
        record = {}
        record[nums[0]] = 0
        for i in range(1, len(nums)):
            print(ans)
            if nums[i] - k in record:
                ans = max(ans, pre_sum[i + 1] - pre_sum[record[nums[i] - k]])
            if k + nums[i] in record:
                ans = max(ans, pre_sum[i + 1] - pre_sum[record[k + nums[i]]])
            if nums[i] in record:
                previous = pre_sum[i] - pre_sum[record[nums[i]]]
                if previous <= 0:
                    record[nums[i]] = i
            else:
                record[nums[i]] = i

        return 0 if ans == float('-inf') else ans


class CountMatchingSubarrays:
    def countMatchingSubarrays(self, nums, pattern):
        ans = 0
        n = len(pattern)
        pattern = [-2] + pattern

        m = len(nums)
        next = [0] * (n + 1)
        F = [0] * m
        j = 0
        for i in range(2, n + 1):
            while j > 0 and pattern[j + 1] != pattern[i]:
                j = next[j]
            if pattern[j + 1] == pattern[i]:
                j += 1
            next[i] = j
        j = 0
        for i in range(1, m):
            current = 0
            if nums[i] < nums[i - 1]:
                current = -1
            elif nums[i] > nums[i - 1]:
                current = 1
            while j > 0 and (j == n or current != pattern[j + 1]):
                j = next[j]
            if pattern[j + 1] == current:
                j += 1
            F[i] = j
            if j == n:
                ans += 1
        return ans


class MaxPalindromesAfterOperations:
    def maxPalindromesAfterOperations(self, words):
        odd, even, ans = 0, 0, 0
        memory = Counter()
        for word in words:
            for ch in word:
                memory[ch] += 1
        for key, value in memory.items():
            even += value // 2
            odd += value % 2
        words.sort(key=lambda x: len(x))
        for word in words:
            if len(word) % 2:
                if odd > 0:
                    odd -= 1
                    even -= len(word) // 2
                else:
                    even -= len(word) // 2 + 1
                    odd += 1
            else:
                even -= len(word) // 2
            if even >= 0:
                ans += 1
        return ans


class LongestCommonPrefix:
    def longestCommonPrefix(self, arr1, arr2):
        memory = set()
        ans = 0
        for num in arr1:
            val = str(num)
            cur = ''
            for i in range(len(val)):
                cur += val[i]
                if cur not in memory:
                    memory.add(cur)
        for num in arr2:
            val = str(num)
            cur = ''
            for i in range(len(val)):
                cur += val[i]
                if cur in memory:
                    ans = max(ans, len(cur))
        return ans


import math


class MinimumPossibleSum:
    def minimumPossibleSum(self, n, target):
        small = min(math.ceil((target - 1) / 2), n)
        rest = n - small
        result1 = ((1 + small) * small // 2) % 1000000007
        result2 = ((target + target + rest - 1) * rest // 2) % 1000000007
        return (result1 + result2) % 1000000007


class MaximumHappinessSum:
    def maximumHappinessSum(self, happiness, k):
        happiness.sort(key=lambda x: -x)
        sum = 0
        delta = 0
        for i in range(k):
            if happiness[i] - delta < 0:
                happiness[i] = 0
            else:
                happiness[i] -= delta
            delta += 1
            sum += happiness[i]
        return sum


class MinimumDeletions:
    def minimumDeletions(self, word, k):
        record = [0] * 26
        for char in word:
            record[ord(char) - 97] += 1
        record.sort()
        ans = float('inf')
        pre = 0
        for i in range(len(record)):
            rest = self.cal_delta(i, record, k)
            if rest <= record[i]:
                ans = min(ans, rest + pre)
                return ans
            ans = min(ans, rest + pre)
            pre += record[i]
        return ans

    def cal_delta(self, index, record, k):
        delta = 0
        for i in range(index + 1, len(record)):
            if record[i] - record[index] - k > 0:
                delta += record[i] - record[index] - k
        return delta


class MinOperations:
    def minOperations(self, k):
        ans = k
        for i in range(k, 1, -1):
            start = (1 + i) * i // 2
            rest = math.ceil((k - i) / i)
            ans = min(ans, start + rest)
        return ans


from sortedcontainers import SortedList


class MostFrequentIDs:
    def mostFrequentIDs(self, nums, freq):
        s = SortedList()
        cnt = Counter()
        ans = []
        for x, change in zip(nums, freq):
            if cnt[x] in s:
                s.remove(cnt[x])
            cnt[x] += change
            s.add(cnt[x])
            ans.append(s[-1])
        return ans


class MaxBottlesDrunk:
    def maxBottlesDrunk(self, numBottles, numExchange):
        ans = 0
        emptyBottles = 0
        fullBottles = numBottles
        while True:
            if fullBottles > 0:
                ans += fullBottles
                emptyBottles += fullBottles
                fullBottles = 0
            else:
                if emptyBottles < numExchange:
                    break
                else:
                    emptyBottles -= numExchange
                    fullBottles = 1
                    numExchange += 1
        return ans


class CountAlternatingSubarrays:
    def countAlternatingSubarrays(self, nums):
        nums.append(nums[-1])
        ans = 0
        record = []
        current = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                current += 1
                continue
            else:
                record.append(current)
                current = 1
        for rec in record:
            ans += (1 + rec) * rec // 2
        return ans


class MinimumLevels:
    def minimumLevels(self, possible):
        sum_p = 0
        for i in range(len(possible)):
            sum_p += possible[i] if possible[i] else -1
        cur = 1 if possible[0] else -1
        rest = sum_p - cur
        if cur - rest > 0:
            return 1
        for i in range(1, len(possible) - 1):
            cur += possible[i] if possible[i] else -1
            rest = sum_p - cur
            if cur > rest:
                return i + 1
        return -1


class MinimumSubarrayLength:
    def minimumSubarrayLength(self, nums, k):
        record = {}
        ans = float('inf')
        for i, val in enumerate(nums):
            temp = {}
            temp.update(record)
            temp[val] = i
            for key in record:
                cur = key | val
                if cur in temp:
                    temp[cur] = max(temp[key], temp[cur])
                else:
                    temp[cur] = temp[key]
            for key in temp:
                if key >= k:
                    ans = min(ans, i - temp[key] + 1)
            record = temp
        return ans


def quick_sort(arr):
    quick_sort_recursive(arr, 0, len(arr) - 1)


def quick_sort_recursive(arr, low, high):
    if low < high:
        partitioning_index = partition(arr, low, high)
        quick_sort_recursive(arr, low, partitioning_index - 1)
        quick_sort_recursive(arr, partitioning_index + 1, high)


def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


class GetSmallestString:
    def getSmallestString(self, s, k):
        cur = 0
        ans = ""
        for i in range(len(s)):
            for chance in range(26):
                current_ord = 97 + chance
                min_dis = min(abs(ord(s[i]) - current_ord), 26 - abs(ord(s[i]) - current_ord))
                if cur + min_dis <= k:
                    cur += min_dis
                    ans += chr(current_ord)
                    break
        return ans


class MinOperationsToMakeMedianK:
    def minOperationsToMakeMedianK(self, nums, k):
        n = len(nums)
        nums.sort()
        mid_index = n // 2
        mid_val = nums[mid_index]
        ans = 0
        if mid_val > k:
            while mid_index >= 0 and nums[mid_index] > k:
                ans += nums[mid_index] - k
                mid_index -= 1
        elif mid_val < k:
            while mid_index < n and nums[mid_index] < k:
                ans += k - nums[mid_index]
                mid_index += 1
        return ans


class MinimumCost:
    def minimumCost(self, n, edges, query):
        ans = [-1] * len(query)
        self.fa = [i for i in range(n)]
        self.record = {i: 1048575 for i in range(n)}
        self.out_edges = [{} for i in range(n)]
        for edge in edges:
            if edge[1] not in self.out_edges[edge[0]]:
                self.out_edges[edge[0]][edge[1]] = edge[2]
            else:
                self.out_edges[edge[0]][edge[1]] &= edge[2]
            if edge[0] not in self.out_edges[edge[1]]:
                self.out_edges[edge[1]][edge[0]] = edge[2]
            else:
                self.out_edges[edge[1]][edge[0]] &= edge[2]
        print(self.out_edges)
        self.visited = [False] * n
        for i in range(n):
            if not self.visited[i]:
                self.dfs(i)
        print(self.fa)
        print(self.record)
        for i, q in enumerate(query):
            x, y = q[0], q[1]
            x, y = self.find(x), self.find(y)
            if x == y:
                ans[i] = self.record[x]
        return ans

    def find(self, x):
        if x == self.fa[x]:
            return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.fa[y] = x

    def dfs(self, x):
        self.visited[x] = True
        cur = self.out_edges[x]
        for child, weight in cur.items():
            self.union(x, child)
            fa = self.find(x)
            self.record[fa] &= weight
            if not self.visited[child]:
                self.dfs(child)


import asyncio
import time

# async def blocking_io():
#     print("Blocking IO started")
#     await asyncio.sleep(2)
#     print("Blocking IO finished")
#
# async def another_task():
#     print("Another task started")
#     await asyncio.sleep(2)
#     print("Another task finished")
#
# async def main():
#     print("Main started")
#     task1 = asyncio.create_task(another_task())
#     await blocking_io()
#     await task1
#     print("Main finished")

# asyncio.run(main())

import asyncio


async def worker1():
    print("start worker1")
    print("finish worker1")


async def worker2():
    print("start worker2")
    time.sleep(3)
    # await worker1()
    print("finish worker2")


async def worker3():
    print("start worker3")
    print("finish worker3")


async def main():
    print("start main")
    task3 = asyncio.create_task(worker3())
    task1 = asyncio.create_task(worker1())
    await worker2()
    # await asyncio.sleep(3)
    print("finish main")


def mergeSort(l, r, nums):
    if l >= r:
        return
    mid = (l + r) >> 1
    mergeSort(l, mid, nums)
    mergeSort(mid + 1, r, nums)
    merge(l, mid, r, nums)


def merge(l, mid, r, nums):
    temp = []
    i, j = l, mid + 1
    while i <= mid and j <= r:
        if nums[i] <= nums[j]:
            temp.append(nums[i])
            i += 1
        else:
            temp.append(nums[j])
            j += 1
    while i <= mid:
        temp.append(nums[i])
        i += 1
    while j <= r:
        temp.append(nums[j])
        j += 1
    nums[l: r + 1] = temp


def actual_sort(nums):
    mergeSort(0, len(nums) - 1, nums)


class MinimumOperations:
    def minimumOperations(self, grid):
        m, n = len(grid), len(grid[0])
        record = [[0] * 10 for _ in range(n)]
        for j in range(n):
            for i in range(m):
                val = grid[i][j]
                record[j][val] += 1
        for i in range(len(record)):
            for j in range(len(record[0])):
                record[i][j] = m - record[i][j]
        dp = [[float('inf')] * 10 for _ in range(n)]
        for j in range(10):
            dp[0][j] = record[0][j]
        for i in range(1, n):
            for j in range(10):
                for k in range(10):
                    if k != j:
                        dp[i][j] = min(dp[i - 1][k] + record[i][j], dp[i][j])
        return min(dp[-1])


class NumberOfSpecialChars:
    def numberOfSpecialChars(self, word):
        ans = 0
        record = {}
        for i in range(len(word)):
            val = ord(word[i])
            if val in record and val >= 97:
                record[val] = i
            elif val not in record:
                record[val] = i
        for key, val in record.items():
            if key >= 97 and key - 32 in record and val < record[key - 32]:
                ans += 1
        return ans


import heapq


class FindAnswer:
    def findAnswer(self, n, edges):
        ans = [False] * len(edges)
        out_edges = [[] for _ in range(n)]
        for edge in edges:
            out_edges[edge[0]].append([edge[1], edge[2]])
            out_edges[edge[1]].append([edge[0], edge[2]])

        def cal(node):
            dist = {i: float('inf') for i in range(n)}
            memory = set()
            dist[node] = 0
            q = []
            heapq.heappush(q, (0, node))
            while q:
                top = heapq.heappop(q)
                cur_node, cur_weight = top[1], top[0]
                if cur_node in memory:
                    continue
                memory.add(cur_node)
                for edge in out_edges[cur_node]:
                    target, weight = edge[0], edge[1]
                    if dist[target] > dist[cur_node] + weight:
                        dist[target] = dist[cur_node] + weight
                        heapq.heappush(q, (dist[target], target))
            return dist

        dist0 = cal(0)
        distN = cal(n - 1)
        min_dis = dist0[n - 1]
        if min_dis == float('inf'):
            return ans
        for index, edge in enumerate(edges):
            node1, node2, weight = edge[0], edge[1], edge[2]
            judge1 = dist0[node1] + distN[node2] + weight
            if judge1 == min_dis and min_dis < float('inf'):
                ans[index] = True
                continue
            judge2 = dist0[node2] + distN[node1] + weight
            if judge2 == min_dis and min_dis < float('inf'):
                ans[index] = True
                continue
        return ans


class MinRectanglesToCoverPoints:
    def minRectanglesToCoverPoints(self, points, w):
        points.sort(key=lambda x: x[0])
        ans, last = 1, points[0][0] + w
        for point in points:
            if point[0] <= last:
                continue
            ans += 1
            last = point[0] + w
        return ans


class MinimumTime:
    def minimumTime(self, n, edges, disappear):
        ans = [float('inf')] * n
        out_edges = [[] for _ in range(n)]
        for edge in edges:
            out_edges[edge[0]].append([edge[1], edge[2]])
            out_edges[edge[1]].append([edge[0], edge[2]])
        q = []
        heapq.heappush(q, (0, 0))
        ans[0] = 0
        memory_add = set()
        while q:
            top = heapq.heappop(q)
            cur_node, weight = top[1], top[0]
            if cur_node in memory_add:
                continue
            memory_add.add(cur_node)
            for edge in out_edges[cur_node]:
                target, weight = edge[0], edge[1]
                new_dis = ans[cur_node] + weight
                if ans[target] > new_dis and disappear[target] > new_dis:
                    ans[target] = new_dis
                    heapq.heappush(q, (new_dis, target))
        for i, v in enumerate(ans):
            if v == float('inf'):
                ans[i] = -1
        return ans


class MinimumAddedInteger:
    def minimumAddedInteger(self, nums1, nums2):
        nums1.sort()
        nums2.sort()
        n1, n2 = len(nums1), len(nums2)

        def find_min_x(start):
            count = 0
            judge = 2 - start
            x = nums2[0] - nums1[start]
            j = start + 1
            for i in range(1, n2):
                while nums2[i] - nums1[j] != x:
                    count += 1
                    if count > judge:
                        return False
                    j += 1
                j += 1
            return True

        ans = float('inf')
        for start in range(3):
            if find_min_x(start):
                ans = min(ans, nums2[0] - nums1[start])
        return ans


class MinEnd:
    def minEnd(self, n, x):
        j = 0
        for i in range(32):
            while (1 << j) & x:
                j += 1
            if (1 << i) & n:
                x |= (1 << j)
            j += 1
        return x


class NumberOfSubarrays:
    def numberOfSubarrays(self, nums):
        segmentTree = SegmentTree(nums)
        ans = len(nums)
        last_index = {nums[0]: 0}
        record = {nums[0]: 0}
        for i in range(1, len(nums)):
            if nums[i] not in last_index:
                last_index[nums[i]] = i
                record[nums[i]] = 0
                continue
            cur_area_max = segmentTree.query(last_index[nums[i]], i, 1)
            if cur_area_max <= nums[i]:
                record[nums[i]] = record[nums[i]] + 1
                ans += record[nums[i]]
            else:
                record[nums[i]] = 0
            last_index[nums[i]] = i
        return ans


class Node:
    def __init__(self):
        self.l, self.r, self.maxV = 0, 0, None


class SegmentTree:
    def __init__(self, nums):
        n = len(nums)
        self.segmentTree = [Node() for _ in range(4 * n)]
        self.build_tree(0, n - 1, nums, 1)

    def build_tree(self, l, r, nums, index):
        node = self.segmentTree[index]
        node.l, node.r = l, r
        if l == r:
            node.maxV = nums[l]
            return
        mid = (l + r) >> 1
        self.build_tree(l, mid, nums, index * 2)
        self.build_tree(mid + 1, r, nums, index * 2 + 1)
        node.maxV = max(self.segmentTree[index * 2].maxV, self.segmentTree[index * 2 + 1].maxV)

    def query(self, l, r, index):
        cur_node = self.segmentTree[index]
        ans = float('-inf')
        if l <= cur_node.l and r >= cur_node.r:
            return cur_node.maxV
        mid = (cur_node.l + cur_node.r) >> 1
        if mid >= l:
            ans = max(ans, self.query(l, r, index * 2))
        if r > mid:
            ans = max(ans, self.query(l, r, index * 2 + 1))
        return ans


class NumberOfSubarrays2:
    def numberOfSubarrays(self, nums):
        stack = []
        ans = len(nums)
        last_index = {}
        record = {}
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if nums[i] not in last_index or (stack and stack[-1] > last_index[nums[i]]) or not stack:
                record[nums[i]] = 0
                last_index[nums[i]] = i
                stack.append(i)
                continue
            record[nums[i]] += 1
            ans += record[nums[i]]
            last_index[nums[i]] = i
            stack.append(i)
        return ans


class NumberOfRightTriangles:
    def numberOfRightTriangles(self, grid):
        rows, cols = [0] * len(grid), [0] * len(grid[0])
        memory = []
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    rows[i] += 1
                    cols[j] += 1
                    memory.append((i, j))
        for point in memory:
            i, j = point[0], point[1]
            ans += (rows[i] - 1) * (cols[j] - 1)
        return ans


class NumberOfStableArrays:
    def numberOfStableArrays(self, zero, one, limit):
        mol = 1000000007
        dp = [[[0, 0] for j in range(zero + 1)] for i in range(one + 1)]
        # 初始化
        for j in range(1, zero + 1):
            if j > limit:
                break
            dp[0][j][0] = 1
        for i in range(1, one + 1):
            if i > limit:
                break
            dp[i][0][1] = 1
        for i in range(1, one + 1):
            for j in range(1, zero + 1):
                delta0 = 0 if j - limit - 1 < 0 else dp[i][j - limit - 1][1]
                delta1 = 0 if i - limit - 1 < 0 else dp[i - limit - 1][j][0]
                dp[i][j][0] = (dp[i][j - 1][0] + dp[i][j - 1][1] - delta0) % mol
                dp[i][j][1] = (dp[i - 1][j][1] + dp[i - 1][j][0] - delta1) % mol
        return sum(dp[one][zero]) % mol


class MaxPointsInsideSquare:
    def maxPointsInsideSquare(self, points, s):
        record = [[] for _ in range(26)]
        for i, char in enumerate(s):
            index = ord(char) - 97
            x, y = points[i][0], points[i][1]
            val = max(abs(x), abs(y))
            if not record[index]:
                record[index].append(val)
                continue
            if len(record[index]) == 1:
                if record[index][0] <= val:
                    record[index].append(val)
                else:
                    tail = record[index].pop()
                    record[index].append(val)
                    record[index].append(tail)
            elif len(record[index]) == 2:
                if val < record[index][0]:
                    record[index].pop()
                    tail = record[index].pop()
                    record[index].append(val)
                    record[index].append(tail)
                elif record[index][0] <= val < record[index][1]:
                    record[index].pop()
                    record[index].append(val)
        val = 1000000000
        for rec in record:
            if len(rec) == 2:
                val = min(val, rec[1])
        ans = 0
        for point in points:
            x, y = point[0], point[1]
            cur = max(abs(x), abs(y))
            if val > cur:
                ans += 1
        return ans


class MinimumSubstringsInPartition:
    def minimumSubstringsInPartition(self, s):
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0
        for i in range(1, len(s) + 1):
            record = [0] * 26
            for j in range(i, 0, -1):
                record[ord(s[j - 1]) - 97] += 1
                last = 0
                flag = True
                for k in range(26):
                    if record[k]:
                        if not last:
                            last = record[k]
                        elif last != record[k]:
                            flag = False
                            break
                if flag:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
        return dp[-1]


class MaximumEnergy:
    def maximumEnergy(self, energy, k):
        n = len(energy)
        ans = -float('inf')
        for i in range(k):
            cur = energy[i]
            j = i
            while j + k < n:
                if cur > 0:
                    cur += energy[j + k]
                else:
                    cur = energy[j + k]
                j += k
            ans = max(ans, cur)
        return ans


class MaxScore:
    def maxScore(self, grid):
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        ans = float('-inf')
        for j in range(1, n):
            dp[0][j] = min(dp[0][j - 1], grid[0][j])
            ans = max(ans, grid[0][j] - dp[0][j - 1])
        for i in range(1, m):
            dp[i][0] = min(dp[i - 1][0], grid[i][0])
            ans = max(ans, grid[i][0] - dp[i - 1][0])
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], grid[i][j])
                ans = max(grid[i][j] - dp[i - 1][j], grid[i][j] - dp[i][j - 1], ans)
        return ans


class IsArraySpecial:
    def isArraySpecial(self, nums, queries):
        ans = [False] * len(queries)
        odd, even = {}, {}
        last_odd, last_even = -1, -1
        for i, val in enumerate(nums):
            if last_odd >= 0 and last_even >= 0:
                break
            if val % 2:
                last_odd = i
            else:
                last_even = i
        for i, val in enumerate(nums):
            if val % 2:
                if i - last_odd != 2:
                    odd[i] = i
                else:
                    odd[i] = odd[last_odd]
            else:
                if i - last_even != 2:
                    even[i] = i
                else:
                    even[i] = even[last_even]
            if val % 2:
                last_odd = i
            else:
                last_even = i
        for i, query in enumerate(queries):
            if nums[query[1]] % 2:
                last = odd[query[1]]
                if last - 1 >= 0 and not nums[last - 1] % 2:
                    last -= 1
                ans[i] = last <= query[0]
            else:
                last = even[query[1]]
                if last - 1 >= 0 and nums[last - 1] % 2:
                    last -= 1
                ans[i] = last <= query[0]
        return ans


class SumDigitDifferences:
    def sumDigitDifferences(self, nums):
        ans = 0
        digits = len(str(nums[0]))
        cur_len = 0
        record = [[0] * 10 for _ in range(digits)]
        for i, val in enumerate(nums):
            cur_index = digits - 1
            while val:
                cur = val % 10
                ans += cur_len - record[cur_index][cur]
                record[cur_index][cur] += 1
                cur_index -= 1
                val //= 10
            cur_len += 1
        return ans


import math


class WaysToReachStair:
    def waysToReachStair(self, k):
        if k - 1 > 7:
            time = 4
            while 2 ** time < k:
                time += 1
            if 2 ** time - k >= 0:
                reduce = 2 ** time - k
                return math.comb(time + 1, reduce)
        else:
            return self.cal(k)

    def cal(self, k):
        if not k:
            return 2
        if k == 1:
            return 4
        ans = 0
        time = 2
        while 2 ** time < k:
            time += 1
        while time < 4:
            reduce = 2 ** time - k
            ans += math.comb(time + 1, reduce)
            time += 1
        return ans


class LongestEqualSubarray:
    def longestEqualSubarray(self, nums, k):
        ans = 1
        # 先记录一下每一个元素值的索引位置
        record = {}
        for i, val in enumerate(nums):
            if val not in record:
                record[val] = [i]
            else:
                record[val].append(i)
        # 这个字典是{元素值：三元组}的形式，代表遍历数组的过程中，对于以当前元素值为右端点，决定可以成立的最长左端点，其实就是滑动窗口
        # 第一个元素记录起始位置，第二个元素记录右端点，第三个元素记录已经删除的个数
        memory = {}
        for i, val in enumerate(nums):
            if val not in memory:
                memory[val] = (0, 0, 0)
                continue
            start, lastEnd, cost = memory[val]
            index_end = record[val][lastEnd]
            # 提前判断一下，如果最新的右端点与上一个右端点差值已经大于k了，那证明需要把所有前面的删除掉，要以当前点为全新开始
            if i - index_end - 1 > k:
                memory[val] = (lastEnd + 1, lastEnd + 1, 0)
                continue
            # 开始左端点往右移，直到删除个数 <= k
            while i - index_end - 1 + cost > k:
                cost -= record[val][start + 1] - record[val][start] - 1
                start += 1
            # 更新 memory
            memory[val] = (start, lastEnd + 1, cost + i - index_end - 1)
            # 更新答案
            ans = max(ans, memory[val][1] - memory[val][0] + 1)
        return ans


class MostCompetitive:
    def mostCompetitive(self, nums, k):
        stack = []
        for i, val in enumerate(nums):
            while stack and stack[-1] > val and len(stack) + len(nums) - i - 1 >= k:
                stack.pop()
            stack.append(val)
        return stack[:k]


class FindWinners:
    def findWinners(self, matches):
        winners, losers = {}, {}
        ans = [[], []]
        for i, match in enumerate(matches):
            if match[1] not in losers:
                losers[match[1]] = 1
            else:
                losers[match[1]] += 1
            if match[0] not in winners:
                winners[match[0]] = 1
            else:
                winners[match[0]] += 1
        for k, v in winners.items():
            if k not in losers:
                ans[0].append(k)
        for k, v in losers.items():
            if v == 1:
                ans[1].append(k)
        ans[0].sort()
        ans[1].sort()
        return ans


class CountDays:
    def countDays(self, days, meetings):
        ans = days - meetings[0][1] + meetings[0][0] - 1
        meetings.sort(key=lambda x: x[0])
        last = meetings[0][1]
        for i in range(1, len(meetings)):
            if meetings[i][0] <= last:
                if meetings[i][1] > last:
                    ans -= meetings[i][1] - last
                    last = meetings[i][1]
            else:
                ans -= meetings[i][1] - meetings[i][0] - 1
                last = meetings[i][1]
        return ans


class ClearStars:
    def clearStars(self, s):
        record = [[] for _ in range(26)]
        delete = set()
        ans = ''
        for i, val in enumerate(s):
            index = ord(val) - 97
            if val != "*":
                record[index].append(i)
                continue
            for j, inner in enumerate(record):
                if inner:
                    delete.add(inner.pop())
                    break
        for i, val in enumerate(s):
            if val == '*' or i in delete:
                continue
            ans += val
        return ans


# from collections import deque
class MaxTotalReward:
    def maxTotalReward(self, rewardValues):
        rewardValues.sort()
        new = []
        for val in rewardValues:
            if new and val == new[-1]:
                continue
            new.append(val)
        memory = set()
        memory.add(new[0])
        record = [0, new[0]]
        for i in range(1, len(new)):
            temp = []
            for val in record:
                if val + new[i] not in memory and val < new[i]:
                    temp.append(val + new[i])
                    memory.add(val + new[i])
            record += temp
        ans = -1
        for ele in memory:
            ans = max(ans, ele)
        return ans


class FindWinningPlayer:
    def findWinningPlayer(self, skills, k):
        record = [0] * len(skills)
        skills.append(1000001)
        stack = []
        for i, val in enumerate(skills):
            while stack and skills[stack[-1]] < val:
                record[stack[-1]] = i - stack[-1] - 1
                stack.pop()
            stack.append(i)
        maxV = min(k, len(record) - 1)
        i = record[0] + 1
        if i - 1 >= maxV:
            return 0
        while i < len(record):
            if record[i] + 1 >= maxV or i + record[i] >= len(record) - 1:
                return i
            i += record[i] + 1
        return i


class MaximumLength:
    def maximumLength(self, nums, k):
        dp = [[1] * (k + 1) for _ in range(len(nums))]
        for i in range(1, len(nums)):
            for inner in range(i):
                if nums[i] == nums[inner]:
                    dp[i][0] = max(dp[i][0], dp[inner][0] + 1)
        for i in range(1, len(nums)):
            for j in range(1, k + 1):
                dp[i][j] = dp[i][j - 1]
                for inner in range(i):
                    if nums[i] != nums[inner]:
                        dp[i][j] = max(dp[i][j], dp[inner][j - 1] + 1)
                    else:
                        dp[i][j] = max(dp[i][j], dp[inner][j] + 1)
        ans = 0
        for i in range(len(dp)):
            ans = max(ans, dp[i][-1])
        return ans


class CountCompleteDayPairs:
    def countCompleteDayPairs(self, hours):
        memory = [0] * 24
        ans = 0
        for i, val in enumerate(hours):
            cur = val % 24
            if cur:
                ans += memory[24 - cur]
                memory[cur] += 1
                continue
            ans += memory[0]
            memory[0] += 1
        return ans


class MaximumTotalDamage:
    def maximumTotalDamage(self, power):
        power.sort()
        record = {power[0]: power[0]}
        list_record = [(0, 0)] * 3
        for i in range(1, len(power)):
            if power[i] == power[i - 1]:
                record[power[i]] += power[i]
            else:
                record[power[i]] = power[i]
                list_record.append((power[i - 1], record[power[i - 1]]))
        list_record.append((power[len(power) - 1], record[power[len(power) - 1]]))
        dp = [0] * len(list_record)
        for i in range(3, len(dp)):
            dp[i] = dp[i - 1]
            val = list_record[i][0]
            cur = (val - 1, val - 2)
            if list_record[i - 1][0] not in cur or not list_record[i - 1][0]:
                dp[i] = max(dp[i], dp[i - 1] + list_record[i][1])
                continue
            if list_record[i - 2][0] not in cur or not list_record[i - 2][0]:
                dp[i] = max(dp[i], dp[i - 2] + list_record[i][1])
                continue
            dp[i] = max(dp[i], dp[i - 3] + list_record[i][1])
        return dp[-1]


class SpiralOrder:
    def spiralOrder(self, matrix):
        m, n = len(matrix), len(matrix[0])
        ans = []
        left_up, right_up, left_bottom, right_bottom = [0, 0], [0, n - 1], [m - 1, 0], [m - 1, n - 1]
        while m > 1 and n > 1:
            for j in range(left_up[1], right_up[1]):
                ans.append(matrix[left_up[0]][j])
            for i in range(right_up[0], right_bottom[0]):
                ans.append(matrix[i][right_bottom[1]])
            for j in range(right_bottom[1], left_bottom[1], -1):
                ans.append(matrix[right_bottom[0]][j])
            for i in range(left_bottom[0], left_up[0], -1):
                ans.append(matrix[i][left_bottom[1]])
            m -= 2
            n -= 2
            left_up[0] += 1
            left_up[1] += 1
            left_bottom[0] -= 1
            left_bottom[1] += 1
            right_up[0] += 1
            right_up[1] -= 1
            right_bottom[0] -= 1
            right_bottom[1] -= 1
        if m == 1:
            for j in range(left_up[1], right_up[1] + 1):
                ans.append(matrix[left_up[0]][j])
            return ans
        if n == 1:
            for i in range(left_up[0], left_bottom[0] + 1):
                ans.append(matrix[i][left_up[1]])
            return ans
        return ans


class GameOfLife:
    def gameOfLife(self, board):
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                sur = (
                    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j),
                    (i + 1, j + 1))
                count = 0
                for ele in sur:
                    if 0 <= ele[0] < m and 0 <= ele[1] < n:
                        if board[ele[0]][ele[1]] > 0:
                            count += 1
                if board[i][j]:
                    if count < 2 or count > 3:
                        board[i][j] += 1
                elif count == 3:
                    board[i][j] -= 1
        for i in range(m):
            for j in range(n):
                if board[i][j] < 0:
                    board[i][j] = 1
                elif board[i][j] > 1:
                    board[i][j] = 0


class Rotate:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        count = len(matrix) // 2
        n = len(matrix)
        start = [0, 0]
        while count > 0:
            length = n - start[1]
            for j in range(start[1], length - 1):
                x, y = start[0], j
                after = matrix[x][y]
                for k in range(4):
                    x, y = y, n - 1 - x
                    previous = matrix[x][y]
                    matrix[x][y] = after
                    after = previous
            start[0] += 1
            start[1] += 1
            count -= 1


class SetZeroes:
    def setZeroes(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        rows = [False] * m
        cols = [False] * n
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows[i] = True
                    cols[j] = True
        for i in range(m):
            for j in range(n):
                if rows[i] or cols[j]:
                    matrix[i][j] = 0


class RemoveKdigits:
    def removeKdigits(self, num, k):
        stack = []
        for i, char in enumerate(num):
            while stack and int(char) < int(stack[-1]) and k > 0:
                stack.pop()
                k -= 1
            stack.append(char)
        if k > 0:
            stack = stack[:-k]
        i = 0
        while i < len(stack) and stack[i] == '0':
            i += 1
        if i == len(stack):
            return '0'
        return ''.join(stack[i:])


class RemoveDuplicateLetters:
    def removeDuplicateLetters(self, s):
        record = {}
        for c in s:
            if c in record:
                record[c] += 1
            else:
                record[c] = 1
        stack = []
        memory = set()
        for i, char in enumerate(s):
            if char in memory:
                continue
            while stack and ord(stack[-1]) > ord(char) and record[stack[-1]] > 1:
                record[stack[-1]] -= 1
                stack.pop()
            stack.append(char)
            memory.add(char)
        return ''.join(stack)


class MinimumArea:
    def minimumArea(self, grid):
        m, n = len(grid), len(grid[0])
        left, right = n - 1, 0
        up, down = m - 1, 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    left = min(left, j)
                    right = max(right, j)
                    up = min(up, i)
                    down = max(down, i)
        return (up - down - 1) * (right - left - 1)


class MaximumTotalCost:
    def maximumTotalCost(self, nums):
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0] - nums[1], nums[1] + nums[0])
        for i in range(2, len(nums)):
            dp[i] = max(nums[i - 1] - nums[i] + dp[i - 2], nums[i] + dp[i - 1])
        return dp[-1]


class MinOperations1:
    def minOperations(self, nums):
        ans = 0
        for i in range(len(nums)):
            if not nums[i]:
                if i + 2 >= len(nums):
                    return -1
                nums[i + 1] = (nums[i + 1] + 1) % 2
                nums[i + 2] = (nums[i + 2] + 1) % 2
                ans += 1
        return ans


class MinOperations2:
    def minOperations(self, nums):
        record = [0]
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                record.append(i)
        if not nums[record[0]]:
            return len(record)
        return len(record) - 1


class MaximumLength1:
    def maximumLength(self, nums, k):
        nums = [0] + nums
        dp = [[0] * k for _ in range(len(nums))]
        memory = {i: 0 for i in range(k)}
        ans = 0
        for i in range(1, len(nums)):
            for j in range(k):
                if nums[i] % k <= j and j - nums[i] % k in memory:
                    dp[i][j] = dp[memory[j - nums[i] % k]][j] + 1
                if j + k - nums[i] % k in memory:
                    dp[i][j] = max(dp[memory[j + k - nums[i] % k]][j] + 1, dp[i][j])
                ans = max(ans, dp[i][j])
            memory[nums[i] % k] = i
        return ans


from collections import deque


class MinimumDiameterAfterMerge:
    def minimumDiameterAfterMerge(self, edges1, edges2):
        n, m = len(edges1) + 1, len(edges2) + 1
        out_edges1 = [[] for _ in range(n)]
        out_edges2 = [[] for _ in range(m)]
        in_degree1 = [0] * n
        in_degree2 = [0] * m
        for edge in edges1:
            out_edges1[edge[0]].append(edge[1])
            out_edges1[edge[1]].append(edge[0])
            in_degree1[edge[0]] += 1
            in_degree1[edge[1]] += 1
        for edge in edges2:
            out_edges2[edge[0]].append(edge[1])
            out_edges2[edge[1]].append(edge[0])
            in_degree2[edge[0]] += 1
            in_degree2[edge[1]] += 1

        def topSort(out_edge, in_degree):
            memory = set()
            q = deque()
            layer = 0
            for i, val in enumerate(in_degree):
                if val == 1:
                    q.append((i, layer))
                    memory.add(i)
            while q:
                if len(memory) == len(in_degree) and len(q) == 2 and q[0][1] == q[1][1]:
                    return q[0][1] + 1, q[0][1] * 2 + 1
                top, layer = q.popleft()
                for node in out_edge[top]:
                    if node in memory:
                        continue
                    in_degree[node] -= 1
                    if in_degree[node] == 1:
                        layer += 1
                        q.append((node, layer))
                        memory.add(node)
            return layer, layer * 2

        layer1, maxlength1 = topSort(out_edges1, in_degree1)
        layer2, maxlength2 = topSort(out_edges2, in_degree2)
        return max(layer2 + layer1 + 1, maxlength1, maxlength2)


class MinimumCost01:
    def minimumCost(self, m, n, horizontalCut, verticalCut):
        for i, val in enumerate(horizontalCut):
            horizontalCut[i] = (val, i, 0)
        for i, val in enumerate(verticalCut):
            verticalCut[i] = (val, i, 1)
        record = horizontalCut + verticalCut
        record.sort(key=lambda x: -x[0])
        rowCount, colCount, ans = 1, 1, 0
        for i, val in enumerate(record):
            if val[2]:
                rowCount += 1
                ans += colCount * val[0]
            else:
                colCount += 1
                ans += rowCount * val[0]
        return ans


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class ModifiedList:
    def modifiedList(self, nums, head):
        memory = set(nums)
        start = ListNode(val=-1, next=head)
        cur = start
        while cur:
            while cur.next and cur.next.val in memory:
                cur.next = cur.next.next
            cur = cur.next
        return start.next

