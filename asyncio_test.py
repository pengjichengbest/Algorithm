# coding:utf-8
"""
    Author: apple
    Date: 15/6/2024
    File: asyncio_test.py
    ProjectName: Algorithm
    Time: 15:28
"""

import threading
from queue import Queue

# class ThreadBot(threading.Thread):
#     def __init__(self):
#         super().__init__(target=self.manage_table)
#         self.cutlery = Cutlery(knives=0, forks=0)
#         self.tasks = Queue()
#
#     def manage_table(self):
#         while True:
#             task = self.tasks.get()
#             if task == 'prepare table':
#                 kitchen.give(to=self.cutlery, knives=4, forks=4)
#             elif task == 'clear table':
#                 self.cutlery.give(to=kitchen, knives=4, forks=4)
#             elif task == 'shutdown':
#                 return
#
#
# from attr import attrs, attrib
#
#
# @attrs
# class Cutlery:
#     knives = attrib(default=0)
#     forks = attrib(default=0)
#
#     def give(self, to: 'Cutlery', knives=0, forks=0):
#         self.change(-knives, -forks)
#         to.change(knives, forks)
#
#     def change(self, knives, forks):
#         self.knives += knives
#         self.forks += forks
#
#
# kitchen = Cutlery(knives=100, forks=100)
# bots = [ThreadBot() for i in range(10)]
#
# import sys
#
# for bot in bots:
#     for i in range(int(sys.argv[1])):
#         bot.tasks.put('prepare table')
#         bot.tasks.put('clear table')
#     bot.tasks.put('shutdown')
# print('Kitchen inventory before service', kitchen)
# for bot in bots:
#     bot.start()
# for bot in bots:
#     bot.join()
# print('Kitchen inventory after service', kitchen)


import asyncio
import time


async def compute_intensive_task():
    await test()
    # 模拟一个计算密集型任务
    start_time = time.time()
    while time.time() - start_time < 5:
        pass  # 计算密集型任务，CPU一直在工作
    print("Compute intensive task completed")


async def io_bound_task():
    print("IO bound task started")
    await asyncio.sleep(2)  # 模拟一个IO操作
    print("IO bound task completed")


async def test():
    print("test start")
    await asyncio.sleep(4)
    print("test task completed")


async def main():
    task1 = asyncio.create_task(compute_intensive_task())
    task2 = asyncio.create_task(io_bound_task())

    await task1
    await task2


if __name__ == "__main__":
    asyncio.run(main())
