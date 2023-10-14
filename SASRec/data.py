import pandas as pd


class Data:
    def __init__(self, path):
        self.path = path
        self.user_nums, self.item_nums = None, None

    def data_partition(self):
        with open(self.path, 'r') as f:
            for line in f.readlines():
                pass