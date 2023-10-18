import pandas as pd
import numpy as np
from collections import defaultdict

# 处理的大概思路没问题，但是处理后得到的数据与参考代码的数据仍然不同
# 所以直接采样参考代码的数据，该文件提供处理思路


# 将原始数据item_id进行映射
def mapping(df):
    mapping_dict = {}

    item_ids = df['item_id'].values

    for item_id in item_ids:
        # 先按照参考代码的数据，从1开始映射
        if item_id not in mapping_dict.keys():
            mapping_dict[item_id] = len(mapping_dict) + 1

    mapping_data = []

    for idx, row in df.iterrows():
        user_id, item_id = row['user_id'], row['item_id']
        mapping_data.append([user_id, mapping_dict[item_id]])

    mapping_data = np.array(mapping_data)

    return mapping_data


# discard users and items with fewer than 5 related actions
def remove_less_than_5(data):
    u_i_dict, i_u_dict = defaultdict(list), defaultdict(list)

    # 需要被移除的用户和物品
    removed_user, removed_item = [], []

    for i in range(len(data)):
        user, item, timestamp = data[i, :]
        u_i_dict[user].append(item)
        i_u_dict[item].append(user)

    for (k1, v1), (k2, v2) in zip(u_i_dict.items(), i_u_dict.items()):
        if len(v1) < 5:
            removed_user.append(k1)

        if len(v2) < 5:
            removed_item.append(k2)

    return removed_user, removed_item


# 预处理原始的rating.dat文件
def pre_processing(file_path, save_path):
    # 不需要记录分数，只要是出现在rating.rat文件中的数据都是implicit feedback
    data = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, item_id, rating, timestamp = line.split("::")
            data.append([int(user_id), int(item_id), int(timestamp)])

    data = np.array(data)
    removed_user, removed_item = remove_less_than_5(data)

    user_nums, item_nums = 0, 0
    user_ids, item_ids, timestamps = [], [], []
    for i in range(len(data)):
        user, item, timestamp = data[i, :]
        if user in removed_user or item in removed_item:
            continue
        user_ids.append(user)
        item_ids.append(item)
        timestamps.append(timestamp)

        user_nums = max(user_nums, user)
        item_nums = max(item_nums, item)

    # 将数据存入dataframe中，根据user_id和timestamp进行排序
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'timestamp': timestamps
    })

    sorted_df = df.sort_values(by=['user_id', 'timestamp'], ascending=True)
    # print(sorted_df)
    mapping_data = mapping(sorted_df)

    print(max(mapping_data[:, 1]))

    # save
    np.savetxt(save_path, mapping_data, fmt='%d', delimiter=' ')


file_path = '../data/ml-1m/ratings.dat'
save_path = './my_processed_data/ml-1m.txt'
pre_processing(file_path, save_path)
