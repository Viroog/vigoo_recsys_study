import pandas as pd
import numpy as np

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

    # for i in range(len(df)):
    #     _, item_id, _ = df.iloc[i, :]
    #     print(item_id)
    #     # 先按照参考代码的数据，从1开始映射
    #     if item_id not in mapping_dict.keys():
    #         mapping_dict[item_id] = len(mapping_dict) + 1
    #
    mapping_data = []

    for idx, row in df.iterrows():
        user_id, item_id = row['user_id'], row['item_id']
        mapping_data.append([user_id, mapping_dict[item_id]])

    mapping_data = np.array(mapping_data)

    return mapping_data


# 预处理原始的rating.dat文件
def pre_processing(file_path, save_path):
    # 不需要记录分数，只要是出现在rating.rat文件中的数据都是implicit feedback
    user_ids, item_ids, timestamps = [], [], []
    # 将数据存入dataframe中，根据user_id和timestamp进行排序
    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, item_id, rating, timestamp = line.split("::")
            user_ids.append(int(user_id))
            item_ids.append(int(item_id))
            timestamps.append(int(timestamp))

    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'timestamp': timestamps
    })

    sorted_df = df.sort_values(by=['user_id', 'timestamp'], ascending=True)
    # print(sorted_df)
    mapping_data = mapping(sorted_df)

    # save
    np.savetxt(save_path, mapping_data, fmt='%d', delimiter=' ')


file_path = '../data/ml-1m/ratings.dat'
save_path = './data/ml-1m.txt'
pre_processing(file_path, save_path)
