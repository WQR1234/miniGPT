### TODO: prepare SFT data similar to `prepare.py`
import os
import sys
import tiktoken
import numpy as np

import json
import time

enc = tiktoken.get_encoding("gpt2")

# sft_path = sys.argv[1]

block_size = 1024
pad_token = enc.encode_single_token('<|endoftext|>')  # 填充时，填充终止符所对应的token


# 读取sft_data.jsonl文件的每条数据
data = []
with open('../sft_data.jsonl', 'r', encoding='utf-8') as f:  # 读取前一目录下的sft_data.jsonl文件
    for line in f:
        json_data = json.loads(line.strip())
        q = json_data['question']
        a = json_data['answer']


        data.append([q, a])

train_data, val_data = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]  # 9：1分为训练集与测试集
train_data = sum(train_data, [])
val_data = sum(val_data, [])


def tokenize_qa_list(qa_list: list[str]):
    # 生成的ndarray列数为block_size+3, 前n+1列为数据（X取0~block_size, Y取1~block_size+1）,
    # 最后两列分别记录question与answer的tokens的长度，用于计算loss mask。
    tokens_list = enc.encode_ordinary_batch(qa_list)
    n = len(qa_list)
    ids = []
    for i in range(0, n, 2):
        row = tokens_list[i]+tokens_list[i+1]

        # 将每行tokens填充或截断至block_size+1
        if len(row)>=block_size+1:
            row = row[:block_size+1]
        else:
            row = row + [pad_token]*(block_size+1-len(row))
        row.append(len(tokens_list[i]))
        row.append(len(tokens_list[i+1]))
        ids.append(row)

    return ids


# 转为np.ndarray
train_ids = np.asarray(tokenize_qa_list(train_data), dtype=np.uint16)
val_ids = np.asarray(tokenize_qa_list(val_data), dtype=np.uint16)

# save numpy array to file [name]/train.bin and [name]/val.bin

dataset = 'sft'
if not os.path.exists(dataset):
    os.mkdir(dataset)
train_ids.tofile(os.path.join(dataset, "train.bin"))
val_ids.tofile(os.path.join(dataset, 'val.bin'))


# # 验证读取到的值与原值相同
# time.sleep(1)
train_ids_read = np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r')
val_ids_read = np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r')

assert np.array_equal(train_ids, train_ids_read.reshape(-1, block_size+3))
assert np.array_equal(val_ids, val_ids_read.reshape(-1, block_size+3))
print('相同')


###