import os
import sys
import tiktoken
import numpy as np

import json
import time

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

names = sys.argv[1:]

### TODO: read data from ([name].jsonl for name in names)
### TODO: combine multiple files(if needed) into one single data file
### TODO: split data for train(0.9) and valid (0.1)
data = []
for name in names:
    with open(f'{name}.jsonl', 'r', encoding='utf-8') as f:  # 读取当前目录下的jsonl文件，文件由names指定
        for line in f:
            json_data = json.loads(line.strip())
            data.append(json_data['text'])
train_data, val_data = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]
train_data = ''.join(train_data)
val_data = ''.join(val_data)
# print(len(train_data))
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array

train_ids, val_ids = enc.encode_ordinary(train_data), enc.encode_ordinary(val_data)
# print(len(train_ids))
train_ids = np.asarray(train_ids, dtype=np.uint16)
val_ids = np.asarray(val_ids, dtype=np.uint16)

# print(enc.decode(train_ids[:100]))

# print(train_ids.shape)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
dataset = 'pretrain'
if not os.path.exists(dataset):
    os.mkdir(dataset)
train_ids.tofile(os.path.join(dataset, "train.bin"))
val_ids.tofile(os.path.join(dataset, 'val.bin'))


# # 验证读取到的值与原值相同
# time.sleep(1)
train_ids_read = np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r')
val_ids_read = np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r')
assert np.array_equal(train_ids, train_ids_read)
assert np.array_equal(val_ids, val_ids_read)
print('相同')


### python prepare.py wiki-zh-subset-AA wiki-zh-subset-AB wiki-zh-subset-AC wiki-zh-subset-AD wiki-zh-subset-AE wiki-zh-subset-AF wiki-zh-subset-AG wiki-zh-subset-AH wiki-zh-subset-AI wiki-zh-subset-AJ wiki-zh-subset-AK