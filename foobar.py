import tiktoken

import os
import numpy as np
import json
import torch

enc = tiktoken.get_encoding("gpt2")


# st = enc.encode_ordinary('\n')
# print(st)

# with open('sft_data.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         json_data = json.loads(line.strip())
#         q = json_data['question']
#         a = json_data['answer']
#         qt = enc.encode_ordinary(q)
#         at = enc.encode_ordinary(a)
#         qat = enc.encode_ordinary(q+a)

#         print(qt)
#         print(at)
#         print(qat)
#         print(qt+at==qat)

#         # print(qat[:qat.index(st[0])]==qt)
#         # print(qat[qat.index(st[0])+1:]==at)

#         break

# s1 = "hwllo world!     "
# s2 = "tiktoken is great!  "
# t1 = enc.encode_ordinary(s1)
# t2 = enc.encode_ordinary(s2)
# t3 = enc.encode_ordinary(s1+s2)
# print(t1)
# print(t2)
# print(t3)
# print(t1+t2==t3)

# print(enc.decode([198, 198]))

# print(enc.encode_single_token(' '))

# sft_data = []
# with open('qa.txt', 'r', encoding='utf-8') as f:
#     data = {}
#     for line in f:
#         if line[:3]=='问题：':
#             data['question'] = line[3:].strip()
#         elif line[:3]=='答案：':
#             data['answer'] = line[3:].strip()
        
#             sft_data.append(data.copy())
#             data.clear()

# sft_data = [json.dumps(x, ensure_ascii=False)+'\n' for x in sft_data]
# # print(sft_data)
# with open('sft_data.jsonl', 'a+', encoding='utf-8') as f:
#     f.writelines(sft_data)


# input_text = "北京是中国的"
# print(enc.encode_ordinary(input_text))

checkpoint = torch.load('ckpt.bin', map_location='cpu')

print(checkpoint)

# state_dict = torch.load('pytorch_model.bin', map_location='cpu')
# unwanted_postfix = 'masked_bias'
# state_dict = {key: value for key, value in state_dict.items() if not key.endswith(unwanted_postfix)} 
# model_args = {
#     "n_layer": 12,
#     "n_head": 12,
#     "n_embd": 768,
#     'block_size': 1024,
#     'dropout': 0.1,
#     'bias': True,
#     'vocab_size': 50304,
# }

# checkpoint = {
#     'model': state_dict,
#     'model_args': model_args,
#     # 'iter_num': iter_num,
#     # 'best_val_loss': best_val_loss,
#     # 'config': config,
# }

# out_dir = "out-best"
# if not os.path.exists(out_dir):
#     os.mkdir(out_dir)
# torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))


from data_utils import *

# dataset = 'sft'
# init_data_sft(dataset)

# x, y, loss_mask = get_batch_sft('train', 1, 256, 'cpu')
# print(x)
# print(y)

# print(loss_mask)

# tokens = y[0].tolist()
# output = enc.decode(tokens)
# output = output.replace('<|endoftext|>', '')
# print(output)


# y = y[loss_mask==1]
# output = enc.decode(y.tolist())
# output = output.replace('<|endoftext|>', '')
# print(output)

