import os

import torch
import numpy as np

train_data = None
val_data = None

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    
    ### TODO: 读取+初始化sft数据
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ###

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
    
def get_batch_sft(split, batch_size, block_size, device): 
    ### TODO: 获取sft数据的批次（batch）+ 构建损失函数掩码（loss_mask）
    global train_data, val_data

    # 读取到的数据已摊平为一行，需先转为（n, block_size+3）
    train_data = np.reshape(train_data, (-1, block_size+3))
    val_data = np.reshape(val_data, (-1, block_size+3))
    data = train_data if split == 'train' else val_data

    # 随机取batch_size行
    ix = np.random.choice(data.shape[0], batch_size, replace=False)  
    x = torch.from_numpy(data[ix, 0:block_size].astype(np.int64))
    y = torch.from_numpy(data[ix, 1:block_size+1].astype(np.int64))

    # 根据最后两列（记录question与answer的tokens长度）计算loss_mask,即question tokens length ~ question tokens length + answer tokens length 全取1
    loss_mask = torch.zeros_like(x, dtype=torch.float64)
    qa_idx = data[:, -2:]
    for i, row_idx in enumerate(ix):
        loss_mask[i, qa_idx[row_idx, 0]-1:min(block_size, qa_idx[row_idx, 0]+qa_idx[row_idx, 1]-1)] = 1.

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    ###
    
    return x, y, loss_mask