### TODO: Implement metrics Perplexity, Rouge-L, etc.

import os
import sys
import torch
import tiktoken

from my_model import GPTConfig, MiniGPT
from data_utils import init_data_pretrain, init_data_sft, get_batch_pretrain, get_batch_sft


out_dir = 'out-best'
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dataset = ''
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024


exec(open('configurator.py').read())

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
# print(checkpoint['model_args'])
# print(checkpoint['model'].keys())
# sys.exit()
config = GPTConfig(**checkpoint['model_args'])
# print(config)
# print(checkpoint['model'].keys())
model = MiniGPT(config)
print(model)


# sys.exit()

state_dict = checkpoint['model']
model.load_state_dict(state_dict)
# load_weight(model, state_dict)

# sys.exit()

model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')

if dataset == 'pretrain':
    init_data_pretrain(dataset)
elif dataset == 'sft':
    init_data_sft(dataset)
else:
    sys.exit()


@torch.no_grad()
def calculate_perplexity():
    if dataset == 'pretrain':
        X, Y, loss_mask = get_batch_pretrain('val', batch_size, block_size, device)
    else:
        X, Y, loss_mask = get_batch_sft('val', batch_size, block_size, device)
    _, loss = model(X, Y, loss_mask)
    return torch.exp2(loss)


def longest_common_subsequence(seq1, seq2):
    # 计算最长公共子序列
    m = len(seq1)
    n = len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    
    lcs_length = dp[-1][-1]
    lcs = []

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            if dp[i][j] == dp[i - 1][j]:
                break
            elif dp[i][j] == dp[i][j - 1]:
                continue
            else:
                lcs.append(seq1[i - 1])
                i -= 1
                j -= 1
                break
    
    lcs.reverse()
    return lcs


def calculate_Rouge_L(X, Y, beta):
    lcs = longest_common_subsequence(X, Y)
    R_lcs = len(lcs) / len(X)
    P_lcs = len(lcs) / len(Y)

    rouge_l = (1+beta*beta)*R_lcs*P_lcs / (R_lcs+beta*beta*P_lcs)
    return rouge_l

###