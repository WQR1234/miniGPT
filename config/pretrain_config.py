import time

dataset = ''
out_dir = f'out-' + str(int(time.time()))
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False


gradient_accumulation_steps = 1
batch_size = 4
block_size = 1024 # context of up to 256 previous characters

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = 4000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only