### TODO: build gradio app based on sample.py
import os
import torch
import tiktoken
from my_model import GPTConfig, MiniGPT

import gradio as gr
import time

out_dir = 'out-best'
device = 'cpu'

enc = tiktoken.get_encoding("gpt2")

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

config = GPTConfig(**checkpoint['model_args'])
model = MiniGPT(config)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model.to(device)


def stream_generate_text(message, history, max_length, temperature, top_k):
    input_text = ''
    for user_input, bot_response in history:
        input_text += user_input + bot_response
    input_text += message
    input_ids = enc.encode_ordinary(input_text)
    x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

    for idx in model.stream_generate(x, max_length, temperature, top_k):
        yield enc.decode(idx[0].tolist())



demo = gr.ChatInterface(
    stream_generate_text,
    additional_inputs=[
        gr.Slider(100, 250, 150, step=10, label="最大token数"),
        gr.Slider(0, 1, 0.8, step=0.1, label="温度"),
        gr.Slider(value=50, step=10, label='前 K 个最可能的词汇')

    ]
)

if __name__ == "__main__":
    demo.launch(share=False)


###