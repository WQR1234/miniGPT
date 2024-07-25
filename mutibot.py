import os
import torch
import tiktoken
from my_model import GPTConfig, MiniGPT

import gradio as gr
from gradio_client import Client
import time
import random

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


other_url = "http://127.0.0.1:7861"
client = Client(other_url)



with gr.Blocks() as demo:
    chatbot1 = gr.Chatbot()
    chatbot2 = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    max_length = gr.Slider(100, 250, 150, step=10, label="最大token数")
    temperature = gr.Slider(0, 1, 0.8, step=0.1, label="温度")
    top_k = gr.Slider(value=50, step=10, label='前 K 个最可能的词汇')

    def user(user_message, history):
        return "", history + [[user_message, None]], history + [[user_message, None]]


    def stream_generate_text(history, max_new_tokens, temperature, top_k):
        history[-1][1] = ""
        input_text = ''
        for user_input, bot_response in history:
            input_text += user_input + bot_response
        input_ids = enc.encode_ordinary(input_text)
        x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

        for idx in model.stream_generate(x, max_new_tokens, temperature, top_k):
            output_text = enc.decode(idx[0].tolist())
            if output_text != "<|endoftext|>":
                history[-1][1] = output_text
                yield history
    
    def other_bot(history, max_length, temperature, top_k):
        history[-1][1] = ""
        input_text = ''
        for user_input, bot_response in history:
            input_text += user_input + bot_response
        history[-1][1] = client.predict(input_text, max_length, temperature, top_k)
        return history

    msg.submit(user, [msg, chatbot1], [msg, chatbot1, chatbot2], queue=False).then(
        stream_generate_text, [chatbot1, max_length, temperature, top_k], chatbot1
    ).then(
        other_bot, [chatbot2, max_length, temperature, top_k], chatbot2
    )
    clear.click(lambda: None, None, [chatbot1, chatbot2], queue=False)

demo.launch(share=False)
