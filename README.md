# MiniGPT

## 安装第三方库
```
pip3 install torch torchvision torchaudio tiktoken gradio
```
**注意若有GPU则应安装GPU版pytorch,见[pytorch官网](https://pytorch.org/get-started/locally/)**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**注意numpy大版本应为1**


MiniGPT目前已提供or已给出框架的部分内容如下列举。

## 数据预处理

首先进入数据目录:
```bash
cd data/
```

- 准备数据：
    我们在清华云盘准备了预训练数据，请根据作业要求预先下载。并将其放在data文件夹下。
    
- 数据预处理（需实现）：

    ```
    python prepare.py [dataset_names] # tokenize
    ```
    通过`[dataset_names]`指定若干个数据集，将他们统一处理为一份数据（包含训练集`train.bin`与验证集`val.bin`）。

## 模型训练

通过运行如下命令启动训练：
```bash
python train.py config/train_config.py --dataset=[dataset_name]
```
其中`--dataset`参数指定使用数据在`data/`下的二级目录名。

在训练过程中，会自动通过`torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))`保存训练过程中生成的模型



微调（在原有模型基础上继续训练），运行如下指令：
```bash
python train.py config/train_config.py --dataset=[dataset_name] --init_from=finetune --ckpt_dir=[/path/to/ckpt/dir]
```
其中`--dataset`参数指定使用数据在`data/`下的二级目录名, `--ckpt_dir`参数指定加载的训练模型目录位置

## 模型推理

通过运行如下命令加载训练完毕的模型权重进行推理：

```bash
python sample.py --out_dir=[/dir/to/training/output] --save_path=/path/to/save/output # or add prompts by --start=FILE:/path/to/prompts.txt
```

其中：
- `--out_dir`参数指定使用的模型权重的目录（由模型训练过程生成）。
- `--save_path`参数指定生成文本的保存路径，不设置则不保存仅打印。
- `--start`参数可以设置指导模型生成的prompt。可以在`prompts.txt`文件中逐行给出输入的各个prompt

## 可视化界面
```bash
python sample_gradio.py
```
然后打开http://127.0.0.1:7860


# 附：
## GPT2
### [英文视频](https://www.youtube.com/watch?v=l8pRSuU81PU)
### [中文视频](https://www.bilibili.com/video/BV12s421u7sZ)
### [代码](https://github.com/karpathy/nanoGPT)  

## [模型文件下载](https://drive.google.com/file/d/1G9tAASBqgs1OC1P6C64sBnOhTLoEVSoC/view?usp=drive_link)
