### TODO: add your import
import matplotlib.pyplot as plt  
import os  

def visualize_loss(train_loss_list: list, train_interval: int, val_loss_list: list, val_interval: int, dataset_name: str, out_dir: str): 
    ### TODO: visualize loss of training & validation and save to [out_dir]/loss.png 
    # 计算训练和验证的迭代次数  
    train_iterations = [i * train_interval for i in range(len(train_loss_list))]  
    val_iterations = [i * val_interval for i in range(len(val_loss_list))]  
    
    # 创建输出目录（如果不存在）  
    os.makedirs(out_dir, exist_ok=True)  
    
    # 创建图形  
    plt.figure(figsize=(10, 5))  
    
    # 绘制训练损失  
    plt.plot(train_iterations, train_loss_list, label='Training Loss', color='blue', marker='o')  
    
    # 绘制验证损失  
    plt.plot(val_iterations, val_loss_list, label='Validation Loss', color='red', marker='x')  
    
    # 添加图例  
    plt.legend()  
    # 添加标题和标签  
    plt.title(f'Loss Curves for {dataset_name}')  
    plt.xlabel('Iterations')  
    plt.ylabel('Loss')  
    
    # 保存图像  
    plt.grid()  
    plt.tight_layout()  
    plt.savefig(os.path.join(out_dir, 'loss.png'))  
    # 关闭图形以节省资源  
    plt.close()  

    ###
