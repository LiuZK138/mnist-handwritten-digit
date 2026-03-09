# 手写数字识别系统 (MNIST)

基于 PyTorch 实现的卷积神经网络，用于识别手写数字（0-9），准确率达到 98.8%。项目包含完整的训练代码和一个带 GUI 的实时识别演示程序。

##  功能特点

- [x] **CNN 网络**：两个卷积层 + 两个全连接层
- [x] **训练与测试**：在 MNIST 数据集上训练，测试准确率 98.8%
- [x] **GUI 实时识别**：用鼠标直接写数字，点击按钮实时识别
- [x] **置信度可视化**：显示模型对每个数字的把握程度

##  运行方式
1. 运行 `mnist_gui.py`
2. 在弹出的画板上用鼠标写数字
3. 点击“识别”按钮查看结果

### 环境要求
- Python 3.8 或更高版本
- 操作系统：Windows / macOS / Linux 均可

### 安装依赖
在终端（命令行）中运行以下命令安装所需库：
```bash
pip install torch torchvision matplotlib pillow
```

##  运行效果
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/1398f2ce-2237-4bae-a508-d1c087edf4c7" />
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/03b247a3-aa42-46c0-87e2-3baf7fd9270d" />

##  关于作者
LiuZK - 考研复试准备期间完成的深度学习入门项目，体现快速学习和动手实践能力。
