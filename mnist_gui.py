import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. 定义网络（和你训练时用的一模一样）----------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------- 2. 加载模型（如果没有就训练一个）----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

try:
    # 尝试加载之前训练好的模型
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    print("✅ 加载已有模型成功！")
except:
    print("🔄 没找到已保存的模型，开始快速训练一个...")
    # 加载数据
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    # 训练一个简单的模型（只跑2轮，够用了）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(2):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} done')
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("✅ 模型训练完成并已保存！")

model.eval()

# ---------- 3. 创建GUI画板 ----------
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🖊️ 手写数字识别 - 鼠标直接画")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # 创建画布（280x280，方便缩放成28x28）
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, 
                                bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # 保存上一个点的坐标
        self.last_x = None
        self.last_y = None
        
        # 创建按钮和标签
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        self.recognize_btn = tk.Button(button_frame, text="🔍 识别", command=self.recognize,
                                       bg='#4CAF50', fg='white', font=('Arial', 12), padx=20)
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(button_frame, text="🗑️ 清空", command=self.clear_canvas,
                                   bg='#f44336', fg='white', font=('Arial', 12), padx=20)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = tk.Label(root, text="在这里写个数字吧 👆", 
                                      font=('Arial', 14), fg='#333333')
        self.result_label.pack(pady=10)
        
        # 用于存储画的图像
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
    
    def paint(self, event):
        """鼠标移动时画线"""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在画布上画线
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                   width=15, fill='black', capstyle=tk.ROUND, smooth=True)
            # 在PIL图像上画线
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=15)
        self.last_x = x
        self.last_y = y
    
    def reset(self, event):
        """松开鼠标时重置坐标"""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """清空画布"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="在这里写个数字吧 👆")
    
    def recognize(self):
        """识别画的数字"""
        # 1. 图像预处理：缩放到28x28
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 2. 转换成numpy数组并反转颜色（MNIST是黑底白字，我们是白底黑字）
        img_array = np.array(img)
        img_array = 255 - img_array  # 反转：黑变白，白变黑
        
        # 3. 归一化
        img_array = img_array / 255.0
        
        # 4. 转换成tensor并增加维度
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # 5. 预测
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1).squeeze()
            confidence = probs[pred].item() * 100
        
        # 6. 显示结果
        self.result_label.config(text=f"✨ 我猜是：{pred}  (置信度：{confidence:.1f}%)")
        
        # 7. 顺便画个柱状图看看（可选）
        self.show_confidence_chart(probs.cpu().numpy(), pred)
    
    def show_confidence_chart(self, probs, pred):
        """显示每个数字的置信度柱状图"""
        plt.figure(figsize=(8, 4))
        plt.bar(range(10), probs, color=['red' if i == pred else 'steelblue' for i in range(10)])
        plt.xlabel('数字')
        plt.ylabel('置信度')
        plt.title('识别结果置信度分布')
        plt.xticks(range(10))
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # 在柱子上标数值
        for i, p in enumerate(probs):
            plt.text(i, p + 0.02, f'{p:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# ---------- 4. 启动应用 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()