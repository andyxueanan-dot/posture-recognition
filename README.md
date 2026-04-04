# 智能正姿衣物 - 体态识别模型

## 项目简介
基于UCI人体活动识别数据集，使用1D-CNN + 知识蒸馏训练的体态识别模型。
输入561个传感器特征，识别6种姿态：走路、上楼、下楼、坐姿、站姿、躺卧。

## 模型效果
| 模型 | 准确率 | 参数量 |
|---|---|---|
| 教师模型 | 98.06% | 491,142 |
| 学生模型 | 78.98% | 10,374 |
| 压缩比 | - | 47.3倍 |

## 文件说明
| 文件 | 用途 |
|---|---|
| `student_model.pt` | 最终轻量模型，用于部署 |
| `teacher_model.pt` | 教师模型，仅供参考 |
| `X_mean.npy` | 归一化均值参数，部署必用 |
| `X_std.npy` | 归一化标准差参数，部署必用 |
| `model_info.json` | 模型详细信息 |
| `training_curves.png` | 训练曲线图 |
| `main.py` | 训练代码 |

## 使用方法
```python
import torch
import numpy as np

# 加载模型
model = torch.load('student_model.pt')
model.eval()

# 加载归一化参数
X_mean = np.load('X_mean.npy')
X_std  = np.load('X_std.npy')

# 输入新传感器数据（561个数字）
new_data = np.array([...])
new_data = (new_data - X_mean) / X_std

# 预测
x = torch.tensor(new_data).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    pred = model(x).argmax(1).item()

label_names = ['走路', '上楼', '下楼', '坐姿', '站姿', '躺卧']
print(f"预测结果：{label_names[pred]}")
```

## 数据来源
UCI Human Activity Recognition Using Smartphones Dataset
