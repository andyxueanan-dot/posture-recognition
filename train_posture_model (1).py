import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 消除OpenMP警告

# ================================================================
# 智能正姿衣物 - 体态识别模型训练（PyTorch 知识蒸馏版）
# 输入文件：sensor_posture_data_X.npy / sensor_posture_data_y.npy
# ================================================================

# ============ 只需要改这里！============
X_PATH = r"D:\PythonProject1\sensor_posture_data_X.npy"   # 特征文件路径
Y_PATH = r"D:\PythonProject1\sensor_posture_data_y.npy"   # 标签文件路径
# =======================================

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'

# ============ 检查环境 ============
print("=" * 55)
print("  体态识别模型训练系统（PyTorch 知识蒸馏）")
print("=" * 55)
print(f"\n  PyTorch 版本: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  运行设备: {device}")

# ============ 1. 加载数据 ============
print("\n[1/6] 加载数据...")
X = np.load(X_PATH).astype(np.float32)
y_raw = np.load(Y_PATH)
y = (y_raw - 1).astype(np.int64)  # 标签从1~6转成0~5

label_names = ['走路', '上楼', '下楼', '坐姿', '站姿', '躺卧']
num_classes = len(np.unique(y))

print(f"  特征矩阵: {X.shape}")
print(f"  标签数组: {y.shape}，类别数: {num_classes}")
for i in range(num_classes):
    print(f"    {label_names[i]}: {np.sum(y == i)} 条")

# ============ 2. 数据预处理 ============
print("\n[2/6] 数据预处理...")

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

os.makedirs('models', exist_ok=True)
np.save('models/X_mean.npy', X_mean)
np.save('models/X_std.npy',  X_std)

n   = len(X_norm)
idx = np.random.RandomState(42).permutation(n)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

X_train, y_train = X_norm[idx[:n_train]],            y[idx[:n_train]]
X_val,   y_val   = X_norm[idx[n_train:n_train+n_val]], y[idx[n_train:n_train+n_val]]
X_test,  y_test  = X_norm[idx[n_train+n_val:]],      y[idx[n_train+n_val:]]

def make_loader(X, y, shuffle=False):
    X_t = torch.tensor(X).unsqueeze(1)  # (N, 1, 561)
    y_t = torch.tensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

print(f"  训练:{X_train.shape}  验证:{X_val.shape}  测试:{X_test.shape}")

# ============ 3. 定义模型 ============

class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).squeeze(-1))


class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).squeeze(-1))


# ============ 4. 训练教师模型 ============
print("\n[3/6] 训练教师模型...")

ce_loss = nn.CrossEntropyLoss()

def run_epoch(model, loader, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss   = ce_loss(logits, y_b)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(y_b)
    return total_loss / len(loader), correct / total

teacher     = TeacherModel(num_classes).to(device)
t_optimizer = optim.Adam(teacher.parameters(), lr=1e-3)
t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, patience=5, factor=0.5)

best_val, patience_count = float('inf'), 0
t_train_accs, t_val_accs = [], []

for epoch in range(50):
    tr_loss, tr_acc = run_epoch(teacher, train_loader, t_optimizer, train=True)
    vl_loss, vl_acc = run_epoch(teacher, val_loader, train=False)
    t_scheduler.step(vl_loss)
    t_train_accs.append(tr_acc)
    t_val_accs.append(vl_acc)
    print(f"  Epoch {epoch+1:3d}/50  训练:{tr_acc:.4f}  验证:{vl_acc:.4f}  损失:{vl_loss:.4f}")
    if vl_loss < best_val:
        best_val, patience_count = vl_loss, 0
        torch.save(teacher.state_dict(), 'models/teacher_best.pth')
    else:
        patience_count += 1
        if patience_count >= 12:
            print("  早停触发"); break

teacher.load_state_dict(torch.load('models/teacher_best.pth', weights_only=True))

# ============ 5. 知识蒸馏训练学生模型 ============
print("\n[4/6] 知识蒸馏训练学生模型...")

TEMPERATURE, ALPHA = 3.0, 0.7
student     = StudentModel(num_classes).to(device)
s_optimizer = optim.Adam(student.parameters(), lr=1e-3)
s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, patience=5, factor=0.5)
kl_loss     = nn.KLDivLoss(reduction='batchmean')

best_val, patience_count = float('inf'), 0
s_train_losses, s_val_losses = [], []
teacher.eval()

for epoch in range(50):
    student.train()
    epoch_loss, steps = 0.0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        with torch.no_grad():
            t_probs = torch.softmax(teacher(X_b) / TEMPERATURE, dim=1)
        s_logits = student(X_b)
        s_probs  = torch.log_softmax(s_logits / TEMPERATURE, dim=1)
        loss = ALPHA * kl_loss(s_probs, t_probs) * (TEMPERATURE**2) + \
               (1 - ALPHA) * ce_loss(s_logits, y_b)
        s_optimizer.zero_grad()
        loss.backward()
        s_optimizer.step()
        epoch_loss += loss.item(); steps += 1

    vl_loss, vl_acc = run_epoch(student, val_loader, train=False)
    s_scheduler.step(vl_loss)
    s_train_losses.append(epoch_loss / steps)
    s_val_losses.append(vl_loss)
    print(f"  Epoch {epoch+1:3d}/50  蒸馏损失:{epoch_loss/steps:.4f}  验证:{vl_acc:.4f}")
    if vl_loss < best_val:
        best_val, patience_count = vl_loss, 0
        torch.save(student.state_dict(), 'models/student_best.pth')
    else:
        patience_count += 1
        if patience_count >= 12:
            print("  早停触发"); break

student.load_state_dict(torch.load('models/student_best.pth', weights_only=True))

# ============ 6. 评估模型 ============
print("\n[5/6] 评估模型...")

def evaluate(model, loader, name):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.extend(model(X_b.to(device)).argmax(1).cpu().numpy())
            labels.extend(y_b.numpy())
    preds, labels = np.array(preds), np.array(labels)
    acc = np.mean(preds == labels)
    print(f"\n  {name} 测试准确率: {acc*100:.2f}%")
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() == 0: continue
        print(f"    {label_names[i]}: {np.mean(preds[mask]==labels[mask])*100:.2f}%")
    return acc

teacher_acc = evaluate(teacher, test_loader, "教师模型")
student_acc = evaluate(student, test_loader, "学生模型")

# ============ 7. 保存文件 ============
print("\n[6/6] 保存文件...")

torch.save(teacher, 'models/teacher_model.pt')
torch.save(student, 'models/student_model.pt')

t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())

with open('models/model_info.json', 'w', encoding='utf-8') as f:
    json.dump({
        'teacher_params':    t_params,
        'student_params':    s_params,
        'compression_ratio': round(t_params / s_params, 1),
        'teacher_accuracy':  round(float(teacher_acc), 4),
        'student_accuracy':  round(float(student_acc), 4),
        'num_classes':       num_classes,
        'label_names':       label_names,
    }, f, indent=2, ensure_ascii=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_train_accs, label='训练准确率')
axes[0].plot(t_val_accs,   label='验证准确率')
axes[0].set_title('教师模型训练曲线'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('准确率'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(s_train_losses, label='蒸馏训练损失')
axes[1].plot(s_val_losses,   label='蒸馏验证损失')
axes[1].set_title('学生模型蒸馏曲线'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('损失'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150)

print("\n" + "=" * 55)
print("  训练完成！")
print("=" * 55)
print(f"  教师模型准确率 : {teacher_acc*100:.2f}%")
print(f"  学生模型准确率 : {student_acc*100:.2f}%")
print(f"  准确率保持     : {student_acc/teacher_acc*100:.1f}%")
print(f"  参数压缩比     : {t_params/s_params:.1f}x  ({t_params:,} → {s_params:,})")
print("\n  输出文件（在 models/ 文件夹）：")
print("    teacher_model.pt          教师模型")
print("    student_model.pt          学生模型")
print("    model_info.json           模型信息与准确率")
print("    X_mean.npy / X_std.npy    归一化参数（部署必用）")
print("    training_curves.png       训练曲线图")
print("=" * 55)
