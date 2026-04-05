import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
import glob
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ============ 1. 数据预处理模块 ============
def load_sensor_data_from_csv(csv_path, time_steps=100, sensor_cols=None, label_col='label'):
    """
    从单个CSV文件加载传感器数据
    """
    df = pd.read_csv(csv_path)

    if sensor_cols is None:
        # 自动检测传感器列：假设除了标签列外的所有列都是传感器
        sensor_cols = [col for col in df.columns if col != label_col]

    # 提取传感器数据和标签
    sensor_data = df[sensor_cols].values.astype(np.float32)

    if label_col in df.columns:
        labels = df[label_col].values
    else:
        # 如果没有标签列，从文件名推断
        filename = os.path.basename(csv_path)
        if "straight" in filename.lower() or "端正" in filename:
            labels = np.zeros(len(sensor_data))
        elif "forward" in filename.lower() or "前倾" in filename:
            labels = np.ones(len(sensor_data))
        elif "side" in filename.lower() or "侧弯" in filename:
            labels = np.ones(len(sensor_data)) * 2
        else:
            labels = np.zeros(len(sensor_data))

    return sensor_data, labels, sensor_cols

def create_time_series_samples(data, labels, time_steps=100, overlap=0.5):
    """
    将连续数据分割成固定长度的时间序列样本
    """
    samples = []
    sample_labels = []

    step = int(time_steps * (1 - overlap))  # 步长
    n_samples = (len(data) - time_steps) // step + 1

    for i in range(n_samples):
        start_idx = i * step
        end_idx = start_idx + time_steps

        sample = data[start_idx:end_idx]

        if len(sample) < time_steps:
            # 填充
            padding = np.zeros((time_steps - len(sample), sample.shape[1]))
            sample = np.vstack([sample, padding])

        # 使用窗口内最频繁的标签
        window_labels = labels[start_idx:end_idx]
        unique, counts = np.unique(window_labels, return_counts=True)
        label = unique[np.argmax(counts)] if len(unique) > 0 else 0

        samples.append(sample)
        sample_labels.append(label)

    return np.array(samples), np.array(sample_labels)

def prepare_dataset(csv_folder, time_steps=100, n_sensors=7, output_dir='processed_data'):
    """
    准备完整的数据集
    """
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    all_labels = []

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(csv_folder, "**/*.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    if not csv_files:
        raise ValueError(f"在 {csv_folder} 中没有找到CSV文件")

    print(f"找到 {len(csv_files)} 个CSV文件")

    sensor_columns = None

    for i, csv_file in enumerate(csv_files):
        print(f"处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")

        data, labels, sensor_cols = load_sensor_data_from_csv(csv_file, time_steps)

        if sensor_columns is None:
            sensor_columns = sensor_cols

        samples, sample_labels = create_time_series_samples(data, labels, time_steps)

        all_samples.append(samples)
        all_labels.append(sample_labels)

    # 合并所有样本
    X = np.vstack(all_samples)
    y = np.hstack(all_labels)

    # 归一化
    data_mean = np.mean(X, axis=(0, 1), keepdims=True)
    data_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X - data_mean) / data_std

    # 分割数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 保存数据
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # 保存归一化参数
    norm_params = {
        'mean': data_mean.tolist(),
        'std': data_std.tolist(),
        'sensor_columns': sensor_columns
    }

    with open(os.path.join(output_dir, 'normalization_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)

    # 统计数据
    stats = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'time_steps': time_steps,
        'n_sensors': X.shape[2],
        'n_classes': len(np.unique(y)),
        'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    }

    with open(os.path.join(output_dir, 'data_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n数据集准备完成:")
    print(f"  总样本数: {len(X)}")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  类别数: {len(np.unique(y))}")
    print(f"  类别分布: {stats['class_distribution']}")

    return X_train, X_val, X_test, y_train, y_val, y_test, stats

# ============ 2. 模型构建模块 ============
def build_teacher_model(input_shape, num_classes):
    """构建教师模型（复杂但准确）"""
    inputs = keras.Input(shape=input_shape)

    # 卷积块1
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # 卷积块2
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # 卷积块3
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    # 全连接层
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='teacher_model')
    return model

def build_student_model(input_shape, num_classes):
    """构建学生模型（轻量适合部署）"""
    inputs = keras.Input(shape=input_shape)

    # 更轻量的卷积块
    x = layers.Conv1D(16, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # 更少的全连接层
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='student_model')
    return model

# ============ 3. 知识蒸馏核心模块 ============
class KnowledgeDistillation(keras.Model):
    """知识蒸馏模型"""

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs, training=False):
        student_logits = self.student_model(inputs, training=training)
        teacher_logits = self.teacher_model(inputs, training=False)  # 教师模型不训练

        # 应用温度缩放
        if training:
            teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
            student_probs = tf.nn.softmax(student_logits / self.temperature)
            return student_probs, teacher_probs, student_logits
        else:
            return student_logits

    def compute_loss(self, student_probs, teacher_probs, student_logits, y_true):
        # 蒸馏损失（学生模仿教师）
        distillation_loss = keras.losses.KLDivergence()(
            teacher_probs, student_probs
        ) * (self.temperature ** 2)

        # 学生自身的分类损失
        student_loss = self.loss_fn(y_true, student_logits)

        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss, distillation_loss, student_loss

# ============ 4. 训练模块 ============
def train_teacher_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=100):
    """训练教师模型"""
    print("\n" + "="*50)
    print("训练教师模型")
    print("="*50)

    teacher = build_teacher_model(input_shape, num_classes)

    teacher.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    teacher.summary()

    # 回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/teacher_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # 训练
    history = teacher.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return teacher, history

def train_student_with_distillation(teacher, X_train, y_train, X_val, y_val, input_shape, num_classes,
                                    temperature=3.0, alpha=0.5, epochs=100):
    """使用知识蒸馏训练学生模型"""
    print("\n" + "="*50)
    print("知识蒸馏训练学生模型")
    print("="*50)
    print(f"温度参数: {temperature}")
    print(f"蒸馏权重(α): {alpha}")

    # 构建学生模型
    student = build_student_model(input_shape, num_classes)

    # 构建蒸馏模型
    distillation_model = KnowledgeDistillation(
        teacher_model=teacher,
        student_model=student,
        temperature=temperature,
        alpha=alpha
    )

    # 编译
    distillation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    student.summary()

    # 自定义训练循环
    print("\n开始蒸馏训练...")

    # 准备数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # 训练循环
    train_losses = []
    val_losses = []
    distill_losses = []
    student_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 训练阶段
        epoch_loss = 0
        epoch_distill = 0
        epoch_student = 0
        steps = 0

        for batch_x, batch_y in train_dataset:
            with tf.GradientTape() as tape:
                # 前向传播
                student_probs, teacher_probs, student_logits = distillation_model(batch_x, training=True)

                # 计算损失
                total_loss, distill_loss, stu_loss = distillation_model.compute_loss(
                    student_probs, teacher_probs, student_logits, batch_y
                )

            # 反向传播
            gradients = tape.gradient(total_loss, distillation_model.student_model.trainable_variables)
            distillation_model.optimizer.apply_gradients(
                zip(gradients, distillation_model.student_model.trainable_variables)
            )

            epoch_loss += total_loss
            epoch_distill += distill_loss
            epoch_student += stu_loss
            steps += 1

        # 验证阶段
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0

        for batch_x, batch_y in val_dataset:
            predictions = distillation_model(batch_x, training=False)
            val_loss += distillation_model.loss_fn(batch_y, predictions)

            # 计算准确率
            pred_labels = tf.argmax(predictions, axis=1)
            correct += tf.reduce_sum(tf.cast(pred_labels == batch_y, tf.float32))
            total += len(batch_y)
            val_steps += 1

        # 记录损失
        train_losses.append(epoch_loss / steps)
        distill_losses.append(epoch_distill / steps)
        student_losses.append(epoch_student / steps)
        val_losses.append(val_loss / val_steps)

        val_acc = (correct / total).numpy()

        print(f"训练损失: {train_losses[-1]:.4f} "
              f"(蒸馏: {distill_losses[-1]:.4f}, 学生: {student_losses[-1]:.4f})")
        print(f"验证损失: {val_losses[-1]:.4f}, 验证准确率: {val_acc:.4f}")

        # 早停检查
        if len(val_losses) > 10 and np.mean(val_losses[-5:]) > np.mean(val_losses[-10:-5]):
            print("验证损失不再下降，提前停止")
            break

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='总损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(distill_losses, label='蒸馏损失')
    plt.plot(student_losses, label='学生损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('损失分量曲线')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/distillation_training.png', dpi=300, bbox_inches='tight')
    plt.show()

    return student, train_losses, val_losses

# ============ 5. 模型评估模块 ============
def evaluate_model(model, X_test, y_test, name="模型"):
    """评估模型性能"""
    print(f"\n评估 {name}:")

    # 计算准确率
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)

    print(f"测试准确率: {accuracy:.4f}")

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes, digits=4)

    print("\n分类报告:")
    print(report)

    # 可视化混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴
    class_names = ['端正', '前倾', '侧弯', '其他'][:len(np.unique(y_test))]
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=f'{name} 混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')

    # 添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f'models/confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, cm

# ============ 6. 模型保存和转换模块 ============
def save_models(teacher_model, student_model, stats):
    """保存所有模型"""
    os.makedirs('models', exist_ok=True)

    # 1. 保存Keras模型
    print("\n保存Keras模型...")
    teacher_model.save('models/teacher_model.keras')
    student_model.save('models/student_model.keras')

    # 2. 转换为TFLite格式
    print("转换为TFLite格式...")

    # 学生模型转换为TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]

    tflite_model = converter.convert()

    with open('models/student_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # 3. 转换为C数组格式（用于嵌入式设备）
    print("转换为C数组格式...")

    c_array = []
    c_array.append(f"// 知识蒸馏训练的学生模型\n")
    c_array.append(f"// 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    c_array.append(f"// 模型信息: {stats}\n\n")
    c_array.append("#ifndef MODEL_DATA_H\n")
    c_array.append("#define MODEL_DATA_H\n\n")
    c_array.append(f"const unsigned char g_model[] = {{\n  ")

    for i, byte in enumerate(tflite_model):
        c_array.append(f"0x{byte:02x},")
        if (i + 1) % 16 == 0:
            c_array.append("\n  ")

    c_array.append("\n};\n")
    c_array.append(f"const unsigned int g_model_len = {len(tflite_model)};\n\n")
    c_array.append("#endif // MODEL_DATA_H\n")

    with open('models/model_data.h', 'w') as f:
        f.write("".join(c_array))

    # 4. 保存模型信息
    model_info = {
        'teacher_params': teacher_model.count_params(),
        'student_params': student_model.count_params(),
        'compression_ratio': teacher_model.count_params() / student_model.count_params(),
        'teacher_input_shape': teacher_model.input_shape[1:],
        'student_input_shape': student_model.input_shape[1:],
        'num_classes': student_model.output_shape[-1],
        'saved_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n模型保存完成:")
    print(f"  Teacher model: models/teacher_model.keras ({teacher_model.count_params():,} 参数)")
    print(f"  Student model: models/student_model.keras ({student_model.count_params():,} 参数)")
    print(f"  TFLite model: models/student_model.tflite ({len(tflite_model)/1024:.2f} KB)")
    print(f"  C header: models/model_data.h")
    print(f"  压缩比: {model_info['compression_ratio']:.1f}x")

    return tflite_model

# ============ 7. 测试推理模块 ============
def test_tflite_inference():
    """测试TFLite模型推理"""
    print("\n测试TFLite推理...")

    # 加载测试数据
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')

    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path='models/student_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"输入形状: {input_details[0]['shape']}")
    print(f"输出形状: {output_details[0]['shape']}")

    # 测试几个样本
    correct = 0
    n_test = min(20, len(X_test))

    for i in range(n_test):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        pred = np.argmax(output[0])
        true = y_test[i]

        if pred == true:
            correct += 1

        if i < 3:  # 显示前3个样本的详细结果
            print(f"样本 {i+1}: 预测={pred}, 真实={true}, 正确={pred==true}")
            print(f"  概率分布: {output[0]}")

    print(f"\nTFLite模型推理准确率: {correct}/{n_test} ({correct/n_test:.2%})")

# ============ 8. 主训练流程 ============
def main():
    """主训练流程"""
    print("="*60)
    print("1D-CNN 知识蒸馏训练系统")
    print("="*60)

    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)

    # 步骤1: 准备数据
    print("\n[1/6] 准备数据集...")

    # 如果有CSV文件，使用真实数据
    csv_folder = "your_csv_data"  # 修改为您的CSV文件夹路径

    if os.path.exists(csv_folder) and len(glob.glob(os.path.join(csv_folder, "*.csv"))) > 0:
        print("使用CSV文件数据...")
        X_train, X_val, X_test, y_train, y_val, y_test, stats = prepare_dataset(
            csv_folder, time_steps=100, output_dir='processed_data'
        )
    else:
        print("使用模拟数据（因为没有找到CSV文件）...")

        # 生成模拟数据
        n_samples = 2000
        time_steps = 100
        n_sensors = 7
        n_classes = 4

        # 生成模拟的传感器数据
        X = np.random.randn(n_samples, time_steps, n_sensors).astype(np.float32)

        # 生成标签（模拟不同体态的模式）
        y = np.random.randint(0, n_classes, n_samples)

        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        stats = {
            'total_samples': n_samples,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'time_steps': time_steps,
            'n_sensors': n_sensors,
            'n_classes': n_classes
        }

        # 保存数据
        np.save('processed_data/X_train.npy', X_train)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/X_val.npy', X_val)
        np.save('processed_data/y_val.npy', y_val)
        np.save('processed_data/X_test.npy', X_test)
        np.save('processed_data/y_test.npy', y_test)

    # 步骤2: 训练教师模型
    print("\n[2/6] 训练教师模型...")
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    teacher_model, teacher_history = train_teacher_model(
        X_train, y_train, X_val, y_val,
        input_shape, num_classes, epochs=50
    )

    # 步骤3: 评估教师模型
    print("\n[3/6] 评估教师模型...")
    teacher_accuracy, _ = evaluate_model(teacher_model, X_test, y_test, "教师模型")

    # 步骤4: 知识蒸馏训练学生模型
    print("\n[4/6] 知识蒸馏训练学生模型...")
    student_model, train_losses, val_losses = train_student_with_distillation(
        teacher_model, X_train, y_train, X_val, y_val,
        input_shape, num_classes,
        temperature=3.0, alpha=0.7, epochs=50
    )

    # 步骤5: 评估学生模型
    print("\n[5/6] 评估学生模型...")
    student_accuracy, _ = evaluate_model(student_model, X_test, y_test, "学生模型")

    # 步骤6: 保存模型
    print("\n[6/6] 保存模型文件...")
    tflite_model = save_models(teacher_model, student_model, stats)

    # 最终报告
    print("\n" + "="*60)
    print("知识蒸馏训练完成！")
    print("="*60)
    print(f"教师模型准确率: {teacher_accuracy:.4f}")
    print(f"学生模型准确率: {student_accuracy:.4f}")
    print(f"准确率保持: {student_accuracy/teacher_accuracy:.2%}")
    print(f"参数压缩比: {teacher_model.count_params()/student_model.count_params():.1f}x")
    print(f"模型大小: {len(tflite_model)/1024:.2f} KB")
    print("\n模型文件已保存到 'models/' 目录:")
    print("  - teacher_model.keras: 教师模型")
    print("  - student_model.keras: 学生模型")
    print("  - student_model.tflite: 嵌入式TFLite格式")
    print("  - model_data.h: C头文件（用于嵌入式设备）")
    print("  - model_info.json: 模型信息")
    print("\n下一步: 将 'student_model.tflite' 或 'model_data.h' 部署到嵌入式设备")
    print("="*60)

    # 测试推理
    test_tflite_inference()

    return teacher_model, student_model
    # 10 运行程序

    if __name__ == "__main__":
    get_csv_preparation_instructions()

    response = input("\n是否开始训练？(y/n): ")

    if response.lower() in ['y', 'yes']:
        # 修改这里的路径为您的CSV文件夹路径
        csv_folder = "your_csv_data_folder"  # 修改这里！

        if os.path.exists(csv_folder):
            print(f"\n使用CSV文件夹: {csv_folder}")
        else:
            print(f"\nCSV文件夹 {csv_folder} 不存在，将使用模拟数据")

        # 运行主训练流程
        teacher, student = main()

        print("\n训练完成！您现在可以:")
        print("1. 将 'models/student_model.tflite' 部署到ESP32/STM32")
        print("2. 在嵌入式代码中包含 'models/model_data.h'")
        print("3. 使用相同的归一化参数预处理传感器数据")
        print("4. 运行实时体态识别")
    else:
        print("已取消训练。请准备好CSV数据后重新运行。")