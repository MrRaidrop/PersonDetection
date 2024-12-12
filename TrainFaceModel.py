import os
import random
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# **数据生成器**
def face_detection_data_generator(csv_file, annotations_dir, binary_base_dir, rgb_base_dir, size=(200, 200), batch_size=32):
    """
    仅加载包含人脸数据的生成器，用于训练和测试。
    """
    def parse_voc_xml(xml_path, img_width, img_height):
        """解析 XML 文件并返回归一化 bbox。"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find('object')
            if obj is None:
                return None  # 无标注数据
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) / img_width
            ymin = int(bbox.find('ymin').text) / img_height
            xmax = int(bbox.find('xmax').text) / img_width
            ymax = int(bbox.find('ymax').text) / img_height
            return [xmin, ymin, xmax, ymax]
        except Exception as e:
            print(f"Error parsing XML file {xml_path}: {e}")
            return None

    data = pd.read_csv(csv_file)
    while True:
        images, bboxes = [], []
        for _, row in data.sample(frac=1).iterrows():
            img_path_binary = os.path.join(binary_base_dir, row['x:image'].strip())
            img_path_rgb = os.path.join(rgb_base_dir, row['x:image'].strip())
            xml_path = os.path.join(annotations_dir, os.path.splitext(os.path.basename(img_path_binary))[0] + '.xml')

            # 加载 RGB 图像
            img_rgb = cv2.imread(img_path_rgb)
            if img_rgb is not None:
                h, w = img_rgb.shape[:2]
                bbox = parse_voc_xml(xml_path, w, h)
                if bbox:
                    img_rgb = cv2.resize(img_rgb, size).astype('float32') / 255.0
                    images.append(img_rgb)
                    bboxes.append(bbox)

            # 加载 binary 图像（转为伪彩色）
            img_binary = cv2.imread(img_path_binary, cv2.IMREAD_GRAYSCALE)
            if img_binary is not None:
                img_binary = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2RGB)
                h, w = img_binary.shape[:2]
                bbox = parse_voc_xml(xml_path, w, h)
                if bbox:
                    img_binary = cv2.resize(img_binary, size).astype('float32') / 255.0
                    images.append(img_binary)
                    bboxes.append(bbox)

            # 输出 batch
            if len(images) >= batch_size:
                yield np.array(images), np.array(bboxes)
                images, bboxes = [], []

# **构建单分支模型**
def create_face_detection_model(input_shape):
    """
    构建仅输出边界框的模型。
    """
    inputs = layers.Input(shape=input_shape)

    # 卷积网络
    x = layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

    # 边界框输出
    regression_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)

    # 定义模型
    model = models.Model(inputs=inputs, outputs=regression_output)
    return model

# **路径配置**
train_csv = './dataV2/binary/train_binary.csv'
test_csv = './dataV2/binary/test_binary.csv'
annotations_dir = './dataV2/binary/Annotations'
binary_base_dir = './dataV2/binary'
rgb_base_dir = './dataV2/rgb'

train_samples = pd.read_csv(train_csv).shape[0]
test_samples = pd.read_csv(test_csv).shape[0]

# **数据生成器**
train_generator = face_detection_data_generator(
    train_csv, annotations_dir, binary_base_dir, rgb_base_dir, batch_size=32
)
test_generator = face_detection_data_generator(
    test_csv, annotations_dir, binary_base_dir, rgb_base_dir, batch_size=32
)

# **创建模型**
model = create_face_detection_model(input_shape=(200, 200, 3))

# **编译模型**
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mse']
)

# **学习率调整和早停**
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# **训练模型**
history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // 32,
    validation_data=test_generator,
    validation_steps=test_samples // 32,
    epochs=5         ,
    callbacks=[lr_scheduler, early_stopping]
)

# 保存模型
model.save('face_detection_model.h5')
print("Model saved as face_detection_model.h5")

# **可视化训练曲线**
def plot_training_curves(history):
    """
    绘制训练和验证的损失曲线。
    """
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_curves(history)
