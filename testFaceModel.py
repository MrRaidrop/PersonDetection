import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf


# 定义绘制边界框的函数
def draw_bbox(image, bbox, confidence=None):
    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = bbox
    xmin, xmax = int(xmin * w), int(xmax * w)
    ymin, ymax = int(ymin * h), int(ymax * h)
    color = (0, 255, 0)  # 绿色框为人脸

    # 绘制边界框
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    # 添加分类标签和置信度
    text = "Face"
    if confidence is not None:
        text += f" ({confidence:.2f})"
    cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


# 加载训练好的模型
face_model = tf.keras.models.load_model('face_detection_model.h5')


# 测试函数
def test_face_detection_on_dataset(test_csv, annotations_dir, base_dir, model, size=(200, 200), num_samples=20):
    # 加载数据并筛选有人像的图片
    data = pd.read_csv(test_csv)
    face_data = data[data['y:label'] == 1]  # 筛选有人的图片
    indices = np.random.choice(len(face_data), num_samples, replace=False)

    plt.figure(figsize=(15, 15))
    for idx, i in enumerate(indices):
        row = face_data.iloc[i]
        img_path = os.path.join(base_dir, row['x:image'].strip())
        xml_path = os.path.join(annotations_dir, os.path.splitext(os.path.basename(img_path))[0] + '.xml')

        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法加载图像: {img_path}")
            continue
        img_resized = cv2.resize(image, size).astype('float32') / 255.0

        # 加载标注边界框
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bbox = root.find('object/bndbox')
                xmin = int(bbox.find('xmin').text) / image.shape[1]
                ymin = int(bbox.find('ymin').text) / image.shape[0]
                xmax = int(bbox.find('xmax').text) / image.shape[1]
                ymax = int(bbox.find('ymax').text) / image.shape[0]
                true_bbox = [xmin, ymin, xmax, ymax]
            except Exception as e:
                print(f"Error parsing XML file {xml_path}: {e}")
                continue
        else:
            print(f"Annotation file not found: {xml_path}")
            continue

        # 模型预测
        input_image = np.expand_dims(img_resized, axis=0)
        pred_bbox = model.predict(input_image, verbose=0)[0]

        # 绘制预测边界框
        image_with_bbox = draw_bbox(image.copy(), pred_bbox)

        # 显示图像
        plt.subplot(5, 4, idx + 1)
        plt.imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image {idx + 1}")

    plt.tight_layout()
    plt.show()


# 调用测试函数
test_face_detection_on_dataset(
    test_csv='./dataV2/binary/test_binary.csv',
    annotations_dir='./dataV2/binary/Annotations',
    base_dir='./dataV2/rgb',
    model=face_model,
    num_samples=20
)
