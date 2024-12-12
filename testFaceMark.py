import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree


# 定义类别映射
class_names = ['face', 'not-face']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

# 图像尺寸
image_size = (200, 200)


# XML 解析函数
def parse_xml(xml_file):
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    # 获取类别和边界框
    label = root.find('object/name').text
    bbox = root.find('object/bndbox')

    # 边界框坐标
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)

    return label, [xmin, ymin, xmax, ymax]


# 可视化函数
def visualize_annotations(images_dir, annotations_dir, num_samples=20):
    """
    在有 XML 标注的范围内随机选择 num_samples 张图片并可视化标注
    """
    xml_files = sorted(os.listdir(annotations_dir))
    img_files = [xml_file.replace('.xml', '.png') for xml_file in xml_files]

    # 仅保留有标注对应的图像
    valid_samples = [(img_file, xml_file) for img_file, xml_file in zip(img_files, xml_files)
                     if os.path.exists(os.path.join(images_dir, img_file))]

    # 随机选择 num_samples 个文件
    samples = random.sample(valid_samples, min(num_samples, len(valid_samples)))

    plt.figure(figsize=(15, 15))

    for idx, (img_file, xml_file) in enumerate(samples):
        img_path = os.path.join(images_dir, img_file)
        xml_path = os.path.join(annotations_dir, xml_file)

        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法加载图像 {img_path}，跳过。")
            continue

        # 解析 XML
        try:
            label, bbox = parse_xml(xml_path)
        except Exception as e:
            print(f"警告：无法解析 XML 文件 {xml_path}: {e}")
            continue

        # 绘制边界框
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 转换为 RGB 并显示
        plt.subplot(5, 4, idx + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"{label} ({img_file})")

    plt.tight_layout()
    plt.show()


# 配置路径
binary_dir = './dataV2/rgb'
annotations_dir = os.path.join(binary_dir, 'Annotations')
images_dir = os.path.join(binary_dir, 'portrait')

# 调用可视化函数
visualize_annotations(images_dir, annotations_dir, num_samples=20)
