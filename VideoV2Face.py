import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 加载模型
person_model = load_model('person_V2.h5', compile=False)  # 用于检测是否有人
face_model = load_model('face_detection_model2.h5', compile=False)  # 用于检测人脸边框

# 重新编译模型（若必要）
person_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
face_model.compile(optimizer='adam', loss={'classification': 'binary_crossentropy', 'regression': 'mse'})

# 摄像头捕获对象
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 输入图像尺寸
INPUT_SIZE_PERSON = (200, 200)  # 输入到 person_V2.h5 模型的尺寸
INPUT_SIZE_FACE = (200, 200)    # 输入到 face_detection_model.h5 模型的尺寸

# 初始化保存检测结果的变量
last_face_bbox = None
last_face_prob = 0.0
last_detection_time = 0
DISPLAY_THRESHOLD = 1.0  # 在最近 1 秒内检测到的框继续显示

def preprocess_frame(frame, size):
    """预处理图像以适应模型输入"""
    img = cv2.resize(frame, size)  # 调整大小
    img = img / 255.0  # 归一化到 [0, 1]
    img = np.expand_dims(img, axis=0)  # 添加批次维度
    return img

def draw_bbox(image, bbox, label, color=(0, 255, 0)):
    """在图像上绘制边界框和标签"""
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理帧
    processed_person_frame = preprocess_frame(frame, INPUT_SIZE_PERSON)

    # 使用 person_V2.h5 检测是否有人
    person_logits = person_model.predict(processed_person_frame)[0]
    person_prob = tf.nn.softmax(person_logits).numpy()
    person_detected = np.argmax(person_prob) == 1  # 是否检测到人

    if person_detected:
        # 如果检测到人，用 face_detection_model.h5 检测人脸边框
        processed_face_frame = preprocess_frame(frame, INPUT_SIZE_FACE)
        preds = face_model.predict(processed_face_frame)

        # 获取边界框（face_model 输出的内容）
        face_bbox = preds[0]  # 假设模型直接输出边界框坐标

        # 绘制绿框
        draw_bbox(frame, face_bbox, label="Face Detected", color=(0, 255, 0))
    else:
        cv2.putText(frame, "No Person Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示视频流
    cv2.imshow('Real-Time Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print(preds)
