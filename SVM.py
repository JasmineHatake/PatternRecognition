import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载和预处理数据集
def load_dataset(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (128, 128))
            images.append(resized_image.flatten())  # 将图像展平为一维向量
            labels.append(label)
    return np.array(images), np.array(labels)

# 加载训练集和测试集
train_images, train_labels = load_dataset("C:/Users/32118/Desktop/ai_python_learning/Dataset/Train")
test_images, test_labels = load_dataset("C:/Users/32118/Desktop/ai_python_learning/Dataset/Test")

# 将图像数据归一化
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 使用支持向量机进行分类
svm_model = SVC(kernel='linear')  # 使用线性核的支持向量机
svm_model.fit(train_images, train_labels)

# 在测试集上进行预测
predictions = svm_model.predict(test_images)

# 计算准确率
accuracy = accuracy_score(test_labels, predictions)
print("Test accuracy:", accuracy)
