'''
https://blog.csdn.net/qq_40714949/article/details/109863595
LeNet结构:
C1卷积层:卷积层的意义在于利用不同的卷积核扫描图像，可以提取图像不同的特征
    -参数
        -卷积核种类6
        -卷积核大小5*5
        -输出特征图数量6
        -输出特征图大小 123*123
        -神经元数量 123*123*6
        -可训练参数 (5*5+1)*6
        -连接数 123*123*(5*5+1)*6,卷积层每个特征图各像素都与其对应的卷积核的各个参数间有连接，但由于有权值共享机制
S2池化层:池化是进行下采样操作，即对图像进行压缩可以降低卷积层输出的特征向量，改善过拟合现象
        -输入:123*123的特征图6张
        -采样区域3*3
        -输出特征图大小:41*41
        -输出特征图数量 6
        -神经元数量:41*41*6
        -可训练参数3*
C3卷积层
S4池化层
C5卷积层
F6全连接层
OUTPUT输出层
'''
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
            images.append(resized_image)
            labels.append(label)
    return np.array(images), np.array(labels)
# 加载训练集和测试集
train_images, train_labels = load_dataset("C:/Users/32118/Desktop/ai_python_learning/Dataset/Train")
test_images, test_labels = load_dataset("C:/Users/32118/Desktop/ai_python_learning/Dataset/Test")

# 将图像数据归一化
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
# 将训练集划分为训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
#定义LeNet模型
#
def build_lenet(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
# 构建LeNet模型
model = build_lenet(input_shape=train_images[0].shape, num_classes=len(np.unique(train_labels)))
# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))
# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
