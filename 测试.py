import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data_dir="C:/Users/32118/Desktop/ai_python_learning/Dataset/Train"
data_dir_="C:/Users/32118/Desktop/ai_python_learning/Dataset/Test"
images=[]
labels=[]
images_=[]
labels_=[]
_,axes=plt.subplots(1,4)
# 加载和预处理数据集
def load_dataset(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image=np.float32(image)/255.0#归一化
            images.append(image)
            labels.append(folder)
    return np.array(images), np.array(labels)

#获取训练集图像和测试集图像
X_train,y_train =load_dataset(data_dir)
n_samples=len(X_train)
print(n_samples)
X_train_scaled=X_train.reshape((n_samples,-1))
print(X_train_scaled.size)