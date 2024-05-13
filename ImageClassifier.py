'''
HOG特征提取流程:
1.图像预处理:
    ·gamma矫正(调整图像的对比度，降低局部阴影和光照变化造成的影响，同时抑制噪音)和和灰度化
2.计算图像像素点梯度值，得到梯度图：梯度大小,梯度幅度,梯度方向
3.图像划分多个cell,统计cell内梯度直方图
4.将2*2个cell联合成一个block,对每个block做块内梯度归一化
在opencv中,Sobel算子是一种常用于图像处理中的边缘检测算子
Sobel离散差分算子:图像可以看作一个离散的网络，每个网格点代表图像的一个像素
差分通常用于描述数据的局部变化，用于检测图像中的边缘或者轮廓。
'''
import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

data_dir="C:/Users/32118/Desktop/ai_python_learning/Dataset/Train"
data_dir_="C:/Users/32118/Desktop/ai_python_learning/Dataset/Test"
images=[]
labels=[]
images_=[]
labels_=[]
# 加载和预处理数据集
def load_dataset(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path,0)
            image=np.power(image/float(np.max(image)),1.5)
            image = cv2.resize(image, (128, 128))
            fd, hog_image = hog(image,
                    orientations=9,#把180°分成几份
                    pixels_per_cell=(8, 8),#一个cell中包含的像素个数
                    cells_per_block=(1, 1),#一个block中包含的cell个数
                    visualize=True)#是否返回一个hog图像用于显示
            images.append(fd)
            labels.append(folder)
    return np.array(images), np.array(labels)


X_train,y_train =load_dataset(data_dir)
X_test,y_test =load_dataset(data_dir_)

# 将标签字符串映射成不同数字
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

clf = MLPClassifier(max_iter=500, solver='adam', random_state=1,alpha=0.0001,hidden_layer_sizes=(100,))
clf.fit(X_train, y_train_encoded)
y_pred_encoded = clf.predict(X_test)

print("Classification report for clf %s:\n%s\n"
      %(clf,metrics.classification_report(y_test_encoded,y_pred_encoded)))
disp=ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test_encoded)
disp.figure_.suptitle("Confusion Matrix")

print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()