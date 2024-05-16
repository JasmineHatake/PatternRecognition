import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data_dir="C:/Users/32118/Desktop/ai_python_learning/Dataset/Train"
data_dir_="C:/Users/32118/Desktop/ai_python_learning/Dataset/Test"
images=[]
labels=[]
images_=[]
labels_=[]
for label,folder in enumerate(os.listdir(data_dir)):
    folder_path=os.path.join(data_dir,folder)
    for image_file in os.listdir(folder_path):
        image_path=os.path.join(folder_path,image_file)
        image=cv2.imread(image_path)
        #cv2.resize()函数缩放图像
        resized_image=cv2.resize(image,(128,128))
        images.append(resized_image)
        labels.append(folder)

for label_,folder_ in enumerate(os.listdir(data_dir_)):
    folder_path_=os.path.join(data_dir_,folder_)
    for image_file_ in os.listdir(folder_path_):
        image_path_=os.path.join(folder_path_,image_file_)

        image_=cv2.imread(image_path_)
        #cv2.resize()函数缩放图像
        resized_image_=cv2.resize(image_,(128,128))
        images_.append(resized_image_)
        labels_.append(folder_)

# 将图像数据转换为 NumPy 数组
X_train = np.array(images)
X_test = np.array(images_)
y_train = np.array(labels)
y_test = np.array(labels_)


# 将标签字符串映射成不同数字
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(len(X_train), -1))  # 将每个图像的特征向量展平为一维数组并归一化
X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))
'''
# 定义超参数的搜索范围
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'alpha': [0.0001, 0.001, 0.01],
}
#创建MLP分类器
clf = MLPClassifier(max_iter=500, solver='adam', random_state=1)
# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=10)
grid_search.fit(X_train_scaled, y_train_encoded)

# 打印最佳参数组合和对应的得分
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_clf = grid_search.best_estimator_'''
clf = MLPClassifier(max_iter=500, solver='lbfgs', random_state=1,alpha=0.0001,hidden_layer_sizes=(100,))
clf.fit(X_train_scaled, y_train_encoded)
y_pred_encoded = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print("Accuracy:", accuracy)