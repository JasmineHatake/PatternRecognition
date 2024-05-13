import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image=np.float32(image)/255.0#归一化
            images.append(image)
            labels.append(folder)
    return np.array(images), np.array(labels)
#获取训练集图像和测试集图像
X_train,y_train =load_dataset(data_dir)
X_test,y_test =load_dataset(data_dir_)

# 获取训练集和测试集的数字标签
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)



n_samples=len(X_train)
n_samples_=len(X_test)

X_train_scaled=X_train.reshape((n_samples,-1))
X_test_scaled=X_test.reshape((n_samples_,-1))
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
clf = MLPClassifier(max_iter=500, solver='adam', random_state=1,alpha=0.0001,hidden_layer_sizes=(100,))
clf.fit(X_train_scaled, y_train_encoded)
y_pred_encoded = clf.predict(X_test_scaled)

print("Classification report for clf %s:\n%s\n"
      %(clf,metrics.classification_report(y_test_encoded,y_pred_encoded)))
disp=ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test_encoded)
disp.figure_.suptitle("Confusion Matrix")

print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()