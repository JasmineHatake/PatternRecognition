import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def load_dataset(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, 0)
            image = np.power(image / float(np.max(image)), 1.5)
            image = cv2.resize(image, (128, 128))
            fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
            images.append(fd)
            labels.append(folder)
    return np.array(images), np.array(labels)

data_dir = "C:/Users/32118/Desktop/ai_python_learning/Dataset/Train"
data_dir_ = "C:/Users/32118/Desktop/ai_python_learning/Dataset/Test"

X_train, y_train = load_dataset(data_dir)
X_test, y_test = load_dataset(data_dir_)

# 将标签字符串映射成不同数字
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 定义要尝试的参数网格
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
# 使用交叉验证函数来评估每个参数组合的性能
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train_encoded)
# 输出最佳参数组合
print("Best parameters found: ", grid_search.best_params_)

# 在测试集上进行预测
y_pred_encoded = grid_search.predict(X_test)

print("Classification report for clf %s:\n%s\n"
      %(grid_search,metrics.classification_report(y_test_encoded,y_pred_encoded)))

disp=ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test_encoded)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()
