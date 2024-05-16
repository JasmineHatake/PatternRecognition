import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

digits=datasets.load_digits()
_,axes=plt.subplots(2,4)
images_and_labels=list(zip(digits.images,digits.target))

for ax,(image,label) in zip(axes[0,:],images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Training:%i'%label)
#********对手写数据集进行预处理，将图像数据转化为可以输入机器学习模型的格式********
#获取样本数量
n_samples=len(digits.images)
#对图像进行了reshape操作，将每张图像的二维数组（表示图像的像素值）转化为一维数组，-1让NumPy自动计算另一个维度的大小，以保持数据的总量不变
data=digits.images.reshape((n_samples,-1))
#创建一个支持向量机分类器对象
classifier=svm.SVC(gamma=0.001)
X_train,X_test,y_train,y_test=train_test_split(
    data,digits.target,test_size=0.5,shuffle=False
)
classifier.fit(X_train,y_train)
predicted=classifier.predict(X_test)
#将测试集的图像数据和对应的预测结果进行了组合
images_and_predictions=list(zip(digits.images[n_samples//2:],predicted))
for ax,(image,prediction) in zip(axes[1,:],images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Prediction:%i'%prediction)
#打印出分类器的名称以及生成的分类报告
print("Classification report for classifier %s:\n%s\n"
      %(classifier,metrics.classification_report(y_test,predicted)))
#用于绘制混淆矩阵
disp=ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
#打印出混淆矩阵的内容
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()