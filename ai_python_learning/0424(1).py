from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets,metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

digits=datasets.load_digits()
_,axes=plt.subplots(2,4)
images_and_labels=list(zip(digits.images,digits.target))
for ax,(image,label) in zip(axes[0,:],images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    ax.set_title('Training:%i'%label)
n_samples=len(digits.images)
#numpy.reshape(a,newshape,order='C'),a要重排的数组，newshape新的形状，order可选
#该函数将存储在digits.images中的图像数据重新排列成一个二维数组，图像失去空间结构信息
data=digits.images.reshape((n_samples,-1))
clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(500,20),random_state=1)
X_train,X_test,y_train,y_test=train_test_split(
    data,digits.target,test_size=0.5,shuffle=False
)
clf.fit(X_train,y_train)
predicted=clf.predict(X_test)
images_and_predictions=list(zip(digits.images[n_samples//2:],predicted))
for ax,(image,prediction) in zip(axes[1,:],images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Prediction:%i'%prediction)
print("Classification report for clf %s:\n%s\n"
      %(clf,metrics.classification_report(y_test,predicted)))
disp=ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()