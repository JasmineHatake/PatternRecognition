import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets
n_neighbors= 15 
#导入需要处理的数据集
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
h=.02
cmap_light=ListedColormap(['orange','cyan','cornflowerblue'])
cmap_bold=ListedColormap(['darkorange','c','darkblue'])

for weights in ['uniform','distance']:
    clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    
    #生成网格点坐标
    xx,yy=np.meshgrid((np.arange(x_min,x_max,h)),np.arange(y_min,y_max,h))
    
    #xx.ravel()将网格点坐标展平为一维数组，np.c_()按列连接数组，得到一个二维数组，每一行是一个网格点坐标
    #clf.predict()对于传入的展平后的网格点坐标，用于对每个网格点进行分类预测
    #Z是一个一维数组，表示对每个网格点的预测结果
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    #将分类结果Z调整为与网格点坐标一样的形状，以便与网格点坐标对应
    Z=Z.reshape(xx.shape)

    plt.figure()
    #根据网格中的数值对应到颜色映射cmap_light中的颜色并将颜色填充到网格中
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    #绘制训练数据
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
    #限定x轴显示的范围
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("3-Class classification(k=%i,weights='%s')"%(n_neighbors,weights))
    plt.show()