import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
n_neighbors= 7
X=np.array([(1,0),(0,1),(0,-1),(0,0),(0,2),(0,-2),(-2,0)])
y=np.array([1,1,1,2,2,2,2])
h=.02
cmap_light=ListedColormap(['orange','cyan'])
cmap_bold=ListedColormap(['darkorange','c'])
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
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto', vmin=1, vmax=2)
    #绘制训练数据
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)

    #测试数据
    X_test=np.array([(0.2,0.3)])
    y_pred=clf.predict(X_test)
    test_color=cmap_bold.colors[y_pred[0]-1]
    print(y_pred)
    plt.scatter(X_test[:,0],X_test[:,1],c=test_color,edgecolor='k',s=80,marker='s',label='Test data')
    
    #限定x轴显示的范围
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("3-Class classification(k=%i,weights='%s')"%(n_neighbors,weights))
    plt.show()