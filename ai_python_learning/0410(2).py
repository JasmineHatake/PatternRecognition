import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn import neighbors
n_neighbors= 4
#X[0]搞笑镜头X[1]拥抱镜头X[2]打斗镜头
X=np.array([(45,2,9),(21,17,5),(54,9,11),(39,0,31),(5,2,57),(3,2,65),(2,3,55),(6,4,21),(7,46,4),(9,39,8),(9,38,2),(8,34,17)])
#喜剧1，动作2，爱情3
y=np.array([1,1,1,1,2,2,2,2,3,3,3,3])
h=3
cmap_light=ListedColormap(['orange','cyan','cornflowerblue'])
cmap_bold=ListedColormap(['darkorange','c','darkblue'])

for weights in ['uniform','distance']:
    clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    #绘制训练数据
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=cmap_bold,edgecolor='k',s=20)

    X_test=np.array([(23,6,21)])
    y_pred=clf.predict(X_test)
    test_color=cmap_bold.colors[y_pred[0]-1]
    print(y_pred)
    
    #绘制测试数据
    ax.scatter(X_test[:,0],X_test[:,1],X_test[:,2],c=test_color,cmap=cmap_bold,edgecolor='k',s=80,marker='s')
    #分类边界
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    z_min,z_max=X[:,2].min()-1,X[:,2].max()+1  
    #生成网格点坐标
    xx,yy,zz=np.meshgrid((np.arange(x_min,x_max,h)),np.arange(y_min,y_max,h),np.arange(z_min,z_max,h))

    Z=clf.predict(np.c_[xx.ravel(),yy.ravel(),zz.ravel()])
    #将分类结果Z调整为与网格点坐标一样的形状，以便与网格点坐标对应
    Z=Z.reshape(xx.shape)
    #绘制网格点
    
    
    ax.set_title("3-Class classification(k=%i,weights='%s')"%(n_neighbors,weights))
    plt.show()