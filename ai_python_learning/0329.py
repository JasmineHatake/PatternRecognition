import time   
import numpy as np  
# matplotlib.pyplot绘图接口
import matplotlib.pyplot as plt   
from sklearn.cluster import MiniBatchKMeans, KMeans  
# sklearn.metric.pairwise.pairwise_distances_argmin用于计算点与点之间的最近距离，并返回最近距离对应的索引值
from sklearn.metrics.pairwise import pairwise_distances_argmin  
# sklearn.datasets.make_blobs用于生成聚类问题的模拟数据集
from sklearn.datasets import make_blobs  
  
# #############################################################################  

np.random.seed(0)  

#每个小批量的样本数量

batch_size = 45  
centers = [[1, 1], [-1, -1], [1, -1]]  
n_clusters = len(centers)  
X, labels_true = make_blobs(n_samples=3000, centers=n_clusters, cluster_std=0.7)  
  
# #############################################################################  
# Compute clustering with Means  
  
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)  
t0 = time.time()  
k_means.fit(X)  
t_batch = time.time() - t0  
  
# #############################################################################  
# Compute clustering with MiniBatchKMeans  
  
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,  
                      n_init=10, max_no_improvement=10, verbose=0)  
t0 = time.time()  
mbk.fit(X)  
t_mini_batch = time.time() - t0  
  
# #############################################################################  
# Plot result  

#创建了一个大小为8*3英寸的绘图
fig = plt.figure(figsize=(8, 3)) 
# 调整子图位置 
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)  
colors = ['#4EACC5', '#FF9C34', '#4E9A06']  
  
# We want to have the same colors for the same cluster from the  
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per  
# closest one.  

#
k_means_cluster_centers = k_means.cluster_centers_  
order = pairwise_distances_argmin(k_means.cluster_centers_,  
                                  mbk.cluster_centers_)  
# order变量包含了用来重新排列mbk.cluster_centers_的索引顺序，通过使用mbk.cluster_centers_[order]确保两个聚类中心坐标对应
mbk_means_cluster_centers = mbk.cluster_centers_[order]  
# 得到包含每个样本聚类标签的数组  
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)  
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)  
  
# KMeans  
#创建一个子图，添加到fig的1行3列的第1列位置
ax = fig.add_subplot(1, 3, 1)  

# range用于生成整数序列0~n_cluster-1
# zip函数用于将多个可迭代对象中对应位置的元素打包成一个元组，可以让我们可视化时为每个聚类指定不同的颜色以区分
for k, col in zip(range(n_clusters), colors):  
    #创建了一个布尔数组，k_means_labels是包含每个样本所属聚类的标签数组，k是当前遍历地聚类索引，将与当前聚类索引k相等的元素设置为true
    my_members = k_means_labels == k
    #k_means_cluster_centers是包含了所有聚类中心坐标的数组
    cluster_center = k_means_cluster_centers[k]  
    #用于绘制属于当前聚类的样本点
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',  
            markerfacecolor=col, marker='.')
    #这行代码用于绘制当前聚类中心
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  
            markeredgecolor='k', markersize=6)  

ax.set_title('KMeans')  
ax.set_xticks(())  
ax.set_yticks(())  
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (  
    t_batch, k_means.inertia_))  
  


# MiniBatchKMeans  
ax = fig.add_subplot(1, 3, 2)  
for k, col in zip(range(n_clusters), colors):  
    my_members = mbk_means_labels == k  
    cluster_center = mbk_means_cluster_centers[k]  
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',  
            markerfacecolor=col, marker='.')  
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  
            markeredgecolor='k', markersize=6)  
ax.set_title('MiniBatchKMeans')  
ax.set_xticks(())  
ax.set_yticks(())  
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %  
         (t_mini_batch, mbk.inertia_))  
  


# Initialise the different array to all False  
different = (mbk_means_labels == 4)  
ax = fig.add_subplot(1, 3, 3)  
  
for k in range(n_clusters):  
    different += ((k_means_labels == k) != (mbk_means_labels == k))  
  
identic = np.logical_not(different)  
ax.plot(X[identic, 0], X[identic, 1], 'w',  
        markerfacecolor='#bbbbbb', marker='.')  
ax.plot(X[different, 0], X[different, 1], 'w',  
        markerfacecolor='m', marker='.')  
ax.set_title('Difference')  
ax.set_xticks(())  
ax.set_yticks(())  
  
plt.show()  