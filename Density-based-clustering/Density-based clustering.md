# sklearn实现密度聚类DBSCAN算法
> 主题：机器学习——聚类——密度聚类
> 参考URL：[CSDN笔记](https://blog.csdn.net/FAICULTY/article/details/79430164)
## 2.1 函数说明
在scikit-learn中，DBSCAN算法类为`sklearn.cluster.DBSCAN`.
```
from sklearn.cluster import DBSCAN
y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
```
参数说明：
- eps：距离阈值$\epsilon$；
- min_samples：邻域样本数阈值$MinPts$；
- X：要分类的数据集。
## 2.2 DBSCAN算法优缺点说明
- **优点**
  - 可以对任意形状的稠密数据集进行聚类，相对的，K-Means只适用于凸的数据集。
  - 可以在聚类的同时发现异常点（是的！），对数据集中的异常点不敏感。聚类结果没有偏倚，相对的，K-Means等聚类算法初始值对聚类结果有很大影响。
- **缺点**
  - 若样本密度不均匀、聚类之间差距相差很大时，聚类质量较差，此时不适合用DBSCAN算法，会导致异常点识别较多或同一个簇中样本过少。如果样本数据集较大，聚类收敛时间较长，此时可以对搜索最近邻时建立[KD树](https://blog.csdn.net/zkk12345/article/details/76404219)或者[球树（ball-tree）](https://blog.csdn.net/weixin_41770169/article/details/81634307)进行规模限制来改进。补充知识👉[最近邻的思想，距离度量，KD树和球树](https://www.cnblogs.com/pinard/p/6061661.html)。
  - 调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值$\epsilon$，邻域样本数阈值$MinPts$联合调参，不同的参数组合对最后的聚类效果有较大影响。
 ## 2.3 算法实现
 ![](Density-based%20clustering_md_files/9fa268b0-0510-11ed-a2ea-8b08400b08e0.jpeg?v=1&type=image)
 若要预测新数据，可以参考👉[这个链接](https://www.codenong.com/27822752/)
