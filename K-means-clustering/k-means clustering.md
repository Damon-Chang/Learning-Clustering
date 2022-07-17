# k-means算法实现聚类
> 主题：机器学习——聚类算法——k均值
> 参考URL：[CSDN笔记](https://blog.csdn.net/qq_38563206/article/details/120435976)
## 1 函数说明
` form sklearn.cluster import KMeans `
` class sklearn.cluster.KMeans(n_clusters=8,init='k-means++',n_init=10,max_iter=300,tol=0.0001,verbose=0,random_state=None,copy_x=True,algorithm='lloyd')`
在[scikit-learn](https://so.csdn.net/so/search?q=scikit-learn&spm=1001.2101.3001.7020)中，包括两个K-Means的算法，一个是传统的K-Means算法，对应的类是KMeans。另一个是基于采样的Mini Batch K-Means算法，对应的类是MiniBatchKMeans。一般来说，使用K-Means的算法调参是比较简单的。
用[KMeans](https://so.csdn.net/so/search?q=KMeans&spm=1001.2101.3001.7020)类的话，一般要注意的仅仅就是k值的选择，即参数n_clusters；如果是用MiniBatchKMeans的话，也仅仅多了需要注意调参的参数batch_size，即我们的Mini Batch的大小。
当然KMeans类和MiniBatchKMeans类可以选择的参数还有不少，但是大多不需要怎么去调参。下面我们就看看KMeans类和MiniBatchKMeans类的一些主要参数。

**Kmeans参数说明**
- **n_clusters**: 即我们的k值，一般需要多试一些值以获得较好的聚类效果。k值好坏的评估标准在下面会讲。
- **max_iter**： 最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不是凸的，可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。
-  **n_init**：用不同的初始化质心运行算法的次数。由于K-Means是结果受初始值影响的局部最优的迭代算法，因此需要多跑几次以选择一个较好的聚类效果，默认是10，一般不需要改。如果你的k值较大，则可以适当增大这个值。
- **init：** 即初始值选择的方式，可以为完全随机选择'random',优化过的'k-means++'或者自己指定初始化的k个质心。一般建议使用默认的'k-means++'。
- **algorithm**：有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means算法， “elkan”是我们原理篇讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是 “elkan”，否则就是"full"。一般来说建议直接用默认的"auto"

**MiniBatchKMeans类主要参数**
- n_cluster：k值
- max_iter：最大迭代次数
- n_int：用不同的初始化质心运行算法的次数。这里和KMeans类意义稍有不同，KMeans类里的n_init是用同样的训练集数据来跑不同的初始化质心从而运行算法。而MiniBatchKMeans类的n_init则是每次用不一样的[采样](https://so.csdn.net/so/search?q=%E9%87%87%E6%A0%B7&spm=1001.2101.3001.7020)数据集来跑不同的初始化质心运行算法。
- init_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。
- init：初始值的选择方式
- init_size：用来做质心初始值候选的样本个数，默认是batch_size的3倍，一般用默认值就可以了。
- reassignment_ratio：某个类别质心被重新赋值的最大次数比例，这个和max_iter一样是为了控制算法运行时间的。这个比例是占样本总数的比例，乘以样本总数就得到了每个类别质心可以重新赋值的次数。如果取值较高的话算法收敛时间可能会增加，尤其是那些暂时拥有样本数较少的质心。默认是0.01。如果数据量不是超大的话，比如1w以下，建议使用默认值。如果数据量超过1w，类别又比较多，可能需要适当减少这个比例值。具体要根据训练集来决定。
- max_no_improvement：即连续多少个Mini Batch没有改善聚类效果的话，就停止算法， 和reassignment_ratio，  max_iter一样是为了控制算法运行时间的。默认是10.一般用默认值就足够了。

## 2 K值的评估标准
Calinski-Harabasz Index，这个计算简单直接，得到的Calinski-Harabasz分数值ss越大则聚类效果越好。Calinski-Harabasz分数值s的数学计算公式是：
$$s(k)=tr(Bk)tr(Wk)m−kk−1s(k)=tr(Bk)tr(Wk)m−kk−1$$
其中m为训练集样本数，k为类别数。Bk为类别之间的协方差矩阵，Wk为类别内部数据的协方差矩阵。tr为矩阵的迹。
也就是说，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。在scikit-learn中， Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score.
```
from sklearn import metrics
metrics.calinski_harabaz_score(X, y_pred)
```
## 3 代码实现
**KMeans**
```
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()　　
```
![](k-means%20clustering_md_files/c2a63400-0578-11ed-b8fb-3587475aafb3.jpeg?v=1&type=image)
 **用Calinski-Harabasz Index评估的聚类分数**
 ```
 from sklearn import metrics
 metrics.calinski_harabasz_score(data, y_pred)
 ```
 `28.09297207606`
 **MiniBatchKMeans**
 ```
y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 5, random_state=9).fit_predict(X)
score= metrics.calinski_harabaz_score(X, y_pred)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
```
![](k-means%20clustering_md_files/83ba5850-057a-11ed-b8fb-3587475aafb3.jpeg?v=1&type=image)
  
  **用Calinski-Harabasz Index评估的聚类分数**
 ```
 from sklearn import metrics
 metrics.calinski_harabasz_score(data, y_pred)
 ```
`
Out[25]:
24.35535782529911`
