# scipy/sklearn实现层次聚类
> 主题：机器学习——聚类——层次聚类
> 参考内容URL：[CSDN笔记](https://blog.csdn.net/pentiumCM/article/details/105695414)
> 
## 1 scipy生成聚类树
### 1.1 函数说明
#### 1.1.1 linkage函数
 ` def linkage(y,method='single',metric = 'euclidean',optimal_ordering=False):`
函数说明：
1. 参数（Parameters）：
    1. y:数据，可以是一维距离矩阵（距离向量）或2维观测矢量数据（坐标点的矩阵）；   
    2. method：计算距离的方法
		    - single：最近邻；
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423105226144.png#pic_center)
		    - complete：最远邻；	   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423105317702.png#pic_center)
		    - average：平均距离；
		    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423104101957.png#pic_center)
		    - weighted：在压缩距离矩阵上执行加权/WPGMA（[加权平均法](https://www.mun.ca/biology/scarr/2900_UPGMA.htm)；[对比](https://www.mun.ca/biology/scarr/UPGMA_vs_WPGMA.html)；[UPGMA非加权平均法](http://www.icp.ucl.ac.be/~opperd/private/upgma.html)）链接。如簇u由簇s,t组成，那么簇u到簇v的距离为![插入图片描述](https://img-blog.csdnimg.cn/20200423103159459.png#pic_center)
		    - centroid：质心距离，把类与类中的质心间距离作为类间距![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423105704682.png#pic_center)
		    - ward：将要合并的群集的方差最小化。
2. 返回值（Returns）：
    1. Z（ndarray）：层次聚类编码为一个linkage矩阵。Z共有四列组成，第 1 字段与第 2 字段分别为聚类簇的编号，在初始距离前每个初始值被从0 ~ n-1进行标识，每生成一个新的聚类簇就在此基础上增加一对新的聚类簇进行标识（比如初始有30个聚类簇，第1个和第17个合并之后形成一个31号聚类簇），第 3 个字段表示前两个聚类簇之间的距离，第 4 个字段表示新生成聚类簇所包含的元素的个数。
3. [linkage scipy官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)

#### 1.1.2 fcluster函数
` def fcluster(Z, t, criterion='inconsistent',depth = 2,R-None,monocrit=None):`
函数说明：从给定linkage矩阵的层次聚类中形成平面聚类。
1. 参数（Parameters):
    1. Z：linkage矩阵，记录了层次聚类的层次信息；
    2. t：是一个聚类的距离的阈值，相当于聚类层级树状图的纵轴值；
    3. criterionstr：形成扁平簇的准则：
        - inconsistent：？
        - distance：以簇之间距离作为划分准则。
2. 返回值（Returns）：
    1. fcluster（ndarray）：长度为n的数组。f[i]是原始观测值i所属的平面簇数。
3. [fcluster scipy官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster)

## 1.2 算法实现
使用[西瓜数据集4.0](http://whatbeg.com/2016/04/22/xiguadataset.html)
样本数据：
![](Hierachical-Clustering_md_files/d9b73300-0420-11ed-9161-c38e1eb50327.jpeg?v=1&type=image)
linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)，层次聚类编码为一个linkage矩阵。
```
Z = linkage(data_new, 'average')
print('聚类过程\n', Z)
```
聚类过程
 [[0.00000000e+00 2.30000000e+01 3.17647603e-02 2.00000000e+00]
 [2.20000000e+01 2.90000000e+01 3.88329757e-02 2.00000000e+00]
 [2.50000000e+01 2.70000000e+01 4.02616443e-02 2.00000000e+00]
 ……
 [4.60000000e+01 5.00000000e+01 1.81114720e-01 8.00000000e+00]
 [5.10000000e+01 5.50000000e+01 2.62026573e-01 1.50000000e+01]
 [4.80000000e+01 5.40000000e+01 2.79452411e-01 1.50000000e+01]
 [5.60000000e+01 5.70000000e+01 3.29199576e-01 3.00000000e+01]]

 根据层级聚类结果linkage矩阵将结果以树状图表示出来
```
plt.figure()
dn = dendrogram(Z)#dengrogram树状图
plt.show()
```
![](Hierachical-Clustering_md_files/7adcb570-0421-11ed-9161-c38e1eb50327.jpeg?v=1&type=image)

从给定linkage链接矩阵定义的层次聚类中形成平面聚类，这里0.2表示距离阈值，在0.2以下的距离下分类（层级图纵轴以0.2划线以下的分类）。
```
f = fcluster(Z, 0.2, 'distance')
print('平面聚类结果:\n', f)
```

` 平面聚类结果:    
[3 2 2 4 4 3 3 4 2 4 3 1 4 2 4 2 1 1 4 4 4 2 1 3 4 2 1 2 1 1]`

实现的代码戳👉[代码](http://localhost:8888/notebooks/a%E5%AD%A6%E4%B8%9A%E4%BF%A1%E8%AE%A1/a%E7%A0%94%E7%A9%B6%E7%94%9F/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB/Hierarchical-clustering.ipynb)

## 2 sklearn实现聚类算法
### 2.1 函数说明
sklearn库下的层次聚类是在sklearn.cluster的 AgglomerativeClustering中：
```
def __init__(self, n_clusters = 2, affinity = 'euclidean', memory = None, 
					connectivity = None, compute_full_tree = 'auto', linkage = 'ward',
					distance_threshold = None):
```
AgglomerativeClustering 类的构造函数的参数有 簇的个数 `n_clusters`，连接方法 `linkage`，连接度量选项 `affinity` 三个重要参数：
- n_clusters：用户指定分类簇数；
- linkage：簇间距离计算策略；同上，有ward，complete，average，single。
     -   ward：将要合并的群集的方差最小化。
    -   complete：完全距离/最大距离，使用两组中所有观察值之间的最大距离。
    -   single：最小距离，使用两组中所有观测值之间的最小距离。
    -   average：平均距离，使用两组的每个观测值的距离平均值。
- affinity：距离计算方法，可以是Euclidean（欧氏距离），l1，l2，Manhattan（曼哈顿距离），cosine（余弦），或precomputed（预计算）。
	- 如果链接为ward，则仅接受欧氏距离；
	- precomputed需要距离矩阵作为拟合方法的输入
	> ` 距离矩阵`的生成方法：假设用户有 n 个观测点，那么先依次构造这 n 个点两两间的距离列表，即长度为 n*(n-1)/2 的距离列表，然后通过 scipy.spatial.distance 的 dist 库的 squareform 函数就可以构造距离矩阵了。这种方式的好处是用户可以使用自己定义的方法计算任意两个观测点的距离，然后再进行聚类。后面也会给出基于自定义的距离矩阵进行的聚类算法。
**scikit-learn 1.1.1**  
[Other versions](http://scikit-learn.org/dev/versions.html)

Please  [cite us](https://scikit-learn.org/stable/about.html#citing-scikit-learn)  if you use the software.

-   [`sklearn.cluster`.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#)
    -   [Examples using  `sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#examples-using-sklearn-cluster-agglomerativeclustering)

`sklearn.cluster.AgglomerativeClustering ` [官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
### 2.2 算法实现
核心代码
```
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
clustering = ac.fit(data)

print("每个数据所属的簇编号", clustering.labels_)
print("每个簇的成员", clustering.children_)
```
算法运行结果
```
每个数据所属的簇编号 [2 2 0 0 1]
每个簇的成员 [[0 1]
 			[2 3]
 			[5 6]
 			[4 7]]
```
👉[完整代码](http://localhost:8888/notebooks/a%E5%AD%A6%E4%B8%9A%E4%BF%A1%E8%AE%A1/a%E7%A0%94%E7%A9%B6%E7%94%9F/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB/Hierarchical-clustering.ipynb)

