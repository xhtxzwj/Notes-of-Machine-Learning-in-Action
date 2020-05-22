关于集成学习, sklearn中是ensemble模块, 即sklearn.ensemble. 里面包含了AdaBoost, 随机森林等常用的集成方法, 本文用的是AdaBoostClassifier.

![](https://i.loli.net/2019/09/27/Q89EfhCLZSIq3Xu.png)



# 一 ensemble.AdaBoostClassifier

![](https://i.loli.net/2019/09/26/peDU3uocmRtyZOH.png)



**参数说明**

AdaBoostClassifier中主要的参数如下: 

- **base_estimator:  ** **基分类器, 可选参数,  默认为决策树 (DecisionTreeClassifier)** .  理论上可以选择任何一个分类或者回归学习器,  不过需要支持样本权重.  我们常用的一般是CART决策树或者神经网络MLP.  默认是决策树,  即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier,  而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor.  另外有一个要注意的点是,  如果我们选择的AdaBoostClassifier算法是SAMME.R,  则我们的弱分类学习器还需要支持概率预测,  也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba.  
- **algorithm:  ** **算法,  也就是模型提升准则.  可选参数,  默认为SAMME.R**.  scikit-learn实现了两种Adaboost分类算法,  SAMME和SAMME.R.  两者的主要区别是弱学习器权重的度量,  SAMME使用对样本集分类效果作为弱学习器权重,  而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重.  由于SAMME.R使用了概率度量的连续值,  迭代一般比SAMME快,  因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R.  我们一般使用默认的SAMME.R就够了,  但是要注意的是使用了SAMME.R,   则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器.  SAMME算法则没有这个限制.  
- **n_estimators:  ** **基分类器提升 (循环) 次数, 整数型,  可选参数,  默认为50.**  弱学习器的最大迭代次数,  或者说最大的弱学习器的个数.  一般来说n_estimators太小,  容易欠拟合,  n_estimators太大,  又容易过拟合,  一般选择一个适中的数值.  默认是50.  在实际调参的过程中,  我们常常将n_estimators和下面介绍的参数learning_rate一起考虑.  
- **learning_rate:  ** **学习率,  表示梯度收敛速度,  浮点型,  可选参数,  默认为1.0**.  每个弱学习器的权重缩减系数,  取值范围为0到1,  对于同样的训练集拟合效果,  较小的v意味着我们需要更多的弱学习器的迭代次数.  通常我们用步长和迭代最大次数一起来决定算法的拟合效果.  所以这两个参数n_estimators和learning_rate要一起调参.  一般来说,  可以从一个小一点的v开始调参,  默认是1.  
- **random_state:  ** **随机种子设置,  整数型,  可选参数,  默认为None**.  如果RandomState的实例,  random_state是随机数生成器; 如果None,  则随机数生成器是由np.random使用的RandomState实例.  



**主要方法**:

同时, AdaBoostClassifier中还有一些方法, 如下:

![](https://i.loli.net/2019/09/26/sElaWDGmreXu14j.png)

decision_function(X): 返回决策函数值 (比如svm中的决策距离) 

fit(X,Y): 在数据集 (X,Y) 上训练模型.  

get_parms(): 获取模型参数

predict(X): 预测数据集X的结果.  

predict_log_proba(X): 预测数据集X的对数概率.  

predict_proba(X): 预测数据集X的概率值.  

score(X,Y): 输出数据集 (X,Y) 在模型上的准确率.  

staged_decision_function(X): 返回每个基分类器的决策函数值

staged_predict(X): 返回每个基分类器的预测数据集X的结果.  

staged_predict_proba(X): 返回每个基分类器的预测数据集X的概率结果.  

staged_score(X, Y): 返回每个基分类器的预测准确率.  



# 二 利用Sklearn构建AdaBoost分类器

机器学习实战中的"在一个难数据集上应用 AdaBoost ",  马疝病数据集上应用AdaBoost分类器.

具体代码如下:

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def loadDataSet(fileName):
    #特征数目
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    bdt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),algorithm='SAMME', n_estimators= 10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum()/ len(testArr)*100))

    scores = bdt.score(testArr, testLabelArr)
    print("Score: %.3f%%" % (scores*100))
```

上面的我用了两种方式来看准确率的情况, 第一个是自己构建的错误率情况, 第二个是用里面的score方法,测出准确率情况. 

结果如下:

![](https://i.loli.net/2019/09/26/UAzpcasZTLGIqSt.png)





# 三 性能度量

对学习器泛化性能的评估, 在有了可行的实验估计方法后, 还需要有**衡量模型泛化能力的评价标准**,  这就是**性能度量**.

在对比不同模型的能力时,  使用**不同的性能度量往往会导致不同的评判结果**;  这意味着**模型的"好坏"是相对的**,  什么样的模型是好,  不仅取决于**算法**和**数据**,  还决定于**任务需求**. 



给定样例集 $D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \ldots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}$ ,  其中 $y_{i}$ 是示例 $\boldsymbol x_{i}$ 的真实标记.  要评估学习器 $f$ 的性能,  就要把**学习器预测结果 $f(\boldsymbol x)$** 与**真实标记 $y$** 进行**比较**.

**回归任务**中常用的性能度量是"**均方误差**" (mean squared error)
$$
E(f ; D)=\frac{1}{m} \sum_{i=1}^{m}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}\right)^{2}\tag{2.2}
$$
更一般的,  对于数据分布 $ \mathcal{D}$ 和概率密度函数 $p(\cdot)$ ,  均方误差可描述为
$$
E(f ; \mathcal{D})=\int_{\boldsymbol{x} \sim \mathcal{D}}(f(\boldsymbol{x})-y)^{2} p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}\tag{2.3}
$$

## (一) 错误率与精度

**错误率**是**分类错误的样本数**占样本**总数**的比例,  **精度**则是分类正确的样本数占样本总数的比例.

对于样例集 $D$ ,  分类错误率定义为
$$
E(f ; D)=\frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right) \neq y_{i}\right)\tag{2.4}
$$
精度则为
$$
\begin{aligned} \operatorname{acc}(f ; D) &=\frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right)=y_{i}\right) \\ &=1-E(f ; D) \end{aligned}\tag{2.5}
$$
更为一般的,  对于数据分布 $ \mathcal{D}$ 和概率密度函数 $p(\cdot)$ ,  错误率与精度可以定义为
$$
E(f ; \mathcal{D})=\int_{\boldsymbol{x} \sim \mathcal{D}} \mathbb{I}(f(\boldsymbol{x}) \neq y) p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}\tag{2.6}
$$

$$
\begin{aligned} \operatorname{acc}(f ; \mathcal{D}) &=\int_{\boldsymbol{x} \sim \mathcal{D}} \mathbb{I}(f(\boldsymbol{x})=y) p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x} \\ &=1-E(f ; \mathcal{D}) \end{aligned}\tag{2.7}
$$



## (二) 查准率、查全率与 $F1$ 

错误率衡量了有**多少比例的瓜被判别错误** ,  但"**挑出的西瓜中有多少比例是好瓜**"—**查准率**  或者"**所有好瓜中有多少比例被挑了出来**"—**查全率**,   这两个指标同样非常重要.

### 1 查准率和查全率的定义

对于二分类问题,  可将样例根据其**真实类别**与**学习器预测类别**的组合划分为**真正例** (**true positive**)、**假正例** (**false positive**)、**真反例** (**true negative**) 、**假反例** (**false negative**) 四种情形，令 **$TP$**、 **$FP$**、 **$TN$**、 **$FN$** 分别表示其对应的样例数,  则显然有 $TP+FP+TN+FN=$ 样例总数.  分类结果的<font color=red>"**混淆矩阵**" </font>(confusion matrix) 如表 $2.1$ 所示:

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190624092652.png)

**查准率 $P$** 和**查全率 $R$** 分别定义为
$$
P=\frac{T P}{T P+F P}\tag{2.8}
$$

$$
R=\frac{T P}{T P+F N}\tag{2.9}
$$

**查准率和查全率互为矛盾体:**

查准率和查全率是一对矛盾的度量.  一般来说,  **查准率高时,  查全率往往偏低**;  而**查全率高时,  查准率往往偏低**.

### 2 $P-R$ 曲线

根据**学习器的预测结果对样例进行排序**,  排在**前面**的是学习器认为**"最可能 "是正例的样本**,  排在最后的则是学习器认为**"最不可能"是正例的样本**.  

按**此顺序逐个把样本作为正例进行预测**,  则每次可以计算出**当前的查全率、查准率**.  以查准率为纵轴、查全率为横轴作图,   就得到了**查准率-查全率曲线**,  简称"**$P-R$ 曲线**".  如下图

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190624100606.png)

$P-R$ 图直观地显示出学习器在样本总体上的查全率、 查准率 .

- **1 包住**:  在进行比较时,  若一个学习器的 $P-R$ 曲线被另一个学习器的曲线完全"**包住** " ,  则可断言**后者的性能优于前者**, 如图 $2.3$ 中学习去 A 的性能优于学习器 $C$ ;
- **2 交叉**:  如果两个学习器的 $P-R$ 曲线发生了**交叉**,  例如图 $2.3$ 中的  学习器 $A$ 与 $B$ ,  则难以一般性地断言两者孰优孰劣,  只能在具体的查准率或查全率条件下进行比较. 这时一个比较合理的判据是比较 $P-R$ 曲线节面积的大小.  但这个值不太容易估算. 下面的**平衡点**就时一个综合考虑查准率、查全率的性能度量指标.

### 3 平衡点

"**平衡点**" (Break-Event Point，简称 $BEP$) 就是这样一个度量,  它是**"查准率=查全率"**时的取值,  例如图 $2.3$ 中学习器 $C$ 的 $BEP$ 是 $0.65$,  而基于 $BEP$ 的比较,  可认为学习器 **$A$ 优于 $B$** . 

### 4 $F1$ 度量

但 $BEP$ 还是过于简化了些,  更常用的是 $F1$ 度量,   $F1$ 的定义如下:
$$
F 1=\frac{2 \times P \times R}{P+R}=\frac{2 \times TP }{样例总数+TP-TN}\tag{2.10}
$$

### 5 $F1$ 的一般形式 $F_{\beta}$ 

在一些实际应用中, 对**查准率核查全率的重视程度有所不同**.  因此引入 $F1$ 度量的一般形式—$F_{\beta}$ ,  能让我们表达出对查准率/查全率的不同偏好,  具体定义为:
$$
F_{\beta}=\frac{\left(1+\beta^{2}\right) \times P \times R}{\left(\beta^{2} \times P\right)+R}\tag{2.11}
$$
其中 $\beta >0$ 度量了**查全率对查准率的相对重要性**.  $\beta=1$ 时退化为标准的 $F1$ ; $\beta>1$ 时, **查全率**有更大影响;  $\beta<1$ 时, **查准率**有更大影响.



### 6 全局性能度量

如何在 $n$ 个二分类混淆矩阵上**综合考察查准率和查全率** 

**宏查准率和宏查全率**

先在各混淆矩阵上**分别计算出查准率和查全率**,  记为 $\left(P_{1}, R_{1}\right),\left(P_{2}, R_{2}\right), \ldots,\left(P_{n}, R_{n}\right)$ ,  再**计算平均值**,  这样就得到"**宏查准率**" ($macro-P$) 、 "**宏查全率**" ($macro-R$) ，以及相应的"**宏 $F1$** " ($macro-F1$):  
$$
\operatorname{macro-P}=\frac{1}{n} \sum_{i=1}^{n} P_{i}\tag{2.12}
$$

$$
\operatorname{macro-} R=\frac{1}{n} \sum_{i=1}^{n} R_{i}\tag{2.13}
$$

$$
\operatorname{macro-} F 1=\frac{2 \times \operatorname{macro-} P \times \operatorname{macro-R}}{\operatorname{macro}-P+\operatorname{macro}-R}\tag{2.14}
$$

**微查准率和微查全率**

先将各混淆矩阵的对应元素进行**平均**，得到 $T P, F P, T N, F N$ 的平均值,  分别记为 $\overline{T P}, \overline{F P}, \overline{T N}, \overline{F N}$ ,  再基于这些平均值计算出"**微查准率** "($micro-P$) 、"**微查全率**" ($micro-R$)和"**微 $F1$**" ($micro-F1$) 
$$
\operatorname{micro}-P=\frac{\overline{T P}}{\overline{T P}+\overline{F P}}\tag{2.15}
$$

$$
\operatorname{micro}-R=\frac{\overline{T P}}{\overline{T P}+\overline{F N}}\tag{2.16}
$$

$$
\operatorname{micro}-F 1=\frac{2 \times \operatorname{micro-} P \times \operatorname{micro}-R}{\operatorname{micro}-P+\operatorname{micro}-R}\tag{2.17}
$$



**小结:**

**先定义查准率 $P$ 和查全率 $R$ ,  接着引入 $P-R$ 曲线, 如果完全包住, 这包住的为更优学习器,  其他情况这计算面积;  但 $P-R$ 曲线中计算面积不易求得,  接着引入比较"平衡点"($BEP$), 即查准率=查全率的点;  但平衡点也过于简单, 于是引入 $F1$ ;  但 $F1$ 中, 查准率与查全率的地位相同的, 为了在实际使用中对查准率或查全率有所侧重, 引入 $F1$ 的一般形式 $F_{\beta}$ ; 最后, 对于多个混淆矩阵, 如何计算全局的 $F1$ , 分为宏和微.**

即 <font color=red>查准率 $P$ 和查全率 $R$ → $P-R$曲线 →"平衡点"($BEP$) → $F1$ →$F_{\beta}$ </font>





## (三) $ROC$ 与 $AUC$

### 1 ROC曲线

很多学习器是**为测试样本产生一个实值**或**概率预测**,  然后将这个**预测值**与一个**分类阔值** (threshold) 进行**比较**,  若**大于阈值**则分为**正类**,  **否则为反类**.  这个实值或概率预测结果的好坏,  直接决定了学习器的**泛化能力**.  

根据这个实值或概率预测结果,  我们可将测试样本进行**排序**,  **"最可能"是正例**的排在最**前面**,  **"最不可能"是正例**的排在最**后面**.  这样,  分类过程就相当于在这个排序中以某个**"截断点**" (cut point) 将**样本分为两部分**,  **前一部分判作正例,  后一部分则判作反例**.  

可根据任务需求来采用不同的截断点,  若我们更重视**"查准率"**,  则可选择排序中**靠前的位置进行截断**;  若更重视"**查全率**",  则可选择**靠后的位置进行截断**.  因此,  **排序本身的质量好坏**,  体现了综合考虑学习器在**不同任务下的"期望泛化性能"的好坏**,  或者说**"一般情况下"泛化性能的好坏**. 



$ROC$ 全称是"**受试者工作特征**" ( Receiver Operating Characteristic ) **曲线** .  与 $P-R$ 曲线相似,  我们根据学习器的**预测结果**对**样例进行排序**,  按此顺序**逐个**把样本作为**正例**进行**预测**,  每次计算出两个重要量的值,  分别以它们为横、纵坐标作图,  就得到了"$ROC$ 曲线.  与 $P-R$ 曲线使用查准率、查全率为纵、横轴不同,  $ROC$ 曲线的**纵轴**是"**真正例率**" (True Positive Rate,  简称 $TPR$ ),  **横轴**是"**假正例率**" (False PositiveRate，简称 $FPR$) ,  基于表 $2.1$ 中的符号,  两者分别定义为:
$$
\mathrm{TPR}=\frac{T P}{T P+F N}\tag{2.18}
$$

$$
\mathrm{FPR}=\frac{F P}{T N+F P}\tag{2.19}
$$

显示 $ROC$ 曲线的图称为"$ROC$ 图". 如图 $2.4(a)$ 

![](https://i.loli.net/2019/06/24/5d108440e173e22856.png)



现实任务中通常是利用有限个测试样例来绘制 $ROC$ 图，此时仅能获得**有限个** (真正例率,  假正例率) **坐标对**,  无法产生图 $2.4(a)$ 中的光滑 $ROC$ 曲线,  只能绘制出如图 $2.4(b)$ 所示的近似 $ROC$ 曲线.



### 2 $AUC$ 曲线

进行学习器的比较时,   与 $P-R$ 图相似,  若一个学习器的 $ROC$ 曲线被另 一个学习器的曲线完全"**包住**",  则可断言**后者**的性能**优**于前者;  若两个学习器的 $ROC$ 曲线发生**交叉**,  则难以一般性地断言两者孰优孰劣.  此时如果一定要进行比较,  则较为合理的判据是比较 **$ROC$ 曲线下的面积**,  即 $AUC$ (Area UnderROC Curve),  如图 $2.4$ 所示

$AUC$ **的计算方法:**

$AUC$ 可通过对 $ROC$ 曲线下各部分的面积求和而得.  假定 $ROC$ 曲线是由坐标为 $\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{m}, y_{m}\right)\right\}$的点按序连接而形成 $\left(x_{1}=0, x_{m}=1\right)$,  参见图 $2.4(b) $,  则 $AUC$ 可估算为:
$$
\mathrm{AUC}=\frac{1}{2} \sum_{i=1}^{m-1}\left(x_{i+1}-x_{i}\right) \cdot\left(y_{i}+y_{i+1}\right)\tag{2.20}
$$
**排序损失:**

$AUC$ 考虑的是样本预测的排序质量,  因此它与**排序误差**有紧密联系.  给定 $m^{+}$ 个正例和 $m^{-}$ 个反例,  令 $D^{+}$ 和$D^{-}$ 分别表示正、反例集合,  则排序"**损失**" (loss)定义为:
$$
\ell_{\text {rank}}=\frac{1}{m^{+} m^{-}} \sum_{x^{+} \in D^{+}} \sum_{x^{-} \in D^{-}}\left(\mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)<f\left(\boldsymbol{x}^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)=f\left(\boldsymbol{x}^{-}\right)\right)\right)\tag{2.21}
$$
即考虑每一对正、反例,  若**正例的预测值小于反例**,  则**记一个"罚分”**,  若**相等**,  则**记 $0.5$ 个"罚分"**.  



$\ell_{\text {rank}}$ 对应的是 $ROC$ 曲线之上的面积:
$$
\mathrm{AUC}=1-\ell_{\text {rank}}\tag{2.22}
$$

> **注1:**  公式 (2.22) 不是很明白, 暂放. 





## (四) 代价敏感错误率与代价曲线

现实任务中,  会有这样的情况:  **不同类型的错误所造成的后果不同**.

**为权衡不同类型错误所造成的不同损失**,  可为错误赋予"**非均等代价**" (unequal cost). 

以二分类任务为例,  我们可根据任务的领域知识设定一个"**代价矩阵**" (cost matrix) ,  如表 $2.2$ 所示，其中 **$\cos t_{i j}$ 表示将第 $i$ 类样本预测为第 $j$ 类样本的代价**.  一般来说,  $\cos t_{i i}=0$ ; 若将第 $0$ 类判别为第 $1$ 类所造成的损失更大,  则$\cos t_{01}>\cos t_{10}$ ;  损失程度相差越大,  $cost_{01}$ 与 $\cos t_{10}$ 值的差别越大.

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190624172033.png)

前面介绍的性能度量，都隐式地假设了**均等代价**.  在非均等代价下,  我们所希望的不再是**简单**地最小化**错误次数**,  而是希望**最小化"总体代价"**(total cost) .

若将表 $2.2$ 中的第 $0$ 类作为正类、第 $1$ 类作为反类,  令 $D^{+}$ 与 $D^{-}$ 分别代表样例集 $D$ 的正例子集和反例子集,  则"**代价敏感**" (cost-sensitive)**错误率**为 
$$
\begin{aligned} E(f ; D ; \operatorname{cost})=& \frac{1}{m}\left(\sum_{x_{i} \in D^{+}} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right) \neq y_{i}\right) \times \operatorname{cost}_{01}\right.\\ &+\sum_{\boldsymbol{x}_{i} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right) \neq y_{i}\right) \times \operatorname{cost}_{10} ) \end{aligned}\tag{2.23}
$$
**代价曲线：**

在非均等代价下,  $ROC$ 曲线不能直接反映出学习器的期望总体代价,  而**"代价曲线"** (cost curve) 则可达到该目的.  代价曲线图的**横轴是取值为 $[0,1]$的正例概率代价**
$$
P(+) \cos t=\frac{p \times \cos t_{01}}{p \times \cos t_{01}+(1-p) \times \cos t_{10}}\tag{2.24}
$$
 其中 **$p$ 是样例为正例的概率**

**纵轴是取值为 $[0,1]$ 的归一化代价**
$$
\operatorname{cost}_{\text {norm}}=\frac{\text { FNR } \times p \times \cos t_{01}+\mathrm{FPR} \times(1-p) \times \cos t_{10}}{p \times \cos t_{01}+(1-p) \times \operatorname{cost}_{10}}\tag{2.25}
$$
其中 $FPR$ 是式 $(2.19)$ 定义的**假正例率**， $FNR =1 - TPR$ 是**假反例率** 

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190624202110.png)