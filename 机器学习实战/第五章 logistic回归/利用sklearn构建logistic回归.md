<center><font size=6>第 5 章 利用sklearn构建logistic回归</font></center>
sklearn.linear_model模块实现了广义线性回归模型. 它包括了岭回归, 贝叶斯回归, lasso回归, 基于最小角度回归的弹性网络估计和坐标下降法等. 在本章, 使用**LogisticRegression**.

![](https://i.loli.net/2019/08/30/HAkCLx5wiVDpfv1.png)

# 一 LogisticRegression

LogisticRegression一共有15个参数:

![](https://i.loli.net/2019/08/30/TKerHf3gILtCQBJ.png)





**参数说明**

各个参数的具体含义如下:

- **penalty:  **  **惩罚项,  str类型,  可选参数为l1和l2,  默认为l2**.  用于指定惩罚项中使用的规范.  newton-cg、sag和lbfgs求解算法只支持L2规范.  弹性网络仅有saga求解算法支持. 如果是‘none', 就没有应用具体规则.

- **dual:  **  **对偶或原始方法,  bool类型,  默认为False(即dual)**.  对偶方法只用在求解**线性多核(liblinear)**的L2惩罚项上.  当样本数量>样本特征的时候,  dual通常设置为False.  

- **tol:  **  **停止求解的标准,  float类型,  默认为1e-4**.  就是求解到多少的时候,  停止,  认为已经求出最优解.  

- **c:  **  **正则化系数 λ 的倒数,  float类型,  默认为1.0**.  必须是**正浮点型数**.  像SVM一样,  **越小**的数值表示**越强的正则化**.  

- **fit_intercept:  **  **是否存在截距或偏差,  bool类型,  默认为True**.  

- **intercept_scaling:  ** **仅在求解算法(solver)为"liblinear"**,  且**fit_intercept设置为True时有用**.  float类型,  **默认为1**.  

- **class_weight:  **用于表示**模型中各个类别的权重**,  可以是一个**字典或者'balanced'字符串**,  **默认不输入**,  为**None**, 也就是不考虑权重,即所有类别的权重为1.   

  如果选择输入的话,  可以选择**balanced让类库自己计算类型权重**,  或者**自己输入各个类型的权重**.  

  - 如,  0-1二元模型,  可定义class_weight={0:0.9,1:0.1},  这样类型0的权重为90%,  而类型1的权重为10%; 
  - 若class_weight选择balanced,  那么类库会根据训练样本量来计算权重.  **某种类型样本量越多,  则权重越低,  样本量越少,  则权重越高**.  当class_weight为balanced时,  类权重计算方法如下:  n_samples / (n_classes * np.bincount(y)).  n_samples为样本数,  n_classes为类别数量,  np.bincount(y)会输出每个类的样本数,  例如y=[1,0,0,1,1],则np.bincount(y)=[2,3].  

- **random_state:  **随机数种子,  int类型,  可选参数,  默认为无,  仅在正则化优化算法为sag,liblinear时有用.  

- **solver:  **srt类型, 优化算法的选择. solver参数决定了我们对逻辑回归损失函数的优化方法,  共有五个可选参数,  即newton-cg,lbfgs,liblinear,sag,saga.  默认为liblinear. 详细介绍各个算法的优劣和适用范围:

  - liblinear适用于小数据集,  而sag和saga适用于大数据集,  因为速度更快.  
  - 对于多分类问题,  只有newton-cg,sag,saga和lbfgs能够处理多项损失,  而liblinear受限于一对剩余(OvR).
  - newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数,  因此不能用于没有连续导数的L1正则化,  只能用于L2正则化.  而liblinear和sagaL1正则化和L2正则化都可以.

- **max_iter:**  算法收敛最大迭代次数,  int类型,  默认为10.  仅在正则化优化算法为newton-cg, sag和lbfgs才有用,  算法收敛的最大迭代次数.  

- **multi_class：**分类方式选择参数,  str类型,  可选参数为ovr和multinomial,  默认为ovr. 

- **verbose：**日志冗长度,  int类型.  默认为0.  就是不输出训练过程,  1的时候偶尔输出结果,  大于1,  对于每个子模型都输出.  

- **warm_start：**热启动参数,  bool类型.  默认为False.  如果为True,  则下一次训练是以追加树的形式进行 (重新使用上一次的调用作为初始化) .  

- **n_jobs：**并行数.  int类型,  默认为1.  1的时候,  用CPU的一个内核运行程序,  2的时候,  用CPU的2个内核运行程序.  为-1的时候,  用所有CPU的内核运行程序.  

- **l1_radio:**  弹性网络混合参数, float或None,  可选 (默认为None) . 0 <= l1_ratio <= 1.仅在惩罚='elasticnet'时使用.  设置``l1_ratio = 0相当于使用penalty ='l2',  而设置l1_ratio = 1相当于使用penalty ='l1'.  对于0 <l1_ratio <1,  惩罚是L1和L2的组合.  



**主要方法**

同时,  LogisticRegression 还有一些方法, 具体如下: 

![](https://i.loli.net/2019/09/02/io1vbhmUu2FVKPf.png)



# 二 利用sklearn中logistic模块进行分类

直接看代码:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
函数说明: 利用sklearn.liner_model.LogisticRgression进行拟合和
            准确度的测量.

Parameters:
    无
Returns:
    errorRate - 分类错误率
"""
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    #读取测试数据集进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        #每一行作为一个样本,样本数加一
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    classifier = LogisticRegression()
    classifier.fit(np.array(trainingSet), trainingLabels)
    test_accury = classifier.score(testSet,testLabels)
    print(test_accury)

if __name__ == '__main__':
    colicTest()
```

结果如下:

![](https://i.loli.net/2019/09/02/UI3W7tC64piqh2f.png)



可以对比之前自己构建的logistic分类器计算测试集的准确度的代码,  可见用sklearn构建的logistic分类的代码简洁了不是一行两行. 而且内部还有很多参数可调.





logistic回归(实际是分类),  就这么多了. 下一节, 开始, 比较难的支持向量机.

