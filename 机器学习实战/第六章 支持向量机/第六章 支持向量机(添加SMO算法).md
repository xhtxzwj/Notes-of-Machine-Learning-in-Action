# 6.1 间隔与支持向量机

## 6.1.1 问题的提出

给定训练样本集 $D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \ldots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}, y_{i} \in\{-1,+1\}$ ,  分类学习最基本的想法就是基于训练集 $D$ 在样本空间中找到一个划分超平面,  将不同类别的样本分开.  能将训练样本分开的划分超平面有很多. 如下图.

![](https://i.loli.net/2019/06/03/5cf479a73040845917.png)

可以看出, 粗线条的划分超平面"容忍性"最好.  换句话说,  这个划分超平面所产生的分类结果是最鲁棒的,  对未见示例的泛化能力最强.



## 6.1.2 数学表示

在样本空间中,  划分超平面可通过如下线性方程来描述:
$$
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=0 \tag{6.1}
$$
其中,  $\boldsymbol{w}=\left(w_{1} ; w_{2} ; \dots ; w_{d}\right)$ 是法向量,  决定了超平面的方向;  $b$ 为位移项,  决定了超平面与原点之间的距离,  划分超平面可被法向量 $\boldsymbol w$ 和位移 $b$ 确定. 将法向量记为 $(\boldsymbol w, b)$ .

样本空间中任意点 $\boldsymbol x$ 到超平面 $(\boldsymbol w, b)$ 的距离可以写成
$$
r=\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|} \tag{6.2}
$$
其中, $\|\boldsymbol{w}\|$ 是向量 $\boldsymbol w$ 的 $L_{2}$ 范数, 也就是  $\|\boldsymbol{w}\|_{2}$ 的简写.  向量的 $L_{2}$ 范数就是向量的模.



假设超平面 $(\boldsymbol w, b)$ 能将训练样本正确分类,  即对于 $\left(\boldsymbol{x}_{i}, y_{i}\right) \in D$ ,  若 $y_{i}=+1$ ,  则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b>0$ ;  若 $y_{i}=-1$,  则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b<0$ . 

**令**
$$
\left\{\begin{array}{ll}{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \geqslant+1,} & {y_{i}=+1} \\ {\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \leqslant-1,} & {y_{i}=-1}\end{array}\right. \tag{6.3}
$$
> ------
>
> **注: 为何是令大于1和小于-1?**
>
> 因为这样标记方便我们将上述 (6.3) 变成如下的形式:
> $$
> y_{i}({\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b) \geqslant+1,}
> $$
> 正是因为标签为1和-1,  才方便我们将约束条件变成一个约束方程,  从而方便我们的计算

------

如图 (6.2) 

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190603100556.png)

如图 (6.2) , 距离超平面最近的这几个训练样本点使式 (6.3) 的等号成立, 它们被称为"支持向量机"(support vector) , 两个异类支持向量到超平面的距离之和为:
$$
\gamma=\frac{2}{\|\boldsymbol{w}\|} \tag{6.4}
$$
称为**"间隔"(margin)**

欲找到具有"最大间隔"(maximum margin)的划分超平面,  也就是找到能满足式(6.3)中约束的参数 $\boldsymbol w$ 和 $b$ , 使得 $\gamma$ 最大,  即
$$
\begin{array}{l}{\max _{\boldsymbol{w}, b} \frac{2}{\|\boldsymbol{w}\|}} \\ {\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}\tag{6.5}
$$
最大化间隔, 即最大化 $\|\boldsymbol{w}\|^{-1}$ , 也就是等价于最小化 $\|\boldsymbol{w}\|^{2}$ .  则 (6.5) 可重写为
$$
\color{red} \begin{array}{l}{\min_{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}} \\ {\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array} \tag{6.6}
$$
这就是<font color=red>**支持向量机**(support vector machine, 简称SVM)的**基本型**.</font>





# 6.2 对偶问题

## 6.2.1 对偶问题

我们的目的就是根据式 (6.6) 来求得最大间隔划分超平面所对应的模型
$$
f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\tag{6.7}
$$
其中 $\boldsymbol w$ 和 $b$ 是模型参数. (6.6)本身是一个凸二次规划问题, 可以直接求解, 但是有更高效的方法.



对式 (6.6) 使用拉格朗日乘子法可得到其"对偶问题"(dual problem). 具体来说, 对式 (6.6) 的每条约束添加拉格朗日乘子 $\alpha_{i} \geqslant 0$ , 则该问题的拉格朗日函数可以写成
$$
\color{red} L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)\tag{6.8}
$$

> ------
>
> **注1:**  <font color=red>**关于拉格朗日对偶性详解**</font>
>
> 令 $\theta(w)=\max _{\alpha_{i} \geq 0} L(w, b, \alpha)$ , 易验证,当某个约束条件不满足时,  例如 $y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)< 1$ ,  那么显然有 $\theta(w)=\infty$ (只要令 $\alpha_{i}=\infty$ 即可).  而当所有约束条件都满足时,  则有 $\theta(w)=\frac{1}{2}\|\boldsymbol w\|^{2}$,  亦即最初要最小化的量.   
>
> 因此,  在要求约束条件得到满足的情况下最小化 $\frac{1}{2}\|\boldsymbol w\|^{2}$,  实际上等价于直接最小化  (当然,  这里也有约束条件,  就是 $\alpha_{i} \geq 0, i=1, \dots, n$) , 因为如果约束条件没有得到满足,   $θ(w) $会等于无穷大,  自然不会是我们所要求的最小值 
>
> 具体写出来,  目标函数变成了:  
> $$
> \min _{\boldsymbol w, b} \theta(\boldsymbol w)=\min _{\boldsymbol w, b} \max _{\alpha_{i} \geq 0} L(\boldsymbol w, b, \boldsymbol \alpha)=p^{*}
> $$
> 把最小和最大的位置交换一下, 变成:
> $$
> \max _{\alpha_{i} \geq 0} \min _{\boldsymbol w, b} L(\boldsymbol w, b, \boldsymbol \alpha)=d^{*}
> $$

------

其中, $\boldsymbol{\alpha}=\left(\alpha_{1} ; \alpha_{2} ; \ldots ; \alpha_{m}\right)$ . 令 $L(\boldsymbol{w}, b, \boldsymbol{\alpha})$ 对 $\boldsymbol w$ 和 $b$ 的偏导为零可得
$$
\boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i} \tag{6.9}
$$

$$
0=\sum_{i=1}^{m} \alpha_{i} y_{i}\tag{6.10}
$$

> ------
>
> **注2:**  **(6.9)和 (6.10) 推导**
>
> 对 $\boldsymbol w$ 求偏导有:
>
> $\frac{\partial L}{\partial w}=\frac{1}{2} \times 2 \times \boldsymbol{w}+0-\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}-0=0 \Longrightarrow \boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}$ (1. $\frac {\partial \boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}}{\partial \boldsymbol{w}} = \boldsymbol{x}_{i}$ 2. $\frac {\partial \boldsymbol{w}^{\mathrm{T}} \boldsymbol{w}}{\partial \boldsymbol{w}} = 2\boldsymbol{w}$)
>
> $\frac {\partial L}{\partial b}=0+0-0-\sum_{i=1}^{m}\alpha_iy_i=0  \Longrightarrow  \sum_{i=1}^{m}\alpha_iy_i=0$

------

将式 (6.9)带入式 (6.8) , 即可将 $L(\boldsymbol{w}, b, \boldsymbol{\alpha})$ 中的 $\boldsymbol w$ 和 $b$ 消去, 再考虑式 (6.10) 的约束, 就得到式<font color=red> **(6.6) 的对偶问题** </font>

> ------
>
> **注3:** **对偶式推导过程**
>
> $\begin{aligned}
> L(\boldsymbol{w},b,\boldsymbol{\alpha})  &=\frac{1}{2}\boldsymbol{w}^T\boldsymbol{w}+\sum_{i=1}^m\alpha_i -\sum_{i=1}^m\alpha_iy_i\boldsymbol{w}^T\boldsymbol{x}_i-\sum_{i=1}^m\alpha_iy_ib \\
> &=\frac {1}{2}\boldsymbol{w}^T\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i-\boldsymbol{w}^T\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i+\sum _{i=1}^m\alpha_
> i -b\sum _{i=1}^m\alpha_iy_i \\
> & = -\frac {1}{2}\boldsymbol{w}^T\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i+\sum _{i=1}^m\alpha_i -b\sum _{i=1}^m\alpha_iy_i
> \end{aligned}$
>
> 又 $0=\sum_{i=1}^{m} \alpha_{i} y_{i}$ , 进一步化简得到
>
> $\begin{aligned}
> L(\boldsymbol{w},b,\boldsymbol{\alpha}) &= -\frac {1}{2}\boldsymbol{w}^T\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i+\sum _{i=1}^m\alpha_i \\
> &=-\frac {1}{2}(\sum_{i=1}^{m}\alpha_iy_i\boldsymbol{x}_i)^T(\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i)+\sum _{i=1}^m\alpha_i \\
> &=-\frac {1}{2}\sum_{i=1}^{m}\alpha_iy_i\boldsymbol{x}_i^T\sum _{i=1}^m\alpha_iy_i\boldsymbol{x}_i+\sum _{i=1}^m\alpha_i \\
> &=\sum _{i=1}^m\alpha_i-\frac {1}{2}\sum_{i=1 }^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j
> \end{aligned}$
>
> 所以,$ \max _{\alpha_{i} \geq 0} \min _{\boldsymbol w, b} L(\boldsymbol w, b, \boldsymbol \alpha)$ , 最后 $L(\boldsymbol w, b, \boldsymbol \alpha)$ 的只有参数 $\alpha_i$ ($\alpha_j$ 一样). 因此 
>
> $ \max _{\alpha_{i} \geq 0} \min _{\boldsymbol w, b} L(\boldsymbol w, b, \boldsymbol \alpha)=\max _{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}$

------

<font color=red> **(6.6) 的对偶问题** </font>
$$
\color{red} \max _{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \tag{6.11}
$$

$$
\color{red} s.t. \begin{array}{l}{\sum\limits_{i=1}^{m} \alpha_{i} y_{i}=0} \\ {\alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m}\end{array}
$$





解出 $\boldsymbol \alpha$ 后, 求出 $\boldsymbol w$ 和$b$ 即可得到模型
$$
\begin{aligned} f(\boldsymbol{x}) &=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}+b \end{aligned} \tag{6.12}
$$
### 6.2.2 KKT条件

从对偶问题 (6.11)解出的 $\alpha_{i}$ 是式 (6.8) 中的拉格朗日乘子, 它恰对应着训练样本$\left(\boldsymbol{x}_{i}, y_{i}\right)$ . 注意到式 (6.6) 中有不等式约束,  因此上述过程需满足 KKT(Karush-Kuhn-Tucker) 条件, 即要求 
$$
\color{red} \left\{\begin{array}{l}{\alpha_{i} \geqslant 0} \\ {y_{i} f\left(\boldsymbol{x}_{i}\right)-1 \geqslant 0} \\ {\alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1\right)=0}\end{array}\right.
$$

> **注4:** 关于拉格朗日对偶性以及KKT条件, 详细知识参见统计学习方法附录C,  



**具体含义:**

于是,  对任意训练样本 $\left(\boldsymbol{x}_{i}, y_{i}\right)$,  总有 $\alpha_{i}=0$ 或 $y_{i} f\left(\boldsymbol{x}_{i}\right)=1$ .

- 若 $\alpha_{i}=0$ ,  则该样本将不会在式 (6.12) 的求和中出现,  也就不会对 $f(\boldsymbol x)$ 有任何影响;  
- 若 $\alpha_{i}>0$,  则必有 $y_{i} f\left(\boldsymbol{x}_{i}\right)=1$ ,  所对应的样本点位于最大间隔边界上,  是一个支持向量.  
- 这显示出支持向量机的一个重要性质:训练完成后,  大部分的训练样本都不需保留,  最终模型仅与支持向量有关 



## 6.2.3 SMO算法简介

#### 1 SMO简介

不难发现,  (6.11) 是一个二次规划问题,  可使用通用的二次规划算法来求解;  然而,  该问题的规模正比于训练样本数 ,  这会在实际任务中造成很大的开销.  为了避开这个障碍,  人们通过利用问题本身的特性,  提出了很多高效算法,   SMO (Sequential Minimal Optimization) 是其中一个著名的代表 [Platt ,  1998].

SMO 的基本思路是先固定 $\alpha_{i}$ 之外的所有参数,  然后求 $\alpha_{i}$ 上的极值.  由于存在约束$\sum_{i=1}^{m} \alpha_{i} y_{i}=0$,  若固定 $\alpha_{i}$ 之外的其他变量,  则 $\alpha_{i}$ 可由其他变量导出.  于是,  SMO 每次选择两个变量 $\alpha_{i}$ 和 $\alpha_{j}$ ,  并固定其他参数.  这样,  在参数初始化后,  SMO 不断执行如下两个步骤直至收敛: 

- 选取一对需更新的变量 $\alpha_{i}$ 和 $\alpha_{j}$ ;
- 固定 $\alpha_{i}$ 和 $\alpha_{j}$ 以外的参数,  求解式 (6.11) 获得更新后的 $\alpha_{i}$ 和 $\alpha_{j}$ 



注意到只需选取的 $\alpha_{i}$ 和 $\alpha_{j}$ 中有一个不满足 KKT 条件 (6.13),  目标函数就会在选代后减小.  直观来看,  KKT 条件违背的程度越大,  则变量更新后可能导致的目标函数值减幅越大.  于是,  SMO 先选取违背 KKT 条件程度最大的变量. 第二个变量应选择一个使目标函数值减小最快的变量,  但由于比较各变量所对应的目标函数值减幅的复杂度过高,  因此 SMO 采用了一个启发式:  **使选取的两变量所对应样本之间的问隔最大**. 一种直观的解释是,  这样的两个变量有很大的差别,  与对两个相似的变量进行更新相比,  对它们进行更新会带给目标函数值更大的变化. 



#### 2 SMO的数学简要解释

SMO 算法之所以高效,  恰由于在固定其他参数后,  仅优化两个参数的过程能做到非常高效.具体来说,  仅考虑 $\alpha_{i}$ 和 $\alpha_{j}$ 时,  式 (6.11) 中的约束可重写为
$$
\alpha_{i} y_{i}+\alpha_{j} y_{j}=c, \quad \alpha_{i} \geqslant 0, \quad \alpha_{j} \geqslant 0 \tag{6.14}
$$
其中, 
$$
c=-\sum_{k \neq i, j} \alpha_{k} y_{k} \tag{6.15}
$$
是使 $\sum_{i=1}^{m} \alpha_{i} y_{i}=0$ 成立的常数. 用
$$
\alpha_{i} y_{i}+\alpha_{j} y_{j}=c \tag{6.16}
$$
消去式 (6.11) 中的变量 $\alpha_{j}$ ,  那么就得到一个关于 $\alpha_{i}$ 的单变量二次规划问题,  仅有的约束是 $\alpha_{i} \geqslant 0$ . 不难发现,  这样的二次规划问题具有闭式解,  于是不必调用数值优化算法即可高效地计算出更新后的 $\alpha_{i}$ 和 $\alpha_{j}$ .



**接下来确定便宜项 $b$** . 注意到对任意支持向量 $(\boldsymbol x_{s}, y_{s})$ (也即是$\alpha_{i}>0$ 时, 为支持向量) , 都有 $y_{s} f\left(\boldsymbol{x}_{s}\right)=1$ , 根据式 (6.12) 进一步有:
$$
y_{s}\left(\sum_{i \in S} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{s}+b\right)=1 \tag{6.17}
$$
其中, $S=\left\{i | \alpha_{i}>0, i=1,2, \dots, m\right\}$ 为所有支持向量的下标集.  理论上, 可选任意支持向量并通过求解式 (6.17) 获得 $b$ , 但现实任务中常采用一种更鲁棒的做法:  使用所有支持向量求解的平均值
$$
b=\frac{1}{|S|} \sum_{s \in S}\left(\frac{1}{y_{s}}-\sum_{i \in S} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{s}\right)\tag{6.18}
$$
其中, $|S|$ 表示集合 $S$ 中元素的数量.





# 6.3 核函数

## 6.3.1 线性不可分的情况

前面的都是假设训练样本是线性可分的,  即存在一个划分超平面能将训练样本正确分类.然而在现实任务中,  原始样本空间内也许并不存在一个能正确划分两类样本的超平面.  如图 6.3 中的" 异或 " 问题就不是线性可分的.

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190603201624.png)

对这样的问题,  **可将样本从原始空间映射到一个更高维的特征空间,  使得样本在这个特征空间内线性可分.** 

如在图 6.3 中,  若将原始的二维空间映射到一个合适的三维空间,  就能找到一个合适的划分超平面.

同时,  **如果原始空间是有限维,  即属性数有限,  那么一定存在一个高维特征空间使样本可分** 



## 6.3.2 数学表示

令 $\phi(\boldsymbol{x})$ 表示将 $\boldsymbol x$ 映射后的特征向量,  于是, 在特征空间中划分超平面所对应的模型可表示为:
$$
f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b \tag{6.19}
$$
其中, $\boldsymbol w$ 和 $b$ 是模型参数. 类似于式 (6.6), 有
$$
\begin{array}{l}{\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}} \\ {\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \phi\left(\boldsymbol{x}_{i}\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array} \tag{6.20}
$$
其对偶问题是
$$
\max _{\boldsymbol \alpha} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right) \tag{6.21}
$$

$$
s.t. \begin{array}{l}{\sum\limits_{i=1}^{m} \alpha_{i} y_{i}=0} \\ {\alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m}\end{array}
$$

$\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right)$ 表示样本 $\boldsymbol{x}_{i}$ 与 $\boldsymbol{x}_{j}$ 映射到特征空间之后的内积. 计算较为困难.  为避开,  设想这样一个函数:
$$
\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\left\langle\phi\left(\boldsymbol{x}_{i}\right), \phi\left(\boldsymbol{x}_{j}\right)\right\rangle=\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right) \tag{6.22}
$$
即  $\boldsymbol{x}_{i}$ 与 $\boldsymbol{x}_{j}$ 在特征空间的内积等于它们在原始样本空间中通过函数 $\kappa(\cdot, \cdot)$ 计算的结果.

于是, 式 (6.21) 可以重写为:
$$
\max _{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \tag{6.23}
$$

$$
s.t. \begin{array}{l}{\sum\limits_{i=1}^{m} \alpha_{i} y_{i}=0} \\ {\alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m}\end{array}
$$

求解后即可得到
$$
\begin{aligned} f(\boldsymbol{x}) &=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi(\boldsymbol{x})+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)+b \end{aligned} \tag{6.24}
$$
这里的函数 $\kappa(\cdot, \cdot)$ 就是<font color=red>**"核函数"**</font>(kernel function)

式 (6.24) 显示出模型最优解可通过训练样本的函数展开, 这一展开式亦称 "支持向量展式"



## 6.3.3 有关核函数的定理

### (一) 定理6.1 (核函数)

定理6.1 (核函数)  令 $\mathcal{X}$ 为输入空间,   $\kappa(\cdot, \cdot)$ 是定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称函数,  则 $k$ 是核函数当且仅当对于任意数据 $D=\left\{\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{m}\right\}$ ,  "核矩阵'' (kernel matrix) $\mathbf{K}$ 总是本正定的:
$$
\mathbf{K}=\left[\begin{array}{ccccc}{\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{m}\right)}\end{array}\right]
$$
定理 6.1 表明,  **只要一个对称函数所对应的核矩阵半正定,  它就能作为核函数使用.**  事实上,  **对于一个半正定核矩阵,  总能找到一个与之对应的映射 $\phi$** .  换言之,  任何一个核函数都隐式地定义了一个称为"再生核希尔伯特空间" (Reproducing Kernel Hilbert Space,  简称 RKHS) 的特征空间. 

### (二) 常用的核函数及核函数的相关组合

表 6.1 列出了集中常用的核函数

![](https://raw.githubusercontent.com/xhtxzwj/picfiles/master/20190603210644.png)

此为, 还可以通过函数组合得到相关的核函数. 

- 若 $k_{1}$ 和 $k_{2}$ 为核函数,  则对于任意正数 $\gamma_{1}, \gamma_{2}$ ,  其线性组合
  $$
  \gamma_{1} \kappa_{1}+\gamma_{2} \kappa_{2} \tag{6.25}
  $$
  也是核函数

- 若 $k_{1}$ 和 $k_{2}$ 为核函数,  则核函数的直积
  $$
  \kappa_{1} \otimes \kappa_{2}(\boldsymbol x, \boldsymbol z)=\kappa_{1}(\boldsymbol x, \boldsymbol z) \kappa_{2}(\boldsymbol x, \boldsymbol z) \tag{6.26}
  $$
  也是核函数

- 若 $k_{1}$ 为核函数,  则对于任意函数 $g(\boldsymbol{x})$ ,
  $$
  \kappa(\boldsymbol{x}, \boldsymbol{z})=g(\boldsymbol{x}) \kappa_{1}(\boldsymbol{x}, \boldsymbol{z}) g(\boldsymbol{z})
  $$
  也是函数



# 6.4 软间隔与正则化

## 6.4.1 软间隔的概念

前面我们假定训练样本在**样本空间**或**特征空间**中是**线性可分**的,  即存在一个超平面能将不同类的样本完全划分开.  然而,  在现实任务中往往很难确定合适的核函数使得训练样本在特征空间中线性可分;  即使恰好找到了 某个核函数使训练集在特征空间中线性可分,  也很难断定这个貌似线性可分的结果不是由于**过拟合**所造成的.  

缓解该问题的一个办法是<font color=red>**允许支持向量机在一些样本上出错**</font>.为此,  要引入"**软间隔**" (soft margin)的概念,  如图6.4所示.

![](https://i.loli.net/2019/06/04/5cf5d97f9bd8d70600.png)



## 6.4.2 数学表示

前面介绍的支持向量机形式是**要求所有样本均满足约束 (6.3)**,  即**所有样本都必须划分正确**,  这称为" **硬间隔**" (hard margin),  而**软间隔则是允许某些样本不满足约束**
$$
y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1\tag{6.28}
$$
在**最大化间隔的同时,  不满足约束的样本应尽可能的少**.  则**优化目标**可以写为
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)\tag{6.29}
$$
其中 $C>0$ 是一个常数,  $\ell_{0 / 1}$ 是 "0/1损失函数"
$$
\ell_{0 / 1}(z)=\left\{\begin{array}{ll}{1,} & {\text { if } z<0} \\ {0,} & {\text { otherwise }}\end{array}\right. \tag{6.30}
$$
<font color=green>**显然,  当 $C$ 为无穷大时, 式 (6.29) 迫使所有样本均满足约束 (6.28) ,  于是式 (6.29) 等价于 式 (6.6) ;  当 $C$ 取有限值时,  式 (6.29) 允许一些样本不满足约束.**</font>



然而,  $\ell_{0 / 1}$ 非凸、非连续,  数学性质不太好,  使得式 (6.29) 不易直接求解. 人们通常用其他一些函数来代替 $\ell_{0 / 1}$,   称为"**替代损失**" (surrogate loss).  替代损失函数一般具有较好的数学性质,  如它们通常是凸的连续函数且是  $\ell_{0 / 1}$ 的上界.  图 6.5 给出了三种常用的替代损失函数: 

- hinge 损失:  $\ell_{\text {hinge}}(z)=\max (0,1-z)\tag{6.31}$

- 指数损失(exponential loss):  $\ell_{e x p}(z)=\exp (-z)\tag{6.32}$

- 对率损失(logistic loss):   $\ell_{\exp }(z)=\exp (-z) \tag{6.33}$

![](https://i.loli.net/2019/06/04/5cf6202aa024751841.png)

若采用hinge损失,  则式 (6.29) 变成
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right) \tag{6.34}
$$

> 注5:  同6.29一样的思路来理解



接着,  引入"**松弛变量**"(slack variable) $\xi_{i} \geqslant 0$ ,  可将式 (6.34) 重写为
$$
\color{red} \min _{\boldsymbol{w}, b, \xi_{i}} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \tag{6.35}
$$

$$
\color{red} s.t. \quad y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i}
$$

​                                                                                  $\color{red}\xi_{i} \geqslant 0, i=1,2, \ldots, m$

这就是常用的<font color=red>**"软间隔支持向量机"**</font>的基本式



> 注6: 式(6.35)是式(6.34)的上限,  最小化式(6.35)的同时也会最化小式(6.34),  这是因为:
>
> 由式(6.35)的约束条件 $y_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i}$ 可得 $\xi_{i} \geqslant 1-y_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)$,  再加上约束条件 $\xi_{i} \geqslant 0$,  即 $\xi_{i} \geqslant \max \left(0,1-y_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)\right)$ ,  因此式(6.35)是式(6.34)的上限 



显然,  式 (6.35) 中每个样本都有一个对应的松弛变量,  用以表征该样本不满足约束 (6.28) 的程度.  

与式 (6.6) 相似,  这还是一个二次规划问题.  于是,  类似于式 (6.8) ,  通过拉格朗日乘子法可得到式 (6.35) 的拉格朗日函数
$$
\begin{aligned} L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})=& \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\ &+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i} \end{aligned} \tag{6.36}
$$
其中,  $\alpha_{i} \geqslant 0, \mu_{i} \geqslant 0$ 是拉格朗日乘子.

> ------
>
> 注7:  **拉格朗日基本形式**为
> $$
> \min _{x \in \mathbf{R}^{n}} f(x)
> $$
>
> $$
> s.t. \quad c_{i}(x) \leqslant 0, \quad i=1,2, \cdots, k
> $$
>
> $$
> h_{j}(x)=0, \quad j=1,2, \cdots, l
> $$
>
> 
> 
> 引入拉格朗日函数有
>$$
> L(x, \alpha, \beta)=f(x)+\sum_{i=1}^{k} \alpha_{i} c_{i}(x)+\sum_{j=1}^{l} \beta_{j} h_{j}(x)
> $$

------

令 $L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})$ 对 $\boldsymbol w, b, \xi_{i}$ 的偏导为零可得
$$
\boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i} \tag{6.37}
$$

$$
0=\sum_{i=1}^{m} \alpha_{i} y_{i}\tag{6.38}
$$

$$
C=\alpha_{i}+\mu_{i}\tag{6.39}
$$

将式 (6.37)-(6.39) 代入式 (6.36) 即可得到<font color=red>**软间隔支持向量机的对偶问题**</font>

------


$$
\color{red} \max _{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \tag{6.40}
$$

$$
\color{red} s.t. \sum_{i=1}^{m} \alpha_{i} y_{i}=0
$$

​                                                                                         $\color{red}0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \ldots, m$

------

软间隔的对偶问题式 (6.40) 与硬间隔下的对偶问题 (6.11) 对比可看出,  两者唯一的差别就再与对偶变量的约束不同: **前者是 $0 \leqslant \alpha_{i} \leqslant C$ ,  后者是 $0 \leqslant \alpha_{i}$ .** 于是,  可采用 6.2节中同样的算法求解式 (6.40);  在引入**核函数**后能得到与式 (6.24) 同样的**支持向量展式**. 



类似于式 (6.13),  对软间隔支持向量机,  KKT条件要求
$$
\color{red} \left\{\begin{array}{l}{\alpha_{i} \geqslant 0, \quad \mu_{i} \geqslant 0} \\ {y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i} \geqslant 0} \\ {\alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i}\right)=0} \\ {\xi_{i} \geqslant 0, \quad \mu_{i} \xi_{i}=0}\end{array}\right.\tag{6.41}
$$


对任意训练样本 $\left(\boldsymbol{x}_{i}, y_{i}\right)$ ,  总有 $\alpha_{i}=0$ 或 $y_{i} f\left(\boldsymbol{x}_{i}\right)=1-\xi_{i}$ .   

- 若 $\alpha_{i}=0$ ,  则该样本不会对 $f(\boldsymbol x)$ 有任何影响;  
- 若  $\alpha_{i}>0$ ,  则必有 $y_{i} f\left(\boldsymbol{x}_{i}\right)=1-\xi_{i}$ ,  即该样本是支持向量:   
  - 由式 (6.39)可知, 若 $\alpha_{i}<C$ ,  则 $\mu_{i}>0$ ,  进而有 $\xi_{i}=0$ ,  即该样本恰在最大间隔边界上;  
  - 若 $\alpha_{i}=C$ ,  则有 $\mu_{i}=0$ , 
    - 此时若 $\xi_{i} \leqslant 1$ , 则该样本落在最大间隔内部, 
    - 若 $\xi_{i}>1$ , 则该样本被错误分类.   

由此可看出,  软间隔支持向量机的最终模型**仅与支持向量有关**,  即通过采用hinge损失函数仍**保持了稀疏性**. 











# 6.5 SMO算法详解

## 6.5.1 概论

### (一) 概念

支持向量机的学习问题可以形式化为求解凸二次规划问题.   这样的凸二次规划问题具有全局最优解,   并且有许多最优化算法可以用于这一问题的求解.   但是当训练样本容量很大时,   这些算法往往变得非常低效,   以致无法使用.   

所以,   如何高效地实现支持向量机学习就成为一个重要的问题.   目前人们已提出许多快速实现算法.   本节讲述其中的序列最小最优化 (sequential minimal optimization,   SMO)  算法,   这种算法1998年由Platt提出.   

### (二) 目标问题–-软间隔对偶问题

首先,  软间隔的对偶问题前面已经说过了, 也就是 (6.40) 



------

$$
\color{red} \max _{\alpha} \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \tag{6.40}
$$

$$
\color{red} s.t. \sum_{i=1}^{m} \alpha_{i} y_{i}=0
$$

​                                                                               $\color{red}0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \ldots, m$

------

同时, 对软间隔支持向量机,  KKT条件要求
$$
\color{red} \left\{\begin{array}{l}{\alpha_{i} \geqslant 0, \quad \mu_{i} \geqslant 0} \\ {y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i} \geqslant 0} \\ {\alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i}\right)=0} \\ {\xi_{i} \geqslant 0, \quad \mu_{i} \xi_{i}=0}\end{array}\right.\tag{6.41}
$$


接下来, 我们变换一下, 先把max变换为min, 然后把 $\boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}$ 用核函数表示为 $K(\boldsymbol x_{i},\boldsymbol x_{j})$ , 关于核函数可参考 6.3.2 小节知识.

那么, 就可以变换为:
$$
\color{red}\min _{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(\boldsymbol x_{i}, \boldsymbol x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \tag{7.98}
$$

$$
\color{red} \text { s.t. } \quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \tag{7.99}
$$

​	     		$\color{red} 0\leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N \tag{7.100}$

在这个问题中, 变量时拉格朗日乘子,  一个变量 $\alpha_{i}$ 对应一个样本点 $(\boldsymbol x_{i},y_{i})$; 变量的总数等于训练样本容量.

### (三) SMO算法的思路

SMO算法是一种启发式算法,  其基本思路是:  **如果所有变量的解都满足此最优化问题的KKT条件 (Karush-Kuhn-Tucker conditions) ,  那么这个最优化问题的解就得到了**.  **因为KKT条件是该最优化问题的充分必要条件**.   否则,  选择**两个变量**,  **固定其他变量**,  **针对这两个变量构建一个二次规划问题**.   这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解,  因为这会使得原始二次规划问题的目标函数值变得更小.   重要的是,  这时子问题可以通过解析方法求解,  这样就可以大大提高整个算法的计算速度.   子问题有两个变量,  **一个是违反KKT条件最严重的那一个,  另一个由约束条件自动确定**.  如此,  SMO算法将原问题不断分解为子问题并对子问题求解,  进而达到求解原问题的目的.   

注意,   **子问题的两个变量中只有一个是自由变量**,  假设 $\alpha_{1},\alpha_{2}$ 为两个变量,  $\alpha_{3}, \alpha_{4}, \cdots, \alpha_{N}$ 固定,   那么由等式约束 (7.99) 可知:
$$
\alpha_{1}=-y_{1} \sum_{i=2}^{N} \alpha_{i} y_{i}
$$
如果 $\alpha_{2}$ 确定, 那么 $\alpha_{1}$ 也随之确定.  所以子问题中同时更新两个变量.

整个SMO算法包括两个部分:  **求解两个变量二次规划的解析方法**和**选择变量的启发式方法** 



## 6.5.2 两个变量二次规划的求解方法

### (一) 优化问题的改写

 不失一般性,  假设选择的两个变量是 $\alpha_{1},\alpha_{2}$ ,  其他变量 $\alpha_{i}(i=3,4, \cdots, N)$ 是固定的.  于是SMO的最优化问题(7.98) ~ (7.100)的子问题可以写成:  
$$
\color{red} \begin{aligned} \min _{\alpha_{1}, \alpha_{2}} \quad W\left(\alpha_{1}, \alpha_{2}\right)=& \frac{1}{2} K_{11} \alpha_{1}^{2}+\frac{1}{2} K_{22} \alpha_{2}^{2}+y_{1} y_{2} K_{12} \alpha_{1} \alpha_{2} \\ &-\left(\alpha_{1}+\alpha_{2}\right)+y_{1} \alpha_{1} \sum_{i=3}^{N} y_{i} \alpha_{i} K_{i 1}+y_{2} \alpha_{2} \sum_{i=3}^{N} y_{i} \alpha_{i} K_{i 2} \end{aligned}\tag{7.101}
$$

$$
\color{red}\text { s.t. } \quad \alpha_{1} y_{1}+\alpha_{2} y_{2}=-\sum_{i=3}^{N} y_{i} \alpha_{i}=\zeta \tag{7.102}
$$

$$
\color{red} 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots,N \tag{7.103}
$$

其中,  $K_{i j}=K\left(x_{i}, x_{j}\right), i, j=1,2, \cdots, N$ ,   $\zeta$ 是常数,  目标函数式 (7.101) 中省略了不含 $\alpha_{1},\alpha_{2}$ 的常数项.

> 注: (7.101)的推导过程
>
> 直接带入即可化简, 可得到结果, 但是需要注意以下两个计算过程:
>
> - 但是要注意一个是 $K_{12}=K_{21}$ , 
> - $y_{1} \alpha_{1} \sum_{i=3}^{N} y_{i} \alpha_{i} K_{i 1}$ =$y_{1} \alpha_{1} \sum_{j=3}^{N} y_{j} \alpha_{i} K_{1 j}$ ,  所以可以合并在一起.



### (二) 约束条件

为了求解两个变量的二次规划问题(7.101)～(7.103),  首先分析约束条件,  然后在此约束条件下求极小.  

由于只有两个变量 $\alpha_{1},\alpha_{2}$ ,  约束可以用二维空间中的图形表示（如图7.8所示）  

![](https://i.loli.net/2019/09/05/n4TCsVNpeHrqLzZ.png)

> 注:  观察约束条件 (7.102) 和 (7.103) , 因为 $y_{1}$ 和 $y_{2}$ 的取值只有{-1,1}, 所以当 $y_{1} \neq y_{2}$ 时, 也就变为图 7.8 左边图, $y_{1} = y_{2}$ 时, 也就时图 7.8 右边图 

不等式约束 (7.103) 使得 $(\alpha_{1},\alpha_{2})$ 在盒子 $[0, C] \times[0, C]$ 内, 等式约束 (7.102) 使 $(\alpha_{1},\alpha_{2})$ 在平行于盒子 $[0, C] \times[0, C]$ 的对角线的直线上.  **因此要求的是目标函数在一条平行于对角线的线段上的最优值**.  这使得两个变量的最优化问题成为**实质上的单变量的最优化问题**,  不妨考虑为变量 $\alpha_{2}$ 的最优化问题.   



假设问题 (7.101)～(7.103) 的初始可行解为 $\alpha_{1}^{\text {old }}, \alpha_{2}^{\text {old }}$ , 最优解为 $\alpha_{1}^{\text {new }}, \alpha_{2}^{\text {new }}$ ,  并且假设在沿着约束方向未经剪辑时 $\alpha_{2}$ 的最优解为 $\alpha^{\text {new,unc}}_{2}$ .

由于 $\alpha_{2}^{\text {new }}$ 需满足不等式约束 (7.103) , 所以最优值 $\alpha_{2}^{\text {new }}$ 的取值范围必须满足条件
$$
L \leqslant \alpha_{2}^{\mathrm{new}} \leqslant H
$$
其中, $L$ 和 $H$ 是 $\alpha_{2}^{\text {new }}$所在的对角线段端点的界.  如果 $y_{1} \neq y_{2}$ , 即图 7.8 左边图, 则有:
$$
L=\max \left(0, \alpha_{2}^{\mathrm{old}}-\alpha_{1}^{\mathrm{old}}\right), \quad H=\min \left(C, C+\alpha_{2}^{\mathrm{old}}-\alpha_{1}^{\mathrm{old}}\right)
$$
如果 $y_{1} = y_{2}$ , 即图 7.8 右边图, 则
$$
L=\max \left(0, \alpha_{2}^{\text {old }}+\alpha_{1}^{\text {old }}-C\right), \quad H=\min \left(C, \alpha_{2}^{\text {old }}+\alpha_{1}^{\text {old }}\right)
$$

> 注:<font color=red> **$L$ 和 $H$ 的推导过程** </font>
>
> 首先根据原问题的约束条件和初始解,最优解有:
> $$
> \begin{array}{l}{\alpha_{1}^{n e w} y_{1}+\alpha_{2}^{n e w} y_{2}=\alpha_{1}^{o l d} y_{1}+\alpha_{2}^{o l d} y_{2}=\zeta} \\ {\quad 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N}\end{array}
> $$
> **第一种情况, 当 $y_{1} \neq y_{2}$ , 即图 7.8 左边图,** 那么有:
> $$
> \alpha_{1}^{o l d}-\alpha_{2}^{o l d}=\alpha_{1}^{new}-\alpha_{2}^{new}=\zeta
> $$
> 进行如下推导:
> $$
> \alpha_{2}^{new}=\alpha_{1}^{new}-(\alpha_{1}^{o l d}-\alpha_{2}^{o l d})\tag{I}
> $$
> 这里需要注意的一点是, $\alpha_{2}^{new}$ 是待求解的, $\alpha_{1}^{new}$ 是变化的.
>
> 又 ${\quad 0 \leqslant \alpha_{2}^{new} \leqslant C}$,  ${\quad 0 \leqslant \alpha_{1}^{new} \leqslant C}$,  那么 $\alpha_{2}^{new}$ 的最小值最小只能到0 , 什么时候取 $0$ 呢, 就是 $(\alpha_{1}^{o l d}-\alpha_{2}^{o l d})<0$ 时, 当 $(\alpha_{1}^{o l d}-\alpha_{2}^{o l d})>0$ 时, 最小值就是 在 $\alpha_{1}^{new}=0$ 时, $I$ 式变为:
>
> $\alpha_{2}^{new}=-(\alpha_{1}^{o l d}-\alpha_{2}^{o l d})$ , 因此, L的取值范围就是 $L=\max \left(0, \alpha_{2}^{\mathrm{old}}-\alpha_{1}^{\mathrm{old}}\right)$
>
> 同理可以求得 $H$ 的取值范围:
> $$
> H=\min \left(C, C+\alpha_{2}^{\mathrm{old}}-\alpha_{1}^{\mathrm{old}}\right)
> $$
> 具体可以参见下图:
>
> ![](https://i.loli.net/2019/09/05/GfsNtClOPkM617D.png)
>
> 
>
> **第二种情况,  如果 $y_{1} = y_{2}$ , 即图 7.8 右边图,** 
>
> 可以根据同样的方法,推导得到 $L$ 和 $H$ 的取值范围:
> $$
> L=\max \left(0, \alpha_{2}^{\text {old }}+\alpha_{1}^{\text {old }}-C\right), \quad H=\min \left(C, \alpha_{2}^{\text {old }}+\alpha_{1}^{\text {old }}\right)
> $$
> 取值范围如下图:
>
> ![](https://i.loli.net/2019/09/05/i1GpmaZRTFN7nqK.png)
>
> 



### (三) 两个变量的解

首先求沿着约束方向未经剪辑**即未考虑不等式约束(7.103)时** $\alpha_{2}$ 的最优解 $\alpha^{\text {new,unc}}_{2}$, 然后再求剪辑后 $\alpha_{2}$ 的解 $\alpha_{2}^{\text {new }}$

为了后面公式的简洁, 记:
$$
g(x)=\sum_{i=1}^{N} \alpha_{i} y_{i} K\left(x_{i}, x\right)+b \tag{7.104}
$$
令
$$
E_{i}=g\left(x_{i}\right)-y_{i}=\left(\sum_{j=1}^{N} \alpha_{j} y_{j} K\left(x_{j}, x_{i}\right)+b\right)-y_{i}, \quad i=1,2 \tag{7.105}
$$
**当 $i=1,2$ 时,  $g(x)$ 为 $x$ 的预测值, $E_{i}$ 为函数 $g(x)$ 对输入 $x_{i}$ 的预测值与真实输出 $y_{i}$ 之差.**



------

<font color=red>**定理 7.6**  两个变量的解</font>

最优化问题 (7.101)～(7.103) 沿着约束方向<font color=red>**未经剪辑时的解是** </font>:
$$
\alpha_{2}^{\text {new, unc }}=\alpha_{2}^{\text {old }}+\frac{y_{2}\left(E_{1}-E_{2}\right)}{\eta} \tag{7.106}
$$
其中,  
$$
\eta=K_{11}+K_{22}-2 K_{12}=\left\|\Phi\left(x_{1}\right)-\Phi\left(x_{2}\right)\right\|^{2} \tag{7.107}
$$
$\Phi\left(x_{1}\right)$ 是输入空间到特征空间的映射,  $E_{i}, \quad i=1,2$ , 由式 (7.105) 给出.

<font color=red>**经剪辑后 $\alpha_{2}$ 的解**</font>是
$$
\alpha_{2}^{\mathrm{new}}=\left\{\begin{array}{ll}{H,} & {\alpha_{2}^{\mathrm{new}, \mathrm{unc}}>H} \\ {\alpha_{2}^{\mathrm{new}, \mathrm{unc}},} & {L \leqslant \alpha_{2}^{\mathrm{new}, \mathrm{unc}} \leqslant H} \\ {L,} & {\alpha_{2}^{\mathrm{new}, \mathrm{unc}}<L}\end{array}\right. \tag{7.108}
$$
由 $\alpha_{2}^{\text {new }}$ 求得 $\alpha_{1}^{\text {new }}$是:
$$
\alpha_{1}^{\mathrm{new}}=\alpha_{1}^{\mathrm{old}}+y_{1} y_{2}\left(\alpha_{2}^{\mathrm{old}}-\alpha_{2}^{\mathrm{new}}\right) \tag{7.109}
$$

------

> **注: 关于定理的推导过程.**
>
> 1. 关于**未经剪辑时的解的推导**过程:
>
> 请参考李航<统计学习方法> p127-128的证明
>
> 2. 经剪辑后的解 (7.108) 的解释:
>
> 要使其满足不等式约束必须将其限制在区间 $[L,  H]$内,  从而得到 $\alpha_{2}^{\text {new }}$ 的表达式 (7.108)
>
> 3. $\alpha_{1}^{\text {new }}$ 的解 (7.109) 的解释:
>
> 由等式约束 (7.102) , 得到  $\alpha_{1}^{\text {new }}$ 的表达式 (7.109) 



## 6.5.3 变量的选择方法

SMO算法在每个子问题中选择两个变量优化,  其中至少一个变量是违反KKT条件的.

### (一) 第 1 个变量的选择

SMO称选择第1个变量的过程为外层循环.  外层循环在训练样本中选取违反 KKT 条件最严重的样本点,  并将其对应的变量作为第1个变量.  具体地,  检验训练样本点 $\left(x_{i}, y_{i}\right)$ 是否满足KKT条件,  即
$$
\alpha_{i}=0 \Leftrightarrow y_{i} g\left(x_{i}\right) \geqslant 1 \tag{7.111}
$$

$$
0<\alpha_{i}<C \Leftrightarrow y_{i} g\left(x_{i}\right)=1 \tag{7.112}
$$

$$
\alpha_{i}=C \Leftrightarrow y_{i} g\left(x_{i}\right) \leqslant 1 \tag{7.113}
$$

其中,  $g\left(x_{i}\right)=\sum_{j=1}^{N} \alpha_{j} y_{j} K\left(x_{i}, x_{j}\right)+b$ .

> 注1 : 关于 (7.111)~(7.113) 的推导:
>
> 1. $\alpha_{i}=0$ 
>
> 由(6.39) 知: $C=\alpha_{i}+\mu_{i}$ , 可得:
> $$
> \mu_{i}=C
> $$
> 再由对偶问题的 kkt 条件 (6.41) 中的 $ \mu_{i} \xi_{i}=0$ 可知:
> $$
> \xi_{i}=0
> $$
> 再由 kkt 条件中的 ${y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i} \geqslant 0}$ (或者原始问题的约束条件, 是一样的), 有:
> $$
> y_{i} g\left(x_{i}\right) \geqslant 1
> $$
>
> 2. $0<\alpha_{i}<C$
>
> 若  $\alpha_{i}>0$ ,  则必有 $y_{i} f\left(\boldsymbol{x}_{i}\right)=1-\xi_{i}$ ,  即该样本是支持向量, 由式 (6.39), 即 $C=\alpha_{i}+\mu_{i}$ 可知, 若 $\alpha_{i}<C$ ,  则 $\mu_{i}>0$ ,  根据 $ \mu_{i} \xi_{i}=0$ ,  进而有 $\xi_{i}=0$ ,  即该样本恰在最大间隔边界上; 所以有, $y_{i} f\left(\boldsymbol{x}_{i}\right)=1$ , 即 $y_{i} g\left(x_{i}\right)=1$
>
> 3. $\alpha_{i}=C$ 
>
> 首先,  $\xi_{i}\geqslant0$ , 同时, 由于 $\alpha_{i}=C$ , 那么由 $\alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i}\right)=0$ 可得,  $\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i}\right)=0$, 所以 $y_{i} f(\boldsymbol{x}_{i})\leqslant1$



> 注2: 其实  (7.111)~(7.113) 就是 kkt 条件 (6.41) 的充要条件, 两者可以互相推出.



检验是在精度 $\varepsilon$ 范围内进行的.  在检验过程中,  外层循环首先遍历所有满足条件 $0<\alpha_{i}<C$ 的样本点,  即在间隔边界上的支持向量点,  检验它们是否满足KKT条件.  如果这些样本点都满足KKT条件,  那么遍历整个训练集,  检验它们是否满足KKT条件.  



### (二) 第 2 个变量的选择

SMO称选择第2个变量的过程为内层循环.  假设在外层循环中已经找到第1个变量 $\alpha_{1}$ ,  现在要在内层循环中找第2个变量 $\alpha_{2}$ .  第2个变量选择的标准是希望能使 $\alpha_{2}$ 有足够大的变化.  由式 (7.106) 和式(7.108) 可知,   是依赖于$|E_{1}-E_{2}|$ 的,   为了加快计算速度,   一种简单的做法是选择 $\alpha_{2}$ ,   使其对应的 $|E_{1}-E_{2}|$ 最大.  因为 $\alpha_{1}$ 已定,   $E_{1}$ 也确定了.  如果 $E_{1}$ 是正的,  那么选择最小的 $E_{i}$ 作为$E_{2}$;  如果 $E_{1}$ 是负的,   那么选择最大的 $E_{i}$作为$E_{2}$.  为了节省计算时间,   将所有 $E_{i}$ 值保存在一个列表中.  在特殊情况下,   如果内层循环通过以上方法选择的 $\alpha_{2}$ 不能使目标函数有足够的下降,  那么采用以下启发式规则继续选择 $\alpha_{2}$ .  遍历在间隔边界上的支持向量点,   依次将其对应的变量作为 $\alpha_{2}$ 试用,   直到目标函数有足够的下降.  若找不到合适的 $\alpha_{2}$ ,   那么遍历训练数据集;  若仍找不到合适的 $\alpha_{2}$ ,   则放弃第1个 $\alpha_{1}$ ,   再通过外层循环寻求另外的 $\alpha_{1}$  



### (三) 计算阈值 $b$ 和差值 $E_{i}$ 

在每次完成两个变量的优化后,  都要重新计算阈值b。 当 $0<\alpha_{1}^{\mathrm{new}}<\mathrm{C}$ 时,  由 KKT 条件 (7.112) 可知: 
$$
\sum_{i=1}^{N} \alpha_{i} y_{i} K_{i 1}+b=y_{1}
$$

> 注:  $\left(x_{1}, y_{1}\right)$ 也满足(7.112), 两边同乘以 $y_{1}$ , 有:
> $$
> y^{2}_{1} g\left(x_{i}\right)=y_{1}
> $$
> 又 $y^{2}_{1}=1$ , 即可得到上述结论



于是, 可得:
$$
b_{1}^{\mathrm{new}}=y_{1}-\sum_{i=3}^{N} \alpha_{i} y_{i} K_{i 1}-\alpha_{1}^{\mathrm{new}} y_{1} K_{11}-\alpha_{2}^{\mathrm{new}} y_{2} K_{21} \tag{7.114}
$$
由 $E_{1}$ 的定义式 (7.105) 有:
$$
E_{1}=\sum_{i=3}^{N} \alpha_{i} y_{i} K_{i 1}+\alpha_{1}^{\mathrm{old}} y_{1} K_{11}+\alpha_{2}^{\mathrm{old}} y_{2} K_{21}+b^{\mathrm{old}}-y_{1}
$$
式 (7.114) 的前两项可以通过 $E_{1}$ 改写为:
$$
y_{1}-\sum_{i=3}^{N} \alpha_{i} y_{i} K_{i 1}=-E_{1}+\alpha_{1}^{\mathrm{old}} y_{1} K_{11}+\alpha_{2}^{\mathrm{old}} y_{2} K_{21}+b^{\mathrm{old}} 
$$
带入式 (7.114) , 可得:
$$
b_{1}^{\text {new }}=-E_{1}-y_{1} K_{11}\left(\alpha_{1}^{\text {new }}-\alpha_{1}^{\text {eld }}\right)-y_{2} K_{21}\left(\alpha_{2}^{\text {new }}-\alpha_{2}^{\text {old }}\right)+b^{\text {old }} \tag{7.115}
$$
那么, 同样的, 如果 $0<\alpha_{2}^{\mathrm{new}}<C$ , 则有:
$$
b_{2}^{\text {new }}=-E_{2}-y_{1} K_{12}\left(\alpha_{1}^{\text {new }}-\alpha_{1}^{\text {old }}\right)-y_{2} K_{22}\left(\alpha_{2}^{\text {new }}-\alpha_{2}^{\text {old }}\right)+b^{\text {old }} \tag{7.116}
$$

- **如果 $\alpha_{1}^{\mathrm{new}}$ , $\alpha_{2}^{\mathrm{new}}$ 同时满足条件 $0<\alpha_{i}^{\mathrm{new}}<C, i=1,2$ (也就是 $b_{1}^{\text {new }}$ 和 $b_{2}^{\text {new }}$ 都有效的时候), 他们是相等的, 即  $b^{\text {new }}=b_{1}^{\text {new }}=  b_{2}^{\text {new }}$** 
- **如果 $\alpha_{1}^{\mathrm{new}}$ , $\alpha_{2}^{\mathrm{new}}$ 是 0 或者 C , 那么  $b_{1}^{\text {new }}$ 和 $b_{2}^{\text {new }}$ 以及他们两者之间的数都是符合 KKT 条件的阈值, 这时选择它们的中点作为 $b^{n e w}=\frac{b_{1}^{n e w}+b_{2}^{n e w}}{2}$**





### 6.5.4 SMO算法总结

------

<font color=red>**算法 7.5 (SMO算法)** </font>

**输入:** 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$ , 其中, $x_{i} \in \mathcal{X}=\mathbf{R}^{n}$ , $y_{i} \in \mathcal{Y}=\{-1,+1\}$, $\quad i=1,2, \cdots, N$ , 精度 $\mathcal{E}$ 

**输出:** 近似解 $\hat{\alpha}$ 

- (1) 取初值 $\alpha^{(0)}=0$ ,  令 $k=0$ ;

- (2) 按照 6.5.3 变量的选择方法中第一个变量选择, 选择第一个变量 $\alpha_{1}^{(k)}$ , 按照第二个变量选择方法选择第二个变量 $\alpha_{2}^{(k)}$ ,  根据式 (7.106) , 求出新的 $\alpha_{2}^{\text {new, unc }}$ , 
  $$
  \alpha_{2}^{\text {new, unc }}=\alpha_{2}^{(k)}+\frac{y_{2}\left(E_{1}-E_{2}\right)}{\eta}
  $$

- (3) 按照下式 (即式 (7.108) ) 求出 $\alpha_{2}^{(k+1)}$ 
  $$
  \alpha_{2}^{(k+1)}=\left\{\begin{array}{ll}{H,} & {\alpha_{2}^{\mathrm{new}, \mathrm{unc}}>H} \\ {\alpha_{2}^{\mathrm{new}, \mathrm{unc}},} & {L \leqslant \alpha_{2}^{\mathrm{new}, \mathrm{unc}} \leqslant H} \\ {L,} & {\alpha_{2}^{\mathrm{new}, \mathrm{unc}}<L}\end{array}\right.
  $$

- (4) 利用  $\alpha_{2}^{(k+1)}$ 和  $\alpha_{1}^{(k+1)}$ 的关系(即式 (7.109) ) , 求出   $\alpha_{1}^{(k+1)}$ .
  $$
  \alpha_{1}^{(k+1)}=\alpha_{1}^{(k)}+y_{1} y_{2}\left(\alpha_{2}^{(k)}-\alpha_{2}^{(k+1)}\right)
  $$

- (5) 按照 6.5.3 变量的选择方法中的 (三) 计算阈值 $b$ 和差值 $E_{i}$ , 计算 $b^{k+1}$ 和 $E_{i}$ 

- (6) 在精度 $\mathcal{E}$ 范围内检查是否满足如下的终止条件:
  $$
  \sum_{i=1}^{N} \alpha_{i} y_{i}=0
  $$

  $$
  0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
  $$

  $$
  \alpha_{i}^{k+1}=0 \Rightarrow y_{i} g\left(x_{i}\right) \geq 1
  $$

  $$
  0<\alpha_{i}^{k+1}<C \Rightarrow y_{i} g\left(x_{i}\right)=1
  $$

  $$
  \alpha_{i}^{k+1}=C \Rightarrow y_{i} g\left(x_{i}\right) \leq 1
  $$

- (7) 如果满足则结束, 返回 $\alpha_{i}^{k+1}$, 否则转到步骤 (2) 

------


