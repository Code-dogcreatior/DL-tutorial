# Diffusion Model

当前主要有四大生成模型：生成对抗模型、变微分自动编码器、流模型以及扩散模型。**扩散模型（diffusion models）**是当前深度生成模型中新SOTA。扩散模型在图片生成任务中超越了原SOTA：GAN，并且在诸多应用领域都有出色的表现，如计算机视觉，NLP、波形信号处理、多模态建模、分子图建模、时间序列建模、对抗性净化等。此外，扩散模型与其他研究领域有着密切的联系，如稳健学习、表示学习、强化学习。本文主要是从其数学原理以及代码方面对扩散模型进行讲解。





## 先验知识

### 贝叶斯法则

先验概率 $P\left ( \theta\right ) : $根据以往的经验和分析得到的，假设$\theta$ 发生的概率；
似然函数 $P(x\mid\theta)$ : 在假设 $\theta$ 已发生的前提下，发生事件 $x$ 的概率；
后验概率 $P\left(\theta\mid x\right)$: 在事件 $x$ 已发生的前提下，假设 $\theta$ 成立的概率；
标准化常量 $P( x) : $在已知所有假设下，事件$x$ 发生的概率；
$$
P\left(\theta\mid x\right)=\frac{P(x|\theta)P(\theta)}{P(x)}
$$

### 贝叶斯公式

$$
&P(A,B)=P(B\mid A)P(A)=P(A\mid B)P(B) \\
&P(A,B,C)=P(C\mid B,A)P(B,A)=P(C\mid B,A)P(B\mid A)P(A) \\
&P(B,C\mid A)=P(B\mid A)P(C\mid A,B)
$$

若满足马尔科夫链关系 $A\to B\to C_{\prime}$ 即当前时刻的概率分布仅与上一时刻有关，则有：
$$
&P(A,B,C)=P(C\mid B,A)P(B,A)=P(B\mid A)P(A)\color{red}{P(C\mid B)} \\
&P(B,C\mid A)=P(B\mid A)\color{red}{P(C\mid B)}
$$

### 高斯分布的概率密度函数、高斯函数的叠加公式

给定均值为 $\mu$ , 方差为 $\sigma^2$ 的单一变量高斯分布 $\mathcal{N}(\mu,\sigma^2)$ , 其概率密度函数为：

很多时候，为了方便起见，可以将前面的常数系数去掉，写成：
$$
q(x)\propto exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)\quad\Leftrightarrow\quad q(x)\propto\exp\!\left(-\frac{1}{2}\left(\frac{1}{\sigma^2}x^2-\frac{2\mu}{\sigma^2}x+\frac{\mu^2}{\sigma^2}\right)\right)
$$
 给定两个高斯分布 $X\sim\mathcal{N}(\mu_1,\sigma_1^2)$ , $Y\sim\mathcal{N}(\mu_2,\sigma_2^2)$ ,则它们叠加后的 $aX+bY$ 满足：

$$
aX+bY\sim\mathcal{N}(a\times\mu_1+b\times\mu_2,a^2\times\sigma_1^2+b^2\times\sigma_2^2)
$$

### KL散度

#### 1.定义

- 两个概率分布(probability distribution)间差异的非对称性度量；
- 参与计算的一个概率分布为真实分布，另一个为理论（拟合）分布，相对熵表示使用理论分布拟合真实分布时产生的信息损耗。

设 $P(x),Q(x)$ 是随机变量 $X$ 上的两个概率分布，则在离散随机变量的情形下，KL散度的定义为：

$$
\operatorname{KL}(P\|Q)=\sum P(x)\log\frac{P(x)}{Q(x)}
$$

在连续随机变量的情形下，KL散度的定义为：
$$
\operatorname{KL}(P\|Q)=\int P(x)\log\frac{P(x)}{Q(x)}dx
$$

#### 2.标准正态分布KL散度计算: $\mathcal{N}\left(\mu,\sigma^2\right)\text{与}\mathcal{N}\left(0,1\right)$

正态分布 $X\sim\mathcal{N}\left(\mu,\sigma^2\right)$ 的概率密度函数为：

$$
p(x)=\frac1{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}
$$

标准正态分布 $X\sim\mathcal{N}\left(0,1\right)$ 的概率密度函数为：
$$
q(x)=\frac1{\sqrt{2\pi}}e^{-\frac{x^2}2}
$$
KL散度计算:
$$
\begin{aligned}
&KL\left(\mathcal{N}\left(\mu,\sigma^{2}\right)\left\Vert\mathcal{N}(0,1)\right)\right. \\
&=\sum p(x)\log\frac{p(x)}{q(x)} \\
&=\int\frac1{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}\left(\log\frac{\frac1{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}}{\frac1{\sqrt{2\pi}}e^{-x^2/2}}\right)dx \\
& =\int\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-(x-\mu)^{2}/2\sigma^{2}}\log\biggl\{\frac{1}{\sqrt{\sigma^{2}}}\exp\biggl\{\frac{1}{2}\bigl[x^{2}-(x-\mu)^{2}/\sigma^{2}\bigr]\biggr\}\biggr\}dx  \\
& =\frac{1}{2}\int\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-(x-\mu)^{2}/2\sigma^{2}}\left[-\log\sigma^{2}+x^{2}-(x-\mu)^{2}/\sigma^{2}\right]dx  \\
& =\frac{1}{2}\int p(x)\left[-\log\sigma^2+x^2-(x-\mu)^2/\sigma^2\right]dx 
\end{aligned}
$$
整个结果分为三项积分，第一项实际上就是 $-log\sigma^2$ 乘以概率密度的积分 (也就是 1) , 所以结果是 $-log\sigma^2$ ; 第二项实际是正态分布的二阶矩，熟悉正态分布应该都清楚正态分布的二阶矩为$\mu^2+\sigma^2$;而根据定义，第三项实际上就是“-方差除以方差=-1”。所以总结果就是：

##### 2.1零阶矩、一阶矩、二阶矩

**意义**
**物理意义**
  在物理学中，矩是表示距离和物理量乘积的物理量，表征物体的空间分布。矩通常需要一个 参考点 （基点或参考系）来定义距离。如力和参考点距离乘积得到的力矩（或扭矩），原则上任何物理量和距离相乘都会产生力矩，质量，电荷分布等。

如果点表示质量：

- 零阶矩 是总质量
- 一阶原点矩 是质心
- 二阶原点矩 是转动惯量

如果点表示高度：

- 零阶矩 是所有点高度之和
- 一阶原点矩 是点的位置和对应高度乘积之和，表示所有高度的中心
- 二阶中心矩 是所有点的高度波动范围

**数学意义**

数学上，“矩”是一组点组成的模型的特定的数量测度。
定义： 设 $X$ 和 $Y$ 是离散随机变量，$c$ 为常数，$k$ 为正整数， 如果$E(|X-c|^k)$存在，则称$E(|X-c|^k)$为X关于点$c$的$k$阶矩。

$c=0$ 时，称为$k$阶原点矩；
$C=E(X)$时，称为k阶中心矩。
如果$E(|X-c1|^p\bullet|Y-c2|^q)$存在，则称其为$X，Y $关于 $ c$ 点的 $ p+q$ 阶矩。
$c1=c2=0$ 时，称为 $p+q$ 阶混合原点矩$c1=E(X), c2=E(Y)$时，称为 $p+q$ 阶混合中心距。
如果 $X,Y$ 是连续型的，则$\int_k(x-c)^kdx$ 称为X关于点 c 的 k阶原点矩， $\iint_{p+q}(x-x_0)^p\bullet(y-y_0)^qdxdy$ 称为 $X,Y$ 关于点 $c$ 的 $p+q$ 阶混合中心矩。矩的本质是数学期望，而期望的计算公式应该是$E(x)=\int x\bullet f(x)dx$,其中 $f(x)$是 $x$ 的概率密度，也就说上面的公式其实默认了所有随机变量出现的概率相等。

#### 3.正态分布KL散度计算：$X\sim\mathcal{N}\left(\mu_1,\sigma_1^2\right)\text{与}\mathcal{N}\left(\mu_2,\sigma_2^2\right)$

正态分布：$X\sim\mathcal{N}\left(\mu_{1},\sigma_{1}^{2}\right)$的概率密度函数：
$$
p(x)=\frac1{\sqrt{2\pi{\sigma_1}^2}}e^{-(x-\mu_1)^2/2{\sigma_1}^2}
$$


正态分布：$X\sim\mathcal{N}\left(\mu_2,\sigma_2^2\right)$的概率密度函数：
$$
q(x)=\frac1{\sqrt{2\pi\sigma_2{^2}}}e^{-(x-\mu_2)^2/2\sigma_2{^2}}
$$
KL散度计算：
$$
\begin{aligned}
&KL\left(\mathcal{N}\left(\mu_{1},\sigma_{1}^{2}\right)\|\mathcal{N}(\mu_{2},\sigma_{2}^{2})\right) \\
&=\sum p(x)\log\frac{p(x)}{q(x)} \\
&=\int\frac1{\sqrt{2\pi\sigma_1^2}}e^{-(x-\mu_1)^2/2\sigma_1^2}\left(\log\frac{\frac1{\sqrt{2\pi\sigma_1^2}}e^{-(x-\mu_1)^2/2\sigma_1^2}}{\frac1{\sqrt{2\pi\sigma_2^2}}e^{-(x-\mu_2)^2/2\sigma_2^2}}\right)dx \\
&=\int\frac{1}{\sqrt{2\pi\sigma_1^2}}e^{-\left(x-\mu_1\right)^2/2\sigma_1^2}\log\left\{\frac{\sqrt{\sigma_2^2}}{\sqrt{\sigma_1^2}}\exp\left\{\frac{1}{2}\left[\frac{(x-\mu_2)^2}{\sigma_2^2}-\frac{(x-\mu_1)^2}{\sigma_1^2}\right]\right\}\right\}dx \\
&=\frac12\int\frac1{\sqrt{2\pi\sigma_1^2}}e^{-(x-\mu_1)^2/2\sigma_1^2}\left[\log\sigma_2^2-\log\sigma_1^2+\frac{(x-\mu_2)^2}{\sigma_2^2}-\frac{(x-\mu_1)^2}{\sigma_1^2}\right]dx \\
&=\frac{1}{2}\int p(x)\left[\log\sigma_{2}^{2}-\log\sigma_{1}^{2}+\frac{(x-\mu_{2})^{2}}{\sigma_{2}^{2}}-\frac{(x-\mu_{1})^{2}}{\sigma_{1}^{2}}\right]dx
\end{aligned}
$$
整个结果分为四项积分，第一项实际上就是 $log\sigma_{2}^{2}$ 乘以概率密度的积分 分(也就是 1), 所以结果是 $log\sigma_2^2$; 第二项实际上就是 $-log\sigma_1^2$ 乘以概率密度的积分 (也就是 1)), 所以结果是 $-log\sigma_1^2$ ; 第三项实际是异正态分布的二阶矩，熟悉正态分布的朋友应该都清楚异正态分布的二阶矩为

 $\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{\sigma_3^2}$ ; 而根据定义，第四项实际上就是“-方差除以方差=-1”。所以总结果就是：

$$
KL\left(\mathcal{N}\left(\mu_1,\sigma_1^2\right)\|\mathcal{N}(\mu_2,\sigma_2^2)\right)=\frac{1}{2}\left(\log\sigma_2^2-\log\sigma_1^2+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{\sigma_2^2}-1\right)
$$

#### 凸优化-简介

在这里先介绍有用到的坐标下降法，当然实际上还有很多，例如对偶上升法、增强拉格朗日等等，我们仅仅在这里看最基础的部分，但我也列下了相关一些资料，如果有时间后面会填坑。

**Source**

- A. Beck and L. Tetruashvili (2013), *On the convergence of block coordinate descent type methods*
- X. Li, T. Zhao, R. Arora, H. Liu, M. Hong (2016), *An improved convergence analysis of cyclic block coordinate descent-type methdos for strongly convex minimization*
- R. Sun, M. Hong (2015), *Improved iteration complexity bounds of cyclic block coordinate descent for convex problems*
- S. Wright (2015), *Coordinate Descent Algorithms*
- CMU 10-725, *Convex Optimization*
- Boyd, Vandenberghe, *Convex Optimization*

##### Coordinate Descent（坐标下降法）

它考虑的就是这么一个问题：**如果函数在每一个维度上都达到了极小值，能否保证函数已经找到了极小值点**？

如果写成数学式子，就是：
$$
\forall\delta,f(x+\delta e_i)\geq f(x)\to f(x)=\min_zf(z)
$$
这里$e_i$是一个仅仅只有第$i$维为1，其余维度均为0的一个列向量。要解决这个问题，很显然需要对函数做一些限制，如果我们函数是凸并且光滑的，那么很好办了。
 假如说第$i$维上达到了极小值，并且对应的自变量值为$x^*$,这就说明$\frac{\partial f}{\partial x_i}(x^*)=0$,那么每一个维度都满足这个要求，就说明$\nabla f(x)=0$,在函数凸的情况下这当然是一个极小值点，也就是说这个推断是没问题的。

但是如果说函数没有光滑性，但是有凸性，这可不可以呢？**答案是否定的**。

![image-20231213025146059](C:\Users\56201\AppData\Roaming\Typora\typora-user-images\image-20231213025146059.png)

可以看出当我们画出指定点对应的函数等高线时，让它往左右走一点，或者往上下走一点，都会导致函数值的增大（红线所画定的方向），而这就覆盖了所能走的所有方向，也就是说它确确实实满足“任意一个维度上都达到了极小值”这个条件。但是很明显它并不是一个极小值点，蓝色点对应的才是极小值点。

凸性如果没有，就变成了一个非凸优化问题，那自然不可能是我们“凸”优化讨论的范围。但是目前的讨论看来，难道光滑性是一个必要条件？当然也不是，事实上，对于一种特殊的情况
$$
\quad f(x)=g(x)+\sum_{i=1}^nh_i(x_i).
$$
 也即$f$可以拆分成两个函数$g,h$,其中$g$是凸函数并且光滑，$h$可以继续拆分为$h_1,\ldots,h_n$ ($n$是$x$ 的维数),并且每一个$h_i$都是凸函数(有的地方会说$h$是可分的)。在这种情况下，我们是可以导出这个结论的，但稍微需要一些技巧。

 注意到根据条件，我们有

$$
\begin{array}{c}f(y)-f(x)\geq\nabla g(x)^T(y-x)+\sum_{i=1}^n[h_i(y_i)-h_i(x_i)]\\=\sum_{i=1}^n[\nabla_ig(x)(y_i-x_i)+h_i(y_i)-h_i(x_i)]\geq0\end{array}
$$
 最后一个不等式是为什么？ 根据条件，在每一个维度上可以取到极小值，所以其实只需要根据次梯度的一阶最优性条件，就可以得到
$$
\quad0\in\nabla_ig(x)+\partial h_i(x_i)
$$
 也就是说$-\nabla_ig(x)$是$h_i(x_i)$的一个次梯度，那么根据次梯度定义，自然会有

$$
\quad h_i(y_i)\geq h_i(x_i)-\nabla_ig(x)^T(y_i-x_i)
$$
 移项一下就可以得到结论。
 上面这几个例子事实上就说明了，如果我们在每一个维度下都优化到最好，对于某些情况是可以认为优化问题已经解决的，这也就是坐标下降法可被应用的理论基础。而对于坐标下降法，一般来说采用的迭代方式是
$$
\quad x_i^{(k)}=\arg\min_{x_i}f(x_1^{(k)},\ldots,x_{i-1}^{(k)},x_i,x_{i+1}^{(k-1)},\ldots,x_n^{(k-1)})
$$

## DDPM

去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）在2020年被提出，向世界展示了扩散模型的强大能力，带动了扩散模型的火热。

### 前向过程

**前向过程是不断加噪的过程，加入的噪声随着时间步增加增多，**根据马尔可夫定理，加噪后的这一时刻与前一时刻的相关性最高也与要加的噪音有关（是与上一时刻的影响大还是要加的噪音影响大，当前向时刻越往后，噪音影响的权重越来越大了，因为刚开始加一点噪声就有效果，之后要加噪声越来越多 ）

给定初始图像 $x_0$ ,向其中逐步添加高斯噪声，加噪过程持续$T$ 次，产生一系列带噪图像，达到
破坏图像的目的。由 $x_{t-1}$ 加噪至 $x_t$ 的过程中，所加噪声的方差为 $\beta_t$ ,又称扩散率，是一个给定的，大于0 小于 1 的，随扩散步数增加而逐渐增大的值。定义扩散过程如下式：

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}z_t,z_t\sim\mathcal{N}(0,\boldsymbol{I})
$$
根据定义，加噪过程可以看作在上一步的基础上乘了一个系数，然后加上均值为 0, 方差为 $\beta_t$ 的高斯分布。所以加噪过程是确定的，并不是可学习的过程，将其写成概率分布的形式，则有：

$$
q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\boldsymbol{I})
$$
 此外，加噪过程是一个马尔科夫链过程，所以联合概率分布可以写成下式：

$$
q(x_1,x_2,\cdots,x_T|x_0)=q(x_1|x_0)q(x_2|x_1)\cdots q(x_T|x_{T-1})=\prod_{t=1}^Tq(x_t|x_{t-1})
$$
 定义 $\alpha_t=1-\beta_t$ ,即 $\alpha_t+\beta_t=1$ ,代入 $x_t$ 表达式并迭代推导，可得 $x_0$ 到 $x_t$ 的公式：

$$
\begin{aligned}
&x_{t}=\sqrt{1-\mu_{t}x_{t-1}}+\sqrt{\mu_{t}z_{t}}=\sqrt{\alpha_{t}x_{t-1}}+\sqrt{\mu_{t}z_{t}} \\
&=\sqrt{\alpha_t}\color{red}{(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}}z_{t-1})}+\sqrt{\beta_t}z_t \\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t\beta_{t-1}}z_{t-1}+\sqrt{\beta_t}z_t \\
&=\sqrt{\alpha_t\alpha_{t-1}}\color{red}{(\sqrt{\alpha_{t-2}}x_{t-3}+\sqrt{\beta_{t-2}}z_{t-2})}+\sqrt{\alpha_t\beta_{t-1}}z_{t-1}+\sqrt{\beta_t}z_t \\
&=\sqrt{\alpha_t\alpha_{t-1}\alpha_{t-2}}x_{t-3}+\sqrt{\alpha_t\alpha_{t-1}\beta_{t-2}}z_{t-2}+\sqrt{\alpha_t\beta_{t-1}}z_{t-1}+\sqrt{\beta_t}z_{t} \\
&=\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}x_0+\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_2\beta_1}z_1+\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_3\beta_2}z_2+\cdots  \\
&+\sqrt{\alpha_t\alpha_{t-1}\beta_{t-2}}z_{t-2}+\sqrt{\alpha_t\beta_{t-1}}z_{t-1}+\sqrt{\beta_t}z_t
\end{aligned}
$$

$$
\text{注意看}\alpha_t=1-\beta_t
$$

$β$不断增大,论文中是0.0001~0.002,所以之后$α$越来越小，由此,$x_t$随着前向时刻越往后，噪音影响的权重越来越大，z是服从高斯分布的噪音。

上式从第二项到最后一项都是**独立的高斯噪声**，它们的均值都为0，方差为各自系数的平方。根据**高斯分布的叠加公式**，它们的**和**满足均值为0，方差为各项方差之和的高斯分布。又有上式每一项系数的平方和（包括第一项）为1，证明如下：

$$
\begin{aligned}
&\alpha_t\alpha_{t-1}\cdots\alpha_1+\alpha_t\alpha_{t-1}\cdots\alpha_2\beta_1+\alpha_t\alpha_{t-1}\cdots\alpha_3\beta_2+\cdots+\alpha_t\beta_{t-1}+\beta_t \\
&=\alpha_t\alpha_{t-1}\cdots\alpha_3(\alpha_2+\beta_2)+\cdots+\alpha_t\alpha_{t-1}\beta_{t-2}+\alpha_t\beta_{t-1}+\beta_t \\
&=\alpha_t\alpha_{t-1}\cdots\alpha_3\times{\color{red}{1}}+\cdots+\alpha_t\alpha_{t-1}\beta_{t-2}+\alpha_t\beta_{t-1}+\beta_t \\
&=\cdots\cdots=\alpha_t+\beta_t=1
\end{aligned}
$$
 那么，将 $\alpha_t\alpha_{t-1}\cdots\alpha_1$ 记作 $\bar{\alpha}_t$ ,则正态噪声的方差之和为 $1-\bar{\alpha}_t$ , $x_t$ 可表示为：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\bar{z}_t,\quad\bar{z}_t\sim\mathcal{N}(0,\boldsymbol{I})
$$
 由该式可以看出，$x_t$ 实际上是原始图像 $x_{0}$ 和随机噪声 $\bar{z}_t$ 的线性组合，即只要给定初始值，以
 及每一步的扩散率，就可以得到任意时刻的 $x_t$ ,写成概率分布的形式：
$$
q(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\boldsymbol{I})
$$
 当加噪步数$T$ 足够大时，$\bar{\alpha}_t$ 趋向于 0, $1-\bar{\alpha}_t$ 趋向于 1, 所以 $x_T$ 趋向于**标准高斯分布**。



### 逆向过程



### Code

这里写一个简单案例演示加噪的代码

```python
import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image

betas = torch.linspace(0.02, 1e-4, 1000).double()
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_m1_alphas_bar = torch.sqrt(1 - alphas_bar)

img = Image.open('car.png')  # 读取图片
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()  # 转换为tensor
])
x_0 = trans(img)
img_list = [x_0]
noise = torch.randn_like(x_0)
for i in range(15):
    x_t = sqrt_alphas_bar[i] * x_0 + sqrt_m1_alphas_bar[i] * noise
    img_list.append(x_t)
all_img = torch.stack(img_list, dim=0)
all_img = make_grid(all_img)
save_image(all_img, 'car_noise.png')
```



## DDIM

```python
import torch as th
import torch.nn as nn
import math
import some_module 
import numpy as np
def timestep_embedding(timesteps,dim,max_period=10000):
    half = dim //2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0,end=half,dtype=th.float32) /half
    ).to(device=timesteps.device)

    args = timesteps[:,None].float()*freqs[None]
    embedding = th.cat([th.cos(args),th.sin(args)],dim=-1)
    if dim % 2:
        embedding = th.cat([embedding,th.zeros_like(embedding[:,:1],dim=-1)])

class TimestepBlock(nn.Module):
    def forward(self,x,emb):

class TimestepEmbedSquential(nn.Sequential,TimestepBlock):
    def forward(self,x,emb):
        for layer in self:
            if isinstance(layer,TimestepBlock):
                x = layer(x,emb)
            else:
                x = layer(x)
            return x 
        

class ResBlock(TimestepBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
                 ):
            super().__init__()
            self.channels=channels
            self.emb_channels = emb_channels
            self.dropout = dropout
            self.out_channels = out_channels or channels
            self.use_conv = use_conv
            self.dims=dims

            self.in_layers = nn.Sequential(
                normalization(channels),
                SiLU(),
                # conv_nd(padding = 1,3)
            )
    SiLU()
    linear(
        emb_channels,
        2*self.out_channels if use_scale_shift_norm else self.out_channels
    )

def forward(self,x,emb):
    # x [N x C x ...]
    return checkpoint(
        self._forward,(x,emb),self.parameters(),self.use_checkpoint
    )
def _forward(self,x,emb):
    h =self.in_layers(x)
    emb_out =self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[...,None]
    if self.use_scale_shift_norm:
        out_norm,out_rest = self.out_layers[0],self.out_layers[1:]
        scale,shift = th.chunk(emb_out,2,dim=1)
        h = out_norm(h) *(1+scale) +shift
        h = out_rest(h)
    else:
        h = h+emb_out
        h = self.out_layers(h)
    return self.skip_connection(x) +h
#qkv
class AttentionBlock(nn.Module):
    def __init__(self,channels,num_heads=1,use_checkpoint=False):
        super.__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv= conv_nd(1,channels,channels*3,1)
        self.attention = QKVAttention()
        self.proj_out
    def forward(self,x):
        return checkpoint(self._forward,(x,),self.parameters)
    def _forward(self,x):
        return (x+h).reshape(b,c,*spatial)
    
def get_named_beta_schedule(schedule_name,num_diffusion_timesteps):
    if schedule_name =='linear':
        scale =1000/num_diffusion_timesteps
        beta_start = scale *0.00001
        beta_end = scale *0.02
        return np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64
        )
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t : math.cos((t+0.08)/1.008*math.pi/2)**2

        )
    else :
        raise NotImplementedError(f"unknown beta")
    
def betas_for_alpha_bar(num_diffusion_timesteps,alpha_bar,max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i/num_diffusion_timesteps
        t2 = (i+1)/num_diffusion_timesteps
        betas.append(min(1-alpha_bar(t2)/alpha_bar(t1),max_beta))
    return np.array(betas)

#diffusion_utils_2.py
```

一步到多步的



```python
#q_sample x_0 x_t
```



## Appendix

