# Diffusion Model

当前主要有四大生成模型：生成对抗模型、变微分自动编码器、流模型以及扩散模型。**扩散模型（diffusion models）**是当前深度生成模型中新SOTA。扩散模型在图片生成任务中超越了原SOTA：GAN，并且在诸多应用领域都有出色的表现，如计算机视觉，NLP、波形信号处理、多模态建模、分子图建模、时间序列建模、对抗性净化等。此外，扩散模型与其他研究领域有着密切的联系，如稳健学习、表示学习、强化学习。本文主要是从其数学原理以及代码方面对扩散模型进行讲解。

## 先验知识

### 贝叶斯法则

先验概率 $P\left ( \theta\right ) : $根据以往的经验和分析得到的，假设$\theta$ 发生的概率；
似然函数 $P(x\mid\theta)$ : 在假设 $\theta$ 已发生的前提下，发生事件 $x$ 的概率；
后验概率 $P\left(\theta\mid x\right)$: 在事件 $x$ 已发生的前提下，假设 $\theta$ 成立的概率； 标准化常量 $P( x) : $在已知所有假设下，事件$x$ 发生的概率；
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
定义： 设X 和 Y 是离散随机变量，c 为常数，k 为正整数， 如果$E(|X-c|^k)$存在，则称$E(|X-c|^k)$为X关于点c的k阶矩。

$c=0$ 时，称为k阶原点矩；
$C=E(X)$时，称为k阶中心矩。
如果$E(|X-c1|^p\bullet|Y-c2|^q)$存在，则称其为X，Y 关于 c 点的 p+q 阶矩。
$c1=c2=0$ 时，称为 p+q 阶混合原点矩c1=E(X), c2=E(Y)时，称为 p+q 阶混合中心距。
如果 X,Y 是连续型的，则$\int_k(x-c)^kdx$ 称为X关于点 c 的 k阶原点矩， $\iint_{p+q}(x-x_0)^p\bullet(y-y_0)^qdxdy$ 称为 X,Y 关于点 c 的 p+q 阶混合中心矩。矩的本质是数学期望，而期望的计算公式应该是$E(x)=\int x\bullet f(x)dx$,其中 f(x)是 x 的概率密度，也就说上面的公式其实默认了所有随机变量出现的概率相等。

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

β不断增大,论文中是0.00010.002,所以之后α越来越小，由此,$x_t$随着前向时刻越往后，噪音影响的权重越来越大，z是服从高斯分布的噪音。

上式从第二项到最后一项都是**独立的高斯噪声**，它们的均值都为0，方差为各自系数的平方。根据**高斯分布的叠加公式**，它们的**和**满足均值为0，方差为各项方差之和的高斯分布。又有上式每一项系数的平方和（包括第一项）为1，证明如下：

$\alpha_t\alpha_{t-1}\cdots\alpha_1+\alpha_t\alpha_{t-1}\cdots\alpha_2\beta_1+\alpha_t\alpha_{t-1}\cdots\alpha_3\beta_2+\cdots+\alpha_t\beta_{t-1}+\beta_t$

$$
\begin{aligned}
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

## Appendix

### DDPM in Stable Diffusion

```python
"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        self.make_it_fit = make_it_fit
        if reset_ema: assert exists(ckpt_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None and not self.force_null_conditioning:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key in ['class_label', 'cls']:
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            if self.cond_stage_key in ["class_label", "cls"]:
                xc = self.cond_stage_model.get_unconditional_conditioning(batch_size, device=self.device)
                return self.get_learned_conditioning(xc)
            else:
                raise NotImplementedError("todo")
        if isinstance(c, list):  # in case the encoder gives us a list
            for i in range(len(c)):
                c[i] = repeat(c[i], '1 ... -> b ...', b=batch_size).to(self.device)
        else:
            c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        return c

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', "cls"]:
                try:
                    xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                    log['conditioning'] = xc
                except KeyError:
                    # probably no "human_label" in batch
                    pass
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            if self.model.conditioning_key == "crossattn-adm":
                uc = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            if hasattr(self, "scripted_diffusion_model"):
                # TorchScript changes names of the arguments
                # with argument cc defined as context=cc scripted model will produce
                # an error: RuntimeError: forward() is missing value for argument 'argument_3'.
                out = self.scripted_diffusion_model(x, t, cc)
            else:
                out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class LatentUpscaleDiffusion(LatentDiffusion):
    def __init__(self, *args, low_scale_config, low_scale_key="LR", noise_level_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        # assumes that neither the cond_stage nor the low_scale_model contain trainable params
        assert not self.cond_stage_trainable
        self.instantiate_low_stage(low_scale_config)
        self.low_scale_key = low_scale_key
        self.noise_level_key = noise_level_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        self.low_scale_model = model.eval()
        self.low_scale_model.train = disabled_train
        for param in self.low_scale_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, log_mode=False):
        if not log_mode:
            z, c = super().get_input(batch, k, force_c_encode=True, bs=bs)
        else:
            z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                                  force_c_encode=True, return_original_cond=True, bs=bs)
        x_low = batch[self.low_scale_key][:bs]
        x_low = rearrange(x_low, 'b h w c -> b c h w')
        x_low = x_low.to(memory_format=torch.contiguous_format).float()
        zx, noise_level = self.low_scale_model(x_low)
        if self.noise_level_key is not None:
            # get noise level from batch instead, e.g. when extracting a custom noise level for bsr
            raise NotImplementedError('TODO')

        all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}
        if log_mode:
            # TODO: maybe disable if too expensive
            x_low_rec = self.low_scale_model.decode(zx)
            return z, all_conds, x, xrec, xc, x_low, x_low_rec, noise_level
        return z, all_conds

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True,
                   unconditional_guidance_scale=1., unconditional_guidance_label=None, use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, x_low, x_low_rec, noise_level = self.get_input(batch, self.first_stage_key, bs=N,
                                                                          log_mode=True)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["x_lr"] = x_low
        log[f"x_lr_rec_@noise_levels{'-'.join(map(lambda x: str(x), list(noise_level.cpu().numpy())))}"] = x_low_rec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_tmp = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # TODO explore better "unconditional" choices for the other keys
            # maybe guide away from empty text label and highest noise level and maximally degraded zx?
            uc = dict()
            for k in c:
                if k == "c_crossattn":
                    assert isinstance(c[k], list) and len(c[k]) == 1
                    uc[k] = [uc_tmp]
                elif k == "c_adm":  # todo: only run with text-based guidance?
                    assert isinstance(c[k], torch.Tensor)
                    #uc[k] = torch.ones_like(c[k]) * self.low_scale_model.max_noise_level
                    uc[k] = c[k]
                elif isinstance(c[k], list):
                    uc[k] = [c[k][i] for i in range(len(c[k]))]
                else:
                    uc[k] = c[k]

            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        return log


class LatentFinetuneDiffusion(LatentDiffusion):
    """
         Basis for different finetunas, such as inpainting or depth2image
         To disable finetuning mode, set finetune_keys to None
    """

    def __init__(self,
                 concat_keys: tuple,
                 finetune_keys=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 keep_finetune_dims=4,
                 # if model was trained without concat mode before and we would like to keep these channels
                 c_concat_log_start=None,  # to log reconstruction of c_concat codes
                 c_concat_log_end=None,
                 *args, **kwargs
                 ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", list())
        super().__init__(*args, **kwargs)
        self.finetune_keys = finetune_keys
        self.concat_keys = concat_keys
        self.keep_dims = keep_finetune_dims
        self.c_concat_log_start = c_concat_log_start
        self.c_concat_log_end = c_concat_log_end
        if exists(self.finetune_keys): assert exists(ckpt_path), 'can only finetune from a given checkpoint'
        if exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

            # make it explicit, finetune by including extra input channels
            if exists(self.finetune_keys) and k in self.finetune_keys:
                new_entry = None
                for name, param in self.named_parameters():
                    if name in self.finetune_keys:
                        print(
                            f"modifying key '{name}' and keeping its original {self.keep_dims} (channels) dimensions only")
                        new_entry = torch.zeros_like(param)  # zero init
                assert exists(new_entry), 'did not find matching parameter to modify'
                new_entry[:, :self.keep_dims, ...] = sd[k]
                sd[k] = new_entry

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if not (self.c_concat_log_start is None and self.c_concat_log_end is None):
            log["c_concat_decoded"] = self.decode_first_stage(c_cat[:, self.c_concat_log_start:self.c_concat_log_end])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                         batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                 batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc_full,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log


class LatentInpaintDiffusion(LatentFinetuneDiffusion):
    """
    can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn.
    To disable finetuning mode, set finetune_keys to None
     """

    def __init__(self,
                 concat_keys=("mask", "masked_image"),
                 masked_image_key="masked_image",
                 *args, **kwargs
                 ):
        super().__init__(concat_keys, *args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for inpainting'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        c_cat = list()
        for ck in self.concat_keys:
            cc = rearrange(batch[ck], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            bchw = z.shape
            if ck != self.masked_image_key:
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super(LatentInpaintDiffusion, self).log_images(*args, **kwargs)
        log["masked_image"] = rearrange(args[0]["masked_image"],
                                        'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
        return log


class LatentDepth2ImageDiffusion(LatentFinetuneDiffusion):
    """
    condition on monocular depth estimation
    """

    def __init__(self, depth_stage_config, concat_keys=("midas_in",), *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.depth_model = instantiate_from_config(depth_stage_config)
        self.depth_stage_key = concat_keys[0]

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for depth2img'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        assert len(self.concat_keys) == 1
        c_cat = list()
        for ck in self.concat_keys:
            cc = batch[ck]
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            cc = self.depth_model(cc)
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )

            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min + 0.001) - 1.
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        depth = self.depth_model(args[0][self.depth_stage_key])
        depth_min, depth_max = torch.amin(depth, dim=[1, 2, 3], keepdim=True), \
                               torch.amax(depth, dim=[1, 2, 3], keepdim=True)
        log["depth"] = 2. * (depth - depth_min) / (depth_max - depth_min) - 1.
        return log


class LatentUpscaleFinetuneDiffusion(LatentFinetuneDiffusion):
    """
        condition on low-res image (and optionally on some spatial noise augmentation)
    """
    def __init__(self, concat_keys=("lr",), reshuffle_patch_size=None,
                 low_scale_config=None, low_scale_key=None, *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.reshuffle_patch_size = reshuffle_patch_size
        self.low_scale_model = None
        if low_scale_config is not None:
            print("Initializing a low-scale model")
            assert exists(low_scale_key)
            self.instantiate_low_stage(low_scale_config)
            self.low_scale_key = low_scale_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        self.low_scale_model = model.eval()
        self.low_scale_model.train = disabled_train
        for param in self.low_scale_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for upscaling-ft'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        assert len(self.concat_keys) == 1
        # optionally make spatial noise_level here
        c_cat = list()
        noise_level = None
        for ck in self.concat_keys:
            cc = batch[ck]
            cc = rearrange(cc, 'b h w c -> b c h w')
            if exists(self.reshuffle_patch_size):
                assert isinstance(self.reshuffle_patch_size, int)
                cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                               p1=self.reshuffle_patch_size, p2=self.reshuffle_patch_size)
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            if exists(self.low_scale_model) and ck == self.low_scale_key:
                cc, noise_level = self.low_scale_model(cc)
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        if exists(noise_level):
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c], "c_adm": noise_level}
        else:
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        log["lr"] = rearrange(args[0]["lr"], 'b h w c -> b c h w')
        return log


class ImageEmbeddingConditionedLatentDiffusion(LatentDiffusion):
    def __init__(self, embedder_config, embedding_key="jpg", embedding_dropout=0.5,
                 freeze_embedder=True, noise_aug_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_key = embedding_key
        self.embedding_dropout = embedding_dropout
        self._init_embedder(embedder_config, freeze_embedder)
        self._init_noise_aug(noise_aug_config)

    def _init_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def _init_noise_aug(self, config):
        if config is not None:
            # use the KARLO schedule for noise augmentation on CLIP image embeddings
            noise_augmentor = instantiate_from_config(config)
            assert isinstance(noise_augmentor, nn.Module)
            noise_augmentor = noise_augmentor.eval()
            noise_augmentor.train = disabled_train
            self.noise_augmentor = noise_augmentor
        else:
            self.noise_augmentor = None

    def get_input(self, batch, k, cond_key=None, bs=None, **kwargs):
        outputs = LatentDiffusion.get_input(self, batch, k, bs=bs, **kwargs)
        z, c = outputs[0], outputs[1]
        img = batch[self.embed_key][:bs]
        img = rearrange(img, 'b h w c -> b c h w')
        c_adm = self.embedder(img)
        if self.noise_augmentor is not None:
            c_adm, noise_level_emb = self.noise_augmentor(c_adm)
            # assume this gives embeddings of noise levels
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        if self.training:
            c_adm = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(c_adm.shape[0],
                                                                               device=c_adm.device)[:, None]) * c_adm
        all_conds = {"c_crossattn": [c], "c_adm": c_adm}
        noutputs = [z, all_conds]
        noutputs.extend(outputs[2:])
        return noutputs

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, **kwargs):
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True,
                                           return_original_cond=True)
        log["inputs"] = x
        log["reconstruction"] = xrec
        assert self.model.conditioning_key is not None
        assert self.cond_stage_key in ["caption", "txt"]
        xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
        log["conditioning"] = xc
        uc = self.get_unconditional_conditioning(N, kwargs.get('unconditional_guidance_label', ''))
        unconditional_guidance_scale = kwargs.get('unconditional_guidance_scale', 5.)

        uc_ = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
        ema_scope = self.ema_scope if kwargs.get('use_ema_scope', True) else nullcontext
        with ema_scope(f"Sampling"):
            samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=True,
                                             ddim_steps=kwargs.get('ddim_steps', 50), eta=kwargs.get('ddim_eta', 0.),
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_, )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samplescfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log

```

### DDIM in Stable Diffusion

```python
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
```



