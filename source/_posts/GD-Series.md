---
title: 梯度下降法家族
date: 2017-12-23 19:51:16
categories: ML
tags:
     - GD
     - Momentum
description: 根据一个通式说明梯度下降法全家族。
---

# 用一个通式说明梯度下降

梯度下降优化法经历了 SGD→SGDM→NAG→AdaGrad→AdaDelta→Adam→Nadam 这样的发展历程。之所以会不断地提出更加优化的方法，究其原因，是引入了动量（Momentum）这个概念。最初，人们引入一阶动量来给梯度下降法加入惯性（即，越陡的坡可以允许跑得更快些）。后来，在引入二阶动量之后，才真正意味着“自适应学习率”优化算法时代的到来。

我们用一个通式来描述梯度下降法，掌握了这个通式，你就也可以自己设计自己的优化算法了。

首先定义：

- 待优化参数：$w$
- 目标函数：$f(w)$
- 初始学习率：$\alpha$

然后进行迭代优化。在每个 epoch $t$ ：

> 1. 计算目标函数关于当前参数的梯度：$g_{t} = \nabla f(w_{t})$
> 2. 根据历史梯度计算一阶动量和二阶动量：$m_{t} = \phi(g_{1}, g_{2}, \cdots , g_{t}); \, V_{t} = \psi(g_{1}, g_{2}, \cdots , g_{t})$
> 3. 计算当前时刻的下降梯度：$\eta_{t} = \alpha \cdot m_{t} \Big/ \sqrt{V_{t}}$
> 4. 根据下降梯度进行更新：$w_{t+1} = w_{t} - \eta_{t}$

## SGD

SGD 没有动量的概念，即：$m_{t} = g_{t}; \, V_{t} = I^{2}$

此时，$\eta_{t} = \alpha \cdot g_{t}$

SGD 最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。

## SGD with Momentum

为了抑制 SGD 的震荡，SGDM 引入了惯性，即一阶动量。如果发现是陡坡，就用惯性跑得快一些。此时：$m_{t} = \beta_{1} \cdot m_{t-1} + (1 - \beta_{1}) \cdot g_{t}$

我们把这个式子递归一下就可以看出，一阶动量其实是各个时刻梯度方向的指数移动平均值，约等于最近 $1 / (1 - \beta_{1})$ 个时刻的梯度向量和的平均值。

$\beta_{1}$ 的经验值为 0.9 ，也就是每次下降时更偏向于此前累积的下降方向，并稍微偏向于当前下降方向。（就像拐弯时刹不住车，或者说得好听点——漂移）。

## SGD with Nesterov Acceleration

SGD 还有一个问题是会被困在一个局部最优点里。就像被一个小盆地周围的矮山挡住了视野，看不到更远的更深的沟壑。

Nesterov 提出了一个方法是既然我们有了动量，那么我们可以在步骤 1 中先不考虑当前的梯度。每次决定下降方向的时候先按照一阶动量的方向走一步试试，然后在考虑这个新地方的梯度方向。此时的梯度就变成了：$g_{t} = \nabla f(w_{t} - \alpha \cdot m_{t-1})$ 。

我们用这个梯度带入 SGDM 中计算 $m_{t}$ 的式子里去，然后再计算当前时刻应有的梯度并更新这一次的参数。

## AdaGrad

终于引入了二阶动量。

想法是这样：神经网络中有大量的参数，对于经常更新的参数，我们已经积累了大量关于它们的知识，不希望它们被单个样本影响太大，希望学习速率慢一些；而对于不经常更新的参数，我们对于它们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，学习速率大一些。

那么怎么样去度量历史更新频率呢？采用二阶动量——该维度上，所有梯度值的平方和：$V_{t} = \sum_{\tau = 1}^{t} g_{\tau}^{2}$ 。

回顾步骤 3 中的下降梯度：$\eta_{t} = \alpha \cdot m_{t} \Big/ \sqrt{V_{t}}$ 。我们发现引入二阶动量的意义上给了学习率一个缩放比例，从而达到了自适应学习率的效果（Ada = Adaptive）。（一般为了防止分母为 0 ，会对二阶动量做一个平滑。）

这个方法有一个问题是，因为二阶动量是单调递增的，所以学习率会很快减至 0 ，这样可能会使训练过程提前结束。

## AdaDelta / RMSProp

由于 AdaGrad 的学习率单调递减太快，我们考虑改变二阶动量的计算策略：不累计全部梯度，只关注过去某一窗口内的梯度。这个就是名字里 Delta 的来历。

修改的思路很直接，前面我们说过，指数移动平均值大约是过去一段时间的平均值，因此我们用这个方法来计算二阶累积动量：$V_{t} = \beta_{2} \cdot V_{t-1} + (1 - \beta_{2}) \cdot g_{t}^{2}$ 。

## Adam / Nadam

讲到这个 Adam 和 Nadam 就自然而然的出现了，既然有了一阶动量和二阶动量，干嘛不都用起来呢？所以：

Adam = Adaptive + Momentum。

Nadam = Nesterov + Adam

※上文中提到的 $\beta_{1}$ 和 $\beta_{2}$ 就是在一些深度学习框架里提到一阶动量和二阶动量时要传递的两个系数。

# 不忘初心，为什么还要使用 SGD？

既然 Adam 和 Nadam 这么厉害，为什么学术界在发论文时还有很多大牛只使用 SGD ？

理由很简单：应用是应用，学术是学术。

SGD 虽然不那么“聪明”，但是它“单纯”。所有的控制都在你自己手中，用它才不会影响你对其它因素的研究。

而其它的算法实在太“快”了，所以它们有时不收敛，有时不能达到最优解。

# 最后的一句话

算法没有好坏，最适合数据的才是最好的，永远记住：No free lunch theorem。