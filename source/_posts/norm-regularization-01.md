---
title: 浅说范数规范化（一）—— L0 范数、L1 范数、L2 范数
date: 2017-09-07 14:43:10
categories: ML
tags:
      - Norm regularization
      - Convex optimization
description: 浅说机器学习问题中出现的范数规则化问题，本篇着重讲 L0 范数、L1 范数、L2 范数。
---

# 我们需要范数规则化的原因

监督机器学习用一句话总结就是：Minimize your error while regularizing your parameters。
其中“minimize error"是目的，一般采用最小化损失函数的做法来达到。
而“regularizing parameters”就是一种保障，它可以防止模型发生过拟合，让模型的参数规模尽量向着“简单”的方向进化。（根据奥卡姆剃刀原理 Occam's Razor，简单的模型虽然不尽准确，却也有更好的泛化能力。）

所以监督机器学习一般可以用一个公式来代表：
$$
w^{\ast} = \mathop{\arg\min}_{w} \sum_{i} L(y_{i}, f(x_{i};w)) + \lambda\Omega(w)
$$
对于第一项的损失函数，我们不在本文中做过多讨论，大概说一下：

- Square loss —— 最小二乘
- Hinge loss —— SVM
- Exp loss —— Boosting
- Log loss —— Logistic regression

等等。
总之，不同的 loss 函数有不同的拟合特性。

下面我们重点说一说规则项 $ \Omega(w) $ 。

# 关于规则项

大多时候规则项都是用来限制参数的复杂程度的。所以一般用参数 $w$ 的某些性质来约束，常见的就是几种范数：L0 范数、L1 范数、L2 范数、迹范数、Frobenius 范数和核范数。

本篇，着重说 L0 范数、L1 范数 和 L2 范数。

（稍微声明一下：一些大家都知道的知识我就不在此赘述了。）

## L0 范数和 L1 范数

这两个范数都是保证参数稀疏性的所以放在一起说。

L0 范数是向量中非 0 元素的个数，这个想必大家都知道了。
用它来限制稀疏性本身很好，可惜它不是凸的，求解它将成为一个 NP-hard 问题。
所以我们用能够完全包络 L0 范数的一个凸包—— L1 范数来近似的代替它[^1]。

特征稀疏的好处有以下两点：

- 便于特征选择。现实世界中，问题的特征的数量往往是很大的，而起决定性作用的往往是一小部分，所以我们在建立简单模型的时候，会先考虑舍弃权重快速收敛于 0 的特征。
- 更具可解释性。例如对于癌症预测问题，可能有上千个特征，如果主要特征只有 10 个，就可以解释为癌症的发生几乎更和这 10 个特征息息相关，其它的暂不考虑影响也不大。

## L2 范数

考虑到 L1 范数在顶点处是不可微的，人们又引入了 L2 范数。

关于 L2 范数如何提升模型泛化能力就不赘述了，在此讲一下它对优化计算所作出的贡献。

优化问题有两个难题，一：局部最小值，二：病态（ill-condition）问题。
第一个问题很容易理解，那么第二个问题提到的病态又是什么呢？
简单来说，假设有一个方程 $AX = b$ ，如果 $A$ 和 $b$ 稍微发生改变就会引起 $X$ 的巨大变化的话，我们就称这个方程组系统是病态（ill-condition）的。反之就是良态（well-condition）的。

举个例子，在人脸识别中，如果一个人粘了个假睫毛就不认识了，那就说明她的脸是“病态的”（笑）。

定义：方阵 $A$ 是非奇异的，那么 $A$ 的条件数（condition number）定义为
$$
\kappa(A) = \lVert A \rVert \lVert A^{-1} \rVert
$$
经过简单的证明（请参考本系列的附录 A），我们可以得到以下的结论
$$
\frac{\lVert \Delta x \rVert}{\lVert x \rVert} \le \kappa(A) \cdot \frac{\lVert \Delta b \rVert}{\lVert b \rVert} \\
\frac{\lVert \Delta x \rVert}{\lVert x + \Delta x \rVert} \le \kappa(A) \cdot \frac{\lVert \Delta A \rVert}{\lVert A \rVert}
$$
因此可以认为，condition number 描述的是一个矩阵（或它形成的线性系统）的稳定性（或敏感度）的度量。
如果一个矩阵的 condition number 在 1 附近，那么它是 well-condition 的；反之，它是 ill-condition 的。

考虑线性回归的解析解
$$
w^{\ast} = (X^{\mathsf{T}}X)^{-1}X^{\mathsf{T}}y
$$
如果样本的数目比样本的维度还要小的时候，矩阵 $ X^{\mathsf{T}}X $ 将会不是满秩的，也就不可逆。
但如果加上 L2 范数规则项，解就变成
$$
w^{\ast} = (X^{\mathsf{T}}X + \lambda I)^{-1}X^{\mathsf{T}}y
$$
此时，就可以直接求逆了。

另外，通常我们并不适用解析解求解，而是使用牛顿迭代法求解。
此时，加入规则项的两一个好处就是它可以将目标函数变为 $ \lambda $ 强凸（$\lambda$ -strongly convex）的。

定义：当 $f$ 满足
$$
f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\lambda}{2} \lVert y-x \rVert ^{2}
$$
时，我们称 $f$ 为 $ \lambda $ 强凸（$\lambda$ -strongly convex）的。

对比一阶泰勒展开
$$
f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + o (\lVert y-x \rVert)
$$
我们可以发现，这种强凸性质不仅仅要求函数在某条切线的上方，而且还要求函数整体在某个二次函数图像上方，即已经具备一定的“向上弯曲度”。
所以这种函数的下降速度非常快，而且很稳定。

[^1]: [Ramirez, C. & V. Kreinovich & M. Argaez. Why l1 is a good approximation to l0: A geometric explanation [J]. *Engineering*, 2013, 7 (3): 203-207](https://www.researchgate.net/publication/290729378_Why_l1_is_a_good_approximation_to_l0_A_geometric_explanation)

