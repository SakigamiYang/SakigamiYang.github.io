---
title: 神经网络中均方误差与交叉熵作为损失函数的共同点
date: 2017-12-03 12:39:14
categories: ML
tags: 
     - Deep learning
     - Loss function
description: 从数学角度推导了均方误差和交叉熵误差的敏感度的等价性。
---

# 先说结论

在神经网络中有两种可以选择的损失函数，即均方误差和交叉熵误差。这两种损失函数在形式上很不一样，而且分别适用与不同的地方（回归或分类）。但是在适用梯度下降法学习最优参数时，它们是一致的。

考虑神经网络的 BP 算法，其最核心的一步是计算敏感度 $\delta$ ，采用不同损失函数和激活函数的神经网络在 BP 算法上的差异也主要存在于敏感度上。而下文中，我们就分别讨论了上述两种损失函数在不同场景下的情况，得到了它们的敏感度是相等的结论。

用到的符号：

- $net$ 或 $net_{i}$ ，净输出值，即 $net = w^{\mathsf{T}}x$ 。
- $a$ 或 $a_{i}$ ，神经元的激活函数输出值，即 $a=f(net)$ 。

# 均方误差——线性回归

线性回归使用均方误差作为损失函数。其激活函数为 identical ，即$a=net=w^{\mathsf{T}}x$ 。
其损失：
$$
J(w) = \frac{1}{2} (a-y)^{2}=\frac{1}{2} (net-y)^{2}
$$
输出神经元的敏感度：
$$
\delta = \frac{\partial J}{\partial net} = net-y = a-y
$$

# 交叉熵——逻辑回归

逻辑回归使用最大似然估计方法估计参数。

## 二分类逻辑回归

其激活函数为 sigmoid 函数 ，即$a=\sigma(net)=\sigma(w^{\mathsf{T}}x)=\frac{1}{1+\exp(-w^{\mathsf{T}}x)}$ 。
其损失：
$$
J(w) = -L(w) = -y \ln a - (1-y) \ln (1-a)
$$
其中：
$$
L(w) = y \ln a + (1-y) \ln (1-a)
$$
是所谓的 log-likelihood。

输出神经元的敏感度：
$$
\delta = \frac{\partial J}{\partial net} = \frac{\partial J}{\partial a} \frac{\partial a}{\partial net} = \frac{a-y}{(1-a)a}(1-a)a = a-y
$$

## 多分类逻辑回归

其激活函数为 softmax 函数，即$a=\mathrm{softmax}(net)=\frac{\exp(net_{i})}{\sum_{j=1}^{N}\exp(net_{j})}$ 。

其损失：
$$
J(w) = -\sum_{j=1}^{N} y_j \ln a_j = \sum_{j=1}^{N} y_j (\ln \sum_{k=1}^{N} e^{net_{k}}-net_{j})
$$
第 $i$ 个输出神经元的敏感度：
$$
\delta_{i} = \frac{\partial J}{\partial net_{i}} = \sum_{j=1}^{N} y_j (\frac{\sum_{k=1}^{N} e^{net_{k}}\frac{\partial net_{k}}{\partial net_{i}}}{\sum_{k=1}^{N} e^{net_{k}}} - \frac{\partial net_{j}}{\partial net_{i}}) = \sum_{j=1}^{N} y_j  \frac{e^{net_{i}}}{\sum_{k=1}^{N} e^{net_{k}}} - \sum_{j=1}^{N} y_j  \frac{\partial net_{j}}{\partial net_{i}} = a_{i}-y_{i}
$$
