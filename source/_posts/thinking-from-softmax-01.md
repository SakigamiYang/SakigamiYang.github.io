---
title: 由 Softmax 想到的（一）——广义线性模型
date: 2017-08-10 21:12:46
categories: ML
tags: 
     - Generalized Linear Model
     - Exponential family distribution
     - Logistic
     - Softmax
description: 由好友 史博士 关于 Softmax 的问题而想到的一些脑洞。本文讲解了 Logistic 模型和 Softmax 模型的来历。和广义线性模型的相关知识。
---

昨日被好友史博士问了个问题：为什么是 Softmax？Softmax到底是怎么来的？
于是脑洞大开，写几篇博客把广义线性模型、梯度、次梯度、激活函数这几块有一些关联的概念大概串一下。

# 广义线性模型 及 指数分布族

本质上，线性模型、Logistic 模型和 Softmax 模型都是由一个东西推理出来的。
就是广义线性模型（Generalized Linear Model）。
这些分布之所以长成这个样子，是因为我们对标签的分布进行了假设。

- 标签是连续的（正态分布，由中心极限定律所得）——线性模型。
- 标签是二分类（多次观察可看做是二项分布）—— Logistic 模型。
- 标签是多分类（多次观察可看做是多项分布）—— Softmax 模型。

另外，只要 $y$ 的分布是指数分布族（exponential family distribution）的，都可以用一种通用的方法推导出 $h(x)$ 。



广义线性模型的定义：

1. 给定特征属性 $x$ 和参数 $\theta$ 后，$y$ 的条件概率 $P(y \mid x; \, \theta)$ 服从指数分布族。
2. 预测 $T(y)$ 的期望，即计算 $E[T(y) \mid x]$ 。
3. $\eta$ 与 $x$ 之间是线性的，即 $\eta=\theta^{\mathsf{T}}x$ 。

指数分布的形式如下：
$$
P(y; \, \eta) = b(y) \ast \exp(\eta^{\mathsf{T}}T(y)-a(\eta))
$$

# 二项分布—— Logistic 模型的意义

伯努利分布（Bernoulli distribution），也叫 0-1 分布，如果实验多次就变成了二项分布（binomial distribution）。
所以二项分布也叫 $n$ 重伯努利分布，或者 $n$ 重伯努利实验。
概率密度函数为：$P(y; \, \phi) = \phi^{y}(1-\phi)^{1-y}$ 。
把它变为指数分布族的形式：$P(y; \, \phi) = \exp(y\ln \frac{\phi}{1-\phi}+\ln(1-\phi))$ 。
对比指数分布族，有：$\eta=\ln \frac{\phi}{1-\phi} \Rightarrow \phi=\frac{1}{1+e^{-\eta}}$ 。
根据广义线性模型的第三点的线性关系。可得：
$$
\phi = \frac{1}{1+e^{-\theta^{\mathsf{T}}x}}
$$
所以我们用 Logistic 函数来估计伯努利分布的参数。
直观的想就是我们估计一个东西的表现更像 0 还是更像 1，而不是准确的等于哪一个。

# 多项分布—— Softmax 模型的意义

多项分布（multinomial distribution）也是 $n$ 重实验，只不过每次不是伯努利分布，而是有 $k$ 个选择的“伯努利分布”。跟推导 Logistic 模型的手法相似，将多项分布化为指数分布族，在考虑线性关系，可得：
$$
P(y^{(i)}=j \mid x^{(i)}; \, \theta)=\frac{e^{\theta_{j}^{\mathsf{T}}x^{(i)}}}{\sum_k e^{\theta_{k}^{\mathsf{T}}x^{(i)}}}
$$
注意在推导中有一个小技巧，就是 Softmax 模型总是有一个“冗余”的参数，道理很简单，如果我们有 $k$ 个分类，并且一个东西的表现不太像某 $k-1$ 类的话，那必然猜测它属于第 $k$ 类。

另外，当 $k=2$ 时，Softmax 模型退化为 Logistic 模型。