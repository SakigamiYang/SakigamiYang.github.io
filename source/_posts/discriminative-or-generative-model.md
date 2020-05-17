---
title: 论判别模型和生成模型
date: 2018-02-04 22:17:38
categories: ML
tags:
     - Discriminative model
     - Generative model
description: 讨论了判别模型和生成模型的定义、理论推导及优缺点。
---

# 一个粗略的解释

**判别模型（Discriminative model）**：能够直接判断出输入的数据会输出什么东西的模型。比如对于一个输入，直接告诉你这个输入是 $+1$ 类，还是 $-1$ 类。

常见的判别模型有：线性回归（Linear Regression）、逻辑斯谛回归（Logistic Regression）、支持向量机（SVM）、传统神经网络（Traditional Neural Networks)、线性判别分析（Linear Discriminative Analysis）、条件随机场（Conditional Random Field）等。

**生成模型（Generative model）**：描述猜测目标的可能的分布的模型。比如给一个输入，告诉你输出为各个类别的概率。

常见的生成模型有：朴素贝叶斯（Naive Bayes）、隐马尔可夫模型（Hidden Markov Model）、贝叶斯网络（Bayes Networks）、隐含狄利克雷分布（Latent Dirichlet Allocation）。

# 理论解释

定义：

- 训练数据 $(C, X)$ ，$C=\{c_1, c_2, \cdots, c_n\}$ 是n个训练样本的标签，$X=\{x_1, x_2, \cdots, x_n\}$ 是n个训练样本的特征。
- 单个测试数据 $(\tilde{c}, \tilde{x})$ ，$\tilde{c}$ 是测试数据的标签，$\tilde{x}$ 是测试数据的特征。

## 判别模型

训练完毕后，输入测试数据，直接给出 $P(\tilde{c} | \tilde{x}) = P(\tilde{c} | \tilde{x}, C, X)$ 。我们认为这个条件分布是由模型的参数 $\theta$ 决定的，即 $P(\tilde{c} | \tilde{x},\theta)$ 。对于求解结果，我们有：
$$
\begin{aligned}
&P(\tilde{c} | \tilde{x}) \\
=&P(\tilde{c} | \tilde{x}, C, X) \\
=&\int P(\tilde{c}, \theta | \tilde{x}, C, X) \, d \theta \\
=&\int P(\tilde{c} | \tilde{x}, \theta) \cdot P(\theta | C, X) \, d \theta \\
=&\int P(\tilde{c} | \tilde{x}, \theta) \cdot \frac{P(C | X, \theta) \cdot P(\theta)}{\int P(C | X, \theta) \cdot P(\theta) \, d \theta} \, d \theta
\end{aligned}
$$
实际上，求这个式子里的两个积分是相当复杂的，而且几乎不可能拿到全部的信息。

所以我们采用 variational inference 的方法来解决这个问题。如果训练样本足够多的话，可以使用 $\theta$ 的最大后验分布 $\theta_{MAP}$ 来对 $\theta$ 进行点估计（point estimate）。此时有， $P(\tilde{c} | \tilde{x}) = P(\tilde{c} | \tilde{x}, C, X) = P(\tilde{c} | \tilde{x},\theta_{MAP})$ 。

至于求最大后验分布，考虑到 $P(C|X) = \int P(C | X, \theta) \cdot P(\theta) \, d \theta$ 是一个常数，我们只要求 $P(C | X, \theta) \cdot P(\theta)$ 的最大值就可以，于是问题转变为求最大似然函数。

事实上，在假设噪声为高斯分布的前提下，最小误差平方和优化问题等价于求最大似然函数。（有兴趣的同学可以自己证明一下。）

总结一下，判别模型求解的思路是：条件分布→模型参数后验概率最大→似然函数/参数先验最大→最大似然。

## 生成模型

直接求出联合分布 $P(\tilde{x}, \tilde{c})$ 。

以朴素贝叶斯为例：$P(\tilde{x}, \tilde{c}) = P(\tilde{x} | \tilde{c}) \cdot P(\tilde{c})$ 。

总结一下，生成模型求解的思路是：联合分布→求解类别先验概率和类别条件概率。

# 两种模型的优缺点

## 判别模型

优点：

1. 节省计算资源，需要的样本数量少于生成模型。
2. 准确率较生成模型高。
3. 由于直接预测结果的概率，而不需要求解每一个类别的条件概率，所以允许对数据进行抽象（比如降维、构造等）。

缺点：

生成模型的优点它都没有。

## 生成模型

优点：

1. 因为结果给出的是联合分布，不仅能计算条件分布，还可以给出其它信息（比如边缘分布）。
2. 收敛速度比较快，当样本数量较多时，可以更快地收敛于真实模型。
3. 能够应付隐变量存在的情况，如高斯混合模型（Gaussian Mixture Model）。

缺点：

1. 需要更多的样本和计算，尤其是为了更准确估计类别条件分布，需要增加样本的数目。