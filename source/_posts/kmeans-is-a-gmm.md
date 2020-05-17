---
title: K-means 是 GMM 的一种特例
date: 2017-10-23 23:23:56
categories: ML
tags:
     - K-means
     - GMM
     - EM Algorithm
description: 从一个 GMM 的特定生成模型讨论了 K-means 与 EM 算法的关系。
---

# 对一个特定 GMM 模型的讨论

考虑到在非监督学习 K-means 中，那些不存在的所谓的“标签”也可以被当成“不可见因素”的想法。
本文通过一个特定的 GMM（高斯混合模型，Gaussian Mixture Model）模型讨论了 K-means 算法在该特定模型下可以看做一个 EM 算法。因此可以认为 K-means 算法与 EM 算法在某种先验条件下存在交集。

首先给出以下两个问题的描述：

**聚类问题**：给定数据点 $x_{1}, x_{2}, \cdots, x_{N} \in \mathbb{R}^{m}$ ，给定分类数目 $K$ ，求出 $K$ 个类中心 $\mu_{1}, \mu_{2}, \cdots, \mu_{K}$ ，使得所有点到距离该点最近的类中心的距离的平方和 $\sum\limits_{i=1}^{N} \min\limits_{1 \le k \le K} \lVert x_{i} - \mu_{k} \rVert_{2}^{2}$ 最小。

**含隐变量的最大似然问题**：给定数据点 $x_{1}, x_{2}, \cdots, x_{N} \in \mathbb{R}^{m}$ ，给定分类数目 $K$ ，考虑如下生成模型，
$$
p(x, z \mid \mu_{1}, \mu_{2}, \cdots, \mu_{K}) \propto 
\left \{
\begin{aligned}
&\exp \left ( -\lVert x-\mu_{z} \rVert_{2}^{2} \right ) & \qquad \lVert x-\mu_{z} \rVert_{2} = \min_{1 \le k \le K} \lVert x-\mu_{k} \rVert_{2} \\
&0 & \qquad \lVert x-\mu_{z} \rVert_{2} > \min_{1 \le k \le K} \lVert x-\mu_{k} \rVert_{2}
\end{aligned}
\right.
$$
模型中 $z \in \{1, 2, \cdots, K\}$ 为隐变量。

这个模型的意义是：先验的假设真的存在那么 $K$ 个中心，只不过看不到而已。而所有的数据确实也是根据这 $K$ 个中心生成的，且这个生成概率是以其最近的“隐形中心”为中心呈高斯分布，并且不认为（零概率）该数据点是由离其较远的“隐形中心”生成的。（因此这个模型的条件还是很苛刻的。）

结论：用 EM 算法解这个含隐变量的最大似然问题等价于用 K-means 算法解聚类问题。

**E 步骤**：

1. 计算
   $$
   p(z_{i} \mid x_{i}, \{\mu_{k}\}^{(t)}) \propto 
   \left \{
   \begin{aligned}
   &1 & \qquad \lVert x_{i}-\mu_{z_{i}}^{(t)} \rVert_{2} = \min_{1 \le k \le K} \lVert x_{i}-\mu_{k}^{(t)} \rVert_{2} \\
   &0 & \qquad \lVert x_{i}-\mu_{z_{i}}^{(t)} \rVert_{2} > \min_{1 \le k \le K} \lVert x_{i}-\mu_{k}^{(t)} \rVert_{2}
   \end{aligned}
   \right.
   $$
   以及
   $$
   p(Z \mid X, \{ \mu_{k} \}^{(t)}) = \prod_{i=1}^{N} p(z_{i} \mid x_{i}, \{ \mu_{k} \}^{(t)})
   $$
   这里使用正比于符号是考虑离数据点最近的中心可能不止一个。

   下面为了简单只讨论只有一个中心，且中心编号为 $y_{i}$ 的情况。

2. 计算目标函数
   $$
   \begin{aligned}
   Q(\{ \mu_{k} \} \mid \{ \mu_{k} \}^{(t)}) & = E_{Z \mid X, \{ \mu_{k} \}^{(t)}} [\log p(X, Z \mid \{ \mu_{k} \})] \\
   & = \sum_{i=1}^{N} E_{z_{i} \mid x_{i}, \{ \mu_{k} \}^{(t)}} [\log p(x_{i}, z_{i} \mid \{ \mu_{k} \})] \\
   & = \sum_{i=1}^{N} \log p(x_{i}, z_{i}=y_{i} \mid \{ \mu_{k} \}) \\
   & = \mathrm{const} - \sum_{i=1}^{N} \lVert x_{i} - \mu_{y_{i}} \rVert_{2}^{2}
   \end{aligned}
   $$
   所以最大化目标函数就等价于最小化 $Q'(\{ \mu_{k} \} \mid \{ \mu_{k} \}^{(t)}) = \sum\limits_{i=1}^{N} \lVert x_{i} - \mu_{y_{i}} \rVert_{2}^{2}$ 。

**M 步骤**：

找寻 $\{ \mu_{k} \}$ 使得 $Q'$ 最小。

将指标集 $\{1, 2, \cdots, K \}$ 分割为 $K$ 个，$I_{k} = \{ i \mid y_{i}=k \}$ ，则
$$
Q'(\{ \mu_{k} \} \mid \{ \mu_{k} \}^{(t)}) = \sum\limits_{i=1}^{K} \sum_{i \in I_{k}} \lVert x_{i} - \mu_{k} \rVert_{2}^{2}
$$
由于求和的每一项都是非负的，所以每一个内层求和 $\sum\limits_{i \in I_{k}} \lVert x_{i} - \mu_{k} \rVert_{2}^{2}$ 都达到最小时，总和最小。

这和 K-means 最小化二范数平方和是一样的。

# 进阶讨论

以上的讨论是为了简单起见提出了一个简化的 GMM 模型，这个模型里没有提到高斯分布的协方差矩阵的问题。实际上，K-means 在简化 GMM 时，应该是限制协方差矩阵 $\Sigma = \sigma I$ ，再让 $\sigma \to 0$ 。更多的讨论在 [这篇论文](http://icml.cc/2012/papers/291.pdf) 的 2.1 节可以找到。