---
title: 浅谈 PCA 和 LFM 的异同
date: 2017-08-08 20:25:59
categories: ML
tags: 
      - PCA
      - LFM
---

在机器学习中有两种算法非常相似，却又很不同。
它们就是“PCA（Principal Component Analysis）”和“LFM（Latent Factor Model）”。
许多人都知道，PCA 常用于数据的降维，而接触到 LFM 最多的地方就是大家都知道的推荐系统。
事实上，这两种方法都涉及到矩阵的分解，也都是为了得到数据中隐含的信息。
（说到矩阵分解， PCA 是特征值分解；而 LFM 经常是被分解为两个矩阵的乘积，如 user-item 矩阵分解。）

下面详细讲一下这两个算法的不同点。

<!-- more -->

1. PCA

   PCA的目的是尽可能简化数据，即当数据维度 $p​$ 很大时，我们希望能够以一个更小维度的 $q​$ 维数据来代替它，而又不损失原始数据的随机程度，或者说是想要保持原始数据的分散程度。
   即，希望找到能够代替原始数据 $\lbrace X_{1}, X_{2}, \cdots, X_{p} \rbrace$ 的一组数据 $\lbrace Y_{1}, Y_{2}, \cdots, Y_{q} \rbrace$ ，使得
   $$
   Y_{i} = \sum_{j=1}^{p} \alpha_{j}^{(i)} X_{j}
   $$
   由上式的线性关系可知，PCA可以看做是相关性（correlation） 的推广。我们从视觉上给随机变量所展示出来的数据一个分散程度的定义——“波动”。那么 correlation 就可以说，某两个随机变量的涨落相似，如果它们的相关性系数很高。

   同样，如果将某个 $Y_{i}$ 分解为 $X_{j}$ 的线性组合时，某些 $\alpha_{j}^{(i)}$ 很大（而其他的 $\alpha_{j}^{(i)}$ 接近 $0$ 甚至等于 $0$ ），那么我们就可以认为 $Y_{i}$ 与这些 $X_{j}$ 共沉浮，即相关性很高。当我们尽可能的排除掉那些 $\alpha_{j}^{(i)} $ 接近 $0$ 的维度时，对于剩下的维度，我们就可以说：原始的 $p$ 维数据的“波动”，大部分都反映在这个 $q$ 维的子空间里。

   因此，PCA 所给出的 $q$ 个特征向量（eigenvector）就是描述这个“波动”的 $q$ 个主方向，而 $q$ 个特征值（eigenvalue）就是描述这个“波动”在这 $q$ 个主方向上波动的幅度。

   所以，PCA 才被用来降维，也就是我们用它来关注数据“最乱”的几个方向，其它的方向变化不大，不重要。

2. LFM

   目标：解释 $\lbrace X_{1}, X_{2}, \cdots, X_{p} \rbrace$ 之间的相关性，即希望找到一些 $Z_{k}$，有
   $$
   \forall X_{i}, \text{ 能找到一系列 } \beta_{k}^{(i)}, \text{ 使得 } X_{i} = \sum_{k=1}^{r} \beta_{k}^{(i)} Z_{k}
   $$
   一般，假设 $\lbrace Z_{1}, Z_{2}, \cdots, Z_{r} \rbrace$ 是零均值、单位方差且不相关的。
   （注意这里，所有的 $\beta^{\mathsf{T}}$ 就有点像 user 矩阵，所有的 $Z$ 就有点像 item 矩阵。所以 user-item 矩阵分解是LFM的一种应用。）

   同时，对于不同的 $X$ ，他们的那些 $\beta$ 越相似，这些 $X$ 就越相似。
   这也是我们能用它来判定有哪些用户相似，或者有哪些商品相似的原因。

   ​

**结论**
PCA：归纳。找出分散的根源。
LFM：分解。将随机变量分解为同一组变量的线性组合，来试图解释它们相关性的来源。
