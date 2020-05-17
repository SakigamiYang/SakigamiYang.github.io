---
title: 浅说范数规范化（二）—— 核范数
date: 2017-09-09 09:42:10
categories: ML
tags:
      - Norm regularization
      - Convex optimization
      - Matrix theory
description: 浅说机器学习问题中出现的范数规则化问题，本篇着重讲核范数。
---

# 矩阵补全（matrix completion）问题

这个问题最早火起来是因为 Netflix 公司悬赏 100 万美金的公开挑战，奖励给能够改进公司现行矩阵补全算法 10% 以上最优胜的队伍。最后的结果是 2009 年 9 月 BellKors Pragmatic 拿走了奖金。

什么是矩阵补全问题呢？用 Netflix 的数据集来作说明。
简单的来说就是一个电影评分系统要根据非常稀疏的现有数据集（一个用户可能只 rate 了几十部电影）来推断整个用户群对不同电影的评分。

这个问题在推荐系统、图像处理等方面都有广泛的应用。
接着，这类问题一般都有隐含的假设，即最终的矩阵应该是低秩（low rank）的。
这其实也很好理解，因为我们一般会觉得：1、不同用户对于电影的偏好可以分成聚落（cliques），比如按照观众的年龄来分，年龄相仿的观众口味往往相近；2、电影也可以分成大致几种不同的题材（genres），如：爱情片、动作片、科幻片等。所以会有低秩的特性。
简单来说，这个矩阵的行和列会有“协同”的特性，这也是这个问题的别名协同过滤（collaborative filtering）的得名原因。
另外，低秩限制会比较实用，人们都比较喜欢得到稀疏解，使得整个问题更有可诠释性。

所以这个问题的目标函数被定义成下面的样子：
$$
\begin{aligned}
&\mathrm{minimize} \quad  \mathrm{rank}(Z) \\
&\text{subject to} \quad  \sum_{(i,j):\text{Observed}} (Z_{ij} - X_{ij})^{2} \le \delta
\end{aligned} \\
\text{Impute missing } X_{ij} \text{ with } Z_{ij}
$$
然而问题来了，这个问题是非凸的，而且这种问题的规模都非常大，没办法在如此庞大的 NP-hard 问题中找到全局最优解。

早先时候，人们使用了一些启发式算法。
后来人们发现，核范数是矩阵秩的一个很好的凸近似。
（原因有点类似与 L1 范数是 L0 范数的一个凸近似。）
下面就介绍一下核范数的一些性质。

# 核范数

## 定义

矩阵 $X$ 的核范数定义为：
$$
\lVert X \rVert _{\ast} = \mathrm{tr}\left( \sqrt{X^{\mathsf{T}}X} \right)
$$
显而易见，核范数也可以等价地定义为矩阵特征值的和，考虑 $X$ 的特征值分解 $X=U \Sigma V^{\mathsf{T}}$ 显然有：
$$
\begin{aligned}
\mathrm{tr}\left( \sqrt{X^{\mathsf{T}}X} \right) & =  \mathrm{tr} \left( \sqrt{(U \Sigma V^{\mathsf{T}})^{\mathsf{T}}U \Sigma V^{\mathsf{T}}} \right) \\
& = \mathrm{tr} \left( \sqrt{V \Sigma^{\mathsf{T}} U^{\mathsf{T}} U \Sigma V^{\mathsf{T}}} \right) \\
& = \mathrm{tr} \left( \sqrt{V \Sigma^{2} V^{\mathsf{T}}} \right)  \quad \left( \Sigma^{\mathsf{T}} = \Sigma \right) \\
& = \mathrm{tr} \left( \sqrt{V^{\mathsf{T}} V \Sigma^{2}} \right) \\
& = \mathrm{tr} (\Sigma)
\end{aligned}
$$

## 凸性的证明

首先，矩阵诱导范数是凸的，即：令 $ f_x (A) = \lVert Ax \rVert _{p} \quad (p \ge 1) $ ，则 $f_x$ 凸，故 $\lVert A \rVert _{p} = \sup\limits_{\Vert x \rVert _{p}=1} f_x (A)$ 凸。特别的，$\lVert A \rVert_{2}$ 凸。
因为 $\lVert A \rVert_{\ast}$ 和 $\lVert A \rVert_{2}$ 是对偶范数，所以 $\lVert A \rVert_{\ast}$ 凸。（$\lVert A \rVert_{\ast}=\sup \limits_{\lVert X \rVert_{2}=1} \mathrm{tr} \left( A^{\mathsf{T}}X \right)$）

## 梯度的求解

基于上述 S.V.D 的假设，我们可以轻易地得到，
$$
\frac{\partial \lVert X \rVert_{\ast}}{\partial X} = \frac{\partial \mathrm{tr} (\Sigma)}{\partial X} = \frac{\mathrm{tr} (\partial \Sigma)}{\partial X}
$$
所以我们需要解出 $\partial \Sigma$ ，
考虑 $X=U \Sigma V^{\mathsf{T}}$ ，因此：
$$
\begin{aligned}
& & \partial X & = (\partial U )\Sigma V^{\mathsf{T}} + U (\partial \Sigma) V^{\mathsf{T}} + U \Sigma (\partial V^{\mathsf{T}}) \\
&\Rightarrow \, & \partial \Sigma & = U^{\mathsf{T}} (\partial X) V - U^{\mathsf{T}} (\partial U) \Sigma - \Sigma (\partial V^{\mathsf{T}}) V \\
& & & = U^{\mathsf{T}} (\partial X) V \qquad (- U^{\mathsf{T}} (\partial U) \Sigma - \Sigma (\partial V^{\mathsf{T}}) V = 0)
\end{aligned}
$$
所以：
$$
\frac{\partial \lVert X \rVert_{\ast}}{\partial X} = \frac{\mathrm{tr} (\partial \Sigma)}{\partial X} = \frac{\mathrm{tr} (U^{\mathsf{T}} (\partial X)V)}{\partial X} = \frac{\mathrm{tr} (V U^{\mathsf{T}} (\partial X))}{\partial X} = (V U^{\mathsf{T}})^{\mathsf{T}} = U V^{\mathsf{T}}
$$

# 低秩问题的近似

定义了核范数以后，我们就可以将低秩优化问题近似成相应的核范数优化问题了。
即：
$$
\begin{aligned}
&\mathrm{minimize} \quad  \lVert Z \rVert_{\ast} \\
&\text{subject to} \quad  \sum_{(i,j):\text{Observed}} (Z_{ij} - X_{ij})^{2} \le \delta
\end{aligned} \\
\text{Impute missing } X_{ij} \text{ with } Z_{ij}
$$
