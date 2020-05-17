---
title: 浅说范数规范化（附录）
date: 2017-09-13 11:14:14
categories: ML
tags:
      - Norm regularization
      - Matrix theory
description: 【浅说范数规范化】系列的附录，包含一些数学相关证明和知识。
---

# 附录 A：矩阵条件数 $\kappa({A})$

在本系列第一篇里提到病态矩阵的时候，说到了矩阵条件数的一些性质，下面给一个简单的证明。

事实上由于条件数是描述线性方程系统 $AX = b$ 的变化敏感度的一个量，所以我们给这个系统的每个量一个微扰，然后解出 $x$ 的变化程度：
$$
\begin{aligned}
& \quad (A+\Delta A)(x + \Delta x) = b+\Delta b \\
& \Rightarrow \Delta x = A^{-1}[\Delta b - (\Delta A) x - (\Delta A)(\Delta x)] \\
& \Rightarrow \lVert \Delta x \rVert \le \lVert A^{-1} \rVert (\lVert \Delta b \rVert + \lVert \Delta A \rVert \lVert x \rVert + \lVert \Delta A\rVert \lVert \Delta x\rVert ) \\
& \Rightarrow (1-\lVert A^{-1} \rVert \lVert \Delta A \rVert) \lVert \Delta x \rVert \le \lVert A^{-1} \rVert (\lVert \Delta b \rVert + \lVert \Delta A \rVert \lVert x \rVert) \\
& \Rightarrow \lVert \Delta x \rVert \le \frac{\lVert A^{-1} \rVert}{1-\lVert A^{-1} \rVert \lVert \Delta A \rVert} \left( \lVert \Delta b \rVert + \lVert \Delta A \rVert \lVert x \rVert \right) \qquad (\because \, \lVert A^{-1} \rVert \lVert \Delta A \rVert \le 1) \\
& \Rightarrow \frac {\lVert \Delta x \rVert}{\lVert x \rVert} \le \frac{\lVert A^{-1} \rVert \lVert A \rVert}{1-\lVert A^{-1} \rVert \lVert \Delta A \rVert} \left( \frac{\lVert \Delta b \rVert}{\lVert b \rVert} + \frac{\lVert \Delta A \rVert}{\lVert A \rVert} \right) \qquad (\because \, \lVert b \rVert \le \lVert A \rVert \lVert x \rVert)
\end{aligned}
$$
当 $\Delta A = 0$ 时，有：
$$
\frac{\lVert \Delta x \rVert}{\lVert x \rVert} \le \kappa(A) \frac{\lVert \Delta b \rVert}{\lVert b \rVert}
$$
当 $\Delta b = 0$ 时，有：
$$
\frac{\lVert \Delta x \rVert}{\lVert x \rVert} \le \kappa(A) \frac{\lVert \Delta A \rVert}{\lVert A \rVert}(1+o(\lVert A^{-1} \rVert))
$$
一个常用的条件数是 2-条件数：
$$
\kappa_{2}(A) = \lVert A^{-1} \rVert _{2} \lVert A \rVert _{2} = \sqrt{\frac{\lambda _{max}}{\lambda _{min}}}
$$
其中，$\lambda_{max}$ 和 $\lambda_{min}$ 分别是 $A^{\mathsf{H}}A$ 的最大特征值和最小特征值。
（从这个角度上来看，条件数也确实是衡量矩阵敏感度的一个值。）

