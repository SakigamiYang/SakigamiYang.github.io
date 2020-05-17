---
title: 生成对抗网络（二）—— Wasserstein GAN 简述及一个 TensorFlow 示例
date: 2017-11-04 11:21:51
categories: ML
tags:
     - Deep learning
     - GAN
description: 对 Wasserstein GAN 进行了简介，并提供了一个 TensorFlow 示例。
---

# 原始 GAN 到底出了什么问题

在近似最优判别器下，最小化生成器的 loss 等价于最小化 $P_{\text{data}}$ 和 $P_g$ 之间的 JS 散度，而由于 $P_{\text{data}}$ 和 $P_g$ 几乎不可能有不可忽略的重叠，所以无论它们相距多远，JS 散度都是常数 $\log 2$ ，最终导致生成器的梯度近似为 $0$ ，梯度消失。

又因为最小化目标等价于最小化
$$
KL(P_{g} \Vert P_{\text{data}}) - 2 JS(P_{\text{data}} \Vert P_{g})
$$
这个目标要求同时最小化生成分布和真是分布的 KL 散度，并且最大化两者的 JS 散度。产生了数值上的不稳定。最终造成生成器宁可多生成一些重复但是很“安全”的样本，也不愿意去生成多样性的样本。就发生了所谓的 collapse mode。

# Wasserstein GAN 改动了什么

Wasserstein GAN 与 GAN 相比，只改了四点：

- 判别器最后一层去掉 sigmoid
- 生成器和判别器的 loss 不取 log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数 c
- 不要用基于动量的优化方法（包括 momentum 和 Adam），推荐使用 RMSProp，SGD 也行。

PS：第四点是论文作者从实验中用玄学得到的。因为如果使用 Adam，判别器的 loss 有时候会崩掉，然后 Adam 给出的更新方向与梯度方向夹角的 cos 值就会变成负数，判别器的 loss 梯度就变得不稳定了。但是改用 RMSProp 以后就解决了上述问题。

PPS：推荐使用比较小的 learning rate，论文中使用的是 $\alpha=0.00005$ 。

# 一个 TensorFlow 示例（基于 MNIST 数据库）

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(5):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
```

