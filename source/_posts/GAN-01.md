---
title: 生成对抗网络（一）—— GAN 原理简述及一个 TensorFlow 示例
date: 2017-11-01 16:09:01
categories: ML
tags:
     - Deep learning
     - GAN
description: 对比较火爆的生成对抗网络（GAN）进行了简介，并提供了一个 TensorFlow 示例。
---

# 生成对抗网络（GAN）的基本概念

2014年 Goodfellow 等人提出了一个**生成对抗网络（GAN）**的概念后一直火爆至今，相比于其它生成模型，它有以下几个优点：

- 模型只用到了反向传播，而不需要马尔科夫链。
- 训练时不需要对隐变量做推断。
- 理论上，只要是可微分函数都可以用于构建 D（判别模型）和 G（生成模型），因为能够与深度神经网络结合做深度生成式模型。
- G 的参数更新不是直接来自数据样本，而是使用来自 D 的反向传播。

它的主要思想来源于博弈论中的零和游戏。简单描述起来就是我们有一些真实的数据，也有一些随机生成的假数据。G 负责把这些数据拿过来拼命地模仿成真实数据并把它们藏在真实数据中，而 D 就拼命地要把伪造数据和真实数据分开。经过二者的博弈以后，G 的伪造技术越来越厉害，D 的鉴别技术也越来越厉害。直到 D 再也分不出数据是真实的还是 G 生成的数据的时候（D 判断正确的概率为 $1/2$ ，即随机猜测），我们就达到了目的。此时 G 可以用来模仿生成所谓的“真实数据”了。

为了学习到生成器在数据 $x$ 生的分布 $p_g(x)$ ，我们先定义一个先验的输入噪声变量 $p_z(z)$ ，然后根据 $G(z;\theta_g)$ 将其映射到数据空间中，其中 $G$ 为多层感知机所表征的可微函数。然后利用第二个多层感知机 $D(x;\theta_d)$ ，它的输出为单个标量，表示 $x$ 来源于真实数据的概率。我们训练 $D$ 来最大化正确分配真实样本和生成样本的概率，因此我们就可以通过最小化 $\log (1-D(G(z)))$ （G 生成的数据被分错的损失函数）而同时训练 $G$ 。即：
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))]
$$

# 最优判别器

考虑到
$$
\mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))] = \mathbb{E}_{x \sim p_{g}(x)}[\log (1-D(x)]
$$
我们可以将价值函数 $V(D,G)$ 展开为在全体 $x$ 上的积分形式
$$
V(G, D) = \int_{x} p_{\text{data}}(x) \log (D(x)) + p_g(x) \log (1-D(x)) \, dx
$$
因为求积分最大值可以转化为求被积函数最大值，且 $p_{\text{data}}(x)$ 和 $p_g(x)$ 均为标量，因此我们讨论如下形式的式子的最大值：
$$
f(y) = a \log y + b \log (1-y)
$$
考虑到
$$
f'(y) = 0 \Rightarrow \frac{a}{y}-\frac{b}{1-y}=0 \Rightarrow y=\frac{a}{a+b}
$$
且
$$
f''(y) \Big|_{y=\frac{a}{a+b}} = -\frac{a}{\left( \frac{a}{a+b} \right)^{2}}-\frac{b}{1-\left( \frac{a}{a+b} \right)^{2}}<0
$$
我们得到极大值在 $\frac{a}{a+b}$ 处取到，令 $a=p_{\text{data}}(x), b=p_g(x), y=D(x)$ ，则容易对比出最优判别器
$$
D(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

# 最优生成器

显然，GAN 的目的是让 $p_{\text{data}}=p_g$ 。此时判别器已经完全分辨不出真实数据和生成数据的区别了。即
$$
D_G^{\ast}=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}=\frac{1}{2}
$$
Goodfellow 等人证明，这个最优生成器就是上面那个 min max 的式子。

# 一个 TensorFlow 示例（基于 MNIST 数据库）

简单起见，没有用 CNN。主要是展示 G 与 D 的博弈过程。

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# from PIL import Image

mnist = input_data.read_data_sets("MNIST_data/")
images = mnist.train.images


def xavier_initializer(shape):
    return tf.random_normal(shape=shape, stddev=1.0 / shape[0])


# Generator
z_size = 100  # maybe larger
g_w1_size = 400
g_out_size = 28 * 28

# Discriminator
x_size = 28 * 28
d_w1_size = 400
d_out_size = 1

z = tf.placeholder('float', shape=(None, z_size))
X = tf.placeholder('float', shape=(None, x_size))

# use dict to share variables
g_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(z_size, g_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[g_w1_size])),
    'out': tf.Variable(xavier_initializer(shape=(g_w1_size, g_out_size))),
    'b2': tf.Variable(tf.zeros(shape=[g_out_size])),
}

d_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(x_size, d_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[d_w1_size])),
    'out': tf.Variable(xavier_initializer(shape=(d_w1_size, d_out_size))),
    'b2': tf.Variable(tf.zeros(shape=[d_out_size])),
}


def G(z, w=g_weights):
    # here tanh is better than relu
    h1 = tf.tanh(tf.matmul(z, w['w1']) + w['b1'])
    # pixel output is in range [0, 255]
    return tf.sigmoid(tf.matmul(h1, w['out']) + w['b2']) * 255


def D(x, w=d_weights):
    # here tanh is better than relu
    h1 = tf.tanh(tf.matmul(x, w['w1']) + w['b1'])
    h2 = tf.matmul(h1, w['out']) + w['b2']
    return h2  # use h2 to calculate logits loss


def generate_z(n=1):
    return np.random.normal(size=(n, z_size))


sample = G(z)

dout_real = D(X)
dout_fake = D(G(z))

G_obj = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.ones_like(dout_fake)))
D_obj_real = tf.reduce_mean(  # use single side smoothing
    tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_real, labels=(tf.ones_like(dout_real) - 0.1)))
D_obj_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.zeros_like(dout_fake)))
D_obj = D_obj_real + D_obj_fake

G_opt = tf.train.AdamOptimizer().minimize(G_obj, var_list=g_weights.values())
D_opt = tf.train.AdamOptimizer().minimize(D_obj, var_list=d_weights.values())

# Training
batch_size = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        sess.run(D_opt, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        # run two phases of generator
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)
        })
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)
        })

        g_cost = sess.run(G_obj, feed_dict={z: generate_z(batch_size)})
        d_cost = sess.run(D_obj, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        image = sess.run(G(z), feed_dict={z: generate_z()})
        df = sess.run(tf.sigmoid(dout_fake), feed_dict={z: generate_z()})
        # print i, G cost, D cost, image max pixel, D output of fake
        print(i, g_cost, d_cost, image.max(), df[0][0])

    # You may wish to save or plot the image generated
    # to see how it looks like
    image = sess.run(G(z), feed_dict={z: generate_z()})
    image1 = image[0].reshape([28, 28])
    # print(image1)
    # im = Image.fromarray(image1)
    # im.show()
```

