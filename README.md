# whalegrad 

'Whalegrad' - A Lightweight Deep Learning Framework

Welcome to whalegrad, my personal deep learning framework project. This is learning project for me. and yeah I recreate the wheel. (readme file written by me and modified by chatgpt :) 

![](https://github.com/saurabhaloneai/whalegrad/blob/main/images/whalegrad.png)


Features:

* Lightweight Framework: Whalegrad is a compact deep learning framework built entirely in Python, spanning just under 1000 lines of code. It's designed to be easy to understand and modify.

* Inspired by the Greats: Andrej Karpathy's(GOD) micrograd and tinygrad by geohot(hack) – Their work inspired me to create something of my own.

* Multi-Dimensional Array Support: just like tensor it supports the whalor. it is nothing but just the fancy name 

* Automatic Gradient Computation: yeah it supoorts the autograd 

* Functionality: is it have lots of activation fucntions, loss fucntions, optims and new embeddings.

# Get start with autograd:


```
from whalegrad.engine.whalor import Whalor

a = Whalor([5], requires_grad =True,)
b = Whalor([4], requires_grad = True)
c = a * b
g = c * b 
f = g - a
k = f * a + b

# print(a.shape)
k.backward([1])

# print(a)
print(a.grad)
print(b.grad) 

```