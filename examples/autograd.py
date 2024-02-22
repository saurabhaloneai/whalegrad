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
# print(c.accumulate_grad)