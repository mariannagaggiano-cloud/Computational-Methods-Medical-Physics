import numpy as np
import matplotlib.pyplot as plt

X=np.arange(-5,100,1)
Y=np.arange(-50,50,1)

x0=0
y0=0

#step1
rn1=np.random.rand(1)
x1=-np.log(1-rn1*(1-np.exp(-1)))*1000
print(x1)


