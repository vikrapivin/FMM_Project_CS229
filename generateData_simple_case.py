import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=20191201)
#np.matmul(np.array([[1, -0.5],[-0.5, 1]]),np.random.randn(2,20))

mean1 = np.array([1.9752789582069352, 2.612221568288382])
mean2 = np.array([4.541859413496787, 0.35506076682060406])/2
mean3 = np.array([3.921817103081432, 1.4212370116812414])
mean4 = np.array([0.6694035336660469, 4.319707723507271])
mean5 = np.array([2.1947963962071104, 2.9138320445455426])*2
print(mean1)
print(mean2)
print(mean3)
print(mean4)
print(mean5)

sigma1 = np.array([[3.014872342722073, 2.388503534014302], [2.388503534014302, 2.1695099568429415]])*0.125
sigma2 = np.array([[0.5793524918273765, 0.11838069403547669], [0.11838069403547669, 0.6892525819200488]])*0.125
sigma3 = np.array([[0.4577992085479581, -0.07447718766113565], [-0.07447718766113565, 0.552959770463783]])*0.125
sigma4 = np.array([[1.16093956014782873, -0.5370566978082062], [-0.5370566978082062, 0.50445798899637193]])*0.125
sigma5 = np.array([[1.9455359884256378,  0.1*1.244426411988547], [0.1*1.244426411988547, 2.788164624057118]])*0.125
# sigma1 = np.array([[1,0],[0,1]])*0.25
# sigma2 = np.array([[1,0],[0,1]])*0.25
# sigma3 = np.array([[1,0],[0,1]])*0.25
# sigma4 = np.array([[1,0],[0,1]])*0.25
# sigma5 = np.array([[1,0],[0,1]])*0.25
# sigma1 = np.array([[1,0],[0,1]])*0.125
# sigma2 = np.array([[1,0],[0,1]])*0.125
# sigma3 = np.array([[1,0],[0,1]])*0.125
# sigma4 = np.array([[1,0],[0,1]])*0.125
# sigma5 = np.array([[1,0],[0,1]])*0.125
# sigma1 = np.array([[1,0],[0,1]])*0.01
# sigma2 = np.array([[1,0],[0,1]])*0.01
# sigma3 = np.array([[1,0],[0,1]])*0.01
# sigma4 = np.array([[1,0],[0,1]])*0.01
# sigma5 = np.array([[1,0],[0,1]])*0.01

num1 = 2000*2
num2 = 1000*2
num3 = 1500*2
num4 = 3000*2
num5 = 750*2

#np.random.multivariate_normal((0,0), [[1,-0.5],[-0.5,1]],(2,20))[[0.5793524918273765, 0.11838069403547669], [0.11838069403547669, 0.6892525819200488]][[0.5793524918273765, 0.11838069403547669], [0.11838069403547669, 0.6892525819200488]]
gen1 = np.random.multivariate_normal(mean1, sigma1, num1)
gen2 = np.random.multivariate_normal(mean2, sigma2, num2)
gen3 = np.random.multivariate_normal(mean3, sigma3, num3)
gen4 = np.random.multivariate_normal(mean4, sigma4, num4)
gen5 = np.random.multivariate_normal(mean5, sigma5, num5)

totalDist = np.concatenate((gen1,gen2,gen3,gen4,gen5))
# totalDist = np.concatenate((gen1,gen2,gen3))

xmin = -1
xmax = 8
ymin = -1
ymax = 8
np.save('gen_simple.npy', totalDist)
np.savetxt("gen_simple.csv", totalDist, delimiter=",")
plt.figure()
plt.scatter(totalDist[:,0],totalDist[:,1])
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])
# plt.show()
plt.savefig('gen_unlabeled.pdf')

plt.figure()
plt.scatter(gen1[:,0],gen1[:,1])
plt.scatter(gen2[:,0],gen2[:,1])
plt.scatter(gen3[:,0],gen3[:,1])
plt.scatter(gen4[:,0],gen4[:,1])
plt.scatter(gen5[:,0],gen5[:,1])
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])
plt.savefig('gen_labeled.pdf')