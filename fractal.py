import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

#def duff(Y,t):
#    x,v=Y
#    dx=v
#    dv = g*np.cos(w*t)-d*v-b*x-a*x**3
#    return [dx,dv]
#
#a,b,g,d,w=1,-1,.3,.2,1
#P = 2*np.pi/w
#N1p=10
#dt=P/N1p
#NP=10000
#tmax=NP*P
#t = np.arange(0,tmax+dt,dt)
#Y0 = np.array([1,1])
#Y = spi.odeint(duff,Y0,t)
#x,v = Y.T
#xp,vp=x[::N1p],v[::N1p]
#np.save("xp.npy", xp)
#np.save("vp.npy", vp)
##delimitar o atrator
#xmin,xmax = np.min(xp),np.max(xp) 
#vmin,vmax = np.min(vp),np.max(vp) 

xp = np.load("xp.npy")
vp = np.load("vp.npy")

plt.plot(xp, vp, 'or', markersize=.3, alpha=.7)

#delimi tar o atrator
l =1.5
n=0
def num_quadrados(n,l):
    num = (2**n)
    e = 2*l/num
    return num, e
N = []
E_n = []
nmax=7

for i in range(0,nmax+1):
    num_caixas = 0
    num,e = num_quadrados(n,l)
    E_n.append(e)
    X,Y = np.arange(-l,l+e,e),np.arange(-l,l+e,e)
    print(X,Y,n)
    for j in range(2**n):
        for k in range(2**n):
            if np.any((xp >= X[j]) & (xp <= X[j+1]) & (vp >= Y[k]) & (vp <= Y[k+1])):
                num_caixas +=1

    N.append(num_caixas)
    n+=1
print(N)
N_e = np.array(N)
E_N = np.array(E_n)
print(N_e,E_N)
plt.axhline(-1.5)
plt.axhline(1.5)
plt.axvline(1.5)
plt.axvline(-1.5)

plt.axhline(0,color='green')
plt.axvline(0,color='green')

plt.axhline(-0.75,color='orange')
plt.axvline(-0.75,color='orange')
plt.axhline(0.75,color='orange')
plt.axvline(0.75,color='orange')

x,y=np.log(E_N),np.log(N_e)

plt.show()
coeff = np.polyfit(x,y, 1)
a= coeff[0]
#plt.scatter(x,y)
plt.scatter(E_N,N_e)
plt.xscale('log')
plt.yscale('log')
plt.title(f'D = {np.absolute(a)}')
plt.show()
