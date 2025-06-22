import numpy as np
import matplotlib.pyplot as plt
L = np.array([0,2,3,4,4.5,5.3,6,7,7.5,7.9,8.7,9.2,9.8,10.2,11.5,13,14])/100
t = np.arange(0,50,3)
print(L.size,t.size)

dT = 300-77
# parte 1 conducao solida
#area de conducao solida
def Qcs(k,A,dT,l):
    qcs = k*A*dT/l
    return qcs

esp = 1/1000
def area_secao(e,d):
    return np.pi*(d**2-(d-e*2)**2)/4




# parte 2  radiacao
def Qr(sig,eps,A,T1,T2):
    qr = sig*eps*A*(T2**4-T1**4)
    return qr
sig = 5.670374419e-8
area1 = (2*np.pi*(0.175/2)*(0.592-L))
#area2 = 2*np.pi*(0.0504/2)*0.12
calor_rad_tot = Qr(sig,0.048,area1,77,300)#+Qr()
print(calor_rad_tot)
Qtot=calor_rad_tot




# parte 3 conducao gasosa

P = 133.322*1.9e-4
def Qvac(T1,T2):
    qvac = .0159*.9*P*(T2-T1)
    return qvac

calor_cg_tot = np.array([Qvac(77,300) for i in range(17)])
Qtot+= calor_cg_tot
print(calor_cg_tot)











# parte 4 conveccao


def Qcon(h,A,T1,T2):
    qcon = h*A*(T2-T1)
    return qcon

h=10000
T1=77
T2 = 300
A = np.pi*0.176*L[::-1]
print(A)
calor_conv_tot = Qcon(h,A,T1,T2)
print(calor_conv_tot)
print(Qtot)



Joule_tot = Qtot/ 161.35



print(np.sum(180*(Joule_tot)))


A = np.pi*(0.176**2 - 0.0484**2-2*0.0168**2-0.0124**2)/4
plt.scatter(t*60,L[::-1]*A*1000)
plt.axhline(A*0.12*1000)
plt.show()
#V=np.zeros_like(L)
#V[:-2]=L[::-2]*A*1000 +
plt.scatter(t*60,L)
plt.axhline(0.12)
plt.show()














