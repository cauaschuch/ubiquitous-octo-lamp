import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

# A = Mo B = Os
class p:
    R = 1.986
    def __init__(self, mul1A, num1A, mul2A, num2A, mul1B, num1B, mul2B, num2B):
        self.mul1A = mul1A
        self.num1A = num1A
        self.num1B = num1B
        self.mul1B = mul1B
        self.mul2A = mul2A
        self.mul2B = mul2B
        self.num2A = num2A
        self.num2B = num2B
        self.T = np.arange(min(self.num1A, self.num1B, self.num2A, self.num2B)-1, max(self.num1A, self.num1B, self.num2A, self.num2B)+1,1)
    
    def E1(num,mul,t):
        return np.exp((mul*(num-t))/(p.R*t))
    
    def E2(num1,mul1,num2,mul2,t):
        return np.exp((mul1*(num1-t) - mul2*(num2-t))/(p.R*t))
    
    def ponto(self):
        def f(t):
            return (p.E1(self.num1A, self.mul1A, t) - 1) / (p.E1(self.num1A, self.mul1A, t) - p.E1(self.num1B, self.mul1B, t)) - (p.E1(self.num2A, self.mul2A, t) - 1) / (p.E1(self.num2A, self.mul2A, t) - p.E1(self.num2B, self.mul2B, t))
        return spo.newton(f, self.T.mean())
    
    def plot(self):
        t = self.T
        EAbl = p.E1(self.num1A, self.mul1A, t)
        EAel = p.E1(self.num2A, self.mul2A, t)
        EAbe = p.E2(self.num1A, self.mul1A, self.num2A, self.mul2A, t)
        EBbl = p.E1(self.num1B, self.mul1B, t)
        EBel = p.E1(self.num2B, self.mul2B, t)
        EBbe = p.E2(self.num1B, self.mul1B, self.num2B, self.mul2B, t)

        XBl1 = (EAbl - 1) / (EAbl - EBbl)
        XBb1 = XBl1 * EBbl
        XBl2 = (EAel - 1) / (EAel - EBel)
        XBe1 = XBl2 * EBel
        XBe2 = (EAbe - 1) / (EAbe - EBbe)
        XBb2 = XBe2 * EBbe

        t0= self.ponto()
        X0 = (p.E1(self.num1A, self.mul1A, t0) - 1) / (p.E1(self.num1A, self.mul1A, t0) - p.E1(self.num1B, self.mul1B, t0))

        mask1 = t > t0
        mask2 = t < t0
        plt.scatter(X0, t0, color='black')
        plt.plot(XBl1[mask1], t[mask1], label='B_l1',color='green')
        plt.plot(XBb1[mask1], t[mask1], label='B_b1',color='blue')
        plt.plot(XBl2[mask1], t[mask1], label='B_l2',color='green')
        plt.plot(XBe1[mask1], t[mask1], label='B_e1',color='blue')
        plt.plot(XBe2[mask2], t[mask2], label='B_e2',color='red')
        plt.plot(XBb2[mask2], t[mask2], label='B_b2',color='red')
        plt.hlines(t0, XBe1[mask1][0], XBb1[mask1][0], 'black')
        plt.xlim(0, 1)
        plt.show()
        print(XBl1[mask1])

L = p(2, 2900, 2, 1900, 2.8, 1960, 2, 3300)		
L.plot()
