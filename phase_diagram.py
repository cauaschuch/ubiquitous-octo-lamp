import numpy as np
import matplotlib.pyplot as plt
# A = Mo B = Os







class phase_2:
	def __init__(self, mulA, numA,mulB,numB):#FALTANDO TEM Q SER MUL1A MUL2A ETC
		self.mulA = mulA
		self.numA = num1
		self.numB = numB
		self.mulB = mulB
		self.T = 10
		
		
	def dG(mul, num):
		pass
		
	def E(dG):
		pass
	
	
	def phase_diagram(self):
		pass
	
	
	



R = 1.986
t = np.arange(1000,3500,1)
#Xb = np.zeros(t.size)


dGblMo = 2*(2900-t)
dGelMo = 2*(1900-t)
dGbeMo = dGblMo - dGelMo
EMobl = np.exp(dGblMo/(R*t))
EMoel = np.exp(dGelMo/(R*t))
EMobe = np.exp(dGbeMo/(R*t))



dGblOs = 2.8*(1960-t)
dGelOs = 2*(3300-t)
dGbeOs = dGblOs - dGelOs
EOsbl = np.exp(dGblOs/(R*t))
EOsel = np.exp(dGelOs/(R*t))
EOsbe = np.exp(dGbeOs/(R*t))


#B > L
XOsl1 = (EMobl -1)/(EMobl - EOsbl)
XOsb1 = XOsl1*EOsbl

#E > L
XOsl2 = (EMoel -1)/(EMoel - EOsel)
XOse1 = XOsl2*EOsel
#B > E
XOse2 = (EMobe -1)/(EMobe - EOsbe)
XOsb2 = XOse2*EOsbe


plt.plot(XOsl1,t,label='Os_l1')
plt.plot(XOsb1,t,label='Os_b1')
plt.plot(XOsl2,t,label='Os_l2')
plt.plot(XOse1,t,label='Os_e1')
plt.plot(XOse2,t,label='Os_e2')
plt.plot(XOsb2,t,label='Os_b2')

#X1B_l = X2B_l






plt.legend()
plt.xlim(0,1)
plt.ylim(1800,3500)
plt.show()
