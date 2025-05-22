import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad






A_v = 11.089570428e-10
A_nb = 11.272435172e-10 #angstrons
A_ta = 11.263962942e-10

P_nb = np.array([
    485.59,   
    391.67,   
    307.36,   
    231.69,   
    163.85,   
    103.15,   
    48.73,    
    -0.65,    
    -43.57,   
    -82.10,   
    -116.48,  
    -146.96,  
    -173.85,  
    -197.56,  
    -218.29   
])*1e8 #kbar

P_v = np.array([
     450.55,  362.12,  282.87,  211.90,  148.51,
      94.51,   44.96,   -0.58,  -40.24,  -75.54,
    -106.77, -134.59, -159.00, -180.32, -198.94
])*1e8

P_ta = np.array([
     505.71,  407.91,  320.10,  241.30,  170.64,
     107.35,   50.93,   -0.69,  -45.58,  -85.52,
    -121.29, -153.02, -181.03, -205.54, -227.03
])


dist_v,dist_nb,dist_ta = np.array([
    0.4743416490,
    0.4780914437,
    0.4818120558,
    0.4855041562,
    0.4891683905,
    0.4928053803,
    0.4964157244,
    0.5000499975,
    0.5035587638,
    0.5070925528,
    0.5106018857,
    0.5140872633,
    0.5175491695,
    0.5209880723,
    0.5244044241
]),np.array([
    0.4743416490,
    0.4780914437,
    0.4818120558,
    0.4855041562,
    0.4891683905,
    0.4928053803,
    0.4964157244,
    0.5000499975,
    0.5035587638,
    0.5070925528,
    0.5106018857,
    0.5140872633,
    0.5175491695,
    0.5209880723,
    0.5244044241
]),np.array([
    0.4743416490,
    0.4780914437,
    0.4818120558,
    0.4855041562,
    0.4891683905,
    0.4928053803,
    0.4964157244,
    0.5000499975,
    0.5035587638,
    0.5070925528,
    0.5106018857,
    0.5140872633,
    0.5175491695,
    0.5209880723,
    0.5244044241
])
nb_energia = np.array([
-9200.1081523,
-9200.2549536,
-9200.37282164,
-9200.46438856,
-9200.5320567,
-9200.57801417,
-9200.60428028,
-9200.61268094,
-9200.6048859,
-9200.58245916,
-9200.54679352,
-9200.4991882,
-9200.44085262,
-9200.37286356,
-9200.29628593])*2.179874099e-18 #JOULE

new_A_nb = dist_nb*A_nb*2
new_A_v = dist_v*A_v*2
new_A_ta = dist_ta*A_ta*2
V_v = .25*new_A_v**3
V_ta =.25*new_A_ta**3
V_nb = 0.25*new_A_nb**3 #usar celula primitiva aqui ne


Vo = 0.25*A_nb**3
def P(Vp,Bo,B_):
    return 1.5*Bo*((Vo/Vp)**(7/3)-(Vo/Vp)**(5/3))*(1+0.75*(B_ - 4)*((Vo/Vp)**(2/3)-1))


parametros, V = curve_fit(P,V_nb,P_nb,p0=[300000000000000,5])

Bo,B_nb = parametros
Vp=V_nb
gamma_nb = 2/3 +.5 + 0.5*B_nb
print(B_nb,gamma_nb,'nb')

Vo = 0.25*A_v**3
def P(Vp,Bo,B_):
    return 1.5*Bo*((Vo/Vp)**(7/3)-(Vo/Vp)**(5/3))*(1+(0.75*(B_ -4))*((Vo/Vp)**(2/3)-1))


parametros, V = curve_fit(P,V_v,P_v,p0=[30000000000,5])

Bov,B_v = parametros
gamma_v = 2/3+.5 + 0.5*B_v
print(gamma_v,'v')


Vo = 0.25*A_ta**3
def P(Vp,Bo,B_):
    return 1.5*Bo*((Vo/Vp)**(7/3)-(Vo/Vp)**(5/3))*(1+(0.75*(B_ -4))*((Vo/Vp)**(2/3)-1))


parametros, V = curve_fit(P,V_ta,P_ta,p0=[30000000000000,5])

Bov,B_ta = parametros
gamma_ta= 2/3+.5 + 0.5*B_ta
print(gamma_ta,'ta')


###########################################


kb = 1.380649e-23
def debye_v(gamma,v,debye_o,vo):
    #print(vo,v)
    return debye_o*(vo/v)**gamma

def D(Td_T):
  f = lambda x: (x**3)/((np.exp(x)-1))
  result, error = quad(f, 1e-8, Td_T)
  return result*3*(Td_T)**-3


def Energy(Vp,Eo,Vo,Bo,B_):
    return Eo+(9/16*Bo*Vo)*(B_*(-1+(Vo/Vp)**(2/3))**3+(6-4*(Vo/Vp)**(2/3))*((-1+(Vo/Vp)**(2/3))**2))

def F_vib(gamma,Vp,debye_o,Vo,T):
    Td=debye_v(gamma,Vp,debye_o,Vo)
    return (9/8*kb*Td + kb*T*(3*np.log(1-np.exp(-Td/T))-D(Td/T)))



def fit_E(V_array,E_array,Eo,Vo,Bo,B_):
    parametros, xxx = curve_fit(Energy,V_array,E_array,p0=[Eo,Vo,Bo,B_])
    return parametros[0],parametros[1],parametros[2],parametros[3]


V_array,E_array,Eo,Vo,Bo,B_=V_nb,nb_energia,-9200*2.179874099e-18,.25*A_nb**3,214e9,4,
Eo,Vo,Bo,B_ = fit_E(V_array,E_array,Eo,Vo,Bo,B_)
gamma_nb = 2/3 +.5 + 0.5*B_nb
print(Eo/2.179874099e-18,Vo,Bo,B_,gamma_nb,'AAAAAAAAAA')


def F_tot(v,V_array,E_array,Eo,Vo,Bo,B_,debye_o,T):
    #DEFINO ELE AQUI DENTRO OU FORA?
    energia = Energy(v,Eo,Vo,Bo,B_)
    Fvib = F_vib(gamma_nb,v,debye_o,Vo,T)
    return Fvib  +energia/29#F = energia+Fvib


    
#parametros = fit_E(V_nb,nb_energia,-9200*2.179874099e-18,0.25*A_nb**3,214e9,8)
#print(parametros[0],parametros[1],parametros[2]/1e9,parametros[3])

T = np.linspace(1e-8,600,600)
V_SET = np.linspace(.25*A_nb**3 *0.95,1.1*.25*A_nb**3,1000)
F = np.array([F_tot(v,V_nb,nb_energia,-9200*2.179874099e-18,.25*A_nb**3,214e9,4,330.31,140) for v in V_SET])
plt.plot(V_SET,F.T,label='F(140)')
F = np.array([F_tot(v,V_nb,nb_energia,-9200*2.179874099e-18,.25*A_nb**3,214e9,4,330.31,200) for v in V_SET])
plt.plot(V_SET,F.T,label='F(200)')
F = np.array([F_tot(v,V_nb,nb_energia,-9200*2.179874099e-18,.25*A_nb**3,214e9,4,330.31,0.) for v in V_SET])
plt.plot(V_SET,F.T,label='F (0)')
F = np.array([F_tot(v,V_nb,nb_energia,-9200*2.179874099e-18,.25*A_nb**3,214e9,4,330.31,400) for v in V_SET])
plt.plot(V_SET,F.T,label='F(400)')
#plt.plot(T,F.T[1],label='Fvib')
#plt.plot(T,F.T[2],label='E')
plt.legend()
plt.show()
#pegar p(efermi_ ) modelo de sommerfield



from scipy.optimize import minimize_scalar

Vo_nb = 0.25*A_nb**3
#print(Vo_nb)

Eo,Vo,Bo,B_,gamma_nb,debye_o = -2.005617688180834e-14, 3.581226126012145e-28 ,213484294670.90308 ,4.622613981210721, 3.4809539440617097,330.31


T = np.linspace(0,350,1000)
v_min_nb = np.zeros(len(T))
vi = np.linspace(Vo*0.98,Vo*1.07,2000)
#for i in range(len(T)):
#    def f_tot(v):
#        #DEFINO ELE AQUI DENTRO OU FORA?
#        #energia = Energy(v,Eo,Vo,Bo,B_)
#        energia = Eo+(9/16*Bo*v)*(B_*(-1+(Vo/v)**(2/3))**3+(6-4*(Vo/v)**(2/3))*((-1+(Vo/v)**(2/3))**2))
#        #f_vib = F_vib(gamma_nb,v,debye_o,Vo,T[100])
#        Td=debye_v(gamma_nb,v,debye_o,Vo)
#        f_vib=(9/8*kb*Td + kb*T[i]*(3*np.log(1-np.exp(-Td/T[i]))-D(Td/T[i])))
#        return (energia/29 +f_vib)#F = energia+Fvib
#    f = np.array([f_tot(vi_i) for vi_i in vi])
#    #print(np.min(f),np.argmin(f),i)
#    #v_min_nb[i] = vi[np.argmin(f)]
#    #v_min_nb[i] = minimize_scalar(f_tot, bounds=(Vo*0.995, Vo*1.08), method='bounded').x
#    #v_min_nb[i] = minimize_scalar(f_tot, method='brent').x
np.set_printoptions(precision=12)
#print(v_min_nb)
#np.save("nb_vols.npy", v_min_nb)

v_min_nb = np.load("nb_vols.npy")
plt.plot(T,v_min_nb,label='nb')

A_v = 11.0764e-10
new_A_v = dist_v*A_v*2
V_v = .25*new_A_v**3
v_energia = np.array([-8478.3497467
    ,-8478.48258363
    ,-8478.58919687
    ,-8478.67198687
    ,-8478.73314208
    ,-8478.77465909
    ,-8478.79834734
    ,-8478.80592407
    ,-8478.7988465
    ,-8478.77861556
    ,-8478.746451
    ,-8478.70357179
    ,-8478.65106424
    ,-8478.59018215
    ,-8478.52171922])*2.179874099e-18
V_array,E_array,Eo,Vo,Bo,B_=V_v,v_energia,-8478*2.179874099e-18,.25*A_v**3,204.04e9,4,
Eo,Vo,Bo,B_ = fit_E(V_array,E_array,Eo,Vo,Bo,B_)
gamma_v = 2/3 +.5 + 0.5*B_
print(Eo/2.179874099e-18,Vo,Bo,B_,gamma_v,'AAAAAAAAAA')


debye_o = 346.1964780859712
vi = np.linspace(Vo*0.98,Vo*1.07,2000)
v_min_v = np.zeros(len(T))
#for i in range(len(T)):
#    def f_tot(v):
#        #DEFINO ELE AQUI DENTRO OU FORA?
#        #energia = Energy(v,Eo,Vo,Bo,B_)
#        energia = Eo+(9/16*Bo*v)*(B_*(-1+(Vo/v)**(2/3))**3+(6-4*(Vo/v)**(2/3))*((-1+(Vo/v)**(2/3))**2))
#        #f_vib = F_vib(gamma_v,v,debye_o,Vo,T[100])
#        Td=debye_v(gamma_v,v,debye_o,Vo)
#        f_vib=(9/8*kb*Td + kb*T[i]*(3*np.log(1-np.exp(-Td/T[i]))-D(Td/T[i])))
#        return (energia/29 +f_vib)#F = energia+Fvib
#    f = np.array([f_tot(vi_i) for vi_i in vi])
#    print(np.min(f),np.argmin(f),i)
#    v_min_v[i] = vi[np.argmin(f)]
#    #v_min_nb[i] = minimize_scalar(f_tot, bounds=(Vo*0.995, Vo*1.08), method='bounded').x
#    #v_min_nb[i] = minimize_scalar(f_tot, method='brent').x
#np.set_printoptions(precision=12)
#np.save("v_vols.npy", v_min_v)
v_min_v = np.load("v_vols.npy")
plt.plot(T,v_min_v,label='v')













ta_energia = np.array([
-11557.15548103
,-11557.30798632
,-11557.43042997
,-11557.52554576
,-11557.59582862
,-11557.64355261
,-11557.67078795
,-11557.67950089
,-11557.67133893
,-11557.64802118
,-11557.61092776
,-11557.56143311
,-11557.50078603
,-11557.43017253
,-11557.35065244
    ])*2.179874099e-18
V_array,E_array,Eo,Vo,Bo,B_=V_ta,ta_energia,-11557.68*2.179874099e-18,.25*A_ta**3,223.27e9,4,
Eo,Vo,Bo,B_ = fit_E(V_array,E_array,Eo,Vo,Bo,B_)
gamma_v = 2/3 +.5 + 0.5*B_
print(Eo/2.179874099e-18,Vo,Bo,B_,gamma_v,'AAAAAAAAAA')

T=np.linspace(0,350,350)
debye_o = 301.89286368728955
vi = np.linspace(Vo*0.9999,Vo*1.02,5000)
v_min_v = np.zeros(len(T))
#for i in range(len(T)):
#    def f_tot(v):
#        #DEFINO ELE AQUI DENTRO OU FORA?
#        #energia = Energy(v,Eo,Vo,Bo,B_)
#        energia = Eo+(9/16*Bo*v)*(B_*(-1+(Vo/v)**(2/3))**3+(6-4*(Vo/v)**(2/3))*((-1+(Vo/v)**(2/3))**2))
#        #f_vib = F_vib(gamma_v,v,debye_o,Vo,T[100])
#        Td=debye_v(gamma_v,v,debye_o,Vo)
#        f_vib=(9/8*kb*Td + kb*T[i]*(3*np.log(1-np.exp(-Td/T[i]))-D(Td/T[i])))
#        return (energia/29 +f_vib)#F = energia+Fvib
#    f = np.array([f_tot(vi_i) for vi_i in vi])
#    print(np.min(f),np.argmin(f),i)
#    v_min_v[i] = vi[np.argmin(f)]
#    #v_min_nb[i] = minimize_scalar(f_tot, bounds=(Vo*0.995, Vo*1.08), method='bounded').x
#    #v_min_nb[i] = minimize_scalar(f_tot, method='brent').x
#np.set_printoptions(precision=12)
#np.save("ta_vols.npy", v_min_v)
v_min_ta = np.load("ta_vols.npy")
plt.plot(T,v_min_ta,label='ta')
print(v_min_v[0]**(1/3),v_min_nb[0]**(1/3),v_min_ta[0]**(1/3))
plt.legend()
plt.show()

#dv_dt =      np.zeros(349)
#dv_dt[0:]=v_min_ta[1:] - v_min_ta[0:-1]
#plt.scatter(T[0:-1], dv_dt, label="dV/dT",)
#plt.legend()
#plt.show()


#BULK = v (d²F/dv²)
