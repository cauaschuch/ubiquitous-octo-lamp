import matplotlib.pyplot as plt
import numpy as np

V_mag = """
      310.3       136.9       136.9         0.0         0.0         0.0 
      136.9       310.3       136.9         0.0         0.0         0.0 
      136.9       136.9       310.3         0.0         0.0         0.0 
        0.0         0.0         0.0        94.0         0.0         0.0 
        0.0         0.0         0.0         0.0        94.0         0.0 
        0.0         0.0         0.0         0.0         0.0        94.0 
"""
Nb = """
      355.2       144.0       144.0         0.0         0.0         0.0 
      144.0       355.2       144.0         0.0         0.0         0.0 
      144.0       144.0       355.2         0.0         0.0         0.0 
        0.0         0.0         0.0       107.3         0.0         0.0 
        0.0         0.0         0.0         0.0       107.3         0.0 
        0.0         0.0         0.0         0.0         0.0       107.3 
"""
Ta = """
      371.1       149.3       149.3         0.0         0.0         0.0 
      149.3       371.1       149.3         0.0         0.0         0.0 
      149.3       149.3       371.1         0.0         0.0         0.0 
        0.0         0.0         0.0       113.3         0.0         0.0 
        0.0         0.0         0.0         0.0       113.3         0.0 
        0.0         0.0         0.0         0.0         0.0       113.3 
"""
V11,V12,V44    = 310.3e9,136.9e9,94.0e9
Nb11,Nb12,Nb44 = 355.2e9,144.0e9,107.3e9
Ta11,Ta12,Ta44 = 371.1e9,149.3e9,113.3e9

u_to_kg = 1.66053906660e-27
V_mass  = 50.9415  * u_to_kg  # ≈ 8.459e-26 kg
Nb_mass = 92.9064  * u_to_kg  # ≈ 1.542e-25 kg
Ta_mass = 180.9479 * u_to_kg  # ≈ 3.005e-25 kg
Ni_mass = 58.6934  * u_to_kg  # ≈ 9.743e-26 kg
Si_mass = 28.0855  * u_to_kg  # ≈ 4.666e-26 kg

a_V = 11.089e-10
a_Nb = 11.272e-10
a_Ta = 11.264e-10

vol_prim_V = a_V**3/4
vol_prim_Nb = a_Nb**3/4
vol_prim_Ta = (a_Ta**3)/4


M_V  = (6*V_mass + 16*Ni_mass + 7*Si_mass)
M_Nb = (6*Nb_mass + 16*Ni_mass + 7*Si_mass)
M_Ta = (6*Ta_mass + 16*Ni_mass + 7*Si_mass)

p_V = M_V / vol_prim_V
p_Nb = M_Nb / vol_prim_Nb
p_Ta = M_Ta / vol_prim_Ta

C_A_L_v = np.sqrt(V11 / p_V)
C_A_T_v = np.sqrt(V44 / p_V)
C_B_L_v = np.sqrt(((V11+4*V44+2*V12)/3) / p_V)
C_B_T_v = np.sqrt(((V11+V44-V12)/3) / p_V)
C_Y_L_v = np.sqrt(0.5*(V11+2*V44+V12)/p_V)
C_Y_Th_v = np.sqrt(0.5*(V11-V12)/p_V)
C_Y_T3_v = np.sqrt(V44/p_V)

C_A_L_nb = np.sqrt(V11 / p_Nb)
C_A_T_nb = np.sqrt(V44 / p_Nb)
C_B_L_nb = np.sqrt(((V11 + 4*V44 + 2*V12) / 3) / p_Nb)
C_B_T_nb = np.sqrt(((V11 + V44 - V12) / 3) / p_Nb)
C_Y_L_nb = np.sqrt(0.5 * (V11 + 2*V44 + V12) / p_Nb)
C_Y_Th_nb = np.sqrt(0.5 * (V11 - V12) / p_Nb)
C_Y_T3_nb = np.sqrt(V44 / p_Nb)

C_A_L_ta = np.sqrt(V11 / p_Ta)
C_A_T_ta = np.sqrt(V44 / p_Ta)
C_B_L_ta = np.sqrt(((V11 + 4*V44 + 2*V12) / 3) / p_Ta)
C_B_T_ta = np.sqrt(((V11 + V44 - V12) / 3) / p_Ta)
C_Y_L_ta = np.sqrt(0.5 * (V11 + 2*V44 + V12) / p_Ta)
C_Y_Th_ta = np.sqrt(0.5 * (V11 - V12) / p_Ta)
C_Y_T3_ta = np.sqrt(V44 / p_Ta)

print(C_A_L_v,C_A_T_v,C_B_L_v,C_B_T_v,C_Y_L_v,C_Y_Th_v,C_Y_T3_v)
print(C_A_L_nb,C_A_T_nb,C_B_L_nb,C_B_T_nb,C_Y_L_nb,C_Y_Th_nb,C_Y_T3_nb)
print(C_A_L_ta,C_A_T_ta,C_B_L_ta,C_B_T_ta,C_Y_L_ta,C_Y_Th_ta,C_Y_T3_ta)
