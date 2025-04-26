
import elastic
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
import pandas as pd

plt.rcParams['text.usetex'] = True

V_mag = """
      310.3       136.9       136.9         0.0         0.0         0.0 
      136.9       310.3       136.9         0.0         0.0         0.0 
      136.9       136.9       310.3         0.0         0.0         0.0 
        0.0         0.0         0.0        94.0         0.0         0.0 
        0.0         0.0         0.0         0.0        94.0         0.0 
        0.0         0.0         0.0         0.0         0.0        94.0 
"""
Ni = """
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

materiais = {
    "V (magn.)": elastic.Elastic(V_mag),
    "Nb": elastic.Elastic(Ni),
    "Ta": elastic.Elastic(Ta)
}




# Ângulo azimutal
phi = np.linspace(0, 2 * np.pi, 200)

# Criar os 3 subplots em linha
fig, axs = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(24, 6))

# Primeiro subplot: plano xy (theta = pi/2)
theta = np.pi / 2
for nome, mat in materiais.items():
    f = np.vectorize(mat.Young_2)
    r = f(theta, phi)
    axs[0].plot(phi, r, label=nome, linewidth=2)

axs[0].set_title("Young's modulus", fontsize=12)
axs[0].grid(True)

# Estilo das bordas e ticks
for ax in axs:
    ax.spines['polar'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', length=5, width=2)
    ax.tick_params(axis='both', which='minor', length=3, width=1.5)

# Legenda só no primeiro
axs[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
#phi = 0
#theta = np.linspace(0, 2 * np.pi, 200)
#for nome, mat in materiais.items():
#    f = np.vectorize(mat.Young_2)
#    r = f(theta, phi)
#    axs[1].plot(theta, r, linewidth=2)
#axs[1].set_title("Young", fontsize=12)
#
#phi = np.pi/4
#theta = np.linspace(0, 2 * np.pi, 200)
#for nome, mat in materiais.items():
#    f = np.vectorize(mat.Young_2)
#    r = f(theta, phi)
#    axs[2].plot(theta, r, linewidth=2)
#axs[2].set_title("yz", fontsize=12)
#
#
phi = np.linspace(0, 2 * np.pi, 200)
theta = np.pi / 2
for nome, mat in materiais.items():
    f = np.vectorize(mat.LC_2)
    r = f(theta, phi)
    axs[1].plot(phi, r, label=nome, linewidth=2)
axs[1].plot(np.max(phi),1.8,alpha=0)

axs[1].set_title("Linear compressibility", fontsize=12)
axs[1].grid(True)




phi = np.linspace(0, 2 * np.pi, 200)
for nome,mat in materiais.items():
    f = np.vectorize(lambda x: mat.shear2D([np.pi / 2, x]))
    r = f(phi)
    axs[2].plot(phi, r[0],)

axs[2].set_title("", fontsize=12)
#ax.plot(phi, r[1], color='r')
axs[2].grid(True)

for nome,mat in materiais.items():
    f = np.vectorize(lambda x: mat.shear2D([np.pi / 2, x]))
    r = f(phi)
    axs[3].plot(phi, r[1],)

axs[3].set_title("", fontsize=12)
#ax.plot(phi, r[1], color='r')
axs[3].grid(True)




plt.tight_layout()
plt.show()





















plt.show()


