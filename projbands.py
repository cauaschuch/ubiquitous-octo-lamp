import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
from  ProjBands_maker import *
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Exemplo do caminho em k-space (dado por você)
k_distances = [0.0000, 1.0000, 1.5000, 1.8536, 2.9142, 3.7802, 4.3926, 4.7462, 5.4533, 6.0656, 6.4192]


def remap_xvals(kpoints1, k_distances, total_points):
    norm_kdist = (np.array(k_distances) - k_distances[0]) / (k_distances[-1] - k_distances[0])
    xvals = np.zeros(total_points)
    for i in range(len(kpoints1) - 1):
        start_idx = int(kpoints1[i])
        end_idx = int(kpoints1[i + 1])
        npts = end_idx - start_idx
        if end_idx == total_points:
            # Último intervalo → inclui o ponto final
            xvals[start_idx:end_idx] = np.linspace(norm_kdist[i], norm_kdist[i + 1], npts, endpoint=True)
        else:
            xvals[start_idx:end_idx] = np.linspace(norm_kdist[i], norm_kdist[i + 1], npts, endpoint=False)
    return xvals, norm_kdist

def plot(band_data, Ef, nkstotal, legend):
    # Dados dos pontos de alta simetria e distâncias reais
    kpoints1 = [0.0000, 51., 102., 153., 204., 255., 306., 357., 408., 459., 512.]
    k_distances = [0.0000, 1.0000, 1.5000, 1.8536, 2.9142, 3.7802, 4.3926, 4.7462, 5.4533, 6.0656, 6.4192]
    k_labels = ['$\Gamma$', 'X', 'W', 'K', '$\Gamma$', 'L', 'U', 'W', 'L', 'K|U', 'X']

    # Mapeia todos os índices de 0 a 512 para valores normalizados de x
    xvals_total, norm_kdist = remap_xvals(kpoints1, k_distances, nkstotal)

    # Converte os índices do band_data[:, 0] em inteiros e mapeia para x
    k_indices = band_data[:, 0].astype(int)
    x_mapped = xvals_total[k_indices]

    plt.figure(figsize=(10, 6))
    colors = ["#31688e", "#e9d91b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    map = band_data[:, 2]
    plt.scatter(x_mapped, band_data[:, 1] - Ef, c=map, cmap=cmap, s=25, edgecolors='none')

    plt.scatter(0, 10, color='#31688e', label='Nb-d',marker='s')
    plt.scatter(0, 10, color='#e9d91b', label='Ni-d',marker='s')

    plt.axhline(0, linestyle=(0, (5, 5)), linewidth=2, color='k', alpha=0.7)

    for x in norm_kdist:
        plt.axvline(x,  linewidth=2, color='k', alpha=0.7)
    plt.xticks(norm_kdist, k_labels)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', length=5, width=2)
    ax.tick_params(axis='both', which='minor', length=3, width=1.5)
    plt.ylabel('Energy (eV)')
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig(legend + '.png', dpi=300, bbox_inches='tight')
    plt.show()
def Cplot( band_data, Ef, nkstotal, legend, minlabel = 'Min', maxlabel = 'Max' ):
    nKPTS = nkstotal 
    xmin= 0.
    xmax= 1.
    ymin= -1.
    ymax= 1.

    plt.figure( figsize = ( 10, 6 ) )
    colors = ["#31688e", "#e9d91b"]
    print(band_data[:,0].shape)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    map = band_data[ :, 2]
    plt.scatter( band_data[ :, 0 ]/ float(nKPTS-1) , band_data[ :, 1] - Ef, c =map, cmap = cmap, s = 30, edgecolors = 'none')
    plt.scatter(0,10,color='#31688e',label='Nb')
    plt.scatter(0,10,color='#e9d91b',label='Ni')
    plt.axhline(y=0, linewidth=0.5, color='gray')
    kpoints1 = [0.0000, 51., 102., 153., 204., 255., 306., 357., 408.,459. , 512.]
    kpoints = [i/nKPTS for i in kpoints1]

    for i in kpoints:
        plt.axvline(i,linewidth=2, color='k', alpha=0.7)
    kpoint_labels = ['$\Gamma$', 'X', 'W', 'K', '$\Gamma$', 'L', 'U', 'W', 'L', 'K|U', 'X']

    plt.xticks(kpoints, kpoint_labels)
    plt.ylabel(' eV' )
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.legend()
    plt.savefig( legend + '.png', dpi=300, bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    #filename = '/home/caua/Downloads/qe-7.3.1/PP/examples/projected_bands_example/results/pt.kpdos.out'#silicene.kpdos.out'#'/home/caua/pdos.out'
    #
    # filename='/home/caua/pseudo_tests/Ni_testing/pp_ni.out'
    filename = 'NbNiSi_projbands.out'
    output = filename
    
    # Reading output file of a projwfc run
    data = parse( output )

    # number of kpoints
    nkstotal = get_number_of_kpoinst( data )
    print(nkstotal)
    # number of band
    nbnd  = get_number_of_bands( data )
    print(nbnd)
    # List all projection   
    list_projections( data ) 

    # Indices of s projection
    sdos_index =[36,37,38,39,40,49,50,51,52,53,62,63,64,65,66,75,76,77,78,79,88,89,90,91,92,101,102,103,104,105]

    # Sum all projection with indices in sdos_index
    sdos       = sum_states( data, state_index = sdos_index ) 


    #plot( sdos, Ef, nkstotal, legend = 's-projection' )

    # Indices of p projection
    pdos_index =  [111,112,113,114,115,121,122,123,124,125,131,132,133,134,135,141,142,143,144,145,151,152,153,154,155,161,162,163,164,165,171,172,173,174,175,181,182,183,184,185,191,192,193,194,195,201,202,203,204,205,211,212,213,214,215,221,222,223,224,225,231,232,233,234,235,241,242,243,244,245,251,252,253,254,255,261,262,263,264,265]
    # Sum all projection with indices in pdos_index
    pdos       = sum_states( data, state_index = pdos_index )

    #plot( pdos, Ef, nkstotal, legend = 'p-projection' )

    # Calculate sdos /( sdos + pdos )
    ratio = pdos[ :, -1 ] / ( sdos[ :, -1 ] + pdos[ :, -1 ] + 1.e-10 )  #1.e-10 to avoid divide by 0

    # Construct data for plotting
    ratio_data = sdos.copy( )
    ratio_data[ :, -1 ] = ratio

    plot( ratio_data, 17.56, nkstotal, legend = 'Projected bands')

