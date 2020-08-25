import numpy as np
import matplotlib.pyplot as plt
import auxiliar as aux
import os
from scipy.signal import find_peaks
from scipy import optimize
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

# define some Functions
def fun_find_pk(t,sp):
    mx = np.max(sp)
    q1,_ = np.where(t<1e-6)
    sp[q1] = np.random.normal(np.mean(sp[q1]), 0.05e-3, len(q1))
    locs,_ = find_peaks(sp,distance=500, height = [mx/30,mx])
    pk_sort = np.argsort(sp[locs])
    pk_sort = pk_sort[::-1]
    locs = locs[pk_sort[0:5]]

    return locs

def fun_mass_calib(t,sp):

    locs = fun_find_pk(t,sp)
    my_dpi = 72
    plt.figure(1,figsize=(1000/my_dpi, 600/my_dpi), dpi=my_dpi)
    plt.plot(t*1e6,sp)
    plt.plot(t[locs]*1e6,sp[locs],'*',markersize=10)
    plt.xlim(0,20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.pause(1)
    ar = list(map(int,input('Que masas son? (en orden):  ').split()))
    plt.close('1')
    p = np.polyfit(np.sqrt(ar),t[locs],1)

    kt = np.where(t>p[1])
    masa = ((t[kt]-p[1])/p[0])**2

    # check that the calibration works
    fun_plot_mass(masa,sp[kt[0]])

    plt.pause(0.5)
    comm = input('Lacalibración es correcta? (si/no):  ')
    if comm == 'si':
        plt.close('all')
        return masa, kt, p
    elif comm == 'no':
        plt.close('all')
        plt.pause(0.5)
        fun_mass_calib(t,sp)

    return masa, kt, p

file  = 'canales.mat'

data = loadmat(file)
sp = np.mean(data['yy4'],0)
masa, kt, p = aux.fun_mass_calib(data['t'],sp)

fig = plt.figure(5,figsize=(8,6))
ax  = fig.add_subplot(111)
ax.plot(masa,savgol_filter(np.mean(data['yy4'][3:55:2+1,kt[0]],axis=0),15,2),
label='IR+UV',color='C1')
ax.plot(masa,savgol_filter(np.mean(data['yy4'][3:55:2,kt[0]],axis=0),15,2),
label='UV',color='C0')
ax.set_xlim(-0.2,105)
ax.set_xlabel(r'Mass (m/Z)',fontsize=14)
ax.set_ylabel(r'TOF-MS signal (V)',fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
ax.grid(linestyle='--')

axins = ax.inset_axes([0.5, 0.45, 0.35, 0.35])
axins.plot(masa,savgol_filter(np.mean(data['yy4'][3:55:2+1,kt[0]],axis=0),15,2),
color='C1')
axins.plot(masa,savgol_filter(np.mean(data['yy4'][3:55:2,kt[0]],axis=0),15,2),
color='C0')
# sub region of the original image
x1, x2, y1, y2 = 45, 50, -0.001, 0.04
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticklabels('')
axins.grid(linestyle='--')
ax.indicate_inset_zoom(axins)

plt.rc('text', usetex=True)
plt.tight_layout()
plt.show()
