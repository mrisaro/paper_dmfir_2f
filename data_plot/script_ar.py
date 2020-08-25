#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:13:04 2019
Script to analyze ARgon spectrum
@author: tute
"""
#%% Load the libraries I need

import numpy as np
import matplotlib.pyplot as plt
import auxiliar as aux
import os
from scipy.signal import find_peaks
from scipy import optimize
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib.backends.backend_pdf import PdfPages
my_dpi = 72

#%% Analyzing the first experiment
#root  = '/home/tute/Dropbox/Matias/Laboratorio/019/data_sif4_uv/'
root  = '/home/matias/Dropbox/Matias/Laboratorio/2019/data_ar_uv/'    
#root  = '/home/matias/Dropbox/Matias/Laboratorio/2019/data_test/'
#direc = 'sif4_UV_11'
direc = 'Ar_UV_11'
#direc = 'AR_lamp_UV_1'
#direc = 'C6H8Si_test_UV_1'

path  = root+direc+'/'
 
#pks = np.array([14,28,32,47,66,85])
pks = np.array([12,16,28,40])
Elaser = []
Ipk = []
Wpk = []
tqs = []
tap = []
p_calib = []

for ii in range(14):
    file_form = 'datos'+str(ii+1)+'.mat'
    file = path + file_form
    file_data = path + 'datos'+str(ii+1)+'_read.mat'
    print(file)
    if os.path.exists(file_data):
        data = loadmat(file_data)
        ipk, wpk = aux.fun_bunch_pk(data['masa'][0],data['sp'][0],pks)
        Ipk.append(ipk)
        Wpk.append(wpk)
        [_,_,_,_,t,BNC,Euv] = aux.fun_load_mat(file)
        Elaser.append(Euv[0][0])
        p_calib.append(data['p'].flatten())
        [_,y2,_,y4,t,BNC,Euv] = aux.fun_load_mat(file)
        tap.append(BNC['ch1'][0][0][0][0])          
    else:
        [_,y2,_,y4,t,BNC,Euv] = aux.fun_load_mat(file)
        sp = np.mean(y4,0)
        masa, kt, p = aux.fun_mass_calib(t,sp)
        sp_mean = np.mean(y4[:,kt[0]],0)
        ipk,wpk = aux.fun_bunch_pk(masa,sp_mean,pks)
        print(ipk)
        Ipk.append(ipk)
        Wpk.append(wpk)
        aux.fun_gen_file(masa,sp_mean,p,file_data)
        Elaser.append(Euv[0][0])
        p_calib.append(p.flatten())
    
    tqs.append(BNC['ch2'][0][0][0][0])
    
#%%
ind = np.argsort(Elaser)
E266 = np.array(np.sort(Elaser))
Ipk = np.array(Ipk)[ind]
Ipk = np.array(Ipk)
Wpk = np.array(Wpk)
pp = np.array(p_calib)
#%% Plot differents ions
fig = plt.figure(1,figsize=(1000/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
ax.loglog(E266,Ipk[:,3]*1e3,'o',markersize=8, label = r'Ar$^{+}$ Amp')
ax.loglog(E266,Ipk[:,2]*1e3,'o',markersize=8, label = r'N$_{2}^{+}$ Amp')
ax.loglog(E266,Ipk[:,1]*1e3,'o',markersize=8, label = r'O$^{+}$ Amp')
#ax.loglog(E266,Ipk[:,1]*1e3,'-o',markersize=8, label = r'Si$^{+}$ Amp')
ax.set_xlabel(r'E$_{UV}$ (mJ)', fontsize=18)
ax.set_ylabel(r'V$_{sp}$ (mV)', fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(loc='best',fontsize=20)
ax.grid(linestyle='--',which='minor')
fig.tight_layout()

fig_file_png = path+'ion_energy_laser.png'
fig_file_pdf = path+'ion_energy_laser.pdf'
plt.savefig(fig_file_png)
plt.savefig(fig_file_pdf)

#%% Power law model for the amounts of ions

def pow_func(x, a, b, phi_s):
    return a * 1/(1+(phi_s/x)**b)

def log_func(x, c0, c1, c2):                     # ... and the log of it
   return np.log(pow_func(x, c0, c1, c2))

qp = np.arange(len(E266))

p40, p40_cov = optimize.curve_fit(log_func, E266[qp], np.log(Ipk[qp,3]), 
                                      p0=[0.5,3,12])

e_fit = np.logspace(-1,1.6,100)
I40_fit = pow_func(e_fit, p40[0],p40[1],p40[2])

fig = plt.figure(4,figsize=(1000/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
ax.loglog(E266[qp],Ipk[qp,3]*1e3,'o',markersize=10, 
          label = r'Ar$^{+}$')
ax.loglog(e_fit,I40_fit*1e3,'-',color='r', label = r'Ar$^{+}$ fit')
#ax.loglog(E266[qp],Ipk[qp,4]*1e3,'o',markersize=8, 
#          label = r'SiF$_{2}^{+}$')
#ax.loglog(e_fit,I66_fit*1e3,'--',color='r', label = r'SiF$_{2}^{+}$ fit')
#ax.loglog(E266[qp],Ipk[qp,3]*1e3,'o',markersize=8, 
#          label = r'SiF$^{+}$ Amp')
#ax.loglog(e_fit,I47_fit*1e3,'-.',color='r', label = r'SiF$^{+}$ fit')
ax.set_xlabel(r'E$_{UV}$ (mJ)', fontsize=18)
ax.set_ylabel(r'V$_{sp}$ (mV)', fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(loc='best',fontsize=20)
ax.grid(linestyle='--',which='minor')
ax.grid(linestyle='--',which='major')
fig.tight_layout()

fig_file_png = path+'power_law_fit.png'
fig_file_pdf = path+'power_law_fit.pdf'
plt.savefig(fig_file_png)
plt.savefig(fig_file_pdf)

#%% Save txt file with fit parameters.
file_name = path+'fit_parameters.csv'
param = np.array([p40,np.sqrt(np.diag(p40_cov))])
np.savetxt(file_name, param.T, header="p,cov", fmt="%s",
					delimiter=',', comments='')

#%%
#tap = np.arange(160,310,10)
tap = np.array([140,145,150,155,160,170,180,190,200,210,220,230,240,250])
del_t = p[0]*0.5/np.sqrt(40)*Wpk[:,3]*1e9

fig = plt.figure(3,figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
ax.plot(tap,Ipk[:,3],'-o',markersize = 10,label=r'Ar$^{+}$ pk')
ax.set_xlabel(r't$_{ap}$ ($\mu$s)', fontsize=18)
ax.set_ylabel(r'V$_{pk}$ (V)', fontsize=18)
ax.tick_params(labelsize=14)
ax2 = ax.twinx()
ax2.plot(tap[1:],del_t[1:],'--o',markersize=10,color='C1',
         label=r'Ar$^{+}$ $\delta$t')
ax2.set_ylim(10,80)
ax2.set_ylabel(r'$\delta t$ (ns)', fontsize=18)
ax2.tick_params(labelsize=14)
ax.legend(loc='best',fontsize=20)
ax2.legend(loc=6,fontsize=20)
ax.grid(linestyle='--')
fig.tight_layout()

#%%
file_dat5 = path + 'datos'+str(5)+'_read.mat'
data5 = loadmat(file_dat5)
file_dat10 = path + 'datos'+str(10)+'_read.mat'
data10 = loadmat(file_dat10)

fig = plt.figure(4,figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
#ax.plot(data5['masa'][0],data5['sp'][0]*13,'-',linewidth=3.0,
#        label=r'Ar$^{+}$ (t$_{QS}$ = 200$\mu$s)')
ax.plot(data10['masa'][0],data10['sp'][0]*1.7,'-',linewidth=1.0,
        label=r'Ar$^{+}$ (t$_{QS}$ = 250$\mu$s)')
#ax.plot(data['masa'][0],data['sp'][0],'--',linewidth=3.0,
#    label=r'Ar$^{+}$ (t$_{QS}$ = 300$\mu$s)')
ax.set_xlabel(r'Masa (uma)', fontsize=18)
ax.set_ylabel(r'V$_{det}$ (V)', fontsize=18)
ax.set_xlim(35,39)
ax.tick_params(labelsize=14)
ax.legend(loc='best',fontsize=20)
ax.grid(linestyle='--')
fig.tight_layout()    

#%% figures of signal and delay


fig = plt.figure(5,figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
#ax.plot(tap,Ipk[:,2],'-o',markersize=10,label=r'O$^{+}$')
ax.plot(tap,Ipk[:,3],'-o',markersize=10,label=r'Ar$^{+}$')
ax.set_xlabel(r't$_{ret}$ (ms)', fontsize=18)
ax.set_ylabel(r'V$_{sp}$ (V)', fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(loc='best',fontsize=20)
ax.grid(linestyle='--',which='minor')
ax.grid(linestyle='--',which='major')
fig.tight_layout()


#%% Figures parker valve

fig = plt.figure(6,figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
ax  = fig.add_subplot(111)
ax.plot(data2['masa'][0],data2['sp'][0],label=r'P.V. Off')
ax.plot(data1['masa'][0],data1['sp'][0],label=r'P.V. On')
ax.set_xlim(10,45)
ax.set_xlabel(r'Masa (uma)', fontsize=18)
ax.set_ylabel(r'V$_{tof}$', fontsize=18)
ax.tick_params(labelsize=14)
ax.grid(linestyle='--')
inner_ax = fig.add_axes([0.2, 0.25, 0.45, 0.45]) # x, y, width, height
inner_ax.plot(data2['masa'][0],data2['sp'][0])
inner_ax.plot(data1['masa'][0],data1['sp'][0])
inner_ax.set(title='Zoom In', xlim=(10, 20), ylim=(-.01, .08))
inner_ax.grid(linestyle='--')
ax.legend(loc='best',fontsize=20)
plt.tight_layout()

