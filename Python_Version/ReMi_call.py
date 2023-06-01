# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-03-03 4:02 PM
@File     : Remi_call.py
@Software : PyCharm
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from obspy import read
from ReMi import ReMi
import matplotlib.pyplot as plt

# %% Read sg2 data

print('Please select the folder containing the .sg2 files\n------------')
root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory()  # all .sg2 files should be in the same folder and sorted by order
File = os.listdir(data_path)
FileNames = [f for f in File if f.endswith('.sg2')]
n = len(FileNames)

# Read the sg2 file
for i in range(n):
    trace_temp = read(os.path.join(data_path, FileNames[i]))
    for j in range(len(trace_temp)):
        # Change the j to two digits
        if j < 10:
            trace_temp[j].write(os.path.join(data_path, FileNames[i][:-4] + '_0' + str(j) + '.sac'), format='SAC')
        else:
            trace_temp[j].write(os.path.join(data_path, FileNames[i][:-4] + '_' + str(j) + '.sac'), format='SAC')

# %% Read data

print('Please select the folder containing the .sac files\n------------')
root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory()  # all .sac files should be in the same folder and sorted by order
File = os.listdir(data_path)
FileNames = [f for f in File if f.endswith('.sac')]
n = len(FileNames)
tr = read(os.path.join(data_path, FileNames[0]))
start_time = tr.traces[0].stats.starttime
end_time = tr.traces[0].stats.endtime
N = len(tr.traces[0].data)
samplerate = round(1/tr.traces[0].stats.delta)

desample = 1  # 0: no desample; 1: desample
k = 2  # desample factor
if desample != 0:
    seis_all = np.zeros((int(N/k), n))
    for i in range(n):
        tr = read(os.path.join(data_path, FileNames[i]))
        tr.resample(samplerate/k)
        seis_all[:, i] = tr.traces[0].data
    samplerate = samplerate/k
else:
    seis_all = np.zeros((N, n))
    for i in range(n):
        tr = read(os.path.join(data_path, FileNames[i]))
        seis_all[:, i] = tr.traces[0].data
print(f'File path: {data_path}\nNumber of traces: {n}')
print(f'Sampling rate: {samplerate}Hz\nStart time: {start_time}\nEnd time: {end_time}\n------------')

# %% Parameters

dn = 2  # distance between traces
n0 = 0  # start position
n_select = 10  # number of traces to be selected
loop = 0  # 0: no loop; 1: loop
n_cut = 9  # number of traces to be cut
loop_times = n_select - n_cut + 1  # number of loops
np_num = 200  # number of slowness
npad = 4000  # number of frequency
percent = 5  # percent of taper
nw = 20  # number of windows

# %% Remi

remi = ReMi()
if loop != 0:
    if loop_times + n_cut > n + 1:
        raise Exception('Loop position exceeds maximum node position')
    for n0 in range(loop_times):
        seis = seis_all[:, n0:n0+n_cut]
        print('Start ReMi calculation...')
        RF_STACK, f, p = remi.remi_func(n, dn, seis, nw, np_num, npad, percent, samplerate, loop, n_cut, n_select)
        fig = plt.figure()
        RF_STACK = np.real(RF_STACK)
        p_mesh, f_mesh = np.meshgrid(p, f)
        plt.pcolormesh(f_mesh, p_mesh, RF_STACK, cmap='RdYlBu_r', shading='gouraud')
        plt.xlim([5, 30])
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Slowness (s/m)')
        plt.gca().xaxis.set_label_position('top')
        cbar = plt.colorbar()
        plt.clim(0, 90)
        cbar.set_label('ReMi(TM) Spectral Ratio')
        plt.savefig(os.path.join(data_path, f'fig_0{n0}.png'))
        plt.show()
else:
    seis = seis_all[:, n0:n0+n_select]
    print('Start ReMi calculation...')
    RF_STACK, f, p = remi.remi_func(n, dn, seis, nw, np_num, npad, percent, samplerate, loop, n_cut, n_select)
    fig = plt.figure()
    RF_STACK = np.real(RF_STACK)
    p_mesh, f_mesh = np.meshgrid(p, f)
    plt.pcolormesh(f_mesh, p_mesh, RF_STACK, cmap='RdYlBu_r', shading='gouraud')
    plt.xlim([5, 30])
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Slowness (s/m)')
    plt.gca().xaxis.set_label_position('top')
    cbar = plt.colorbar()
    plt.clim(0, 90)
    cbar.set_label('ReMi(TM) Spectral Ratio')
    plt.savefig(os.path.join(data_path, f'fig_0{n0}.png'))
    plt.show()
print('ReMi calculation finished!\n------------')

# %% Extract dispersion curve

# Find max value of ReMi method
# print('Start extracting dispersion curve...')
# fmin = 5
# fmax = 30
# num_search = np.where(p == 0.006)[0][0]
# no_fmin = np.argmin(np.abs(f[:, 0] - fmin))
# no_fmax = np.argmin(np.abs(f[:, 0] - fmax))
# # Select data corresponding to frequnecy range [fmin,fmax]
# Aplot = RF_STACK[no_fmin:no_fmax + 1, :]
# fplot = f_mesh[no_fmin:no_fmax + 1, :]
# cplot = p_mesh[no_fmin:no_fmax + 1, :]

# remi.extract_disp(data_path, fplot, cplot, Aplot, fmin, fmax, num_search)
# print('Dispersion curve extraction finished!')

# Find peaks of ReMi method
print('Start extracting dispersion curve...')
fmin = 5
fmax = 30
no_fmin = np.argmin(np.abs(f[:, 0] - fmin))
no_fmax = np.argmin(np.abs(f[:, 0] - fmax))
# Select data corresponding to frequnecy range [fmin,fmax]
Aplot = RF_STACK[no_fmin:no_fmax + 1, :]
fplot = f_mesh[no_fmin:no_fmax + 1, :]
cplot = p_mesh[no_fmin:no_fmax + 1, :]
remi.extract_disp(data_path, fplot, cplot, Aplot, fmin, fmax)

print('Dispersion curve extraction finished!')
