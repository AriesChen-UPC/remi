# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-03-09 9:23 AM
@File     : ReMi.py
@Software : PyCharm
"""

import numpy as np
from math import ceil
from tqdm import tqdm
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class ReMi:
    """
        This is a class for ReMi algorithm.

    """
    class_variable = "This is a ReMi class variable"

    def __init__(self):
        self.instance_variable = "This is an ReMi instance variable"

    def mwindow(self, n, percent=10):

        if np.isscalar(n):
            n = int(n)
        else:
            n = len(n)

        if percent > 50 or percent < 0:
            raise ValueError('Invalid percent for mwindow')

        m = 2 * percent * n // 100
        m = 2 * (m // 2)
        h = np.hanning(m)
        w = np.concatenate((h[:m // 2], np.ones(n - m), h[m // 2:][::-1]))

        return w

    def fftrl(self, s, t, percent=0.0, npad=None):

        if npad is None:
            npad = len(t)
        if npad == 0:
            npad = len(t)

        s = np.asarray(s)
        l, m1 = s.shape
        m2 = 1
        nx = 1
        ny = 1

        itr = 0
        if l == 1 and m2 == 1:
            nsamps = m1
            itr = 1
            s = s.reshape(nsamps, 1)
        elif m1 == 1 and m2 == 1:
            nsamps = l
        elif m1 != 1:
            nsamps = l
            nx = m1
            ny = m2
        else:
            raise ValueError("Unrecognizable data array")

        if nsamps != len(t):
            t = t[0] + (t[1] - t[0]) * np.arange(nsamps)
            if npad is None:
                npad = nsamps

        if percent > 0:
            mw = self.mwindow(nsamps, percent)
            mw = mw[:, np.newaxis]
            s *= mw
            del mw

        spec = np.fft.fft(s, n=npad, axis=0)
        spec = spec[:npad // 2 + 1, ...]
        del s

        fnyq = 1. / (2 * (t[1] - t[0]))
        nf = spec.shape[0]
        df = 2 * fnyq / npad
        f = df * np.arange(nf)
        f = np.reshape(f, (-1, 1))

        if itr:
            f = f.T
            spec = spec.T

        return spec, f

    def ifftrl(self, spec, f):

        spec = spec[:, np.newaxis]
        m, n1 = spec.shape
        n2 = 1
        itr = 0

        if (m - 1) * (n1 - 1) == 0:
            if m == 1:
                spec = spec.T
                itr = 1
            nsamps = len(spec)
            nx = 1
            ny = 1
        else:
            nsamps = m
            nx = n1
            ny = n2

        nyq = 0
        rnyq = np.real(spec[-1])
        inyq = np.imag(spec[-1])
        small = 100 * np.finfo(float).eps
        if rnyq == 0 and inyq == 0:
            nyq = 1
        elif rnyq == 0 and np.abs(inyq) < small:
            nyq = 1
        elif rnyq == 0:
            nyq = 0
        elif np.abs(inyq / rnyq) < small:
            nyq = 1

        if nyq:
            L1 = np.arange(0, nsamps)
            L2 = np.arange(nsamps - 2, 0, -1)
        else:
            L1 = np.arange(0, nsamps)
            L2 = np.arange(nsamps - 1, 0, -1)

        symspec = np.concatenate([spec[L1, :], np.conj(spec[L2, :])], axis=0)
        r = np.real(np.fft.ifft(symspec, axis=0))
        n = r.shape[0]
        df = f[1] - f[0]
        dt = 1 / (n * df)
        t = dt * np.arange(0, n)

        if itr == 1:
            r = r.T
            t = t.T
        r = r.reshape(-1)

        return r

    def tptran(self, seis, t, x, pmin, pmax, dp=None):

        if seis.shape[0] != t.shape[0]:
            raise ValueError('t vector incompatible with seis')
        if seis.shape[-1] != x.shape[0]:
            raise ValueError('x vector incompatible with seis')

        if dp is None:
            dp = 0.5 * (pmax - pmin) / (len(x) - 1)

        nt, nx = seis.shape
        nt2 = 2 ** int(np.ceil(np.log2(nt)))
        if nt2 > nt:
            seis = np.vstack((seis, np.zeros((nt2 - nt, nx))))
            tau = (t[1] - t[0]) * np.arange(nt2)
        else:
            tau = t

        seisf, f = self.fftrl(seis, tau)
        p = np.arange(pmin, pmax + dp, dp)
        np_ = len(p)
        stp = np.zeros((nt2, np_))

        for k in range(np_):
            dtx = p[k] * x
            shiftr = np.exp(1j * 2. * np.pi * f * dtx)
            trcf = np.sum(seisf * shiftr, axis=1)
            trcf[-1] = np.real(trcf[-1])
            stp[:, k] = self.ifftrl(trcf, f)
        stp = stp / np_.real

        return stp, tau, p

    def remi_func(self, n, dn, seis, nw=None, np_num=None, npad=None, percent=None, samplerate=None, loop=None, n_cut=None,
                  n_select=None, pmax=None, pmax1=None, pmin=None, pmin1=None):

        if pmax is None:
            pmax = 1e-2
        if pmin is None:
            pmin = 0
        if pmax1 is None:
            pmax1 = 0
        if pmin1 is None:
            pmin1 = -1e-2

        if loop != 0:
            x = np.arange(0, n_cut * dn, dn)
        else:
            x = np.arange(0, n_select * dn, dn)

        dp = (pmax - pmin) / np_num
        t_window = int(len(seis) / samplerate) / nw
        RF_STACK = np.zeros((int(npad / 2) + 1, int((pmax - pmin) / dp + 1)))

        for j in tqdm(range(nw)):
            delta_data = ceil((len(seis) - 1) / nw)
            seis_windows = seis[j * delta_data:(j + 1) * delta_data]
            t = np.linspace(0, t_window, len(seis_windows))
            t = t[:, np.newaxis]

            stp, tau, p = self.tptran(seis_windows, t, x, pmin1, pmax1, dp)
            spec, f = self.fftrl(stp, tau, percent, npad)
            PF2 = (spec * np.conj(spec))

            stp, tau, p = self.tptran(seis_windows, t, x, pmin, pmax, dp)
            spec, f = self.fftrl(stp, tau, percent, npad)
            PF1 = (spec * np.conj(spec))
            PF = PF1 + np.fliplr(PF2)
            RF = PF
            Sf = np.sum(PF, axis=1) / len(p)

            for i in range(len(Sf)):
                RF[i, :] = RF[i, :] / Sf[i]
            RF_STACK = RF_STACK + RF

        return RF_STACK, f, p

    def extract_disp(self, data_path, fplot, cplot, Aplot, fmin, fmax):

        max_value_peak_index = np.zeros((len(Aplot[:, 0]), 1))
        for i in range(len(Aplot[:, 0])):
            peaks, _ = find_peaks(Aplot[i, :], prominence=0.25)
            valleys, _ = find_peaks(-Aplot[i, :])
            if len(peaks) == 0:
                max_value_peak_index[i, 0] = np.nan
            elif len(peaks) == 1:
                if Aplot[i, peaks] < np.mean(Aplot[i, :]):
                    max_value_peak_index[i, 0] = np.nan
                else:
                    max_value_peak_index[i, 0] = peaks[0]
            else:
                A_value = Aplot[i, peaks]
                A_value_max = np.max(A_value)
                if A_value_max < np.mean(Aplot[i, :]):
                    max_value_peak_index[i, 0] = np.nan
                else:
                    A_value_max_index = np.where(A_value == A_value_max)
                    for j in range(len(A_value)):
                        if A_value[j] > A_value_max * 0.75 and peaks[A_value_max_index] - peaks[j] <= 30:
                            max_value_peak = peaks[A_value_max_index]
                            for k in range(len(valleys)):
                                if valleys[k] > max_value_peak:
                                    valley_index = valleys[k]
                                    break
                            if valley_index - peaks[j] > 30:
                                max_value_peak_index[i, 0] = max_value_peak + 15
                            else:
                                max_value_peak_index[i, 0] = round(np.mean([peaks[j], valley_index]))
                            break
                        else:
                            if A_value[j] > A_value_max * 0.75:
                                max_value_peak = peaks[j]
                                for k in range(len(valleys)):
                                    if valleys[k] > max_value_peak:
                                        valley_index = valleys[k]
                                        break
                                if valley_index - peaks[j] > 30:
                                    max_value_peak_index[i, 0] = max_value_peak + 15
                                else:
                                    max_value_peak_index[i, 0] = round(np.mean([peaks[j], valley_index]))
                                break

        p_value = np.zeros((len(Aplot[:, 0]), 1))
        p = cplot[0, :]
        for i in range(len(Aplot[:, 0])):
            if np.isnan(max_value_peak_index[i, 0]):
                if i < 5:
                    max_value_peak_index[i, 0] = np.nanmean(max_value_peak_index[:, 0])
                else:
                    max_value_peak_index[i, 0] = np.mean(max_value_peak_index[i - 5:i, 0])
            p_value[i, 0] = p[int(max_value_peak_index[i, 0])]

        plt.pcolormesh(fplot, cplot, Aplot, cmap='RdYlBu_r', shading='gouraud')
        plt.plot(fplot[:, 0], p_value, 'o', markersize=1, markerfacecolor='r', markeredgecolor='r')
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.xlim(fmin, fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Slowness (s/m)')
        plt.gca().xaxis.set_label_position('top')
        cbar = plt.colorbar()
        plt.clim(0, 90)
        cbar.set_label('ReMi(TM) Spectral Ratio')
        plt.savefig(os.path.join(data_path, 'ReMi(TM) Spectral Ratio.png'))
        plt.show()

        p_value = p_value[:, 0]
        p_value_filter = savgol_filter(p_value, 15, 3, mode='nearest')
        v_value = 1 / p_value_filter
        plt.plot(fplot[:, 0], v_value, color="#0072BD", linewidth=2)
        plt.xlim(fmin, fmax)
        plt.ylim(100, 1000)
        plt.title('Dispersion curve')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Velocity (m/s)')
        plt.grid()
        plt.savefig(os.path.join(data_path, 'Dispersion curve.png'))
        np.savetxt(os.path.join(data_path, 'Dispersion curve.csv'), np.vstack((fplot[:, 0], v_value)).T,
                   delimiter=',', fmt='%f', header='Frequency (Hz), Velocity (m/s)', comments='')
        plt.show()
