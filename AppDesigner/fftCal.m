function [f, P1] = fftCal(signalData, Fs)
%====================================================
% Plot the fft result of target signal.
%
% fftPlot(signalData,Fs)
% 	signalData: The target signal data.
%   Fs: Sampling rate.
%
% Example:
% 	fftPlot(data,Fs)
%
% Email: s15010125@s.upc.edu.cn
% Author: LijiongChen
%
% log:
% 2021-08-04: Complete
%====================================================

N = length(signalData);
sig_fft = fft(signalData);
P2 = abs(sig_fft/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);  
f = Fs*(0:(N/2))/N;

% figure()
% plot(f,P1);
% legend();
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('Frequency (Hz)')
% ylabel('|P1(f)|')
% grid on
end