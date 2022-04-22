function [ RF_STACK,f,p ] = Remi_func( n,dn,seis,nw,np,npad,percent,samplerate,loop,n_cut,n_select,pmax,pmax1,pmin,pmin1)
% REMI_FUNC 利用Remi方法提取频散曲线

%   [ RF_STACK,f,p ] = Remi_func( n,dn,seis,nw,np,npad,percent,samplerate,loop,n_cut,pmax,pmax1,pmin,pmin1 )
%   [ RF_STACK,f,p ] = Remi_func( n,dn,seis,nw,np,npad,percent,samplerate,loop,n_cut)
%   [ RF_STACK,f,p ] = Remi_func( n,dn,seis,nw,samplerate)

%  ******** 输入参数说明*********
% n= 检波器数目
% dn= 检波器间距（m）
% seis= 原始数据矩阵
% nw= 数据开窗数目
% np= 慢度搜索点数
% npad= FFT计算点数
% percent= FFT开窗长度百分比
% samplerate= 原始数据采样率
% loop= 是否循环计算 1代表是，0代表否，default=0
% n_cut= 参与循环计算的节点数目

% pmax= 正方向慢度最大值  default=1e-2;
% pmin= 正方向慢度最小值  default=1e-3;
% pmax1= 负方向慢度最大值 default=-1e-3;
% pmin1= 负方向慢度最小值 default=-1e-2;

% ***********输出参数说明*************
% RF_STACK=全部窗口叠加后的能谱比值矩阵
% f=频率向量
% p=慢度向量

% By Y Zheng, June 2021

%%

if nargin<14
    pmax = 1e-2;  
    pmin = 0;  % 正方向慢度最小值/最大值
    pmax1 = 0;  
    pmin1 = -1e-2;  % 负方向慢度最小值/最大值
end

if nargin < 10
    np = 100;
    npad = 1000;
    percent = 10;
    loop = 0;  % 慢度搜索点数； FFT计算点数；FFT窗口重叠百分比; 是否循环计算
end

if loop ~= 0        % 判断是否进行循环计算
    x = 0:dn:(n_cut-1)*dn;    % 要进行循环计算，节点位置矢量按选取的节点数目计算
else
    x = 0:dn:(n_select-1)*dn;        % 不进行循环计算，节点位置矢量按全节点计算
end

dp = (pmax-pmin)/np;  %计算慢度搜索间隔，该参数直接影响能谱矩阵的第二个维度
t_window = round(length(seis)/samplerate)/nw;  %计算每个窗口时间长度
RF_STACK = zeros(round(npad/2)+1,round((pmax-pmin)/dp+1));  % 预分配叠加后矩阵
h1 = waitbar(0,'please wait');  % 设置进度条

for j = 1:nw  % 窗口叠加计算
    delta_data = floor((length(seis-1))/nw);  % 计算每个窗口数据长度，20表示开20个窗
    seis_windows = seis((j-1)*delta_data+1:j*delta_data,:);
    t = linspace(0,t_window,length(seis_windows));  % 单道数据时间矢量
    t = t';
    %
    [stp,tau,p] = tptran(seis_windows,t,x,pmin1,pmax1,dp);  % 调用Crews tau-p 变换函数，计算负方向Tau-p tran
    [spec,f] = fftrl(stp,tau,percent,npad);  % 调用Crews fft
    PF2 = (spec.*conj(spec));  % 矩阵元取模长
    [stp,tau,p] = tptran(seis_windows,t,x,pmin,pmax,dp);  % 调用Crews tau-p 变换函数,计算正方向Tau-p tran
    [spec,f] = fftrl(stp,tau,percent,npad);  % 调用Crews fft
    PF1 = (spec.*conj(spec));
    PF = PF1+fliplr(PF2);  % 正负两个方向能谱叠加
    RF = PF;
    Sf = (1/length(p))*sum(PF,2);  % 能谱矩阵归一化系数计算

    for i = 1:length(Sf)
        RF(i,:) = RF(i,:)/Sf(i,1);  % 利用能谱归一化系数计算 能谱比
    end
    RF_STACK = RF_STACK+RF;  % 计算所有窗口叠加后的能谱比
    str = ['Progress...',num2str(round(j/(nw)*100)),'%'];  % 进度计算
    waitbar(j/(nw),h1,str)
end
delete(h1);
end

