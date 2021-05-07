clear all;
clc;
[Xtr, Ytr,ytr] = LoadBatch('data_batch_1.mat');
% [Xtr2, Ytr2,ytr2] = LoadBatch('data_batch_2.mat');
% [Xtr3, Ytr3,ytr3] = LoadBatch('data_batch_3.mat');
% [Xtr4, Ytr4,ytr4] = LoadBatch('data_batch_4.mat');
% [Xtr5, Ytr5,ytr5] = LoadBatch('data_batch_5.mat');
[Xval,Yval,yval] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');
Xval = Xval(:,1:1000);
Yval = Yval(:,1:1000);
yval = yval(:,1:1000);

%parametrar
lambda = 0;
decay = 0.95;
trials = 10;
GDparams = setParams(100,0.01,100);
mean = 0;
std = 0.01;
k = size(Ytr,1);
d = size(Xtr,1);
W = mean + randn(k,d)*std;
b = mean + randn(k,1)*std;

for i = 1:trials
   disp(i)
   [W,b,jtrain,jtest] = boosting(Xtr, Ytr, Xval,Yval,GDparams,lambda, W, b);
   GDparams.eta = decay*GDparams.eta;
   
   figure()
   plot(1:GDparams.epochs,jtrain,'r')
   hold on
   plot(1:GDparams.epochs,jtest,'b')
   hold off
   xlabel('epoch');
   ylabel('loss');
   legend('training loss', 'testing loss')
end

%accuracy
[X,~,y] = LoadBatch('test_batch.mat');
acc = ComputeAccuracy(X,y,W,b)