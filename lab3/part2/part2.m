clear all;
clc;
[Xtr1, Ytr1,ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2,ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3, Ytr3,ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4, Ytr4,ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5, Ytr5,ytr5] = LoadBatch('data_batch_5.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');

Xtr = [Xtr1, Xtr2, Xtr3, Xtr4, Xtr5];
Ytr = [Ytr1, Ytr2, Ytr3, Ytr4, Ytr5];
ytr = [ytr1, ytr2, ytr3, ytr4, ytr5];
Xval = Xtr2;
Yval = Ytr2;
yval = ytr2;

%pre-processing
mean_x = mean(Xtr,2);
Xtr = Xtr - repmat(mean_x,[1,size(Xtr,2)]);
Xval = Xval - repmat(mean_x,[1,size(Xval,2)]);
Xte = Xte - repmat(mean_x,[1,size(Xte,2)]);

%parametrar
k = 3;
m = [50 30];
epochs = 50;
batch_size = 100;

pairs = 50;
eta = 0.02;
decay = 0.95;
lambda_max = 0.01;
lambda_min = 1e-7;
l = log10(lambda_min) + (log10(lambda_max)-log10(lambda_min))*rand(1,1);
lambda = 10.^(-l);

for i = 1:pairs
   disp(i)

   GDparams = setParams(batch_size, eta, epochs);
   [W,b,jtrain, jtest] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,k,m);
   if mod(i,10) == 0
       GDparams.eta = GDparams.eta*decay;
   end
%    figure()
%    plot(1:GDparams.epochs,jtrain,'r')
%    hold on
%    plot(1:GDparams.epochs,jtest,'b')
%    hold off
%    xlabel('epoch');
%    ylabel('loss');
%    legend('training loss', 'testing loss')
end

acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k);
disp(['training accuracy:' num2str(acc_tr) '%'])
acc_te = ComputeAccuracy(Xte, yte, W, b, k);
disp(['test accuracy:' num2str(acc_te) '%'])

