clear all;
clc;
[Xtr, Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2,ytr2] = LoadBatch('data_batch_2.mat');
% [Xtr3, Ytr3,ytr3] = LoadBatch('data_batch_3.mat');
% [Xtr4, Ytr4,ytr4] = LoadBatch('data_batch_4.mat');
% [Xtr5, Ytr5,ytr5] = LoadBatch('data_batch_5.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');

Xval = Xtr2(:,1:1000);
Yval = Ytr2(:,1:1000);
yval = ytr2(:,1:1000);

%pre-processing
mean_x = mean(Xtr,2);
Xtr = Xtr - repmat(mean_x,[1,size(Xtr,2)]);
Xval = Xval - repmat(mean_x,[1,size(Xval,2)]);
Xte = Xte - repmat(mean_x,[1,size(Xte,2)]);

%parametrar
m = 50;
epochs = 10;
batch_size = 100;

pairs = 50;
eta = 0.2;
decay = 0.95;
lambda_max = 0.1;
lambda_min = 1e-7;
l = log10(lambda_min) + (log10(lambda_max)-log10(lambda_min))*rand(1,1);
lambda = 10.^(-l);

for i = 1:pairs
   disp(i)
   GDparams = setParams(batch_size, eta, epochs);
   [W,b,jtrain, jtest, ~] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,m);
   if mod(i,10) == 0
       GDparams.eta = GDparams.eta*decay;
   end
   acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
   disp(['training accuracy:' num2str(acc_tr) '%'])
   acc_te = ComputeAccuracy(Xte, yte, W, b);
   disp(['test accuracy:' num2str(acc_te) '%'])
   
   figure()
   plot(1:GDparams.epochs,jtrain,'r')
   hold on
   plot(1:GDparams.epochs,jtest,'b')
   hold off
   xlabel('epoch');
   ylabel('loss');
   legend('training loss', 'testing loss')
end

%% if lazy run 
% eta = 0.0622;
% lambda = 1.6424e-6;
% GDparams = setParams(100, eta, epochs);
% [W,b,jtrain, jtest, flag] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,m);

% acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
% disp(['training accuracy:' num2str(acc_tr) '%'])
% acc_te = ComputeAccuracy(Xte, yte, W, b);
% disp(['test accuracy:' num2str(acc_te) '%'])
% 
% figure()
% plot(1:GDparams.epochs,jtrain,'r')
% hold on
% plot(1:GDparams.epochs,jtest,'b')
% hold off
% xlabel('epoch');
% ylabel('loss');
% legend('training loss', 'testing loss')