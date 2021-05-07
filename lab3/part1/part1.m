clear all;
clc;
[Xtr, Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xval, Yval,yval] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');

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

pairs = 20;
eta = 0.02;
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
   acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k);
   disp(['training accuracy:' num2str(acc_tr) '%'])
   acc_te = ComputeAccuracy(Xte, yte, W, b, k);
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

% eta = 0.438;
% lambda = 1.0907e-4;
% GDparams = setParams(batch_size, eta, epochs);
% [W,b,jtrain, jtest] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,k,m);
% 
% acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k);
% disp(['training accuracy:' num2str(acc_tr) '%'])
% acc_te = ComputeAccuracy(Xte, yte, W, b, k);
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