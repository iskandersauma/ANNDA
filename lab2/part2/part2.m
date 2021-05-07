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
Eta = [];
Lambda = [];
acc_val = [];

trials = 5;
bootstrap = 10000;
eta = 0.0622;
lambda = 1.6424e-6;
GDparams = setParams(batch_size,eta,epochs);
acc = zeros(1,trials);
P = zeros(size(Ytr,1),bootstrap,trials);
decay = 0.95;

for i = 1:trials
   disp(i)
   index = randperm(size(Xtr,2),bootstrap);
   Xtr_i = Xtr(:,index);
   Ytr_i = Ytr(:,index);
   [W,b,jtrain, jtest, flag] = training(Xtr_i,Ytr_i,Xval,Yval,GDparams, lambda,m);
   [acc_i,prob_i] = ComputeAccuracy(Xte,yte,W,b);
   acc(i) = acc_i;
   P(:,:,i) = prob_i;
   GDparams.eta = decay*GDparams.eta;
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