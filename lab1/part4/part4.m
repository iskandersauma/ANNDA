clear all;
clc;
[Xtr, Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');
Xval = Xval(:,1:1000);
Yval = Yval(:,1:1000);
yval = yval(:,1:1000);

%parametrar
mean = 0;
std = 0.01;
k = size(Ytr,1);
d = size(Xtr,1);
W = mean + randn(k,d)*std;
lambda = 0;
delta = 1;
decay = 0.95;

GDparams = setParams(100,0.01,50);
jtrain = zeros(1,GDparams.epochs);
jtest = zeros(1,GDparams.epochs);
for i = 1:GDparams.epochs
  disp(i)
  jtrain(i) = ComputeCost(Xtr, Ytr, W, lambda, delta);
  jtest(i) = ComputeCost(Xval, Yval, W, lambda, delta);
  W = MiniBatchGD(Xtr, Ytr, W, lambda, GDparams, delta);
  GDparams.eta = decay*GDparams.eta;
end

acc_tr = ComputeAccuracy(Xtr, ytr, W);
acc_te = ComputeAccuracy(Xte, yte, W);
disp(['training accuracy: ' num2str(acc_tr) '%'])
disp(['testing accuracy: ' num2str(acc_te) '%'])

figure()
plot(1:GDparams.epochs,jtrain,'r')
hold on
plot(1:GDparams.epochs,jtest,'b')
hold off
xlabel('epoch')
ylabel('loss')
legend('training loss', 'test loss')
