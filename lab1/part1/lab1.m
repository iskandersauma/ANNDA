clear all;
clc;
[Xtr, Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xval,Yval,yval] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');

%parametrar
mean = 0;
std = 0.01;
K = size(Ytr,1);
d = size(Xtr,1);
W = mean + randn(K,d)*std;
b = mean + randn(K,1)*std;
lambda = 0;
batch_size = 50;

GDparams = setParams(20,0.1,40);
jtrain = zeros(1,GDparams.epochs);
jtest = zeros(1,GDparams.epochs);
%training
for i = 1:GDparams.epochs
    disp(i)
    jtrain(i) = ComputeCost(Xtr,Ytr,W,b,lambda);
    jtest(i) = ComputeCost(Xval,Yval,W,b,lambda);
    [W,b] = MiniBatchGD(Xtr, Ytr, W, b, lambda,GDparams);
end

%accuracy
acc_tr = ComputeAccuracy(Xtr,ytr,W,b);
disp(['training accuracy: ' num2str(acc_tr)])
acc_test = ComputeAccuracy(Xte,Yte,W,b);
disp(['test accuracy: ' num2str(acc_test)])

figure()
plot(1:GDparams.epochs,jtrain,'r')
hold on
plot(1:GDparams.epochs,jtest,'b')
hold off
xlabel('epochs');
ylabel('loss');
legend('training loss','validation loss');


for i = 1:K
   im = reshape(W(i,:),32,32,3);
   s_im{i} = (im-min(im(:)))/(max(im(:)) - min(im(:)));
   s_im{i} = permute(s_im{i},[2,1,3]);
end
figure()
montage(s_im,'size',[1,K])











