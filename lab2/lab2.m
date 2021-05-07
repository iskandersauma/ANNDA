clear all;
clc;

filename = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};
X = [];
Y = [];
y = [];
N = 0;
for i = 1:5
    disp(i)
   [XX,YY,yy,NN,hyper_parameters.k,hyper_parameters.d] = LoadBatch(filename{i});
   X = [X XX];
   Y = [Y YY];
   y = [y yy];
   N = N + NN;
end

hyper_parameters.a          = 0.001; % variance
hyper_parameters.eta        = 0.02; % learning rate
hyper_parameters.lambda     = 0.001; % regularization rate
hyper_parameters.batch_size    = 500; % number of batches
hyper_parameters.epochs   = 60; % number of epoches
hyper_parameters.decay_rate = 0.95;
hyper_parameters.m          = 100; % hidden layer.
hyper_parameters.rho        = 0.9; % momentum parameter.

W1 = hyper_parameters.a.*randn(hyper_parameters.m,hyper_parameters.d);
b1 = zeros(hyper_parameters.m,1);
W2 = hyper_parameters.a.*randn(hyper_parameters.k,hyper_parameters.m);
b2 = zeros(hyper_parameters.k,1);
j_train = zeros(hyper_parameters.epochs,1);
j_validation = zeros(hyper_parameters.epochs,1);

split = N-1000;
X_train = X(:,1:split);
X_validation = X(:,split+1:N);
Y_train = Y(:,1:split);
Y_validation = Y(:,split+1:split);
N = split;

for i = 1:20
   disp(i)
   [W1,b1,W2,b2] = MiniBatchGD(X_train,Y_train,W1,b1,W2,b2,hyper_parameters);
   j_train(i) = ComputeCost(X_train,Y_train,W1,b1,W2,b2,hyper_parameters.lambda,hyper_parameters.k);
   %j_validation(i) = ComputeCost(X_validation, Y_validation,W1,b1,W2,b2,hyper_parameters.lambda,hyper_parameters.k);
   if(mod(i,10)==0)
      hyper_parameters.eta = hyper_parameters.eta*hyper_parameters.decay_rate; 
   end
end

[X,~,y,~,k,~] = LoadBatch('test_batch.mat');
acc = ComputeAccuracy(X,y,W1,b1,W2,b2,k)






