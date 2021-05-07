clear all;
clc;
filename = {'data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'};
X = [];
Y = [];
y = [];
N = 0;
for i = 1:5
    [X2, Y2, y2, N2, k, d] = LoadBatch(filename{i});
    X = [X X2];
    Y = [Y Y2];
    y = [y y2];
    N = N + N2;
end

%parametrar
a = 0.01;
eta = 0.02;
lambda = 0.01;
batch_size = 500;
epochs = 50;
W = a.*randn(k,d);
b = a.*rand(k,1);
j_train = zeros(epochs,1);
j_validation = zeros(epochs,1);

split = N - 1000;
X_train = X(:,1:split);
X_validation = X(:,split+1:N);
Y_train = Y(:,1:split);
Y_validation = Y(:,split+1:N);
N = split;

%training
for i = 1:epochs
    disp(i)
   for j = 1:N/batch_size
      j_start = (j-1)*batch_size + 1;
      j_end = j*batch_size;
      ind = j_start:j_end;
      Xbatch = X(:,ind);
      Ybatch = Y(:,ind);
      [W,b] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, k, d, eta);
   end
   j_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
   j_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda);
   if(mod(i,10) == 0)
       eta = eta*0.9;
   end
end

%accuracy
[X,~,y,~,k,~] = LoadBatch('test_batch.mat');
acc = ComputeAccuracy(X,y,W,b)

for i = 1:k
   im = reshape(W(i,:),32,32,3);
   s_im{i} = (im-min(im(:)))/(max(im(:)) - min(im(:)));
   s_im{i} = permute(s_im{i},[2,1,3]);
end
figure()
montage(s_im,'size',[1,k])

inds = 1:epochs;
plot(inds, j_train, inds, j_validation);