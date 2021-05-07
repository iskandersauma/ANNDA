function [W1,b1,W2,b2] = MiniBatchGD(X,Y,W1,b1,W2,b2, hyper_parameters)
[~,N] = size(X);
lambda = hyper_parameters.lambda;
k = hyper_parameters.k;
d = hyper_parameters.d;
m = hyper_parameters.m;
eta = hyper_parameters.eta;
rho = hyper_parameters.rho;
batch_size = hyper_parameters.batch_size;

% momentum
v_w1 = zeros(m,d);
v_b1 = zeros(m,1);
v_w2 = zeros(k,m);
v_b2 = zeros(k,1);

for i = 1:N/batch_size
    j_start = (i-1)*batch_size+1;
    j_end = i*batch_size;
    index = j_start:j_end;
    Xbatch = X(:,index);
    Ybatch = Y(:,index);
    
    [p,H,s]= EvaluateClassifier(Xbatch,W1,b1,W2,b2,k);
    [grad_w1,grad_b1,grad_w2,grad_b2] = ComputeGradient(Xbatch,Ybatch,H,s,p,W1,b1,W2,b2,lambda,k,d,m);
    
    %with momentum
    v_w1 = rho*v_w1 + eta*grad_w1;
    v_b1 = rho*v_b1 + eta*grad_b1;
    v_w2 = rho*v_w2 + eta*grad_w2;
    v_b2 = rho*v_b2 + eta*grad_b2;
    %update weights
    W1 = W1 - v_w1;
    b1 = b1 - v_b1;
    W2 = W2 - v_w2;
    b2 = b2 - v_b2;
    
    %without momentum
%     W1 = W1 - eta*grad_w1;
%     b1 = b1 - eta*grad_b1;
%     W2 = W2 - eta*grad_w2;
%     b2 = b2 - eta*grad_b2;
end


end