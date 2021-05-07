function [Wstar, bstar] = MiniBatchGD(X,Y,W,b,lambda,GDparams,rho)
batch_size = GDparams.batch_size;
eta = GDparams.eta;
N = size(X,2);
v_W = {zeros(size(W{1})),zeros(size(W{2}))};
v_b = {zeros(size(b{1})),zeros(size(b{2}))};

for i = 1:N/batch_size
   jstart = (i-1)*batch_size+1;
   jend = i*batch_size;
   index = jstart:jend;
   Xbatch = X(:,index);
   Ybatch = Y(:,index);
   Xbatch = Xbatch + 0.05*randn(size(Xbatch));
   
   h = intervalues(Xbatch,W,b);
   p = EvaluateClassifier(h,W,b);
   [grad_W,grad_b] = ComputeGradient(Xbatch,Ybatch,p,h,W,b,lambda);
   
   v_W{1} = rho*v_W{1} + eta*grad_W{1};
   v_W{2} = rho*v_W{2} + eta*grad_W{2};
   v_b{1} = rho*v_b{1} + eta*grad_b{1};
   v_b{2} = rho*v_b{2} + eta*grad_b{2};
   
   
   W{1} = W{1} - v_W{1};
   W{2} = W{2} - v_W{2};
   b{1} = b{1} - v_b{1};
   b{2} = b{2} - v_b{2};
end
Wstar = W;
bstar = b;

end