function [Wstar, bstar,mu,v] = MiniBatchGD(X,Y,W,b,lambda,GDparams,rho,alpha,k)
batch_size = GDparams.batch_size;
eta = GDparams.eta;
N = size(X,2);

for i = 1:k
   v_W{i} = zeros(size(W{i}));
   v_b{i} = zeros(size(b{i}));
end

for i = 1:N/batch_size
   jstart = (i-1)*batch_size+1;
   jend = i*batch_size;
   index = jstart:jend;
   Xbatch = X(:,index);
   Ybatch = Y(:,index);
   
   [h,s,mu_i,v_i] = intervalues(Xbatch,W,b,k);
   p = EvaluateClassifier(h,W,b);
   
   for j = 1:k-1
      if i == 1
          mu = mu_i;
          v = v_i;
      else
         mu{j} = alpha*mu{j} + (1-alpha)*mu_i{j};
         v{j} = alpha*v{j} + (1-alpha)*v_i{j};
      end
   end
   
   [grad_W,grad_b] = ComputeGradient(Xbatch,Ybatch,p,h,s,W,lambda,k,mu_i,v_i);
   
   for m = 1:k
       v_W{m} = rho*v_W{m} + eta*grad_W{m};
       v_b{m} = rho*v_b{m} + eta*grad_b{m};
   
       W{m} = W{m} - v_W{m};
       b{m} = b{m} - v_b{m};
   end
end
Wstar = W;
bstar = b;

end