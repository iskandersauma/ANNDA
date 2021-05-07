function [grad_w1,grad_b1,grad_w2,grad_b2] = ComputeGradient(X,Y,H,s,p,W1,b1,W2,b2,lambda,k,d,m)
[~,N] = size(X);
grad_w1 = zeros(m,d);
grad_b1 = zeros(m,1);
grad_w2 = zeros(k,m);
grad_b2 = zeros(k,1);

for i = 1:N
   x1 = X(:,i);
   y1 = Y(:,i);
   p1 = p(:,i);
   h = H(:,i);
   g = -y1'/(y1'*p1)*(diag(p1)-p1*p1');
   grad_b2 = grad_b2 + g';
   grad_w2 = grad_w2 + g'*h';
   
   g = g*W2;
   %g = g*diag(h>0); %for relu
   g = g*diag((exp(-s)./(1+exp(-s)).^2));
   grad_b1 = grad_b1+g';
   grad_w1 = grad_w1 + g'*x1';
end
grad_w2 = grad_w2/N + 2*lambda*W2;
grad_b2 = grad_b2/N;
grad_w1 = grad_w1/N + 2*lambda*W1;
grad_b1 = grad_b1/N;
end