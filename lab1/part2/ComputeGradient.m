function [grad_W, grad_b] = ComputeGradient(X,Y,P,W,lambda,k,d)
[~,N] = size(X);
grad_W = zeros(k,d);
grad_b = zeros(k,1);

for i =1:N
   x = X(:,i);
   y = Y(:,i);
   p = P(:,i);
   g = -y'/(y'*p)*(diag(p)-p*p');
   grad_b = grad_b + g';
   grad_W = grad_W + g'*x';
end
grad_W = grad_W/N+2/lambda*W;
grad_b = grad_b/N;
end