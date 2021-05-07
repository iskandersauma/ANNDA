function [grad_W, grad_b] = ComputeGradient(X,Y,P,W,lambda)
grad_W = zeros(size(W));
grad_b = zeros(size(W,1),1);

for i =1:size(X,2)
   x = X(:,i);
   y = Y(:,i);
   p = P(:,i);
   g = -y'/(y'*p)*(diag(p)-p*p');
   grad_b = grad_b + g';
   grad_W = grad_W + g'*x';
end
grad_W = grad_W/size(X,2)+2*lambda*W;
grad_b = grad_b/size(X,2);

end