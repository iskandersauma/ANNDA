function [grad_W, grad_b] = ComputeGradient(X,Y,P,h,W,b,lambda)
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));


for i =1:size(X,2)
   x = X(:,i);
   y = Y(:,i);
   p = P(:,i);
   h1 = h(:,i);
   g = -y'/(y'*p)*(diag(p)-p*p');
   grad_b2 = grad_b2 + g';
   grad_W2 = grad_W2 + g'*h1';
   g = g*W2;
   h1(find(h1>0)) = 1;
   h1(find(h1<0)) = 0.1;
   g = g*diag(h1);
   grad_b1 = grad_b1 + g';
   grad_W1 = grad_W1 + g'*x';
end
grad_W1 = grad_W1/size(X,2)+2*lambda*W1;
grad_W2 = grad_W2/size(X,2)+2*lambda*W2;
grad_b1 = grad_b1/size(X,2);
grad_b2 = grad_b2/size(X,2);
grad_W = {grad_W1, grad_W2};
grad_b = {grad_b1, grad_b2};

end