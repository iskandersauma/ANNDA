function [grad_W, grad_b] = ComputeGradient(X,Y,P,h,s,W,lambda,k,mu,v)

grad_Wi = zeros(size(Y,1),size(h{end},1));
grad_bi = zeros(size(Y,1),1);
prev = zeros(size(X,2),size(h{end},1));
eps = 0.001;

for i =1:size(X,2)
   y = Y(:,i);
   p = P(:,i);
   h1 = h{end}(:,i);
   g = -y'/(y'*p)*(diag(p)-p*p');
   
   grad_bi = grad_bi + g';
   grad_Wi = grad_Wi + g'*h1';
   
   g = g*W{end};
   prev(i,:) = g*diag(h1>0);
end
grad_W{k} = 2*lambda*W{end} + grad_Wi/size(X,2);
grad_b{k} = grad_bi/size(X,2);

for i = k-1:-1:1
   prev = BN_backward(prev,mu{i},v{i},s{i},eps); 
    
   grad_b{i} = mean(prev)';
   if i == 1
      grad_W{i} = prev'*X'; 
   else
      grad_W{i} = prev'*h{i-1}';
   end
   grad_W{i} = grad_W{i}/size(X,2) + 2*lambda*W{i};
   
   if i > 1
       prev = prev*W{i};
       prev = prev.*(h{i-1} > 0)';
   end
end

end