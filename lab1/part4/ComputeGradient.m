function grad_W = ComputeGradient(X,Y,P,W,lambda,delta)
grad_W = zeros(size(W));
pc = repmat(sum(P.*Y),size(P,1),1);
margin = P - pc + delta;
f = zeros(size(P));
f(find(margin > 0)) = 1;
f(find(Y == 1)) = -1;

for i =1:size(X,2)
   x = X(:,i);
   f1 = f(:,i);
   g = repmat(x',size(W,1),1);
   g(find(f1 == 0),:) = 0;
   g(find(f1 == -1),:) = -length(find(f1 == 1))*g(find(f1 == -1),:);
   grad_W = grad_W + g;
end
grad_W = grad_W/size(X,2)+2*lambda*W;

end