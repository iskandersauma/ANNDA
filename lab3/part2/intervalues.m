function [h,S,mu,v] = intervalues(X,W,b,k)
eps = 0.001;
for i = 1:k-1
   Wi = W{i};
   bi = b{i};
   bi = repmat(bi,1,size(X,2));
   s = Wi*X + bi;
   S{i} = s;
   
   [sbar, mu_i, v_i] = BN_forward(s,eps);
   mu{i} = mu_i;
   v{i} = v_i;
   X = max(0,sbar);
   h{i} = X;
end
end