function [W,b] = init_parameters(M,std)
for i = 1:size(M,2) - 1
   if nargin < 2
      var = 2/(M(i) + M(i+1));
      std = sqrt(var);
   end
   W{i} = std*rand(M(i+1),M(i));
   b{i} = zeros(M(i+1),1);
end
end