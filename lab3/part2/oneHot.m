function out = oneHot(label)
K = length(unique(label));
N = length(label);
out = zeros(K,N);

for i = 1:N
   out(label(i),i) = 1; 
end
end