function out = oneHot(label,K)
N = length(label);
out = zeros(K,N);

for i = 1:N
   out(label(i),i) = 1; 
end
end