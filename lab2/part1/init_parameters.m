function [W,b] = init_parameters(m,d,k,std)
W1 = std*randn(m,d);
W2 = std*randn(k,m);
b1 = zeros(m,1);
b2 = zeros(k,1);
W = {W1,W2};
b = {b1, b2};
end