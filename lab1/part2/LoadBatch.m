function [X,Y,y,N,k,d] = LoadBatch(filename)
A = load(filename);
X = im2double(A.data');
y = A.labels;

N = length(y);
k = length(min(y):max(y));
[d,~] = size(X);
Y = zeros(k,N);

for i = 0:(k-1)
    rows = (y==i);
    Y(i+1, rows) = 1;
end
end