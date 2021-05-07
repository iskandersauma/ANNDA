function [X,Y,y,N,k,d,mu] = LoadBatch(filename)
A = load(filename);
X = im2double(A.data');
mu = mean(X,2);
X = X - repmat(mu, [1,size(X,2)]);
y = A.labels;
N = length(y);
k = length(min(y):max(y));
[d,~] = size(X);
Y = zeros(k,N);
for i = 1:k
    rows = y==(i-1);
    Y(i,rows) = 1;
end
end