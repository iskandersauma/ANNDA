function [sbar, mu, v] = BN_forward(s, eps)
mu = mean(s,2);
v = mean((s- repmat(mu, 1, size(s,2))).^2, 2);
sbar = diag((v+eps).^(-0.5))*(s - repmat(mu, 1, size(s,2)));
end