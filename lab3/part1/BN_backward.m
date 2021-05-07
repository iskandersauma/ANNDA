function g = BN_backward(g, mu, v, S, eps)
g_new = g*diag((v + eps).^(-1.5)).*(S - repmat(mu, 1, size(S, 2)))';
g_new = 0.5*sum(g_new);
g_sum = -sum(g*diag((v + eps).^(-0.5)));
g = g*diag((v + eps).^(-0.5)) + 2/size(S, 2)*repmat(g_new,...
        size(S, 2), 1).*(S - repmat(mu, 1, size(S, 2)))' + ...
        repmat(g_sum, size(S, 2), 1)/size(S, 2);
end