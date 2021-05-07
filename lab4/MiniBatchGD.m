function [RNN, sl, iter, M, min_loss] = MiniBatchGD(RNN, X, Y, n, k, m, eta, iter, M, int_to_char, smooth_loss, min_loss)
e = 1;
textlen = 1000;
sl = [];

while e <= length(X) - n -1
    Xe = X(:, e : e + n - 1);
    Ye = Y(:, e + 1 : e + n);
    
    if e == 1
       hprev = zeros(m,1);
    else
       hprev = h(:,end); 
    end
    [loss, a, h, ~, p] = forward(RNN, Xe, Ye, hprev, n, k, m);
    [RNN, M] = backward(RNN, Xe, Ye, a, h, p, n, m, eta, M);
    
    if loss < min_loss
        min_loss = loss;
    end
    sl = [sl, loss];
    
    if iter == 1 || mod(iter, 5000) == 0
        y = synText(RNN, hprev, X(:,1), textlen, k);
        c = [];
        for i = 1:textlen
            c = [c int_to_char(y(i))];
        end
        disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
        disp(c);
    end
    
    iter = iter + 1;
    e = e + n;
end
end
