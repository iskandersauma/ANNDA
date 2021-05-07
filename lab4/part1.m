clear all;
clc;

book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);

char_to_int = containers.Map('KeyType', 'char', 'ValueType', 'int32');
int_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

keySet = num2cell(book_chars);
valueSet = 1:length(keySet);
newMap1 = containers.Map(keySet, valueSet);
newMap2 = containers.Map(valueSet, keySet);
char_to_int = [char_to_int; newMap1];
int_to_char = [int_to_char; newMap2];

k = length(keySet);
m = 100;
eta = 0.1;
seq_length = 25;
std = 0.01;
RNN.b = zeros(m, 1);      
RNN.c = zeros(k, 1);
RNN.U = randn(m, k)*std;
RNN.W = randn(m, m)*std;
RNN.V = randn(k, m)*std;
M.W = zeros(size(RNN.W));
M.U = zeros(size(RNN.U));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));


X_int = zeros(1,length(book_data));
for i = 1:length(book_data)
    X_int(i) = char_to_int(book_data(i));
end
X = oneHot(X_int,k);

Y = X;
iter = 1;
epochs = 20;
sl = [];
s = 0;
hprev = [];
min_loss = 500;
min_loss_list = [];
for i = 1:epochs
    disp(i)
   [RNN, s, iter, M, min_loss] = MiniBatchGD(RNN,...
       X, Y, seq_length, k, m, eta, iter, M, int_to_char, s(end), min_loss);
   sl = [sl, s];
   min_loss_list(i) = min_loss;
end
plot(1:length(sl),sl)
plot(1:length(min_loss_list),min_loss_list)
