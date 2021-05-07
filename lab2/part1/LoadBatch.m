function [X,Y,y] = LoadBatch(filename)
indata = load(filename);
X = double(indata.data')/255;
y = double(indata.labels') + 1;
Y = oneHot(y);
end