function GDparams = setParams(batch_size,eta,epochs)
if nargin > 0
   GDparams.batch_size = batch_size;
   GDparams.eta = eta;
   GDparams.epochs = epochs;
end
end