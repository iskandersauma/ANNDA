function s = batchNormalize(s,mu,v)
s = cellfun(@(x) (v.^(-0.5))'.*(x-mu),s,'UniformOutput',false);
end