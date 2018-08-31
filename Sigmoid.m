function [Z]=Sigmoid(z,varargin)

optargs={1};
optargs(1:length(varargin))=varargin;
[Mode]=optargs{:};

if Mode==1
    Z=(1+exp(-z))^(-1);
else
    Z=(1+exp(-z))^(-1)*(1-(1+exp(-z))^(-1));
end

end
