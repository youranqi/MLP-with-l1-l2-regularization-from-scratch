function [Z]=Identity(z,varargin)

optargs={1};
optargs(1:length(varargin))=varargin;
[Mode]=optargs{:};

if Mode==1
    Z=z;
else
    Z=1;
end

end