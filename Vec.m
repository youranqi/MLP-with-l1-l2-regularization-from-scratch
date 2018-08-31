function [x]=Vec(X)

x=[];
for i = 1:size(X,1)
    for j = 1:size(X,2)
        x=[x;reshape(X{i,j},numel(X{i,j}),1)];
    end
end
            
end