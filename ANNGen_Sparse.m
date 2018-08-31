function [X,Y,W,b]=ANNGen_Sparse(n,p,q,sparse_ratio,Fun,N_Hidden) 

X=normrnd(0,1,n,p);
Y=zeros(n,q);
L=size(N_Hidden,1)+1;

b=cell(L,1);
b{1,1}=zeros(N_Hidden(1),1);
if L>=3
    for i = 2:(L-1)
        b{i,1}=zeros(N_Hidden(i),1);
    end
end
b{L,1}=zeros(q,1);
a=b;
z=b;

W=cell(L,1);
W{1,1}=zeros(N_Hidden(1),p);
if L>=3
    for i = 2:(L-1)
        W{i,1}=zeros(N_Hidden(i),N_Hidden(i-1));
    end
end
W{L,1}=zeros(q,N_Hidden(L-1));

for l=1:L
    
    n_bl=numel(b{l});nonzero_bl=ceil(n_bl*sparse_ratio);
    b{l}(sort(randsample(1:n_bl,nonzero_bl)))=normrnd(0,1,nonzero_bl,1);
    
    n_Wl=numel(W{l});nonzero_Wl=ceil(n_Wl*sparse_ratio);
    W{l}(sort(randsample(1:n_Wl,nonzero_Wl)))=normrnd(0,1,nonzero_Wl,1);

end

for i =1:size(X,1)
    z{1}=W{1}*X(i,:)'+b{1};
    a{1}=arrayfun(Fun,z{1});
    for j = 2:L
        z{j}=W{j}*a{j-1}+b{j};
        a{j}=arrayfun(Fun,z{j});
    end
    Y(i,:)=a{L};
end

end