function [MSPE,Yhat]=ANNPredict(X,Y,W,b,Fun,N_Hidden) 

p=size(X,2);
n=size(X,1);
q=size(Y,2);
L=size(N_Hidden,1)+1;

a=cell(L,1);
a{1,1}=zeros(N_Hidden(1),1);
if L>=3
    for i = 2:(L-1)
        a{i,1}=zeros(N_Hidden(i),1);
    end
end
a{L,1}=zeros(q,1);
z=a;
Yhat=zeros(size(Y,1),size(Y,2));

for i =1:size(X,1)
    z{1}=W{1}*X(i,:)'+b{1};
    a{1}=arrayfun(Fun,z{1});
    for j = 2:L
        z{j}=W{j}*a{j-1}+b{j};
        a{j}=arrayfun(Fun,z{j});
    end
    Yhat(i,:)=a{L};
end

MSPE=(norm(Yhat-Y,'fro')^2)/n;

end