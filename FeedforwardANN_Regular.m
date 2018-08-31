function [EachW,Eachb]=FeedforwardANN_Regular(X,Y,Fun,N_Hidden,nn,eta,alfa,lambda,varargin) 

optargs={'Epoch',20,0.001};
optargs(1:length(varargin))=varargin;
[Stoprule,Maxepoch,Criticalvalue]=optargs{:};

p=size(X,2);
n=size(X,1);
q=size(Y,2);
L=size(N_Hidden,1)+1;

b=cell(L,1);
b{1,1}=zeros(N_Hidden(1),1);
if L>=3
    for i = 2:(L-1)
        b{i,1}=zeros(N_Hidden(i),1);
    end
end
b{L,1}=zeros(q,1);
Saveb=b;

W=cell(L,1);
W{1,1}=zeros(N_Hidden(1),p);
if L>=3
    for i = 2:(L-1)
        W{i,1}=zeros(N_Hidden(i),N_Hidden(i-1));
    end
end
W{L,1}=zeros(q,N_Hidden(L-1));
DW=W;
SaveW=W;

a=Saveb;
z=Saveb;
delta=Saveb;

EachW=cell(Maxepoch,1);
Eachb=cell(Maxepoch,1);

%Initialize W and b
for i = 1:L
    b{i,1}=normrnd(0,1,size(b{i,1},1),size(b{i,1},2));
    W{i,1}=normrnd(0,1,size(W{i,1},1),size(W{i,1},2)); 
end

% Start SGD with backpropagation
STOP=0;
epoch=0;
Waitbar=waitbar(0,'Please wait...');
while(~STOP)

    epoch=epoch+1;
    
    Randomperm=reshape(randperm(n),nn,(n/nn));
    
    for k=1:size(Randomperm,2)
        
        c.Db=Saveb;
        c.DW=SaveW;
        
        for i=Randomperm(:,k)'
            
            %calculate a and z
            z{1}=W{1}*X(i,:)'+b{1};
            a{1}=arrayfun(Fun,z{1});
            for j = 2:L
                z{j}=W{j}*a{j-1}+b{j};
                a{j}=arrayfun(Fun,z{j});
            end
            
            %calculate the delta using backpropagation
            delta{L}=arrayfun(Fun,z{L},zeros(size(z{L},1),size(z{L},2))).*(a{L}-Y(i,:)');
            for l=(L-1):-1:1
                delta{l}=arrayfun(Fun,z{l},zeros(size(z{l},1),size(z{l},2))).*(W{l+1}'*delta{l+1});
            end
            
            %Obtain the derivatives
            Db=delta;
            DW{1}=delta{1}*X(i,:);
            for u=2:L
                DW{u}=delta{u}*a{u-1}';
            end
            
            for v=1:L
                c.Db{v}=c.Db{v}+Db{v};
                c.DW{v}=c.DW{v}+DW{v};
            end
            waitbar(((epoch-1)*n+(k-1)*nn+find(Randomperm(:,k)==i))/(Maxepoch*n))
        end
        
        for i = 1:L
            b{i}=b{i}-eta.*c.Db{i}/nn;
            W{i}=W{i}-eta.*c.DW{i}/nn;
        end
        
        if lambda~=0
            for i =1:L
                paramb=repmat(eta*lambda*alfa,size(b{i},1),size(b{i},2));
                b{i}=arrayfun(@S,b{i},paramb)./(1+eta*lambda*(1-alfa));
                paramW=repmat(eta*lambda*alfa,size(W{i},1),size(W{i},2));
                W{i}=arrayfun(@S,W{i},paramW)./(1+eta*lambda*(1-alfa));
            end
        end
    end
    
    EachW{epoch}=W;
    Eachb{epoch}=b;
    
    if strcmp(Stoprule,'Epoch')
        STOP=(epoch>=Maxepoch);
    elseif strcmp(Stoprule,'MSPE')
        [MSPE,~]=ANNPredict(X,Y,W,b,Fun,N_Hidden);
        STOP=(MSPE<=Criticalvalue | epoch>=Maxepoch);
    elseif strcmp(Stoprule,'DistanceWb')
        STOP=(norm([Vec(W);Vec(b)]-[Vec(EachW{epoch-1});Vec(Eachb{epoch-1})])<=Criticalvalue | epoch>=Maxepoch);
    end
    
end
close(Waitbar); 
end
