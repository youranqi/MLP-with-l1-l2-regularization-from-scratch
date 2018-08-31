% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
trainimages = loadMNISTImages('train-images-idx3-ubyte');trainimages=trainimages';
trainlabels = loadMNISTLabels('train-labels-idx1-ubyte');
testimages = loadMNISTImages('t10k-images-idx3-ubyte');testimages=testimages';
testlabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
%display_network(trainimages(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

V_trainlabels=zeros(size(trainlabels,1),size(unique(trainlabels),1));
for i = 1:size(trainlabels,1)
    V_trainlabels(i,trainlabels(i)+1)=1;
end

V_testlabels=zeros(size(testlabels,1),size(unique(testlabels),1));
for i = 1:size(testlabels,1)
    V_testlabels(i,testlabels(i)+1)=1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fun=@Sigmoid; % @Identity @Sigmoid
N_Hidden=[30];
nn=10; %size of the batch
eta=3;
alfa=0.3;
lambda=0;
Trainsize=10000;
Testsize=5000;

rng(1);
[EachW,Eachb]=FeedforwardANN_Regular(trainimages(1:Trainsize,:),V_trainlabels(1:Trainsize,:),Fun,N_Hidden,nn,eta,alfa,lambda,'Epoch',10);

Result=zeros(length(EachW),2);
for i =1:length(EachW)

    [MSPE,Yhat]=ANNPredict(testimages(1:Testsize,:),V_testlabels(1:Testsize,:),EachW{i},Eachb{i},Fun,N_Hidden);
    [~,Index]=max(Yhat');
    Number=Index'-1;
    Result(i,:)=[i,sum(Number==testlabels(1:Testsize))/Testsize];

end
Result

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fun=@Sigmoid; % @Identity @Sigmoid
N_Hidden=[30];
nn=10;
eta=3;
alfa=0.3;
lambda=0.0001; 
Trainsize=10000;
Testsize=5000;

rng(1);
[EachW,Eachb]=FeedforwardANN_Regular(trainimages(1:Trainsize,:),V_trainlabels(1:Trainsize,:),Fun,N_Hidden,nn,eta,alfa,lambda,'Epoch',10);

Result=zeros(length(EachW),2);
for i =1:length(EachW)

    [MSPE,Yhat]=ANNPredict(testimages(1:Testsize,:),V_testlabels(1:Testsize,:),EachW{i},Eachb{i},Fun,N_Hidden);
    [~,Index]=max(Yhat');
    Number=Index'-1;
    Result(i,:)=[i,sum(Number==testlabels(1:Testsize))/Testsize];

end
Result


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(0);
Fun=@Sigmoid; % @Identity @Sigmoid
N_Hidden=[60];
[X,Y,TrueW,Trueb]=ANNGen_Sparse(10000,300,200,0.1,Fun,N_Hidden);
Trainsize=7000;
Testsize=3000;
TrainX=X(1:Trainsize,:);TrainY=Y(1:Trainsize,:);
TestX=X((Trainsize+1):(Trainsize+Testsize),:);TestY=Y((Trainsize+1):(Trainsize+Testsize),:);

%%%
nn=10;
eta=3;
alfa=0.3;
lambda=0;

rng(2);
[EachW,Eachb]=FeedforwardANN_Regular(TrainX,TrainY,Fun,N_Hidden,nn,eta,alfa,lambda,'Epoch',10);

Result=zeros(length(EachW),2);
for i =1:length(EachW)

    [MSPE,Yhat]=ANNPredict(TestX,TestY,EachW{i},Eachb{i},Fun,N_Hidden);
    Result(i,:)=[i,MSPE];

end
Result

%%%
nn=10;
eta=3;
alfa=0.3;
lambda=0.0001;

rng(2);
[EachW,Eachb]=FeedforwardANN_Regular(TrainX,TrainY,Fun,N_Hidden,nn,eta,alfa,lambda,'Epoch',10);

Result=zeros(length(EachW),2);
for i =1:length(EachW)

    [MSPE,Yhat]=ANNPredict(TestX,TestY,EachW{i},Eachb{i},Fun,N_Hidden);
    Result(i,:)=[i,MSPE];

end
Result

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Garch.sim=textread('Garch.sim.txt');
ARIMA.sim=textread('ARIMA.sim.txt');
k=3;Trainsize=250;

% Ts=Garch.sim;
% Ts=Ts*1000;
% MinTs=abs(min(Ts));
% Ts=Ts+MinTs;
% MaxTs=max(abs(Ts));
% Ts=Ts/MaxTs;

Ts=ARIMA.sim;
MinTs=abs(min(Ts));
Ts=Ts+MinTs;
MaxTs=max(abs(Ts));
Ts=Ts/MaxTs;

Train=[];
for i=1:(Trainsize)
    
    Train=[Train;Ts(i:(i+k))'];
    
end
TrainX=Train(:,1:k);TrainY=Train(:,(k+1));

Test=[];
for i=(Trainsize+1):(length(Ts)-k)
    
    Test=[Test;Ts(i:(i+k))'];
    
end
TestX=Test(:,1:k);TestY=Test(:,(k+1));

Fun=@Sigmoid; % @Identity @Sigmoid
N_Hidden=[70];
nn=5;
eta=3;
alfa=0.3;
lambda=0.0001; 

rng(3);
[EachW,Eachb]=FeedforwardANN_Regular(TrainX,TrainY,Fun,N_Hidden,nn,eta,alfa,lambda,'Epoch',15);

Result=zeros(length(EachW),2);
for i =1:length(EachW)

    [MSPE,Yhat]=ANNPredict(TestX,TestY,EachW{i},Eachb{i},Fun,N_Hidden);
    Result(i,:)=[i,MSPE];

end
Result

% OYhat=Yhat;

% [(TestY*MaxTs-MinTs)/1000,(Yhat*MaxTs-MinTs)/1000]
% plot((TestY*MaxTs-MinTs)/1000);hold on;plot((Yhat*MaxTs-MinTs)/1000)

%[TestY*MaxTs-MinTs,OYhat*MaxTs-MinTs,Yhat*MaxTs-MinTs]

plot(TestY*MaxTs-MinTs,'-b');
hold on;
plot(OYhat*MaxTs-MinTs,'-g');
hold on;
plot(Yhat*MaxTs-MinTs,'-r');
legend('True Series','Original','Elastic Net');
title('Figure 1: Predictions to the Time Series')

