data=load("C:\Users\Saif\Desktop\ms\sem 1\ML\HW4\hw4data\q2_2_data.mat");
train_data=data.trD;
train_label=data.trLb;
val_data=data.valD;
val_label=data.valLb;
classifiers=zeros(10,4097);

for i=1:10              %10 classes
    disp(i);
    ind=find(train_label==i);
    t_l=-ones(size(train_label,1),1);
    t_l(ind)=1;
    [weights,bias]=binary_SVM(train_data,t_l);
    classifiers(i,1:4096)=weights';
    classifiers(i,4097)=bias;
end
%% 
predicted=zeros(size(val_label));
for i=1:size(val_data,2)
    max_margin=0.0;
    class=0;
    for j=1:10
        margin=classifiers(j,1:4096)*val_data(:,i)+classifiers(j,4097);
        if margin>max_margin
            max_margin=margin;
            class=j;
        end
    end
    predicted(i,1)=class;
end
acc=0;
wrong=0;
for i=1:size(val_label,1)
    if predicted(i,1)~=val_label(i,1)
        wrong=wrong+1;
    end
end
acc=(size(val_label,1)-wrong)/size(val_label,1);
%% 
test_data=data.tstD;
pred=zeros(size(test_data,2),1);
for i=1:size(test_data,2)
    max_margin=-100000;
    class=0;
    for j=1:10
        margin=classifiers(j,1:4096)*test_data(:,i)+classifiers(j,4097);
        if margin>max_margin
            max_margin=margin;
            class=j;
        end
    end
    pred(i,1)=class;
end
csvwrite("submissions.csv",pred);