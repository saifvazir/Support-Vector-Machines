data=load("C:\Users\Saif\Desktop\ms\sem 1\ML\HW4\hw4data\q2_1_data.mat");
train_data=data.trD;
train_label=data.trLb;
val_data=data.valD;
val_label=data.valLb;

C=1;
[r_t,c_t]=size(train_data);
[r_lb,c_lb]=size(train_label);
trans_data=transpose(train_data);
trans_label=transpose(train_label);
X_y_ele_mul=train_data.*trans_label;
H=zeros([c_t,c_t]);
H=(X_y_ele_mul')*(X_y_ele_mul);
H=double(H);
%{
for i=1:c_t
    for j=i:c_t
        H(i,j)=trans_label(1,i)*(trans_data(i,:)*train_data(:,j))*train_label(j,1);
        if i==j
            H(i,j)=H(i,j);
        end
        if i~=j
            H(j,i)=H(i,j);
        end
    end
end
%}
A=[];
b=[];
f(1:c_t,1)=-1;
Aeq=double(trans_label);
beq=0;
lb=zeros(c_t,1);
ub(1:c_t,1)=C;
%options = optimset('Algorithm','interior-point-convex');
[X,fval]=quadprog(H,f,A,b,Aeq,beq,lb,ub);
w=zeros(r_t,1);
bias=0;

for i=1:c_t

    if X(i,1)<1.000e-12
        X(i,1)=0.00000;
    end
end
  
%indices=find(X>0.0000);
%disp(indices);


for i=1:c_t
    w=w+train_label(i,1)*X(i,1)*train_data(:,i);
end

for i=1:c_t
    if X(i,1)>0 && X(i,1)<C
        bias=train_label(i,1)-transpose(w)*train_data(:,i);
        break;
    end
end

[r_valD,c_valD]=size(val_data);
predicted=zeros(c_valD,1);
w_t=transpose(w);

for i=1:c_valD
    predicted(i,1)=sign(w_t*val_data(:,i)+bias);
end
acc=0;
wrong=0;
for i=1:c_valD
    if predicted(i,1)~=val_label(i,1)
        wrong=wrong+1;
    end
end
acc=(c_valD-wrong)/c_valD;

conf_mat=confusionmat(val_label,predicted);
