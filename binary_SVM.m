function [w,bias] = binary_SVM(train_data,train_label)
C=10;
[r_t,c_t]=size(train_data);
%[r_lb,c_lb]=size(train_label);
%trans_data=transpose(train_data);
trans_label=transpose(train_label);
X_y_ele_mul=train_data.*trans_label;
%H=zeros([c_t,c_t]);
H=(X_y_ele_mul')*(X_y_ele_mul);
H=double(H);
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
bias=0.0;
%{
for i=1:c_t

    if X(i,1)<1.000e-10
        X(i,1)=0.00000;
    end
end
%}
zero_indices=find(X<1.000e-5);
X(zero_indices)=0.0000;

%{
for i=1:c_t
    w=w+train_label(i,1)*X(i,1)*train_data(:,i);
end
%}
w=train_data.*trans_label.*X';
w=sum(w,2);
%{
ind=find(X>0&&X<C);
temp=X(ind);
bias=temp(1,1);
%}
for i=1:c_t
    if X(i,1)>0 && X(i,1)<C
        bias=train_label(i,1)-transpose(w)*train_data(:,i);
        break;
    end
end
%{
max_bias=-10000.000;
for i=1:c_t
    if X(i,1)>0 && X(i,1)<C
        bias=train_label(i,1)-transpose(w)*train_data(:,i);
        if bias>max_bias
            max_bias=bias;
        end
    end
end
bias=max_bias;
%}

end %end of function binary svm 