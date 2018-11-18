[trD, trLb, valD, valLb, trRegs, valRegs]=HW4_Utils.getPosAndRandomNeg();
[weights,bias]=binary_SVM(trD,trLb);
HW4_Utils.genRsltFile(weights,bias,valD,'C:\Users\Saif\Desktop\ms\sem 1\ML\HW4\hw4data');
[ap, prec, rec]=HW4_Utils.cmpAP('C:\Users\Saif\Desktop\ms\sem 1\ML\HW4\hw4data\rects', dataset);
%% 
