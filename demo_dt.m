
clear;
n = 20;                   
data = 'appendicitis_base_clustering.mat';      
load(data);                      
gt = y;
K = length(unique(gt));

gamma_all = [0.5,1,5,10,50,100,150,200,300];
lambda3 = 0.02;  
sigma_all = [-6:6];
for gamma_ind =1:length(gamma_all)
    gamma = gamma_all(gamma_ind);
tic
for sigama_ind = 1:length(sigma_all)
    sigma =  2^(sigma_all(sigama_ind));
parfor i = 1:10

[a,b] = size(E);
zz = RandStream('mt19937ar','Seed',i);
RandStream.setGlobalStream(zz);
indx = randperm(b);
EC_end = E(:,indx(1:n));
M = Gbe(EC_end); 
CA = M*M'./n;

C = solver_AWKEC(CA,gamma,lambda3,sigma);
 [U,S,V] = svd(C,'econ');
S = diag(S);
r = sum(S>1e-4*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^4;
results_C  = spectral_clustering(L,K);
ACC_C(i) = Accuracy(results_C,gt);
NMI_C(i)= compute_nmi(results_C,gt);
ARI_C(i)= RandIndex(results_C,gt);
Fscore_C(i)= compute_f(results_C,gt);
end
acc_c_final(gamma_ind,sigama_ind) = mean(ACC_C); 
nmi_c_final(gamma_ind,sigama_ind) = mean(NMI_C);
ari_c_final(gamma_ind,sigama_ind) = mean(ARI_C);
F_c_final(gamma_ind,sigama_ind) = mean(Fscore_C);

end
toc
end
RE=zeros(3,1);
 RE(1,1) = max(max(ari_c_final));
 RE(2,1) = max(max(nmi_c_final));
 RE(3,1) = max(max(acc_c_final));
