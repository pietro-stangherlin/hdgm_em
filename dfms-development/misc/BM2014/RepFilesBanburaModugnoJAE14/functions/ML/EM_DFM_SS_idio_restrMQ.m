function Res = EM_DFM_SS_idioMA_restrMQ(X,Par)


thresh = 1e-4;
r = Par.r;
p = Par.p;
max_iter = Par.max_iter;
R_mat = Par.Rconstr;
q = Par.q;
nQ = Par.nQ;

%--------------------------------------------------------------------------
% Preparation of the data
%--------------------------------------------------------------------------
[T,N] = size(X);

% Standardise x
Mx = nanmean(X);
Wx = (nanstd(X));
xNaN = (X-repmat(Mx,T,1))./repmat(Wx,T,1);
% xNaN = X;

%--------------------------------------------------------------------------
% Initial Conditions
%--------------------------------------------------------------------------

%Removing missing values (for initial estimators)
optNaN.method = 2; % Remove leading and closing zeros
optNaN.k = 3;


[A, C, Q, R, Z_0, V_0] = InitCond(xNaN,r,p,optNaN,R_mat,q,nQ);

% some auxiliary variables for the iterations
previous_loglik = -inf;
num_iter = 0;
LL = -inf;
converged = 0;

% y for the estimation is WITH missing data
y = xNaN';


%--------------------------------------------------------------------------
%THE EM LOOP
%--------------------------------------------------------------------------

%The model can be written as
%y = C*Z + e;
%Z = A*Z(-1) + v
%where y is NxT, Z is (pr)xT, etc

%remove the leading and ending nans for the estimation
optNaN.method = 3;
y_est = remNaNs_spline(xNaN,optNaN)';

while (num_iter < max_iter) & ~converged
    [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik] = EMstep(y_est, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ);
    
    C = C_new;
    R = R_new;
    A = A_new;
    Q = Q_new;

    % Checking convergence
    if num_iter>2
    [converged,decrease(num_iter+1)] = em_converged(loglik, previous_loglik, thresh,1);
    end
    
    LL = [LL loglik];
    previous_loglik = loglik;
    num_iter =  num_iter + 1;
end

%final run of the Kalman filter
Zsmooth = runKF(y, A, C, Q, R, Z_0, V_0)';
Res.x_sm = Zsmooth(2:end,:)*C';
Res.X_sm = repmat(Wx,T,1).*Res.x_sm+repmat(Mx,T,1);
% Res.F = Zsmooth(2:end,:);

%--------------------------------------------------------------------------
%   Loading the structure with the results
%--------------------------------------------------------------------------
Res.C = C;
Res.R = R;
Res.A = A;
Res.Q = Q;
Res.Z_0 = Z_0;
Res.V_0 = V_0;
Res.Mx = Mx;
Res.Wx = Wx;
Res.r = r;
Res.p = p;

% Res.loglik   = LL;
% Res.num_iter = num_iter;
% Res.converge = converged;
% decrease = any(decrease);
% 
%--------------------------------------------------------------------------
%PROCEDURES
%--------------------------------------------------------------------------

function  [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik] = EMstep(y, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio)

[n,T] = size(y);
rC = size(R_mat,2);
pC = max(p,rC);
rp = r*p;
rpC = r*pC;
rC = rC*r;
nM= n - nQ;
% Compute the (expected) sufficient statistics for a single Kalman filter sequence.

%Running the Kalman filter with the current estimates of the parameters
[Zsmooth, Vsmooth, VVsmooth, loglik] = runKF(y, A, C, Q, R, Z_0, V_0);


EZZ = Zsmooth(1:rpC,2:end)*Zsmooth(1:rpC,2:end)'+sum(Vsmooth(1:rpC,1:rpC,2:end),3);                        %E(Z'Z)
EZZ_BB = Zsmooth(1:rpC,1:end-1)*Zsmooth(1:rpC,1:end-1)'+sum(Vsmooth(1:rpC,1:rpC,1:end-1),3); %E(Z(-1)'Z_(-1))
EZZ_FB = Zsmooth(1:rpC,2:end)*Zsmooth(1:rpC,1:end-1)'+sum(VVsmooth(1:rpC,1:rpC,:),3);%E(Z'Z_(-1)) 

EZZ2 = diag(diag(Zsmooth(rpC+1:end,2:end)*Zsmooth(rpC+1:end,2:end)'))+...
    diag(diag(sum(Vsmooth(rpC+1:end,rpC+1:end,2:end),3)));                        %E(Z'Z)
EZZ_BB2 = diag(diag(Zsmooth(rpC+1:end,1:end-1)*Zsmooth(rpC+1:end,1:end-1)'))+...
    diag(diag(sum(Vsmooth(rpC+1:end,rpC+1:end,1:end-1),3))); %E(Z(-1)'Z_(-1))
EZZ_FB2 = diag(diag(Zsmooth(rpC+1:end,2:end)*Zsmooth(rpC+1:end,1:end-1)'))+...
    diag(diag(sum(VVsmooth(rpC+1:end,rpC+1:end,:),3)));%E(Z'Z_(-1)) 


A_new = A;
A_new(1:r,1:rp) = EZZ_FB(1:r,1:rp) * inv(EZZ_BB(1:rp,1:rp));
Q_new = Q;
Q_new(1:r,1:r) = (EZZ(1:r,1:r) - A_new(1:r,1:rp)*EZZ_FB(1:r,1:rp)') / T;

A_new2 = EZZ_FB2 * diag(1./diag((EZZ_BB2)));
Q_new2 = (EZZ2 - A_new2*EZZ_FB2') / T;

A_new(rpC+1:rpC+nM,rpC+1:rpC+nM) = A_new2(1:nM,1:nM);
Q_new(rpC+1:rpC+nM,rpC+1:rpC+nM) = Q_new2(1:nM,1:nM);

% Initial conditions
Z_0 = Zsmooth(:,1); %zeros(size(Zsmooth,1),1); %
V_0(1:rpC,1:rpC) = Vsmooth(1:rpC,1:rpC,1);
V_0(rpC+1:end,rpC+1:end) = diag(diag(Vsmooth(rpC+1:end,rpC+1:end,1)));

%E(Y'Y) & E(Y'Z) 
nanY = isnan(y);
y(nanY) = 0;

C_new = C;

nM = n-nQ;
denom = zeros(nM*r,nM*r);
nom = zeros(nM,r);

for t=1:T
    nanYt = diag(~nanY(1:nM,t));
    denom = denom + kron(Zsmooth(1:r,t+1)*Zsmooth(1:r,t+1)'+Vsmooth(1:r,1:r,t+1),nanYt);
    nom = nom + y(1:nM,t)*Zsmooth(1:r,t+1)'...
        -nanYt*(Zsmooth(rpC+1:rpC+nM,t+1)*Zsmooth(1:r,t+1)'+Vsmooth(rpC+1:rpC+nM,1:r,t+1));
end

vec_C = inv(denom)*nom(:);
C_new(1:nM,1:r) = reshape(vec_C,nM,r);

R_mat = kron(R_mat,eye(r));
q = kron(q,ones(r,1));
for i=n-nQ+1:n
    denom = zeros(rC,rC);
    nom = zeros(1,rC);

       idx_jQ = i-nM;
       i_idio_jQ = (rpC+nM+5*(idx_jQ-1)+1:rpC+nM+5*idx_jQ);
       V_0(i_idio_jQ,i_idio_jQ) = Vsmooth(i_idio_jQ,i_idio_jQ,1);
       A_new(i_idio_jQ(1),i_idio_jQ(1)) = A_new2(i_idio_jQ(1)-rpC,i_idio_jQ(1)-rpC);
       Q_new(i_idio_jQ(1),i_idio_jQ(1)) = Q_new2(i_idio_jQ(1)-rpC,i_idio_jQ(1)-rpC);
    for t=1:T
        nanYt = diag(~nanY(i,t));
        denom = denom + kron(Zsmooth(1:rC,t+1)*Zsmooth(1:rC,t+1)'+Vsmooth(1:rC,1:rC,t+1),nanYt);
        nom = nom + y(i,t)*Zsmooth(1:rC,t+1)';
            nom = nom -...
                nanYt*([1 2 3 2 1]*Zsmooth(i_idio_jQ,t+1)*Zsmooth(1:rC,t+1)'+...
                [1 2 3 2 1]*Vsmooth(i_idio_jQ,1:rC,t+1));
    end

    C_i = inv(denom)*nom';
    C_i_constr = C_i - inv(denom)*R_mat'*inv(R_mat*inv(denom)*R_mat')*(R_mat*C_i-q);
    C_new(i,1:rC) = C_i_constr;

end


R_new = R;


%--------------------------------------------------------------------------

function [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_increased)
% EM_CONVERGED Has EM converged?
% [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
%
% We have converged if the slope of the log-likelihood function falls below 'threshold', 
% i.e., |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
% 'threshold' defaults to 1e-4.
%
% This stopping criterion is from Numerical Recipes in C p423
%
% If we are doing MAP estimation (using priors), the likelihood can decrase,
% even though the mode of the posterior is increasing.

if nargin < 3, threshold = 1e-4; end
if nargin < 4, check_increased = 1; end

converged = 0;
decrease = 0;

if check_increased
    if loglik - previous_loglik < -1e-3 % allow for a little imprecision
        fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik);
        decrease = 1;
    end
end

delta_loglik = abs(loglik - previous_loglik);
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;
if (delta_loglik / avg_loglik) < threshold, converged = 1; end
 

%--------------------------------------------------------------------------

function [ A, C, Q, R, initZ, initV] = InitCond(xNaN,r,p,optNaN,Rcon,q,NQ,i_idio)


rC = size(Rcon,2);
pC = max(p,rC);

OPTS.disp=0;

[x,indNaN] = remNaNs_spline(xNaN,optNaN);


[T,N] = size(x);
NM = N - NQ;
% Eigenval decomp of cov(x) = VDV', only r largest evals
[ v, d ] = eigs(cov(x),r,'lm',OPTS);

% Static predictions
f = x*v;

%--------------------------------------------------------------------------
% Observation equation
%--------------------------------------------------------------------------
C = [v zeros(N,r*(pC-1))];

ff = [];
for kk = 0:rC-1
    ff = [ff f(rC-kk:end-kk,:)];
end

Rcon = kron(Rcon,eye(r));
q = kron(q,ones(r,1));

for i=N-NQ+1:N
    xx_i = xNaN(rC:T,i);
    if sum(~isnan(xx_i)) < size(ff,2)+2
        xx_i = x(rC:T,i);
    end
    ff_i = ff(~isnan(xx_i),:);
    xx_i = xx_i(~isnan(xx_i));
    iff_i = inv(ff_i'*ff_i);
    Cc = iff_i*ff_i'*xx_i;

    Cc = Cc - iff_i*Rcon'*inv(Rcon*iff_i*Rcon')*(Rcon*Cc-q);
    C(i,1:rC*r)=Cc';
end


res = x(rC:end,:) - ff*C(:,1:rC*r)';
resNaN = res;
resNaN(indNaN(rC:end,:)) = nan;
R = diag(nanvar(resNaN));


eyeN = eye(N);
eyeN(:,NM+1:end) = [];
% Initial conditions
C=[C eyeN];

%--------------------------------------------------------------------------
% Transition equation
%--------------------------------------------------------------------------
BM = zeros(NM);
SM = zeros(NM);

[T,N] = size(resNaN);

for i = 1:NM;

    res_i = resNaN(:,i);
    % number of leading zeros
    leadZero = max( find( (1:T)' == cumsum(isnan(res_i)) ) );
    endZero = max( find( (1:T)' == cumsum(isnan(res_i(end:-1:1))) ) );
    
    res_i = res(:,i);
    res_i(end-endZero:endZero) = [];
    res_i(1:leadZero) = [];
    
    BM(i,i) = inv(res_i(1:end-1)'*res_i(1:end-1))*res_i(1:end-1)'*res_i(2:end,:);
    SM(i,i) = cov(res_i(2:end)-res_i(1:end-1)*BM(i,i));
end
initViM = diag(1./diag(eye(size(BM,1))-BM.^2)).*SM;


C = [C [zeros(NM,5*NQ);kron(eye(NQ),[1 2 3 2 1])]];
Rdiag = diag(R);
sig_e = Rdiag(NM+1:N)/19;
R = 1e-04*eye(N);


rho0 = 0.1;
BQ = kron(eye(NQ),[[rho0 zeros(1,4)];[eye(4),zeros(4,1)]]);
temp = zeros(5);
temp(1,1) = 1;
SQ = kron(diag((1-rho0^2)*sig_e),temp);

initViQ = reshape(inv(eye((5*NQ)^2)-kron(BQ,BQ))*SQ(:),5*NQ,5*NQ);


% Estimate A & Q from stacked F(t) = A*F(t-1) + e(t);
z = f;
Z = [];
for kk = 1:p
    Z = [Z z(p-kk+1:end-kk,:)]; % stacked regressors (lagged SPC)
end;
z = z(p+1:end,:);
%% run the var chi(t) = A*chi(t-1) + e(t);


A = zeros(r*pC,r*pC)';
A_temp = inv(Z'*Z)*Z'*z;
A(1:r,1:r*p) = A_temp';
A(r+1:end,1:r*(pC-1)) = eye(r*(pC-1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = zeros(pC*r,pC*r);
e = z  - Z*A_temp; % VAR residuals
Q(1:r,1:r) = cov(e); % VAR covariance matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rp2 = (r*pC)^2;
initV = reshape(inv(eye(rp2)-kron(A,A))*Q(:),r*pC,r*pC);

A = blkdiag(A, BM, BQ);
Q = blkdiag(Q, SM, SQ);

% Initial conditions
initZ = zeros(size(A,1),1); % %[randn(1,r*(nlag+1))]';
initV = blkdiag(initV, initViM, initViQ);

