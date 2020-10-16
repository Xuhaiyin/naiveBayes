% Lawn sprinker example from Russell and Norvig p454
% See www.cs.berkeley.edu/~murphyk/Bayes/usage.html for details.

t1=cputime;
N = 4; 
dag = zeros(N,N); 
C = 1; S = 2; R = 3; W = 4;
dag(C,[R S]) = 1;
dag(R,W) = 1;
dag(S,W)=1;

false = 1; true = 2;
ns = 2*ones(1,N); % binary nodes

bnet = mk_bnet(dag, ns);
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);
%bnet.CPD{W} = tabular_CPD(bnet, W, [0.99 0.1 0.1 0.01 0.01 0.9 0.9 0.99]);

bnet.CPD{W}

CPT = cell(1,N);
for i=1:N
  s=struct(bnet.CPD{i});  % violate object privacy
  CPT{i}=s.CPT;
end

% Generate training data
nsamples =1000;
samples = cell(N, nsamples);
for i=1:nsamples
  samples(:,i) = sample_bnet(bnet);
end
data = cell2num(samples);

% Make a tabula rasa
bnet2 = mk_bnet(dag, ns);
seed = 0;
% rand('state', seed);
bnet2.CPD{C} = tabular_CPD(bnet2, C, 'clamped', 1, 'CPT', [0.5 0.5], ...
			   'prior_type', 'dirichlet', 'dirichlet_weight', 0);
bnet2.CPD{R} = tabular_CPD(bnet2, R, 'prior_type', 'dirichlet', 'dirichlet_weight', 0);
bnet2.CPD{S} = tabular_CPD(bnet2, S, 'prior_type', 'dirichlet', 'dirichlet_weight', 0);
bnet2.CPD{W} = tabular_CPD(bnet2, W, 'prior_type', 'dirichlet', 'dirichlet_weight', 0);

Parameter_MLE=bnet2;   %最大似然估计
CPT_MLE=cell(1,N);
for i=1:N
    s=struct(Parameter_MLE.CPD{i});
    CPT_MLE{i}=s.CPT;
end

Parameter_MLE_W = CPT_MLE{4};

%完整数据时，学习参数的方法主要有两种：最大似然估计learn_params()和贝叶斯方法bayes_update_params(); 

% Find MLEs from fully observed data
bnet4 = learn_params(bnet2, samples);

% Bayesian updating with 0 prior is equivalent to ML estimation
bnet5 = bayes_update_params(bnet2, samples);

CPT4 = cell(1,N);
for i=1:N
  s=struct(bnet4.CPD{i});  % violate object privacy
  CPT4{i}=s.CPT ;
end
CPT4{4}
CPT5 = cell(1,N);
for i=1:N
  s=struct(bnet5.CPD{i});  % violate object privacy
  CPT5{i}=s.CPT ;
  assert(approxeq(CPT5{i}, CPT4{i}));
end
CPT5{4}

t2=cputime;


fprintf('运行时间:%f\n',t2-t1);
