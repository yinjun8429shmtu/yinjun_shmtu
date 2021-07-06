%% objective£ºsum1~V£¨||X_v-U_vP||^2+beta*||U_v||^2£©+alpha*tr(PLwP')+gama*sum1~Nsum1~N(WijInWij)

% data_e: existing data X_E
% ori_new_data: original P
% ori_bas: original U_v
% los_mark: mark of missing data

function [data,bas,new_data,J,W] = IMCRV-2(data_e,ori_new_data,ori_bas,los_mark,options)

alpha = options.alpha;
beta = options.beta;
gama = options.gama;
maxiter = options.maxiter;
stoperr = options.stoperr;
view_num = length(data_e);
[new_dim,data_num] = size(ori_new_data);
dim = zeros(1,view_num); 
tq_l = cell(1,view_num);  
kz_l = cell(1,view_num); 
kz_e = cell(1,view_num); 
tot_mark = 1:data_num;  
data_l = cell(1,view_num); 
data = cell(1,view_num);  

sum_dim = 0;

for view_mark = 1:view_num    
    [dim(view_mark),~] = size(data_e{view_mark});
    sum_dim = sum_dim+dim(view_mark);
     los_mark{view_mark} = sort(los_mark{view_mark});  
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(data_num);
     tq_l{view_mark}(:,ext_mark) = [];
     l_num = length(los_mark{view_mark}); 
     e_num = data_num - l_num;  
     kz_l{view_mark} = zeros(l_num,data_num);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,data_num);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
end
beta = beta*data_num/new_dim;
alpha = alpha*sum_dim/new_dim;
gama = gama*alpha;
new_data = ori_new_data;  
bas = ori_bas;
%%%%

%%%% initialize W
W = ones(data_num,data_num);
W = W/(data_num-1);
W = W-diag(diag(W));
%%%%

D1= diag(sum(W,1));
D2= diag(sum(W,2));

DD = D1 + D2;
WW = W + W';

%%%%%

J = zeros(maxiter,1);
for it_mark = 1:maxiter   
    temp1 = zeros(new_dim,data_num);
    temp2 = zeros(new_dim,data_num); 
     for view_mark = 1:view_num
        %%%% update X_v
        data_l{view_mark} = bas{view_mark}*new_data*tq_l{view_mark}; 
        data{view_mark} = data_l{view_mark}*kz_l{view_mark} + data_e{view_mark}*kz_e{view_mark}; 
        %%%%
        
        %%%% update U_v
        temp3 = data{view_mark}*new_data';
        temp4 = bas{view_mark}*new_data*new_data'+beta*bas{view_mark};
        temp4(abs(temp4)<1e-10) = 1e-10;
        bas{view_mark} = bas{view_mark}.*temp3./temp4;
        %%%%

        temp1 = temp1+bas{view_mark}'*data{view_mark};
        temp2 = temp2+bas{view_mark}'*bas{view_mark}*new_data;
     end

    %%%% update P
    temp1 = temp1 + alpha*new_data*WW;
    temp2 = temp2 + alpha*new_data*DD;
    temp2(abs(temp2)<1e-10) = 1e-10;
    new_data = new_data.*temp1./temp2;
    %%%%

    %%%% update W
    tempW1 = repmat(diag(new_data'*new_data),1,data_num); 
    tempW2 = repmat(diag(new_data'*new_data)',data_num,1);
    tempW3 = new_data'*new_data;
    W = exp((-alpha/gama)*(tempW1+tempW2-2*tempW3));
    W = max(W,1e-300);  
    W = W-diag(diag(W));
    W = W./repmat(sum(W),data_num,1);
    %%%%
    
     tempXN = sum(sum(W.*log(W+eye(data_num)))); 

     D1= diag(sum(W,1));
     D2= diag(sum(W,2));

     DD = D1 + D2;
     WW = W + W';
     LL = DD - WW;
   
    for view_mark = 1:view_num
        J(it_mark) = J(it_mark)+trace((data{view_mark}-bas{view_mark}*new_data)*(data{view_mark}-bas{view_mark}*new_data)')+beta*trace(bas{view_mark}*bas{view_mark}');
    end
    J(it_mark) = J(it_mark) + alpha*trace(new_data*LL*new_data')+gama*tempXN;
    if it_mark>=2 && abs((J(it_mark)-J(it_mark-1))/J(it_mark-1))< stoperr      
          break;
    end
end

   norm_new_data = repmat(sqrt(sum(new_data.*new_data)),size(new_data,1),1);
   norm_new_data = max(norm_new_data,1e-10);
   new_data = new_data./norm_new_data;

