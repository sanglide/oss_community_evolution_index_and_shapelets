clear all
clf

eta = [0:0.0001:1];
BASE = 2;%exp(1);
M = [2:1:5];
m=M-1;

% alpha等于psi hat
%alpha = ones(length(M),max(M)-1);
alpha = rand(length(M),max(M)-1);

for j = 1:length(M)
    alpha(j,:) = alpha(j,:) / sum(alpha(j,[1:(M(j)-1)]));
    % 除开Null社区的分布
    gini_alpha(j) = 1;
    gini_alpha_max(j) = 1;
    % 计算各种熵值
    for i = 1:M(j)-1
        % 除开Null社区的Gini Index，也就是psi hat的gini index值
        gini_alpha(j) = gini_alpha(j) - alpha(j,i) * alpha(j,i);
        gini_alpha_max(j) = gini_alpha_max(j) - (1/m(j)) * (1/m(j));
    end
    
    % 分裂指数 
    I_psi(j,:) =  (1- eta).* m(j) .* gini_alpha(j);
    I_psi_max(j,:) =  (1- eta).* m(j) .* gini_alpha_max(j);
    % 差分作为分裂指数的导数
    for k=1:length(I_psi_max(j,:))-1
        d_psi(j,k) = I_psi_max(j,k+1) - I_psi_max(j,k);
    end
    d_psi(j,length(I_psi_max(j,:))) = d_psi(j,length(I_psi_max(j,:))-1);
    
    
    % 缩减指数
    I_eta(j,:) =  eta .* (m(j)*gini_alpha_max(j) - I_psi(j,:));
    I_eta_max(j,:) =  eta .* (m(j)*gini_alpha_max(j) - I_psi_max(j,:));
    % 差分作为缩减指数的导数
    for k=1:length(I_eta_max(j,:))-1
        d_eta(j,k) = I_eta_max(j,k+1) - I_eta_max(j,k);%H(j) - h(j,:) + eta./(1-eta).* h(j,:) + eta./(1-eta).*log(eta) / log(BASE) + C; % derivative of I_eta wrt. eta
    end
    d_eta(j,length(I_eta_max(j,:))) = d_eta(j,length(I_eta_max(j,:))-1);
    
end

subplot(3,2,1);title("Gini Index，横轴为eta值");grid on;hold on;
%plot(eta, h_max);legend("M="+M);
plot(m, gini_alpha); % entropy of random distribution of psi given m and eta
plot(m, gini_alpha_max, 'b:'); % maximum entropy given m, i.e., -log( 1/(m+1) )
hold off;

subplot(3,2,3);hold on;
plot(eta, I_eta);  % shrink index, random distribution of psi, given eta and m
%plot(eta, H_plot_M, 'b:'); % maximum entropy + m, which is the maximum of shrink index obtained when eta==1 given m
plot(eta, I_eta_max,'r:'); % even distribution of psi, given eta and m
legend("M="+M);
title("缩减指数（实线对应随机分布，虚线对应均匀分布最大熵），横轴为eta值");grid on;hold off;

subplot(3,2,4);plot(eta, d_eta);title("缩减指数对eta求导");legend("M=m+1="+M);grid on; % derivative of shrink index wrt eta, given m, always positive

%subplot(2,2,3);plot([1:M-1], alpha);title("alpha");grid on;
subplot(3,2,5);hold on;
plot(eta, I_psi);
plot(eta, I_psi_max,'r:');
legend("M="+M);
title("分裂指数（实线对应随机分布，虚线对应均匀分布最大熵），横轴为eta值");grid on;hold off;

subplot(3,2,6);plot(eta, d_psi);title("分裂指数对eta求导");legend("M=m+1="+M);grid on; 

%subplot(2,2,4);plot(eta, res);title("eta .* log(eta) ./ (1-eta)");grid on;