clear all
clf

eta = [0:0.0001:1];
BASE = 2;%exp(1);
M = [2:1:5];

% alpha等于psi hat
%alpha = ones(length(M),max(M)-1);
alpha = rand(length(M),max(M)-1);

for j = 1:length(M)
  %  C = log(M(j))/log(BASE); % i.e., m in the paper
    alpha(j,:) = alpha(j,:) / sum(alpha(j,[1:(M(j)-1)]));
    % 包含Null社区的分布
   % H(j) = -log(1/(M(j))) / log(BASE);
    %H_plot(j,:) = H(j) * ones(1, length(eta));
   % H_plot_M(j,:) = (H(j)+M(j)-1) * ones(1, length(eta));
   % h(j,:) = -eta.* log(eta) / log(BASE);
   % h_max(j,:) = -eta.* log(eta) / log(BASE);
   % hh(j,:) = -(1 - eta).* log(1-eta) / log(BASE);
    
    sigma = 0;
    if M(j) == 2
        sigma = 0.5;
    end
    
    % 除开Null社区的分布
    h_alpha(j) = 0;
    h_alpha_max(j) = 0;
    % 计算各种熵值
    for i = 1:M(j)-1
        %h(j) = h(j) - alpha(j,i)*(1-eta) .* log(alpha(j,i)*(1-eta)) / log(BASE);
        %h_max(j,:) = h_max(j,:) - (1-eta)./(M(j)-1) .* log((1-eta)./(M(j)-1)) / log(BASE);
        %hh(j,:) = hh(j,:) - alpha(j,i)*(eta) .* log(alpha(j,i)*(eta)) / log(BASE);
        
        % 除开Null社区的熵值，也就是alpha（psi hat）的熵值
        h_alpha(j) = h_alpha(j) - alpha(j,i) * log(alpha(j,i)) / log(BASE);
        h_alpha_max(j) = h_alpha_max(j) - 1/(M(j)-1) * log(1/(M(j)-1)) / log(BASE);
    end
    
    fix_eta = eta .* log(eta) / log(BASE) - eta;
    
    % 分裂指数 
    I_psi(j,:) =  (1- eta).* (h_alpha(j));%(1- eta).* h(j,:);% (1- eta).* (h(j,:) + fix_eta + C);% (1- eta).* (H(j) - hh(j,:) + C);% (h(j,:)) + eta.*(log(eta) / log(BASE) - 1) + 1;
    I_psi_max(j,:) =  (1- eta).* (h_alpha_max(j));%(h_max(j,:) + eta .* log(eta)/log(BASE)); %(1- eta + 1/M(j)).* (h_max(j,:)); %(1- eta).* (h_max(j,:) + fix_eta + C);
    % 差分作为分裂指数的导数
    for k=1:length(I_psi_max(j,:))-1
        d_psi(j,k) = I_psi_max(j,k+1) - I_psi_max(j,k);%H(j) - h(j,:) + eta./(1-eta).* h(j,:) + eta./(1-eta).*log(eta) / log(BASE) + C; % derivative of I_eta wrt. eta
    end
    d_psi(j,length(I_psi_max(j,:))) = d_psi(j,length(I_psi_max(j,:))-1);
    %d_psi(j,:) = -2*(h(j,:) + eta .* log(eta) / log(BASE)) + eta - C;%H(j) - hh(j,:) + (1-eta)./(eta).* hh(j,:) + (1-eta)./(eta).*log(eta) / log(BASE) + C; %-2 * h(j,:);
    
    % 缩减指数
    I_eta(j,:) =  eta .* (h_alpha_max(j) - I_psi(j,:) + eta .* sigma);%h_alpha_max(j,:) - I_psi(j,:);%eta .* (1 + h_alpha_max(j,:) - (1- eta).* h_alpha(j,:));%eta .* (H(j) - h(j,:)+C); % eta .* (H(j) - h(j,:));%eta.*(H(j) - h(j,:) - fix_eta + C); % random distribution of psi
    I_eta_max(j,:) = eta .* (h_alpha_max(j) - I_psi_max(j,:)+ eta .* sigma);%eta .* (1 + h_alpha_max(j,:) - (1- eta).* h_alpha_max(j,:));%(eta - 1/M(j)) .* (H(j) - h_max(j,:));%eta.*(H(j) - h_max(j,:)+C); % eta.*(H(j) - h_max(j,:)  - fix_eta + C); % even distribution of psi
    % 差分作为缩减指数的导数
    for k=1:length(I_eta_max(j,:))-1
        d_eta(j,k) = I_eta_max(j,k+1) - I_eta_max(j,k);%H(j) - h(j,:) + eta./(1-eta).* h(j,:) + eta./(1-eta).*log(eta) / log(BASE) + C; % derivative of I_eta wrt. eta
    end
    d_eta(j,length(I_eta_max(j,:))) = d_eta(j,length(I_eta_max(j,:))-1);
    
    %res(j,:) = (eta ./ (1-eta)) .* log(eta) / log(BASE);
end

%subplot(3,2,1);title("信息熵（包含Null社区），横轴为eta值");grid on;hold on;
%%plot(eta, h_max);legend("M="+M);
%plot(eta, h); % entropy of random distribution of psi given m and eta
%plot(eta, H_plot, 'b:'); % maximum entropy given m, i.e., -log( 1/(m+1) )
%plot(eta, h_max, '--'); % entropy given m and eta, even distribution of psi
%legend(["M=m+1="+M,"最大熵="+H]);
%hold off;


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