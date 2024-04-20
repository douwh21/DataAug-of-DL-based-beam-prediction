% Standard deviations of noise injection /sigma evaluation
figure;
hold on;
xlabel('Epoch', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;
load('./evaluation/DataAug_64beam_supervised_640_lr=0.03_evaluation.mat');
markersize = 10;
BL = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
plot(1:length(BL), BL,'LineWidth', 1.5,'Color','c');

load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.01_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL,'m','LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.05_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'g','LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.1_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'Color',[0.8500 0.3250 0.0980], 'LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.2_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'color', [0.5 0 0.5], 'LineWidth', 1.5)


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.3_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'r', 'LineWidth', 1.5)


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.4_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL,'b',  'LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.5_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL,  'color', [0 0.5 0.5],'LineWidth', 1.5);

legend({'$\sigma=0$',"$\sigma=0.01$","$\sigma=0.05$","$\sigma=0.1$","$\sigma=0.2$",'$\sigma=0.3$',"$\sigma=0.4$","$\sigma=0.5$"},'interpreter','latex')
saveas(gcf,"./figures/figure1",'epsc')



% Different settings of power Z in label augmentation method evaluation
figure;
hold on;
xlabel('Epoch', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;
load('./evaluation/DataAug_64beam_supervised_640_lr=0.03_evaluation.mat');
BL = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
plot(1:length(BL), BL,'Color',[0.8500 0.3250 0.0980],'LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=1.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL,'m','LineWidth', 1.5);

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=2.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'g','LineWidth', 1.5);

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=4.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'b',  'LineWidth', 1.5);

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=8.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL,  'color', [0 0.5 0.5], 'LineWidth', 1.5);

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=16.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'color', [0.5 0 0.5], 'LineWidth', 1.5)

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Z=32.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'color', [0.5 0.5 0],'LineWidth', 1.5);


load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Zf=8.0_Z0=2.0_k=6.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
plot(1:length(BL), BL, 'r', 'LineWidth', 1.5);

legend('One-hot [11]',"Z=1","Z=2",'Z=4','Z=8','Z=16','Z=32','Adaptive power scheduler','Interpreter', 'latex')
saveas(gcf,"./figures/figure2",'epsc')



% Ablation study
figure
hold on;
xlabel('Methods', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;
Gn = zeros(1,6);

load('./evaluation/DataAug_64beam_supervised_640_lr=0.03_evaluation.mat');
BL = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
Gn(1)=max(BL);

load('./evaluation/DataAug_64beam_supervised_640_cyclicshift_lr=0.03_p=0.5_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
Gn(2)=max(BL);

load('./evaluation/DataAug_64beam_supervised_640_flipping_lr=0.03_p=0.5_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
Gn(3)=max(BL);

load('./evaluation/DataAug_64beam_supervised_640_noiseinjection_lr=0.03_p=0.5_sigma=0.3_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
Gn(4)=max(BL);

load('./evaluation/DataAug_64beam_supervised_640_labelaug_lr=0.03_Zf=8.0_Z0=2.0_k=6.0_evaluation.mat');
BL = mean(squeeze(mean(BL_eval, 1)), 1);
Gn(5)=max(BL);

load('./evaluation/DataAug_64beam_supervised_640_ourapproach_lr=0.03_p=0.5_Zf=8.0_Z0=2.0_k=6.0_sigma=0.3_evaluation.mat');
BL = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
Gn(6)=max(BL);

color=[[0 0 1];[0 1 0];[0.9290 0.6940 0.1250];[0 1 1];[1 0 1];[1 0 0]];
values = categorical({'A','B','C','D','E','F'});
for i=1:6
    b = bar(values(i),Gn(i),'FaceColor','flat','BarWidth',0.6);
    b.CData(1,:)=color(i,:);
    set(b,'edgecolor','none')
end
set(gca,'XTick',[])
legend("Conventional supervised learning [11]","Input cyclic shifting","Input flipping","Input noise injection","Adaptive label augmentation","Our proposed approach", 'Interpreter', 'latex')
saveas(gcf,"./figures/figure3",'epsc')



% Performance comparison of our proposed approach and conventional supervised learning
figure;
hold on
xlabel('Size of training dataset', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;
power_ratios = zeros(5, 4);
for k=0:1:4
    count=160*4^k;
    load(['./evaluation/DataAug_64beam_supervised_' num2str(count) '_lr=0.03_evaluation.mat']);
    BL1 = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
    BL2 = mean(squeeze(mean(BL_eval(:,3,:,:), 1)), 1);
    power_ratios(k+1, 1) = max(BL1);
    power_ratios(k+1, 2) = max(BL2);
    load(['./evaluation/DataAug_64beam_supervised_' num2str(count) '_ourapproach_lr=0.03_p=0.5_Zf=8.0_Z0=2.0_k=6.0_sigma=0.3_evaluation.mat']);
    BL1 = mean(squeeze(mean(BL_eval(:,1,:,:), 1)), 1);
    BL2 = mean(squeeze(mean(BL_eval(:,3,:,:), 1)), 1);
    power_ratios(k+1, 3) = max(BL1);
    power_ratios(k+1, 4) = max(BL2);
end
plot([160,640,2560,10240,40960], power_ratios(:, 1), 'b*-', 'LineWidth', 1.5);
plot([160,640,2560,10240,40960], power_ratios(:, 2), 'b*--', 'LineWidth', 1.5);
plot([160,640,2560,10240,40960], power_ratios(:, 3), 'ro-', 'LineWidth', 1.5);
plot([160,640,2560,10240,40960], power_ratios(:, 4), 'ro--', 'LineWidth', 1.5);
set(gca,'XScale','log')
xlim([160,40960])
xticks([160 640 2560 10240 40960])
set(gca,'XMinorTick','off','XMinorGrid','off')
legend("Conventional supervised learning top-1 [11]","Conventional supervised learning top-3 [11]",'Our proposed approach top-1','Our proposed approach top-3','Interpreter', 'latex')
saveas(gcf,"./figures/figure4",'epsc')



