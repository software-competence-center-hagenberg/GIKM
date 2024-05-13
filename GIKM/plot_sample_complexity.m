function [] = plot_sample_complexity()
epsilon_arr = linspace(0.01,0.05,50);
N_arr = zeros(2,50);
N_arr(1,:) = (1./(epsilon_arr.^2))*((4 + sqrt(0.5*log(1/0.05)))^2);
N_arr(2,:) = (1./(epsilon_arr.^2))*((4 + sqrt(0.5*log(1/0.00005)))^2);
plot(epsilon_arr,N_arr(1,:),'-o');
hold on;
plot(epsilon_arr,N_arr(2,:),'-*');
grid on;
axis([0.01 0.05 1000 400000]);
xlabel('risk bound \epsilon','FontSize',18,'Interpreter','tex');
ylabel('lower bound on sample complexity','FontSize',18,'Interpreter','tex');
legend({'\delta = 0.05', '\delta = 0.00005'},'FontSize',18);
hold off;
return