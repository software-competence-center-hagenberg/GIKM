function [] = demoGIKM()
[y_data_1,labels_1] = datagen(100,3);
y_data_2 = sin(0.75*pi*y_data_1)+0.1;
labels_2 = labels_1;
[CLF1] = Classifier(y_data_1,labels_1,2,1000);
[CLF2] = Classifier(y_data_2,labels_2,2,1000);
min_y = min([y_data_1 y_data_2],[],2)-0.3;
max_y = max([y_data_1 y_data_2],[],2)+0.3;
[tv1, tv2] = meshgrid(min_y(1):0.05:max_y(1), min_y(2):0.05:max_y(2));
y_data_new = ([tv1(:) tv2(:)])';
[colormatrix] = colorsorder;
close all;
figure,
for i = 1:3
    [hat_y_data] = outputParallelAutoencoders(y_data_new,CLF1.AE_arr_arr{i});
    plot(hat_y_data(1,:),hat_y_data(2,:),'o','color',colormatrix(i,:));
    hold on
    plot(y_data_1(1,labels_1==i),y_data_1(2,labels_1==i),'*','color',colormatrix(i,:)),
    xlabel('y_1','FontSize',14);
    ylabel('y_2','FontSize',14);
end
hold off;
[min_distance_arr{1},labels_arr_arr{1}] = predictionClassifier(y_data_new,CLF1);
decision_map = reshape(labels_arr_arr{1},size(tv1,1),size(tv1,2));
figure; imagesc([min_y(1) max_y(1)],[min_y(2) max_y(2)],decision_map);
hold on;
set(gca,'ydir','normal');
colormap(colormatrix(1:3,:));
for i = 1:3
    plot(y_data_1(1,labels_1==CLF1.labels(i)),y_data_1(2,labels_1==CLF1.labels(i)),'^','MarkerFaceColor',colormatrix(i,:),'MarkerEdgeColor','white');
end
xlabel('y_1','FontSize',14);
ylabel('y_2','FontSize',14);
hold off;

figure,
for i = 1:3
    [hat_y_data] = outputParallelAutoencoders(y_data_new,CLF2.AE_arr_arr{i});
    plot(hat_y_data(1,:),hat_y_data(2,:),'o','color',colormatrix(i,:));
    hold on
    plot(y_data_2(1,labels_2==i),y_data_2(2,labels_2==i),'*','color',colormatrix(i,:)),
    xlabel('y_1','FontSize',14);
    ylabel('y_2','FontSize',14);
end
hold off;
[min_distance_arr{2},labels_arr_arr{2}] = predictionClassifier(y_data_new,CLF2);
decision_map = reshape(labels_arr_arr{2},size(tv1,1),size(tv1,2));
figure; imagesc([min_y(1) max_y(1)],[min_y(2) max_y(2)],decision_map);
hold on;
set(gca,'ydir','normal');
colormap(colormatrix(1:3,:));
for i = 1:3
    plot(y_data_2(1,labels_2==CLF1.labels(i)),y_data_2(2,labels_2==CLF1.labels(i)),'^','MarkerFaceColor',colormatrix(i,:),'MarkerEdgeColor','white');
end
xlabel('y_1','FontSize',14);
ylabel('y_2','FontSize',14);
hold off;

figure,
for i = 1:3
    [hat_y_data_1,distance_1] = AutoencoderFiltering(y_data_new,CLF1.AE_arr_arr{i}{1});
    [hat_y_data_2,distance_2] = AutoencoderFiltering(y_data_new,CLF2.AE_arr_arr{i}{1});
    min_distance = min(distance_1,distance_2);
    hat_y_data = zeros(size(y_data_new));
    hat_y_data(:,distance_1==min_distance) = hat_y_data_1(:,distance_1==min_distance);
    hat_y_data(:,distance_2==min_distance) = hat_y_data_2(:,distance_2==min_distance);
    plot(hat_y_data(1,:),hat_y_data(2,:),'o','color',colormatrix(i,:));
    hold on
    plot(y_data_1(1,labels_1==i),y_data_1(2,labels_1==i),'*','color',colormatrix(i,:)),
    plot(y_data_2(1,labels_2==i),y_data_2(2,labels_2==i),'*','color',colormatrix(i,:)),
    xlabel('y_1','FontSize',14);
    ylabel('y_2','FontSize',14);
end
hold off;
[~,decisions] = combineMultipleClassifiers(min_distance_arr,labels_arr_arr);
decision_map = reshape(decisions,size(tv1,1),size(tv1,2));
figure; imagesc([min_y(1) max_y(1)],[min_y(2) max_y(2)],decision_map);
hold on;
set(gca,'ydir','normal');
colormap(colormatrix(1:3,:));
for i = 1:3
    plot(y_data_1(1,labels_1==CLF1.labels(i)),y_data_1(2,labels_1==CLF1.labels(i)),'^','MarkerFaceColor',colormatrix(i,:),'MarkerEdgeColor','white');
    plot(y_data_2(1,labels_2==CLF1.labels(i)),y_data_2(2,labels_2==CLF1.labels(i)),'^','MarkerFaceColor',colormatrix(i,:),'MarkerEdgeColor','white');
end
xlabel('y_1','FontSize',14);
ylabel('y_2','FontSize',14);
hold off;


return











function [X,y] = datagen(ns,nc)
X = [];
y = [];
for ic = 1:nc %ic: class index
    r = linspace(0.25,0.95,ns);
    t = linspace(ic*2, (ic+1)*2, ns) + 0*randn(1,ns);
    X = [X ; [(r.*sin(t))' (r.*cos(t))']];
    y = [y;ic*ones(ns,1)];
end
X = X';
y = y';
return