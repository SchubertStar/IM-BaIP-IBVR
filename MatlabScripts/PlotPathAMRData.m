clear all
clc

% Load CSV file
filename1 = 'Nexus3DBB/rawROSbagdata/hedge_pos_ang/finalbig1noplateu';  
data1 = readtable(filename1);

filename2 = 'Nexus3DBB/rawROSbagdata/hedge_pos_ang/finalbig1spline.csv'; 
data2 = readtable(filename2);

% Extract camera x and y positions
cam_x1 = data1.cam_x;
cam_y1 = data1.cam_y;

cam_x2 = data2.cam_x;
cam_y2 = data2.cam_y;

% Plot camera trajectory
figure;
hold on
plot(cam_x1, cam_y1, '-o', 'MarkerSize', 4, 'LineWidth', 1.5, 'Color', [0 0 1]);
% plot(cam_x2, cam_y2, '-o', 'MarkerSize', 4, 'LineWidth', 1.5, 'Color', [1 0 0]);

% Set axis limits
xlim([0 8]);
ylim([0 8]);

%plot(2.4, 2.75, 'bo', 'MarkerSize', 16, 'LineWidth', 1.5);
plot(3.6, 3.7, 'ro', 'MarkerSize', 16, 'LineWidth', 1.5);

grid on;
xlabel('cam\_x [m]');
ylabel('cam\_y [m]');
title('Camera Trajectory');
legend('cam\_x vs cam\_y', 'NexusAMR', 'Location', 'Best');
hold off
