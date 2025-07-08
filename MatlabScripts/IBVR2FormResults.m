%% IBVR2FormResults.m
%% 
clear all;
clc;

bagFilename = 'formation_test_2_left1e-4KAL.bag';  % ← your file here
bag = rosbag(bagFilename);

%% 
ibvrSel  = select(bag, 'Topic','/nexus4/ibvr_output');
hedgeSel = select(bag, 'Topic','/nexus4/hedge_pos_ang');

%%
ibvrMsgs  = readMessages(ibvrSel,  'DataFormat','struct');
hedgeMsgs = readMessages(hedgeSel, 'DataFormat','struct');

N_ibvr   = numel(ibvrMsgs);
N_hedge  = numel(hedgeMsgs);
fprintf('Read %d IBVR msgs and %d hedge_pos msgs\n', N_ibvr, N_hedge);

%%
disp('--- IBVROutput fields ---')
disp(fieldnames(ibvrMsgs{1}));
disp('--- hedge_pos fields ---')
disp(fieldnames(hedgeMsgs{1}));

%%
agentIDs = [0, 1];         
nAgents  = numel(agentIDs);

time_ibvr    = nan(N_ibvr,1);
errorMat     = nan(N_ibvr, nAgents);
distanceMat  = nan(N_ibvr, nAgents);
depthMat     = nan(N_ibvr, nAgents);
depthEstMat  = nan(N_ibvr, nAgents);
cmdVx        = nan(N_ibvr,1);
cmdVy        = nan(N_ibvr,1);
cmdWz        = nan(N_ibvr,1);

%%
for i = 1:N_ibvr
    m = ibvrMsgs{i};
    time_ibvr(i) = double(m.Header.Stamp.Sec) + m.Header.Stamp.Nsec*1e-9;
    ids = m.Ids;
    for j = 1:nAgents
        idx = ids(j) + 1;
        errorMat(i,    idx) = m.Error(j);
        distanceMat(i, idx) = m.Distance(j);
        depthMat(i,    idx) = m.Depth(j);
        depthEstMat(i, idx) = m.DepthEst(j);
    end
    cmdVx(i) = m.CmdVx;
    cmdVy(i) = m.CmdVy;
    cmdWz(i) = m.CmdWz;
end

%%
time_hedge = nan(N_hedge,1);
posX       = nan(N_hedge,1);
posY       = nan(N_hedge,1);
posZ       = nan(N_hedge,1);
ang        = nan(N_hedge,1);
for i = 1:N_hedge
    h = hedgeMsgs{i};
    time_hedge(i) = double(h.TimestampMs)*1e-3;
    posX(i)       = h.XM;
    posY(i)       = h.YM;
    posZ(i)       = h.ZM;
    ang(i)        = h.Angle;
end
%%
time_ibvr  = time_ibvr  - time_ibvr(1);
time_hedge = time_hedge - time_hedge(1);

posX_sync = nan(N_ibvr,1);
posY_sync = nan(N_ibvr,1);
posZ_sync = nan(N_ibvr,1);
ang_sync  = nan(N_ibvr,1);

for i = 1:N_ibvr
    [~, j] = min(abs(time_hedge - time_ibvr(i)));
    posX_sync(i) = posX(j);
    posY_sync(i) = posY(j);
    posZ_sync(i) = posZ(j);
    ang_sync(i)  = ang(j);
end
%%
t0 = time_ibvr(1);
t  = time_ibvr - t0;

%%
ax = [0.641, 2.212];    % agent x‐centers
ay = [5.381, 7.200];    % agent y‐centers
colors = {'r','b'};
ccolors = {'b','r'};
colors2 = { ...
    'm', ...                 % shorthand magenta
    [0.5, 0, 0.5] ...        % custom purple
};
Rzone = 3.5;

%%
figure; 
hold on; 
grid on;

% 1) AMR path + start/end markers
plot(posX, posY, 'k:','LineWidth',1.5);
plot(posX(1), posY(1), 'bs','MarkerSize',8,'LineWidth',1.5);
plot(posX(end), posY(end), 'rx','MarkerSize',8,'LineWidth',1.5);

% 2) Agent 0 marker + zone
plot(ax(1), ay(1), 'o', 'Color', ccolors{1}, 'MarkerSize',8, 'LineWidth',2);
text(ax(1)+0.3, ay(1), 'Agent 0', 'FontSize',11, 'Color', ccolors{1}, ...
     'HorizontalAlignment','left', 'VerticalAlignment','middle');
xc0 = ax(1) + Rzone*cos(theta);
yc0 = ay(1) + Rzone*sin(theta);
plot(xc0, yc0, '--', 'Color', ccolors{1}, 'LineWidth',1.2);

% 3) Agent 1 marker + zone
plot(ax(2), ay(2), 'o', 'Color', ccolors{2}, 'MarkerSize',8, 'LineWidth',2);
text(ax(2)+0.3, ay(2), 'Agent 1', 'FontSize',11, 'Color', ccolors{2}, ...
     'HorizontalAlignment','left', 'VerticalAlignment','middle');
xc1 = ax(2) + Rzone*cos(theta);
yc1 = ay(2) + Rzone*sin(theta);
plot(xc1, yc1, '--', 'Color', ccolors{2}, 'LineWidth',1.2)

% 3) enforce 0–8 m bounds & square aspect ratio
xlim([0 8]); 
ylim([0 8]);
daspect([1 1 1]);

% 4) labels + legend (only Path, Zone 0, Zone 1)
xlabel('X [m]'); 
ylabel('Y [m]');
title('NexusAMR Trajectory');
legend('AMR Path','Start','End', ...
       'Agent 0','Zone 0', ...
       'Agent 1','Zone 1', ...
       'Location','eastoutside');

%%
figure; hold on; grid on;
valid = find(any(~isnan(errorMat),1));

h2 = plot(t, errorMat(:,valid(2)), '-b', 'LineWidth',1.5);   % Agent 1
h1 = plot(t, errorMat(:,valid(1)), '-r', 'LineWidth',1.5);   % Agent 0

yline(0, '--k', 'LineWidth',1.5, 'HandleVisibility','off');

% tighten y-axis
axis tight

xlabel('Time [s]');
ylabel('Percieved Image-area error [e_{ij}]');
title('Per-Agent Perception (IBVR) Error');

legend([h1, h2], {'Agent 0','Agent 1'}, 'Location','best');

%%
figure; hold on; grid on;
plot(t, cmdWz, 'g-','LineWidth',1.5);
plot(t, cmdVx, 'b-','LineWidth',1.5);
plot(t, cmdVy, 'r-','LineWidth',1.5);
axis tight
xlabel('Time [s]'); ylabel('Linear Velocity [m/s]');
title('Command Velocities');
legend('v_x','v_y', '\omega','Location','best');

%%
posX_sync = posX_sync(:);
posY_sync = posY_sync(:);

actualDist_sync = nan(N_ibvr, nAgents);

for i = 1:N_ibvr
    for j = 1:nAgents
        dx = posX_sync(i) - ax(j);
        dy = posY_sync(i) - ay(j);
        actualDist_sync(i,j) = sqrt(dx*dx + dy*dy);
    end
end

%--- 3) plot perceived vs actual, pairing by idx
figure; hold on; grid on;

hPer = gobjects(nAgents,1);
hAct = gobjects(nAgents,1);
for idx = 1:nAgents
    hPer(idx) = plot( ...
        t, distanceMat(:,idx), ...
        'Color', colors{idx}, ...
        'LineWidth',  1.5 );

    hAct(idx) = plot( ...
        t, actualDist_sync(:,idx), '--', ...
        'Color', colors2{idx}, ...
        'LineWidth',  1.5 );
end

%--- 4) styling
xlabel('Time [s]');
ylabel('Distance [m]');
title('Perceived (IBVR) Distance vs local GNSS Distance');
axis tight

% dashed zero‐line if you like
yline(3.5,'--k','LineWidth',1.2,'HandleVisibility','off');

% legend: Perceived 0, Actual 0, Perceived 1, Actual 1
legend(...
  [hPer(1), hAct(1), hPer(2), hAct(2)], ...
  {'Perceived 0','Actual 0','Perceived 1','Actual 1'}, ...
  'Location','best' ...
);

hold off;

%%

error_sync = distanceMat - actualDist_sync;

figure; 
hold on; 
grid on;

% reuse your existing color cell (unwrap with {})
for idx = 1:2
    plot( ...
      t, error_sync(:,idx), ...
      'Color',     colors{idx}, ...
      'LineWidth', 1.5 );
end

% zero-error dashed line
yline(0, '--k', 'LineWidth',1.2, 'HandleVisibility','off');

% styling
xlabel('Time [s]');
ylabel('Distance Error [m]');
title('Distance Error (Perceived − local GNSS)');
axis tight

% legend for each agent
legend( ...
  arrayfun(@(k) sprintf('Agent %d',k-1), 1:nAgents, 'Uni',false), ...
  'Location','best' ...
);

hold off;

