%% IBVR1FormResults.m
%% 
clearvars; 
clc;
bagFilename = 'rosbags/formation_test_1_1e-4KAL2.bag'; 
agentID     = 0;            
agentPos    = [3.612, 3.560];      
zoneR       = 0; 

% plotting colors
agentColor  = 'r';
pathColor   = 'k';   % black dashed path
startColor  = 'b';   % blue square
endColor    = 'r';   % red x

%% 2) LOAD ROSBAG & SELECT TOPICS
bag       = rosbag(bagFilename);
ibvrSel   = select(bag,'Topic','/nexus4/ibvr_output');
hedgeSel  = select(bag,'Topic','/nexus4/hedge_pos_ang');
ibvrMsgs  = readMessages(ibvrSel, 'DataFormat','struct');
hedgeMsgs = readMessages(hedgeSel,'DataFormat','struct');

N_ibvr  = numel(ibvrMsgs);
N_hedge = numel(hedgeMsgs);

%% 3) EXTRACT IBVR FOR ONE AGENT
time_ibvr   = nan(N_ibvr,1);
errorVec    = nan(N_ibvr,1);
distanceVec = nan(N_ibvr,1);
depthVec    = nan(N_ibvr,1);
depthEstVec = nan(N_ibvr,1);
cmdVx       = nan(N_ibvr,1);
cmdVy       = nan(N_ibvr,1);
cmdWz       = nan(N_ibvr,1);

for i = 1:N_ibvr
    m = ibvrMsgs{i};
    time_ibvr(i) = double(m.Header.Stamp.Sec) + m.Header.Stamp.Nsec*1e-9;
    ids = m.Ids;
    j = find(ids==agentID,1);
    if ~isempty(j)
        errorVec(i)    = m.Error(j);
        distanceVec(i) = m.Distance(j);
        depthVec(i)    = m.Depth(j);
        depthEstVec(i) = m.DepthEst(j);
    end
    cmdVx(i) = m.CmdVx;
    cmdVy(i) = m.CmdVy;
    cmdWz(i) = m.CmdWz;
end

%% 4) EXTRACT HEDGE POSITIONS
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

%% 5) TIME SYNC (nearest‐neighbor)
t_ibvr_rel  = time_ibvr  - time_ibvr(1);
t_hedge_rel = time_hedge - time_hedge(1);

posX_sync = interp1(t_hedge_rel, posX, t_ibvr_rel, 'nearest');
posY_sync = interp1(t_hedge_rel, posY, t_ibvr_rel, 'nearest');
posZ_sync = interp1(t_hedge_rel, posZ, t_ibvr_rel, 'nearest');
ang_sync  = interp1(t_hedge_rel, ang,  t_ibvr_rel, 'nearest');

t = t_ibvr_rel;  % use for all time‐axis plots

%% 6) ACTUAL DISTANCE + TRUE ERROR
actualDist = sqrt((posX_sync-agentPos(1)).^2 + (posY_sync-agentPos(2)).^2);
trueError  = distanceVec - actualDist;

%% 7) TRAJECTORY PLOT
figure; 
hold on; 
grid on;

hPath  = plot(posX_sync, posY_sync, ':',  'Color',pathColor, 'LineWidth',1.5);
hStart = plot(posX_sync(1), posY_sync(1), 's', 'Color',startColor,'MarkerSize',8,'LineWidth',1.5);
hEnd   = plot(posX_sync(end), posY_sync(end), 'x','Color',endColor,  'MarkerSize',8,'LineWidth',1.5);

hAgent = plot(agentPos(1), agentPos(2), 'o', 'Color',agentColor,'MarkerSize',8,'LineWidth',2);

theta = linspace(0,2*pi,200);
xc_z  = agentPos(1) + zoneR*cos(theta);
yc_z  = agentPos(2) + zoneR*sin(theta);
hZone = plot(xc_z, yc_z, '--', 'Color',agentColor, 'LineWidth',1.2);

xlim([0 8]); 
ylim([0 8]);
daspect([1 1 1]);

xlabel('X [m]'); 
ylabel('Y [m]');
title('NexusAMR Trajectory');

legend(...
  [hPath, hStart, hEnd, hAgent, hZone], ...
  {'AMR Path','Start','End', ...
   sprintf('Agent %d',agentID), 'Desired Zone'}, ...
  'Location','eastoutside' ...
);
%% 8) PER‐AGENT IBVR ERROR
figure; hold on; grid on;
axis tight
hErr = plot(t, errorVec, '-', 'Color',agentColor,'LineWidth',1.5);
yline(0,'--k','LineWidth',1.2,'HandleVisibility','off');
xlabel('Time [s]'); ylabel('Percieved Image-area error [e_{ij}]');
title('Per-Agent Perception (IBVR) Error');
legend(hErr, sprintf('Agent %d',agentID), 'Location','best');

%% 9) IBVR COMMANDED VELOCITIES
figure; hold on; grid on;
axis tight
plot(t, cmdWz, 'g-','LineWidth',1.5);
plot(t, cmdVx, 'b-','LineWidth',1.5);
plot(t, cmdVy, 'r-','LineWidth',1.5);
xlabel('Time [s]'); ylabel('Linear Velocity [m/s]');
title('Commanded Velocities');
legend('\omega','v_x','v_y','Location','best');

%% 10) PERCEIVED VS ACTUAL DISTANCE
figure; hold on; grid on;
axis tight
hPer = plot(t, distanceVec,  '-', 'Color',agentColor,'LineWidth',1.5);
hAct = plot(t, actualDist,    '--','Color',agentColor,'LineWidth',1.5);
yline(zoneR,'--k','LineWidth',1.2,'HandleVisibility','off');
xlabel('Time [s]'); ylabel('Distance [m]');
title('Perceived (IBVR) Distance vs local GNSS Distance');
legend([hPer,hAct], {'Perceived','Actual'}, 'Location','best');

%% 11) TRUE DISTANCE ERROR
figure; hold on; grid on;
hTE = plot(t,trueError,'-','Color',agentColor,'LineWidth',1.5);
yline(0,'--k','LineWidth',1.2,'HandleVisibility','off');
xlabel('Time [s]'); ylabel('Distance Error [m]');
title('True Distance Error (Perceived – Actual)');
legend(hTE, sprintf('Agent %d',agentID), 'Location','best');

%% 12) True Distance Error vs. Relative Yaw
% 1) map into [0,360)
relativeYawRaw = 0 - ang_sync;              % your original
relativeYaw    = mod(relativeYawRaw, 360);  % wrap negatives into 0–360

% 2) make the figure
figure; hold on; grid on;

% scatter of the raw samples
scatter(relativeYaw, trueError, 20, 'filled', 'MarkerFaceAlpha',0.6);

% sort & smooth for trend
[ry_s, idx]  = sort(relativeYaw);
te_s         = trueError(idx);
te_smooth    = movmean(te_s, 25);  % adjust window as desired
plot(ry_s, te_smooth, 'r-', 'LineWidth',2);

% 3) fix axis from 0 to 360
xlim([0 360]);
xticks(0:60:360);    % tick every 60°, adjust as you like

% 4) labels, title, legend, zero‐line
xlabel('Relative Yaw (°)');
ylabel('Local GNSS Distance Error (m)');
title(sprintf('Local GNSS Distance Error vs Relative Yaw',agentID));
yline(0, '--k','LineWidth',1.2,'HandleVisibility','off');
legend('Samples','Smoothed trend','Location','best');
