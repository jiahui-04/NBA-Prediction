%% NBA winner prediction
clc; clear; close all;

%% Load Advanced Stats
path_adv = '/Users/chowjiahui/Documents/MATLAB/Project/NBA Team Advance Datasets';
seasons = 2015:2024;
allStats_adv = struct([]);

for i = 1:length(seasons)
    seasonStart = seasons(i);
    seasonEnd = seasonStart + 1;
    fileName = sprintf('%d-%d NBA Season Summary.csv', seasonStart, seasonEnd);
    in_file = fullfile(path_adv, fileName);
    
    if ~exist(in_file, 'file')
        error('File not found: %s', in_file);
    end
    
    [Team, Age, W, L, PW, PL, MOV, SOS, SRS, ORtg, DRtg, NRtg, Pace, FTr, ...
     ThreePAr, TS, OffeFG, OffTOV, OffORB, OffFTFGA, DefeFG, DefTOV, DefDRB, DefFTFGA] = ...
     textread(in_file, ...
     '%s %f %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', ...
     'delimiter', ',', 'headerlines', 2);

    % Store into struct
    allStats_adv(i).Season = seasonStart;
    allStats_adv(i).Team = Team;
    allStats_adv(i).Age = Age;
    allStats_adv(i).W = W;
    allStats_adv(i).L = L;
    allStats_adv(i).PW = PW;
    allStats_adv(i).PL = PL;
    allStats_adv(i).MOV = MOV;
    allStats_adv(i).SOS = SOS;
    allStats_adv(i).SRS = SRS;
    allStats_adv(i).ORtg = ORtg;
    allStats_adv(i).DRtg = DRtg;
    allStats_adv(i).NRtg = NRtg;
    allStats_adv(i).Pace = Pace;
    allStats_adv(i).FTr = FTr;
    allStats_adv(i).ThreePAr = ThreePAr;
    allStats_adv(i).TS = TS;
    allStats_adv(i).OffeFG = OffeFG;
    allStats_adv(i).OffTOV = OffTOV;
    allStats_adv(i).OffORB = OffORB;
    allStats_adv(i).OffFTFGA = OffFTFGA;
    allStats_adv(i).DefeFG = DefeFG;
    allStats_adv(i).DefTOV = DefTOV;
    allStats_adv(i).DefDRB = DefDRB;
    allStats_adv(i).DefFTFGA = DefFTFGA;
end

%% Combine advanced stats into one table
allTeams_adv = [];
for i = 1:length(allStats_adv)
    if isempty(allStats_adv(i).Team)
        continue;
    end
    nTeams = length(allStats_adv(i).Team);
    S = struct();
    S.Season = repmat(allStats_adv(i).Season, nTeams, 1);
    S.Team = allStats_adv(i).Team;
    S.Age = allStats_adv(i).Age;
    S.W = allStats_adv(i).W;
    S.L = allStats_adv(i).L;
    S.PW = allStats_adv(i).PW;
    S.PL = allStats_adv(i).PL;
    S.MOV = allStats_adv(i).MOV;
    S.SOS = allStats_adv(i).SOS;
    S.SRS = allStats_adv(i).SRS;
    S.ORtg = allStats_adv(i).ORtg;
    S.DRtg = allStats_adv(i).DRtg;
    S.NRtg = allStats_adv(i).NRtg;
    S.Pace = allStats_adv(i).Pace;
    S.FTr = allStats_adv(i).FTr;
    S.ThreePAr = allStats_adv(i).ThreePAr;
    S.TS = allStats_adv(i).TS;
    S.OffeFG = allStats_adv(i).OffeFG;
    S.OffTOV = allStats_adv(i).OffTOV;
    S.OffORB = allStats_adv(i).OffORB;
    S.OffFTFGA = allStats_adv(i).OffFTFGA;
    S.DefeFG = allStats_adv(i).DefeFG;
    S.DefTOV = allStats_adv(i).DefTOV;
    S.DefDRB = allStats_adv(i).DefDRB;
    S.DefFTFGA = allStats_adv(i).DefFTFGA;
    allTeams_adv = [allTeams_adv; struct2table(S)];
end

%% Load Basic Stats 
path_basic = '/Users/chowjiahui/Documents/MATLAB/Project/NBA Team Traditional Datasets';
allStats_basic = struct([]);

for i = 1:length(seasons)
    seasonStart = seasons(i);
    seasonEnd = seasonStart + 1;
    fileName = sprintf('%d-%d NBA Season Summary.csv', seasonStart, seasonEnd);
    in_file = fullfile(path_basic, fileName);
    
    if ~exist(in_file, 'file')
        warning('⚠️ File not found: %s — skipping...', in_file);
        continue;
    end
    
    [Team, GP, W, L, WIN_PCT, Min, PTS, FGM, FGA, FG_PCT, ... 
     ThreePM, ThreePA, ThreeP_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, ...
     AST, TOV, STL, BLK, BLKA, PF, PFD, PlusMinus] = ...
    textread(in_file, ...
     '%s %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', ...
     'delimiter', ',', 'headerlines', 1);

    % Store into struct
    allStats_basic(i).Season = seasonStart;
    allStats_basic(i).Team = Team;
    allStats_basic(i).GP = GP;
    allStats_basic(i).WIN_PCT = WIN_PCT;
    allStats_basic(i).Min = Min;
    allStats_basic(i).PTS = PTS;
    allStats_basic(i).FGM = FGM;
    allStats_basic(i).FGA = FGA;
    allStats_basic(i).FG_PCT = FG_PCT;
    allStats_basic(i).ThreePM = ThreePM;
    allStats_basic(i).ThreePA = ThreePA;
    allStats_basic(i).ThreeP_PCT = ThreeP_PCT;
    allStats_basic(i).FTM = FTM;
    allStats_basic(i).FTA = FTA;
    allStats_basic(i).FT_PCT = FT_PCT;
    allStats_basic(i).OREB = OREB;
    allStats_basic(i).DREB = DREB;
    allStats_basic(i).REB = REB;
    allStats_basic(i).AST = AST;
    allStats_basic(i).TOV = TOV;
    allStats_basic(i).STL = STL;
    allStats_basic(i).BLK = BLK;
    allStats_basic(i).BLKA = BLKA;
    allStats_basic(i).PF = PF;
    allStats_basic(i).PFD = PFD;
    allStats_basic(i).PlusMinus = PlusMinus;
end

%% Combine basic stats into one table
allTeams_basic = [];
for i = 1:length(allStats_basic)
    if isempty(allStats_basic(i).Team)
        continue;
    end
    nTeams = length(allStats_basic(i).Team);
    S = struct();
    S.Season = repmat(allStats_basic(i).Season, nTeams, 1);
    S.Team = allStats_basic(i).Team;
    S.GP = allStats_basic(i).GP;
    S.WIN_PCT = allStats_basic(i).WIN_PCT;
    S.Min = allStats_basic(i).Min;
    S.PTS = allStats_basic(i).PTS;
    S.FGM = allStats_basic(i).FGM;
    S.FGA = allStats_basic(i).FGA;
    S.FG_PCT = allStats_basic(i).FG_PCT;
    S.ThreePM = allStats_basic(i).ThreePM;
    S.ThreePA = allStats_basic(i).ThreePA;
    S.ThreeP_PCT = allStats_basic(i).ThreeP_PCT;
    S.FTM = allStats_basic(i).FTM;
    S.FTA = allStats_basic(i).FTA;
    S.FT_PCT = allStats_basic(i).FT_PCT;
    S.OREB = allStats_basic(i).OREB;
    S.DREB = allStats_basic(i).DREB;
    S.REB = allStats_basic(i).REB;
    S.AST = allStats_basic(i).AST;
    S.TOV = allStats_basic(i).TOV;
    S.STL = allStats_basic(i).STL;
    S.BLK = allStats_basic(i).BLK;
    S.BLKA = allStats_basic(i).BLKA;
    S.PF = allStats_basic(i).PF;
    S.PFD = allStats_basic(i).PFD;
    S.PlusMinus = allStats_basic(i).PlusMinus;
    allTeams_basic = [allTeams_basic; struct2table(S)];
end

%% Clean team names, sort, and combine

% Remove '*' and extra spaces, make uppercase
cleanTeamName = @(x) upper(strtrim(strrep(x, '*', '')));
allTeams_adv.Team = cleanTeamName(allTeams_adv.Team);
allTeams_basic.Team = cleanTeamName(allTeams_basic.Team);

% Standardize team names (fix LA Clippers mismatch)
allTeams_adv.Team = strrep(allTeams_adv.Team, 'LOS ANGELES CLIPPERS', 'LA CLIPPERS');

% Sort tables by Season and Team alphabetically
allTeams_adv = sortrows(allTeams_adv, {'Season','Team'});
allTeams_basic = sortrows(allTeams_basic, {'Season','Team'});

% Merge tables on Season + Team
combinedTable = innerjoin(allTeams_adv, allTeams_basic, 'Keys', {'Season','Team'});

% Check results
fprintf('\n=== COMBINED TABLE SUMMARY ===\n');
for s = seasons
    count = sum(combinedTable.Season == s);
    fprintf('Season %d-%d: %d teams\n', s, s+1, count);
end
fprintf('\nTotal rows: %d\n', height(combinedTable));
fprintf('Total columns: %d\n', width(combinedTable));

% Display final data table
disp(combinedTable)

%% Compute target variable
combinedTable.WinRate = combinedTable.W ./ (combinedTable.W + combinedTable.L);
targetVar = 'WinRate';

%% Select numeric predictors 
numericVars = varfun(@isnumeric, combinedTable, 'OutputFormat', 'uniform');
predictorNames = combinedTable.Properties.VariableNames(numericVars);
predictorNames(strcmp(predictorNames, targetVar)) = [];
X = combinedTable{:, predictorNames};
y = combinedTable.(targetVar);

%% Standardize predictors & target 
X = (X - mean(X,"omitnan")) ./ std(X,"omitnan");
y = (y - mean(y,"omitnan")) ./ std(y,"omitnan");

%% Evaluate each metric 
results = [];
for i = 1:length(predictorNames)
    xi = X(:, i);
    valid = ~isnan(xi) & ~isnan(y);
    xi_v = xi(valid);
    y_v  = y(valid);

    G = [ones(size(xi_v)), xi_v];
    m = (G' * G) \ (G' * y_v);
    y_pred = G * m;

    SS_tot = sum((y_v - mean(y_v)).^2);
    SS_res = sum((y_v - y_pred).^2);
    R2 = 1 - SS_res / SS_tot;

    C = corrcoef(xi_v, y_v);
    r = C(1,2);

    results = [results; {predictorNames{i}, R2, r}];
end

%% Display results 
resultsTable = cell2table(results, 'VariableNames', {'Metric','R2','Correlation'});
% Get the correlation values and sort indices
correlationValues = resultsTable.Correlation;
[~, sortIdx] = sort(abs(correlationValues), 'descend');
resultsTable = resultsTable(sortIdx, :);
disp('🏆 Metrics ranked by predictive strength:');
disp(resultsTable);

%% Top 15 metrics by correlation
topMetrics = resultsTable(1:15, :);
disp('🏆 Top 15 Metrics');
disp(topMetrics);

%% ----- Model 1 (Top 5 metrics) -----
% Visualize Model 1 Metrics vs Win%

selectedMetrics = {'W', 'L', 'MOV', 'NRtg', 'SRS'};
targetVar = 'WinRate'; 

y = combinedTable.WIN_PCT;

for i = 1:length(selectedMetrics)
    metric = selectedMetrics{i};
    x = combinedTable.(metric);
    
    % Remove NaN values
    valid = ~isnan(x) & ~isnan(y);
    x_v = x(valid);
    y_v = y(valid);
    
    % Linear regression
    Xmat = [ones(size(x_v)), x_v];
    beta = Xmat \ y_v;
    y_pred = Xmat * beta;
    
    % Compute R² and correlation
    SS_res = sum((y_v - y_pred).^2);
    SS_tot = sum((y_v - mean(y_v)).^2);
    R2 = 1 - SS_res / SS_tot;
    r = corr(x_v, y_v);
    
    % Create new figure for each metric
    figure('Color','w');
    scatter(x_v, y_v, 'filled'); hold on;
    plot(x_v, y_pred, 'r', 'LineWidth', 2);
    xlabel(metric, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Win Percentage', 'FontWeight', 'bold', 'Color', 'k');
    title(sprintf('%s vs Win%% (R²=%.3f, r=%.3f)', metric, R2, r), ...
          'FontWeight', 'bold', 'Color', 'k');
    grid on;
    
    % Set all axes and tick labels to black
    ax = gca;
    ax.XColor = 'k';
    ax.YColor = 'k';
    ax.GridColor = [0.2 0.2 0.2];  % dark gray grid
    ax.FontSize = 11;
    ax.Color = 'w';  % background stays white
end

%% Correlation Heatmap for Model 1 Metrics
heatmapMetrics = [{'WIN_PCT'}, selectedMetrics];  % include target + predictors

% Extract data for selected metrics
data = combinedTable(:, heatmapMetrics);
dataArray = table2array(data);

% Compute correlation matrix
corrMat = corr(dataArray, 'Rows', 'complete');  % handle NaNs

% Make upper triangle NaN (to show only lower half)
mask = triu(true(size(corrMat)), 1);
corrMat(mask) = NaN;

% Plot one-sided (lower-triangle) heatmap
figure('Color','w','Position',[600 300 600 500]);
h = heatmap(heatmapMetrics, heatmapMetrics, corrMat, ...
    'Colormap', parula, ...
    'ColorLimits', [-1 1], ...
    'CellLabelFormat', '%.2f');

% Customize visuals
h.Title = 'One-Sided Correlation Heatmap: Win% vs Selected NBA Metrics';
h.XLabel = 'Metrics';
h.YLabel = 'Metrics';
h.FontColor = 'k';     
h.FontSize = 12;

% Optional: Adjust font sizes safely
ax = findobj(gcf, 'Type', 'Axes');

%% Predict the most recent season (2024–2025)
latestSeason = max(combinedTable.Season);
trainIdx = combinedTable.Season < latestSeason;   % 2015–2023 for training
testIdx = combinedTable.Season == latestSeason;   % 2024 for testing

% Training data
X_train = combinedTable{trainIdx, selectedMetrics};  % train features
y_train = combinedTable.W(trainIdx) ./ (combinedTable.W(trainIdx) + combinedTable.L(trainIdx));  % train target

% Testing data
X_test  = combinedTable{testIdx, selectedMetrics};   % test features
y_test  = combinedTable.W(testIdx) ./ (combinedTable.W(testIdx) + combinedTable.L(testIdx));   % test target
teams_test = combinedTable.Team(testIdx);           % team names for testing

%% Train the regression model on past data
X_design_train = [ones(size(X_train, 1), 1), X_train];
beta = inv(X_design_train' * X_design_train) * X_design_train' * y_train;
disp(beta)

%% Predict Win% for the latest season
X_design_test = [ones(size(X_test, 1), 1), X_test];
predicted_WinPct_test = X_design_test * beta;

%% Compare predictions vs actuals
results_2024 = table(teams_test, y_test, predicted_WinPct_test, ...
    'VariableNames', {'Team', 'Actual_WinPct', 'Predicted_WinPct'});

disp('=== Predicted vs Actual Win% for 2024–2025 Season ===');
disp(results_2024);

%% Compute model accuracy for the latest season
R2_2024 = 1 - sum((y_test - predicted_WinPct_test).^2) / sum((y_test - mean(y_test)).^2);
fprintf('Model 1 R² (on 2024–2025 season): %.4f\n', R2_2024);

MAE = mean(abs(y_test - predicted_WinPct_test));
fprintf('Mean Absolute Error (on 2024–2025 season): %.4f\n', MAE);

%% Visualization – Actual vs Predicted Win%
figure;
bar(categorical(results_2024.Team), [results_2024.Actual_WinPct, results_2024.Predicted_WinPct]);
legend('Actual Win%', 'Predicted Win%', 'Location', 'northwest');
xlabel('Team');
ylabel('Win Percentage');
title(sprintf('Actual vs Predicted NBA Win%% (%d–%d Season)', latestSeason, latestSeason+1));
grid on;

%% Print predicted champion
[~, bestIdx] = max(results_2024.Predicted_WinPct);
fprintf('\n🏆 Predicted NBA Champion (%d–%d): %s\n', ...
    latestSeason, latestSeason+1, results_2024.Team{bestIdx});

%% Load team data: locations & win probability
% Load team locations 
directory = '/Users/chowjiahui/Documents/MATLAB';
file_name = 'stadiums.csv';
inFile = fullfile(directory, file_name);

if exist(inFile, 'file') % test if file exists
    [team_abbrevs, team, ~, lat, lon] = textread(inFile, ...
        '%s %s %s %f %f', 'delimiter', ',', 'headerlines', 1);
else
    error('This file does not exist.')
end

% Convert team names to uppercase
team = upper(team);

% Rename "LOS ANGELES CLIPPERS" to "LA CLIPPERS"
team(strcmp(team, 'LOS ANGELES CLIPPERS')) = {'LA CLIPPERS'};

%% Mapping predicted win probabilities using color-coded markers
% Create base map
% specify lat & lon limits
latlimit = [25 50];
lonlimit = [-125 -66];

figure;
axesm("MapProjection","mercator", "MapLatLimit",latlimit, ...
    "MapLonLimit",lonlimit) % Mercator projection
% show land area of USA 
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9])
hold on

% Load state borders of USA & Canada (for Toronto Raptors)
    load("usastates.mat");
    bordersLat = [usastates.Lat];
    bordersLon = [usastates.Lon];
    state_names = {usastates.Name};
    geoshow(bordersLat, bordersLon, 'Color', 'k')
    hold on

% load canadian province boundary for Toronto Raptors 
    province_file = "/Users/chowjiahui/Documents/MATLAB/PROVINCE.SHP";
    state_names = [state_names, 'Ontario'];
    geoshow(province_file, "FaceColor", [0.9 0.9 0.9]);

% Plot with color-coded markers based on predicted win probability
% Create colormap: red (low) -> yellow (medium) -> green (high)
cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);

for v = 1:length(team_abbrevs)
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(v) * 99) + 1;
    marker_color = cmap(color_idx, :);
    
    % Plot points
    plotm(lat(v), lon(v), 'o', 'MarkerSize', 10, ...
        'MarkerFaceColor', marker_color, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Add labels with offset for better visibility
    if contains(team_abbrevs{v}, 'GSW')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'LAC')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'NYK')
        offset_lon = 0.5;
    elseif contains(team_abbrevs{v}, 'BRK')
        offset_lat = -1; offset_lon = 0.8;
    elseif contains(team_abbrevs{v}, 'WAS')
        offset_lat = -1.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'CHI')
        offset_lat = -1; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'PHI')
        offset_lat = 0.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'BOS')
        offset_lon = 0.5;
    else 
        offset_lat = 0.5; % default offset for other cities
        offset_lon = 0.3;
    end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{v});
    textm(lat(v) + offset_lat, lon(v) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
end

% Add colorbar & title
cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};
title('NBA Teams Win Probability Map', 'FontSize', 14, 'FontWeight', ...
    'bold');

%% Generate extra zoomed-in maps for California & New York
figure; 
% Create zoomed-in map for California
subplot(1, 2, 1)
lat_cal = [32 39];
lon_cal = [-125 -116];
axesm("MapProjection","mercator", "MapLatLimit", lat_cal, ...
    "MapLonLimit", lon_cal);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in California');
hold on 

cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);
% Plot points for California
for c = 1:length(team_abbrevs)
    if (lat_cal(1)<=lat(c)) && (lat(c)<=lat_cal(2)) && (lon_cal(1)<= ...
            lon(c)) && (lon(c)<=lon_cal(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(c) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(c), lon(c), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);

        % Add labels with offset for better visibility
        if contains(team_abbrevs{c}, 'GSW')
            offset_lat = -0.5; offset_lon = -0.8;
        elseif contains(team_abbrevs{c}, 'LAC')
            offset_lat = -0.5; offset_lon = -0.8;
        else
            offset_lat = 0.3; offset_lon = 0.3;
        end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{c});
    textm(lat(c) + offset_lat, lon(c) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end
end

% create zoomed-in map for Northeast region
subplot(1, 2, 2)
lat_NE = [40 42];
lon_NE = [-75 -73];
axesm("MapProjection","mercator", "MapLatLimit", lat_NE, ...
    "MapLonLimit", lon_NE);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in Northeast US');
hold on 

% Plot points for NE region 
for n = 1:length(team)
    if (lat_NE(1)<=lat(n)) && (lat(n)<=lat_NE(2)) && (lon_NE(1)<= ...
            lon(n)) && (lon(n)<=lon_NE(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(n) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(n), lon(n), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
           
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{n});
    textm(lat(n), lon(n), label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end 
end

cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};

%% ----- Model 2 (Metrics with less than 0.9 correlation) -----
% Visualize Model 2 Metrics vs Win%

selectedMetrics2 = {'ORtg', 'ThreeP_PCT', 'DRtg', 'BLKA', 'Age'};
targetVar = 'WinRate'; 

y = combinedTable.WIN_PCT;

for i = 1:length(selectedMetrics2)
    metric = selectedMetrics2{i};
    x = combinedTable.(metric);
    
    % Remove NaN values
    valid = ~isnan(x) & ~isnan(y);
    x_v = x(valid);
    y_v = y(valid);
    
    % Linear regression
    Xmat = [ones(size(x_v)), x_v];
    beta = Xmat \ y_v;
    y_pred = Xmat * beta;
    
    % Compute R² and correlation
    SS_res = sum((y_v - y_pred).^2);
    SS_tot = sum((y_v - mean(y_v)).^2);
    R2 = 1 - SS_res / SS_tot;
    r = corr(x_v, y_v);
    
    % Create new figure for each metric
    figure('Color','w');
    scatter(x_v, y_v, 'filled'); hold on;
    plot(x_v, y_pred, 'r', 'LineWidth', 2);
    xlabel(metric, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Win Percentage', 'FontWeight', 'bold', 'Color', 'k');
    title(sprintf('%s vs Win%% (R²=%.3f, r=%.3f)', metric, R2, r), ...
          'FontWeight', 'bold', 'Color', 'k');
    grid on;
    
    % Set all axes and tick labels to black
    ax = gca;
    ax.XColor = 'k';
    ax.YColor = 'k';
    ax.GridColor = [0.2 0.2 0.2];  % dark gray grid
    ax.FontSize = 11;
    ax.Color = 'w';  % background stays white
end

%% Correlation Heatmap for Model 2 Metrics
heatmapMetrics = [{'WIN_PCT'}, selectedMetrics2];  % include target + predictors

% Extract data for selected metrics
data = combinedTable(:, heatmapMetrics);
dataArray = table2array(data);

% Compute correlation matrix
corrMat = corr(dataArray, 'Rows', 'complete');  % handle NaNs

% Make upper triangle NaN (to show only lower half)
mask = triu(true(size(corrMat)), 1);
corrMat(mask) = NaN;

% Plot one-sided (lower-triangle) heatmap
figure('Color','w','Position',[600 300 600 500]);
h = heatmap(heatmapMetrics, heatmapMetrics, corrMat, ...
    'Colormap', parula, ...
    'ColorLimits', [-1 1], ...
    'CellLabelFormat', '%.2f');

% Customize visuals
h.Title = 'One-Sided Correlation Heatmap: Win% vs Selected NBA Metrics';
h.XLabel = 'Metrics';
h.YLabel = 'Metrics';
h.FontColor = 'k';     
h.FontSize = 12;

% Optional: Adjust font sizes safely
ax = findobj(gcf, 'Type', 'Axes');

%% Predict the most recent season (2024–2025)
latestSeason = max(combinedTable.Season);
trainIdx = combinedTable.Season < latestSeason;   % 2015–2023 for training
testIdx = combinedTable.Season == latestSeason;   % 2024 for testing

% Training data
X_train = combinedTable{trainIdx, selectedMetrics2};  % train features
y_train = combinedTable.W(trainIdx) ./ (combinedTable.W(trainIdx) + combinedTable.L(trainIdx));  % train target

% Testing data
X_test  = combinedTable{testIdx, selectedMetrics2};   % test features
y_test  = combinedTable.W(testIdx) ./ (combinedTable.W(testIdx) + combinedTable.L(testIdx));   % test target
teams_test = combinedTable.Team(testIdx);           % team names for testing

%% Train the regression model on past data
X_design_train = [ones(size(X_train, 1), 1), X_train];
beta = inv(X_design_train' * X_design_train) * X_design_train' * y_train;
disp(beta)

%% Predict Win% for the latest season
X_design_test = [ones(size(X_test, 1), 1), X_test];
predicted_WinPct_test = X_design_test * beta;

%% Compare predictions vs actuals
results_2024 = table(teams_test, y_test, predicted_WinPct_test, ...
    'VariableNames', {'Team', 'Actual_WinPct', 'Predicted_WinPct'});

disp('=== Predicted vs Actual Win% for 2024–2025 Season ===');
disp(results_2024);

%% Compute model accuracy for the latest season
R2_2024 = 1 - sum((y_test - predicted_WinPct_test).^2) / sum((y_test - mean(y_test)).^2);
fprintf('Model 2 R² (on 2024–2025 season): %.4f\n', R2_2024);

MAE = mean(abs(y_test - predicted_WinPct_test));
fprintf('Mean Absolute Error (on 2024–2025 season): %.4f\n', MAE);

%% Visualization – Actual vs Predicted Win%
figure;
bar(categorical(results_2024.Team), [results_2024.Actual_WinPct, results_2024.Predicted_WinPct]);
legend('Actual Win%', 'Predicted Win%', 'Location', 'northwest');
xlabel('Team');
ylabel('Win Percentage');
title(sprintf('Actual vs Predicted NBA Win%% (%d–%d Season)', latestSeason, latestSeason+1));
grid on;

%% Print predicted champion
[~, bestIdx] = max(results_2024.Predicted_WinPct);
fprintf('\n🏆 Predicted NBA Champion (%d–%d): %s\n', ...
    latestSeason, latestSeason+1, results_2024.Team{bestIdx});

%% Mapping predicted win probabilities using color-coded markers
% Create base map
% specify lat & lon limits
latlimit = [25 50];
lonlimit = [-125 -66];

figure;
axesm("MapProjection","mercator", "MapLatLimit",latlimit, ...
    "MapLonLimit",lonlimit) % Mercator projection
% show land area of USA 
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9])
hold on

% Load state borders of USA & Canada (for Toronto Raptors)
    load("usastates.mat");
    bordersLat = [usastates.Lat];
    bordersLon = [usastates.Lon];
    state_names = {usastates.Name};
    geoshow(bordersLat, bordersLon, 'Color', 'k')
    hold on

% load canadian province boundary for Toronto Raptors 
    province_file = "/Users/chowjiahui/Documents/MATLAB/PROVINCE.SHP";
    state_names = [state_names, 'Ontario'];
    geoshow(province_file, "FaceColor", [0.9 0.9 0.9]);

%% Load team data: locations & win probability
% Load team locations 
directory = '/Users/chowjiahui/Documents/MATLAB';
file_name = 'stadiums.csv';
inFile = fullfile(directory, file_name);

if exist(inFile, 'file') % test if file exists
    [team_abbrevs, team, ~, lat, lon] = textread(inFile, ...
        '%s %s %s %f %f', 'delimiter', ',', 'headerlines', 1);
else
    error('This file does not exist.')
end

% Convert team names to uppercase
team = upper(team);

% Rename "LOS ANGELES CLIPPERS" to "LA CLIPPERS"
team(strcmp(team, 'LOS ANGELES CLIPPERS')) = {'LA CLIPPERS'};

%% Plot with color-coded markers based on predicted win probability
% Create colormap: red (low) -> yellow (medium) -> green (high)
cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);

for v = 1:length(team_abbrevs)
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(v) * 99) + 1;
    marker_color = cmap(color_idx, :);
    
    % Plot points
    plotm(lat(v), lon(v), 'o', 'MarkerSize', 10, ...
        'MarkerFaceColor', marker_color, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Add labels with offset for better visibility
    if contains(team_abbrevs{v}, 'GSW')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'LAC')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'NYK')
        offset_lon = 0.5;
    elseif contains(team_abbrevs{v}, 'BRK')
        offset_lat = -1; offset_lon = 0.8;
    elseif contains(team_abbrevs{v}, 'WAS')
        offset_lat = -1.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'CHI')
        offset_lat = -1; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'PHI')
        offset_lat = 0.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'BOS')
        offset_lon = 0.5;
    else 
        offset_lat = 0.5; % default offset for other cities
        offset_lon = 0.3;
    end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{v});
    textm(lat(v) + offset_lat, lon(v) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
end

% Add colorbar & title
cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};
title('NBA Teams Win Probability Map', 'FontSize', 14, 'FontWeight', ...
    'bold');

%% Generate extra zoomed-in maps for California & New York
figure; 
% Create zoomed-in map for California
subplot(1, 2, 1)
lat_cal = [32 39];
lon_cal = [-125 -116];
axesm("MapProjection","mercator", "MapLatLimit", lat_cal, ...
    "MapLonLimit", lon_cal);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in California');
hold on 

cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);
% Plot points for California
for c = 1:length(team_abbrevs)
    if (lat_cal(1)<=lat(c)) && (lat(c)<=lat_cal(2)) && (lon_cal(1)<= ...
            lon(c)) && (lon(c)<=lon_cal(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(c) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(c), lon(c), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);

        % Add labels with offset for better visibility
        if contains(team_abbrevs{c}, 'GSW')
            offset_lat = -0.5; offset_lon = -0.8;
        elseif contains(team_abbrevs{c}, 'LAC')
            offset_lat = -0.5; offset_lon = -0.8;
        else
            offset_lat = 0.3; offset_lon = 0.3;
        end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{c});
    textm(lat(c) + offset_lat, lon(c) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end
end

% create zoomed-in map for Northeast region
subplot(1, 2, 2)
lat_NE = [40 42];
lon_NE = [-75 -73];
axesm("MapProjection","mercator", "MapLatLimit", lat_NE, ...
    "MapLonLimit", lon_NE);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in Northeast US');
hold on 

% Plot points for NE region 
for n = 1:length(team)
    if (lat_NE(1)<=lat(n)) && (lat(n)<=lat_NE(2)) && (lon_NE(1)<= ...
            lon(n)) && (lon(n)<=lon_NE(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(n) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(n), lon(n), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
           
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{n});
    textm(lat(n), lon(n), label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end 
end

cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};

%% ----- Model 3 (Metrics with more than 0.9 correlation and not in top 5) -----
% Visualize Model 3 Metrics vs Win%

selectedMetrics3 = {'NRtg', 'SRS', 'PW', 'PlusMinus', 'PL'};
targetVar = 'WinRate'; 

y = combinedTable.WIN_PCT;

for i = 1:length(selectedMetrics3)
    metric = selectedMetrics3{i};
    x = combinedTable.(metric);
    
    % Remove NaN values
    valid = ~isnan(x) & ~isnan(y);
    x_v = x(valid);
    y_v = y(valid);
    
    % Linear regression
    Xmat = [ones(size(x_v)), x_v];
    beta = Xmat \ y_v;
    y_pred = Xmat * beta;
    
    % Compute R² and correlation
    SS_res = sum((y_v - y_pred).^2);
    SS_tot = sum((y_v - mean(y_v)).^2);
    R2 = 1 - SS_res / SS_tot;
    r = corr(x_v, y_v);
    
    % Create new figure for each metric
    figure('Color','w');
    scatter(x_v, y_v, 'filled'); hold on;
    plot(x_v, y_pred, 'r', 'LineWidth', 2);
    xlabel(metric, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Win Percentage', 'FontWeight', 'bold', 'Color', 'k');
    title(sprintf('%s vs Win%% (R²=%.3f, r=%.3f)', metric, R2, r), ...
          'FontWeight', 'bold', 'Color', 'k');
    grid on;
    
    % Set all axes and tick labels to black
    ax = gca;
    ax.XColor = 'k';
    ax.YColor = 'k';
    ax.GridColor = [0.2 0.2 0.2];  % dark gray grid
    ax.FontSize = 11;
    ax.Color = 'w';  % background stays white
end

%% Correlation Heatmap for Model 3 Metrics
heatmapMetrics = [{'WIN_PCT'}, selectedMetrics3];  % include target + predictors

% Extract data for selected metrics
data = combinedTable(:, heatmapMetrics);
dataArray = table2array(data);

% Compute correlation matrix
corrMat = corr(dataArray, 'Rows', 'complete');  % handle NaNs

% Make upper triangle NaN (to show only lower half)
mask = triu(true(size(corrMat)), 1);
corrMat(mask) = NaN;

% Plot one-sided (lower-triangle) heatmap
figure('Color','w','Position',[600 300 600 500]);
h = heatmap(heatmapMetrics, heatmapMetrics, corrMat, ...
    'Colormap', parula, ...
    'ColorLimits', [-1 1], ...
    'CellLabelFormat', '%.2f');

% Customize visuals
h.Title = 'One-Sided Correlation Heatmap: Win% vs Selected NBA Metrics';
h.XLabel = 'Metrics';
h.YLabel = 'Metrics';
h.FontColor = 'k';     
h.FontSize = 12;

% Optional: Adjust font sizes safely
ax = findobj(gcf, 'Type', 'Axes');

%% Predict the most recent season (2024–2025)
latestSeason = max(combinedTable.Season);
trainIdx = combinedTable.Season < latestSeason;   % 2015–2023 for training
testIdx = combinedTable.Season == latestSeason;   % 2024 for testing

% Training data
X_train = combinedTable{trainIdx, selectedMetrics3};  % train features
y_train = combinedTable.W(trainIdx) ./ (combinedTable.W(trainIdx) + combinedTable.L(trainIdx));  % train target

% Testing data
X_test  = combinedTable{testIdx, selectedMetrics3};   % test features
y_test  = combinedTable.W(testIdx) ./ (combinedTable.W(testIdx) + combinedTable.L(testIdx));   % test target
teams_test = combinedTable.Team(testIdx);           % team names for testing

%% Train the regression model on past data
X_design_train = [ones(size(X_train, 1), 1), X_train];
beta = inv(X_design_train' * X_design_train) * X_design_train' * y_train;
disp(beta)

%% Predict Win% for the latest season
X_design_test = [ones(size(X_test, 1), 1), X_test];
predicted_WinPct_test = X_design_test * beta;

%% Compare predictions vs actuals
results_2024 = table(teams_test, y_test, predicted_WinPct_test, ...
    'VariableNames', {'Team', 'Actual_WinPct', 'Predicted_WinPct'});

disp('=== Predicted vs Actual Win% for 2024–2025 Season ===');
disp(results_2024);

%% Compute model accuracy for the latest season
R2_2024 = 1 - sum((y_test - predicted_WinPct_test).^2) / sum((y_test - mean(y_test)).^2);
fprintf('Model 3 R² (on 2024–2025 season): %.4f\n', R2_2024);

MAE = mean(abs(y_test - predicted_WinPct_test));
fprintf('Mean Absolute Error (on 2024–2025 season): %.4f\n', MAE);

%% Visualization – Actual vs Predicted Win%
figure;
bar(categorical(results_2024.Team), [results_2024.Actual_WinPct, results_2024.Predicted_WinPct]);
legend('Actual Win%', 'Predicted Win%', 'Location', 'northwest');
xlabel('Team');
ylabel('Win Percentage');
title(sprintf('Actual vs Predicted NBA Win%% (%d–%d Season)', latestSeason, latestSeason+1));
grid on;

%% Print predicted champion
[~, bestIdx] = max(results_2024.Predicted_WinPct);
fprintf('\n🏆 Predicted NBA Champion (%d–%d): %s\n', ...
    latestSeason, latestSeason+1, results_2024.Team{bestIdx});

%% Mapping predicted win probabilities using color-coded markers
% Create base map
% specify lat & lon limits
latlimit = [25 50];
lonlimit = [-125 -66];

figure;
axesm("MapProjection","mercator", "MapLatLimit",latlimit, ...
    "MapLonLimit",lonlimit) % Mercator projection
% show land area of USA 
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9])
hold on

% Load state borders of USA & Canada (for Toronto Raptors)
    load("usastates.mat");
    bordersLat = [usastates.Lat];
    bordersLon = [usastates.Lon];
    state_names = {usastates.Name};
    geoshow(bordersLat, bordersLon, 'Color', 'k')
    hold on

% load canadian province boundary for Toronto Raptors 
    province_file = "/Users/chowjiahui/Documents/MATLAB/PROVINCE.SHP";
    state_names = [state_names, 'Ontario'];
    geoshow(province_file, "FaceColor", [0.9 0.9 0.9]);

%% Load team data: locations & win probability
% Load team locations 
directory = '/Users/chowjiahui/Documents/MATLAB';
file_name = 'stadiums.csv';
inFile = fullfile(directory, file_name);

if exist(inFile, 'file') % test if file exists
    [team_abbrevs, team, ~, lat, lon] = textread(inFile, ...
        '%s %s %s %f %f', 'delimiter', ',', 'headerlines', 1);
else
    error('This file does not exist.')
end

% Convert team names to uppercase
team = upper(team);

% Rename "LOS ANGELES CLIPPERS" to "LA CLIPPERS"
team(strcmp(team, 'LOS ANGELES CLIPPERS')) = {'LA CLIPPERS'};

%% Plot with color-coded markers based on predicted win probability
% Create colormap: red (low) -> yellow (medium) -> green (high)
cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);

for v = 1:length(team_abbrevs)
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(v) * 99) + 1;
    marker_color = cmap(color_idx, :);
    
    % Plot points
    plotm(lat(v), lon(v), 'o', 'MarkerSize', 10, ...
        'MarkerFaceColor', marker_color, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Add labels with offset for better visibility
    if contains(team_abbrevs{v}, 'GSW')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'LAC')
        offset_lat = -1.5; offset_lon = -1.2;
    elseif contains(team_abbrevs{v}, 'NYK')
        offset_lon = 0.5;
    elseif contains(team_abbrevs{v}, 'BRK')
        offset_lat = -1; offset_lon = 0.8;
    elseif contains(team_abbrevs{v}, 'WAS')
        offset_lat = -1.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'CHI')
        offset_lat = -1; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'PHI')
        offset_lat = 0.5; offset_lon = -2;
    elseif contains(team_abbrevs{v}, 'BOS')
        offset_lon = 0.5;
    else 
        offset_lat = 0.5; % default offset for other cities
        offset_lon = 0.3;
    end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{v});
    textm(lat(v) + offset_lat, lon(v) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
end

% Add colorbar & title
cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};
title('NBA Teams Win Probability Map', 'FontSize', 14, 'FontWeight', ...
    'bold');

%% Generate extra zoomed-in maps for California & New York
figure; 
% Create zoomed-in map for California
subplot(1, 2, 1)
lat_cal = [32 39];
lon_cal = [-125 -116];
axesm("MapProjection","mercator", "MapLatLimit", lat_cal, ...
    "MapLonLimit", lon_cal);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in California');
hold on 

cmap = [linspace(0.8,0,100)', linspace(0,1,100)', zeros(100,1)];
colormap(cmap);
% Plot points for California
for c = 1:length(team_abbrevs)
    if (lat_cal(1)<=lat(c)) && (lat(c)<=lat_cal(2)) && (lon_cal(1)<= ...
            lon(c)) && (lon(c)<=lon_cal(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(c) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(c), lon(c), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);

        % Add labels with offset for better visibility
        if contains(team_abbrevs{c}, 'GSW')
            offset_lat = -0.5; offset_lon = -0.8;
        elseif contains(team_abbrevs{c}, 'LAC')
            offset_lat = -0.5; offset_lon = -0.8;
        else
            offset_lat = 0.3; offset_lon = 0.3;
        end
    
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{c});
    textm(lat(c) + offset_lat, lon(c) + offset_lon, label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end
end

% create zoomed-in map for Northeast region
subplot(1, 2, 2)
lat_NE = [40 42];
lon_NE = [-75 -73];
axesm("MapProjection","mercator", "MapLatLimit", lat_NE, ...
    "MapLonLimit", lon_NE);
geoshow("landareas.shp", "FaceColor", [0.9 0.9 0.9]);
geoshow(bordersLat, bordersLon, 'Color', 'k') % only need US borders
title('Win Probability of NBA Teams in Northeast US');
hold on 

% Plot points for NE region 
for n = 1:length(team)
    if (lat_NE(1)<=lat(n)) && (lat(n)<=lat_NE(2)) && (lon_NE(1)<= ...
            lon(n)) && (lon(n)<=lon_NE(2))
    % Determine color based on win probability
    color_idx = round(predicted_WinPct_test(n) * 99) + 1;
    marker_color = cmap(color_idx, :);
    % Plot points
    plotm(lat(n), lon(n), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ...
        marker_color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
           
    % Label with team name and win probability
    label_text = sprintf('%s\n', team_abbrevs{n});
    textm(lat(n), lon(n), label_text, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end 
end

cb = colorbar;
cb.Label.String = 'Win Probability';
cb.Ticks = [0 0.25 0.5 0.75 1];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};
