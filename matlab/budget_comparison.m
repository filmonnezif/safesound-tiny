%% WristEcho / SafeSound: recall vs false-alert budget (MobileNetV3 MN04 int8, DSP wake gate on)
% Plots the device-level results of the two home alert budgets: 1 and 3 false alerts per hour.
% Values: results/phase5c_home1.json and results/phase5c_home3.json, produced by the Python
% evaluation (python -m safesound.phase5c --policy home1 / home3), locked test set.
% github.com/filmonnezif/safesound-tiny

classes = ["Fire alarm" "Siren" "Doorbell" "Knock" "Baby cry" "Car horn"];
recall_1h = [92 77 77 78 67 60];     % event recall (%), home budget 1 alert/h
recall_3h = [91 90 80 80 71 68];     % event recall (%), home budget 3 alerts/h
fa_home   = [1.0 2.8];               % observed false alerts per hour, real home (SINS)
fa_tv     = [28 77];                 % false alerts per hour, varied/TV-like audio (AudioSet)

macro = [mean(recall_1h) mean(recall_3h)];
gain  = recall_3h - recall_1h;                         % what the extra budget buys per class
fprintf('Macro recall: %.1f%% (1/h)  ->  %.1f%% (3/h), +%.1f points\n', macro, diff(macro));
fprintf('Home false alerts: %.1f/h -> %.1f/h (x%.1f)\n', fa_home, fa_home(2) / fa_home(1));
fprintf('Recall gained per extra home false alert: %.1f points\n', diff(macro) / diff(fa_home));
disp(table(classes', recall_1h', recall_3h', gain', ...
     'VariableNames', {'Sound', 'Recall_1_per_h', 'Recall_3_per_h', 'Change_points'}));

figure('Name', 'Alert budget comparison', 'Color', 'w', 'Position', [60 40 980 600]);

subplot(1, 2, 1);
b = bar([recall_1h' recall_3h'], 'grouped');
b(1).FaceColor = [0.00 0.33 0.71]; b(2).FaceColor = [0.01 0.79 0.72];
set(gca, 'XTickLabel', classes, 'FontSize', 10); xtickangle(30);
ylim([0 100]); ylabel('Event recall (%)'); grid on;
legend(sprintf('1 alert/h  (macro %.1f%%)', macro(1)), ...
       sprintf('3 alerts/h (macro %.1f%%)', macro(2)), 'Location', 'southwest');
title('Recall per sound');
for k = 1:numel(classes)                               % print the change above each pair
    text(k, max(recall_1h(k), recall_3h(k)) + 3, sprintf('%+d', gain(k)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

subplot(1, 2, 2);
plot(fa_home, macro, 'o-', 'LineWidth', 2, 'MarkerSize', 9, 'Color', [0.00 0.33 0.71]); hold on;
plot(fa_home, [recall_1h(1) recall_3h(1)], 's--', 'LineWidth', 1.5, 'Color', [0.85 0.2 0.2]);
plot(fa_home, [recall_1h(2) recall_3h(2)], '^--', 'LineWidth', 1.5, 'Color', [0.95 0.55 0.1]);
hold off; grid on; xlim([0.5 3.3]); ylim([55 100]);
xlabel('Observed false alerts per hour (home)'); ylabel('Event recall (%)');
legend('Macro (6 sounds)', 'Fire alarm', 'Siren', 'Location', 'southeast');
title('Trade-off: recall vs false alerts');
text(fa_home(1), macro(1) - 4, '1.0/h', 'HorizontalAlignment', 'center');
text(fa_home(2), macro(2) - 4, '2.8/h', 'HorizontalAlignment', 'center');

sgtitle('MN04 int8 + DSP wake gate: home alert budget 1/h vs 3/h (locked test set)');
