function safesound_demo(source)
% SAFESOUND_DEMO  Simple DSP demo: detect a smoke/fire alarm with classical DSP.
%
%   safesound_demo            built-in test sound: quiet room, a smoke alarm
%                             (ISO 8201 T3 beeps at 3.1 kHz), then a door slam
%   safesound_demo('x.wav')   any audio file
%   safesound_demo('mic')     record 10 s from the computer microphone
%
% What it shows (same detector as the paper, safesound/dsp.py):
%   1. STFT spectrogram (64 ms Hann window, 10 ms hop)
%   2. noise floor per frequency bin (minimum follower)
%   3. alarm evidence: a narrow tone in 2-5 kHz that stands out from its band,
%      from the noise floor, and from the low band (100-1500 Hz)
%   4. a 2 s window score every 0.5 s: steady frequency + beeps switching on/off
%   5. the usual "loudness" (energy) detector for comparison: it ignores the
%      quiet alarm but fires on the door slam
% Only base MATLAB is needed (no toolboxes).

if nargin < 1, source = 'synthetic'; end
fs = 16000;
[x, name] = get_audio(source, fs);

%% 1) STFT power spectrogram
NFFT = 1024; HOP = 160; FPS = fs / HOP;
w = 0.5 - 0.5 * cos(2 * pi * (0:NFFT-1)' / (NFFT - 1));     % Hann window
T = 1 + floor((numel(x) - NFFT) / HOP);
P = zeros(NFFT/2 + 1, T);
for t = 1:T
    seg = x((t-1)*HOP + (1:NFFT)) .* w;
    X = fft(seg);
    P(:, t) = abs(X(1:NFFT/2 + 1)).^2 + 1e-10;
end
f  = (0:NFFT/2)' * fs / NFFT;          % bin frequencies (Hz)
tf = ((0:T-1) * HOP + NFFT/2) / fs;    % frame centre times (s)

%% 2) noise floor: drops at once, rises 6 dB per 2 s
N = noise_floor(P, FPS);
PdB = 10 * log10(P);  NdB = 10 * log10(N);

%% 3) alarm evidence per frame
bA = find(f >= 2000 & f < 5000);       % alarm band
bL = find(f >= 100  & f < 1500);       % low band (speech, TV, appliances)
[peak, k] = max(PdB(bA, :), [], 1);                        % strongest bin
prom = peak - median(PdB(bA, :), 1);                       % stands out in its band
NdBA = NdB(bA, :);
snr  = peak - NdBA(sub2ind(size(NdBA), k, 1:T));           % above the noise floor
dom  = ramp(peak - max(PdB(bL, :), [], 1), -6, 6);         % louder than low band
a    = ramp(prom, 10, 25) .* ramp(snr, 6, 20) .* dom;      % 0..1 per frame
afreq = f(bA(k))';                                         % tone frequency

% energy detector: broadband level above its own floor
lvl  = sum(P, 1);
loud = ramp(10 * log10(lvl ./ noise_floor(lvl, FPS)), 10, 30);

%% 4) 2 s window scores every 0.5 s
W = round(2 * FPS);
nW = floor(numel(x) / (0.5 * fs));
tw = (1:nW) * 0.5;                     % window end times
sAlarm = zeros(1, nW); sLoud = zeros(1, nW);
for i = 1:nW
    e = min(T, floor((i * 0.5 * fs - NFFT) / HOP) + 1);
    if e <= 0, continue; end
    s = max(1, e - W + 1):e;
    on = a(s) > 0.3;
    edges = sum(diff(on) == 1);                            % beeps switching on
    if sum(on) > 3
        steady = ramp(-std(afreq(s(on)), 1), -60, -15);    % fixed frequency
    else
        steady = 0;
    end
    sAlarm(i) = pct(a(s), 90) * steady * (0.6 + 0.4 * min(edges, 3) / 3);
    sLoud(i)  = max(loud(s));
end

%% 5) decisions
TH = 0.4;                                          % alarm threshold
hit = sAlarm >= TH;
alert = hit & ([false hit(1:end-1)] | [false false hit(1:end-2)]);  % 2 of last 3
energyOn = sLoud >= 0.1;                           % energy detector fires

fprintf('\nSafeSound DSP demo: %s (%.1f s)\n', name, numel(x) / fs);
if any(alert)
    fprintf('  ALARM detected first at %.1f s (DSP alarm detector)\n', tw(find(alert, 1)));
else
    fprintf('  no alarm detected\n');
end
fprintf('  alarm detector on in %d windows, energy detector on in %d windows\n', ...
        sum(alert), sum(energyOn));

%% 6) plots
figure('Name', 'SafeSound DSP demo', 'Color', 'w');
tx = (0:numel(x)-1) / fs;

subplot(4, 1, 1);
plot(tx, x, 'Color', [0.3 0.3 0.3]); xlim([0 tx(end)]);
title(['Input: ' name], 'Interpreter', 'none'); ylabel('amplitude');

subplot(4, 1, 2);
imagesc(tf, f / 1000, PdB); axis xy; ylim([0 8]); colormap(gca, 'parula');
clim_ = [pct(PdB(:), 50), pct(PdB(:), 99.9)]; caxis(clim_);   % tones stand out
hold on; plot(tf([1 end]), [2 2], 'w--', tf([1 end]), [5 5], 'w--'); hold off;
ylabel('kHz'); title('Spectrogram (dB); dashed = alarm band 2-5 kHz');

subplot(4, 1, 3);
plot(tf, a, 'r', tf, loud, 'b'); ylim([-0.05 1.05]); xlim([0 tx(end)]);
legend('alarm tone evidence', 'loudness above floor', 'Location', 'northwest');
title('Per-frame DSP evidence');

subplot(4, 1, 4);
stairs(tw, sAlarm, 'r', 'LineWidth', 1.5); hold on;
stairs(tw, double(energyOn), 'b--');
plot([0 tw(end)], [TH TH], 'k:');
yl = 1.1;
for i = find(alert)
    patch([tw(i)-0.5 tw(i) tw(i) tw(i)-0.5], [0 0 yl yl], 'r', ...
          'FaceAlpha', 0.15, 'EdgeColor', 'none');
end
hold off; ylim([0 yl]); xlim([0 tx(end)]); xlabel('time (s)');
legend('alarm score (2 s window)', 'energy detector fires', 'threshold', ...
       'ALARM alert', 'Location', 'northwest');
title('Decision: shaded = alarm alert');

ws = warning('off', 'all');                       % MATLAB Online prints audio warnings
try
    sound(x / max(abs(x) + eps) * 0.8, fs);        % play what was analysed
catch
end
warning(ws);
end

%% ---------------------------------------------------------------------------
function [x, name] = get_audio(source, fs)
if strcmpi(source, 'synthetic')
    rng(1);
    t = (0:20*fs-1)' / fs;
    room = filter(1, [1 -0.97], randn(size(t)));          % fan / room rumble
    room = 0.02 * room / std(room);
    c = mod(t - 5, 4);                                     % T3: 3 x 0.5 s beeps, 1.5 s pause
    on = (t >= 5 & t < 13) & (c < 2.5) & (mod(c, 1) < 0.5);
    x = room + 0.01 * on .* sin(2 * pi * 3100 * t);        % quiet smoke alarm, 5-13 s
    i = 16 * fs;                                           % door slam at 16 s
    x(i+1:i+800) = x(i+1:i+800) + 0.5 * randn(800, 1) .* exp(-(0:799)' / 150);
    name = 'synthetic: room, smoke alarm 5-13 s, door slam 16 s';
elseif strcmpi(source, 'mic')
    r = audiorecorder(fs, 16, 1);
    disp('Recording 10 s ...'); recordblocking(r, 10); disp('done');
    x = getaudiodata(r); name = 'microphone';
else
    [x, fs0] = audioread(source);
    x = mean(x, 2);
    if fs0 ~= fs                                           % simple resampling
        x = interp1((0:numel(x)-1)' / fs0, x, (0:1/fs:(numel(x)-1)/fs0)', 'linear');
    end
    [~, name] = fileparts(source);
end
x = x(:);
end

function N = noise_floor(P, FPS)
% minimum follower per row: drops to a quieter value at once, rises 6 dB per 2 s
up = 10^(6 / 10 / (2 * FPS));
srt = sort(P, 2);
n = srt(:, max(1, round(0.1 * size(P, 2))));       % start at the 10th percentile
N = zeros(size(P));
for t = 1:size(P, 2)
    n = min(n * up, P(:, t));
    N(:, t) = n;
end
end

function y = ramp(x, lo, hi)
y = min(max((x - lo) / (hi - lo), 0), 1);
end

function v = pct(x, p)
% percentile with linear interpolation (same as numpy's default)
x = sort(x(:));
r = 1 + (numel(x) - 1) * p / 100;
lo = floor(r); hi = ceil(r);
v = x(lo) + (r - lo) * (x(hi) - x(lo));
end
