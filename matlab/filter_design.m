%% WristEcho / SafeSound: the filters used in the system (base MATLAB only)
% 1) Hann analysis window (STFT of the DSP detectors): spectrum, main lobe and side lobes
% 2) 30 Hz high-pass biquad (device pre-filter): magnitude, phase, poles and zeros
% 3) 40-band mel filterbank (front end of the neural network)
% 4) adaptive noise floor (minimum follower): fast attack, slow release (+6 dB per 2 s)
% Same parameters as safesound/dsp.py, safesound/frontend.py and firmware/detector.
% github.com/filmonnezif/safesound-tiny

fs = 16000;

%% 1) Hann window, N = 1024 (64 ms)
N = 1024;
w = 0.5 - 0.5 * cos(2 * pi * (0:N-1)' / (N - 1));
Nz = 64 * N;                                         % zero padding for a smooth spectrum
W = abs(fft(w, Nz)); W = 20 * log10(W / max(W) + 1e-12);
fb = (0:Nz-1)' / Nz * N;                             % frequency in FFT bins
half = fb <= 8;
sl = max(W(fb > 2 & fb <= 8));                       % highest side lobe
fprintf('Hann N=%d: bin spacing %.2f Hz, main lobe +-2 bins (%.0f Hz wide), highest side lobe %.1f dB\n', ...
        N, fs / N, 4 * fs / N, sl);

%% 2) 2nd-order Butterworth high-pass, fc = 30 Hz (RBJ biquad, as in the firmware)
fc = 30; Q = 1 / sqrt(2);
w0 = 2 * pi * fc / fs; al = sin(w0) / (2 * Q); c = cos(w0); a0 = 1 + al;
b = [(1 + c) / 2, -(1 + c), (1 + c) / 2] / a0;
a = [1, -2 * c / a0, (1 - al) / a0];
f = logspace(0, log10(8000), 2000);                  % 1 Hz .. 8 kHz
z = exp(1j * 2 * pi * f / fs);
H = polyval(b, z) ./ polyval(a, z);                  % frequency response on the unit circle
fprintf('HPF b = [%.6f %.6f %.6f], a = [1 %.6f %.6f]\n', b, a(2:3));
fprintf('HPF gain: %.1f dB at 10 Hz, %.1f dB at 30 Hz, %.2f dB at 100 Hz, %.3f dB at 1 kHz\n', ...
        interp1(f, 20 * log10(abs(H)), [10 30 100 1000]));
p = roots(a); q = roots(b);
fprintf('poles |p| = %.5f (stable: inside the unit circle), zeros at z = %.0f (double)\n', abs(p(1)), real(q(1)));

%% 3) mel filterbank: 40 triangles, 50-8000 Hz, area-normalised (Slaney)
nmel = 40; nfft = 512; nb = nfft / 2 + 1;
hz2mel = @(x) 2595 * log10(1 + x / 700); mel2hz = @(m) 700 * (10 .^ (m / 2595) - 1);
edges = mel2hz(linspace(hz2mel(50), hz2mel(8000), nmel + 2));
fk = linspace(0, fs / 2, nb);
FB = zeros(nmel, nb);
for m = 1:nmel
    lo = edges(m); ce = edges(m + 1); hi = edges(m + 2);
    FB(m, :) = max(0, min((fk - lo) / (ce - lo), (hi - fk) / (hi - ce))) * 2 / (hi - lo);
end
fprintf('mel bank: 40 bands, centres %.0f Hz .. %.0f Hz, first band %.0f Hz wide, last %.0f Hz wide\n', ...
        edges(2), edges(end - 1), edges(3) - edges(1), edges(end) - edges(end - 2));

%% 4) adaptive noise floor on one STFT bin: quiet room, louder room, tone bursts
FPS = 100;                                           % 10 ms hop
t = (0:20 * FPS - 1) / FPS;
rng(2);
Pdb = -60 + 3 * randn(size(t));                      % background, -60 dB
Pdb(t >= 8) = Pdb(t >= 8) + 12;                      % room gets 12 dB louder at 8 s
burst = (t >= 3 & t < 5) | (t >= 13 & t < 15);       % a tone in this bin
Pdb(burst) = -25 + randn(1, nnz(burst));
up = 6 / (2 * FPS);                                  % floor may rise 6 dB per 2 s
Ndb = zeros(size(t)); n = prctile10(Pdb);
for k = 1:numel(t)
    n = min(n + up, Pdb(k));                         % drops at once, rises slowly
    Ndb(k) = n;
end

%% figure
figure('Name', 'Filter design', 'Color', 'w', 'Position', [60 40 1000 640]);
subplot(2, 2, 1);
plot(fb(half), W(half), 'LineWidth', 1.2); grid on; ylim([-100 5]);
xlabel('frequency offset (FFT bins, 15.6 Hz each)'); ylabel('dB');
title(sprintf('Hann window N=1024: side lobes %.0f dB', sl));

subplot(2, 2, 2);
yyaxis left;  semilogx(f, 20 * log10(abs(H)), 'LineWidth', 1.4); ylabel('magnitude (dB)'); ylim([-60 5]);
yyaxis right; semilogx(f, angle(H) * 180 / pi, '--'); ylabel('phase (deg)');
grid on; xlabel('Hz'); title('30 Hz Butterworth high-pass (biquad)');
axes('Position', [0.80 0.66 0.10 0.13]);             % pole-zero inset
th = linspace(0, 2 * pi, 200); plot(cos(th), sin(th), 'k:'); hold on;
plot(real(p), imag(p), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(real(q(1)), imag(q(1)), 'bo', 'MarkerSize', 7); hold off; axis equal; axis([-1.2 1.2 -1.2 1.2]);
title('poles x / zeros o', 'FontSize', 8); set(gca, 'FontSize', 7);

subplot(2, 2, 3);
plot(fk, FB', 'LineWidth', 0.8); grid on; xlim([0 8000]);
xlabel('Hz'); ylabel('weight'); title('40-band mel filterbank (50-8000 Hz)');

subplot(2, 2, 4);
plot(t, Pdb, 'Color', [0.6 0.6 0.6]); hold on;
plot(t, Ndb, 'r', 'LineWidth', 2);
plot(t, Ndb + 10, 'r:', 'LineWidth', 1); hold off; grid on; ylim([-75 -15]);
xlabel('time (s)'); ylabel('dB');
legend('bin power', 'noise floor', 'floor + 10 dB', 'Location', 'northwest');
title('Adaptive noise floor (fast down, +6 dB / 2 s up)');

function v = prctile10(x)                           % 10th percentile without toolboxes
    s = sort(x(:)); v = s(max(1, round(0.1 * numel(s))));
end
