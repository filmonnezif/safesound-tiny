"""Export trained students to int8 TensorFlow Lite (for TFLite Micro on the ESP32-S3) and check
the real int8 model against the PyTorch float model and our int8 simulation.

    python -m safesound.export --root /content/drive/MyDrive/safesound-tiny --data /content/cache

The DS-CNN is rebuilt in Keras with the same (batch-norm-folded) weights. PyTorch pads 1 on
every side, so convolutions use explicit ZeroPadding2D + 'valid' (TF 'same' pads asymmetric
for stride 2). Layout: PyTorch (B, 1, mels, frames) == Keras (B, mels, frames, 1).
Outputs (on Drive, results/export/): <name>_int8.tflite, <name>_int8.h (C array for ESP-IDF),
frontend.json (mel filterbank and normalisation constants), export.json (checks, sizes).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safesound.frontend import FMAX, FMIN, HOP, N_FFT, N_MELS, SR, _FB, stream_windows
from safesound.models import DSCNN
from safesound.quant import fold_bn, quantize

NAMES = ["p4_XL_logmel_kd", "p4_L_logmel_kd", "p4_M_logmel_kd"]


def keras_model(folded: DSCNN):
    import tensorflow as tf
    L = tf.keras.layers
    mean = folded.feat_mean.numpy().reshape(1, -1, 1, 1).astype(np.float32)
    inv = (1.0 / folded.feat_std.numpy()).reshape(1, -1, 1, 1).astype(np.float32)
    inp = L.Input((N_MELS, 100, 1), name="logmel")
    x = L.Lambda(lambda z: (z - mean) * inv, name="normalise")(inp)
    for i, mod in enumerate(folded.body):
        conv = mod.conv
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy()
        x = L.ZeroPadding2D(conv.padding[0])(x) if conv.padding[0] else x
        if conv.groups > 1:
            layer = L.DepthwiseConv2D(conv.kernel_size, strides=conv.stride, padding="valid")
            x = layer(x)
            layer.set_weights([w.transpose(2, 3, 0, 1), b])
        else:
            layer = L.Conv2D(conv.out_channels, conv.kernel_size, strides=conv.stride, padding="valid")
            x = layer(x)
            layer.set_weights([w.transpose(2, 3, 1, 0), b])
        if mod.relu:
            x = L.ReLU()(x)
    x = L.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="freq_mean")(x)            # (B, T, C)
    x = L.Lambda(lambda z: tf.reduce_max(z, axis=1) + tf.reduce_mean(z, axis=1), name="time_pool")(x)
    fc = L.Dense(folded.fc.out_features, name="logits")
    out = fc(x)
    fc.set_weights([folded.fc.weight.detach().numpy().T, folded.fc.bias.detach().numpy()])
    return tf.keras.Model(inp, out)


def to_tflite_int8(km, calib):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None, :, :, None].astype(np.float32)] for c in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(blob, feats):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = i["quantization"], o["quantization"]
    out = []
    for f in feats:
        it.set_tensor(i["index"], np.clip(np.round(f[None, :, :, None] / si + zi), -128, 127).astype(np.int8))
        it.invoke()
        out.append((it.get_tensor(o["index"]).astype(np.float32) - zo) * so)
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    return np.concatenate(out), ops


def c_array(blob, name):
    body = ",".join(f"0x{b:02x}" for b in blob)
    return (f"// {name}: int8 TFLite model, {len(blob)} bytes\n#pragma once\n#include <stdint.h>\n"
            f"alignas(16) const uint8_t {name}[] = {{{body}}};\nconst unsigned int {name}_len = {len(blob)};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/content/cache"))
    a = ap.parse_args()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from safesound.bench import Clips
    from safesound.phases import calib_feats, load_student
    from safesound.sim import CONFIGS
    out_dir = a.root / "results" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    choice = json.load(open(a.root / "results" / "phase2_choice.json"))["config"]

    # evaluation windows: real test event clips (first 150) streamed exactly like the benchmark
    tp = Clips(a.data, "test_pos")
    eval_feats = np.concatenate([stream_windows(tp.audio(i), "logmel") for i in range(150)])
    calib = calib_feats(a.data, CONFIGS[choice], "logmel", n_batches=4).numpy()[:1000]
    report = {}
    for name in NAMES:
        model = load_student(a.root / "checkpoints" / f"{name}.pt").cpu()
        folded = fold_bn(model)
        km = keras_model(folded)
        with torch.no_grad():
            ref = folded(torch.from_numpy(eval_feats)).numpy()
            sim = quantize(model.cpu(), torch.from_numpy(calib)).cpu()(torch.from_numpy(eval_feats)).numpy()
        kf = km.predict(eval_feats[..., None], verbose=0)
        blob = to_tflite_int8(km, calib)
        q, ops = run_tflite(blob, eval_feats)
        sig = lambda z: 1 / (1 + np.exp(-z))
        report[name] = {
            "tflite_bytes": len(blob), "ops": ops,
            "keras_vs_torch_max_abs_logit": float(np.abs(kf - ref).max()),
            "tflite_vs_float_prob_mae": float(np.abs(sig(q) - sig(ref)).mean()),
            "tflite_vs_sim_prob_mae": float(np.abs(sig(q) - sig(sim)).mean()),
            "tflite_vs_float_corr": float(np.corrcoef(q.ravel(), ref.ravel())[0, 1]),
            "decision_agreement_at_0.5": float(((sig(q) > 0.5) == (sig(ref) > 0.5)).mean()),
            "eval_windows": int(len(eval_feats)),
            "feat_mean": folded.feat_mean.numpy().ravel().tolist(),
            "feat_std": folded.feat_std.numpy().ravel().tolist(),
        }
        (out_dir / f"{name}_int8.tflite").write_bytes(blob)
        (out_dir / f"{name}_int8.h").write_text(c_array(blob, f"{name}_int8"))
        print(name, json.dumps({k: v for k, v in report[name].items() if not k.startswith("feat_")}), flush=True)
        if name == NAMES[0]:
            json.dump({"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
                       "window": "periodic hann", "compression": "log(mel_power_int16_scale + 1)",
                       "frames_per_window": 100, "mel_filterbank": _FB.tolist(),
                       "normalisation": "per model, see export.json (feat_mean, feat_std)"},
                      open(out_dir / "frontend.json", "w"))
    json.dump(report, open(out_dir / "export.json", "w"), indent=1)


if __name__ == "__main__":
    main()
