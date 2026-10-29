> **History:** the generator and its ONNX/mobile scripts were removed on 2026-10-28; the files are in `archive/motion_generate_2026-10-28/` and the commands below only run if they are copied back. The measurements stay valid for the data they used.

# What blocks running this on a phone (audit, 2026-10-22)

Target in `notes/Text -> Rigged, Animated Creature Pipeline Plan.md`: Galaxy S23 Ultra (Snapdragon), offline, about 100-150 MB on disk, under 2 s per creature. Everything below was measured on this Mac (M-series CPU / GPU), not on a phone; phone numbers are not measured. `scripts/generate/export_onnx.py` reproduces the model export; the experiments were scratch scripts, described here.

## Summary

The neural core is not the problem: it exports to ONNX, matches torch to 1e-6, quantizes to about 100 MB and runs in tens of milliseconds. What blocks a phone is (1) the CLIP text tower, which is the largest model and breaks under the obvious quantization, (2) about 2,300 lines of numpy / scipy Python that have no phone equivalent yet (IK proxy, smoothing, the retargeter, the walk/run constraint), (3) the character library (4 GB), and (4) the on-device player. None of (2)-(4) has been started.

## 1. Models (measured)

| model | parameters | fp32 | fp16 | int8 | ONNX export vs torch (eval mode) | CPU time on this Mac, 4 threads |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 text tower | 63.4 M (token table 25.3 M = 101 MB of the 254 MB) | 254 MB | 127 MB | 63 MB (weight-only) | max error 4.5e-6 | 13.6 ms per prompt |
| diffusion denoiser (MLP) | 28.5 M | 114 MB | 57 MB | 28.6 MB | 5e-6 | 1.4 ms per call (fp32), 0.5 ms (int8) |
| autoencoder decoder (+ aux head) | 3.5 M (the encoder, 2.4 M, is not needed) | 14.1 MB | 7 MB | 6.3 MB | 1.6e-6 | 2.7 ms (fp32), 2.0 ms (int8) |
| total | 95.4 M | 382 MB | 191 MB | about 98 MB | | |

Full sampling (50 DDIM steps with classifier-free guidance = 100 denoiser calls) + latent-norm cap + decode, 24 backflip seeds, same noise for every row (gate metrics: smooth / completes the turn / ends upright / all three):

| configuration | smooth | completes | upright | all three | time per sample |
|---|---|---|---|---|---|
| torch fp32 on the GPU (MPS) | 42% | 62% | 33% | 12% | 158 ms |
| ONNX fp32 (denoiser + decoder) | 42% | 62% | 33% | 12% | 60 ms |
| ONNX int8 denoiser, fp32 decoder | 46% | 62% | 33% | 12% | 21 ms |
| ONNX int8 denoiser + decoder | 38% | 62% | 33% | 8% | 20 ms |

The ONNX fp32 model reproduces torch exactly (latent difference 0.000). int8 changes the latent by 4.3% and the decoded features by 0.016 on average; the gate rates stay within the noise of 24 seeds (the last row's 8% is two seeds). Phones are slower than this Mac's CPU, perhaps several times, which still leaves the sampler at well under a second; not measured.

**CLIP text tower: the one that breaks.** ONNX dynamic int8 (activations quantized too) takes the centred embedding to cosine **0.36** against fp32 (the pipeline subtracts the mean caption embedding, which magnifies errors), the nearest training caption agrees for 2% of 300 held-out captions: unusable, per-tensor or per-channel. Keeping activations in higher precision works (weights stored low, emulated with fp32 compute):

| storage | size | centred cosine to fp32 (mean / 5th percentile, 200 captions) | generated motion (backflip, front flip, walk, run; 24 seeds each) |
|---|---|---|---|
| fp32 | 254 MB | 1 | baseline |
| all weights fp16 | 127 MB | 1.000 / 1.000 | identical to the baseline in every number |
| token table int8 per row, transformer fp16 | about 63 MB | 0.999 / 0.997 | not run |
| token table int8 per row + transformer int8 per channel, no calibration | about 63 MB | 0.996 / 0.991 | within noise (flip rates move by 4-8 points, walk and run speed by 1-2%) |

(The nearest-caption agreement is 94% for fp16 and 76-78% for the int8 variants; many training captions are near duplicates, so top-1 is a noisy measure. These were emulated by rounding the weights in the ONNX file, not exported in a phone format: a real deployment needs W8A16 kernels or dequantize-at-load.) Not tried: a small student text encoder distilled from CLIP (the motion model only sees about 3,000 distinct captions plus the front end's rewrites; the shape matching needs open vocabulary, so it would stay on CLIP), MobileCLIP2-S0 (same 63 M, research-only licence), static (calibrated) int8.

Budget: int8 stack about 63 + 28.6 + 6.3 = **98 MB**, plus data tables (below) about 10 MB, plus a few characters: 110-130 MB, inside the plan's 100-150 MB but with little margin; with fp16 CLIP it is about 190 MB.

NPU: the dynamic int8 models use operators that the Qualcomm QNN execution provider does not run; the NPU wants a calibrated QDQ (static) model. Not needed for speed (CPU is fast enough), and not tried.

## 2. Runtime code that is Python only (measured sizes and times)

| part | lines | what it needs | time on this Mac |
|---|---|---|---|
| IK proxy `proxy_from_features` + body-frame smoothing (`decoded_to_character.py`) | 340 | numpy, `scipy.spatial.transform.Rotation`, `scipy.ndimage.gaussian_filter1d` / `median_filter` | 65 ms per candidate (x16 in the gate) |
| rule-based retargeter (`retarget.py`) + quaternion helpers (`convert_clips.py`) | 1,065 + 560 (partly) | numpy; role labelling by joint names; reads the target's `model.glb`, writes a clip GLB | about 130 ms retarget + 120 ms writing the proxy GLB (file round trips that a phone would skip) |
| walk / run speed constraint (`speed_sampler.py`, `gait_check.py`, `walk_check.py`) | 375 + 106 + 130 | the real clips' features (a 127 MB array) for per-caption speed / duration priors; the decoder is called at every one of the 101 sampling steps | decoder x101: 0.27 s on the GPU, about 0.2 s as int8 ONNX on the CPU |
| text front end (`text_frontend.py`) | 192 | regex and a 13 KB word list; the CLIP BPE tokenizer (vocab + merges, about 1.6 MB) is a separate port | 0 ms |
| sampler loop, latent-norm cap, candidate gate (`generate_motion.py`, `diffusion_test.sample`) | 223 + | numpy; the cap needs the 5,500 training captions' embeddings (17 MB fp32, 5.6 MB fp16) and their latent norms | gate: 1.19 s for 4 candidates, 2.55 s for 10 on the desktop (mostly Python IK) |

Together about 2,300 lines to port (Kotlin / C++), or to run through an embedded Python (for example Chaquopy: not tested here, and whether its numpy / scipy builds suffice is not checked). The data tables the Python code reads at run time can be precomputed: the 127 MB feature array reduces to per-caption speed / duration statistics (kilobytes), the latent-norm table to about 6 MB, the shape index (3,570 characters x 512, fp16) is 3.7 MB.

## 3. Characters

2,792 retarget-target characters, 4.0 GB of `model.glb` (median 0.4 MB, 90th percentile 2.9 MB, largest 277 MB). Not shippable as a set: a curated subset (a few dozen, a few MB each) or an on-demand download, plus precomputed per-character retarget data (role labels, rest pose, scale) instead of parsing a GLB on the phone. Licences of the Mixamo and Objaverse assets for redistribution inside an app have not been checked. Shape retrieval returns existing characters only, so the shape part is limited to whatever is shipped.

## 4. Not started

- The on-device player: skinned-mesh animation from the generated per-frame rotations (the plan's "assembly"), in an engine or a custom renderer; GLB export would be optional.
- Android integration, memory and battery measurements, thermal behaviour, first-launch time (here the models load in about 20 s from the external drive).
- Retargeter robustness on phone-side data (the 47 quadruped targets, rigs without labelled roles).

## 5. Not phone-specific, but limits the product

Flips complete and land upright in only about 8-9% of single samples (a gate with several tries hides this: 16 tries took about 2.5 s on the desktop); the run clips are short (0.6 s); the language is limited to the 6,055 clips' vocabulary; no new bodies are generated.

## 6. Memory (measured with ONNX Runtime on this Mac, fresh process; interpreter + `import onnxruntime` baseline of 49 MB subtracted)

| what is loaded | model files | RAM added by loading | after one full generation | peak |
|---|---|---|---|---|
| int8 denoiser + int8 decoder, default ORT | 35 MB | 89 MB | 95 MB | 102 MB |
| same, memory arena off (`enable_cpu_mem_arena=False`; also `session.disable_prepacking`) | 35 MB | 51 MB | 59 MB | 64 MB |
| fp32 denoiser + fp32 decoder | 128 MB | 169 MB | 177 MB | 188 MB |
| CLIP text tower, fp32 | 254 MB | **494 MB** (1.9x the file) | 502 MB | 529 MB |
| CLIP, optimisation off | 254 MB | 729 MB | 736 MB | 774 MB |
| CLIP + int8 pair together (default options) | 289 MB | 548 MB | 551 MB | 579 MB |

Findings: (1) ONNX Runtime holds about twice the file size for CLIP (the weights plus the runtime's packed or constant-folded copies) and the arena / pre-packing switches that save 40% on the int8 models do nothing for CLIP; (2) deleting the CLIP session did not give the memory back to the operating system in this test (RSS stayed), so "load CLIP, embed, free it" is not a way to lower the peak here (a separate process that exits would; not tried); (3) activations are negligible (the whole generation adds 6-8 MB: the denoiser works on 2 x 256 numbers, the decoder on 128 tokens x 256 with 4 heads). A naive port (CLIP fp32 + int8 pair) therefore needs about **0.6 GB** for the models. With a weight-only int8 or fp16 CLIP file (63 / 127 MB) the figure is probably 2x that, 130-250 MB, but that was not measured (ORT CPU cannot run an fp16 file directly, and the weight-only int8 CLIP was only emulated), so the realistic range for the models is about 0.2-0.35 GB, not yet measured.

Everything else in RAM: data tables about 11 MB (5,500 caption embeddings fp16 5.6 MB, shape index 3.7 MB, BPE vocab 1.6 MB, lexicon 13 KB); animation buffers well under 1 MB (at most 128 frames x about 65 joints x 7 numbers) plus a few MB of numpy temporaries in the Python version; **the character**: vertex + index buffers are small (random sample of 160 targets: median 0.3 MB, 90th percentile 0.8 MB, largest 2.5 MB) but decoded textures are not: median about 0, 90th percentile 22 MB, largest in the sample 134 MB, 10 of the 160 have a 2048 px or larger texture (RGBA with mipmaps), and the three biggest files in the library decode to 268, 403 and 805 MB (36 textures of 2048 px in the worst). The shipped characters need textures capped and compressed (GPU formats such as ASTC / ETC2, 4-8x smaller than RGBA) or stripped. Renderer / engine overhead: not measured. For reference the desktop server (torch on the GPU, all models and tables) is 722 MB resident.

## 7. The "mobile" mode of the generator page

`viewer/generate.html` has a "Runtime" block: tick "mobile runtime" to run the CLIP text encoder, the 50-step sampler and the decoder through ONNX Runtime on the CPU with the int8 models (`scripts/generate/mobile_runtime.py`, `generate_motion.py --mobile --threads N`, `POST /api/generate` fields `mobile`, `threads`; the response has `timing_ms`). The page shows the time of each step and, for mobile, the neural time multiplied by a "phone slowdown" factor you set (default 4; an assumption, not a measurement). Backflip, seed 3, 4 candidates, Aciana, this Mac: desktop torch/GPU 1,231 ms (sampler 672); mobile with 4 threads 356-372 ms (text encoder 19-22, sampler 82-85, decoder 11-13, Python IK + gate 152-167, retarget + write 74-90), with 1 thread 497 ms (neural 238 of it); 16 tries (10 candidates used) 840 ms; creating the sessions the first time takes about 1 s (not counted). Walk and run fall back to the desktop sampler (the speed constraint is not ported) and the page says so. **Why generation seemed slow with the box ticked (fixed 2026-10-22):** the model work was not the cause (first mobile call 1.3 s end to end, later calls 0.3 s). Every result reloaded the embedded player, which asked `/api/retargets` for the list of every retargeted clip: 13-17 s per call (it scans the 3,570 character folders on the external drive, and the generated clips from testing had grown the list to 401). Now the embedded player asks `/api/retarget?target=&name=` for its single clip (0.08 s), the full list is cached in memory (cleared when a clip is generated through the page; `?refresh=1` forces a rebuild), and ticking the mobile box creates the ONNX sessions in the background (`POST /api/warm`). Click to player showing the clip in a real browser: 1.3-2.5 s; first generate after ticking mobile 0.67 s. The standalone `compare.html` still builds the full list on its first load after each generation (about 14-17 s).

`data/mobile/` holds the three ONNX files (289 MB); `onnxruntime` was added to `environment.yml`.

## Suggested order

1. Port the text path (BPE tokenizer, front-end rules) and run denoiser + decoder + CLIP (weight-only int8 or fp16) with ONNX Runtime on a phone: measures the real latency, the only unknown in section 1.
2. Port the IK proxy and retargeter with a test that compares against the Python output on the clips in `data/decoded` (the Python is the reference).
3. Replace the walk / run constraint's data with precomputed tables.
4. Choose and package a character subset; write the player.
