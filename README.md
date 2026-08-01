# EdgeQuake

**An earthquake early-warning system prototype for Taiwan — blind-tested on
real historical events, running live on community sensor data, and audited
automatically after every significant earthquake.**

> **What this is, in three precise statements.**
> The **system** is an EEW prototype: the full chain (waveforms → AI phase
> picking → association → location → magnitude → county alert decision →
> notification) exists and runs, validated by blind tests on events the
> models never saw. The **website** is an earthquake console: live CWA/USGS
> bulletins, a real-time detection layer on the ExpTech community network,
> and auditable replays of historical events. It is **not** a warning
> *service*: that requires real-time access to the national seismic network,
> 24/7 operations, and legal authority — all of which belong to the CWA.
> Everything here is labeled accordingly.

[中文版說明在下方](#中文說明) | English first, Chinese below.

---

## Highlights

- **Streaming replay engine**: ring buffer + sliding-window inference over
  historical waveforms at true real-time pacing; picks with confidence and
  per-step latency (p50 ≈ 4–9 ms on laptop CPU — far under the 500 ms hop).
- **Cross-domain benchmark** (protocol after Münchmeyer et al. 2022): the same
  PhaseNet architecture, two public pretrained weights, three test domains.
  Off-the-shelf PhaseNet loses **0.24 P-wave F1** when moved from its home
  region to Taiwan — the quantified case for local fine-tuning.
- **Taiwan fine-tune** (CWA 2019, 70k traces, Kaggle T4, frozen BatchNorm +
  amplitude clipping): **P F1 0.660 → 0.702, S F1 0.557 → 0.680**, with the
  biggest gain exactly where domain shift hurt most (S recall 0.52 → 0.69).
- **Live engine with pluggable sources**: the same streaming pipeline runs on
  historical replays (honest rehearsals — the engine never sees the future),
  on the **ExpTech TREM community MEMS network in true real time** (123
  stations, trigger-mode detection), and is SeedLink-ready for raw waveform
  feeds. Real notifications (email / Telegram / webhook) fire on a
  quality-gated alert decision.
- **AI early magnitude (MagNet-style, distance-conditioned)**: final-magnitude
  regression from the first 3 s of P, with a Gaussian uncertainty head.
  Honest blind-test verdict: fast but saturating for M6.5+ — quantified, not
  hidden (see Phase 5).
- **Automated shadow audit**: a GitHub Actions workflow downloads CWA's
  official strong-motion waveforms (~12 min after every significant quake),
  replays the engine blind, and commits the "had it been running" timeline
  back to this repo — a public, self-accumulating verification log.
- **Honest engineering log**: eleven real failures (upstream Windows URL bug,
  BatchNorm-stats poisoning by dead channels, label-channel misalignment,
  deep-solution mispick absorption, coda re-triggering, ...) with root causes
  and fixes — see [Pitfalls](#pitfalls-what-actually-broke).

![Full-chain replay of the 2024-04-03 Hualien M7.2 earthquake — AI picks drive location, magnitude and city countdowns, with the official CWA timeline overlaid](outputs/convergence_0403.gif)

*The 2024-04-03 Hualien M7.2 earthquake replayed end-to-end from raw GDMS
waveforms: our fine-tuned PhaseNet picks 110/112 stations, location converges
to ~18 km of the epicenter within 4 s of origin, and the PGA-based magnitude
reads M6.0 at the moment CWA's 9-second first report said M6.2 — an
independent open-source chain reproducing the operational system's early
underestimation of M7+ events (saturation is physics, not implementation).*

## Results

Phase-picking F1 on time-split test sets, ±0.5 s tolerance, threshold 0.3
unless noted:

| Weights (training domain) | Test domain | P F1 | S F1 | Note |
|---|---|---|---|---|
| original (NCEDC, N. California) | NCEDC in-domain (paper) | 0.896 | 0.801 | reference |
| original | Iquique (Chile) | 0.873 | 0.771 | mild cross-domain drop |
| original | **CWA (Taiwan)** | **0.660** | **0.557** | large drop; recall 0.59 |
| stead (STEAD, global) | Iquique (Chile) | 0.390 | 0.281 | severe miscalibration |
| stead | **CWA (Taiwan)** | **0.228** | **0.220** | best-threshold P only 0.25 |
| **cwa-ft (this repo)** | **CWA (Taiwan)** | **0.702** | **0.680** @thr 0.5 | +0.04 P / +0.12 S |

![Fine-tuned PhaseNet on CWA test years — residuals, threshold sweep, precision-recall](outputs/eval_cwa_2020_2021_seisbench-phasenet-ft-phasenet_cwa_ft.png)

Three takeaways: (1) Taiwan's domain gap is far larger than Chile's — mixed
instrumentation (CWASN short-period/broadband + TSMIP strong-motion) and local
geology matter; (2) "trained on global data" does not imply cross-domain
robustness — STEAD weights lose to single-network weights everywhere we
tested; (3) a fixed confidence threshold is not transferable across weights
(STEAD's optimal S threshold is 0.80) — calibration must be a first-class
metric.

## Quick start

```bash
pip install -r requirements.txt

# streaming replay demo (bundled real earthquake)
python scripts/demo_replay.py

# evaluation pipeline sanity check (synthetic dataset, seconds)
python scripts/make_synthetic_dataset.py
python scripts/eval_picking.py --dataset data/synthetic_test --limit 50

# small real-data benchmark (Iquique, ~5 GB download on first use)
python scripts/eval_picking.py --dataset iquique --limit 500

# Taiwan (CWA) — size-preview guard prevents accidental 100 GB downloads:
python scripts/eval_picking.py --preview-cwa _2019 _2020 _2021
python scripts/fetch_cwa_hf.py          # fast Hugging Face route (~27 GB / 3 yrs)
python scripts/eval_picking.py --dataset cwa --chunks _2020 _2021 --confirm-download --limit 1000

# evaluate the fine-tuned weights
python scripts/eval_picking.py --dataset cwa --chunks _2020 _2021 --confirm-download \
    --limit 1000 --state-dict outputs/phasenet_cwa_ft.pt
```

Fine-tuning runs on Kaggle (free GPU): build the compact training subset
locally with `scripts/make_cwa_train_subset.py` (~3.5 GB from the cached 2019
chunk), upload it as a Kaggle Dataset, then run
`kaggle/edgequake_cwa_finetune.ipynb`.

Phase 2 (location + magnitude):

```bash
python scripts/build_event_catalog.py                       # 1,317 multi-station events
python scripts/demo_convergence.py --event 20121013190      # 2020 Yilan M6.6 deep event
python scripts/make_convergence_gif.py --event 20021510590  # animated replay GIF
```

![Streaming replay demo — waveform with picks, rolling P/S probabilities, per-step inference latency](outputs/replay_demo.png)

## Repository layout

```
src/edgequake/
├── pickers/        # Picker interface; SeisBench PhaseNet (+ finetuned ckpt), TF fallback
├── replay/         # ring buffer + replay engine (streaming inference, latency, triggers)
├── eval/           # benchmark protocol: residuals, PR/threshold sweeps, false alarms
└── data/           # dataset loaders with download guards (CWA/CEED size preview)
scripts/            # demo, evaluation CLI, dataset builders, CWA fetch/fix utilities
kaggle/             # fine-tuning notebook (BN-freeze + amplitude-clip recipe)
outputs/            # evaluation figures/JSON + finetuned weights (1.1 MB)
```

## Pitfalls (what actually broke)

| Problem | Root cause | Fix in this repo |
|---|---|---|
| All SeisBench dataset downloads 404 on Windows | upstream uses `os.path.join` for URLs → backslash | monkeypatch in `data/loader.py` |
| CWA/CEED never fall back to Hugging Face | upstream `compile_from_source` defaults to False | enabled in loader |
| CWA via SeisBench repo = 43 GB/yr uncompressed at ~1.3 MB/s | repo stores raw HDF5 | `fetch_cwa_hf.py`: compressed HF route, resumable |
| pandas crash on CWA metadata | mixed timestamp formats (some rows lack fractional seconds) | `fix_cwa_metadata.py` |
| Our own eval mislabeled TSMIP windows | arrival samples stored at original 200 Hz, waveforms resampled to 100 Hz | rescale arrivals by `trace_sampling_rate_hz` (P F1 0.629 → 0.660) |
| First fine-tune silently destroyed the model | labeller outputs [P,S,Noise] but PhaseNet channels are "NPS" | permute labels to model order + pre-training sanity assert (initial loss must be < 2) |
| Second fine-tune collapsed to constant output at inference | dead channels → per-window std-normalize → 1e9-scale values → BatchNorm running_var poisoned to 1.4×10¹⁸ | freeze BN during fine-tune + clip inputs to ±30σ |
| Live engine drifted to depth 149 km, M8.1 mid-event | stations whose true P fell below threshold got S/coda picked as "P"; a free-depth solver absorbs systematically-late picks with tiny residuals | incremental association (new picks must match predicted P ±3.5 s) + S-leg gating + hard 80 km depth ceiling |
| Phantom second event 100 s after the real one closed | mainshock coda keeps producing low-confidence picks | event refractory window + declaration requires ≥2 picks with conf ≥0.5 (coda triggers are uniformly weak) |
| Aggregated AI-magnitude σ reached 0.07 while the error was 2.2 | inverse-variance weighting assumes independent stations; event-level bias is shared | correlation floor: σ_agg² = 1/Σw + 0.3² |
| Live tab looked like a real earthquake during replays | rehearsal state rendered next to real CWA data | replay runs are labeled 演練/rehearsal, never pulse red, and their map overlay is opt-in |

The meta-lesson: **loss is a proxy; task-level evaluation is not optional.**
The collapsed model had a beautiful training loss (0.115) and a
plausible-looking val loss (0.21 ≈ the score of always predicting "noise"),
yet scored F1 = 0 on the actual picking task.

## Phase 2 — Multi-station location & magnitude convergence

Built on catalog P/S picks from 1,317 multi-station test-year events: a
pick-based hypocenter locator (vectorized 3-D grid search + bounded
least-squares, homogeneous velocity model, origin time from P legs), station
bootstrap → 1σ error ellipses, and PGA-attenuation magnitude
(`log10(PGA) = a·M + b·log10(R) + c`, fitted on the 2019 training year only —
out-of-sample MAE **0.21 (M4–5)** / **0.23 (M5–6)**). Everything runs as an
anytime estimator: each new trigger updates location + uncertainty + magnitude.

Replay results on real events:

- **Shallow M5.7 (2020-02-15, depth 8 km)**: magnitude available at **+6 s**
  after the first trigger (M5.6 ± 0.5, catalog M5.73); 1σ ellipse shrinks
  from 69×17 km to 9×3 km by +10 s.
- **Deep offshore M6.6 (2020-12-10 Yilan, depth 76 km)**: epicenter error
  122 km at 3 stations → 17 km at 40 stations (+11.5 s); depth only becomes
  constrained once S arrivals join — the physics behind why deep offshore
  events are hard for every EEW system.

Three findings the convergence experiments surfaced:

1. **Aperture beats station count.** The first 20 stations (arrival order,
   small aperture) can mislocate by 200+ km with P only; the *same number* of
   azimuthally spread stations achieves 13 km. Early-warning's fundamental
   geometric limit, measured.
2. **East-coast azimuthal gap → systematic bias.** For a coastal event with
   no seaward stations, the estimate carries a ~30 km landward bias that
   persists at 40 stations — the same mechanism that challenges operational
   warnings for offshore Taiwan events.
3. **Bootstrap ellipses understate true error.** The 1σ ellipse (data noise
   only) shrinks to 9×3 km while the true error stays ~24 km: velocity-model
   error is invisible to resampling. Displayed uncertainty must include a
   model-error floor — a calibration lesson that carries into the Phase 3
   decision layer (city countdowns as ranges, not points).

Known limits (deliberate v1 scope): homogeneous vp floor ≈ 10–15 km epicenter
error (a 1-D Taiwan model + station corrections is the upgrade); PGA
saturation degrades M6+ magnitudes (MAE 0.64) — the same physics behind
operational underestimation of large events.

## Phase 3 — Full-chain historical replays

`scripts/ingest_gdms.py` turns raw GDMS miniSEED into replay JSON: fine-tuned
PhaseNet picks on continuous records (velocity channels preferred, HL
fallback) + physical PGA via instrument response (dataless SEED). The Phase 2
engine then replays end-to-end — **waveforms → AI picks → location →
magnitude → city countdowns** — with the official CWA timeline overlaid.

| Event | Geometry | Location | Magnitude vs official |
|---|---|---|---|
| **2024-04-03 Hualien M7.2** (offshore) | azimuthal gap seaward | ~18 km @ origin+4 s, 110/112 stations picked | ours M6.0 / CWA M6.2 at the 9-s report; M6.6 / M6.8 by ~15-18 s — parallel underestimation (M7+ PGA saturation) |
| **2025-01-21 Chiayi Dapu ML6.4** (inland) | surrounded by stations | **1.3 km @ origin+8 s** | M6.8 ± 0.5 (slight over; one 2.1 g near-field station) |
| replay caveats | — | pre-origin noise picks filtered by origin time (a production system needs phase association, e.g. GaMMA) | catalog-final PGA used once S+2 s has passed |

Deliverable: a **unified bilingual console** (`docs/index.html`, one
self-building HTML) with two views — *Event Replay* (second-by-second
re-run of 0403/Dapu: ShakeAlert-style alert banner, predicted-intensity
contours, station intensity coloring, waveform strips with AI phase marks)
and *Live Monitor* (MapLibre map with NLSC tiles, CWA bulletins ~30 s poll,
USGS worldwide, live-engine status overlay). Deployable as-is to GitHub
Pages or Vercel (`vercel/` is a ready deployment unit with a serverless
CWA-key proxy).

## Phase 4 — Live streaming engine

`scripts/run_live.py` hosts the real engine: per-station ring buffers on a
time grid, PhaseNet every second, **incremental association** (a new pick
joins only if it matches the current solution's predicted P arrival — the
defense against S-mispicks being absorbed by a runaway-deep solution),
S-leg gating, an event refractory window against coda re-triggers, a hard
80 km crustal depth ceiling, and a quality gate (≥6 stations, ellipse
≤80 km) in front of any alert flag. Sources are pluggable: `replay`
(historical waveforms streamed honestly), `seedlink` (validated against
GEOFON; no public Taiwan feed exists), `trem` / `trem-sim` (Phase 6).
Notifications (email SMTP / Telegram bot / webhook, e.g. a Google Apps
Script mail relay) fire once per event, in daemon threads, each carrying a
research-prototype disclaimer.

Every one of those guards exists because a rehearsal broke without it —
the live engine was debugged by feeding it 0403 with no origin-time crutch,
and each failure mode (deep drift to 149 km, phantom coda events 100 s
after closure, alert from a 4-station garbage solution) became a rule.

## Phase 5 — AI early magnitude (honest result)

A 90k-parameter MagNet-style CNN regresses **final catalog magnitude from
the first 3 s of P** (3-component window + log-amplitude + hypocentral
distance from the *live* location estimate, never the truth), with a
Gaussian head so every estimate carries σ. Protocol: train 2019, test
2020–21, blind-test 2024/2025 events — the model is always out-of-time.

| | v1 (no distance) | v2 (distance-conditioned) |
|---|---|---|
| test MAE (2020–21, single station) | 0.59 | **0.52** |
| M6+ bias | −1.83 | **−1.12** |
| calibration \|z\|<1 (ideal 68%) | 77% | **66%** |
| blind 0403 (true M7.2) | 5.0 | 5.1 |
| blind Dapu (true M6.4) | 4.8 | 5.1 |

The verdict we publish rather than bury: distance conditioning fixes what
it should (near-zero bias below M5, halved bias at M6+, honest σ), but the
first 3 s of a M7 rupture physically resembles a M5–6 — **magnitude
saturation is an information limit, not a model bug**. In the live engine
the AI estimate therefore serves as a fast, conservative second opinion
next to the physics chain (which needs S+PGA but converges to the right
answer); growing-window "anytime" prediction is the documented next step.
A textbook detail the diagnostics surfaced: station-level predictions rise
with distance (4.3 near → 5.3 far) because distant stations see more of
the rupture's growth before their P+3 s window closes.

## Phase 6 — Real-time detection & automated audit

Three data layers, each with the strongest access Taiwan actually offers:

1. **True real time — ExpTech TREM** (community MEMS network, open HTTP
   API, per-station PGA/PGV/intensity at 1 Hz): the engine's trigger mode
   maps PGA jumps to arrivals and reuses the entire physics chain.
   `run_live.py --source trem --notify` is a genuine standing detector —
   when a felt earthquake occurs, it detects, locates, sizes, and messages
   your phone, from your own pipeline. (Community data, attributed,
   reference-only, 1 s minimum poll.)
2. **Official post-event waveforms — CWA E-A0015-004** (~12 min after each
   significant quake): `scripts/ingest_cwa_wave.py` parses the ASCII
   3-component gal-calibrated records and replays the engine blind. A
   GitHub Actions workflow (`.github/workflows/audit.yml`) does this
   automatically and commits the audit. First real audit (2026-07-31
   Taitung M4.7): first location origin+4.9 s, magnitude converged to
   M4.76 vs CWA M4.7 (17 km epicenter error), **and the alert gate
   correctly stayed silent** — true negatives are part of the record.
3. **Research-grade history — CWA GDMS + CWA Benchmark**: the blind-test
   and training substrate of Phases 1–3.

## Roadmap

- ~~Phases 0–6~~ **done**: replay engine → cross-domain benchmark → Taiwan
  fine-tune → locator/magnitude → full-chain replays + console → live engine
  + notifications → AI early magnitude → TREM real-time + automated audit.
- **Always-on hosting**: move the (torch-free) TREM trigger engine to a free
  always-on VM (GCP e2-micro) + push engine state to a cloud relay so the
  public console shows live detection.
- **Anytime magnitude (v3)**: growing-window MagNet heads (P+3/6/9 s) against
  M6+ saturation; replay-tab AI curve.
- **Model upgrades**: 1-D Taiwan velocity model + station corrections; phase
  association (GaMMA) to retire the origin-time noise filter; Pd-based
  magnitude; S-velocity legs for TREM triggers.
- **Hardware edge**: a personal Raspberry Shake — the one path to literal
  onsite early warning (P at the home sensor → alert before S) for an
  individual; would slot in as another engine source.
- **Phase 1 extras**: false-alarm rate on CWANoise, confidence calibration
  curves, longer training + partial BN unfreeze (current 0.70 is a
  conservative 15-epoch single-year recipe; in-domain ceiling is ~0.85+).

## Data & acknowledgements

- **CWA Benchmark** (Taiwan): Tang, K.-W., K.-Y. Chen, D.-Y. Chen, T.-L. Chin,
  and T.-Y. Hsu (2024), *The CWA Benchmark: A Seismic Dataset from Taiwan for
  Seismic Research*, SRL, doi:10.1785/0220230393.
- **PhaseNet**: Zhu, W. and G. C. Beroza (2019), GJI, doi:10.1093/gji/ggy423.
- **SeisBench**: Woollam, J. et al. (2022), SRL, doi:10.1785/0220210324.
- **Benchmark protocol**: Münchmeyer, J. et al. (2022), JGR: Solid Earth,
  doi:10.1029/2021JB023499.
- Iquique dataset: Woollam et al. (2019); STEAD: Mousavi et al. (2019).
- CWASN waveforms obtained via the CWA Geophysical Database Management
  System (GDMS): CWA Seismographic Network, doi:10.7914/SN/T5.
- **CWA Open Data** (opendata.cwa.gov.tw): earthquake bulletins
  (E-A0015/16) and post-event strong-motion waveforms (E-A0015-004).
- **ExpTech TREM** (exptech.dev): community MEMS real-time network, used
  read-only via their open API with attribution — community data, reference
  only, not an official source.
- Map data: © CARTO / OpenStreetMap contributors; NLSC (內政部國土測繪中心)
  WMTS; earthquake feeds by USGS Earthquake Hazards Program.

This is a research prototype. It must not be used as, or presented as, an
operational earthquake warning service. Redistribution of real-time strong-
motion alerts in Taiwan requires an agreement with the CWA.

## License

[MIT](LICENSE)

---

<a name="中文說明"></a>

# 中文說明

**台灣地震早期預警系統原型——以真實歷史事件盲測驗證、在社群測網上即時運行、
每次顯著地震後自動接受稽核。**

> **定位，三句話講清楚。** 這個**系統**是 EEW 原型：完整偵測鏈（波形 → AI
> 相位辨識 → 事件關聯 → 定位 → 規模 → 縣市警報判定 → 通知）全部存在且在
> 運行，並以模型從未見過的事件盲測驗證。這個**網站**是地震主控台：CWA/USGS
> 即時速報、ExpTech 社群測網上的即時偵測層、與可稽核的歷史事件回放。它
> **不是**警報「服務」：那需要國家測網的即時介接、24 小時運維與法定權責——
> 這些屬於中央氣象署。本專案所有介面均如實標示。

## 重點成果

- **串流回放引擎**：ring buffer + 滑動視窗推論，真實時間節奏回放；輸出含信心
  值的 picks 與逐步延遲（筆電 CPU p50 約 4–9 ms，遠低於 500 ms 的串流節奏）。
- **跨域基準**（依 Münchmeyer et al. 2022 協定）：同一 PhaseNet 架構、兩組公開
  預訓練權重、三個測試域。現成 PhaseNet 從原生區域搬到台灣，**P 波 F1 掉了
  0.24**——「台灣需要在地微調」的量化證據。
- **台灣微調**（CWA 2019 年 7 萬條、Kaggle T4、凍結 BatchNorm + 振幅裁剪）：
  **P F1 0.660 → 0.702、S F1 0.557 → 0.680**，受益最大的正是跨域傷最重的
  S 波（recall 0.52 → 0.69）。
- **誠實的工程記錄**：七個真實故障（上游 Windows 網址 bug、死通道毒壞
  BatchNorm 統計、標籤通道錯位……）連同根因與修法，見上方 Pitfalls 表。

## 三個發現

1. 台灣的 domain gap 遠大於智利——混合儀器（CWASN 短周期/寬頻 + TSMIP 強震儀）
   與在地地質都有影響。
2. 「用全球大資料訓練」不保證跨域穩健——STEAD 權重在所有測試域都輸給單一
   區域網訓練的權重。
3. 固定信心門檻不可跨權重沿用（STEAD 的 S 最佳門檻高達 0.80）——校準必須是
   第一級評估指標。

## 核心教訓

**Loss 只是代理指標，任務級評估不可省。** 崩潰的模型有漂亮的 train loss
（0.115）和看似合理的 val loss（0.21——恰好是「永遠回答噪音」的分數），
但在真正的相位辨識任務上 F1 = 0。

## Phase 2：多站定位與規模收斂

以目錄 P/S 到時建立可解釋的定位器（向量化網格搜尋＋有界最佳化、bootstrap
誤差橢圓）與 PGA 衰減規模估計（僅用 2019 訓練年擬合；樣本外 M4–5 誤差
±0.21）。實測回放：淺層 M5.7 在**首站觸發後 6 秒**即報出 M5.6±0.5，橢圓從
69×17 km 收斂至 9×3 km；宜蘭外海 76 km 深震誤差由 122 km（3 站）收斂至
17 km（40 站，+11.5 秒）。三個發現：**測站幾何比數量重要**（同樣 20 站，
到達順序取樣誤差 200+ km、方位角分散僅 13 km）；**東岸方位角缺口造成約
30 km 系統偏差**；**bootstrap 橢圓蓋不住速度模型誤差**（顯示的不確定性
必須加上模型誤差地板——Phase 3 決策層的城市倒數將以區間呈現）。

## Phase 3–6：從回放到活著的系統

**Phase 3（完整鏈回放＋主控台）**：GDMS 原始波形 → 微調 PhaseNet → 定位
→ 規模 → 縣市 PWS 判定，逐秒重演 0403 花蓮 M7.2 與 2025 大埔 ML6.4，
官方時間線並排對照。成果是單檔雙語主控台（`docs/index.html`）：回放分頁
（警示橫幅、預測震度圈、測站實測著色、波形＋AI 相位標記）＋即時分頁
（MapLibre 地圖、NLSC 圖磚、CWA 速報 30 秒輪詢、USGS 全球、引擎狀態疊加），
可直接部署 GitHub Pages／Vercel（`vercel/` 為現成部署單元）。

**Phase 4（live 引擎）**：真正的串流引擎——環形緩衝、每秒推論、增量關聯
（新 pick 須吻合當前解的預測 P 到時，防 S 誤認拖深）、S 腳閘門、事件冷卻期
（防 coda 假事件）、80 km 硬性深度上限、警報品質閘門（≥6 站且橢圓 ≤80 km）。
資料源可插拔：replay／SeedLink／TREM。通知（Email／Telegram／webhook，
含 Apps Script 郵件中繼）每事件一次、附研究原型聲明。

**Phase 5（AI 早期規模——誠實的結果）**：9 萬參數 CNN 從 P 波前 3 秒回歸
最終規模（距離條件化 v2；訓練 2019、測試 2020–21、盲測 2024/25）。測試集
MAE 0.59→0.52、M6+ 偏差 −1.83→−1.12、σ 校準近乎理想；但盲測 M7.2 仍收在
5.1——**前 3 秒的 M7 破裂在物理上就像 M5–6，規模飽和是資訊極限不是模型
bug**。故 AI 在引擎中定位為「快而保守的第二意見」，growing-window anytime
預測列為下一步。

**Phase 6（真即時＋自動稽核）**：三層資料架構——ExpTech TREM 社群測網
（公開 API、秒級 PGA/震度，引擎觸發模式真即時偵測；`--source trem
--notify` 掛機即是一台真的偵測器）；CWA E-A0015-004 官方強震波形（震後
約 12 分鐘發布，GitHub Actions 自動下載→引擎盲測→稽核紀錄 commit 回
repo——首筆稽核：2026-07-31 台東 M4.7，發震後 4.9 秒定位、最終 M4.76 對
官方 M4.7、警報閘門正確保持沉默）；GDMS 歷史資料（研究級盲測基底）。

本專案為研究原型，不得作為或宣稱為即時地震警報服務；台灣即時強震警報之
再散布需與中央氣象署簽約。ExpTech TREM 為社群觀測資料，僅供參考。
授權：MIT。
