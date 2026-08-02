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

**Live console:** <https://edgequake-three.vercel.app> — real-time monitor
(CWA + USGS + live engine state relayed from a 24/7 cloud VM), full replays
of the 2024 Hualien M7.2 and 2025 Dapu ML6.4 earthquakes, and a public
**audit log** where every significant Taiwan earthquake is automatically
re-run through the engine and compared against the official catalog.

[中文版說明在下方](#中文說明) | English first, Chinese below.

---

## At a glance

- **First real-event audit** (2026-07-31 00:58 Taiwan time, Taitung M4.7,
  fully automated):
  first location **origin+4.9 s**, EEW issuance criteria met at
  **origin+9.2 s** (CWA's official performance: 10–20 s), final magnitude
  **M4.81 vs CWA M4.7** with 17 km epicenter error — and the public-alert
  gate **correctly stayed silent**. True negatives are part of the record.
- **Blind full-chain replays**: 2024-04-03 Hualien M7.2 located to ~18 km
  within 4 s of origin (110/112 stations picked); 2025 Dapu ML6.4 located
  to **1.3 km at origin+8 s**.
- **Taiwan fine-tune** of PhaseNet: P F1 **0.660 → 0.702**, S F1
  **0.557 → 0.680** — after quantifying a 0.24 P-F1 cross-domain drop that
  makes the case for local training.
- **Site-effect correction** from 327k catalog records (737 station terms)
  halves the mean final-magnitude error across the blind-test suite
  (0.17 → 0.09).
- **Impact, not just magnitude**: every estimate carries a PAGER-style
  population-exposure figure (WorldPop 1 km grid, ~0.4 ms per evaluation)
  and the most similar historical earthquakes from a 53-year catalog.
- **Runs on free infrastructure end to end**: a GCP e2-micro VM detects on
  137 community MEMS stations 24/7, relays state through a serverless
  Redis to the public page, GitHub Actions performs the post-event audits,
  and Gemini writes bilingual event reports — total cost $0.

## How it works

**Detection chain.** Per-station ring buffers on a shared time grid; the
fine-tuned PhaseNet picks P/S every second (waveform sources) or PGA-jump
triggers stand in for arrivals (MEMS sources). Picks pass **incremental
association** (a new pick must match the current solution's predicted P
within ±3.5 s — the defense against runaway-deep solutions absorbing
mispicks), then a vectorized grid-search locator with bootstrap error
ellipses and a hard 80 km crustal depth ceiling. Magnitude inverts a
PGA-attenuation relation fitted on the 2019 training year only, with
per-station **empirical site terms** removing ground-condition bias; a
distance-conditioned CNN gives a fast second opinion from the first 3 s
of P (honestly documented as saturating for M6.5+ — an information limit,
not a model bug).

**Two-tier alerting.** An **EEW tier** mirrors the CWA issuance rule
(M≥4.5 and predicted intensity ≥3) for timestamping and audit comparison;
the **PWS tier** mirrors the national cell-broadcast criteria and drives
actual notifications (email / Telegram / webhook), guarded by quality
gates earned from real failures: ≥6 stations, error ellipse ≤80 km, and
≥25 gal actually observed somewhere — because a real M6.5+ shakes the
ground, and phantom events never do.

**Live operation.** `run_live.py --source trem` runs 24/7 on the ExpTech
TREM community MEMS network (open API, attributed, reference-only) from a
free cloud VM, pushing state snapshots to Upstash Redis; the public
console reads them through a serverless endpoint, so during a real
earthquake any visitor sees the epicenter, P/S wavefronts, triggered
stations, magnitude convergence, population exposure, and the closest
historical analogs — live.

**Self-auditing loop.** ~12 minutes after every significant quake, CWA
publishes the official strong-motion waveforms. A GitHub Actions workflow
downloads them, replays the engine blind, writes a machine-readable audit
record (timing, magnitudes, errors, alert decisions, exposure,
`pop_version`), has an LLM narrate it bilingually (restricted to numbers
the pipeline computed — nothing invented), and commits everything back to
this repo. The console's audit tab renders the accumulating log. No
cherry-picking is possible: every event that qualifies gets audited.

## Verified results

Phase picking (time-split test sets, ±0.5 s tolerance):

| Weights (training domain) | Test domain | P F1 | S F1 |
|---|---|---|---|
| original (NCEDC, N. California) | NCEDC in-domain (paper) | 0.896 | 0.801 |
| original | Iquique (Chile) | 0.873 | 0.771 |
| original | **CWA (Taiwan)** | **0.660** | **0.557** |
| stead (STEAD, global) | CWA (Taiwan) | 0.228 | 0.220 |
| **cwa-ft (this repo)** | **CWA (Taiwan)** | **0.702** | **0.680** |

![Full-chain replay of the 2024-04-03 Hualien M7.2 earthquake](outputs/convergence_0403.gif)

Blind replays against the official record:

| Event | Location | Magnitude vs official |
|---|---|---|
| 2024-04-03 Hualien M7.2 (offshore) | ~18 km @ origin+4 s | M6.0 vs CWA M6.2 at their 9-s first report — an independent chain reproducing the operational system's early M7+ underestimation (PGA saturation is physics) |
| 2025-01-21 Dapu ML6.4 (inland) | 1.3 km @ origin+8 s | site-corrected final M6.34 (Δ0.06) |
| 2026-07-31 Taitung M4.7 (audit; local date) | 17 km, origin+4.9 s | final M4.81 (Δ0.11); EEW at origin+9.2 s; alert correctly silent |

Site-effect correction, final magnitude vs catalog:

| Event | raw | site-corrected |
|---|---|---|
| Taitung M4.7 | Δ0.06 | Δ0.11 |
| Hualien M7.2 | Δ0.19 | **Δ0.09** |
| Dapu M6.4 | Δ0.25 | **Δ0.06** |

Warning-time reality (measured, not promised): for a Hualien-offshore
event the full chain issues at about origin+12–20 s, giving Taipei ~15 s
of lead before destructive S waves — the same order as the official
system, because both are bound by the same physics. Inland events above
their epicenter (Dapu, Meinong) sit in the blind zone; the replay tab
shows exactly that.

## Honest limits

Point-source GMPE with average-site county predictions; magnitude
saturation for M6.5+ from 3 s of P; homogeneous velocity model
(~10–15 km location floor); community MEMS data is reference-only and
its station codes carry no site terms yet; population exposure assumes
static residential distribution; and this is a research prototype — not,
and never presented as, an operational warning service.

## Pitfalls (the ones worth telling)

| Problem | Root cause | Fix |
|---|---|---|
| Fine-tune collapsed to constant output — with a beautiful training loss | dead channels → per-window std-normalize → 1e9-scale values poisoned BatchNorm running stats | freeze BN during fine-tune + clip inputs to ±30σ; **loss is a proxy, task-level evaluation is not optional** |
| First fine-tune silently destroyed the model | labeller output order ≠ model channel order ("NPS") | permute labels + a pre-training sanity assert (initial loss must be < 2) |
| Live engine drifted to depth 149 km, M8.1 mid-event | S/coda picked as "P" at low-SNR stations; a free-depth solver absorbs late picks with tiny residuals | incremental association ±3.5 s + S-leg gating + hard 80 km ceiling |
| First live night: two phantom M6.6+ alert emails, no earthquake | 2.5 gal trigger threshold below the urban MEMS noise floor (2–4 gal); random noise associations fit a deep/offshore solution and "3 gal at 200 km" inverts to M6.6 | 8 gal threshold + 2-poll persistence + 150 km declaration radius + alerts require ≥25 gal observed |
| First LLM report truncated at 298 chars | Gemini flash "thinks" by default and thinking tokens count against `maxOutputTokens` | thinkingBudget 0 + larger cap + completeness check with retry |
| A 921-day M6.4 aftershock wore the "921 集集大地震" label | famous-event names matched by date only | names attach only when magnitude matches the mainshock ±0.4 |

A longer engineering log (13 entries) exists — the best stories are told
in person.

## Quick start

```bash
pip install -r requirements.txt

# replay a bundled real earthquake through the streaming engine
python scripts/demo_replay.py

# rebuild the console from the replay data
python scripts/build_dashboard.py

# rehearse the live engine on 0403 (simulated TREM feed, honest timeline)
python scripts/run_live.py --source trem-sim --event 0403 --speed 4

# run it for real (community MEMS network, notifications via env config)
python scripts/run_live.py --source trem --notify
```

Model training/evaluation pipelines (SeisBench + Kaggle recipes) are in
`scripts/` and run against the public CWA Benchmark dataset; dataset
download guards prevent accidental 100 GB pulls.

## Repository layout

```
src/edgequake/
├── pickers/        # PhaseNet interface (+ fine-tuned checkpoint)
├── replay/         # ring buffer + streaming replay engine
├── live/           # live engine, TREM source, notifier, state relay
├── location/       # locator, magnitude, site terms, replay simulation
├── models/         # AI early-magnitude (MagNet-style)
├── impact.py       # PAGER-style population exposure
└── similar.py      # historical similar-event retrieval
scripts/            # run_live, audit ingest, builders, data fetchers
assets/             # population grid, quake catalog, coastline (all versioned)
docs/ · vercel/     # the built console + Vercel deployment unit (incl. /api)
outputs/            # model weights, site terms, replay data, audit archive
.github/workflows/  # shadow-audit + data-freshness automation
```

## Roadmap

Self-calibrating site terms for TREM MEMS stations from accumulated live
data; growing-window "anytime" AI magnitude against M6+ saturation; a 1-D
Taiwan velocity model; phase association (GaMMA); a Raspberry Shake as a
true on-site sensor node.

## Data & acknowledgements

- **CWA Benchmark** (Taiwan): Tang, K.-W., K.-Y. Chen, D.-Y. Chen, T.-L. Chin,
  and T.-Y. Hsu (2024), *The CWA Benchmark: A Seismic Dataset from Taiwan for
  Seismic Research*, SRL, doi:10.1785/0220230393.
- **PhaseNet**: Zhu, W. and G. C. Beroza (2019), GJI, doi:10.1093/gji/ggy423.
- **SeisBench**: Woollam, J. et al. (2022), SRL, doi:10.1785/0220210324.
- **Benchmark protocol**: Münchmeyer, J. et al. (2022), JGR: Solid Earth,
  doi:10.1029/2021JB023499.
- Iquique dataset: Woollam et al. (2019); STEAD: Mousavi et al. (2019).
- CWASN waveforms via the CWA Geophysical Database Management System
  (GDMS): CWA Seismographic Network, doi:10.7914/SN/T5.
- **CWA Open Data** (opendata.cwa.gov.tw): earthquake bulletins
  (E-A0015/16) and post-event strong-motion waveforms (E-A0015-004).
- **ExpTech TREM** (exptech.dev): community MEMS real-time network, used
  read-only via their open API with attribution — community data, reference
  only, not an official source.
- **WorldPop** (www.worldpop.org): Taiwan 1 km population grid, Global2
  release R2025A (constrained UN-adjusted, 2026 estimate), University of
  Southampton. CC BY 4.0.
- **USGS FDSN event service** (earthquake.usgs.gov): 1973–present Taiwan
  regional catalog for similar-event retrieval; USGS Earthquake Hazards
  Program feeds on the console.
- Map data: © CARTO / OpenStreetMap contributors; NLSC (內政部國土測繪中心)
  WMTS.

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

**線上主控台：** <https://edgequake-three.vercel.app> ——即時監測（CWA 速報
＋USGS 全球＋雲端引擎即時狀態）、0403 花蓮 M7.2 與大埔 ML6.4 的完整回放、
以及公開稽核紀錄。

## 一眼看懂的數字

首筆全自動稽核（2026-07-31 台灣時間 00:58 台東 M4.7）：發震後 **4.9 秒**首次定位、
**9.2 秒**達強震即時警報發布條件（官方效能 10–20 秒）、最終規模 **M4.81
對官方 M4.7**、震央誤差 17 公里，且國家級警報閘門**正確保持沉默**——
「沒發警報」也是紀錄的一部分。盲測回放：0403 花蓮 M7.2 於發震後 4 秒定位
至 18 公里內；大埔 ML6.4 於發震後 8 秒定位至 **1.3 公里**。PhaseNet 台灣
微調 P F1 0.660→0.702、S 0.557→0.680；場址效應修正讓盲測最終規模平均
誤差**減半**（0.17→0.09）。每次估計同步輸出曝險人口（WorldPop 1 km 網格，
單次 0.4 毫秒）與 53 年目錄中最相似的歷史地震。整套系統跑在免費資源上：
GCP 免費 VM 全天候偵測 137 個社群測站、Upstash 中繼即時狀態、GitHub
Actions 自動稽核、Gemini 生成雙語事件報告——總成本 0 元。

## 系統怎麼運作

引擎逐秒推論（波形源用微調 PhaseNet，MEMS 源以 PGA 跳升代相位），新 pick
須吻合當前解的預測 P 到時（±3.5 秒）才能加入——這是防止誤 pick 把解拖向
深部的關鍵防線；定位用向量化網格搜尋＋bootstrap 誤差橢圓＋80 km 硬性深度
上限；規模反演僅以 2019 訓練年擬合的衰減式，並以 737 站的經驗場址項消除
地盤偏差；AI 從 P 波前 3 秒給出快速第二意見（M6.5+ 飽和如實記載——那是
資訊極限，不是模型 bug）。警報分兩級：EEW 級對齊 CWA 發布條件（M≥4.5 且
預估震度 ≥3）供稽核對比；PWS 級對齊國家級警報門檻並實際發信，且必須
≥6 站、橢圓 ≤80 km、**某站實測 ≥25 gal**——真的 M6.5 會真的搖，幽靈事件
不會。每次顯著地震後約 12 分鐘，官方強震波形一釋出，GitHub Actions 便
自動下載、盲測重演、寫入稽核紀錄、由 LLM 敘述成雙語報告（僅准敘述管線
算出的數字）、commit 回 repo——無法挑選、每筆都留。

## 誠實的限制

點震源假設、均勻速度模型（定位地板約 10–15 km）、M6.5+ 規模飽和、社群
測站資料僅供參考且尚無場址項、曝險人口為常住靜態估計。本專案為研究原型，
不得作為或宣稱為即時地震警報服務；台灣即時強震警報之再散布需與中央氣象署
簽約。ExpTech TREM 為社群觀測資料，僅供參考。授權：MIT。
