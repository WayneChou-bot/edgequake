# EdgeQuake AI — Phase 0 可行性原型

> Historical seismic waveform replayed as a real-time stream — **not** a live
> earthquake warning service.

把歷史地震波形當成即時串流回放，用預訓練 PhaseNet 做滑動視窗滾動推論，
輸出逐時間點 P/S 機率、觸發 pick（含信心值與偵測延遲）、並量測每次推論的
wall-clock 延遲。這是規劃書 v2 的 Phase 0：驗證整條「串流 → ring buffer →
滑窗推論 → 觸發 → 視覺化」管線可行。

## 首次執行結果（本原型已驗證）

- 測試波形：ObsPy 內建真實地震記錄（BW.RJOB，100 Hz 三分量）＋ 30 秒自身背景噪音暖機
- P pick：34.51 s，信心 0.264，偵測延遲 0.49 s
- S pick：35.72 s，信心 0.657，偵測延遲 0.28 s（S−P ≈ 1.2 s，符合極近震）
- 推論延遲（CPU）：p50 ≈ 9 ms、p95 ≈ 12 ms（首次呼叫含框架暖機 ~400 ms，屬一次性）
- 60 秒背景噪音中零誤觸發

![demo](outputs/replay_demo.png)

## 安裝與執行

```bash
pip install -r requirements.txt
python scripts/demo_replay.py                  # 內建示範事件
python scripts/demo_replay.py your_event.mseed # 任何三分量 miniSEED
```

Picker 解析順序：

1. **SeisBench 預訓練 PhaseNet**（正常筆電路徑；首次執行會自動下載權重並快取）
2. **AI4EPS 官方 TensorFlow checkpoint**（離線備援；權重在 GitHub）：

```bash
git clone --depth 1 https://github.com/AI4EPS/PhaseNet.git
pip install tensorflow-cpu tf_keras
TF_USE_LEGACY_KERAS=1 python scripts/demo_replay.py
```

## 程式結構

```
src/edgequake/
├── pickers/
│   ├── base.py               # Picker 介面：window -> (N,P,S) 逐點機率
│   ├── seisbench_picker.py   # SeisBench from_pretrained（預設）
│   └── tf_phasenet.py        # AI4EPS TF checkpoint（離線備援）
├── replay/
│   ├── ring_buffer.py        # 固定容量環形緩衝區（含冷啟動零填充）
│   └── engine.py             # 回放引擎：hop 推進、滾動推論、觸發、延遲量測
└── viz.py                    # 三聯圖：波形+picks / P,S 機率時間線 / 推論延遲
scripts/demo_replay.py        # 端到端 demo
```

## 原型階段就學到的三件事（面試素材）

1. **冷啟動假警報**：ring buffer 零填充的邊界對模型看起來像一個震相 onset，
   第一版在 t=0 產生信心 0.875 的假 P。修正：緩衝區未填滿前抑制觸發
   （真實測站永遠不會在事件上冷啟動，回放系統要模擬這一點）。
2. **門檻即校準問題**：本事件（模型訓練域外的德國測站）P 峰值 0.264，
   PhaseNet 預設門檻 0.3 會漏報、0.25 會抓到——「信心值該多少才值得行動」
   正是 Phase 1 的 calibration / abstention 研究主題，這裡已經有了第一個實例。
3. **延遲結構**：穩態推論 ~9 ms（CPU）遠小於 500 ms hop，瓶頸不在模型；
   首次呼叫的框架暖機（~400 ms）在真實系統要在開機時做，不能落在事件中。

## 已知限制（誠實列出）

- 單測站、單事件、域外資料——尚未在台灣資料（CWA Benchmark）上驗證
- 暖機噪音是拼接合成，僅用於管線驗證
- 觸發邏輯是最簡單的門檻 + refractory，尚無關聯、去重、多站融合
- pick 時刻取窗內機率峰值，未做 sub-sample 精化

---

# Phase 1 — 資料載入與相位辨識評估（已交付）

遵循 Münchmeyer et al. 2022「Which picker fits my data?」協定的簡化版：
每條測試 trace 切一個視窗（標註到時隨機落在視窗中央區，避免模型利用固定位置
先驗），跑一次 picker 並**快取逐點機率曲線**，之後在快取上掃描門檻——
模型只推論一次，門檻掃描零成本。

## 指標

P/S 各自的 precision / recall / F1（容忍 ±0.5 s，可調）、殘差 MAE / RMSE / std、
門檻掃描曲線、PR 曲線，以及**雜訊誤報率（false alarms per hour）**——
無標註的 trace 自動當作雜訊窗計入誤報統計。

## 使用方式

```bash
# 1) 先用合成資料集驗證管線（本機生成，數秒完成）
python scripts/make_synthetic_dataset.py
python scripts/eval_picking.py --dataset data/synthetic_test --limit 50

# 2) 小型真實資料集煙霧測試（Iquique ~5 GB，首次自動下載）
python scripts/eval_picking.py --dataset iquique --limit 500

# 3) CWA 台灣資料——先看下載量再決定！
python scripts/eval_picking.py --preview-cwa _2019 _2020 _2021
python scripts/eval_picking.py --dataset cwa --chunks _2019 --confirm-download --limit 1000
```

## ⚠ CWA 下載防護（重要）

CWA 全量約 **836 GB**。SeisBench 的 chunk 以年為單位（`_2011`…`_2021`），
但實際下載的是 **4 年合併的 tar.gz**——指定 `_2019` 也會拉下整包
merge2019_2021。因此 `load_cwa()` 沒有 `confirm=True` 一律拒絕下載，
`--preview-cwa` 會先向 Hugging Face 查詢真實檔案大小。
筆電建議路線：先用 Iquique 驗證流程 → 查大小 → 視情況用外接 SSD 或
雲端（Kaggle/Colab）跑 CWA。

## 合成資料集驗證結果（管線正確性，非模型品質）

40 個注入事件全數命中（P/S F1 = 1.0）、10 個純噪音窗零誤報。
P 殘差集中在 +0.19 s 且 std 僅 0.027 s——這是**系統性偏移**：合成資料的
「名目真值」來自 Phase 0 的 hop 解析度 picks，本身就有偏差；模型高度一致的
offset 反而證明了殘差統計管線在正確運作（真實資料集的人工標註不會有此現象）。

## Phase 1 核心結果：跨域 domain-shift 對照表（2026-07-29 實測）

同一個 PhaseNet 架構、兩組公開預訓練權重、三個測試域（各 500–1000 個測試窗，
容忍 ±0.5 s，時間切分測試集）：

| 權重（訓練域） | 測試域 | P F1 @0.3 | S F1 @0.3 | 最佳門檻 F1 (P/S) | P 殘差 std |
|---|---|---|---|---|---|
| original（NCEDC 北加州）| NCEDC 同域（論文值）| 0.896 | 0.801 | — | 0.052 s |
| original | Iquique（智利）| 0.873 | 0.771 | ≈同 0.3 | 0.131 s |
| original | **CWA（台灣）** | **0.660** | **0.557** | — | 0.170 s |
| stead（STEAD 全球）| Iquique（智利）| 0.390 | 0.281 | 0.45 @0.70 / 0.65 @0.80 | 0.129 s |
| stead | **CWA（台灣）** | **0.228** | **0.220** | — | 0.171 s |
| **cwa-ft（本專案，2019 微調）** | **CWA（台灣）** | **0.702** | 0.635 | 0.70 @0.30 / **0.68 @0.50** | 0.172 s |

（CWA 數字為修正 TSMIP 到時縮放 bug 後的版本，2026-07-30 重測。）

### Phase 1b：台灣微調結果（2026-07-30）

用 CWA 2019 年 70,000 條 traces（Kaggle T4，15 epochs，凍結 BatchNorm +
振幅裁剪 ±30σ）微調 PhaseNet original 權重：

- **P F1 0.660 → 0.702**（precision 0.749 → 0.812）
- **S F1 0.557 → 0.680**（recall 0.522 → **0.690**，受益最大——S 波形受在地
  地質影響最深，最需要台灣資料）
- 殘差分布集中無偏移，時間精度未因微調劣化
- 誠實註記：距同域訓練天花板（~0.85+）仍有空間；future work：更多 epochs
  + LR 衰減、部分解凍 BN、加入 2015–2018 年份。訓練用 2019（官方 dev 年），
  2020–2021 測試集全程未接觸。

**訓練配方的兩個教訓（已寫入 notebook 防呆）**：
1. 標籤通道順序必須對齊模型（labeller [P,S,N] vs. PhaseNet "NPS"）——錯位
   訓練會靜默地毀掉模型；已加「訓練前 pretrained loss < 2」sanity assert。
2. CWA 死通道 trace 經逐窗標準化會產生 10⁹ 級數值，毒壞 BatchNorm running
   stats（實測 running_var 達 1.4×10¹⁸ → 推論模式輸出與輸入無關的常數，
   train loss 卻正常）——教訓：**loss 是代理指標，任務級 F1 評估不可省**。
   修補：微調凍結 BN + 輸入裁剪。

三個發現：

1. **台灣的 domain gap 遠大於智利**：original 權重 P F1 從同域 0.896 →
   智利 0.873（−0.02）→ 台灣 0.629（−0.27）。台灣測試的 recall 只有 0.56
   （漏一半的 P），且殘差分布右偏（系統性晚報）——CWA 混合 CWASN 短周期／
   寬頻與 TSMIP 強震儀（200 Hz 重取樣）、事件規模分布也偏大，儀器與訊號
   特性都與 NCEDC 不同。**這就是「必須用台灣資料微調」的量化證據。**
2. **「全球大資料」不等於跨域泛化**：STEAD 權重（百萬級全球資料訓練）在兩個
   測試域都遠遜於單一區域網訓練的 original 權重，在台灣連最佳門檻都只有
   P F1 0.25。
3. **固定門檻慣例不可信**：同樣是「信心 0.3」，original 權重下接近最佳操作點，
   STEAD 權重下卻對應大量誤報（S 最佳門檻在 0.80）。信心值的語義隨權重而變
   ——校準（calibration）必須作為第一級評估指標，這是本專案 Trustworthy AI
   軸線的第一個實證。

### 踩坑記錄（真實工程問題，含解法）

| 問題 | 根因 | 解法（本 repo） |
|---|---|---|
| SeisBench 資料集下載一律 404（Windows）| 上游用 `os.path.join` 組 URL → 反斜線 | `loader._apply_windows_url_fix()` monkeypatch |
| CEED/CWA 不會走 Hugging Face 下載 | 上游 `compile_from_source` 預設 False | loader 明確開啟 |
| CWA 走 SeisBench 庫要抓 43 GB/年（1.3 MB/s）| 庫存的是未壓縮 HDF5 | `scripts/fetch_cwa_hf.py` 強制 HF 壓縮包路線（27 GB／三年）|
| CWA metadata 使 pandas 崩潰 | `source_origin_time` 少數列缺小數秒 | `scripts/fix_cwa_metadata.py` 一次性正規化 |
| 本專案評估器自身 bug：TSMIP 200 Hz 的到時樣本未隨重取樣縮放 | metadata 到時以原始取樣率記錄，波形被重取樣到 100 Hz | `eval/picking.py` 依 `trace_sampling_rate_hz` 縮放；修正後台灣 P F1 0.629→0.660 |
| Kaggle 端下載訓練資料失敗 | 2018 年單檔 88.6 GB 超出 Kaggle 磁碟；長串流無續傳被斷線 | 改為本機產出 3.5 GB 精簡訓練集（`make_cwa_train_subset.py`）上傳 Kaggle Dataset |

## 下一步（規劃書 v2）

Phase 1b：**台灣微調**（Kaggle GPU，目標把 P F1 從 0.63 拉回 0.85+）、
校準曲線（信心 vs. 命中率）、abstention 分析、CWANoise 誤報率量測。
Phase 2：多站收斂（到時差定位、誤差橢圓、anytime prediction）。
Phase 3：0403 花蓮／2025 大埔案例回放 + PWS 門檻決策層 + MapLibre 儀表板。
