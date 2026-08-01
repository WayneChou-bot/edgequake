# 部署看板到 Vercel

這個資料夾是可獨立部署的單元：`index.html`（看板頁）＋ `api/cwa.js`
（CWA open data 代理，API key 只存在伺服器端，不會外洩到前端）。

## 步驟

1. **申請 CWA API key（免費）**
   https://opendata.cwa.gov.tw → 註冊 → 會員中心 → API授權碼，
   格式像 `CWA-XXXXXXXX-...`。

2. **部署（二選一）**
   - GitHub 匯入：Vercel 網站 → Add New Project → 選 edgequake repo →
     **Root Directory 設成 `vercel`** → Deploy。
   - CLI：`npm i -g vercel`，然後在 `vercel/` 資料夾裡執行 `vercel`。

3. **設定環境變數**
   Vercel 專案 → Settings → Environment Variables →
   `CWA_API_KEY` = 你的授權碼 → Redeploy。

4. 開啟部署網址即可。頁面每 60 秒抓 `/api/cwa`（伺服器端快取 30 秒，
   不會打爆 CWA 配額），USGS 世界地震由瀏覽器直接抓。

## 本機開發（不用部署、不用 key）

    python scripts/poll_cwa.py --mock --serve   # 假資料
    python scripts/poll_cwa.py --serve          # 真資料（需 CWA_API_KEY）
    → http://localhost:8700

`index.html` 是 `web/monitor.html` 的複本——改了 web 版記得同步
（`copy ..\web\monitor.html index.html`）。

## 注意

- 這是「速報看板」（CWA 報告延遲約 1–2 分鐘），不是地震預警。
  頁面上已標示。
- live 引擎（`run_live.py`）跑在本機；Vercel 只放看板前端。之後要把
  引擎狀態上雲，再加一個狀態中繼（如 Upstash）即可。
