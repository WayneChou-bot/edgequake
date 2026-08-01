/**
 * EdgeQuake 警報通知 — Google Apps Script webhook（ASU 反饋系統同款架構）
 *
 * 運作：引擎（或 GitHub 稽核 workflow）偵測到事件時 POST 到本腳本網址，
 * 腳本驗證 token 後用你的 Gmail 寄出通知信 — 不需要任何 SMTP 密碼。
 *
 * 設定步驟：
 *   1. script.google.com → 新專案 → 貼上本檔
 *   2. 專案設定 → 指令碼屬性 → 新增：
 *        WEBHOOK_TOKEN = 自訂一串隨機文字
 *        MAIL_TO       = 收件人（逗號分隔，可留自己）
 *   3. 部署 → 新增部署作業 → 網頁應用程式 →
 *        執行身分：我；誰可以存取：任何人 → 取得網址
 *   4. 引擎那邊設環境變數：
 *        set EQ_WEBHOOK_URL=https://script.google.com/macros/s/xxxx/exec
 *        set EQ_WEBHOOK_TOKEN=跟步驟2相同的token
 *      然後 python scripts/notify_test.py 測試。
 *
 * 注意：MailApp 一般帳戶每日約 100 封上限 — 地震警報用量綽綽有餘。
 */
function doPost(e) {
  var props = PropertiesService.getScriptProperties();
  var token = (e && e.parameter && e.parameter.token) || "";
  if (token !== props.getProperty("WEBHOOK_TOKEN")) {
    return ContentService.createTextOutput("forbidden");
  }
  var payload;
  try {
    payload = JSON.parse(e.postData.contents);
  } catch (err) {
    return ContentService.createTextOutput("bad payload");
  }
  var to = props.getProperty("MAIL_TO");
  if (!to) {
    return ContentService.createTextOutput("no recipients");
  }
  MailApp.sendEmail({
    to: to,
    subject: payload.subject || "EdgeQuake 通知",
    body: (payload.body || "") +
      "\n\n— EdgeQuake 自動通知（研究原型，非官方警報，請勿回覆）",
  });
  return ContentService.createTextOutput("sent");
}
