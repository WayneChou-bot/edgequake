// Vercel serverless proxy for CWA open data (keeps the API key server-side).
// Set CWA_API_KEY in the Vercel project's Environment Variables.
const API = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore';
const DATASETS = [['E-A0015-001', 'significant'], ['E-A0016-001', 'local']];
const I_ORDER = {'1級': 1, '2級': 2, '3級': 3, '4級': 4, '5弱': 5,
                 '5強': 5.5, '6弱': 6, '6強': 6.5, '7級': 7};

function normIntensity(txt) {
  const m = String(txt || '').match(/([1-7])(級|弱|強)?/);
  if (!m) return null;
  return m[1] + (m[2] === '弱' || m[2] === '強' ? m[2] : '級');
}

function parseReport(eq, kind) {
  try {
    const info = eq.EarthquakeInfo || {};
    const epi = info.Epicenter || {};
    const mag = info.EarthquakeMagnitude || {};
    let best = null, bestV = -1;
    for (const area of (eq.Intensity || {}).ShakingArea || []) {
      const it = normIntensity(area.AreaIntensity);
      const v = I_ORDER[it] ?? -1;
      if (v > bestV) { best = it; bestV = v; }
    }
    const ev = {
      t: String(info.OriginTime || '').slice(0, 16),
      lat: parseFloat(epi.EpicenterLatitude),
      lon: parseFloat(epi.EpicenterLongitude),
      depth: Math.round(parseFloat(info.FocalDepth || 0)),
      mag: parseFloat(mag.MagnitudeValue),
      loc: String(epi.Location || ''),
      maxI: best, url: String(eq.Web || '') || null, kind,
    };
    if (!isFinite(ev.lat) || !isFinite(ev.lon) || !isFinite(ev.mag))
      return null;
    return ev;
  } catch { return null; }
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const key = process.env.CWA_API_KEY;
  if (!key) {
    res.status(500).json({ error: 'CWA_API_KEY not configured' });
    return;
  }
  const events = [];
  for (const [ds, kind] of DATASETS) {
    try {
      const url = `${API}/${ds}?Authorization=${encodeURIComponent(key)}` +
                  `&limit=15&format=JSON`;
      const r = await fetch(url);
      const data = await r.json();
      for (const eq of (data.records || {}).Earthquake || []) {
        const ev = parseReport(eq, kind);
        if (ev) events.push(ev);
      }
    } catch { /* dataset down: continue with the other */ }
  }
  events.sort((a, b) => (a.t < b.t ? 1 : -1));
  res.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=120');
  res.status(200).json({
    updated: new Date(Date.now() + 8 * 3600e3).toISOString()
      .slice(0, 19).replace('T', ' '),
    source: 'cwa-opendata(vercel)',
    events: events.slice(0, 30),
  });
}
