// Public live-engine state: reads the snapshot the engine pushes to
// Upstash Redis (see src/edgequake/live/relay.py). Edge-cached 1 s so
// viewer traffic barely touches Redis. Returns null when no engine is
// pushing (key expires after 90 s) — the console then hides the card.
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const tok = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  res.setHeader('Cache-Control', 's-maxage=1, stale-while-revalidate=3');
  if (!url || !tok) {
    res.status(200).json(null);
    return;
  }
  try {
    const r = await fetch(`${url}/get/eq_state`, {
      headers: { Authorization: `Bearer ${tok}` },
    });
    const d = await r.json();
    res.status(200).json(d && d.result ? JSON.parse(d.result) : null);
  } catch {
    res.status(200).json(null);
  }
}
