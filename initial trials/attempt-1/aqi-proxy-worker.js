/**
 * AQI proxy — deploy this to Cloudflare Workers (free tier is enough).
 *
 * Why this exists: browsers can't safely call the AQI provider directly
 * because that would expose your API token in client-side code, and the
 * risk-assessment app's own sandbox can't reach external APIs at all.
 * This Worker sits in between: the app calls this Worker, this Worker
 * calls the World Air Quality Index (WAQI) API with a server-side token.
 *
 * Setup:
 *   1. Sign up for a free token at https://aqicn.org/data-platform/token/
 *   2. Install the Cloudflare CLI:  npm install -g wrangler
 *   3. wrangler login
 *   4. wrangler secret put WAQI_TOKEN      (paste your token when prompted)
 *   5. wrangler deploy
 *   6. Copy the resulting *.workers.dev URL into CONFIG.AQI_PROXY_URL
 *      in risk-assessment.html
 *
 * You'll also need a minimal wrangler.toml alongside this file:
 *
 *   name = "aqi-proxy"
 *   main = "aqi-proxy-worker.js"
 *   compatibility_date = "2024-01-01"
 */

export default {
  async fetch(request, env) {
    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }

    const { searchParams } = new URL(request.url);
    const lat = searchParams.get("lat");
    const lon = searchParams.get("lon");

    if (!lat || !lon) {
      return json({ error: "Missing lat/lon query params" }, 400);
    }

    const token = env.WAQI_TOKEN; // set via `wrangler secret put WAQI_TOKEN`
    if (!token) {
      return json({ error: "Server misconfigured: missing WAQI_TOKEN" }, 500);
    }

    try {
      const upstream = await fetch(
        `https://api.waqi.info/feed/geo:${lat};${lon}/?token=${token}`
      );
      const data = await upstream.json();

      if (data.status !== "ok") {
        return json({ error: "AQI provider error", detail: data.data }, 502);
      }

      return json({
        aqi: data.data.aqi,
        city: data.data.city?.name ?? null,
        dominantPollutant: data.data.dominentpol ?? null,
        measuredAt: data.data.time?.s ?? null
      });
    } catch (err) {
      return json({ error: "Upstream request failed", detail: String(err) }, 502);
    }
  }
};

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  };
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() }
  });
}
