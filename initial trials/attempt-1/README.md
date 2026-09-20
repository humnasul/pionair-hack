# Risk check app — deployment guide

Two pieces, deployed separately:

- `risk-assessment.html` — the frontend. Static file, no build step.
- `aqi-proxy-worker.js` — a small backend that fetches real AQI data
  server-side (so your API token never sits in browser-visible code).

## Why two pieces

Browsers can't be trusted to hold a private API key, and the AQI provider
doesn't allow direct calls from arbitrary websites. The Worker sits in the
middle: the frontend calls the Worker, the Worker calls the AQI provider
with a secret token, and only the resulting number comes back.

## 1. Deploy the AQI proxy (Cloudflare Workers, free tier)

```bash
npm install -g wrangler
wrangler login
```

Create `wrangler.toml` next to `aqi-proxy-worker.js`:

```toml
name = "aqi-proxy"
main = "aqi-proxy-worker.js"
compatibility_date = "2024-01-01"
```

Get a free token from https://aqicn.org/data-platform/token/, then:

```bash
wrangler secret put WAQI_TOKEN
wrangler deploy
```

Wrangler prints a URL like `https://aqi-proxy.yoursubdomain.workers.dev`.
Test it directly in a browser:

```
https://aqi-proxy.yoursubdomain.workers.dev/?lat=42.44&lon=-76.5
```

You should get back JSON like `{"aqi": 38, "city": "Ithaca, NY", ...}`.

## 2. Point the frontend at it

Open `risk-assessment.html`, find this near the top of the `<script>`:

```javascript
const CONFIG = {
  AQI_PROXY_URL: "https://YOUR-WORKER-URL.workers.dev"
};
```

Replace the URL with the one Wrangler gave you.

## 3. Host the frontend

`risk-assessment.html` is a single static file — any of these work:

- **Vercel / Netlify**: drag-and-drop deploy, or connect a GitHub repo
- **Cloudflare Pages**: same account as the Worker, convenient to keep both together
- **GitHub Pages**: free, works fine for a static file like this

Once deployed to its own URL (not embedded in an iframe), the browser's
location permission prompt will behave normally — that's also what fixes
the geolocation issue from before, since iframe-embedded pages (like a
Claude artifact) don't reliably get permission delegation from their host.

## Notes on accuracy

- AQI values are now real, live data from WAQI.
- The risk *weights* (how much a given AQI moves the score) are still the
  literature-derived coefficients from the earlier research pass, not
  fitted to real outcome data yet. See the PyMC discussion earlier in this
  conversation for the next step once you have training data.
