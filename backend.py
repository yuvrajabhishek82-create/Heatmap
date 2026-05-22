"""
India Attention Map — FastAPI Backend
======================================
- Serves the frontend (index.html) as a static file at /
- Google News RSS ingestion for all Indian states (8 languages)
- OpenAI GPT-4o-mini for AI summaries and narrative classification
- Decay-weighted attention scoring (36h half-life)
- In-memory store with full state — swap for PostgreSQL later
- Auto-ingests on startup; POST /api/ingest/run to refresh manually

Deploy on Railway:
  1. Set env var OPENAI_API_KEY
  2. Push to GitHub → connect Railway → done
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    openai_client = None
    HAS_OPENAI = False

# ============================================================================
# CONSTANTS
# ============================================================================

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu and Kashmir",
]

NARRATIVES = [
    "unemployment", "nationalism", "religion", "corruption", "economy",
    "inflation", "caste", "language politics", "regional identity",
    "education", "law & order", "governance", "elections", "security",
    "border issues", "environment", "farmer issues", "infrastructure",
    "tribal issues", "migration",
]

LANG_CODES = ["en-IN", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN", "kn-IN", "ml-IN"]

STATE_ALIASES: dict[str, list[str]] = {
    "Tamil Nadu":        ["tamil nadu", "tn", "chennai", "coimbatore", "madurai", "tirunelveli"],
    "Uttar Pradesh":     ["uttar pradesh", "up", "lucknow", "varanasi", "noida", "kanpur", "ayodhya", "prayagraj"],
    "Maharashtra":       ["maharashtra", "mumbai", "pune", "nagpur", "nashik", "thane"],
    "Bihar":             ["bihar", "patna", "gaya", "muzaffarpur", "bhagalpur"],
    "West Bengal":       ["west bengal", "bengal", "kolkata", "calcutta", "howrah", "siliguri"],
    "Karnataka":         ["karnataka", "bengaluru", "bangalore", "mysuru", "hubli", "mangaluru"],
    "Kerala":            ["kerala", "thiruvananthapuram", "kochi", "kozhikode", "wayanad"],
    "Gujarat":           ["gujarat", "ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar"],
    "Rajasthan":         ["rajasthan", "jaipur", "jodhpur", "udaipur", "kota", "jaisalmer"],
    "Madhya Pradesh":    ["madhya pradesh", "mp", "bhopal", "indore", "gwalior", "jabalpur"],
    "Telangana":         ["telangana", "hyderabad", "warangal", "secunderabad"],
    "Andhra Pradesh":    ["andhra pradesh", "vijayawada", "visakhapatnam", "amaravati", "tirupati"],
    "Punjab":            ["punjab", "amritsar", "ludhiana", "chandigarh", "jalandhar"],
    "Haryana":           ["haryana", "gurugram", "gurgaon", "faridabad", "panipat"],
    "Odisha":            ["odisha", "orissa", "bhubaneswar", "cuttack", "puri"],
    "Jharkhand":         ["jharkhand", "ranchi", "jamshedpur", "dhanbad"],
    "Chhattisgarh":      ["chhattisgarh", "raipur", "bilaspur", "bastar"],
    "Assam":             ["assam", "guwahati", "dispur", "dibrugarh"],
    "Delhi":             ["delhi", "new delhi", "ncr"],
    "Jammu and Kashmir": ["kashmir", "jammu", "srinagar", "j&k", "pahalgam", "gulmarg"],
    "Himachal Pradesh":  ["himachal", "shimla", "manali", "dharamshala"],
    "Uttarakhand":       ["uttarakhand", "dehradun", "haridwar", "char dham"],
    "Goa":               ["goa", "panaji"],
    "Manipur":           ["manipur", "imphal"],
    "Nagaland":          ["nagaland", "kohima"],
    "Mizoram":           ["mizoram", "aizawl"],
    "Tripura":           ["tripura", "agartala"],
    "Meghalaya":         ["meghalaya", "shillong"],
    "Arunachal Pradesh": ["arunachal", "itanagar", "tawang", "arunachal pradesh", "east siang", "west kameng"],
    "Sikkim":            ["sikkim", "gangtok", "namchi", "gyalshing"],
    "Nagaland":          ["nagaland", "kohima", "dimapur", "naga", "nagaland state"],
    "Mizoram":           ["mizoram", "aizawl", "lunglei", "mizo", "mizoram state"],
    "Meghalaya":         ["meghalaya", "shillong", "tura", "khasi", "garo hills"],
    "Tripura":           ["tripura", "agartala", "tripuri", "tripura state"],
    "Manipur":           ["manipur", "imphal", "manipuri", "meitei", "churachandpur"],
}

NARRATIVE_KEYWORDS: dict[str, list[str]] = {
    "unemployment":      ["unemployment", "jobless", "hiring", "recruitment exam", "vacancy", "layoffs", "naukri"],
    "corruption":        ["scam", "corruption", "bribe", "ed raid", "cbi", "paper leak", "embezzle"],
    "religion":          ["temple", "mosque", "festival", "communal", "religious", "yatra", "waqf"],
    "economy":           ["gdp", "growth", "investment", "industry", "msme", "export", "manufacturing"],
    "inflation":         ["inflation", "price rise", "petrol price", "onion price", "tomato price", "cpi"],
    "caste":             ["caste", "dalit", "obc", "reservation", "quota", "sc/st"],
    "language politics": ["hindi imposition", "three-language", "kannada signage", "tamil", "marathi"],
    "regional identity": ["regional", "sons of soil", "outsider", "marathi manoos", "bhumiputra"],
    "education":         ["school", "university", "neet", "jee", "exam", "syllabus", "student strike"],
    "law & order":       ["crime", "arrest", "encounter", "murder", "rape", "kidnap", "police"],
    "governance":        ["cm", "cabinet", "minister", "policy", "scheme", "yojana", "bill", "assembly"],
    "elections":         ["election", "vote", "poll", "campaign", "candidate", "bypoll"],
    "border issues":     ["border", "loc", "infiltration", "drone", "bsf", "china border", "lac", "myanmar"],
    "environment":       ["pollution", "air quality", "climate", "flood", "drought", "heatwave", "cyclone"],
    "farmer issues":     ["farmer", "msp", "agriculture", "kisan", "crop", "monsoon", "harvest"],
    "infrastructure":    ["highway", "metro", "airport", "expressway", "bullet train", "road"],
    "nationalism":       ["nation", "patriotic", "anti-national", "tiranga", "independence"],
    "tribal issues":     ["tribal", "adivasi", "vanvasi", "forest rights", "scheduled tribe"],
    "migration":         ["migrant", "migration", "remittance", "gulf return", "exodus"],
    "security":          ["terror", "blast", "attack", "naxal", "maoist", "encounter", "ied"],
}

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anger":   ["outrage","fury","angry","slam","blast","condemn","protest","demand","furious",
                "backlash","controversy","clash","dispute","accuse","oppose","reject","denounce",
                "criticize","scam","corruption","fraud","arrested","violence","riot","agitation"],
    "anxiety": ["worry","concern","anxious","alarm","uncertain","crisis","panic","shortage",
                "inflation","unemployment","struggling","challenging","difficult","slowdown",
                "debt","deficit","flood","drought","disaster","tension","unrest","instability"],
    "hope":    ["hope","optimism","progress","growth","achievement","milestone","breakthrough",
                "investment","develop","improve","success","launch","inaugurate","record",
                "boost","rise","expand","reform","initiative","recovery","revival","surge"],
    "pride":   ["pride","honor","celebrate","victory","historic","proud","remarkable","excellence",
                "awarded","recognition","heritage","culture","national","champion","gold","medal",
                "win","landmark","tribute","honour","distinguished","celebrated"],
    "fear":    ["threat","danger","attack","warn","terror","menace","risk","unsafe","incident",
                "killing","crime","conflict","border","infiltration","security","casualty",
                "explosion","shoot","rape","abduct","missing","hostage","military"],
}

# ============================================================================
# IN-MEMORY STORE
# ============================================================================

class Store:
    def __init__(self):
        self.signals: dict[str, list[dict]] = {s: [] for s in INDIAN_STATES}
        self.scores: dict[str, dict] = {}
        self.seen_urls: set[str] = set()
        self.last_ingest: datetime | None = None
        self.ingest_running: bool = False
        # 8-slot daily history per state (circular)
        self.history: dict[str, list[float]] = {s: [] for s in INDIAN_STATES}

    def add_signal(self, state: str, sig: dict) -> bool:
        """Returns True if the signal was new (not a dupe)."""
        url = sig.get("source_url", "")
        if url and url in self.seen_urls:
            return False
        if url:
            self.seen_urls.add(url)
        if state not in self.signals:
            self.signals[state] = []
        self.signals[state].append(sig)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        self.signals[state] = [
            s for s in self.signals[state][-600:]
            if s.get("published_at", datetime.now(timezone.utc)) > cutoff
        ]
        return True

    def snapshot_history(self):
        """Call once a day to push current scores into history."""
        for state in INDIAN_STATES:
            score = self.scores.get(state, {}).get("attention", 0)
            hist = self.history.setdefault(state, [])
            hist.append(round(score, 1))
            self.history[state] = hist[-8:]  # keep last 8 readings

store = Store()

# ============================================================================
# GEO-TAG
# ============================================================================

def geotag_states(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for state, aliases in STATE_ALIASES.items():
        if any(re.search(rf"\b{re.escape(a)}\b", lower) for a in aliases):
            found.append(state)
    return found or []

# ============================================================================
# CLASSIFICATION (keyword-based, fast, no API cost)
# ============================================================================

def classify_narratives(text: str) -> list[str]:
    lower = text.lower()
    scores: dict[str, int] = {}
    for narrative, keywords in NARRATIVE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            scores[narrative] = score
    return sorted(scores, key=lambda k: -scores[k])[:3]

def classify_emotion(text: str) -> dict[str, float]:
    """Returns {} if no emotion keywords found."""
    lower = text.lower()
    raw: dict[str, float] = {}
    for emo, words in EMOTION_KEYWORDS.items():
        score = sum(1 for w in words if w in lower)
        if score > 0:
            raw[emo] = float(score)
    if not raw:
        return {}
    total = sum(raw.values())
    return {k: round(v / total, 3) for k, v in raw.items()}

# ============================================================================
# AI SUMMARY (OpenAI GPT-4o-mini)
# ============================================================================

async def ai_summary(state: str, headlines: list[str], attention: float) -> str:
    """Generate a neutral 2-sentence AI intelligence summary for a state."""
    if not HAS_OPENAI or not headlines:
        return (
            f"{state} is registering an attention score of {attention:.0f}. "
            f"Signals are drawn from {len(headlines)} recent headlines across multiple sources."
        )
    bullets = "\n".join(f"- {h}" for h in headlines[:15])
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=120,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a neutral political intelligence analyst. "
                        "Describe what these headlines collectively reveal about the attention and narrative patterns "
                        "in the given Indian state. Be factual and observational. "
                        "Do not endorse any political position. Do not predict outcomes. "
                        "Write exactly 2 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": f"State: {state}\n\nRecent headlines:\n{bullets}",
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ai_summary] OpenAI error for {state}: {e}")
        return (
            f"{state} shows an attention score of {attention:.0f}, "
            f"driven by {len(headlines)} recent signals across key narrative categories."
        )

# ============================================================================
# RSS INGESTION
# ============================================================================

# State-specific RSS queries for better signal coverage
STATE_QUERIES: dict[str, list[str]] = {
    "Nagaland":          ["Nagaland Naga ceasefire Kohima", "Nagaland government NSCN"],
    "Mizoram":           ["Mizoram Aizawl governance", "Mizoram Myanmar refugee"],
    "Meghalaya":         ["Meghalaya Shillong NPP", "Meghalaya coal mining environment"],
    "Tripura":           ["Tripura Agartala BJP politics", "Tripura Bangladesh border"],
    "Manipur":           ["Manipur ethnic conflict Meitei Kuki", "Manipur Imphal violence"],
    "Arunachal Pradesh": ["Arunachal Pradesh China LAC border", "Arunachal Pradesh governance"],
    "Sikkim":            ["Sikkim Gangtok flood", "Sikkim SKM government"],
    "Assam":             ["Assam flood Brahmaputra", "Assam NRC Guwahati politics"],
    "Gujarat":           ["Gujarat economy investment Ahmedabad", "Gujarat BJP politics"],
    "Goa":               ["Goa tourism mining politics", "Goa governance Panaji"],
}

def build_rss_urls(state: str) -> list[str]:
    """Return 2-3 targeted RSS URLs for a state."""
    queries = STATE_QUERIES.get(state, [
        f"{state} politics government news",
        f"{state} latest news today",
    ])
    return [
        f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        for q in queries[:2]
    ]

def build_rss_url(state: str, lang_code: str) -> str:
    """Legacy single URL — used as fallback."""
    query = state.replace(" ", "+") + "+politics+news"
    return f"https://news.google.com/rss/search?q={query}&hl={lang_code}&gl=IN&ceid=IN:{lang_code.split('-')[0]}"

async def ingest_state(client: httpx.AsyncClient, state: str) -> int:
    added = 0
    urls_tried: set[str] = set()
    rss_urls = build_rss_urls(state)

    for url in rss_urls:
        if url in urls_tried:
            continue
        urls_tried.add(url)
        try:
            r = await client.get(url)
            feed = feedparser.parse(r.text)
        except Exception as e:
            print(f"[ingest] {state} fetch error: {e}")
            continue

        for entry in feed.entries[:12]:
            title = entry.get("title", "")
            body = entry.get("summary", "")
            source_url = entry.get("link", "")
            if not title:
                continue

            text = f"{title} {body}"
            tagged = geotag_states(text)
            # Also always add to the queried state even if not geo-matched
            tagged = list(set(tagged + [state]))

            try:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pub = datetime.now(timezone.utc)

            narratives = classify_narratives(text)
            emotions = classify_emotion(text)
            intensity = min(1.0, 0.4 + 0.2 * len(narratives))

            sig = {
                "source": entry.get("source", {}).get("title", "google_news"),
                "source_url": source_url,
                "title": title,
                "body": body[:400],
                "language": "en",
                "published_at": pub,
                "narratives": narratives,
                "emotions": emotions,
                "intensity": intensity,
            }
            for s in tagged:
                if s in INDIAN_STATES:
                    if store.add_signal(s, sig):
                        added += 1
    return added


async def run_ingest(states: list[str] | None = None) -> int:
    if store.ingest_running:
        print("[ingest] Already running, skipping.")
        return 0
    store.ingest_running = True
    states = states or INDIAN_STATES
    print(f"[ingest] Starting cycle — {len(states)} states — {datetime.now(timezone.utc).isoformat()}")
    total = 0
    try:
        async with httpx.AsyncClient(
            timeout=12,
            headers={"User-Agent": "IndiaAttentionMap/1.0 (research)"},
            follow_redirects=True,
        ) as client:
            sem = asyncio.Semaphore(10)
            async def with_sem(s):
                async with sem:
                    return await ingest_state(client, s)
            results = await asyncio.gather(*[with_sem(s) for s in states], return_exceptions=True)
            total = sum(r for r in results if isinstance(r, int))
    finally:
        store.ingest_running = False
    store.last_ingest = datetime.now(timezone.utc)
    print(f"[ingest] Done — {total} new signals")
    await recompute_all_scores()
    store.snapshot_history()
    return total


# ============================================================================
# SCORING ENGINE
# ============================================================================

async def recompute_state_score(state: str) -> dict:
    sigs = store.signals.get(state, [])
    now = datetime.now(timezone.utc)

    # Decay-weighted signal volume (half-life 36h)
    weighted = sum(
        2 ** (-(now - s["published_at"]).total_seconds() / 129_600) * s.get("intensity", 0.5)
        for s in sigs
    )
    attention = round(100 * math.tanh(weighted / 22), 1)

    # 24h windows for delta
    sigs_24h     = [s for s in sigs if (now - s["published_at"]).total_seconds() < 86_400]
    sigs_48_24h  = [s for s in sigs if 86_400 <= (now - s["published_at"]).total_seconds() < 172_800]
    delta_24h    = round(float(len(sigs_24h) - len(sigs_48_24h)), 1)
    velocity     = round(delta_24h / max(1, len(sigs_48_24h)), 3)

    # Narrative breakdown
    nar_counts: dict[str, int] = {}
    for s in sigs_24h:
        for n in s.get("narratives", []):
            nar_counts[n] = nar_counts.get(n, 0) + 1
    total_nar = max(1, sum(nar_counts.values()))
    top_narratives = sorted(nar_counts.items(), key=lambda kv: -kv[1])[:5]
    narrative_breakdown = [
        {"name": n, "val": round(c / total_nar * 100, 1), "dir": "up"}
        for n, c in top_narratives
    ]

    # Emotion breakdown — skip signals with no keyword matches
    emo_totals: dict[str, float] = {k: 0.0 for k in EMOTION_KEYWORDS}
    for s in sigs_24h:
        sig_emos = s.get("emotions", {})
        if not sig_emos:
            continue
        for k, v in sig_emos.items():
            if k in emo_totals:
                emo_totals[k] += v
    total_emo = sum(emo_totals.values())
    if total_emo > 0:
        emotions = {k: round(v / total_emo, 3) for k, v in emo_totals.items() if emo_totals[k] > 0}
    else:
        emotions = {}

    # Top articles (deduplicated by source)
    seen_src: set[str] = set()
    articles = []
    for s in sorted(sigs_24h, key=lambda x: x.get("intensity", 0), reverse=True):
        src = s.get("source", "unknown")
        if src in seen_src:
            continue
        seen_src.add(src)
        sig_emos = s.get("emotions", {})
        dom_emo = max(sig_emos.items(), key=lambda x: x[1])[0] if sig_emos else None
        articles.append({
            "src": src,
            "txt": s["title"],
            "url": s.get("source_url", "#"),
            "emotion": dom_emo,
            "narratives": s.get("narratives", [])[:2],
        })
        if len(articles) >= 12:
            break

    # Rising topics vs 48-24h window
    nar_prev: dict[str, int] = {}
    for s in sigs_48_24h:
        for n in s.get("narratives", []):
            nar_prev[n] = nar_prev.get(n, 0) + 1
    rising = []
    falling = []
    for n, c in top_narratives[:4]:
        prev = nar_prev.get(n, 0)
        if prev == 0:
            rising.append({"t": n, "pct": "new"})
        else:
            pct = round((c - prev) / prev * 100)
            if pct > 10:
                rising.append({"t": n, "pct": f"+{pct}%"})
            elif pct < -10:
                falling.append({"t": n, "pct": f"{pct}%"})

    # Historical timeline (pad with current if short)
    hist = list(store.history.get(state, []))
    while len(hist) < 7:
        hist.insert(0, max(0.0, attention - 5))
    hist.append(attention)
    hist = hist[-8:]

    # AI summary (async — run only if we have new articles)
    headlines = [s["title"] for s in sigs_24h[:15]]
    summary = await ai_summary(state, headlines, attention)

    dominant_emotion = max(emotions, key=lambda k: emotions[k]) if emotions else None
    dominant_narrative = top_narratives[0][0] if top_narratives else "governance"

    return {
        "name": state,
        "attention": attention,
        "delta_24h": delta_24h,
        "velocity": velocity,
        "dominant_emotion": dominant_emotion,
        "dominant_narrative": dominant_narrative,
        "emotions": emotions,
        "narratives": narrative_breakdown,
        "rising": rising,
        "falling": falling,
        "summary": summary,
        "articles": articles,
        "timeline": hist,
        "signal_count": len(sigs_24h),
    }


async def recompute_all_scores():
    tasks = {state: recompute_state_score(state) for state in INDIAN_STATES}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for state, result in zip(tasks.keys(), results):
        if isinstance(result, dict):
            store.scores[state] = result
        else:
            print(f"[score] Error for {state}: {result}")
    # Cache insights after scoring
    try:
        store.cache_insights = build_insights()
        print(f"[insights] Cached {len(store.cache_insights.get('rising',[]))} rising narratives")
    except Exception as e:
        print(f"[insights] Error: {e}")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="India Attention Map API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def continuous_ingest():
    """Run full ingest on startup then every 15 minutes."""
    while True:
        try:
            print(f"[continuous] Starting full ingest cycle")
            await run_ingest(states=INDIAN_STATES)
        except Exception as e:
            print(f"[continuous] Error: {e}")
        await asyncio.sleep(900)  # 15 minutes

@app.on_event("startup")
async def startup():
    print("[startup] Pulse of India backend starting...")
    asyncio.create_task(continuous_ingest())


# ── Serve frontend ──────────────────────────────────────────────────────────

# HTML served via base64 decode — avoids all string escaping issues
import base64 as _b64
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjE0cHggMjBweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MTRweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsIHJnYmEoMjI0LDkwLDQwLDAuMDYpIDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMTAwJSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoOHB4KTsKICBtYXgtd2lkdGg6OTAwcHg7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMjVlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjEwcHg7Cn0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNHB4LDEuNHZ3LDE4cHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTtsZXR0ZXItc3BhY2luZzotMC4wMWVtOwp9Ci5zaS10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWFjY2VudCl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5tYXAtY2FyZDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNzAlIDUwJSBhdCAzNSUgMCUsIHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCA4MCUgMTAwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDMpIDAlLCB0cmFuc3BhcmVudCA2MCUpOwp9Ci5tYXAtdG9wewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTJweCAxOHB4IDA7Cn0KLm1hcC10aXRsZS1ibG9jayAubXR7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxN3B4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoubWFwLXRpdGxlLWJsb2NrIC5tc3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA2ZW07bWFyZ2luLXRvcDoycHh9Ci5sZWdlbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1kaW0pfQoubGVnZW5kLWJhcnsKICBoZWlnaHQ6M3B4O3dpZHRoOjgwcHg7Ym9yZGVyLXJhZGl1czoycHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQsIzBlMjAzNSwjMWE1NTgwIDI1JSwjOGE1YzE4IDU1JSwjYzAzODFhIDgwJSwjZTAxMDIwKTsKfQoubGF5ZXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7CiAgcGFkZGluZzoxMHB4IDIwcHggNnB4Owp9Ci5sYXllci1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5sdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjNweH0KLmx0YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjNweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQoubHRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDgpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpfQoubHRhYntkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NXB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OnZpc2libGV9Ci5sdGFiLWluZm97d2lkdGg6MTNweDtoZWlnaHQ6MTNweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4yKTtmb250LXNpemU6OHB4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc3R5bGU6aXRhbGljO2ZvbnQtd2VpZ2h0OjYwMDtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuMzUpO2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y3Vyc29yOmhlbHA7ZmxleC1zaHJpbms6MDt0cmFuc2l0aW9uOmFsbCAwLjE1cztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEwMH0KLmx0YWItaW5mbzpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjZW50KTtjb2xvcjp2YXIoLS1hY2NlbnQpfQoubHRhYi1pbmZvOjphZnRlcntjb250ZW50OmF0dHIoZGF0YS10aXApO3Bvc2l0aW9uOmFic29sdXRlO2JvdHRvbTpjYWxjKDEwMCUgKyAxMHB4KTtsZWZ0OjA7d2lkdGg6MjMwcHg7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjtsZXR0ZXItc3BhY2luZzowO3RleHQtdHJhbnNmb3JtOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMnM7ei1pbmRleDoxMDAwMDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNSl9Ci5sdGFiLWluZm86aG92ZXI6OmFmdGVye29wYWNpdHk6MX0KLmx0YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KCi5tYXAtc3ZnLXdyYXB7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZzoxMnB4IDE2cHggMTZweDsKfQoubWFwLWlubmVye3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxLzE7d2lkdGg6MTAwJX0KI2luZGlhLW1hcHt3aWR0aDoxMDAlO2hlaWdodDoxMDAlO2Rpc3BsYXk6YmxvY2s7b3ZlcmZsb3c6dmlzaWJsZX0KCi8qIG1hcCBzdGF0ZSBzdHlsZXMgKi8KI2luZGlhLW1hcCAuc3RhdGV7CiAgY3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpmaWx0ZXIgMC4yNXMgZWFzZSwgc3Ryb2tlLXdpZHRoIDAuMnMgZWFzZSwgc3Ryb2tlIDAuMnMgZWFzZTsKfQojaW5kaWEtbWFwIC5zdGF0ZTpob3ZlcnsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjcpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjFweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMjUpIGRyb3Atc2hhZG93KDAgMCAxMHB4IHJnYmEoMjU1LDI1NSwyNTUsMC4yKSk7Cn0KI2luZGlhLW1hcCAuc3RhdGUuc2VsZWN0ZWR7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC45KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjRweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMzUpIGRyb3Atc2hhZG93KDAgMCAxNnB4IHJnYmEoMjU1LDI1NSwyNTUsMC4zKSk7Cn0KCi8qIGFuaW1hdGVkIHB1bHNlIHJpbmdzICovCi5wdWxzZS1yaW5ne2ZpbGw6bm9uZTtwb2ludGVyLWV2ZW50czpub25lfQoucHVsc2UtcmluZy5wMXthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCBpbmZpbml0ZX0KLnB1bHNlLXJpbmcucDJ7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgMC45cyBpbmZpbml0ZX0KQGtleWZyYW1lcyBwcnsKICAwJXtyOjQ7b3BhY2l0eTowLjc7c3Ryb2tlLXdpZHRoOjEuMn0KICAxMDAle3I6MjY7b3BhY2l0eTowO3N0cm9rZS13aWR0aDowLjJ9Cn0KCi8qIGF0bW9zcGhlcmljIGdsb3cgYmVoaW5kIGhvdCBzdGF0ZXMgKi8KLnN0YXRlLWdsb3d7cG9pbnRlci1ldmVudHM6bm9uZTtmaWxsOm5vbmV9CkBrZXlmcmFtZXMgZ2xvd1B1bHNlezAlLDEwMCV7b3BhY2l0eTowLjEyfTUwJXtvcGFjaXR5OjAuMjJ9fQoKLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDUsNywxMiwwLjk1KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6OXB4OwogIHBhZGRpbmc6MTJweCAxNHB4O29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xMnM7ei1pbmRleDoyMDttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmciAxZnI7Z2FwOjE4cHg7Cn0KLm5hci1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNHB4KTtvdmVyZmxvdzpoaWRkZW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubmMtdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO2NvbG9yOnZhcigtLWluayl9Ci5uYy1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA1ZW07bWFyZ2luLXRvcDoycHh9Ci5uYy1ib2R5e3BhZGRpbmc6MTNweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjB9CgoubW9tLWl0ewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweDsKICBwYWRkaW5nOjdweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLm1vbS1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoubW9tLXJre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO3dpZHRoOjEzcHg7ZmxleC1zaHJpbms6MH0KLm1vbS1pbmZ7ZmxleDoxfQoubW9tLW5te2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQoubW9tLXN0e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9Ci5tb20tcGN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtmb250LXdlaWdodDo0MDA7ZmxleC1zaHJpbms6MH0KLm1vbS1wYy5ye2NvbG9yOnZhcigtLXJpc2UpfQoubW9tLXBjLmZ7Y29sb3I6dmFyKC0tZmFsbCl9Ci5tb20tdHJ7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDttYXJnaW46M3B4IDAgMDtvdmVyZmxvdzpoaWRkZW59Ci5tb20tZmx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHh9CgoucmVnLWl0ewogIGRpc3BsYXk6ZmxleDtnYXA6OXB4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgcGFkZGluZzo4cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2N1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246b3BhY2l0eSAwLjE1czsKfQoucmVnLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5yZWctaXQ6aG92ZXJ7b3BhY2l0eTowLjc1fQoucmVnLWJhZGdlewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4wN2VtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTQpOwogIGNvbG9yOnZhcigtLWFjY2VudCk7ZmxleC1zaHJpbms6MDttYXJnaW4tdG9wOjJweDt3aGl0ZS1zcGFjZTpub3dyYXA7Cn0KLnJlZy1mbHtmbGV4OjE7Zm9udC1zaXplOjExLjVweDtsaW5lLWhlaWdodDoxLjV9Ci5yZWctZnJvbXtjb2xvcjp2YXIoLS1mYWludCl9Ci5yZWctYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7bWFyZ2luOjAgNHB4fQoucmVnLXRve2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQoucmVnLXRte2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7ZmxleC1zaHJpbms6MDttYXJnaW4tdG9wOjJweH0KCi8qIEZBVlMgKi8KLmZhdnN7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzZweCA0MHB4Owp9Ci5mYXZzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLmZhdnMtcm93e2Rpc3BsYXk6ZmxleDtnYXA6MTBweDtvdmVyZmxvdy14OmF1dG87cGFkZGluZy1ib3R0b206M3B4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFye2hlaWdodDoycHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXItdGh1bWJ7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjFweH0KLmZhdi1jYXJkewogIGZsZXg6MCAwIDE5MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTBweDtwYWRkaW5nOjEycHg7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7Cn0KLmZhdi1jYXJkOmhvdmVye2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIyKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDIpfQouZmMtaGVhZHtkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6YmFzZWxpbmU7bWFyZ2luLWJvdHRvbTo3cHh9Ci5mYy1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7Y29sb3I6dmFyKC0taW5rKX0KLmZjLXNje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLmZjLXJvd3tkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6M3B4fQouZmMtcm93IC52e2NvbG9yOnZhcigtLWRpbSk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4fQouZmF2cy1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTppdGFsaWM7cGFkZGluZzo0cHggMH0KCi8qIEZPT1QgKi8KLmZvb3R7dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzo0OHB4IDM2cHggNjBweDttYXgtd2lkdGg6NTgwcHg7bWFyZ2luOjAgYXV0bztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5mb290LW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW4tYm90dG9tOjE0cHh9Ci5mb290LWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEycHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjg7bWFyZ2luLWJvdHRvbToxMnB4fQouZm9vdC1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSl9CgovKiBhbmltYXRpb25zICovCkBrZXlmcmFtZXMgZmFkZVVwe2Zyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVZKDZweCl9dG97b3BhY2l0eToxO3RyYW5zZm9ybTpub25lfX0KLm1hcC1jYXJkLC5zdGF0ZS1wYW5lbCwubmFyLWNhcmQsLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbjpmYWRlVXAgMC41NXMgY3ViaWMtYmV6aWVyKC4yLC44LC4yLDEpIGJhY2t3YXJkc30KLm5hci1jYXJkOm50aC1jaGlsZCgyKXthbmltYXRpb24tZGVsYXk6MC4wN3N9Ci5uYXItY2FyZDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuMTRzfQouc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uLWRlbGF5OjAuMDVzfQoKQG1lZGlhKG1heC13aWR0aDoxMTAwcHgpewogIC5tYWlue2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgLnN0YXRlLXBhbmVse21heC1oZWlnaHQ6bm9uZX0KICAubmFyLXJvd3tncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQp9Cjwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+Cgo8ZGl2IGNsYXNzPSJ0b3BiYXIiPgogIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLW1hcmsiPjxzcGFuIGNsYXNzPSJicmFuZC1wdWxzZS1kb3QiPjwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLXRleHQtYmxvY2siPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtbmFtZSI+PGVtIGNsYXNzPSJicmFuZC1wdWxzZS13b3JkIj5QdWxzZTwvZW0+IG9mIEluZGlhPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtdGFnbGluZSI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3NwYW4+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJ0b3BiYXItciI+CiAgICA8ZGl2IGNsYXNzPSJsaXZlLWluZGljYXRvciI+CiAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gSEVSTyAtLT4KPHNlY3Rpb24gY2xhc3M9Imhlcm8iIHN0eWxlPSJwYWRkaW5nLXRvcDo4MHB4O3BhZGRpbmctYm90dG9tOjI0cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuIj4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTt3aWR0aDo2MDBweDtoZWlnaHQ6MzUwcHg7dG9wOi02MHB4O2xlZnQ6LTgwcHg7YmFja2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSBhdCA0MCUgNTAlLHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLHRyYW5zcGFyZW50IDY1JSk7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7YW5pbWF0aW9uOmFtYmllbnRTaGlmdCAxMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUgYWx0ZXJuYXRlIj48L2Rpdj4KICA8c3R5bGU+QGtleWZyYW1lcyBhbWJpZW50U2hpZnR7MCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMCl9MTAwJXt0cmFuc2Zvcm06dHJhbnNsYXRlWCgyNHB4KSB0cmFuc2xhdGVZKC0xMnB4KX19PC9zdHlsZT4KICA8ZGl2IGNsYXNzPSJoZXJvLWV5ZWJyb3ciIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPkNvbGxlY3RpdmUgYXR0ZW50aW9uICZtaWRkb3Q7IEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1icmFuZC1ibG9jayIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE4cHg7bWFyZ2luLWJvdHRvbToxNnB4O3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MSI+CiAgICA8ZGl2IGNsYXNzPSJoZXJvLXB1bHNlLXNpZ25hbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJocHMtY29yZSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMiI+PC9zcGFuPgogICAgPC9kaXY+CiAgICA8aDEgY2xhc3M9Imhlcm8tYnJhbmQtbmFtZSI+PGVtPlB1bHNlPC9lbT4gb2YgSW5kaWE8L2gxPgogIDwvZGl2PgogIDxwIGNsYXNzPSJoZXJvLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9wPgogIDxwIGNsYXNzPSJoZXJvLWRlc2MiPk9ic2VydmUgaG93IEluZGlhJ3MgbmFycmF0aXZlcyBhbmQgcHVibGljIGF0dGVudGlvbiBzaGlmdCBpbiByZWFsIHRpbWUuPC9wPgogIDxwIGNsYXNzPSJoZXJvLXN1Yi1saW5lIj5PYnNlcnZpbmcgSW5kaWEgaW4gbW90aW9uLjwvcD4KCiAgPCEtLSBMSVZFIFNUQVRTIFNUUklQIC0tPgo8ZGl2IGlkPSJzdGF0cy1zdHJpcCIgc3R5bGU9IgogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MjsKICBiYWNrZ3JvdW5kOnJnYmEoOSwxMywyMSwwLjkpOwogIGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4wOCk7CiAgcGFkZGluZzowIDM2cHg7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7CiI+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIiBpZD0ic2Mtc2lnbmFscyI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+U2lnbmFscyB0cmFja2VkPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1zaWduYWxzLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiPkxpdmUgaW5nZXN0aW9uPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLWhvdHRlc3QiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciIgb25jbGljaz0ic2VsZWN0SG90dGVzdCgpIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5IaWdoZXN0IGF0dGVudGlvbjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPkNsaWNrIHRvIGV4cGxvcmU8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1hbmdlci12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj5PdXRyYWdlICYgcHJvdGVzdCBzaWduYWxzPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIHJpc2luZyBuYXJyYXRpdmU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLW5hcnJhdGl2ZS12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtbmFycmF0aXZlLXN1YiI+TmF0aW9uYWwgc2lnbmFsIHN1cmdlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCBjb29saW5nPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29saW5nLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29saW5nLXN1YiI+U2lnbmFsIGRlY2F5PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPHN0eWxlPgouc3RhdC1jZWxsewogIGZsZXg6MTtwYWRkaW5nOjEwcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2FwOjJweDsKICB0cmFuc2l0aW9uOmJhY2tncm91bmQgMC4xNXM7Cn0KLnN0YXQtY2VsbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci5zdGF0LWRpdnt3aWR0aDoxcHg7YmFja2dyb3VuZDpyZ2JhKDE2MCwxOTAsMjMwLDAuMDcpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luOjhweCAwfQouc2MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO21hcmdpbi10b3A6MXB4fQouc2Mtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQo8L3N0eWxlPgoKCiAgPCEtLSBTSUdOQVRVUkUgSU5TSUdIVCArIE5BUlJBVElWRSBTVFJJUCBzaWRlIGJ5IHNpZGUgLS0+CiAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoxOHB4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7bWFyZ2luLXRvcDoxNnB4O21hcmdpbi1ib3R0b206MDttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbi1sZWZ0OmF1dG87bWFyZ2luLXJpZ2h0OmF1dG87cGFkZGluZzowIDM2cHg7Ij4KICAgIDxkaXYgY2xhc3M9InNpZ25hdHVyZS1pbnNpZ2h0IiBzdHlsZT0ibWFyZ2luLXRvcDowO2ZsZXg6MTttaW4td2lkdGg6MCI+CiAgICAgIDxkaXYgY2xhc3M9InNpLWxhYmVsIj5DdXJyZW50IG5hdGlvbmFsIG5hcnJhdGl2ZSBzaGlmdDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaS10ZXh0IiBpZD0ic2lnLWluc2lnaHQiPjxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHgiPk9ic2VydmluZyBzaWduYWxzLi4uPC9zcGFuPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaS1zdWIiIGlkPSJzaWctdGFncyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+TmFycmF0aXZlIHNoaWZ0czwvc3Bhbj4KICAgICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjJweDsiPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIGFjdGl2ZSIgZGF0YS1wZXJpb2Q9IjNtIj4zTTwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIiBkYXRhLXBlcmlvZD0iNm0iPjZNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSIxeSI+MVk8L2J1dHRvbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDwhLS0gc2hpZnRzIGxpc3QgLS0+CiAgICAgIDxkaXYgc3R5bGU9ImZsZXg6MTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3BhZGRpbmc6MTBweCAxNHB4O2dhcDo2cHg7IiBpZD0ic2hpZnQtbGlzdCI+PC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKCjwhLS0gTUFJTjogTUFQICsgU1RBVEUgUEFORUwgLS0+CjxkaXYgY2xhc3M9Im1haW4iPgoKICA8ZGl2IGNsYXNzPSJtYXAtY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJtYXAtdG9wIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLXRpdGxlLWJsb2NrIj4KICAgICAgICA8ZGl2IGNsYXNzPSJtdCI+SW5kaWEgJm1kYXNoOyBjb2xsZWN0aXZlIGF0dGVudGlvbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1zIiBpZD0ibWFwLW1ldGEiPjMwIHN0YXRlcyAmbWlkZG90OyBsaXZlIHNpZ25hbCBjb21wb3NpdGU8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImxlZ2VuZCI+PHNwYW4+cXVpZXQ8L3NwYW4+PGRpdiBjbGFzcz0ibGVnZW5kLWJhciI+PC9kaXY+PHNwYW4+YWN0aXZlPC9zcGFuPjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJsYXllci1yb3ciPgogICAgICA8c3BhbiBjbGFzcz0ibGF5ZXItbGFiZWwiPlZpZXc8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9Imx0YWJzIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiBhY3RpdmUiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJlbW90aW9uIj5FbW90aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJUaGUgZG9taW5hbnQgZW1vdGlvbmFsIHRvbmUg4oCUIGFueGlvdXMsIGFuZ3J5LCBob3BlZnVsLCBwcm91ZCBvciBmZWFyZnVsLiBSZXZlYWxzIHRoZSBwc3ljaG9sb2dpY2FsIHVuZGVyY3VycmVudCBvZiBwb2xpdGljYWwgYXR0ZW50aW9uLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9InZlbG9jaXR5Ij5Nb21lbnR1bSA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iSXMgYXR0ZW50aW9uIHJpc2luZyBvciBmYWxsaW5nPyBSaXNpbmcgPSBuYXJyYXRpdmUgYWNjZWxlcmF0aW5nLiBDb29saW5nID0gbG9zaW5nIHRyYWN0aW9uLiBTaG93cyBzdGF0ZXMgZW50ZXJpbmcgb3IgZXhpdGluZyBhIHBvbGl0aWNhbCBjeWNsZS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLXN2Zy13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLWlubmVyIj4KICAgICAgICA8c3ZnIGlkPSJpbmRpYS1tYXAiIHZpZXdCb3g9IjAgMCA4MDAgODAwIiBwcmVzZXJ2ZUFzcGVjdFJhdGlvPSJ4TWlkWU1pZCBtZWV0Ij4KICAgICAgICAgIDxkZWZzPgogICAgICAgICAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImFtYkdsb3ciIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjUwJSI+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0icmdiYSgyMjQsOTAsNDAsMC4wNCkiLz4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgICAgICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICAgICAgICAgIDxmaWx0ZXIgaWQ9InN0YXRlR2xvdyIgeD0iLTMwJSIgeT0iLTMwJSIgd2lkdGg9IjE2MCUiIGhlaWdodD0iMTYwJSI+CiAgICAgICAgICAgICAgPGZlR2F1c3NpYW5CbHVyIGluPSJTb3VyY2VHcmFwaGljIiBzdGREZXZpYXRpb249IjgiIHJlc3VsdD0iYmx1ciIvPgogICAgICAgICAgICAgIDxmZUNvbXBvc2l0ZSBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJibHVyIiBvcGVyYXRvcj0ib3ZlciIvPgogICAgICAgICAgICA8L2ZpbHRlcj4KICAgICAgICAgIDwvZGVmcz4KICAgICAgICAgIDxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJ1cmwoI2FtYkdsb3cpIi8+CiAgICAgICAgICA8ZyBpZD0ibWFwLWdsb3ciPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtc3RhdGVzIj48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXB1bHNlcyI+PC9nPgogICAgICAgIDwvc3ZnPgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC10b29sdGlwIiBpZD0idG9vbHRpcCI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gU1RBVEUgUEFORUwgLS0+CiAgPGRpdiBjbGFzcz0ic3RhdGUtcGFuZWwiIGlkPSJzdGF0ZS1kZXRhaWwiPgogICAgPGRpdiBjbGFzcz0icGFuZWwtZW1wdHkiPgogICAgICA8c3ZnIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxIj4KICAgICAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiA4djRNMTIgMTZoLjAxIi8+CiAgICAgIDwvc3ZnPgogICAgICA8ZGl2IGNsYXNzPSJwZS10Ij5TZWxlY3QgYSBzdGF0ZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJwZS1zIj5DbGljayBhbnkgcmVnaW9uIG9uIHRoZSBtYXA8YnIvPnRvIG9wZW4gaXRzIG5hcnJhdGl2ZSBwYW5lbC48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKPC9kaXY+Cgo8IS0tIE5BUlJBVElWRSBST1cgLS0+CjxkaXYgY2xhc3M9Im5hci1yb3ciPgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJpc2luZyBuYXJyYXRpdmVzPC9zcGFuPjwvZGl2PgogICAgPGRpdiBpZD0icmlzaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+PHNwYW4gY2xhc3M9Im5jLWRvdCBmYWxsIj48L3NwYW4+PHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9ImRlY2xpbmluZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJlZ2lvbmFsIHNoaWZ0czwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9InJlZ2lvbmFsLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSBSRVBMQVkgSU5ESUEgLS0+CjxzZWN0aW9uIGNsYXNzPSJyZXBsYXktc2VjdGlvbiI+CiAgPGRpdiBjbGFzcz0icmVwbGF5LWhlYWRlciI+CiAgICA8ZGl2PjxkaXYgY2xhc3M9InJlcGxheS1sYWJlbCI+UmVwbGF5IEluZGlhPC9kaXY+PGRpdiBjbGFzcz0icmVwbGF5LXN1YiI+V2F0Y2ggaG93IGNvbGxlY3RpdmUgYXR0ZW50aW9uIHNoaWZ0ZWQgb3ZlciB0aW1lPC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyZXBsYXktY29udHJvbHMiPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4gYWN0aXZlIiBkYXRhLXBlcmlvZD0iN2QiPjcgZGF5czwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4iIGRhdGEtcGVyaW9kPSIzMGQiPjMwIGRheXM8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIiBkYXRhLXBlcmlvZD0iNm0iPjYgbW9udGhzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biIgZGF0YS1wZXJpb2Q9ImVsZWN0aW9uIj5FbGVjdGlvbiAyMDI0PC9idXR0b24+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktc2NydWJiZXIiPgogICAgPGRpdiBjbGFzcz0icnAtdHJhY2siIGlkPSJycC10cmFjayI+PGRpdiBjbGFzcz0icnAtZmlsbCIgaWQ9InJwLWZpbGwiPjwvZGl2PjxkaXYgY2xhc3M9InJwLXRodW1iIiBpZD0icnAtdGh1bWIiPjwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0icnAtZGF0ZXMiIGlkPSJycC1kYXRlcyI+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0icmVwbGF5LXBsYXliYWNrIj4KICAgIDxidXR0b24gY2xhc3M9InJwLXBsYXkiIGlkPSJycC1wbGF5LWJ0biIgb25jbGljaz0idG9nZ2xlUmVwbGF5KCkiPgogICAgICA8c3ZnIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwb2x5Z29uIHBvaW50cz0iMiwxIDksNSAyLDkiIGlkPSJycC1wbGF5LWljb24iLz48L3N2Zz4KICAgIDwvYnV0dG9uPgogICAgPGRpdiBjbGFzcz0icnAtY3VycmVudC1kYXRlIiBpZD0icnAtY3VycmVudC1kYXRlIj5TZWxlY3QgYSBwZXJpb2QgYW5kIHByZXNzIHBsYXk8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJwLXNwZWVkIj48c3BhbiBjbGFzcz0icnAtc3BlZWQtbGFiZWwiPlNwZWVkPC9zcGFuPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQgYWN0aXZlIiBkYXRhLXNwZD0iMSI+MXg8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iMiI+Mng8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iNCI+NHg8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InJlcGxheS1zbmFwc2hvdCI+PGRpdiBjbGFzcz0icnAtc25hcC1sYWJlbCI+TmFycmF0aXZlIHNuYXBzaG90IGF0IHRoaXMgbW9tZW50PC9kaXY+PGRpdiBjbGFzcz0icnAtc25hcC1zdGF0ZXMiIGlkPSJycC1zbmFwLXN0YXRlcyI+PGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5QcmVzcyBwbGF5IHRvIG9ic2VydmUgSW5kaWEgaW4gbW90aW9uLjwvZGl2PjwvZGl2PjwvZGl2Pgo8L3NlY3Rpb24+CjxzdHlsZT4KLnJlcGxheS1zZWN0aW9ue3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggMzZweH0KLnJlcGxheS1oZWFkZXJ7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtZW5kO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206MjBweDtnYXA6MjBweDtmbGV4LXdyYXA6d3JhcH0KLnJlcGxheS1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIwcHg7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWluayk7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLnJlcGxheS1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjRweH0KLnJlcGxheS1jb250cm9sc3tkaXNwbGF5OmZsZXg7Z2FwOjRweDtmbGV4LXdyYXA6d3JhcH0KLnJwLWJ0bntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO3BhZGRpbmc6NXB4IDEycHg7Ym9yZGVyLXJhZGl1czo0cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Y29sb3I6dmFyKC0tZmFpbnQpO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQoucnAtYnRuLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5yZXBsYXktc2NydWJiZXJ7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MThweCAyMHB4IDE0cHg7bWFyZ2luLWJvdHRvbToxMnB4fQoucnAtdHJhY2t7cG9zaXRpb246cmVsYXRpdmU7aGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoycHg7Y3Vyc29yOnBvaW50ZXI7bWFyZ2luLWJvdHRvbToxMHB4fQoucnAtZmlsbHtwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MCU7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQscmdiYSgyMjQsOTAsNDAsMC40KSx2YXIoLS1hY2NlbnQpKTtib3JkZXItcmFkaXVzOjJweH0KLnJwLXRodW1ie3Bvc2l0aW9uOmFic29sdXRlO3RvcDo1MCU7dHJhbnNmb3JtOnRyYW5zbGF0ZSgtNTAlLC01MCUpO3dpZHRoOjEycHg7aGVpZ2h0OjEycHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JvcmRlcjoycHggc29saWQgcmdiYSg5LDEzLDIxLDAuOCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCk7bGVmdDowJTtjdXJzb3I6Z3JhYn0KLnJwLWRhdGVze2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJlcGxheS1wbGF5YmFja3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNHB4O21hcmdpbi1ib3R0b206MTZweH0KLnJwLXBsYXl7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjI1KTtjb2xvcjp2YXIoLS1hY2NlbnQpO2N1cnNvcjpwb2ludGVyO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE1c30KLnJwLWN1cnJlbnQtZGF0ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO2ZsZXg6MX0KLnJwLXNwZWVke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweH0KLnJwLXNwZWVkLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1yaWdodDoycHh9Ci5ycC1zcGR7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O3BhZGRpbmc6M3B4IDhweDtib3JkZXItcmFkaXVzOjNweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDp0cmFuc3BhcmVudDtjb2xvcjp2YXIoLS1mYWludCk7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xNXN9Ci5ycC1zcGQuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIpfQoucmVwbGF5LXNuYXBzaG90e2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTJweDtwYWRkaW5nOjE2cHggMjBweH0KLnJwLXNuYXAtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEycHh9Ci5ycC1zbmFwLXN0YXRlc3tkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjhweH0KLnJwLWxvZy1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSk7Zm9udC1zdHlsZTppdGFsaWM7cGFkZGluZzo0cHggMH0KLnJwLXN0YXRlLWNhcmR7cGFkZGluZzo4cHggMTJweDtib3JkZXItcmFkaXVzOjZweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO21pbi13aWR0aDoxNDBweH0KLnJwLXN0YXRlLW5hbWV7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bWFyZ2luLWJvdHRvbTozcHh9Ci5ycC1zdGF0ZS1uYXJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJwLXN0YXRlLWF0dHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCl9Cjwvc3R5bGU+CjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIE9ic2VydmF0aW9uLjwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICBpZighcm93c3x8IXJvd3MubGVuZ3RoKSByZXR1cm47CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocm93LmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb21FbW89cm93LmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihlbW9zKXx8bnVsbDsKICAgICAgdmFyIGVudHJ5PXthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbUVtbyxkb21pbmFudF9uYXJyYXRpdmU6cm93LmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgICAgTElWRVtyb3cubmFtZV09ZW50cnk7CiAgICAgIGlmKCFTRFtyb3cubmFtZV0pIFNEW3Jvdy5uYW1lXT1PYmplY3QuYXNzaWduKHt9LERFRkFVTFQpOwogICAgICBPYmplY3QuYXNzaWduKFNEW3Jvdy5uYW1lXSxlbnRyeSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgLy8gUmUtcmVuZGVyIG1vbWVudHVtIGFmdGVyIHNob3J0IGRlbGF5IHRvIGVuc3VyZSBTRCBpcyBzdGFibGUKICAgIHNldFRpbWVvdXQocmVuZGVyTW9tZW50dW0sIDUwMCk7CiAgICBpZihTRUwmJkxJVkVbU0VMXSYmZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpKSByZW5kZXJQYW5lbChTRUwpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tBUEldJyxlLm1lc3NhZ2UpO30KfQoKZnVuY3Rpb24gdXBkYXRlQWxsU3RyaXBzKCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CiAgdmFyIGhvdHRlc3Q9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKT4oYVsxXS5hdHRlbnRpb258fDApP2I6YTt9LGVudHJpZXNbMF0pOwogIHNldFRleHQoJ3NjLWhvdHRlc3QtdmFsJyxob3R0ZXN0WzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXN1YicsJ0F0dGVudGlvbiAnK2hvdHRlc3RbMV0uYXR0ZW50aW9uLnRvRml4ZWQoMSkpOwogIHZhciB0b3BBbmdlck5tPW51bGwsdG9wQW5nZXJQY3Q9MDsKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgdmFyIGU9a3ZbMV0uZW1vdGlvbnN8fHt9OwogICAgdmFyIGE9ZS5hbmdlcnx8MDsKICAgIGlmKGE+MCYmYTw9MSkgYT1NYXRoLnJvdW5kKGEqMTAwKTsKICAgIGlmKGE+dG9wQW5nZXJQY3Qpe3RvcEFuZ2VyUGN0PWE7dG9wQW5nZXJObT1rdlswXTt9CiAgfSk7CiAgaWYodG9wQW5nZXJObSYmdG9wQW5nZXJQY3Q+MCl7CiAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLHRvcEFuZ2VyTm0pOwogICAgc2V0VGV4dCgnc2MtYW5nZXItc3ViJywnQW5nZXIgJytNYXRoLnJvdW5kKHRvcEFuZ2VyUGN0KSsnJSBvZiBzaWduYWxzJyk7CiAgfSBlbHNlIHsKICAgIC8vIEZhbGwgYmFjayB0byBkb21pbmFudF9lbW90aW9uPWFuZ2VyCiAgICB2YXIgYW5nZXJEb209ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5kb21pbmFudF9lbW90aW9uPT09J2FuZ2VyJzt9KTsKICAgIGlmKGFuZ2VyRG9tLmxlbmd0aCl7CiAgICAgIHZhciB0b3BCeUF0dD1hbmdlckRvbS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICAgICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BCeUF0dFswXSk7CiAgICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsJ0RvbWluYW50IGVtb3Rpb246IGFuZ2VyJyk7CiAgICB9CiAgfQogIHZhciBjb29saW5nPWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLnZlbG9jaXR5fHwwKTwoYVsxXS52ZWxvY2l0eXx8MCk/YjphO30sZW50cmllc1swXSk7CiAgc2V0VGV4dCgnc2MtY29vbGluZy12YWwnLGNvb2xpbmdbMF0pO3NldFRleHQoJ3NjLWNvb2xpbmctc3ViJywnVmVsb2NpdHkgJytjb29saW5nWzFdLnZlbG9jaXR5LnRvRml4ZWQoMykpOwogIHZhciBuYz17fTtlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSluY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdPShuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdfHwwKSsxO30pOwogIHZhciB0bj1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgaWYodG4pe3NldFRleHQoJ3NjLW5hcnJhdGl2ZS12YWwnLHRuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3RuWzBdLnNsaWNlKDEpKTtzZXRUZXh0KCdzYy1uYXJyYXRpdmUtc3ViJywnRG9taW5hbnQgYWNyb3NzICcrdG5bMV0rJyBzdGF0ZXMnKTt9Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUvJytlbmNvZGVVUklDb21wb25lbnQobmFtZSkpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMoZC5lbW90aW9uc3x8e30pOwogICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgIFNEW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZSwKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSxzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OmQubmFycmF0aXZlSGlzdG9yeXx8REVGQVVMVC5uYXJyYXRpdmVIaXN0b3J5LHNpZ25hbF9jb3VudDpkLnNpZ25hbF9jb3VudHx8MH07CiAgICBpZighTElWRVtuYW1lXSlMSVZFW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmV9OwogICAgTElWRVtuYW1lXS5lbW90aW9ucz1lbW9zO0xJVkVbbmFtZV0uZG9taW5hbnRfZW1vdGlvbj1kb207CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbmFtZSxlLm1lc3NhZ2UpO3JldHVybiBTRFtuYW1lXXx8REVGQVVMVDt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICAvLyB0b3BiYXIKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGl2ZS1jb3VudCcpOwogICAgaWYoZWwmJmQudG90YWxfc2lnbmFscykgZWwudGV4dENvbnRlbnQ9ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk7CiAgICB2YXIgbWV0YT1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLW1ldGEnKTsKICAgIGlmKG1ldGEmJmQuYXNfb2YpIG1ldGEudGV4dENvbnRlbnQ9JzMwIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoZC5hc19vZikudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicpOwogICAgLy8gc3RhdHMgc3RyaXAKICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJywgZC50b3RhbF9zaWduYWxzP2QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOictJyk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICB9Y2F0Y2goZSl7fQp9CgpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZihlbCllbC50ZXh0Q29udGVudD12YWw7fQoKZnVuY3Rpb24gdXBkYXRlU3RyaXBOYXJyYXRpdmUoKXt1cGRhdGVBbGxTdHJpcHMoKTt9CmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXt9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hJbnNpZ2h0cygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2luc2lnaHRzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICB2YXIgc2lnPWQuc2lnbmF0dXJlOwogICAgaWYoc2lnKXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9IkluZGlhJ3MgYXR0ZW50aW9uIGlzIHNoaWZ0aW5nIGZyb20gPGVtPiIrc2lnLmZhZGluZysiPC9lbT4gdG93YXJkIDxlbT4iK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFuZCA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiLiAiK3NpZy5ob3R0ZXN0X3N0YXRlKyIgbGVhZHMgbmF0aW9uYWwgYXR0ZW50aW9uLiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC50YWdzKXRFbC5pbm5lckhUTUw9ZC50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgfQogICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgIGlmKHJFbCYmZC5yaXNpbmcmJmQucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQucmlzaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytuLnN0YXRlcy5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKSsnJTtiYWNrZ3JvdW5kOiNlMDVhMjgiPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgaWYoZkVsJiZkLmZhbGxpbmcmJmQuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6IzNiYjhkOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgIGlmKGdFbCYmZC5yZWdpb25hbCYmZC5yZWdpb25hbC5sZW5ndGgpZ0VsLmlubmVySFRNTD1kLnJlZ2lvbmFsLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW4iPjxzcGFuIGNsYXNzPSJuaS1uYW1lIj4nK3IucmVnaW9uKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+JytyLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PjxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytyLmhvdHRlc3Rfc3RhdGUrJyDCtyAnK3IudG9wX25hcnJhdGl2ZSsnPC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlKTt9KTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDEyMDAwMCk7CiAgICAgIHNldEludGVydmFsKGZldGNoSW5zaWdodHMsMzYwMDAwMCk7fQogIH0sMTUwMDApOwp9CgovLyBOQVJSQVRJVkUgREFUQQp2YXIgU0hJRlRTPXsKICAnM20nOlsKICAgIHtmYWRpbmc6J0luZmxhdGlvbicsZmFkaW5nTm90ZTonZWFzaW5nIG5hdGlvbmFsbHknLHJpc2luZzonQm9yZGVyIHNlY3VyaXR5JyxyaXNpbmdOb3RlOidwb3N0LWluY2lkZW50IHN1cmdlJ30sCiAgICB7ZmFkaW5nOidFbGVjdGlvbiByaGV0b3JpYycsZmFkaW5nTm90ZToncG9zdC1jeWNsZSBmYWRlJyxyaXNpbmc6J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3N0ZWFkeSByaXNlJ30sCiAgICB7ZmFkaW5nOidGYXJtZXIgcHJvdGVzdHMnLGZhZGluZ05vdGU6J21vbWVudHVtIGxvc3QnLHJpc2luZzonVW5lbXBsb3ltZW50IGFueGlldHknLHJpc2luZ05vdGU6J3lvdXRoIHNpZ25hbCBzdXJnZSd9LAogIF0sCiAgJzZtJzpbCiAgICB7ZmFkaW5nOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZhZGluZ05vdGU6J3ByZS1lbGVjdGlvbiBwZWFrJyxyaXNpbmc6J0NvcnJ1cHRpb24gYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3Bvc3QtY3ljbGUgcHVzaCd9LAogICAge2ZhZGluZzonUmVsaWdpb3VzIG5hdGlvbmFsaXNtJyxmYWRpbmdOb3RlOidwbGF0ZWF1IHBoYXNlJyxyaXNpbmc6J0Vjb25vbWljIGFueGlldHknLHJpc2luZ05vdGU6J2Nvc3Qtb2YtbGl2aW5nJ30sCiAgICB7ZmFkaW5nOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZmFkaW5nTm90ZToncmliYm9uLWN1dHRpbmcgZG9uZScscmlzaW5nOidMYXcgJiBvcmRlcicscmlzaW5nTm90ZTonY3JpbWUgbmFycmF0aXZlIHJpc2UnfSwKICBdLAogICcxeSc6WwogICAge2ZhZGluZzonUGFuZGVtaWMgcmVjb3ZlcnknLGZhZGluZ05vdGU6J2ZhZGVkIGVhcmx5IHllYXInLHJpc2luZzonSW5mbGF0aW9uJyxyaXNpbmdOb3RlOidkb21pbmF0ZWQgbWlkLXllYXInfSwKICAgIHtmYWRpbmc6J1JlZ2lvbmFsIGlkZW50aXR5JyxmYWRpbmdOb3RlOidsYW5ndWFnZS1sZWQgcGVhaycscmlzaW5nOidTZWN1cml0eSAmIGJvcmRlcnMnLHJpc2luZ05vdGU6J2dlb3BvbGl0aWNhbCBlc2NhbGF0aW9uJ30sCiAgICB7ZmFkaW5nOidHb3Zlcm5hbmNlIG9wdGltaXNtJyxmYWRpbmdOb3RlOidwb2xpY3kgaG9uZXltb29uIGVuZCcscmlzaW5nOidDb3JydXB0aW9uICYgc2NhbXMnLHJpc2luZ05vdGU6J2FjY291bnRhYmlsaXR5IGN5Y2xlJ30sCiAgXSwKfTsKdmFyIFJFR19TSElGVFM9WwogIHtzdGF0ZTonVGFtaWwgTmFkdScsZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLHRvOidGZWRlcmFsIHJlc291cmNlIGRpc3B1dGVzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonQmloYXInLGZyb206J0VsZWN0aW9uIHJoZXRvcmljJyx0bzonVW5lbXBsb3ltZW50ICYgZXhhbSBzY2FtcycsdGltZTonNiB3a3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2tzJ30sCiAge3N0YXRlOidSYWphc3RoYW4nLGZyb206J0Zhcm1lciBwcm90ZXN0cycsdG86J0hlYXQgd2F2ZSDCtyBFbnZpcm9ubWVudCcsdGltZTonMiB3a3MnfSwKICB7c3RhdGU6J0thcm5hdGFrYScsZnJvbTonTWluaW5nIGNvbnRyb3ZlcnN5Jyx0bzonTGFuZ3VhZ2Ugc2lnbmFnZSBwb2xpdGljcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyBpbmZyYXN0cnVjdHVyZScsdG86J0FpciBxdWFsaXR5IGNyaXNpcycsdGltZTonMTAgZGF5cyd9LAogIHtzdGF0ZTonTWFuaXB1cicsZnJvbTonR292ZXJuYW5jZSAmIGNhYmluZXQnLHRvOidFdGhuaWMgdGVuc2lvbnMgwrcgQUZTUEEnLHRpbWU6JzUgd2tzJ30sCiAge3N0YXRlOidQdW5qYWInLGZyb206J1Bvd2VyIGNyaXNpcycsdG86J0JvcmRlciBzZWN1cml0eSDCtyBEcm9uZXMnLHRpbWU6JzMgd2tzJ30sCl07CnZhciBNT0NLX1I9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbiBwcmVzc3VyZScsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBkYXRhPVNISUZUU1twZXJpb2RdfHxTSElGVFNbJzNtJ107CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MPWRhdGEubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDowO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo4cHg7b3ZlcmZsb3c6aGlkZGVuOyI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjZweCAxMHB4O2JvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhbGwpO21hcmdpbi1ib3R0b206M3B4OyI+ZmFkaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLmZhZGluZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5mYWRpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9IndpZHRoOjI4cHg7ZmxleC1zaHJpbms6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNDU7Zm9udC1zaXplOjEzcHg7Ij7ihpI8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6OHB4IDEwcHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1yaXNlKTttYXJnaW4tYm90dG9tOjNweDsiPnJpc2luZzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuMjsiPicrcy5yaXNpbmcrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK3MucmlzaW5nTm90ZSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0YWIpewogIHRhYi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHQpe3QuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICB0YWIuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVuZGVyU3RyaXAodGFiLmRhdGFzZXQucGVyaW9kKTsKICB9KTsKfSk7CgpmdW5jdGlvbiByZW5kZXJNb21lbnR1bSgpewogIC8vIFJlYWQgZnJvbSBTRCAocG9wdWxhdGVkIGJ5IGZldGNoQWxsU3RhdGVzIGZyb20gbGl2ZSBBUEkpCiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgbmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpOwogIHZhciBmYWxsaW5nPXNvcnRlZC5zbGljZSgtNSkucmV2ZXJzZSgpOwogIHZhciBteD1yaXNpbmcubGVuZ3RoP3Jpc2luZ1swXVsxXToxMDA7CgogIC8vIFdyaXRlIHRvIHJpc2luZy1saXN0IChtYXRjaGVzIG5hci1yb3cgSFRNTCkKICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogIGlmKHJFbCYmcmlzaW5nLmxlbmd0aCl7CiAgICByRWwuaW5uZXJIVE1MPXJpc2luZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im5pLW5hbWUiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKSsnJTtiYWNrZ3JvdW5kOiNlMDVhMjgiPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byBkZWNsaW5pbmctbGlzdAogIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgaWYoZkVsJiZmYWxsaW5nLmxlbmd0aCl7CiAgICBmRWwuaW5uZXJIVE1MPWZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuWzFdL214KjEwMCkrJyU7YmFja2dyb3VuZDojM2JiOGQ4Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gcmVnaW9uYWwtbGlzdCDigJQgdG9wIHN0YXRlIHBlciByZWdpb24gZnJvbSBMSVZFCiAgdmFyIHJlZ2lvbnM9ewogICAgJ05vcnRoJzpbJ0RlbGhpJywnVXR0YXIgUHJhZGVzaCcsJ1B1bmphYicsJ0hhcnlhbmEnLCdIaW1hY2hhbCBQcmFkZXNoJywnVXR0YXJha2hhbmQnLCdKYW1tdSBhbmQgS2FzaG1pciddLAogICAgJ0Vhc3QnOlsnV2VzdCBCZW5nYWwnLCdCaWhhcicsJ0poYXJraGFuZCcsJ09kaXNoYSddLAogICAgJ1dlc3QnOlsnTWFoYXJhc2h0cmEnLCdHdWphcmF0JywnUmFqYXN0aGFuJywnR29hJ10sCiAgICAnU291dGgnOlsnVGFtaWwgTmFkdScsJ0thcm5hdGFrYScsJ0tlcmFsYScsJ0FuZGhyYSBQcmFkZXNoJywnVGVsYW5nYW5hJ10sCiAgICAnTkUnOlsnQXNzYW0nLCdNYW5pcHVyJywnTmFnYWxhbmQnLCdNaXpvcmFtJywnTWVnaGFsYXlhJywnVHJpcHVyYScsJ0FydW5hY2hhbCBQcmFkZXNoJywnU2lra2ltJ10sCiAgICAnQ2VudHJhbCc6WydNYWRoeWEgUHJhZGVzaCcsJ0NoaGF0dGlzZ2FyaCddLAogIH07CiAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogIGlmKGdFbCl7CiAgICB2YXIgcmVnSXRlbXM9T2JqZWN0LmVudHJpZXMocmVnaW9ucykubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgdmFyIHJlZ2lvbj1rdlswXSxzdGF0ZXM9a3ZbMV07CiAgICAgIHZhciB0b3A9c3RhdGVzLm1hcChmdW5jdGlvbihzKXtyZXR1cm4ge25hbWU6cyxhdHQ6KExJVkVbc10mJkxJVkVbc10uYXR0ZW50aW9uKXx8MH07fSkKICAgICAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiLmF0dC1hLmF0dDt9KVswXTsKICAgICAgaWYoIXRvcHx8IXRvcC5hdHQpIHJldHVybiBudWxsOwogICAgICB2YXIgbmFyPShMSVZFW3RvcC5uYW1lXSYmTElWRVt0b3AubmFtZV0uZG9taW5hbnRfbmFycmF0aXZlKXx8J+KAlCc7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzo4cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjEyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+JytyZWdpb24rJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3RvcC5hdHQudG9GaXhlZCgxKSsnPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+Jyt0b3AubmFtZSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHg7Ij4nK25hcisnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5maWx0ZXIoQm9vbGVhbikuam9pbignJyk7CiAgICBpZihyZWdJdGVtcykgZ0VsLmlubmVySFRNTD1yZWdJdGVtczsKICB9Cn0KCgovLyBTVEFURSBEQVRBCnZhciBTRD17fTsKCnZhciBMSVZFPXt9OwpmdW5jdGlvbiBub3JtYWxpemVFbW90aW9ucyhlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm57fTt2YXIgdmFscz1PYmplY3QudmFsdWVzKGUpLHRvdD12YWxzLnJlZHVjZShmdW5jdGlvbihzLHYpe3JldHVybiBzK3Y7fSwwKTtpZih0b3Q8PTApcmV0dXJue307aWYodG90PD0xLjAxKXt2YXIgb3V0PXt9O09iamVjdC5rZXlzKGUpLmZvckVhY2goZnVuY3Rpb24oayl7b3V0W2tdPU1hdGgucm91bmQoZVtrXSoxMDApO30pO3JldHVybiBvdXQ7fXJldHVybiBlO30KZnVuY3Rpb24gZG9taW5hbnRFbW90aW9uKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybiBudWxsO3ZhciBteD0wLGRvbT1udWxsO09iamVjdC5lbnRyaWVzKGUpLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdPm14KXtteD1rdlsxXTtkb209a3ZbMF07fX0pO3JldHVybiBkb207fQpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZighZWwpcmV0dXJuO2VsLnRleHRDb250ZW50PXZhbDtpZih2YWwmJnZhbCE9PSctJyl7ZWwuY2xhc3NMaXN0LnJlbW92ZSgnbG9hZGluZycpO319Cgp2YXIgREVGQVVMVD17CiAgYXR0ZW50aW9uOjAsZGVsdGE6MCx2ZWxvY2l0eTowLAogIGVtb3Rpb25zOnt9LGRvbWluYW50X2Vtb3Rpb246bnVsbCxkb21pbmFudF9uYXJyYXRpdmU6bnVsbCwKICBuYXJyYXRpdmVzOltdLHJpc2luZzpbXSxmYWxsaW5nOltdLAogIHN1bW1hcnk6JycsYXJ0aWNsZXM6W10sdGltZWxpbmU6W10sCiAgbmFycmF0aXZlSGlzdG9yeTpbXSxzaWduYWxfY291bnQ6MCwKfTsKCmZ1bmN0aW9uIGcobil7cmV0dXJuIFNEW25dfHxPYmplY3QuYXNzaWduKHt9LERFRkFVTFQpO30KCmZ1bmN0aW9uIGFDKHMpewogIC8vIER5bmFtaWMgc2NhbGU6IGFsd2F5cyBzcHJlYWQgZnVsbCBjb2xvciByYW5nZSBhY3Jvc3MgYWN0dWFsIGRhdGEKICAvLyBHZXQgbWluL21heCBmcm9tIGN1cnJlbnQgU0QgdG8gbm9ybWFsaXplCiAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oZCl7cmV0dXJuIGQuYXR0ZW50aW9ufHwwO30pOwogIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3Jlcyk7CiAgdmFyIG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAvLyBOb3JtYWxpemUgMC0xCiAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzLW1uKS8obXgtbW4pKSk7CiAgLy8gTWFwIHRvIGNvbG9yIHN0b3BzOiBkYXJrIGJsdWUg4oaSIHRlYWwg4oaSIGFtYmVyIOKGkiBvcmFuZ2Ug4oaSIHJlZAogIGlmKG48MC4xMikgcmV0dXJuICcjMGQxZTMwJzsKICBpZihuPDAuMjUpIHJldHVybiAnIzBlM2Q2YSc7CiAgaWYobjwwLjM4KSByZXR1cm4gJyMwZDVmOTAnOwogIGlmKG48MC41MCkgcmV0dXJuICcjMGU3YWFhJzsKICBpZihuPDAuNjIpIHJldHVybiAnIzFhOTA5MCc7CiAgaWYobjwwLjcyKSByZXR1cm4gJyNjODcwMTAnOwogIGlmKG48MC44MikgcmV0dXJuICcjZDg0MDEwJzsKICBpZihuPDAuOTIpIHJldHVybiAnI2NjMTgwOCc7CiAgcmV0dXJuICcjZmYwMDEwJzsKfQpmdW5jdGlvbiBlQyhlKXsKICB2YXIgbXg9MCxkb209J3ByaWRlJzsKICBmb3IodmFyIGsgaW4gZSl7aWYoZVtrXT5teCl7bXg9ZVtrXTtkb209azt9fQogIHJldHVybiAoe2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9KVtkb21dfHwnIzMzYWFjYyc7Cn0KZnVuY3Rpb24gdkModil7CiAgaWYodj4wLjIpIHJldHVybiAnI2RjMDgxOCc7CiAgaWYodj4wLjEpIHJldHVybiAnI2UwNWEyOCc7CiAgaWYodj4wLjAyKSByZXR1cm4gJyNjYzg4MjInOwogIGlmKHY8LTAuMDUpIHJldHVybiAnIzIyOTliYic7CiAgcmV0dXJuICcjMTUyMDMwJzsKfQoKdmFyIGxheWVyPSdhdHRlbnRpb24nLFNFTD1udWxsLEZBVlM9bmV3IFNldCgpOwoKLy8gTUFQCmZ1bmN0aW9uIHByb2pfKHcsaCxwYWQpewogIHBhZD1wYWR8fDIwOwogIHZhciBtaW5Mb249NjguMSxtYXhMb249OTcuNCxtaW5MYXQ9Ni41LG1heExhdD0zNy4xOwogIHZhciBzY1g9KHctcGFkKjIpLyhtYXhMb24tbWluTG9uKTsKICB2YXIgc2NZPShoLXBhZCoyKS8obWF4TGF0LW1pbkxhdCk7CiAgdmFyIHNjPU1hdGgubWluKHNjWCxzY1kpOwogIHZhciBveD1wYWQrKHctcGFkKjItKG1heExvbi1taW5Mb24pKnNjKS8yOwogIHZhciBveT1wYWQrKGgtcGFkKjItKG1heExhdC1taW5MYXQpKnNjKS8yOwogIHJldHVybiBmdW5jdGlvbihsb24sbGF0KXtyZXR1cm4gW294Kyhsb24tbWluTG9uKSpzYywgb3krKG1heExhdC1sYXQpKnNjXTt9Owp9CmZ1bmN0aW9uIGdlbzJwYXRoKGdlb20scGopewogIHZhciBkPScnOwogIGZ1bmN0aW9uIHJpbmcoY3Mpe3ZhciBzPScnO2NzLmZvckVhY2goZnVuY3Rpb24oYyxpKXt2YXIgcD1waihjWzBdLGNbMV0pO3MrPShpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KTtyZXR1cm4gcysnWic7fQogIGlmKGdlb20udHlwZT09PSdQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7CiAgZWxzZSBpZihnZW9tLnR5cGU9PT0nTXVsdGlQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHApe3AuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pO30pOwogIHJldHVybiBkOwp9CmZ1bmN0aW9uIGN0cihnZW9tKXsKICB2YXIgcHRzPVtdOwogIGZ1bmN0aW9uIGNvbChjKXtpZih0eXBlb2YgY1swXT09PSdudW1iZXInKSBwdHMucHVzaChjKTtlbHNlIGMuZm9yRWFjaChjb2wpO30KICBjb2woZ2VvbS5jb29yZGluYXRlcyk7CiAgaWYoIXB0cy5sZW5ndGgpIHJldHVybiBbMCwwXTsKICByZXR1cm4gW3B0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzBdO30sMCkvcHRzLmxlbmd0aCxwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFsxXTt9LDApL3B0cy5sZW5ndGhdOwp9CmZ1bmN0aW9uIHNOYW1lKHByb3BzKXsKICB2YXIgcmF3PXByb3BzLnN0X25tfHxwcm9wcy5OQU1FXzF8fHByb3BzLm5hbWV8fHByb3BzLk5BTUV8fCcnOwogIHZhciBtYXA9eydMYWRha2gnOidKYW1tdSBhbmQgS2FzaG1pcicsJ0phbW11ICYgS2FzaG1pcic6J0phbW11IGFuZCBLYXNobWlyJywnVXR0YXJhbmNoYWwnOidVdHRhcmFraGFuZCcsJ0FuZGFtYW4gYW5kIE5pY29iYXInOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdBbmRhbWFuICYgTmljb2JhciBJc2xhbmQnOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdOQ1Qgb2YgRGVsaGknOidEZWxoaScsJ1BvbmRpY2hlcnJ5JzonUHVkdWNoZXJyeScsJ0RhZHJhIGFuZCBOYWdhciBIYXZlbGknOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1JywnRGFtYW4gYW5kIERpdSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnfTsKICByZXR1cm4gbWFwW3Jhd118fHJhdzsKfQoKdmFyIGNhY2hlZEdlbz1udWxsOwoKYXN5bmMgZnVuY3Rpb24gbG9hZE1hcChhdHRlbXB0KXsKICBhdHRlbXB0ID0gYXR0ZW1wdHx8MTsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaCgnaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L2doL3VkaXQtMDAxL2luZGlhLW1hcHMtZGF0YUBtYXN0ZXIvdG9wb2pzb24vaW5kaWEuanNvbicpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciB0b3BvPWF3YWl0IHIuanNvbigpOwogICAgY2FjaGVkR2VvPXRvcG9qc29uLmZlYXR1cmUodG9wbyx0b3BvLm9iamVjdHMuc3RhdGVzKTsKICAgIHJlbmRlck1hcChjYWNoZWRHZW8pOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDEwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDMwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDYwMDApOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW21hcF0gbG9hZCBmYWlsZWQgYXR0ZW1wdCAnK2F0dGVtcHQrJzonLGUubWVzc2FnZSk7CiAgICBpZihhdHRlbXB0PDUpewogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7bG9hZE1hcChhdHRlbXB0KzEpO30sIGF0dGVtcHQqMjAwMCk7CiAgICB9IGVsc2UgewogICAgICB2YXIgbWk9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpOwogICAgICBpZihtaSkgbWkuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjojMmEzYTRhO3BhZGRpbmc6NDBweDt0ZXh0LWFsaWduOmNlbnRlcjtmb250LWZhbWlseTptb25vc3BhY2U7Zm9udC1zaXplOjExcHgiPk1hcCB1bmF2YWlsYWJsZSDigJQgcmVmcmVzaCB0byByZXRyeTwvZGl2Pic7CiAgICB9CiAgfQp9CgpmdW5jdGlvbiByZW5kZXJNYXAoc3RhdGVzKXsKICB2YXIgdz04MDAsaD04MDAscGo9cHJval8odyxoLDI4KTsKICB2YXIgc2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1zdGF0ZXMnKTsKICB2YXIgcGc9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1wdWxzZXMnKTsKICB2YXIgZ2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1nbG93Jyk7CiAgc2cuaW5uZXJIVE1MPScnO3BnLmlubmVySFRNTD0nJztnZy5pbm5lckhUTUw9Jyc7CgogIHN0YXRlcy5mZWF0dXJlcy5mb3JFYWNoKGZ1bmN0aW9uKGYpewogICAgaWYoIWYuZ2VvbWV0cnkpIHJldHVybjsKICAgIHZhciBubT1zTmFtZShmLnByb3BlcnRpZXMpLGQ9ZyhubSk7CiAgICB2YXIgcGF0aEVsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdwYXRoJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkJyxnZW8ycGF0aChmLmdlb21ldHJ5LHBqKSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3N0YXRlJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnLG5tKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsJ3JnYmEoMjU1LDI1NSwyNTUsMC4wNyknKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzAuNScpOwogICAgc2cuYXBwZW5kQ2hpbGQocGF0aEVsKTsKCiAgICB2YXIgY3Q9Y3RyKGYuZ2VvbWV0cnkpLGNwPXBqKGN0WzBdLGN0WzFdKTsKCiAgICAvLyBBdG1vc3BoZXJpYyBnbG93IGZvciBoaWdoLWF0dGVudGlvbiBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj02NSl7CiAgICAgIHZhciBnbG93RWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2VsbGlwc2UnKTsKICAgICAgdmFyIGdsb3dSPU1hdGgubWluKDYwLDIwK2QuYXR0ZW50aW9uKjAuNSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7Z2xvd0VsLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgncngnLGdsb3dSKTtnbG93RWwuc2V0QXR0cmlidXRlKCdyeScsZ2xvd1IqMC43KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnb3BhY2l0eScsJzAuMDgnKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsdGVyJywndXJsKCNzdGF0ZUdsb3cpJyk7CiAgICAgIGdsb3dFbC5zdHlsZS5hbmltYXRpb249J2dsb3dQdWxzZSAnKygyLjUrTWF0aC5yYW5kb20oKSkrJ3MgZWFzZS1pbi1vdXQgJysoTWF0aC5yYW5kb20oKSoyKSsncyBpbmZpbml0ZSc7CiAgICAgIGdnLmFwcGVuZENoaWxkKGdsb3dFbCk7CiAgICB9CgogICAgLy8gRHVhbCBwdWxzZSByaW5ncyBmb3IgdmVyeSBob3Qgc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NzIpewogICAgICBbMCwxXS5mb3JFYWNoKGZ1bmN0aW9uKGkpewogICAgICAgIHZhciByaW5nPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdjaXJjbGUnKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtyaW5nLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdwdWxzZS1yaW5nIHAnKyhpKzEpKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJyxhQyhkLmF0dGVudGlvbikpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcxJyk7CiAgICAgICAgcmluZy5zdHlsZS5hbmltYXRpb25EZWxheT0oTWF0aC5yYW5kb20oKSoyLjUpKydzJzsKICAgICAgICBwZy5hcHBlbmRDaGlsZChyaW5nKTsKICAgICAgfSk7CiAgICB9CiAgfSk7CiAgYXBwbHlMYXllcigpOwogIGF0dGFjaEludGVyYWN0aW9ucygpOwp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyksZD1nKG5tKSxmaWxsOwogICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJykgZmlsbD1hQyhkLmF0dGVudGlvbik7CiAgICBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgdmFyIGx2PUxJVkVbbm1dO3ZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGVtMj0obHYmJmx2LmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsdi5lbW90aW9ucykubGVuZ3RoKT9sdi5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZGU9KGx2JiZsdi5kb21pbmFudF9lbW90aW9uKXx8ZC5kb21pbmFudF9lbW90aW9ufHxkb21pbmFudEVtb3Rpb24oZW0yKTsKICAgICAgaWYoIWRlJiZkLmF0dGVudGlvbj4yKXt2YXIgbnA9ZC5kb21pbmFudF9uYXJyYXRpdmV8fCcnO2RlPW5wLm1hdGNoKC9ib3JkZXJ8dGVycm9yfHNlY3VyaXR5fGNvbmZsaWN0L2kpPydmZWFyJzpucC5tYXRjaCgvc2NhbXxjb3JydXB0fHByb3Rlc3R8YXJyZXN0L2kpPydhbmdlcic6bnAubWF0Y2goL2RldmVsb3B8aW52ZXN0fGdyb3d0aHxsYXVuY2gvaSk/J2hvcGUnOidhbnhpZXR5Jzt9CiAgICAgIGZpbGw9ZGU/KGVNYXBbZGVdfHxlQyhlbTIpKTplQyhlbTIpfHwnIzMzNDQ1NSc7CiAgICB9CiAgICBlbHNlIGZpbGw9dkMoZC52ZWxvY2l0eSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsZmlsbCk7CiAgICAoZnVuY3Rpb24oKXsKICAgICAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oeCl7cmV0dXJuIHguYXR0ZW50aW9ufHwwO30pOwogICAgICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpLG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAgICAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChkLmF0dGVudGlvbi1tbikvKG14LW1uKSkpOwogICAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxsYXllcj09PSdhdHRlbnRpb24nP01hdGgubWF4KDAuMywwLjMrbiowLjcpOjAuODUpOwogICAgfSkoKTsKICB9KTsKfQoKZnVuY3Rpb24gYXR0YWNoSW50ZXJhY3Rpb25zKCl7CiAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsZnVuY3Rpb24oZSl7CiAgICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyksZD1nKG5tKTsKICAgICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgICAgdmFyIGhpc3Q9ZC5uYXJyYXRpdmVIaXN0b3J5OwogICAgICB2YXIgbGF0ZXN0PWhpc3QmJmhpc3QubGVuZ3RoP2hpc3RbaGlzdC5sZW5ndGgtMV0udG9waWM6J+KAlCc7CiAgICAgIC8vIER5bmFtaWMgdG9vbHRpcCBjb250ZW50IGJhc2VkIG9uIGFjdGl2ZSBsYXllcgogICAgICB2YXIgbGF5ZXJSb3dzPScnOwogICAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgICAgICBsYXllclJvd3M9CiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+QXR0ZW50aW9uIGluZGV4PC9zcGFuPjxzdHJvbmc+JytkLmF0dGVudGlvbisnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgIChkLmRlbHRhIT09MD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+MjRoIHNoaWZ0PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrKGQuZGVsdGE+MD8nI2UwNWEyOCc6JyMzYmI4ZDgnKSsnIj4nKyhkLmRlbHRhPjA/JysnOicnKStkLmRlbHRhKyc8L3N0cm9uZz48L2Rpdj4nOicnKSsKICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Ub3AgbmFycmF0aXZlPC9zcGFuPjxzdHJvbmc+Jyt0b3ArJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgdmFyIGVMaXN0PU9iamVjdC5lbnRyaWVzKGQuZW1vdGlvbnMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgICAgIHZhciByYXdUPWVMaXN0LnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgICAgIGlmKHJhd1Q+MCYmcmF3VDw9MS4wMSl7ZUxpc3Q9ZUxpc3QubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm4gW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7fQogICAgICAgIHZhciB0b3Q9TWF0aC5tYXgoMSxlTGlzdC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKSk7CiAgICAgICAgaWYoIWVMaXN0fHwhZUxpc3QubGVuZ3RoKXsKICAgICAgICAgIGxheWVyUm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RW1vdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPkNvbGxlY3RpbmcuLi48L3N0cm9uZz48L2Rpdj4nOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICB2YXIgZG9tRW1vPWVMaXN0WzBdOwogICAgICAgICAgbGF5ZXJSb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Eb21pbmFudDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3BhbFtkb21FbW9bMF1dKyciPicrZG9tRW1vWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtb1swXS5zbGljZSgxKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgICAgZUxpc3Quc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+JytrdlswXSsnPC9zcGFuPjxzdHJvbmc+JytNYXRoLnJvdW5kKGt2WzFdKjEwMC90b3QpKyclPC9zdHJvbmc+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgICAgICB9fSBlbHNlIHsKICAgICAgICB2YXIgdkRpcj1kLnZlbG9jaXR5PjAuMDU/J1Jpc2luZyc6ZC52ZWxvY2l0eTwtMC4wNT8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICAgICAgdmFyIHZDb2w9ZC52ZWxvY2l0eT4wLjA1PycjZTA1YTI4JzpkLnZlbG9jaXR5PC0wLjA1PycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICAgICAgbGF5ZXJSb3dzPQogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPk1vbWVudHVtPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdkNvbCsnIj4nKyhkLnZlbG9jaXR5PjA/JysnOicnKStkLnZlbG9jaXR5LnRvRml4ZWQoMikrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RGlyZWN0aW9uPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdkNvbCsnIj4nKyhkLnZlbG9jaXR5PjAuMT8nUmlzaW5nIGZhc3QnOmQudmVsb2NpdHk+MC4wMj8nUmlzaW5nJzpkLnZlbG9jaXR5PC0wLjA1PydDb29saW5nJzonU3RhYmxlJykrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+MjRoIHNpZ25hbHM8L3NwYW4+PHN0cm9uZz4nKyhkLmRlbHRhPj0wPycrJzonJykrZC5kZWx0YSsnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfQogICAgICB0aXAuaW5uZXJIVE1MPQogICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1uIj4nK25tKyc8L2Rpdj4nKwogICAgICAgIGxheWVyUm93cysKICAgICAgICAnPGRpdiBjbGFzcz0idHQtbmFyIj48c3Ryb25nPkN1cnJlbnQgbmFycmF0aXZlPC9zdHJvbmc+JytsYXRlc3QrJzwvZGl2Pic7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE4MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTQwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0xOwogICAgfSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbGVhdmUnLGZ1bmN0aW9uKCl7dGlwLnN0eWxlLm9wYWNpdHk9MDt9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7c2VsZWN0XyhwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJykpO30pOwogIH0pOwp9CgovLyBTVEFURSBQQU5FTApmdW5jdGlvbiBzZWxlY3RfKG5tKXsKICBTRUw9bm07CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXtwLmNsYXNzTGlzdC50b2dnbGUoJ3NlbGVjdGVkJyxwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk9PT1ubSk7fSk7CiAgcmVuZGVyUGFuZWwobm0pOwogIGZldGNoRGV0YWlsKG5tKS50aGVuKGZ1bmN0aW9uKGQpe2lmKFNFTD09PW5tKSByZW5kZXJQYW5lbChubSk7fSk7Cn0KCmZ1bmN0aW9uIHJlbmRlclBhbmVsKG5tKXsKICB2YXIgZD1nKG5tKSxwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYoIXBhbmVsKSByZXR1cm47CiAgdmFyIGlzRmF2PUZBVlMuaGFzKG5tKSxkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPyd1cCc6J2RuJzsKICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICB2YXIgZW1vdGlvbnM9ZC5lbW90aW9ucyYmT2JqZWN0LmtleXMoZC5lbW90aW9ucykubGVuZ3RoP2QuZW1vdGlvbnM6e2FueGlldHk6MjAsYW5nZXI6MTUsaG9wZToyNSxwcmlkZToyNSxmZWFyOjE1fTsKICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vdGlvbnMpOwogIC8vIE5vcm1hbGl6ZTogQVBJIG1heSByZXR1cm4gMC0xIGZsb2F0cyBPUiAwLTEwMCBpbnRlZ2VycwogIHZhciByYXdUb3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgaWYocmF3VG90PjAgJiYgcmF3VG90PD0xLjAxKXsgZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm4gW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7IH0KICB2YXIgdG90PU1hdGgubWF4KDEsZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogIHZhciBjdW1BPS1NYXRoLlBJLzIsY3g9MzgsY3k9MzgsUj0zMyxyaT0yMDsKICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgdmFyIGs9a3ZbMF0sdj1rdlsxXSxmcj12L3RvdCxhMT1jdW1BLGEyPWN1bUErZnIqTWF0aC5QSSoyOwogICAgY3VtQT1hMjt2YXIgbGc9KGEyLWExKT5NYXRoLlBJPzE6MDsKICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgIHZhciB4Mz1jeCtNYXRoLmNvcyhhMikqcmkseTM9Y3krTWF0aC5zaW4oYTIpKnJpLHg0PWN4K01hdGguY29zKGExKSpyaSx5ND1jeStNYXRoLnNpbihhMSkqcmk7CiAgICByZXR1cm4gJzxwYXRoIGQ9Ik0nK3gxLnRvRml4ZWQoMSkrJywnK3kxLnRvRml4ZWQoMSkrJyBBJytSKycsJytSKycgMCAnK2xnKycgMSAnK3gyLnRvRml4ZWQoMSkrJywnK3kyLnRvRml4ZWQoMSkrJyBMJyt4My50b0ZpeGVkKDEpKycsJyt5My50b0ZpeGVkKDEpKycgQScrcmkrJywnK3JpKycgMCAnK2xnKycgMCAnK3g0LnRvRml4ZWQoMSkrJywnK3k0LnRvRml4ZWQoMSkrJyBaIiBmaWxsPSInK3BhbFtrXSsnIiBvcGFjaXR5PSIwLjkiLz4nOwogIH0pLmpvaW4oJycpOwoKICB2YXIgdGw9ZC50aW1lbGluZSx0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgdmFyIHR3PTI2MCx0aD02Mix0cD01OwogIHZhciBwdHM9dGwubWFwKGZ1bmN0aW9uKHYsaSl7cmV0dXJuIFt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICB2YXIgcEQ9cHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAoaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSkuam9pbignJyk7CiAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICB2YXIgYWM9YUMoZC5hdHRlbnRpb24pOwoKICB2YXIgaGlzdD1kLm5hcnJhdGl2ZUhpc3Rvcnl8fFtdOwoKICBwYW5lbC5pbm5lckhUTUw9CiAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgJzxkaXY+PGRpdiBjbGFzcz0ic3AtZWsiPk5hcnJhdGl2ZSBwYW5lbDwvZGl2PjxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PjwvZGl2PicrCiAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKGlzRmF2Pydvbic6JycpKyciIG9uY2xpY2s9InRvZ2dsZUZhdihcJycrbm0rJ1wnKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoaXNGYXY/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAnPC9idXR0b24+JysKICAgICc8L2Rpdj4nKwoKICAgIC8vIE5hcnJhdGl2ZSBoaXN0b3J5IHRpbWVsaW5lIOKAlCBzaWduYXR1cmUgZmVhdHVyZQogICAgKGhpc3QubGVuZ3RoPwogICAgICAnPGRpdiBjbGFzcz0ibmFyLXRpbWVsaW5lIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJudC1sYWJlbCI+TmFycmF0aXZlIGV2b2x1dGlvbjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im50LWZsb3ciPicrCiAgICAgICAgICBoaXN0Lm1hcChmdW5jdGlvbihoKXsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJudC1zdGVwICcraC5jbHMrJyI+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibnQtZG90Ij48L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJudC1jb250ZW50Ij4nKwogICAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im50LXRvcGljIj4nK2gudG9waWMrJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibnQtd2hlbiI+JytoLndoZW4rJzwvZGl2PicrCiAgICAgICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOicnKSsKCiAgICAnPGRpdiBjbGFzcz0iaW5zaWdodCI+JytkLnN1bW1hcnkrJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JytkLmF0dGVudGlvbisnPC9kaXY+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTK2QuZGVsdGErJzwvZGl2PjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKGQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+TmFycmF0aXZlIGJyZWFrZG93bjwvZGl2PicrCiAgICAgIChkLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlcy5sZW5ndGg/CiAgICAgICAgJzxkaXYgY2xhc3M9Im5hci1saXN0Ij4nKwogICAgICAgIGQubmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICB2YXIgbm09bi5uYW1lP24ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hbWUuc2xpY2UoMSk6bi5uYW1lOwogICAgICAgICAgdmFyIHZhbD10eXBlb2Ygbi52YWw9PT0nbnVtYmVyJz9uLnZhbDowOwogICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPicrCiAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS1sYWJlbCI+JytubSsobi5kaXI9PT0ndXAnPycgPHNwYW4gc3R5bGU9ImNvbG9yOiNlMDVhMjg7Zm9udC1zaXplOjlweCI+4oaRPC9zcGFuPic6bi5kaXI9PT0nZG93bic/JyA8c3BhbiBzdHlsZT0iY29sb3I6IzNiYjhkODtmb250LXNpemU6OXB4Ij7ihpM8L3NwYW4+JzonJykrJzwvZGl2PicrCiAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS12YWwiPicrdmFsLnRvRml4ZWQoMSkrJyU8L2Rpdj4nKwogICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCx2YWwqMi41KSsnJTtiYWNrZ3JvdW5kOicrKG4uZGlyPT09J3VwJz8nI2UwNWEyOCc6bi5kaXI9PT0nZG93bic/JyMzYmI4ZDgnOicjMzM0NDU1JykrJyI+PC9kaXY+PC9kaXY+JysKICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICc8L2Rpdj4nOgogICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMDtsaW5lLWhlaWdodDoxLjYiPkxvdy1zaWduYWwgcmVnaW9uLiBOYXRpb25hbCBwcmVzcyBjb3ZlcmFnZSBpcyBsaW1pdGVkIGZvciB0aGlzIHN0YXRlIOKAlCByZWdpb25hbCBsYW5ndWFnZSBzb3VyY2VzIGFyZSBiZWluZyBtb25pdG9yZWQuPC9kaXY+JykrCiAgICAnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+TW92ZW1lbnQ8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ibXYtZ3JpZCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibXYtYmxvY2sgdXAiPjxkaXYgY2xhc3M9Im12LWgiPlJpc2luZzwvZGl2PicrCiAgICAgICAgICAoZC5yaXNpbmcubGVuZ3RoP2QucmlzaW5nLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im12LWl0Ij48c3Ryb25nPicrci50Kyc8L3N0cm9uZz48c3Bhbj4nK3IucGN0Kyc8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKTonPGRpdiBjbGFzcz0ibXYtaXQiIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPlN0YWJsZTwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtdi1ibG9jayBkbiI+PGRpdiBjbGFzcz0ibXYtaCI+RmFsbGluZzwvZGl2PicrCiAgICAgICAgICAoZC5mYWxsaW5nLmxlbmd0aD9kLmZhbGxpbmcubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0ibXYtaXQiPjxzdHJvbmc+JytyLnQrJzwvc3Ryb25nPjxzcGFuPicrci5wY3QrJzwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpOic8ZGl2IGNsYXNzPSJtdi1pdCIgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+U3RhYmxlPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkVtb3Rpb25hbCByZWdpc3RlcjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJlbS1yb3ciPicrCiAgICAgICAgJzxzdmcgY2xhc3M9ImVtLWRvbnV0IiB2aWV3Qm94PSIwIDAgNzYgNzYiPicrYXJjcysnPC9zdmc+JysKICAgICAgICAnPGRpdiBjbGFzcz0iZW0tbGVnIj4nKwogICAgICAgICAgZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSkubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgICAgICAgdmFyIGs9a3ZbMF0sdj1rdlsxXTsKICAgICAgICAgICAgdmFyIGRlc2M9e2FueGlldHk6J1VuY2VydGFpbnR5ICYgd29ycnknLGFuZ2VyOidPdXRyYWdlICYgcHJvdGVzdCcsaG9wZTonT3B0aW1pc20gJiBncm93dGgnLHByaWRlOidBY2hpZXZlbWVudCAmIGlkZW50aXR5JyxmZWFyOidUaHJlYXQgcGVyY2VwdGlvbid9OwogICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImVtLWl0ZW0iIHN0eWxlPSJtYXJnaW4tYm90dG9tOjFweCI+JysKICAgICAgICAgICAgICAnPHNwYW4gY2xhc3M9ImVtLXN3IiBzdHlsZT0iYmFja2dyb3VuZDonK3BhbFtrXSsnIj48L3NwYW4+JysKICAgICAgICAgICAgICAnPHNwYW4gY2xhc3M9ImVtLW4iPicray5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStrLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICAgICAnPHNwYW4gY2xhc3M9ImVtLXAiPicrTWF0aC5yb3VuZCh2KjEwMC90b3QpKyclPC9zcGFuPicrCiAgICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICAgICAodj09PWVMLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pWzBdWzFdPwogICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6MXB4IDAgNHB4IDEycHg7Ym9yZGVyLWxlZnQ6MXB4IHNvbGlkICcrcGFsW2tdKyc7bWFyZ2luLWxlZnQ6M3B4O21hcmdpbi1ib3R0b206M3B4OyI+JytkZXNjW2tdKyc8L2Rpdj4nOgogICAgICAgICAgICAnJyk7CiAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QXR0ZW50aW9uIOKAlCA4IGRheXM8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0idGwtd3JhcCI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgJyt0dysnICcrdGgrJyIgc3R5bGU9IndpZHRoOjEwMCU7aGVpZ2h0OjEwMCUiPicrCiAgICAgICAgICAnPGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJ0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJyIgeDE9IjAiIHgyPSIwIiB5MT0iMCIgeTI9IjEiPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMC4yNSIvPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwIi8+JysKICAgICAgICAgICc8L2xpbmVhckdyYWRpZW50PjwvZGVmcz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrYUQrJyIgZmlsbD0idXJsKCN0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJykiLz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrcEQrJyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSInK2FjKyciIHN0cm9rZS13aWR0aD0iMS4yIi8+JysKICAgICAgICAgIHB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gJzxjaXJjbGUgY3g9IicrcFswXSsnIiBjeT0iJytwWzFdKyciIHI9IicrKGk9PT1wdHMubGVuZ3RoLTE/Mi4yOjEuMikrJyIgZmlsbD0iJythYysnIi8+Jzt9KS5qb2luKCcnKSsKICAgICAgICAnPC9zdmc+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFscyA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nK2QuYXJ0aWNsZXMubGVuZ3RoKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImFydC1saXN0Ij4nKwogICAgICAgIGQuYXJ0aWNsZXMubWFwKGZ1bmN0aW9uKGEpe3JldHVybiAnPGRpdiBjbGFzcz0iYXJ0LWl0ZW0iPjxkaXYgY2xhc3M9ImFydC1zcmMiPicrYS5zcmMrJzwvZGl2PjxkaXYgY2xhc3M9ImFydC10eHQiPicrYS50eHQrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJykrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2Pic7Cn0KCmZ1bmN0aW9uIHRvZ2dsZUZhdihubSl7CiAgaWYoRkFWUy5oYXMobm0pKSBGQVZTLmRlbGV0ZShubSk7ZWxzZSBGQVZTLmFkZChubSk7CiAgcmVuZGVyUGFuZWwoU0VMKTtyZW5kZXJGYXZzKCk7Cn0KZnVuY3Rpb24gcmVuZGVyRmF2cygpewogIHZhciByb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Zhdi1yb3cnKTsKICBpZighRkFWUy5zaXplKXtyb3cuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4nO3JldHVybjt9CiAgcm93LmlubmVySFRNTD1BcnJheS5mcm9tKEZBVlMpLm1hcChmdW5jdGlvbihubSl7CiAgICB2YXIgZD1nKG5tKSxkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPycjZTA1YTI4JzonIzNiYjhkOCc7CiAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJmYXYtY2FyZCIgb25jbGljaz0ic2VsZWN0XyhcJycrbm0rJ1wnKSI+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLWhlYWQiPjxzcGFuIGNsYXNzPSJmYy1uYW1lIj4nK25tKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImZjLXNjIj4nK2QuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+TmFycmF0aXZlPC9zcGFuPjxzcGFuIGNsYXNzPSJ2Ij4nK3RvcCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPjI0aDwvc3Bhbj48c3BhbiBjbGFzcz0idiIgc3R5bGU9ImNvbG9yOicrZEMrJyI+JytkUytkLmRlbHRhKyc8L3NwYW4+PC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9Cgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oYyl7CiAgYy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbih4KXt4LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgYy5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtsYXllcj1jLmRhdGFzZXQubGF5ZXI7YXBwbHlMYXllcigpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHVwZGF0ZUNsb2NrKCl7CiAgdmFyIG5vdz1uZXcgRGF0ZSgpLGlzdD1uZXcgRGF0ZShub3cuZ2V0VGltZSgpK25vdy5nZXRUaW1lem9uZU9mZnNldCgpKjYwMDAwKzE5ODAwMDAwKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2xvY2snKS50ZXh0Q29udGVudD1TdHJpbmcoaXN0LmdldEhvdXJzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRNaW51dGVzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRTZWNvbmRzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnIElTVCc7Cn0Kc2V0SW50ZXJ2YWwodXBkYXRlQ2xvY2ssMTAwMCk7dXBkYXRlQ2xvY2soKTsKCi8vIElOSVQg4oCUIHdhaXQgZm9yIERPTQpmdW5jdGlvbiBpbml0KCl7CiAgcmVuZGVyU3RyaXAoJzNtJyk7CiAgcmVuZGVyTW9tZW50dW0oKTsKICAvLyBMb2FkIG1hcCB3aXRoIHJldHJ5IGlmIHRvcG9qc29uIG5vdCByZWFkeQogIHZhciBtYXBBdHRlbXB0cz0wOwogIGZ1bmN0aW9uIHRyeUxvYWRNYXAoKXsKICAgIGlmKHR5cGVvZiB0b3BvanNvbj09PSd1bmRlZmluZWQnKXsKICAgICAgaWYobWFwQXR0ZW1wdHMrKzwxMCkgc2V0VGltZW91dCh0cnlMb2FkTWFwLDMwMCk7CiAgICAgIHJldHVybjsKICAgIH0KICAgIGxvYWRNYXAoKTsKICB9CiAgdHJ5TG9hZE1hcCgpOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtzdGFydFBvbGxpbmcoKTt9LDgwMCk7CiAgLy8gRmV0Y2ggaW5zaWdodHMgYWZ0ZXIgZGF0YSBpcyBsb2FkZWQKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw1MDAwKTsKICAvLyBSZXRyeSBtYXAgaWYgc3RpbGwgZW1wdHkgYWZ0ZXIgM3MKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICBpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKXsKICAgICAgY29uc29sZS5sb2coJ1tpbml0XSBtYXAgZW1wdHksIHJldHJ5aW5nIGxvYWRNYXAnKTsKICAgICAgbG9hZE1hcCgpOwogICAgfQogIH0sMzAwMCk7CiAgLy8gQW5kIGFnYWluIGF0IDZzCiAgc2V0VGltZW91dChmdW5jdGlvbigpewogICAgaWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aCl7CiAgICAgIGxvYWRNYXAoKTsKICAgIH0KICB9LDYwMDApOwp9CmlmKGRvY3VtZW50LnJlYWR5U3RhdGU9PT0nbG9hZGluZycpewogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ0RPTUNvbnRlbnRMb2FkZWQnLCBpbml0KTsKfSBlbHNlIHsKICAvLyBBbHJlYWR5IGxvYWRlZCDigJQgYnV0IHdhaXQgb25lIHRpY2sgdG8gZW5zdXJlIGFsbCBzY3JpcHRzIHBhcnNlZAogIHNldFRpbWVvdXQoaW5pdCwgMCk7Cn0KCi8vIFJFUExBWSBJTkRJQQp2YXIgUkVQTEFZX1BFUklPRFM9eyc3ZCc6e2RheXM6NyxsYWJlbDonUGFzdCA3IGRheXMnfSwnMzBkJzp7ZGF5czozMCxsYWJlbDonUGFzdCAzMCBkYXlzJ30sJzZtJzp7ZGF5czoxODAsbGFiZWw6J1Bhc3QgNiBtb250aHMnfSwnZWxlY3Rpb24nOntkYXlzOjkwLGxhYmVsOidFbGVjdGlvbiBzZWFzb24gMjAyNCd9fTsKdmFyIHJlcGxheVBlcmlvZD0nN2QnLHJlcGxheVBvcz0wLHJlcGxheVBsYXlpbmc9ZmFsc2UscmVwbGF5VGltZXI9bnVsbCxyZXBsYXlTcGVlZD0xLGxhc3RTbmFwUG9zPS0xOwpmdW5jdGlvbiBmbXREYXRlKGQpe3JldHVybiBkLnRvTG9jYWxlRGF0ZVN0cmluZygnZW4tSU4nLHtkYXk6J251bWVyaWMnLG1vbnRoOidzaG9ydCd9KTt9CmZ1bmN0aW9uIGluaXRSZXBsYXkoKXsKICB2YXIgcD1SRVBMQVlfUEVSSU9EU1tyZXBsYXlQZXJpb2RdLG5vdz1uZXcgRGF0ZSgpLHN0YXJ0PW5ldyBEYXRlKG5vdy1wLmRheXMqODY0MDAwMDApOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtZGF0ZXMnKTsKICBpZihlbCllbC5pbm5lckhUTUw9JzxzcGFuPicrZm10RGF0ZShzdGFydCkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjMzKSkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjY2KSkrJzwvc3Bhbj48c3Bhbj5Ub2RheTwvc3Bhbj4nOwogIHNldFJlcGxheVBvcygwKTsKfQpmdW5jdGlvbiBzZXRSZXBsYXlQb3MocG9zKXsKICByZXBsYXlQb3M9TWF0aC5tYXgoMCxNYXRoLm1pbigxLHBvcykpOwogIHZhciBmaWxsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1maWxsJyksdGh1bWI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRodW1iJyksZGF0ZUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1jdXJyZW50LWRhdGUnKTsKICBpZihmaWxsKWZpbGwuc3R5bGUud2lkdGg9KHJlcGxheVBvcyoxMDApKyclJzsKICBpZih0aHVtYil0aHVtYi5zdHlsZS5sZWZ0PShyZXBsYXlQb3MqMTAwKSsnJSc7CiAgdmFyIHA9UkVQTEFZX1BFUklPRFNbcmVwbGF5UGVyaW9kXSxub3c9bmV3IERhdGUoKSxzdGFydD1uZXcgRGF0ZShub3ctcC5kYXlzKjg2NDAwMDAwKSxjdXI9bmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3JlcGxheVBvcypwLmRheXMqODY0MDAwMDApOwogIGlmKGRhdGVFbClkYXRlRWwudGV4dENvbnRlbnQ9Zm10RGF0ZShjdXIpKycg4oCUICcrcC5sYWJlbDsKICB2YXIgc2NhbGU9MC4zNStyZXBsYXlQb3MqMC42NTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLHNhPShkLmF0dGVudGlvbnx8MCkqc2NhbGU7CiAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4gKHguYXR0ZW50aW9ufHwwKSpzY2FsZTt9KTsKICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxLG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzYS1tbikvKG14LW1uKSkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKHNhKSk7cC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsTWF0aC5tYXgoMC4yLDAuMituKjAuOCkpOwogIH0pOwogIGlmKE1hdGguYWJzKHJlcGxheVBvcy1sYXN0U25hcFBvcyk+MC4xMil7bGFzdFNuYXBQb3M9cmVwbGF5UG9zO3VwZGF0ZVJlcGxheVNuYXBzaG90KHJlcGxheVBvcyk7fQp9CmZ1bmN0aW9uIHVwZGF0ZVJlcGxheVNuYXBzaG90KHBvcyl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uYXR0ZW50aW9uPjA7fSkubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm57bmFtZTprdlswXSxhdHQ6TWF0aC5yb3VuZCgoa3ZbMV0uYXR0ZW50aW9ufHwwKSooMC4zNStwb3MqMC42NSkpLG5hcjooa3ZbMV0ubmFycmF0aXZlcyYma3ZbMV0ubmFycmF0aXZlc1swXT9rdlsxXS5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCcpfTt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pLnNsaWNlKDAsNik7CiAgdmFyIHNuYXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXNuYXAtc3RhdGVzJyk7CiAgaWYoIXNuYXApcmV0dXJuOwogIGlmKCF0b3AubGVuZ3RoKXtzbmFwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5ObyBzaWduYWwgZGF0YSB5ZXQuPC9kaXY+JztyZXR1cm47fQogIHNuYXAuaW5uZXJIVE1MPXRvcC5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuICc8ZGl2IGNsYXNzPSJycC1zdGF0ZS1jYXJkIj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1uYW1lIj4nK3MubmFtZSsnPC9kaXY+PGRpdiBjbGFzcz0icnAtc3RhdGUtbmFyIj4nK3MubmFyKyc8L2Rpdj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1hdHQiPkF0dGVudGlvbiAnK3MuYXR0Kyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwp9CmZ1bmN0aW9uIHRvZ2dsZVJlcGxheSgpewogIHJlcGxheVBsYXlpbmc9IXJlcGxheVBsYXlpbmc7CiAgdmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpOwogIGlmKHJlcGxheVBsYXlpbmcpe2lmKHJlcGxheVBvcz49MC45OSlzZXRSZXBsYXlQb3MoMCk7aWYoaWNvbilpY29uLnNldEF0dHJpYnV0ZSgncG9pbnRzJywnMywyIDcsMiA3LDggMyw4IE04LDIgMTIsMiAxMiw4IDgsOCcpO3J1blJlcGxheSgpO30KICBlbHNle2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO30KfQpmdW5jdGlvbiBydW5SZXBsYXkoKXsKICBjbGVhckludGVydmFsKHJlcGxheVRpbWVyKTsKICByZXBsYXlUaW1lcj1zZXRJbnRlcnZhbChmdW5jdGlvbigpewogICAgcmVwbGF5UG9zKz0wLjAwMypyZXBsYXlTcGVlZDsKICAgIGlmKHJlcGxheVBvcz49MSl7cmVwbGF5UG9zPTE7c2V0UmVwbGF5UG9zKDEpO3JlcGxheVBsYXlpbmc9ZmFsc2U7dmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpO2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO3JldHVybjt9CiAgICBzZXRSZXBsYXlQb3MocmVwbGF5UG9zKTsKICB9LDYwKTsKfQooZnVuY3Rpb24oKXt2YXIgdHJhY2s9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRyYWNrJyk7aWYoIXRyYWNrKXJldHVybjt2YXIgZHJhZz1mYWxzZTsKdHJhY2suYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vkb3duJyxmdW5jdGlvbihlKXtkcmFnPXRydWU7dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXtpZighZHJhZylyZXR1cm47dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2V1cCcsZnVuY3Rpb24oKXtpZihkcmFnKXtkcmFnPWZhbHNlO2lmKCFyZXBsYXlQbGF5aW5nKWFwcGx5TGF5ZXIoKTt9fSk7fSkoKTsKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYnRuKXtidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7ZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYil7Yi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTtidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVwbGF5UGVyaW9kPWJ0bi5kYXRhc2V0LnBlcmlvZDtyZXBsYXlQb3M9MDtsYXN0U25hcFBvcz0tMTtpbml0UmVwbGF5KCk7fSk7fSk7CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGJ0bil7YnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe2RvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGIpe2IuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7YnRuLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlcGxheVNwZWVkPXBhcnNlSW50KGJ0bi5kYXRhc2V0LnNwZCk7fSk7fSk7CmluaXRSZXBsYXkoKTsKc2V0VGltZW91dChmdW5jdGlvbigpewogIC8vIEF1dG8tc2VsZWN0IGhvdHRlc3Qgc3RhdGUgZnJvbSBMSVZFIGRhdGEKICB2YXIgc3JjPU9iamVjdC5rZXlzKExJVkUpLmxlbmd0aD9MSVZFOlNEOwogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoc3JjKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApewogICAgdmFyIGVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9IicrdG9wWzBdKyciXScpOwogICAgaWYoZWwpIHNlbGVjdF8odG9wWzBdKTsKICB9Cn0sMzAwMCk7CnNldFRpbWVvdXQocmVuZGVyRmF2cywyNDAwKTsKPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPgo="

@app.get("/", include_in_schema=False)
async def serve_frontend():
    from fastapi.responses import HTMLResponse
    html = _b64.b64decode(FRONTEND_HTML_B64).decode('utf-8')
    return HTMLResponse(content=html, status_code=200)


# ── Heatmap endpoint ────────────────────────────────────────────────────────

@app.get("/api/states")
async def list_states():
    """Lightweight data for every state — what the map heatmap reads."""
    if not store.scores:
        await recompute_all_scores()
    return [
        {
            "name": s["name"],
            "attention": s["attention"],
            "delta_24h": s["delta_24h"],
            "velocity": s["velocity"],
            "dominant_emotion": s.get("dominant_emotion", "anxiety"),
            "dominant_narrative": s.get("dominant_narrative", "governance"),
        }
        for s in store.scores.values()
    ]


# ── State detail panel ──────────────────────────────────────────────────────

@app.get("/api/state/{name}")
async def state_detail(name: str):
    # Try to find state (case-insensitive)
    matched = next((s for s in INDIAN_STATES if s.lower() == name.lower()), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"State '{name}' not found")
    score = store.scores.get(matched)
    if not score:
        score = await recompute_state_score(matched)
        store.scores[matched] = score
    return score


# ── Daily snapshot ──────────────────────────────────────────────────────────

@app.get("/api/snapshot/daily")
async def daily_snapshot():
    if not store.scores:
        return {"error": "no data yet — ingestion in progress, try again in ~60s"}
    scored = list(store.scores.values())
    hottest    = max(scored, key=lambda s: s["attention"])
    coolest    = min(scored, key=lambda s: s["attention"])
    fastest_up = max(scored, key=lambda s: s.get("delta_24h", 0))
    fastest_dn = min(scored, key=lambda s: s.get("delta_24h", 0))

    # Most polarized: highest emotional intensity (min emotion dominates least)
    def polarization(s):
        emos = list(s.get("emotions", {}).values())
        return max(emos) - min(emos) if emos else 0

    most_polarized = max(scored, key=polarization)

    return {
        "hottest_state":      hottest["name"],
        "hottest_score":      hottest["attention"],
        "coolest_state":      coolest["name"],
        "fastest_rising":     fastest_up["name"],
        "fastest_cooling":    fastest_dn["name"],
        "most_polarized":     most_polarized["name"],
        "top_national_narrative": (
            max(
                {n["name"]: n["val"] for s in scored for n in s.get("narratives", [])}.items(),
                key=lambda x: x[1],
                default=("governance", 0),
            )[0]
        ),
        "as_of": store.last_ingest.isoformat() if store.last_ingest else None,
        "total_signals": sum(len(v) for v in store.signals.values()),
    }


# ── Manual ingest trigger ───────────────────────────────────────────────────

@app.post("/api/ingest/run")
async def manual_ingest(background_tasks: BackgroundTasks):
    if store.ingest_running:
        return {"status": "already_running"}
    background_tasks.add_task(run_ingest)
    return {"status": "started", "message": "Ingestion running in background"}


# ── Health ──────────────────────────────────────────────────────────────────

def build_insights() -> dict:
    """Build national insights from live signal data."""
    try:
        scored = [s for s in store.scores.values() if isinstance(s, dict) and s.get("signal_count", 0) > 0]
        if not scored:
            # Fall back to all scored states if none have signal_count
            scored = [s for s in store.scores.values() if isinstance(s, dict) and s.get("attention", 0) > 0]
        if not scored:
            return {"error": "no data"}
    except Exception as e:
        return {"error": str(e)}

    nar_rising: dict[str, float] = {}
    nar_falling: dict[str, float] = {}
    nar_all: dict[str, float] = {}

    for s in scored:
        for n in s.get("narratives", []):
            nm2 = n["name"]
            val = n.get("val", 0)
            nar_all[nm2] = nar_all.get(nm2, 0) + val
            if n.get("dir") == "up":
                nar_rising[nm2] = nar_rising.get(nm2, 0) + val
            elif n.get("dir") == "down":
                nar_falling[nm2] = nar_falling.get(nm2, 0) + val

    top_rising  = sorted(nar_rising.items(),  key=lambda x: x[1], reverse=True)[:5]
    top_falling = sorted(nar_falling.items(), key=lambda x: x[1], reverse=True)[:5]

    main_rising  = top_rising[0][0]  if top_rising  else "governance"
    main_falling = top_falling[0][0] if top_falling else "inflation"
    sec_rising   = top_rising[1][0]  if len(top_rising) > 1 else "border security"
    hottest = max(scored, key=lambda s: s.get("attention", 0))

    regions = {
        "North":   ["Delhi","Uttar Pradesh","Punjab","Haryana","Himachal Pradesh","Uttarakhand","Jammu and Kashmir"],
        "East":    ["West Bengal","Bihar","Jharkhand","Odisha"],
        "West":    ["Maharashtra","Gujarat","Rajasthan","Goa"],
        "South":   ["Tamil Nadu","Karnataka","Kerala","Andhra Pradesh","Telangana"],
        "NE":      ["Assam","Manipur","Nagaland","Mizoram","Meghalaya","Tripura","Arunachal Pradesh","Sikkim"],
        "Central": ["Madhya Pradesh","Chhattisgarh"],
    }
    regional = []
    for region, states in regions.items():
        rs = [s for s in scored if s.get("name") in states]
        if not rs:
            continue
        na: dict[str, float] = {}
        for s in rs:
            for n in s.get("narratives", []):
                na[n["name"]] = na.get(n["name"], 0) + n.get("val", 0)
        if not na:
            continue
        top_n = max(na.items(), key=lambda x: x[1])
        hs = max(rs, key=lambda s: s.get("attention", 0))
        regional.append({"region": region, "top_narrative": top_n[0],
                         "hottest_state": hs["name"], "attention": round(hs.get("attention", 0), 1)})

    rising_cards = []
    for nar, val in top_rising[:5]:
        st = [s["name"] for s in scored
              if any(n["name"] == nar and n.get("dir") == "up" for n in s.get("narratives", []))][:3]
        rising_cards.append({"narrative": nar, "signal_share": round(val, 1), "states": st})

    falling_cards = []
    for nar, val in top_falling[:5]:
        st = [s["name"] for s in scored
              if any(n["name"] == nar and n.get("dir") == "down" for n in s.get("narratives", []))][:3]
        falling_cards.append({"narrative": nar, "signal_share": round(val, 1), "states": st})

    try:
        return {
            "signature": {"fading": main_falling, "rising_primary": main_rising,
                          "rising_secondary": sec_rising, "hottest_state": hottest["name"]},
            "tags": [
                {"label": main_falling.capitalize(), "dir": "down"},
                {"label": main_rising.capitalize(),  "dir": "up"},
                {"label": sec_rising.capitalize(),   "dir": "up"},
            ],
            "rising":   rising_cards,
            "falling":  falling_cards,
            "regional": regional,
            "as_of":    store.cache_built_at.isoformat() if store.cache_built_at else None,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/insights")
async def get_insights():
    """Cached national narrative insights — refreshed each ingest cycle."""
    if hasattr(store, "cache_insights") and store.cache_insights:
        return store.cache_insights
    if store.scores:
        return build_insights()
    return {"error": "warming up"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "openai_configured": HAS_OPENAI,
        "last_ingest": store.last_ingest.isoformat() if store.last_ingest else None,
        "ingest_running": store.ingest_running,
        "total_signals": sum(len(v) for v in store.signals.values()),
        "states_with_data": sum(1 for v in store.signals.values() if v),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
