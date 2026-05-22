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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjE0cHggMjBweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MTRweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsIHJnYmEoMjI0LDkwLDQwLDAuMDYpIDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMTAwJSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoOHB4KTsKICBtYXgtd2lkdGg6OTAwcHg7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMjVlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjEwcHg7Cn0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNHB4LDEuNHZ3LDE4cHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTtsZXR0ZXItc3BhY2luZzotMC4wMWVtOwp9Ci5zaS10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWFjY2VudCl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5tYXAtY2FyZDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNzAlIDUwJSBhdCAzNSUgMCUsIHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCA4MCUgMTAwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDMpIDAlLCB0cmFuc3BhcmVudCA2MCUpOwp9Ci5tYXAtdG9wewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTJweCAxOHB4IDA7Cn0KLm1hcC10aXRsZS1ibG9jayAubXR7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxN3B4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoubWFwLXRpdGxlLWJsb2NrIC5tc3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA2ZW07bWFyZ2luLXRvcDoycHh9Ci5sZWdlbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1kaW0pfQoubGVnZW5kLWJhcnsKICBoZWlnaHQ6M3B4O3dpZHRoOjgwcHg7Ym9yZGVyLXJhZGl1czoycHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQsIzBlMjAzNSwjMWE1NTgwIDI1JSwjOGE1YzE4IDU1JSwjYzAzODFhIDgwJSwjZTAxMDIwKTsKfQoubGF5ZXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7CiAgcGFkZGluZzoxMHB4IDIwcHggNnB4Owp9Ci5sYXllci1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5sdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjNweH0KLmx0YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjNweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQoubHRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDgpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpfQoubHRhYntkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NXB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OnZpc2libGV9Ci5sdGFiLWluZm97d2lkdGg6MTNweDtoZWlnaHQ6MTNweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4yKTtmb250LXNpemU6OHB4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc3R5bGU6aXRhbGljO2ZvbnQtd2VpZ2h0OjYwMDtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuMzUpO2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y3Vyc29yOmhlbHA7ZmxleC1zaHJpbms6MDt0cmFuc2l0aW9uOmFsbCAwLjE1cztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEwMH0KLmx0YWItaW5mbzpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjZW50KTtjb2xvcjp2YXIoLS1hY2NlbnQpfQoubHRhYi1pbmZvOjphZnRlcntjb250ZW50OmF0dHIoZGF0YS10aXApO3Bvc2l0aW9uOmFic29sdXRlO2JvdHRvbTpjYWxjKDEwMCUgKyAxMHB4KTtsZWZ0OjA7d2lkdGg6MjMwcHg7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjtsZXR0ZXItc3BhY2luZzowO3RleHQtdHJhbnNmb3JtOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMnM7ei1pbmRleDoxMDAwMDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNSl9Ci5sdGFiLWluZm86aG92ZXI6OmFmdGVye29wYWNpdHk6MX0KLmx0YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KCi5tYXAtc3ZnLXdyYXB7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZzoxMnB4IDE2cHggMTZweDsKfQoubWFwLWlubmVye3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxLzE7d2lkdGg6MTAwJX0KI2luZGlhLW1hcHt3aWR0aDoxMDAlO2hlaWdodDoxMDAlO2Rpc3BsYXk6YmxvY2s7b3ZlcmZsb3c6dmlzaWJsZX0KCi8qIG1hcCBzdGF0ZSBzdHlsZXMgKi8KI2luZGlhLW1hcCAuc3RhdGV7CiAgY3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpmaWx0ZXIgMC4yNXMgZWFzZSwgc3Ryb2tlLXdpZHRoIDAuMnMgZWFzZSwgc3Ryb2tlIDAuMnMgZWFzZTsKfQojaW5kaWEtbWFwIC5zdGF0ZTpob3ZlcnsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjcpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjFweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMjUpIGRyb3Atc2hhZG93KDAgMCAxMHB4IHJnYmEoMjU1LDI1NSwyNTUsMC4yKSk7Cn0KI2luZGlhLW1hcCAuc3RhdGUuc2VsZWN0ZWR7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC45KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjRweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMzUpIGRyb3Atc2hhZG93KDAgMCAxNnB4IHJnYmEoMjU1LDI1NSwyNTUsMC4zKSk7Cn0KCi8qIGFuaW1hdGVkIHB1bHNlIHJpbmdzICovCi5wdWxzZS1yaW5ne2ZpbGw6bm9uZTtwb2ludGVyLWV2ZW50czpub25lfQoucHVsc2UtcmluZy5wMXthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCBpbmZpbml0ZX0KLnB1bHNlLXJpbmcucDJ7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgMC45cyBpbmZpbml0ZX0KQGtleWZyYW1lcyBwcnsKICAwJXtyOjQ7b3BhY2l0eTowLjc7c3Ryb2tlLXdpZHRoOjEuMn0KICAxMDAle3I6MjY7b3BhY2l0eTowO3N0cm9rZS13aWR0aDowLjJ9Cn0KCi8qIGF0bW9zcGhlcmljIGdsb3cgYmVoaW5kIGhvdCBzdGF0ZXMgKi8KLnN0YXRlLWdsb3d7cG9pbnRlci1ldmVudHM6bm9uZTtmaWxsOm5vbmV9CkBrZXlmcmFtZXMgZ2xvd1B1bHNlezAlLDEwMCV7b3BhY2l0eTowLjEyfTUwJXtvcGFjaXR5OjAuMjJ9fQoKLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDUsNywxMiwwLjk1KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6OXB4OwogIHBhZGRpbmc6MTJweCAxNHB4O29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xMnM7ei1pbmRleDoyMDttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmciAxZnI7Z2FwOjE4cHg7Cn0KLm5hci1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNHB4KTtvdmVyZmxvdzpoaWRkZW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubmMtdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO2NvbG9yOnZhcigtLWluayl9Ci5uYy1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA1ZW07bWFyZ2luLXRvcDoycHh9Ci5uYy1ib2R5e3BhZGRpbmc6MTNweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjB9CgoubW9tLWl0ewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweDsKICBwYWRkaW5nOjdweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLm1vbS1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoubW9tLXJre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO3dpZHRoOjEzcHg7ZmxleC1zaHJpbms6MH0KLm1vbS1pbmZ7ZmxleDoxfQoubW9tLW5te2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQoubW9tLXN0e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9Ci5tb20tcGN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtmb250LXdlaWdodDo0MDA7ZmxleC1zaHJpbms6MH0KLm1vbS1wYy5ye2NvbG9yOnZhcigtLXJpc2UpfQoubW9tLXBjLmZ7Y29sb3I6dmFyKC0tZmFsbCl9Ci5tb20tdHJ7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDttYXJnaW46M3B4IDAgMDtvdmVyZmxvdzpoaWRkZW59Ci5tb20tZmx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHh9CgoucmVnLWl0ewogIGRpc3BsYXk6ZmxleDtnYXA6OXB4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgcGFkZGluZzo4cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2N1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246b3BhY2l0eSAwLjE1czsKfQoucmVnLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5yZWctaXQ6aG92ZXJ7b3BhY2l0eTowLjc1fQoucmVnLWJhZGdlewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4wN2VtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTQpOwogIGNvbG9yOnZhcigtLWFjY2VudCk7ZmxleC1zaHJpbms6MDttYXJnaW4tdG9wOjJweDt3aGl0ZS1zcGFjZTpub3dyYXA7Cn0KLnJlZy1mbHtmbGV4OjE7Zm9udC1zaXplOjExLjVweDtsaW5lLWhlaWdodDoxLjV9Ci5yZWctZnJvbXtjb2xvcjp2YXIoLS1mYWludCl9Ci5yZWctYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7bWFyZ2luOjAgNHB4fQoucmVnLXRve2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQoucmVnLXRte2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7ZmxleC1zaHJpbms6MDttYXJnaW4tdG9wOjJweH0KCi8qIEZBVlMgKi8KLmZhdnN7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzZweCA0MHB4Owp9Ci5mYXZzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLmZhdnMtcm93e2Rpc3BsYXk6ZmxleDtnYXA6MTBweDtvdmVyZmxvdy14OmF1dG87cGFkZGluZy1ib3R0b206M3B4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFye2hlaWdodDoycHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXItdGh1bWJ7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjFweH0KLmZhdi1jYXJkewogIGZsZXg6MCAwIDE5MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTBweDtwYWRkaW5nOjEycHg7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7Cn0KLmZhdi1jYXJkOmhvdmVye2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIyKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDIpfQouZmMtaGVhZHtkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6YmFzZWxpbmU7bWFyZ2luLWJvdHRvbTo3cHh9Ci5mYy1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7Y29sb3I6dmFyKC0taW5rKX0KLmZjLXNje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLmZjLXJvd3tkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6M3B4fQouZmMtcm93IC52e2NvbG9yOnZhcigtLWRpbSk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4fQouZmF2cy1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTppdGFsaWM7cGFkZGluZzo0cHggMH0KCi8qIEZPT1QgKi8KLmZvb3R7dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzo0OHB4IDM2cHggNjBweDttYXgtd2lkdGg6NTgwcHg7bWFyZ2luOjAgYXV0bztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5mb290LW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW4tYm90dG9tOjE0cHh9Ci5mb290LWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEycHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjg7bWFyZ2luLWJvdHRvbToxMnB4fQouZm9vdC1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSl9CgovKiBhbmltYXRpb25zICovCkBrZXlmcmFtZXMgZmFkZVVwe2Zyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVZKDZweCl9dG97b3BhY2l0eToxO3RyYW5zZm9ybTpub25lfX0KLm1hcC1jYXJkLC5zdGF0ZS1wYW5lbCwubmFyLWNhcmQsLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbjpmYWRlVXAgMC41NXMgY3ViaWMtYmV6aWVyKC4yLC44LC4yLDEpIGJhY2t3YXJkc30KLm5hci1jYXJkOm50aC1jaGlsZCgyKXthbmltYXRpb24tZGVsYXk6MC4wN3N9Ci5uYXItY2FyZDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuMTRzfQouc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uLWRlbGF5OjAuMDVzfQoKQG1lZGlhKG1heC13aWR0aDoxMTAwcHgpewogIC5tYWlue2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgLnN0YXRlLXBhbmVse21heC1oZWlnaHQ6bm9uZX0KICAubmFyLXJvd3tncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQp9Cjwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+Cgo8ZGl2IGNsYXNzPSJ0b3BiYXIiPgogIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLW1hcmsiPjxzcGFuIGNsYXNzPSJicmFuZC1wdWxzZS1kb3QiPjwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLXRleHQtYmxvY2siPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtbmFtZSI+PGVtIGNsYXNzPSJicmFuZC1wdWxzZS13b3JkIj5QdWxzZTwvZW0+IG9mIEluZGlhPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtdGFnbGluZSI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3NwYW4+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJ0b3BiYXItciI+CiAgICA8ZGl2IGNsYXNzPSJsaXZlLWluZGljYXRvciI+CiAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gSEVSTyAtLT4KPHNlY3Rpb24gY2xhc3M9Imhlcm8iIHN0eWxlPSJwYWRkaW5nLXRvcDo4MHB4O3BhZGRpbmctYm90dG9tOjI0cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuIj4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTt3aWR0aDo2MDBweDtoZWlnaHQ6MzUwcHg7dG9wOi02MHB4O2xlZnQ6LTgwcHg7YmFja2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSBhdCA0MCUgNTAlLHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLHRyYW5zcGFyZW50IDY1JSk7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7YW5pbWF0aW9uOmFtYmllbnRTaGlmdCAxMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUgYWx0ZXJuYXRlIj48L2Rpdj4KICA8c3R5bGU+QGtleWZyYW1lcyBhbWJpZW50U2hpZnR7MCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMCl9MTAwJXt0cmFuc2Zvcm06dHJhbnNsYXRlWCgyNHB4KSB0cmFuc2xhdGVZKC0xMnB4KX19PC9zdHlsZT4KICA8ZGl2IGNsYXNzPSJoZXJvLWV5ZWJyb3ciIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPkNvbGxlY3RpdmUgYXR0ZW50aW9uICZtaWRkb3Q7IEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1icmFuZC1ibG9jayIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE4cHg7bWFyZ2luLWJvdHRvbToxNnB4O3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MSI+CiAgICA8ZGl2IGNsYXNzPSJoZXJvLXB1bHNlLXNpZ25hbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJocHMtY29yZSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMiI+PC9zcGFuPgogICAgPC9kaXY+CiAgICA8aDEgY2xhc3M9Imhlcm8tYnJhbmQtbmFtZSI+PGVtPlB1bHNlPC9lbT4gb2YgSW5kaWE8L2gxPgogIDwvZGl2PgogIDxwIGNsYXNzPSJoZXJvLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9wPgogIDxwIGNsYXNzPSJoZXJvLWRlc2MiPk9ic2VydmUgaG93IEluZGlhJ3MgbmFycmF0aXZlcyBhbmQgcHVibGljIGF0dGVudGlvbiBzaGlmdCBpbiByZWFsIHRpbWUuPC9wPgogIDxwIGNsYXNzPSJoZXJvLXN1Yi1saW5lIj5PYnNlcnZpbmcgSW5kaWEgaW4gbW90aW9uLjwvcD4KCiAgPCEtLSBMSVZFIFNUQVRTIFNUUklQIC0tPgo8ZGl2IGlkPSJzdGF0cy1zdHJpcCIgc3R5bGU9IgogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MjsKICBiYWNrZ3JvdW5kOnJnYmEoOSwxMywyMSwwLjkpOwogIGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4wOCk7CiAgcGFkZGluZzowIDM2cHg7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7CiI+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIiBpZD0ic2Mtc2lnbmFscyI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+U2lnbmFscyB0cmFja2VkPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1zaWduYWxzLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiPkxpdmUgaW5nZXN0aW9uPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLWhvdHRlc3QiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciIgb25jbGljaz0ic2VsZWN0SG90dGVzdCgpIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5IaWdoZXN0IGF0dGVudGlvbjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPkNsaWNrIHRvIGV4cGxvcmU8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1hbmdlci12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj5PdXRyYWdlICYgcHJvdGVzdCBzaWduYWxzPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIHJpc2luZyBuYXJyYXRpdmU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLW5hcnJhdGl2ZS12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtbmFycmF0aXZlLXN1YiI+TmF0aW9uYWwgc2lnbmFsIHN1cmdlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCBjb29saW5nPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29saW5nLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29saW5nLXN1YiI+U2lnbmFsIGRlY2F5PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPHN0eWxlPgouc3RhdC1jZWxsewogIGZsZXg6MTtwYWRkaW5nOjEwcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2FwOjJweDsKICB0cmFuc2l0aW9uOmJhY2tncm91bmQgMC4xNXM7Cn0KLnN0YXQtY2VsbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci5zdGF0LWRpdnt3aWR0aDoxcHg7YmFja2dyb3VuZDpyZ2JhKDE2MCwxOTAsMjMwLDAuMDcpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luOjhweCAwfQouc2MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO21hcmdpbi10b3A6MXB4fQouc2Mtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQo8L3N0eWxlPgoKCiAgPCEtLSBTSUdOQVRVUkUgSU5TSUdIVCArIE5BUlJBVElWRSBTVFJJUCBzaWRlIGJ5IHNpZGUgLS0+CiAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoxOHB4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7bWFyZ2luLXRvcDoxNnB4O21hcmdpbi1ib3R0b206MDttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbi1sZWZ0OmF1dG87bWFyZ2luLXJpZ2h0OmF1dG87cGFkZGluZzowIDM2cHg7Ij4KICAgIDxkaXYgY2xhc3M9InNpZ25hdHVyZS1pbnNpZ2h0IiBzdHlsZT0ibWFyZ2luLXRvcDowO2ZsZXg6MTttaW4td2lkdGg6MCI+CiAgICAgIDxkaXYgY2xhc3M9InNpLWxhYmVsIj5DdXJyZW50IG5hdGlvbmFsIG5hcnJhdGl2ZSBzaGlmdDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaS10ZXh0IiBpZD0ic2lnLWluc2lnaHQiPjxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHgiPk9ic2VydmluZyBzaWduYWxzLi4uPC9zcGFuPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaS1zdWIiIGlkPSJzaWctdGFncyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+TmFycmF0aXZlIHNoaWZ0czwvc3Bhbj4KICAgICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjJweDsiPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIGFjdGl2ZSIgZGF0YS1wZXJpb2Q9IjNtIj4zTTwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIiBkYXRhLXBlcmlvZD0iNm0iPjZNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSIxeSI+MVk8L2J1dHRvbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDwhLS0gc2hpZnRzIGxpc3QgLS0+CiAgICAgIDxkaXYgc3R5bGU9ImZsZXg6MTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3BhZGRpbmc6MTBweCAxNHB4O2dhcDo2cHg7IiBpZD0ic2hpZnQtbGlzdCI+PC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKCjwhLS0gTUFJTjogTUFQICsgU1RBVEUgUEFORUwgLS0+CjxkaXYgY2xhc3M9Im1haW4iPgoKICA8ZGl2IGNsYXNzPSJtYXAtY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJtYXAtdG9wIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLXRpdGxlLWJsb2NrIj4KICAgICAgICA8ZGl2IGNsYXNzPSJtdCI+SW5kaWEgJm1kYXNoOyBjb2xsZWN0aXZlIGF0dGVudGlvbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1zIiBpZD0ibWFwLW1ldGEiPjMwIHN0YXRlcyAmbWlkZG90OyBsaXZlIHNpZ25hbCBjb21wb3NpdGU8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImxlZ2VuZCI+PHNwYW4+cXVpZXQ8L3NwYW4+PGRpdiBjbGFzcz0ibGVnZW5kLWJhciI+PC9kaXY+PHNwYW4+YWN0aXZlPC9zcGFuPjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJsYXllci1yb3ciPgogICAgICA8c3BhbiBjbGFzcz0ibGF5ZXItbGFiZWwiPlZpZXc8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9Imx0YWJzIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiBhY3RpdmUiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJlbW90aW9uIj5FbW90aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJUaGUgZG9taW5hbnQgZW1vdGlvbmFsIHRvbmUg4oCUIGFueGlvdXMsIGFuZ3J5LCBob3BlZnVsLCBwcm91ZCBvciBmZWFyZnVsLiBSZXZlYWxzIHRoZSBwc3ljaG9sb2dpY2FsIHVuZGVyY3VycmVudCBvZiBwb2xpdGljYWwgYXR0ZW50aW9uLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9InZlbG9jaXR5Ij5Nb21lbnR1bSA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iSXMgYXR0ZW50aW9uIHJpc2luZyBvciBmYWxsaW5nPyBSaXNpbmcgPSBuYXJyYXRpdmUgYWNjZWxlcmF0aW5nLiBDb29saW5nID0gbG9zaW5nIHRyYWN0aW9uLiBTaG93cyBzdGF0ZXMgZW50ZXJpbmcgb3IgZXhpdGluZyBhIHBvbGl0aWNhbCBjeWNsZS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLXN2Zy13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLWlubmVyIj4KICAgICAgICA8c3ZnIGlkPSJpbmRpYS1tYXAiIHZpZXdCb3g9IjAgMCA4MDAgODAwIiBwcmVzZXJ2ZUFzcGVjdFJhdGlvPSJ4TWlkWU1pZCBtZWV0Ij4KICAgICAgICAgIDxkZWZzPgogICAgICAgICAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImFtYkdsb3ciIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjUwJSI+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0icmdiYSgyMjQsOTAsNDAsMC4wNCkiLz4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgICAgICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICAgICAgICAgIDxmaWx0ZXIgaWQ9InN0YXRlR2xvdyIgeD0iLTMwJSIgeT0iLTMwJSIgd2lkdGg9IjE2MCUiIGhlaWdodD0iMTYwJSI+CiAgICAgICAgICAgICAgPGZlR2F1c3NpYW5CbHVyIGluPSJTb3VyY2VHcmFwaGljIiBzdGREZXZpYXRpb249IjgiIHJlc3VsdD0iYmx1ciIvPgogICAgICAgICAgICAgIDxmZUNvbXBvc2l0ZSBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJibHVyIiBvcGVyYXRvcj0ib3ZlciIvPgogICAgICAgICAgICA8L2ZpbHRlcj4KICAgICAgICAgIDwvZGVmcz4KICAgICAgICAgIDxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJ1cmwoI2FtYkdsb3cpIi8+CiAgICAgICAgICA8ZyBpZD0ibWFwLWdsb3ciPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtc3RhdGVzIj48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXB1bHNlcyI+PC9nPgogICAgICAgIDwvc3ZnPgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC10b29sdGlwIiBpZD0idG9vbHRpcCI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gU1RBVEUgUEFORUwgLS0+CiAgPGRpdiBjbGFzcz0ic3RhdGUtcGFuZWwiIGlkPSJzdGF0ZS1kZXRhaWwiPgogICAgPGRpdiBjbGFzcz0icGFuZWwtZW1wdHkiPgogICAgICA8c3ZnIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxIj4KICAgICAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiA4djRNMTIgMTZoLjAxIi8+CiAgICAgIDwvc3ZnPgogICAgICA8ZGl2IGNsYXNzPSJwZS10Ij5TZWxlY3QgYSBzdGF0ZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJwZS1zIj5DbGljayBhbnkgcmVnaW9uIG9uIHRoZSBtYXA8YnIvPnRvIG9wZW4gaXRzIG5hcnJhdGl2ZSBwYW5lbC48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKPC9kaXY+Cgo8IS0tIE5BUlJBVElWRSBST1cgLS0+CjxkaXYgY2xhc3M9Im5hci1yb3ciPgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJpc2luZyBuYXJyYXRpdmVzPC9zcGFuPjwvZGl2PgogICAgPGRpdiBpZD0icmlzaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+PHNwYW4gY2xhc3M9Im5jLWRvdCBmYWxsIj48L3NwYW4+PHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9ImRlY2xpbmluZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJlZ2lvbmFsIHNoaWZ0czwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9InJlZ2lvbmFsLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSBSRVBMQVkgSU5ESUEgLS0+CjxzZWN0aW9uIGNsYXNzPSJyZXBsYXktc2VjdGlvbiI+CiAgPGRpdiBjbGFzcz0icmVwbGF5LWhlYWRlciI+CiAgICA8ZGl2PjxkaXYgY2xhc3M9InJlcGxheS1sYWJlbCI+UmVwbGF5IEluZGlhPC9kaXY+PGRpdiBjbGFzcz0icmVwbGF5LXN1YiI+V2F0Y2ggaG93IGNvbGxlY3RpdmUgYXR0ZW50aW9uIHNoaWZ0ZWQgb3ZlciB0aW1lPC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyZXBsYXktY29udHJvbHMiPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4gYWN0aXZlIiBkYXRhLXBlcmlvZD0iN2QiPjcgZGF5czwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4iIGRhdGEtcGVyaW9kPSIzMGQiPjMwIGRheXM8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIiBkYXRhLXBlcmlvZD0iNm0iPjYgbW9udGhzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biIgZGF0YS1wZXJpb2Q9ImVsZWN0aW9uIj5FbGVjdGlvbiAyMDI0PC9idXR0b24+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktc2NydWJiZXIiPgogICAgPGRpdiBjbGFzcz0icnAtdHJhY2siIGlkPSJycC10cmFjayI+PGRpdiBjbGFzcz0icnAtZmlsbCIgaWQ9InJwLWZpbGwiPjwvZGl2PjxkaXYgY2xhc3M9InJwLXRodW1iIiBpZD0icnAtdGh1bWIiPjwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0icnAtZGF0ZXMiIGlkPSJycC1kYXRlcyI+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0icmVwbGF5LXBsYXliYWNrIj4KICAgIDxidXR0b24gY2xhc3M9InJwLXBsYXkiIGlkPSJycC1wbGF5LWJ0biIgb25jbGljaz0idG9nZ2xlUmVwbGF5KCkiPgogICAgICA8c3ZnIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwb2x5Z29uIHBvaW50cz0iMiwxIDksNSAyLDkiIGlkPSJycC1wbGF5LWljb24iLz48L3N2Zz4KICAgIDwvYnV0dG9uPgogICAgPGRpdiBjbGFzcz0icnAtY3VycmVudC1kYXRlIiBpZD0icnAtY3VycmVudC1kYXRlIj5TZWxlY3QgYSBwZXJpb2QgYW5kIHByZXNzIHBsYXk8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJwLXNwZWVkIj48c3BhbiBjbGFzcz0icnAtc3BlZWQtbGFiZWwiPlNwZWVkPC9zcGFuPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQgYWN0aXZlIiBkYXRhLXNwZD0iMSI+MXg8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iMiI+Mng8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iNCI+NHg8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InJlcGxheS1zbmFwc2hvdCI+PGRpdiBjbGFzcz0icnAtc25hcC1sYWJlbCI+TmFycmF0aXZlIHNuYXBzaG90IGF0IHRoaXMgbW9tZW50PC9kaXY+PGRpdiBjbGFzcz0icnAtc25hcC1zdGF0ZXMiIGlkPSJycC1zbmFwLXN0YXRlcyI+PGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5QcmVzcyBwbGF5IHRvIG9ic2VydmUgSW5kaWEgaW4gbW90aW9uLjwvZGl2PjwvZGl2PjwvZGl2Pgo8L3NlY3Rpb24+CjxzdHlsZT4KLnJlcGxheS1zZWN0aW9ue3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggMzZweH0KLnJlcGxheS1oZWFkZXJ7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtZW5kO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206MjBweDtnYXA6MjBweDtmbGV4LXdyYXA6d3JhcH0KLnJlcGxheS1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIwcHg7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWluayk7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLnJlcGxheS1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjRweH0KLnJlcGxheS1jb250cm9sc3tkaXNwbGF5OmZsZXg7Z2FwOjRweDtmbGV4LXdyYXA6d3JhcH0KLnJwLWJ0bntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO3BhZGRpbmc6NXB4IDEycHg7Ym9yZGVyLXJhZGl1czo0cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Y29sb3I6dmFyKC0tZmFpbnQpO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQoucnAtYnRuLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5yZXBsYXktc2NydWJiZXJ7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MThweCAyMHB4IDE0cHg7bWFyZ2luLWJvdHRvbToxMnB4fQoucnAtdHJhY2t7cG9zaXRpb246cmVsYXRpdmU7aGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoycHg7Y3Vyc29yOnBvaW50ZXI7bWFyZ2luLWJvdHRvbToxMHB4fQoucnAtZmlsbHtwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MCU7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQscmdiYSgyMjQsOTAsNDAsMC40KSx2YXIoLS1hY2NlbnQpKTtib3JkZXItcmFkaXVzOjJweH0KLnJwLXRodW1ie3Bvc2l0aW9uOmFic29sdXRlO3RvcDo1MCU7dHJhbnNmb3JtOnRyYW5zbGF0ZSgtNTAlLC01MCUpO3dpZHRoOjEycHg7aGVpZ2h0OjEycHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JvcmRlcjoycHggc29saWQgcmdiYSg5LDEzLDIxLDAuOCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCk7bGVmdDowJTtjdXJzb3I6Z3JhYn0KLnJwLWRhdGVze2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJlcGxheS1wbGF5YmFja3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNHB4O21hcmdpbi1ib3R0b206MTZweH0KLnJwLXBsYXl7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjI1KTtjb2xvcjp2YXIoLS1hY2NlbnQpO2N1cnNvcjpwb2ludGVyO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE1c30KLnJwLWN1cnJlbnQtZGF0ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO2ZsZXg6MX0KLnJwLXNwZWVke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweH0KLnJwLXNwZWVkLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1yaWdodDoycHh9Ci5ycC1zcGR7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O3BhZGRpbmc6M3B4IDhweDtib3JkZXItcmFkaXVzOjNweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDp0cmFuc3BhcmVudDtjb2xvcjp2YXIoLS1mYWludCk7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xNXN9Ci5ycC1zcGQuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIpfQoucmVwbGF5LXNuYXBzaG90e2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTJweDtwYWRkaW5nOjE2cHggMjBweH0KLnJwLXNuYXAtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEycHh9Ci5ycC1zbmFwLXN0YXRlc3tkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjhweH0KLnJwLWxvZy1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSk7Zm9udC1zdHlsZTppdGFsaWM7cGFkZGluZzo0cHggMH0KLnJwLXN0YXRlLWNhcmR7cGFkZGluZzo4cHggMTJweDtib3JkZXItcmFkaXVzOjZweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO21pbi13aWR0aDoxNDBweH0KLnJwLXN0YXRlLW5hbWV7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bWFyZ2luLWJvdHRvbTozcHh9Ci5ycC1zdGF0ZS1uYXJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJwLXN0YXRlLWF0dHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCl9Cjwvc3R5bGU+CjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIE9ic2VydmF0aW9uLjwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICBpZighcm93c3x8IXJvd3MubGVuZ3RoKSByZXR1cm47CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocm93LmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb21FbW89cm93LmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihlbW9zKXx8bnVsbDsKICAgICAgdmFyIGVudHJ5PXthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbUVtbyxkb21pbmFudF9uYXJyYXRpdmU6cm93LmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgICAgTElWRVtyb3cubmFtZV09ZW50cnk7CiAgICAgIGlmKCFTRFtyb3cubmFtZV0pIFNEW3Jvdy5uYW1lXT1PYmplY3QuYXNzaWduKHt9LERFRkFVTFQpOwogICAgICBPYmplY3QuYXNzaWduKFNEW3Jvdy5uYW1lXSxlbnRyeSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgaWYoU0VMJiZMSVZFW1NFTF0mJmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKSkgcmVuZGVyUGFuZWwoU0VMKTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbQVBJXScsZS5tZXNzYWdlKTt9Cn0KCmZ1bmN0aW9uIHVwZGF0ZUFsbFN0cmlwcygpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBob3R0ZXN0PWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCk+KGFbMV0uYXR0ZW50aW9ufHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsaG90dGVzdFswXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCdBdHRlbnRpb24gJytob3R0ZXN0WzFdLmF0dGVudGlvbi50b0ZpeGVkKDEpKTsKICB2YXIgdG9wQW5nZXJObT1udWxsLHRvcEFuZ2VyUGN0PTA7CiAgZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXt2YXIgZT1rdlsxXS5lbW90aW9uc3x8e30sYT1lLmFuZ2VyfHwwO2lmKGE+MCYmYTw9MSlhPWEqMTAwO2lmKGE+dG9wQW5nZXJQY3Qpe3RvcEFuZ2VyUGN0PWE7dG9wQW5nZXJObT1rdlswXTt9fSk7CiAgaWYodG9wQW5nZXJObSl7c2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlck5tKTtzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdBbmdlciAnK01hdGgucm91bmQodG9wQW5nZXJQY3QpKyclIG9mIHNpZ25hbHMnKTt9CiAgdmFyIGNvb2xpbmc9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0udmVsb2NpdHl8fDApPChhWzFdLnZlbG9jaXR5fHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1jb29saW5nLXZhbCcsY29vbGluZ1swXSk7c2V0VGV4dCgnc2MtY29vbGluZy1zdWInLCdWZWxvY2l0eSAnK2Nvb2xpbmdbMV0udmVsb2NpdHkudG9GaXhlZCgzKSk7CiAgdmFyIG5jPXt9O2VudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlKW5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV09KG5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV18fDApKzE7fSk7CiAgdmFyIHRuPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXTsKICBpZih0bil7c2V0VGV4dCgnc2MtbmFycmF0aXZlLXZhbCcsdG5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdG5bMF0uc2xpY2UoMSkpO3NldFRleHQoJ3NjLW5hcnJhdGl2ZS1zdWInLCdEb21pbmFudCBhY3Jvc3MgJyt0blsxXSsnIHN0YXRlcycpO30KfQphc3luYyBmdW5jdGlvbiBmZXRjaERldGFpbChuYW1lKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChuYW1lKSk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhkLmVtb3Rpb25zfHx7fSk7CiAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8ZC5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgU0RbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGVtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlLAogICAgICBuYXJyYXRpdmVzOihkLm5hcnJhdGl2ZXN8fFtdKS5tYXAoZnVuY3Rpb24obil7cmV0dXJue25hbWU6bi5uYW1lLHZhbDpuLnZhbCxkaXI6bi5kaXJ8fCdmbGF0J307fSksCiAgICAgIHJpc2luZzpkLnJpc2luZ3x8W10sZmFsbGluZzpkLmZhbGxpbmd8fFtdLHN1bW1hcnk6ZC5zdW1tYXJ5fHxERUZBVUxULnN1bW1hcnksCiAgICAgIGFydGljbGVzOmQuYXJ0aWNsZXN8fFtdLHRpbWVsaW5lOmQudGltZWxpbmV8fERFRkFVTFQudGltZWxpbmUsCiAgICAgIG5hcnJhdGl2ZUhpc3Rvcnk6ZC5uYXJyYXRpdmVIaXN0b3J5fHxERUZBVUxULm5hcnJhdGl2ZUhpc3Rvcnksc2lnbmFsX2NvdW50OmQuc2lnbmFsX2NvdW50fHwwfTsKICAgIGlmKCFMSVZFW25hbWVdKUxJVkVbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZX07CiAgICBMSVZFW25hbWVdLmVtb3Rpb25zPWVtb3M7TElWRVtuYW1lXS5kb21pbmFudF9lbW90aW9uPWRvbTsKICAgIHJldHVybiBTRFtuYW1lXTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbZmV0Y2hEZXRhaWxdJyxuYW1lLGUubWVzc2FnZSk7cmV0dXJuIFNEW25hbWVdfHxERUZBVUxUO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTbmFwKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc25hcHNob3QvZGFpbHknKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIC8vIHRvcGJhcgogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7CiAgICBpZihlbCYmZC50b3RhbF9zaWduYWxzKSBlbC50ZXh0Q29udGVudD1kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTsKICAgIHZhciBtZXRhPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtbWV0YScpOwogICAgaWYobWV0YSYmZC5hc19vZikgbWV0YS50ZXh0Q29udGVudD0nMzAgc3RhdGVzIMK3IHVwZGF0ZWQgJytuZXcgRGF0ZShkLmFzX29mKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyk7CiAgICAvLyBzdGF0cyBzdHJpcAogICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLCBkLnRvdGFsX3NpZ25hbHM/ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk6Jy0nKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogIH1jYXRjaChlKXt9Cn0KCmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKGVsKWVsLnRleHRDb250ZW50PXZhbDt9CgpmdW5jdGlvbiB1cGRhdGVTdHJpcE5hcnJhdGl2ZSgpe3VwZGF0ZUFsbFN0cmlwcygpO30KZnVuY3Rpb24gdXBkYXRlU3RyaXBBbmdlcigpe30KCmZ1bmN0aW9uIHNlbGVjdEhvdHRlc3QoKXsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKFNEKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApIHNlbGVjdF8odG9wWzBdKTsKfQphc3luYyBmdW5jdGlvbiBmZXRjaEluc2lnaHRzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvaW5zaWdodHMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIHZhciBzaWc9ZC5zaWduYXR1cmU7CiAgICBpZihzaWcpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0iSW5kaWEncyBhdHRlbnRpb24gaXMgc2hpZnRpbmcgZnJvbSA8ZW0+IitzaWcuZmFkaW5nKyI8L2VtPiB0b3dhcmQgPGVtPiIrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYW5kIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIuICIrc2lnLmhvdHRlc3Rfc3RhdGUrIiBsZWFkcyBuYXRpb25hbCBhdHRlbnRpb24uIjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLnRhZ3MpdEVsLmlubmVySFRNTD1kLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICB9CiAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgaWYockVsJiZkLnJpc2luZyYmZC5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6I2UwNWEyOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICBpZihmRWwmJmQuZmFsbGluZyYmZC5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrbi5zdGF0ZXMuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMykrJyU7YmFja2dyb3VuZDojM2JiOGQ4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogICAgaWYoZ0VsJiZkLnJlZ2lvbmFsJiZkLnJlZ2lvbmFsLmxlbmd0aClnRWwuaW5uZXJIVE1MPWQucmVnaW9uYWwubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbiI+PHNwYW4gY2xhc3M9Im5pLW5hbWUiPicrci5yZWdpb24rJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3IuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+PGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK3IuaG90dGVzdF9zdGF0ZSsnIMK3ICcrci50b3BfbmFycmF0aXZlKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUubWVzc2FnZSk7fQp9Cgphc3luYyBmdW5jdGlvbiBzdGFydFBvbGxpbmcoKXsKICBhd2FpdCBQcm9taXNlLmFsbChbZmV0Y2hBbGxTdGF0ZXMoKSxmZXRjaFNuYXAoKV0pOwogIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbihlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUpO30pOwogIHZhciBuPTA7CiAgdmFyIHQ9c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXsKICAgIG4rKzthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpOwogICAgaWYoU0VMKSByZW5kZXJQYW5lbChTRUwpOwogICAgaWYobj49MTIpe2NsZWFySW50ZXJ2YWwodCk7c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpO2lmKFNFTClyZW5kZXJQYW5lbChTRUwpO30sMTIwMDAwKTsKICAgICAgc2V0SW50ZXJ2YWwoZmV0Y2hJbnNpZ2h0cywzNjAwMDAwKTt9CiAgfSwxNTAwMCk7Cn0KCi8vIE5BUlJBVElWRSBEQVRBCnZhciBTSElGVFM9ewogICczbSc6WwogICAge2ZhZGluZzonSW5mbGF0aW9uJyxmYWRpbmdOb3RlOidlYXNpbmcgbmF0aW9uYWxseScscmlzaW5nOidCb3JkZXIgc2VjdXJpdHknLHJpc2luZ05vdGU6J3Bvc3QtaW5jaWRlbnQgc3VyZ2UnfSwKICAgIHtmYWRpbmc6J0VsZWN0aW9uIHJoZXRvcmljJyxmYWRpbmdOb3RlOidwb3N0LWN5Y2xlIGZhZGUnLHJpc2luZzonR292ZXJuYW5jZSBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZTonc3RlYWR5IHJpc2UnfSwKICAgIHtmYWRpbmc6J0Zhcm1lciBwcm90ZXN0cycsZmFkaW5nTm90ZTonbW9tZW50dW0gbG9zdCcscmlzaW5nOidVbmVtcGxveW1lbnQgYW54aWV0eScscmlzaW5nTm90ZToneW91dGggc2lnbmFsIHN1cmdlJ30sCiAgXSwKICAnNm0nOlsKICAgIHtmYWRpbmc6J0Nhc3RlIG1vYmlsaXNhdGlvbicsZmFkaW5nTm90ZToncHJlLWVsZWN0aW9uIHBlYWsnLHJpc2luZzonQ29ycnVwdGlvbiBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZToncG9zdC1jeWNsZSBwdXNoJ30sCiAgICB7ZmFkaW5nOidSZWxpZ2lvdXMgbmF0aW9uYWxpc20nLGZhZGluZ05vdGU6J3BsYXRlYXUgcGhhc2UnLHJpc2luZzonRWNvbm9taWMgYW54aWV0eScscmlzaW5nTm90ZTonY29zdC1vZi1saXZpbmcnfSwKICAgIHtmYWRpbmc6J0luZnJhc3RydWN0dXJlIHByaWRlJyxmYWRpbmdOb3RlOidyaWJib24tY3V0dGluZyBkb25lJyxyaXNpbmc6J0xhdyAmIG9yZGVyJyxyaXNpbmdOb3RlOidjcmltZSBuYXJyYXRpdmUgcmlzZSd9LAogIF0sCiAgJzF5JzpbCiAgICB7ZmFkaW5nOidQYW5kZW1pYyByZWNvdmVyeScsZmFkaW5nTm90ZTonZmFkZWQgZWFybHkgeWVhcicscmlzaW5nOidJbmZsYXRpb24nLHJpc2luZ05vdGU6J2RvbWluYXRlZCBtaWQteWVhcid9LAogICAge2ZhZGluZzonUmVnaW9uYWwgaWRlbnRpdHknLGZhZGluZ05vdGU6J2xhbmd1YWdlLWxlZCBwZWFrJyxyaXNpbmc6J1NlY3VyaXR5ICYgYm9yZGVycycscmlzaW5nTm90ZTonZ2VvcG9saXRpY2FsIGVzY2FsYXRpb24nfSwKICAgIHtmYWRpbmc6J0dvdmVybmFuY2Ugb3B0aW1pc20nLGZhZGluZ05vdGU6J3BvbGljeSBob25leW1vb24gZW5kJyxyaXNpbmc6J0NvcnJ1cHRpb24gJiBzY2FtcycscmlzaW5nTm90ZTonYWNjb3VudGFiaWxpdHkgY3ljbGUnfSwKICBdLAp9Owp2YXIgUkVHX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidCaWhhcicsZnJvbTonRWxlY3Rpb24gcmhldG9yaWMnLHRvOidVbmVtcGxveW1lbnQgJiBleGFtIHNjYW1zJyx0aW1lOic2IHdrcyd9LAogIHtzdGF0ZTonV2VzdCBCZW5nYWwnLGZyb206J0J5cG9sbCBwb2xpdGljcycsdG86J0xhdyAmIG9yZGVyIMK3IEJvcmRlcicsdGltZTonNCB3a3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdrcyd9LAogIHtzdGF0ZTonS2FybmF0YWthJyxmcm9tOidNaW5pbmcgY29udHJvdmVyc3knLHRvOidMYW5ndWFnZSBzaWduYWdlIHBvbGl0aWNzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonRGVsaGknLGZyb206J01ldHJvIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3a3MnfSwKICB7c3RhdGU6J1B1bmphYicsZnJvbTonUG93ZXIgY3Jpc2lzJyx0bzonQm9yZGVyIHNlY3VyaXR5IMK3IERyb25lcycsdGltZTonMyB3a3MnfSwKXTsKdmFyIE1PQ0tfUj1bCiAge25hbWU6J0JvcmRlciBzZWN1cml0eScsc3RhdGVzOidKJksgwrcgUHVuamFiIMK3IFJhamFzdGhhbicscGN0OicrNDElJ30sCiAge25hbWU6J1VuZW1wbG95bWVudCcsc3RhdGVzOidCaWhhciDCtyBVUCDCtyBKaGFya2hhbmQnLHBjdDonKzI4JSd9LAogIHtuYW1lOidMYW5ndWFnZSBwb2xpdGljcycsc3RhdGVzOidUTiDCtyBLYXJuYXRha2EgwrcgTUgnLHBjdDonKzIyJSd9LAogIHtuYW1lOidFbnZpcm9ubWVudGFsIGNyaXNpcycsc3RhdGVzOidEZWxoaSDCtyBSYWphc3RoYW4gwrcgQVAnLHBjdDonKzE5JSd9LAogIHtuYW1lOidFdGhuaWMgdGVuc2lvbnMnLHN0YXRlczonTWFuaXB1ciDCtyBBc3NhbSDCtyBXQicscGN0OicrMTclJ30sCl07CnZhciBNT0NLX0Y9WwogIHtuYW1lOidFbGVjdGlvbiByaGV0b3JpYycsc3RhdGVzOidOYXRpb25hbCBwb3N0LWN5Y2xlJyxwY3Q6Jy0zOCUnfSwKICB7bmFtZTonSW5mbGF0aW9uIHByZXNzdXJlJyxzdGF0ZXM6J0Vhc2luZyBuYXRpb25hbGx5JyxwY3Q6Jy0yNCUnfSwKICB7bmFtZTonRmFybWVyIHByb3Rlc3RzJyxzdGF0ZXM6J01vbWVudHVtIGxvc3QnLHBjdDonLTE5JSd9LAogIHtuYW1lOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsc3RhdGVzOidSaWJib24tY3V0dGluZyBkb25lJyxwY3Q6Jy0xNCUnfSwKICB7bmFtZTonUmVsaWdpb3VzIGZlc3RpdmFscycsc3RhdGVzOidQb3N0LXNlYXNvbiBmYWRlJyxwY3Q6Jy0xMSUnfSwKXTsKCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGRhdGE9U0hJRlRTW3BlcmlvZF18fFNISUZUU1snM20nXTsKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NoaWZ0LWxpc3QnKTsKICBpZighZWwpIHJldHVybjsKICBlbC5pbm5lckhUTUw9ZGF0YS5tYXAoZnVuY3Rpb24ocyl7CiAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjA7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjhweDtvdmVyZmxvdzpoaWRkZW47Ij4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6NnB4IDEwcHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFsbCk7bWFyZ2luLWJvdHRvbTozcHg7Ij5mYWRpbmc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjI7Ij4nK3MuZmFkaW5nKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytzLmZhZGluZ05vdGUrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0id2lkdGg6MjhweDtmbGV4LXNocmluazowO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC40NTtmb250LXNpemU6MTNweDsiPuKGkjwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo4cHggMTBweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLXJpc2UpO21hcmdpbi1ib3R0b206M3B4OyI+cmlzaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLnJpc2luZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5yaXNpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHRhYil7CiAgdGFiLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odCl7dC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIHRhYi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtyZW5kZXJTdHJpcCh0YWIuZGF0YXNldC5wZXJpb2QpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHJlbmRlck1vbWVudHVtKCl7CiAgLy8gUmVhZCBmcm9tIFNEIChwb3B1bGF0ZWQgYnkgZmV0Y2hBbGxTdGF0ZXMgZnJvbSBsaXZlIEFQSSkKICB2YXIgbmM9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBuY1tuLm5hbWVdPShuY1tuLm5hbWVdfHwwKStuLnZhbDsKICAgIH0pOwogIH0pOwogIHZhciBzb3J0ZWQ9T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogIHZhciByaXNpbmc9c29ydGVkLnNsaWNlKDAsNSk7CiAgdmFyIGZhbGxpbmc9c29ydGVkLnNsaWNlKC01KS5yZXZlcnNlKCk7CiAgdmFyIG14PXJpc2luZy5sZW5ndGg/cmlzaW5nWzBdWzFdOjEwMDsKCiAgLy8gV3JpdGUgdG8gcmlzaW5nLWxpc3QgKG1hdGNoZXMgbmFyLXJvdyBIVE1MKQogIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgaWYockVsJiZyaXNpbmcubGVuZ3RoKXsKICAgIHJFbC5pbm5lckhUTUw9cmlzaW5nLm1hcChmdW5jdGlvbihuKXsKICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibmktbmFtZSI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsblsxXS9teCoxMDApKyclO2JhY2tncm91bmQ6I2UwNWEyOCI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIGRlY2xpbmluZy1saXN0CiAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICBpZihmRWwmJmZhbGxpbmcubGVuZ3RoKXsKICAgIGZFbC5pbm5lckhUTUw9ZmFsbGluZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im5pLW5hbWUiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKSsnJTtiYWNrZ3JvdW5kOiMzYmI4ZDgiPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byByZWdpb25hbC1saXN0IOKAlCB0b3Agc3RhdGUgcGVyIHJlZ2lvbiBmcm9tIExJVkUKICB2YXIgcmVnaW9ucz17CiAgICAnTm9ydGgnOlsnRGVsaGknLCdVdHRhciBQcmFkZXNoJywnUHVuamFiJywnSGFyeWFuYScsJ0hpbWFjaGFsIFByYWRlc2gnLCdVdHRhcmFraGFuZCcsJ0phbW11IGFuZCBLYXNobWlyJ10sCiAgICAnRWFzdCc6WydXZXN0IEJlbmdhbCcsJ0JpaGFyJywnSmhhcmtoYW5kJywnT2Rpc2hhJ10sCiAgICAnV2VzdCc6WydNYWhhcmFzaHRyYScsJ0d1amFyYXQnLCdSYWphc3RoYW4nLCdHb2EnXSwKICAgICdTb3V0aCc6WydUYW1pbCBOYWR1JywnS2FybmF0YWthJywnS2VyYWxhJywnQW5kaHJhIFByYWRlc2gnLCdUZWxhbmdhbmEnXSwKICAgICdORSc6WydBc3NhbScsJ01hbmlwdXInLCdOYWdhbGFuZCcsJ01pem9yYW0nLCdNZWdoYWxheWEnLCdUcmlwdXJhJywnQXJ1bmFjaGFsIFByYWRlc2gnLCdTaWtraW0nXSwKICAgICdDZW50cmFsJzpbJ01hZGh5YSBQcmFkZXNoJywnQ2hoYXR0aXNnYXJoJ10sCiAgfTsKICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgaWYoZ0VsKXsKICAgIHZhciByZWdJdGVtcz1PYmplY3QuZW50cmllcyhyZWdpb25zKS5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgcmVnaW9uPWt2WzBdLHN0YXRlcz1rdlsxXTsKICAgICAgdmFyIHRvcD1zdGF0ZXMubWFwKGZ1bmN0aW9uKHMpe3JldHVybiB7bmFtZTpzLGF0dDooTElWRVtzXSYmTElWRVtzXS5hdHRlbnRpb24pfHwwfTt9KQogICAgICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pWzBdOwogICAgICBpZighdG9wfHwhdG9wLmF0dCkgcmV0dXJuIG51bGw7CiAgICAgIHZhciBuYXI9KExJVkVbdG9wLm5hbWVdJiZMSVZFW3RvcC5uYW1lXS5kb21pbmFudF9uYXJyYXRpdmUpfHwn4oCUJzsKICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmNlbnRlciI+JysKICAgICAgICAgICc8c3BhbiBjbGFzcz0ibmktbmFtZSI+JytyZWdpb24rJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrdG9wLmF0dC50b0ZpeGVkKDEpKyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK3RvcC5uYW1lKycgwrcgJytuYXIrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuZmlsdGVyKEJvb2xlYW4pLmpvaW4oJycpOwogICAgaWYocmVnSXRlbXMpIGdFbC5pbm5lckhUTUw9cmVnSXRlbXM7CiAgfQp9CgoKLy8gU1RBVEUgREFUQQp2YXIgU0Q9e307Cgp2YXIgTElWRT17fTsKZnVuY3Rpb24gbm9ybWFsaXplRW1vdGlvbnMoZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJue307dmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhlKSx0b3Q9dmFscy5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcyt2O30sMCk7aWYodG90PD0wKXJldHVybnt9O2lmKHRvdDw9MS4wMSl7dmFyIG91dD17fTtPYmplY3Qua2V5cyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGspe291dFtrXT1NYXRoLnJvdW5kKGVba10qMTAwKTt9KTtyZXR1cm4gb3V0O31yZXR1cm4gZTt9CmZ1bmN0aW9uIGRvbWluYW50RW1vdGlvbihlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm4gbnVsbDt2YXIgbXg9MCxkb209bnVsbDtPYmplY3QuZW50cmllcyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXT5teCl7bXg9a3ZbMV07ZG9tPWt2WzBdO319KTtyZXR1cm4gZG9tO30KZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIWVsKXJldHVybjtlbC50ZXh0Q29udGVudD12YWw7aWYodmFsJiZ2YWwhPT0nLScpe2VsLmNsYXNzTGlzdC5yZW1vdmUoJ2xvYWRpbmcnKTt9fQoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjowLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7fSxkb21pbmFudF9lbW90aW9uOm51bGwsZG9taW5hbnRfbmFycmF0aXZlOm51bGwsCiAgbmFycmF0aXZlczpbXSxyaXNpbmc6W10sZmFsbGluZzpbXSwKICBzdW1tYXJ5OicnLGFydGljbGVzOltdLHRpbWVsaW5lOltdLAogIG5hcnJhdGl2ZUhpc3Rvcnk6W10sc2lnbmFsX2NvdW50OjAsCn07CgpmdW5jdGlvbiBnKG4pe3JldHVybiBTRFtuXXx8T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTt9CgpmdW5jdGlvbiBhQyhzKXsKICAvLyBEeW5hbWljIHNjYWxlOiBhbHdheXMgc3ByZWFkIGZ1bGwgY29sb3IgcmFuZ2UgYWNyb3NzIGFjdHVhbCBkYXRhCiAgLy8gR2V0IG1pbi9tYXggZnJvbSBjdXJyZW50IFNEIHRvIG5vcm1hbGl6ZQogIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKGQpe3JldHVybiBkLmF0dGVudGlvbnx8MDt9KTsKICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpOwogIHZhciBteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgLy8gTm9ybWFsaXplIDAtMQogIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwocy1tbikvKG14LW1uKSkpOwogIC8vIE1hcCB0byBjb2xvciBzdG9wczogZGFyayBibHVlIOKGkiB0ZWFsIOKGkiBhbWJlciDihpIgb3JhbmdlIOKGkiByZWQKICBpZihuPDAuMTIpIHJldHVybiAnIzBkMWUzMCc7CiAgaWYobjwwLjI1KSByZXR1cm4gJyMwZTNkNmEnOwogIGlmKG48MC4zOCkgcmV0dXJuICcjMGQ1ZjkwJzsKICBpZihuPDAuNTApIHJldHVybiAnIzBlN2FhYSc7CiAgaWYobjwwLjYyKSByZXR1cm4gJyMxYTkwOTAnOwogIGlmKG48MC43MikgcmV0dXJuICcjYzg3MDEwJzsKICBpZihuPDAuODIpIHJldHVybiAnI2Q4NDAxMCc7CiAgaWYobjwwLjkyKSByZXR1cm4gJyNjYzE4MDgnOwogIHJldHVybiAnI2ZmMDAxMCc7Cn0KZnVuY3Rpb24gZUMoZSl7CiAgdmFyIG14PTAsZG9tPSdwcmlkZSc7CiAgZm9yKHZhciBrIGluIGUpe2lmKGVba10+bXgpe214PWVba107ZG9tPWs7fX0KICByZXR1cm4gKHthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfSlbZG9tXXx8JyMzM2FhY2MnOwp9CmZ1bmN0aW9uIHZDKHYpewogIGlmKHY+MC4yKSByZXR1cm4gJyNkYzA4MTgnOwogIGlmKHY+MC4xKSByZXR1cm4gJyNlMDVhMjgnOwogIGlmKHY+MC4wMikgcmV0dXJuICcjY2M4ODIyJzsKICBpZih2PC0wLjA1KSByZXR1cm4gJyMyMjk5YmInOwogIHJldHVybiAnIzE1MjAzMCc7Cn0KCnZhciBsYXllcj0nYXR0ZW50aW9uJyxTRUw9bnVsbCxGQVZTPW5ldyBTZXQoKTsKCi8vIE1BUApmdW5jdGlvbiBwcm9qXyh3LGgscGFkKXsKICBwYWQ9cGFkfHwyMDsKICB2YXIgbWluTG9uPTY4LjEsbWF4TG9uPTk3LjQsbWluTGF0PTYuNSxtYXhMYXQ9MzcuMTsKICB2YXIgc2NYPSh3LXBhZCoyKS8obWF4TG9uLW1pbkxvbik7CiAgdmFyIHNjWT0oaC1wYWQqMikvKG1heExhdC1taW5MYXQpOwogIHZhciBzYz1NYXRoLm1pbihzY1gsc2NZKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLShtYXhMb24tbWluTG9uKSpzYykvMjsKICB2YXIgb3k9cGFkKyhoLXBhZCoyLShtYXhMYXQtbWluTGF0KSpzYykvMjsKICByZXR1cm4gZnVuY3Rpb24obG9uLGxhdCl7cmV0dXJuIFtveCsobG9uLW1pbkxvbikqc2MsIG95KyhtYXhMYXQtbGF0KSpzY107fTsKfQpmdW5jdGlvbiBnZW8ycGF0aChnZW9tLHBqKXsKICB2YXIgZD0nJzsKICBmdW5jdGlvbiByaW5nKGNzKXt2YXIgcz0nJztjcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cGooY1swXSxjWzFdKTtzKz0oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSk7cmV0dXJuIHMrJ1onO30KICBpZihnZW9tLnR5cGU9PT0nUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pOwogIGVsc2UgaWYoZ2VvbS50eXBlPT09J011bHRpUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihwKXtwLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTt9KTsKICByZXR1cm4gZDsKfQpmdW5jdGlvbiBjdHIoZ2VvbSl7CiAgdmFyIHB0cz1bXTsKICBmdW5jdGlvbiBjb2woYyl7aWYodHlwZW9mIGNbMF09PT0nbnVtYmVyJykgcHRzLnB1c2goYyk7ZWxzZSBjLmZvckVhY2goY29sKTt9CiAgY29sKGdlb20uY29vcmRpbmF0ZXMpOwogIGlmKCFwdHMubGVuZ3RoKSByZXR1cm4gWzAsMF07CiAgcmV0dXJuIFtwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFswXTt9LDApL3B0cy5sZW5ndGgscHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMV07fSwwKS9wdHMubGVuZ3RoXTsKfQpmdW5jdGlvbiBzTmFtZShwcm9wcyl7CiAgdmFyIHJhdz1wcm9wcy5zdF9ubXx8cHJvcHMuTkFNRV8xfHxwcm9wcy5uYW1lfHxwcm9wcy5OQU1FfHwnJzsKICB2YXIgbWFwPXsnTGFkYWtoJzonSmFtbXUgYW5kIEthc2htaXInLCdKYW1tdSAmIEthc2htaXInOidKYW1tdSBhbmQgS2FzaG1pcicsJ1V0dGFyYW5jaGFsJzonVXR0YXJha2hhbmQnLCdBbmRhbWFuIGFuZCBOaWNvYmFyJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnQW5kYW1hbiAmIE5pY29iYXIgSXNsYW5kJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnTkNUIG9mIERlbGhpJzonRGVsaGknLCdQb25kaWNoZXJyeSc6J1B1ZHVjaGVycnknLCdEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpJzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsJ0RhbWFuIGFuZCBEaXUnOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1J307CiAgcmV0dXJuIG1hcFtyYXddfHxyYXc7Cn0KCnZhciBjYWNoZWRHZW89bnVsbDsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoYXR0ZW1wdCl7CiAgYXR0ZW1wdCA9IGF0dGVtcHR8fDE7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goJ2h0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9naC91ZGl0LTAwMS9pbmRpYS1tYXBzLWRhdGFAbWFzdGVyL3RvcG9qc29uL2luZGlhLmpzb24nKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwxMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwzMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciw2MDAwKTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1ttYXBdIGxvYWQgZmFpbGVkIGF0dGVtcHQgJythdHRlbXB0Kyc6JyxlLm1lc3NhZ2UpOwogICAgaWYoYXR0ZW1wdDw1KXsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe2xvYWRNYXAoYXR0ZW1wdCsxKTt9LCBhdHRlbXB0KjIwMDApOwogICAgfSBlbHNlIHsKICAgICAgdmFyIG1pPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKTsKICAgICAgaWYobWkpIG1pLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6IzJhM2E0YTtwYWRkaW5nOjQwcHg7dGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6bW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4Ij5NYXAgdW5hdmFpbGFibGUg4oCUIHJlZnJlc2ggdG8gcmV0cnk8L2Rpdj4nOwogICAgfQogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKZnVuY3Rpb24gYXBwbHlMYXllcigpewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksZmlsbDsKICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBsdj1MSVZFW25tXTt2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBlbTI9KGx2JiZsdi5lbW90aW9ucyYmT2JqZWN0LmtleXMobHYuZW1vdGlvbnMpLmxlbmd0aCk/bHYuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRlPShsdiYmbHYuZG9taW5hbnRfZW1vdGlvbil8fGQuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGVtMik7CiAgICAgIGlmKCFkZSYmZC5hdHRlbnRpb24+Mil7dmFyIG5wPWQuZG9taW5hbnRfbmFycmF0aXZlfHwnJztkZT1ucC5tYXRjaCgvYm9yZGVyfHRlcnJvcnxzZWN1cml0eXxjb25mbGljdC9pKT8nZmVhcic6bnAubWF0Y2goL3NjYW18Y29ycnVwdHxwcm90ZXN0fGFycmVzdC9pKT8nYW5nZXInOm5wLm1hdGNoKC9kZXZlbG9wfGludmVzdHxncm93dGh8bGF1bmNoL2kpPydob3BlJzonYW54aWV0eSc7fQogICAgICBmaWxsPWRlPyhlTWFwW2RlXXx8ZUMoZW0yKSk6ZUMoZW0yKXx8JyMzMzQ0NTUnOwogICAgfQogICAgZWxzZSBmaWxsPXZDKGQudmVsb2NpdHkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGZpbGwpOwogICAgKGZ1bmN0aW9uKCl7CiAgICAgIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiB4LmF0dGVudGlvbnx8MDt9KTsKICAgICAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKSxteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgICAgIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwoZC5hdHRlbnRpb24tbW4pLyhteC1tbikpKTsKICAgICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsbGF5ZXI9PT0nYXR0ZW50aW9uJz9NYXRoLm1heCgwLjMsMC4zK24qMC43KTowLjg1KTsKICAgIH0pKCk7CiAgfSk7Cn0KCmZ1bmN0aW9uIGF0dGFjaEludGVyYWN0aW9ucygpewogIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpewogICAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSk7CiAgICAgIHZhciB0b3A9ZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCc7CiAgICAgIHZhciBoaXN0PWQubmFycmF0aXZlSGlzdG9yeTsKICAgICAgdmFyIGxhdGVzdD1oaXN0JiZoaXN0Lmxlbmd0aD9oaXN0W2hpc3QubGVuZ3RoLTFdLnRvcGljOifigJQnOwogICAgICAvLyBEeW5hbWljIHRvb2x0aXAgY29udGVudCBiYXNlZCBvbiBhY3RpdmUgbGF5ZXIKICAgICAgdmFyIGxheWVyUm93cz0nJzsKICAgICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICAgICAgbGF5ZXJSb3dzPQogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkF0dGVudGlvbiBpbmRleDwvc3Bhbj48c3Ryb25nPicrZC5hdHRlbnRpb24rJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAoZC5kZWx0YSE9PTA/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaGlmdDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonKyhkLmRlbHRhPjA/JyNlMDVhMjgnOicjM2JiOGQ4JykrJyI+JysoZC5kZWx0YT4wPycrJzonJykrZC5kZWx0YSsnPC9zdHJvbmc+PC9kaXY+JzonJykrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+VG9wIG5hcnJhdGl2ZTwvc3Bhbj48c3Ryb25nPicrdG9wKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICAgIHZhciBlTGlzdD1PYmplY3QuZW50cmllcyhkLmVtb3Rpb25zKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgICAgICB2YXIgcmF3VD1lTGlzdC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgICAgICBpZihyYXdUPjAmJnJhd1Q8PTEuMDEpe2VMaXN0PWVMaXN0Lm1hcChmdW5jdGlvbihrdil7cmV0dXJuIFtrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO30KICAgICAgICB2YXIgdG90PU1hdGgubWF4KDEsZUxpc3QucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogICAgICAgIGlmKCFlTGlzdHx8IWVMaXN0Lmxlbmd0aCl7CiAgICAgICAgICBsYXllclJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkVtb3Rpb248L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj5Db2xsZWN0aW5nLi4uPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgICB9IGVsc2UgewogICAgICAgICAgdmFyIGRvbUVtbz1lTGlzdFswXTsKICAgICAgICAgIGxheWVyUm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RG9taW5hbnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JytwYWxbZG9tRW1vWzBdXSsnIj4nK2RvbUVtb1swXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW9bMF0uc2xpY2UoMSkrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAgIGVMaXN0LnNsaWNlKDAsMykubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm4gJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPicra3ZbMF0rJzwvc3Bhbj48c3Ryb25nPicrTWF0aC5yb3VuZChrdlsxXSoxMDAvdG90KSsnJTwvc3Ryb25nPjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgICAgfX0gZWxzZSB7CiAgICAgICAgdmFyIHZEaXI9ZC52ZWxvY2l0eT4wLjA1PydSaXNpbmcnOmQudmVsb2NpdHk8LTAuMDU/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgICAgIHZhciB2Q29sPWQudmVsb2NpdHk+MC4wNT8nI2UwNWEyOCc6ZC52ZWxvY2l0eTwtMC4wNT8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgICAgIGxheWVyUm93cz0KICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Nb21lbnR1bTwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZDb2wrJyI+JysoZC52ZWxvY2l0eT4wPycrJzonJykrZC52ZWxvY2l0eS50b0ZpeGVkKDIpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRpcmVjdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZDb2wrJyI+JysoZC52ZWxvY2l0eT4wLjE/J1Jpc2luZyBmYXN0JzpkLnZlbG9jaXR5PjAuMDI/J1Jpc2luZyc6ZC52ZWxvY2l0eTwtMC4wNT8nQ29vbGluZyc6J1N0YWJsZScpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaWduYWxzPC9zcGFuPjxzdHJvbmc+JysoZC5kZWx0YT49MD8nKyc6JycpK2QuZGVsdGErJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgIH0KICAgICAgdGlwLmlubmVySFRNTD0KICAgICAgICAnPGRpdiBjbGFzcz0idHQtbiI+JytubSsnPC9kaXY+JysKICAgICAgICBsYXllclJvd3MrCiAgICAgICAgJzxkaXYgY2xhc3M9InR0LW5hciI+PHN0cm9uZz5DdXJyZW50IG5hcnJhdGl2ZTwvc3Ryb25nPicrbGF0ZXN0Kyc8L2Rpdj4nOwogICAgICB2YXIgcmVjdD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJykuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICAgIHRpcC5zdHlsZS5sZWZ0PU1hdGgubWluKGUuY2xpZW50WC1yZWN0LmxlZnQrMTQscmVjdC53aWR0aC0xODApKydweCc7CiAgICAgIHRpcC5zdHlsZS50b3A9TWF0aC5taW4oZS5jbGllbnRZLXJlY3QudG9wKzE0LHJlY3QuaGVpZ2h0LTE0MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLm9wYWNpdHk9MTsKICAgIH0pOwogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJyxmdW5jdGlvbigpe3RpcC5zdHlsZS5vcGFjaXR5PTA7fSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe3NlbGVjdF8ocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8gU1RBVEUgUEFORUwKZnVuY3Rpb24gc2VsZWN0XyhubSl7CiAgU0VMPW5tOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7cC5jbGFzc0xpc3QudG9nZ2xlKCdzZWxlY3RlZCcscC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpPT09bm0pO30pOwogIHJlbmRlclBhbmVsKG5tKTsKICBmZXRjaERldGFpbChubSkudGhlbihmdW5jdGlvbihkKXtpZihTRUw9PT1ubSkgcmVuZGVyUGFuZWwobm0pO30pOwp9CgpmdW5jdGlvbiByZW5kZXJQYW5lbChubSl7CiAgdmFyIGQ9ZyhubSkscGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKCFwYW5lbCkgcmV0dXJuOwogIHZhciBpc0Zhdj1GQVZTLmhhcyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8ndXAnOidkbic7CiAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgdmFyIGVtb3Rpb25zPWQuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGQuZW1vdGlvbnMpLmxlbmd0aD9kLmVtb3Rpb25zOnthbnhpZXR5OjIwLGFuZ2VyOjE1LGhvcGU6MjUscHJpZGU6MjUsZmVhcjoxNX07CiAgdmFyIGVMPU9iamVjdC5lbnRyaWVzKGVtb3Rpb25zKTsKICAvLyBOb3JtYWxpemU6IEFQSSBtYXkgcmV0dXJuIDAtMSBmbG9hdHMgT1IgMC0xMDAgaW50ZWdlcnMKICB2YXIgcmF3VG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogIGlmKHJhd1RvdD4wICYmIHJhd1RvdDw9MS4wMSl7IGVMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuIFtrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pOyB9CiAgdmFyIHRvdD1NYXRoLm1heCgxLGVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApKTsKICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgdmFyIGFyY3M9ZUwubWFwKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0sZnI9di90b3QsYTE9Y3VtQSxhMj1jdW1BK2ZyKk1hdGguUEkqMjsKICAgIGN1bUE9YTI7dmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICB2YXIgeDE9Y3grTWF0aC5jb3MoYTEpKlIseTE9Y3krTWF0aC5zaW4oYTEpKlIseDI9Y3grTWF0aC5jb3MoYTIpKlIseTI9Y3krTWF0aC5zaW4oYTIpKlI7CiAgICB2YXIgeDM9Y3grTWF0aC5jb3MoYTIpKnJpLHkzPWN5K01hdGguc2luKGEyKSpyaSx4ND1jeCtNYXRoLmNvcyhhMSkqcmkseTQ9Y3krTWF0aC5zaW4oYTEpKnJpOwogICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICB9KS5qb2luKCcnKTsKCiAgdmFyIHRsPWQudGltZWxpbmUsdG1uPU1hdGgubWluLmFwcGx5KG51bGwsdGwpLHRteD1NYXRoLm1heC5hcHBseShudWxsLHRsKSx0cj1NYXRoLm1heCgxLHRteC10bW4pOwogIHZhciB0dz0yNjAsdGg9NjIsdHA9NTsKICB2YXIgcHRzPXRsLm1hcChmdW5jdGlvbih2LGkpe3JldHVybiBbdHArKGkvKHRsLmxlbmd0aC0xKSkqKHR3LXRwKjIpLHRwKygxLSh2LXRtbikvdHIpKih0aC10cCoyKV07fSk7CiAgdmFyIHBEPXB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gKGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pLmpvaW4oJycpOwogIHZhciBhRD1wRCsnIEwnK3B0c1twdHMubGVuZ3RoLTFdWzBdKycsJysodGgtdHApKycgTCcrcHRzWzBdWzBdKycsJysodGgtdHApKycgWic7CiAgdmFyIGFjPWFDKGQuYXR0ZW50aW9uKTsKCiAgdmFyIGhpc3Q9ZC5uYXJyYXRpdmVIaXN0b3J5fHxbXTsKCiAgcGFuZWwuaW5uZXJIVE1MPQogICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICc8ZGl2PjxkaXYgY2xhc3M9InNwLWVrIj5OYXJyYXRpdmUgcGFuZWw8L2Rpdj48ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhpc0Zhdj8nb24nOicnKSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYoXCcnK25tKydcJykiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKGlzRmF2PydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgJzwvYnV0dG9uPicrCiAgICAnPC9kaXY+JysKCiAgICAvLyBOYXJyYXRpdmUgaGlzdG9yeSB0aW1lbGluZSDigJQgc2lnbmF0dXJlIGZlYXR1cmUKICAgIChoaXN0Lmxlbmd0aD8KICAgICAgJzxkaXYgY2xhc3M9Im5hci10aW1lbGluZSI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibnQtbGFiZWwiPk5hcnJhdGl2ZSBldm9sdXRpb248L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJudC1mbG93Ij4nKwogICAgICAgICAgaGlzdC5tYXAoZnVuY3Rpb24oaCl7CiAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibnQtc3RlcCAnK2guY2xzKyciPicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im50LWRvdCI+PC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibnQtY29udGVudCI+JysKICAgICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJudC10b3BpYyI+JytoLnRvcGljKyc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im50LXdoZW4iPicraC53aGVuKyc8L2Rpdj4nKwogICAgICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzonJykrCgogICAgJzxkaXYgY2xhc3M9Imluc2lnaHQiPicrZC5zdW1tYXJ5Kyc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiPicrZC5hdHRlbnRpb24rJzwvZGl2PjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCBzaGlmdDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrZEMrJyI+JytkUytkLmRlbHRhKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5Ub3AgbmFycmF0aXZlPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCcpKyc8L2Rpdj48L2Rpdj4nKwogICAgJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPk5hcnJhdGl2ZSBicmVha2Rvd248L2Rpdj4nKwogICAgICAoZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXMubGVuZ3RoPwogICAgICAgICc8ZGl2IGNsYXNzPSJuYXItbGlzdCI+JysKICAgICAgICBkLm5hcnJhdGl2ZXMubWFwKGZ1bmN0aW9uKG4pewogICAgICAgICAgdmFyIG5tPW4ubmFtZT9uLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpOm4ubmFtZTsKICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0yIj4nKwogICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktbGFiZWwiPicrbm0rKG4uZGlyPT09J3VwJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojZTA1YTI4O2ZvbnQtc2l6ZTo5cHgiPuKGkTwvc3Bhbj4nOm4uZGlyPT09J2Rvd24nPycgPHNwYW4gc3R5bGU9ImNvbG9yOiMzYmI4ZDg7Zm9udC1zaXplOjlweCI+4oaTPC9zcGFuPic6JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdmFsIj4nK3ZhbC50b0ZpeGVkKDEpKyclPC9kaXY+JysKICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsdmFsKjIuNSkrJyU7YmFja2dyb3VuZDonKyhuLmRpcj09PSd1cCc/JyNlMDVhMjgnOm4uZGlyPT09J2Rvd24nPycjM2JiOGQ4JzonIzMzNDQ1NScpKyciPjwvZGl2PjwvZGl2PicrCiAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAnPC9kaXY+JzoKICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDA7bGluZS1oZWlnaHQ6MS42Ij5Mb3ctc2lnbmFsIHJlZ2lvbi4gTmF0aW9uYWwgcHJlc3MgY292ZXJhZ2UgaXMgbGltaXRlZCBmb3IgdGhpcyBzdGF0ZSDigJQgcmVnaW9uYWwgbGFuZ3VhZ2Ugc291cmNlcyBhcmUgYmVpbmcgbW9uaXRvcmVkLjwvZGl2PicpKwogICAgJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPk1vdmVtZW50PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9Im12LWdyaWQiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im12LWJsb2NrIHVwIj48ZGl2IGNsYXNzPSJtdi1oIj5SaXNpbmc8L2Rpdj4nKwogICAgICAgICAgKGQucmlzaW5nLmxlbmd0aD9kLnJpc2luZy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IGNsYXNzPSJtdi1pdCI+PHN0cm9uZz4nK3IudCsnPC9zdHJvbmc+PHNwYW4+JytyLnBjdCsnPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJyk6JzxkaXYgY2xhc3M9Im12LWl0IiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj5TdGFibGU8L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibXYtYmxvY2sgZG4iPjxkaXYgY2xhc3M9Im12LWgiPkZhbGxpbmc8L2Rpdj4nKwogICAgICAgICAgKGQuZmFsbGluZy5sZW5ndGg/ZC5mYWxsaW5nLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im12LWl0Ij48c3Ryb25nPicrci50Kyc8L3N0cm9uZz48c3Bhbj4nK3IucGN0Kyc8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKTonPGRpdiBjbGFzcz0ibXYtaXQiIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPlN0YWJsZTwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5FbW90aW9uYWwgcmVnaXN0ZXI8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZW0tcm93Ij4nKwogICAgICAgICc8c3ZnIGNsYXNzPSJlbS1kb251dCIgdmlld0JveD0iMCAwIDc2IDc2Ij4nK2FyY3MrJzwvc3ZnPicrCiAgICAgICAgJzxkaXYgY2xhc3M9ImVtLWxlZyI+JysKICAgICAgICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pLm1hcChmdW5jdGlvbihrdil7CiAgICAgICAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV07CiAgICAgICAgICAgIHZhciBkZXNjPXthbnhpZXR5OidVbmNlcnRhaW50eSAmIHdvcnJ5JyxhbmdlcjonT3V0cmFnZSAmIHByb3Rlc3QnLGhvcGU6J09wdGltaXNtICYgZ3Jvd3RoJyxwcmlkZTonQWNoaWV2ZW1lbnQgJiBpZGVudGl0eScsZmVhcjonVGhyZWF0IHBlcmNlcHRpb24nfTsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJlbS1pdGVtIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxcHgiPicrCiAgICAgICAgICAgICAgJzxzcGFuIGNsYXNzPSJlbS1zdyIgc3R5bGU9ImJhY2tncm91bmQ6JytwYWxba10rJyI+PC9zcGFuPicrCiAgICAgICAgICAgICAgJzxzcGFuIGNsYXNzPSJlbS1uIj4nK2suY2hhckF0KDApLnRvVXBwZXJDYXNlKCkray5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICAgICAgICAgJzxzcGFuIGNsYXNzPSJlbS1wIj4nK01hdGgucm91bmQodioxMDAvdG90KSsnJTwvc3Bhbj4nKwogICAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAgICAgKHY9PT1lTC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXVsxXT8KICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjFweCAwIDRweCAxMnB4O2JvcmRlci1sZWZ0OjFweCBzb2xpZCAnK3BhbFtrXSsnO21hcmdpbi1sZWZ0OjNweDttYXJnaW4tYm90dG9tOjNweDsiPicrZGVzY1trXSsnPC9kaXY+JzoKICAgICAgICAgICAgJycpOwogICAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkF0dGVudGlvbiDigJQgOCBkYXlzPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InRsLXdyYXAiPicrCiAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwICcrdHcrJyAnK3RoKyciIHN0eWxlPSJ3aWR0aDoxMDAlO2hlaWdodDoxMDAlIj4nKwogICAgICAgICAgJzxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0idGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKyciIHgxPSIwIiB4Mj0iMCIgeTE9IjAiIHkyPSIxIj4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAuMjUiLz4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMCIvPicrCiAgICAgICAgICAnPC9saW5lYXJHcmFkaWVudD48L2RlZnM+JysKICAgICAgICAgICc8cGF0aCBkPSInK2FEKyciIGZpbGw9InVybCgjdGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKycpIi8+JysKICAgICAgICAgICc8cGF0aCBkPSInK3BEKyciIGZpbGw9Im5vbmUiIHN0cm9rZT0iJythYysnIiBzdHJva2Utd2lkdGg9IjEuMiIvPicrCiAgICAgICAgICBwdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuICc8Y2lyY2xlIGN4PSInK3BbMF0rJyIgY3k9IicrcFsxXSsnIiByPSInKyhpPT09cHRzLmxlbmd0aC0xPzIuMjoxLjIpKyciIGZpbGw9IicrYWMrJyIvPic7fSkuam9pbignJykrCiAgICAgICAgJzwvc3ZnPicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2PicrCgogICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JytkLmFydGljbGVzLmxlbmd0aCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJhcnQtbGlzdCI+JysKICAgICAgICBkLmFydGljbGVzLm1hcChmdW5jdGlvbihhKXtyZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIj48ZGl2IGNsYXNzPSJhcnQtc3JjIj4nK2Euc3JjKyc8L2Rpdj48ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nK2EudHh0Kyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwp9CgpmdW5jdGlvbiB0b2dnbGVGYXYobm0pewogIGlmKEZBVlMuaGFzKG5tKSkgRkFWUy5kZWxldGUobm0pO2Vsc2UgRkFWUy5hZGQobm0pOwogIHJlbmRlclBhbmVsKFNFTCk7cmVuZGVyRmF2cygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdnMoKXsKICB2YXIgcm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXYtcm93Jyk7CiAgaWYoIUZBVlMuc2l6ZSl7cm93LmlubmVySFRNTD0nPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+JztyZXR1cm47fQogIHJvdy5pbm5lckhUTUw9QXJyYXkuZnJvbShGQVZTKS5tYXAoZnVuY3Rpb24obm0pewogICAgdmFyIGQ9ZyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8nI2UwNWEyOCc6JyMzYmI4ZDgnOwogICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZmF2LWNhcmQiIG9uY2xpY2s9InNlbGVjdF8oXCcnK25tKydcJykiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1oZWFkIj48c3BhbiBjbGFzcz0iZmMtbmFtZSI+JytubSsnPC9zcGFuPjxzcGFuIGNsYXNzPSJmYy1zYyI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPk5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj4yNGg8L3NwYW4+PHNwYW4gY2xhc3M9InYiIHN0eWxlPSJjb2xvcjonK2RDKyciPicrZFMrZC5kZWx0YSsnPC9zcGFuPjwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKGMpewogIGMuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7bGF5ZXI9Yy5kYXRhc2V0LmxheWVyO2FwcGx5TGF5ZXIoKTsKICB9KTsKfSk7CgpmdW5jdGlvbiB1cGRhdGVDbG9jaygpewogIHZhciBub3c9bmV3IERhdGUoKSxpc3Q9bmV3IERhdGUobm93LmdldFRpbWUoKStub3cuZ2V0VGltZXpvbmVPZmZzZXQoKSo2MDAwMCsxOTgwMDAwMCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Nsb2NrJykudGV4dENvbnRlbnQ9U3RyaW5nKGlzdC5nZXRIb3VycygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0TWludXRlcygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0U2Vjb25kcygpKS5wYWRTdGFydCgyLCcwJykrJyBJU1QnOwp9CnNldEludGVydmFsKHVwZGF0ZUNsb2NrLDEwMDApO3VwZGF0ZUNsb2NrKCk7CgovLyBJTklUIOKAlCB3YWl0IGZvciBET00KZnVuY3Rpb24gaW5pdCgpewogIHJlbmRlclN0cmlwKCczbScpOwogIHJlbmRlck1vbWVudHVtKCk7CiAgLy8gTG9hZCBtYXAgd2l0aCByZXRyeSBpZiB0b3BvanNvbiBub3QgcmVhZHkKICB2YXIgbWFwQXR0ZW1wdHM9MDsKICBmdW5jdGlvbiB0cnlMb2FkTWFwKCl7CiAgICBpZih0eXBlb2YgdG9wb2pzb249PT0ndW5kZWZpbmVkJyl7CiAgICAgIGlmKG1hcEF0dGVtcHRzKys8MTApIHNldFRpbWVvdXQodHJ5TG9hZE1hcCwzMDApOwogICAgICByZXR1cm47CiAgICB9CiAgICBsb2FkTWFwKCk7CiAgfQogIHRyeUxvYWRNYXAoKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7c3RhcnRQb2xsaW5nKCk7fSw4MDApOwogIC8vIFJldHJ5IG1hcCBpZiBzdGlsbCBlbXB0eSBhZnRlciAzcwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAgIGlmKCFkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5sZW5ndGgpewogICAgICBjb25zb2xlLmxvZygnW2luaXRdIG1hcCBlbXB0eSwgcmV0cnlpbmcgbG9hZE1hcCcpOwogICAgICBsb2FkTWFwKCk7CiAgICB9CiAgfSwzMDAwKTsKICAvLyBBbmQgYWdhaW4gYXQgNnMKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICBpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKXsKICAgICAgbG9hZE1hcCgpOwogICAgfQogIH0sNjAwMCk7Cn0KaWYoZG9jdW1lbnQucmVhZHlTdGF0ZT09PSdsb2FkaW5nJyl7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsIGluaXQpOwp9IGVsc2UgewogIC8vIEFscmVhZHkgbG9hZGVkIOKAlCBidXQgd2FpdCBvbmUgdGljayB0byBlbnN1cmUgYWxsIHNjcmlwdHMgcGFyc2VkCiAgc2V0VGltZW91dChpbml0LCAwKTsKfQoKLy8gUkVQTEFZIElORElBCnZhciBSRVBMQVlfUEVSSU9EUz17JzdkJzp7ZGF5czo3LGxhYmVsOidQYXN0IDcgZGF5cyd9LCczMGQnOntkYXlzOjMwLGxhYmVsOidQYXN0IDMwIGRheXMnfSwnNm0nOntkYXlzOjE4MCxsYWJlbDonUGFzdCA2IG1vbnRocyd9LCdlbGVjdGlvbic6e2RheXM6OTAsbGFiZWw6J0VsZWN0aW9uIHNlYXNvbiAyMDI0J319Owp2YXIgcmVwbGF5UGVyaW9kPSc3ZCcscmVwbGF5UG9zPTAscmVwbGF5UGxheWluZz1mYWxzZSxyZXBsYXlUaW1lcj1udWxsLHJlcGxheVNwZWVkPTEsbGFzdFNuYXBQb3M9LTE7CmZ1bmN0aW9uIGZtdERhdGUoZCl7cmV0dXJuIGQudG9Mb2NhbGVEYXRlU3RyaW5nKCdlbi1JTicse2RheTonbnVtZXJpYycsbW9udGg6J3Nob3J0J30pO30KZnVuY3Rpb24gaW5pdFJlcGxheSgpewogIHZhciBwPVJFUExBWV9QRVJJT0RTW3JlcGxheVBlcmlvZF0sbm93PW5ldyBEYXRlKCksc3RhcnQ9bmV3IERhdGUobm93LXAuZGF5cyo4NjQwMDAwMCk7CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1kYXRlcycpOwogIGlmKGVsKWVsLmlubmVySFRNTD0nPHNwYW4+JytmbXREYXRlKHN0YXJ0KSsnPC9zcGFuPjxzcGFuPicrZm10RGF0ZShuZXcgRGF0ZShzdGFydC5nZXRUaW1lKCkrcC5kYXlzKjg2NDAwMDAwKjAuMzMpKSsnPC9zcGFuPjxzcGFuPicrZm10RGF0ZShuZXcgRGF0ZShzdGFydC5nZXRUaW1lKCkrcC5kYXlzKjg2NDAwMDAwKjAuNjYpKSsnPC9zcGFuPjxzcGFuPlRvZGF5PC9zcGFuPic7CiAgc2V0UmVwbGF5UG9zKDApOwp9CmZ1bmN0aW9uIHNldFJlcGxheVBvcyhwb3MpewogIHJlcGxheVBvcz1NYXRoLm1heCgwLE1hdGgubWluKDEscG9zKSk7CiAgdmFyIGZpbGw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLWZpbGwnKSx0aHVtYj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtdGh1bWInKSxkYXRlRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLWN1cnJlbnQtZGF0ZScpOwogIGlmKGZpbGwpZmlsbC5zdHlsZS53aWR0aD0ocmVwbGF5UG9zKjEwMCkrJyUnOwogIGlmKHRodW1iKXRodW1iLnN0eWxlLmxlZnQ9KHJlcGxheVBvcyoxMDApKyclJzsKICB2YXIgcD1SRVBMQVlfUEVSSU9EU1tyZXBsYXlQZXJpb2RdLG5vdz1uZXcgRGF0ZSgpLHN0YXJ0PW5ldyBEYXRlKG5vdy1wLmRheXMqODY0MDAwMDApLGN1cj1uZXcgRGF0ZShzdGFydC5nZXRUaW1lKCkrcmVwbGF5UG9zKnAuZGF5cyo4NjQwMDAwMCk7CiAgaWYoZGF0ZUVsKWRhdGVFbC50ZXh0Q29udGVudD1mbXREYXRlKGN1cikrJyDigJQgJytwLmxhYmVsOwogIHZhciBzY2FsZT0wLjM1K3JlcGxheVBvcyowLjY1OwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksc2E9KGQuYXR0ZW50aW9ufHwwKSpzY2FsZTsKICAgIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiAoeC5hdHRlbnRpb258fDApKnNjYWxlO30pOwogICAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKSxteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDEsbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHNhLW1uKS8obXgtbW4pKSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoc2EpKTtwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxNYXRoLm1heCgwLjIsMC4yK24qMC44KSk7CiAgfSk7CiAgaWYoTWF0aC5hYnMocmVwbGF5UG9zLWxhc3RTbmFwUG9zKT4wLjEyKXtsYXN0U25hcFBvcz1yZXBsYXlQb3M7dXBkYXRlUmVwbGF5U25hcHNob3QocmVwbGF5UG9zKTt9Cn0KZnVuY3Rpb24gdXBkYXRlUmVwbGF5U25hcHNob3QocG9zKXsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKFNEKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5hdHRlbnRpb24+MDt9KS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybntuYW1lOmt2WzBdLGF0dDpNYXRoLnJvdW5kKChrdlsxXS5hdHRlbnRpb258fDApKigwLjM1K3BvcyowLjY1KSksbmFyOihrdlsxXS5uYXJyYXRpdmVzJiZrdlsxXS5uYXJyYXRpdmVzWzBdP2t2WzFdLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJyl9O30pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSkuc2xpY2UoMCw2KTsKICB2YXIgc25hcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtc25hcC1zdGF0ZXMnKTsKICBpZighc25hcClyZXR1cm47CiAgaWYoIXRvcC5sZW5ndGgpe3NuYXAuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJycC1sb2ctZW1wdHkiPk5vIHNpZ25hbCBkYXRhIHlldC48L2Rpdj4nO3JldHVybjt9CiAgc25hcC5pbm5lckhUTUw9dG9wLm1hcChmdW5jdGlvbihzKXtyZXR1cm4gJzxkaXYgY2xhc3M9InJwLXN0YXRlLWNhcmQiPjxkaXYgY2xhc3M9InJwLXN0YXRlLW5hbWUiPicrcy5uYW1lKyc8L2Rpdj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1uYXIiPicrcy5uYXIrJzwvZGl2PjxkaXYgY2xhc3M9InJwLXN0YXRlLWF0dCI+QXR0ZW50aW9uICcrcy5hdHQrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7Cn0KZnVuY3Rpb24gdG9nZ2xlUmVwbGF5KCl7CiAgcmVwbGF5UGxheWluZz0hcmVwbGF5UGxheWluZzsKICB2YXIgaWNvbj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtcGxheS1pY29uJyk7CiAgaWYocmVwbGF5UGxheWluZyl7aWYocmVwbGF5UG9zPj0wLjk5KXNldFJlcGxheVBvcygwKTtpZihpY29uKWljb24uc2V0QXR0cmlidXRlKCdwb2ludHMnLCczLDIgNywyIDcsOCAzLDggTTgsMiAxMiwyIDEyLDggOCw4Jyk7cnVuUmVwbGF5KCk7fQogIGVsc2V7aWYoaWNvbilpY29uLnNldEF0dHJpYnV0ZSgncG9pbnRzJywnMiwxIDksNSAyLDknKTtjbGVhckludGVydmFsKHJlcGxheVRpbWVyKTthcHBseUxheWVyKCk7fQp9CmZ1bmN0aW9uIHJ1blJlcGxheSgpewogIGNsZWFySW50ZXJ2YWwocmVwbGF5VGltZXIpOwogIHJlcGxheVRpbWVyPXNldEludGVydmFsKGZ1bmN0aW9uKCl7CiAgICByZXBsYXlQb3MrPTAuMDAzKnJlcGxheVNwZWVkOwogICAgaWYocmVwbGF5UG9zPj0xKXtyZXBsYXlQb3M9MTtzZXRSZXBsYXlQb3MoMSk7cmVwbGF5UGxheWluZz1mYWxzZTt2YXIgaWNvbj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtcGxheS1pY29uJyk7aWYoaWNvbilpY29uLnNldEF0dHJpYnV0ZSgncG9pbnRzJywnMiwxIDksNSAyLDknKTtjbGVhckludGVydmFsKHJlcGxheVRpbWVyKTthcHBseUxheWVyKCk7cmV0dXJuO30KICAgIHNldFJlcGxheVBvcyhyZXBsYXlQb3MpOwogIH0sNjApOwp9CihmdW5jdGlvbigpe3ZhciB0cmFjaz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtdHJhY2snKTtpZighdHJhY2spcmV0dXJuO3ZhciBkcmFnPWZhbHNlOwp0cmFjay5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWRvd24nLGZ1bmN0aW9uKGUpe2RyYWc9dHJ1ZTt2YXIgcmVjdD10cmFjay5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtzZXRSZXBsYXlQb3MoKGUuY2xpZW50WC1yZWN0LmxlZnQpL3JlY3Qud2lkdGgpO30pOwpkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpe2lmKCFkcmFnKXJldHVybjt2YXIgcmVjdD10cmFjay5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtzZXRSZXBsYXlQb3MoKGUuY2xpZW50WC1yZWN0LmxlZnQpL3JlY3Qud2lkdGgpO30pOwpkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZXVwJyxmdW5jdGlvbigpe2lmKGRyYWcpe2RyYWc9ZmFsc2U7aWYoIXJlcGxheVBsYXlpbmcpYXBwbHlMYXllcigpO319KTt9KSgpOwpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucnAtYnRuJykuZm9yRWFjaChmdW5jdGlvbihidG4pe2J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXtkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucnAtYnRuJykuZm9yRWFjaChmdW5jdGlvbihiKXtiLmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pO2J0bi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtyZXBsYXlQZXJpb2Q9YnRuLmRhdGFzZXQucGVyaW9kO3JlcGxheVBvcz0wO2xhc3RTbmFwUG9zPS0xO2luaXRSZXBsYXkoKTt9KTt9KTsKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLXNwZCcpLmZvckVhY2goZnVuY3Rpb24oYnRuKXtidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7ZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLXNwZCcpLmZvckVhY2goZnVuY3Rpb24oYil7Yi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTtidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVwbGF5U3BlZWQ9cGFyc2VJbnQoYnRuLmRhdGFzZXQuc3BkKTt9KTt9KTsKaW5pdFJlcGxheSgpOwpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgLy8gQXV0by1zZWxlY3QgaG90dGVzdCBzdGF0ZSBmcm9tIExJVkUgZGF0YQogIHZhciBzcmM9T2JqZWN0LmtleXMoTElWRSkubGVuZ3RoP0xJVkU6U0Q7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhzcmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKHRvcCl7CiAgICB2YXIgZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJyt0b3BbMF0rJyJdJyk7CiAgICBpZihlbCkgc2VsZWN0Xyh0b3BbMF0pOwogIH0KfSwzMDAwKTsKc2V0VGltZW91dChyZW5kZXJGYXZzLDI0MDApOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="

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
