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
    "anger":   ["outrage", "fury", "angry", "slam", "blast", "condemn", "protest", "demand", "furious"],
    "anxiety": ["worry", "concern", "anxious", "alarm", "fear", "uncertain", "crisis", "panic"],
    "hope":    ["hope", "optimism", "progress", "growth", "achievement", "milestone", "breakthrough"],
    "pride":   ["pride", "honor", "celebrate", "victory", "achievement", "first", "historic"],
    "fear":    ["threat", "danger", "attack", "violence", "warn", "terror", "menace"],
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
        self.history: dict[str, list[float]] = {s: [] for s in INDIAN_STATES}
        # Pre-serialized response cache — served instantly to frontend
        self.cache_states: list | None = None          # /api/states response
        self.cache_snapshot: dict | None = None        # /api/snapshot/daily response
        self.cache_state_detail: dict[str, dict] = {}  # /api/state/{name} responses
        self.cache_built_at: datetime | None = None

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
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        self.signals[state] = [
            s for s in self.signals[state][-1000:]
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
    """Classify emotional tone from text. Returns None if no emotion keywords found."""
    lower = text.lower()
    raw: dict[str, float] = {}
    for emo, words in EMOTION_KEYWORDS.items():
        score = sum(1 for w in words if w in lower)
        if score > 0:
            raw[emo] = float(score)

    # If no emotion keywords found, return None so we don't pollute aggregates
    if not raw:
        return {}

    # Small baseline only for emotions that have at least 1 hit
    total = sum(raw.values()) or 1.0
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

# State-specific search terms to get more relevant signals
STATE_SEARCH_TERMS: dict[str, str] = {
    "Gujarat":           "Gujarat economy investment industry",
    "Himachal Pradesh":  "Himachal Pradesh tourism disaster infrastructure",
    "Goa":               "Goa tourism environment coast",
    "Sikkim":            "Sikkim environment border glacier",
    "Mizoram":           "Mizoram refugee border governance",
    "Meghalaya":         "Meghalaya environment coal mining",
    "Nagaland":          "Nagaland Naga peace framework",
    "Tripura":           "Tripura Bangladesh border trade",
    "Arunachal Pradesh": "Arunachal Pradesh China border infrastructure",
    "Andaman and Nicobar Islands": "Andaman Nicobar islands development",
}

def build_rss_url(state: str, lang_code: str) -> str:
    # Use state-specific terms if available, else generic political news
    terms = STATE_SEARCH_TERMS.get(state, state + " politics government")
    query = terms.replace(" ", "+")
    return (
        f"https://news.google.com/rss/search?"
        f"q={query}&hl={lang_code}&gl=IN&ceid=IN:{lang_code.split('-')[0]}"
    )

# Multiple search angles per state for richer signal coverage
STATE_QUERY_ANGLES = [
    "{state} politics government",
    "{state} news today",
    "{state} economy development",
]

async def ingest_state(client: httpx.AsyncClient, state: str) -> int:
    added = 0
    urls_tried = set()

    # Build multiple query URLs for this state
    queries = [
        STATE_SEARCH_TERMS.get(state, f"{state} politics government"),
        f"{state} news",
        f"{state} latest",
    ]
    # Add state capital/city specific query if we have aliases
    aliases = STATE_ALIASES.get(state, [])
    if len(aliases) > 1:
        queries.append(aliases[1] + " news")  # city name

    for query in queries[:3]:  # max 3 queries per state
        url = (
            f"https://news.google.com/rss/search?"
            f"q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        if url in urls_tried:
            continue
        urls_tried.add(url)
        try:
            r = await client.get(url)
            feed = feedparser.parse(r.text)
        except Exception as e:
            print(f"[ingest] {state} fetch error: {e}")
            continue

        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            body = entry.get("summary", "")
            source_url = entry.get("link", "")
            if not title:
                continue

            text = f"{title} {body}"
            tagged = geotag_states(text)
            # Always include the queried state
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
            timeout=10,
            headers={"User-Agent": "IndiaAttentionMap/1.0 (research)"},
            follow_redirects=True,
        ) as client:
            sem = asyncio.Semaphore(12)
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

    # Emotion breakdown — only aggregate signals that have actual emotion keywords
    emo_totals: dict[str, float] = {k: 0.0 for k in EMOTION_KEYWORDS}
    emo_signal_count = 0
    for s in sigs_24h:
        sig_emotions = s.get("emotions", {})
        if not sig_emotions:
            continue  # skip signals with no emotion keywords
        for k, v in sig_emotions.items():
            if k in emo_totals:
                emo_totals[k] += v
        emo_signal_count += 1

    total_emo = sum(emo_totals.values())
    if total_emo > 0:
        emotions = {k: round(v / total_emo, 3) for k, v in emo_totals.items()}
    else:
        # No emotion data — return empty so frontend shows neutral/unknown
        emotions = {}

    # Top articles (deduplicated by source)
    seen_src: set[str] = set()
    articles = []
    for s in sorted(sigs_24h, key=lambda x: x["published_at"], reverse=True):
        src = s.get("source", "unknown")
        if src in seen_src:
            continue
        seen_src.add(src)
        articles.append({
            "src": src,
            "txt": s["title"],
            "url": s.get("source_url", "#"),
        })
        if len(articles) >= 8:
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
    # Build response cache immediately after scoring
    await build_response_cache()


async def build_response_cache():
    """Pre-serialize all API responses into cache for instant serving."""
    now = datetime.now(timezone.utc)

    # Cache /api/states (includes emotions for consistent frontend rendering)
    store.cache_states = [
        {
            "name": s["name"],
            "attention": s["attention"],
            "delta_24h": s["delta_24h"],
            "velocity": s["velocity"],
            "dominant_emotion": s.get("dominant_emotion"),
            "dominant_narrative": s.get("dominant_narrative", "governance"),
            "emotions": s.get("emotions", {}),  # include for map color consistency
        }
        for s in store.scores.values()
        if isinstance(s, dict)
    ]

    # Cache /api/state/{name} detail for every state
    for state, score in store.scores.items():
        if isinstance(score, dict):
            store.cache_state_detail[state] = score

    # Cache /api/snapshot/daily — no filtering, all states included
    all_scored = [s for s in store.scores.values() if isinstance(s, dict) and s]

    if all_scored:
        hottest    = max(all_scored, key=lambda s: s.get("attention", 0))
        fastest_up = max(all_scored, key=lambda s: s.get("delta_24h", 0))
        fastest_dn = min(all_scored, key=lambda s: s.get("delta_24h", 0))
        anger_states = [s for s in all_scored if s.get("emotions", {}).get("anger", 0) > 0]
        peak_anger = max(anger_states, key=lambda s: s["emotions"].get("anger", 0)) if anger_states else hottest
        nar_agg: dict[str, float] = {}
        for s in all_scored:
            for n in s.get("narratives", []):
                nar_agg[n["name"]] = nar_agg.get(n["name"], 0) + n["val"]
        top_nar = max(nar_agg.items(), key=lambda x: x[1], default=("governance", 0))[0]
        store.cache_snapshot = {
            "hottest_state":          hottest["name"],
            "hottest_score":          round(hottest.get("attention", 0), 1),
            "fastest_rising":         fastest_up["name"],
            "fastest_cooling":        fastest_dn["name"],
            "peak_anger_state":       peak_anger["name"],
            "peak_anger_pct":         round(peak_anger["emotions"].get("anger", 0) * 100),
            "top_national_narrative": top_nar,
            "as_of":                  now.isoformat(),
            "total_signals":          sum(len(v) for v in store.signals.values()),
            "credible_states":        len(all_scored),
        }
