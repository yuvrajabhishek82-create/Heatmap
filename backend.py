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
    "Arunachal Pradesh": ["arunachal", "itanagar", "tawang"],
    "Sikkim":            ["sikkim", "gangtok"],
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
    lower = text.lower()
    raw: dict[str, float] = {}
    for emo, words in EMOTION_KEYWORDS.items():
        raw[emo] = sum(1 for w in words if w in lower) + 0.15
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

def build_rss_url(state: str, lang_code: str) -> str:
    query = state.replace(" ", "+") + "+politics+news"
    return (
        f"https://news.google.com/rss/search?"
        f"q={query}&hl={lang_code}&gl=IN&ceid=IN:{lang_code.split('-')[0]}"
    )

async def ingest_state(client: httpx.AsyncClient, state: str) -> int:
    added = 0
    urls_tried = set()
    for lang_code in LANG_CODES[:3]:  # limit to 3 langs to stay polite
        url = build_rss_url(state, lang_code)
        if url in urls_tried:
            continue
        urls_tried.add(url)
        try:
            r = await client.get(url)
            feed = feedparser.parse(r.text)
        except Exception as e:
            print(f"[ingest] {state}/{lang_code} fetch error: {e}")
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
                "language": lang_code.split("-")[0],
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
            timeout=25,
            headers={"User-Agent": "IndiaAttentionMap/1.0 (research)"},
            follow_redirects=True,
        ) as client:
            sem = asyncio.Semaphore(4)
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

    # Emotion breakdown
    emo_totals: dict[str, float] = {k: 0.0 for k in EMOTION_KEYWORDS}
    for s in sigs_24h:
        for k, v in s.get("emotions", {}).items():
            if k in emo_totals:
                emo_totals[k] += v
    total_emo = max(0.001, sum(emo_totals.values()))
    emotions = {k: round(v / total_emo, 3) for k, v in emo_totals.items()}

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

    dominant_emotion = max(emotions, key=lambda k: emotions[k]) if emotions else "anxiety"
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


@app.on_event("startup")
async def startup():
    print("[startup] India Attention Map backend starting...")
    # Kick off first ingestion in background so server is immediately available
    asyncio.create_task(run_ingest())


# ── Serve frontend ──────────────────────────────────────────────────────────

# HTML served via base64 decode — avoids all string escaping issues
import base64 as _b64
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5JbmRpYSBBdHRlbnRpb24gTWFwIOKAlCBQb2xpdGljYWwgTmFycmF0aXZlIEludGVsbGlnZW5jZTwvdGl0bGU+CjxsaW5rIHJlbD0icHJlY29ubmVjdCIgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbSI+CjxsaW5rIHJlbD0icHJlY29ubmVjdCIgaHJlZj0iaHR0cHM6Ly9mb250cy5nc3RhdGljLmNvbSIgY3Jvc3NvcmlnaW4+CjxsaW5rIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20vY3NzMj9mYW1pbHk9RnJhdW5jZXM6b3Bzeix3Z2h0QDkuLjE0NCwzMDA7OS4uMTQ0LDQwMDs5Li4xNDQsNTAwOzkuLjE0NCw2MDAmZmFtaWx5PUpldEJyYWlucytNb25vOndnaHRAMzAwOzQwMDs1MDAmZmFtaWx5PUludGVyK1RpZ2h0OndnaHRAMzAwOzQwMDs1MDA7NjAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnLTA6IzA2MDgwZDsKICAtLWJnLTE6IzBiMGYxNzsKICAtLWJnLTI6IzExMTYxZjsKICAtLXN1cmZhY2U6cmdiYSgyMCwyNiwzOCwwLjU1KTsKICAtLWJvcmRlcjpyZ2JhKDE4MCwyMDAsMjMwLDAuMDcpOwogIC0tYm9yZGVyLXN0cm9uZzpyZ2JhKDE4MCwyMDAsMjMwLDAuMTUpOwogIC0taW5rOiNlOGVlZjc7CiAgLS1pbmstZGltOiM4YTk1YTg7CiAgLS1pbmstZmFpbnQ6IzUyNWQ3MDsKICAtLWFjY2VudDojZmY2YjNkOwogIC0tcmlzZTojZmY2YjNkOwogIC0tZmFsbDojNGNjOWYwOwogIC0tc2VyaWY6J0ZyYXVuY2VzJyxHZW9yZ2lhLHNlcmlmOwogIC0tc2FuczonSW50ZXIgVGlnaHQnLHN5c3RlbS11aSxzYW5zLXNlcmlmOwogIC0tbW9ubzonSmV0QnJhaW5zIE1vbm8nLHVpLW1vbm9zcGFjZSxtb25vc3BhY2U7Cn0KKntib3gtc2l6aW5nOmJvcmRlci1ib3g7bWFyZ2luOjA7cGFkZGluZzowfQpodG1sLGJvZHl7YmFja2dyb3VuZDp2YXIoLS1iZy0wKTtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO292ZXJmbG93LXg6aGlkZGVufQpib2R5ewogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA2MCUgMzUlIGF0IDUwJSAtNSUsIHJnYmEoMjU1LDEwNyw2MSwwLjA3KSwgdHJhbnNwYXJlbnQgNTUlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDQwJSAyNSUgYXQgODUlIDk1JSwgcmdiYSg3NiwyMDEsMjQwLDAuMDQpLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmctMCk7CiAgbWluLWhlaWdodDoxMDB2aDsKfQpib2R5OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjpmaXhlZDtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQtaW1hZ2U6dXJsKCJkYXRhOmltYWdlL3N2Zyt4bWw7dXRmOCw8c3ZnIHhtbG5zPSdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2Zycgd2lkdGg9JzE4MCcgaGVpZ2h0PScxODAnPjxmaWx0ZXIgaWQ9J24nPjxmZVR1cmJ1bGVuY2UgdHlwZT0nZnJhY3RhbE5vaXNlJyBiYXNlRnJlcXVlbmN5PScwLjknIG51bU9jdGF2ZXM9JzInLz48ZmVDb2xvck1hdHJpeCB2YWx1ZXM9JzAgMCAwIDAgMC45IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40O21peC1ibGVuZC1tb2RlOm92ZXJsYXk7Cn0KCi8qIOKUgOKUgCBUT1BCQVIg4pSA4pSAICovCi50b3BiYXJ7CiAgcG9zaXRpb246c3RpY2t5O3RvcDowO3otaW5kZXg6NTA7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjEycHggMzJweDsKICBiYWNrZ3JvdW5kOnJnYmEoNiw4LDEzLDAuODUpOwogIGJhY2tkcm9wLWZpbHRlcjpibHVyKDI0cHgpOwogIGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLmJyYW5ke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEwcHh9Ci5icmFuZC1tYXJrewogIHdpZHRoOjI4cHg7aGVpZ2h0OjI4cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTM1ZGVnLCNmZjZiM2QsI2ZmNGQ2ZCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGJveC1zaGFkb3c6MCAwIDE2cHggcmdiYSgyNTUsMTA3LDYxLDAuMyk7Cn0KLmJyYW5kLW1hcms6OmFmdGVyewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4OwogIGJvcmRlcjoxLjVweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuODUpO2JvcmRlci1yYWRpdXM6MnB4OwogIGNsaXAtcGF0aDpwb2x5Z29uKDUwJSAwJSwxMDAlIDM4JSw4MiUgMTAwJSwxOCUgMTAwJSwwJSAzOCUpOwp9Ci5icmFuZC10ZXh0IC5uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo1MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLmJyYW5kLXRleHQgLnN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0taW5rLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi10b3A6MXB4fQoudG9wYmFyLXJpZ2h0e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE4cHh9Ci5saXZlLXBpbGx7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWluay1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDZlbTsKICBwYWRkaW5nOjRweCAxMHB4O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjIwcHg7Cn0KLmxpdmUtZG90e3dpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6IzRhZGU4MDtib3gtc2hhZG93OjAgMCA3cHggIzRhZGU4MDthbmltYXRpb246bHAgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgbHB7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC40fX0KLmNsb2Nre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWluay1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtfQoKLyog4pSA4pSAIEhFUk8g4pSA4pSAICovCi5oZXJvewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBwYWRkaW5nOjU2cHggMzJweCAyOHB4OwogIG1heC13aWR0aDoxNTIwcHg7bWFyZ2luOjAgYXV0bzsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjQ4cHg7YWxpZ24taXRlbXM6ZW5kOwp9Ci5oZXJvLWxlZnQgLmV5ZWJyb3d7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7bGV0dGVyLXNwYWNpbmc6MC4yOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTZweDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7Cn0KLmhlcm8tbGVmdCAuZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MjBweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KX0KLmhlcm8tbGVmdCBoMXsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC13ZWlnaHQ6MzAwOwogIGZvbnQtc2l6ZTpjbGFtcCgzOHB4LDQuNXZ3LDcycHgpO2xpbmUtaGVpZ2h0OjAuOTc7CiAgbGV0dGVyLXNwYWNpbmc6LTAuMDI1ZW07Cn0KLmhlcm8tbGVmdCBoMSBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1hY2NlbnQpO2ZvbnQtd2VpZ2h0OjQwMH0KLmhlcm8tbGVmdCAuc3ViewogIG1hcmdpbi10b3A6MTZweDtmb250LXNpemU6MTQuNXB4O2xpbmUtaGVpZ2h0OjEuNjsKICBjb2xvcjp2YXIoLS1pbmstZGltKTtmb250LXdlaWdodDozMDA7bWF4LXdpZHRoOjQyMHB4Owp9CgovKiBOQVJSQVRJVkUgUFVMU0UgU1RSSVAg4oCUIHJpZ2h0IHNpZGUgb2YgaGVybyAqLwoubmFycmF0aXZlLXN0cmlwewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZmFjZSk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBvdmVyZmxvdzpoaWRkZW47Cn0KLnN0cmlwLWhlYWRlcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTNweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnN0cmlwLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0taW5rLWZhaW50KX0KLnRpbWUtdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjJweH0KLnRpbWUtdGFiewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1pbmstZmFpbnQpO3BhZGRpbmc6NHB4IDEwcHg7Ym9yZGVyLXJhZGl1czo0cHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQoudGltZS10YWIuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDI1NSwxMDcsNjEsMC4xMik7Ym94LXNoYWRvdzppbnNldCAwIDAgMCAxcHggcmdiYSgyNTUsMTA3LDYxLDAuMil9Ci50aW1lLXRhYjpob3Zlcntjb2xvcjp2YXIoLS1pbmstZGltKX0KCi5zdHJpcC1ib2R5e3BhZGRpbmc6MTRweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjEwcHh9Ci5zdHJpcC1yb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MDttaW4taGVpZ2h0OjUycHh9Ci5zdHJpcC1mcm9tLC5zdHJpcC10b3tmbGV4OjE7cGFkZGluZzowIDRweH0KLnN0cmlwLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTttYXJnaW4tYm90dG9tOjNweH0KLnN0cmlwLWxhYmVsLmZhbGx7Y29sb3I6dmFyKC0tZmFsbCl9Ci5zdHJpcC1sYWJlbC5yaXNle2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjJ9Ci5zdHJpcC1tZXRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO21hcmdpbi10b3A6MnB4fQouc3RyaXAtYXJyb3d7CiAgd2lkdGg6MzZweDtmbGV4LXNocmluazowO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O2ZvbnQtc2l6ZToxNnB4Owp9Ci5zdHJpcC1kaXZpZGVye2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO21hcmdpbjoycHggMH0KCi8qIOKUgOKUgCBNQUlOIEdSSUQg4pSA4pSAICovCi5tYWluLWdyaWR7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNTIwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzJweCAzMnB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM4MHB4OwogIGdyaWQtdGVtcGxhdGUtcm93czphdXRvOwogIGdhcDoyMHB4Owp9CgovKiDilIDilIAgTUFQIENBUkQg4pSA4pSAICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmZhY2UpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGdyaWQtY29sdW1uOjE7Z3JpZC1yb3c6MTsKfQoubWFwLWNhcmQ6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7cG9pbnRlci1ldmVudHM6bm9uZTsKICBiYWNrZ3JvdW5kOnJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDYwJSA0MCUgYXQgNDAlIDAlLHJnYmEoMjU1LDEwNyw2MSwwLjA0KSx0cmFuc3BhcmVudCA2MCUpOwp9Ci5tYXAtY2FyZC1oZWFkZXJ7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjE2cHggMjBweCAwOwp9Ci5tYXAtY2FyZC10aXRsZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtY2FyZC1tZXRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0taW5rLWRpbSl9Ci5sZWdlbmQtYmFye2hlaWdodDo0cHg7d2lkdGg6MTAwcHg7Ym9yZGVyLXJhZGl1czoycHg7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQsIzE1Mjg0MCwjMmU3YTllIDMwJSwjYjg3ZDMwIDYwJSwjZTgzZjIwIDgwJSwjZmYxYTNhKX0KCi5tYXAtY29udHJvbHN7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjEwcHggMjBweCA4cHg7Cn0KLm1hcC1jb250cm9scy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0taW5rLWZhaW50KX0KLmxheWVyLXRhYnN7ZGlzcGxheTpmbGV4O2dhcDozcHh9Ci5sYXllci10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0taW5rLWZhaW50KTtwYWRkaW5nOjRweCAxMHB4O2JvcmRlci1yYWRpdXM6NHB4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLmxheWVyLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDEwNyw2MSwwLjA5KTtib3JkZXItY29sb3I6cmdiYSgyNTUsMTA3LDYxLDAuMjIpfQoubGF5ZXItdGFiOmhvdmVye2NvbG9yOnZhcigtLWluay1kaW0pfQoKLm1hcC13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjEwMCU7CiAgcGFkZGluZzowIDE2cHggMTZweDsKfQoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjEwMCU7YXNwZWN0LXJhdGlvOjEvMTttYXgtaGVpZ2h0OjY2MHB4Owp9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrfQojaW5kaWEtbWFwIC5zdGF0ZXsKICBjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmZpbHRlciAwLjJzLHN0cm9rZSAwLjJzOwp9CiNpbmRpYS1tYXAgLnN0YXRlOmhvdmVyewogIHN0cm9rZTojZmZmICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjEuMiAhaW1wb3J0YW50OwogIGZpbHRlcjpkcm9wLXNoYWRvdygwIDAgMTJweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpIGJyaWdodG5lc3MoMS4yKTsKfQojaW5kaWEtbWFwIC5zdGF0ZS5zZWxlY3RlZHsKICBzdHJva2U6I2ZmZiAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjYgIWltcG9ydGFudDsKICBmaWx0ZXI6ZHJvcC1zaGFkb3coMCAwIDE2cHggcmdiYSgyNTUsMjU1LDI1NSwwLjQpKSBicmlnaHRuZXNzKDEuMyk7Cn0KLnB1bHNlLXJpbmd7ZmlsbDpub25lO3BvaW50ZXItZXZlbnRzOm5vbmU7YW5pbWF0aW9uOnB1bHNlIDIuNHMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHVsc2V7MCV7cjo2O29wYWNpdHk6MC42O3N0cm9rZS13aWR0aDoxLjV9MTAwJXtyOjI4O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4zfX0KLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDYsOCwxMywwLjk2KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyLXN0cm9uZyk7Ym9yZGVyLXJhZGl1czo4cHg7CiAgcGFkZGluZzoxMHB4IDE0cHg7b3BhY2l0eTowO3RyYW5zaXRpb246b3BhY2l0eSAwLjEyczt6LWluZGV4OjEwO21pbi13aWR0aDoxNjBweDsKfQoudHQtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7bWFyZ2luLWJvdHRvbTo2cHh9Ci50dC1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmstZGltKTttYXJnaW4tdG9wOjNweH0KLnR0LXJvdyBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKX0KCi8qIOKUgOKUgCBTVEFURSBQQU5FTCAocmlnaHQgY29sLCByb3cgMSkg4pSA4pSAICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmZhY2UpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7cGFkZGluZzoxOHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpOwogIGdyaWQtY29sdW1uOjI7Z3JpZC1yb3c6MTsKICBvdmVyZmxvdy15OmF1dG87bWF4LWhlaWdodDo3NjBweDsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjRweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlci1zdHJvbmcpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAxNnB4O2NvbG9yOnZhcigtLWluay1mYWludCl9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjI1O21hcmdpbi1ib3R0b206MTRweH0KLnBhbmVsLWVtcHR5IC50e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTdweDtjb2xvcjp2YXIoLS1pbmstZGltKTttYXJnaW4tYm90dG9tOjZweH0KLnBhbmVsLWVtcHR5IC5ze2ZvbnQtc2l6ZToxMnB4O2xpbmUtaGVpZ2h0OjEuNn0KCi5zdGF0ZS1oZWFkZXJ7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNHB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoucy1la3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWluay1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NHB4fQoucy1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtsaW5lLWhlaWdodDoxfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyLXN0cm9uZyk7Y29sb3I6dmFyKC0taW5rLWRpbSk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4ycztwYWRkaW5nOjA7ZmxleC1zaHJpbms6MDsKfQouZmF2LWJ0bjpob3Zlcntjb2xvcjp2YXIoLS1pbmspO2JvcmRlci1jb2xvcjp2YXIoLS1pbmstZGltKX0KLmZhdi1idG4uYWN0aXZle2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjU1LDEwNyw2MSwwLjM1KTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDEwNyw2MSwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHh9Cgouc2NvcmUtcm93ewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHg7CiAgcGFkZGluZzo5cHggMTJweDtib3JkZXItcmFkaXVzOjdweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMjUpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNjb3JlLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWluay1mYWludCl9Ci5zY29yZS12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyNnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wM2VtfQouc2NvcmUtZGVsdGF7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzozcHggN3B4O2JvcmRlci1yYWRpdXM6NHB4O21hcmdpbi1sZWZ0OmF1dG99Ci5zY29yZS1kZWx0YS51cHtjb2xvcjojZmY4YTViO2JhY2tncm91bmQ6cmdiYSgyNTUsMTA3LDYxLDAuMSl9Ci5zY29yZS1kZWx0YS5kbntjb2xvcjojNGNjOWYwO2JhY2tncm91bmQ6cmdiYSg3NiwyMDEsMjQwLDAuMSl9CgouaW5zaWdodC1ib3h7CiAgYm9yZGVyLWxlZnQ6MnB4IHNvbGlkIHZhcigtLWFjY2VudCk7CiAgcGFkZGluZzoxMHB4IDEycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwxMDcsNjEsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDdweCA3cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzcHg7bGluZS1oZWlnaHQ6MS41NTtjb2xvcjp2YXIoLS1pbmspOwogIGZvbnQtc3R5bGU6aXRhbGljO2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLnN1Yi10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0taW5rLWZhaW50KTttYXJnaW46MTRweCAwIDhweDsKICBkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Cn0KLm5hcnJhdGl2ZXN7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6N3B4fQoubmFycmF0aXZle2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIGF1dG87Z2FwOjZweDthbGlnbi1pdGVtczpjZW50ZXJ9Ci5uYXItbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmFyLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1pbmstZGltKX0KLm5hci10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O292ZXJmbG93OmhpZGRlbjttYXJnaW4tdG9wOi00cHh9Ci5uYXItZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuNnN9CgoucmYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjhweH0KLnJmLWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5yZi1ibG9jayAucmh7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0taW5rLWZhaW50KTttYXJnaW4tYm90dG9tOjdweH0KLnJmLWJsb2NrLnVwIC5yaHtjb2xvcjp2YXIoLS1yaXNlKX0KLnJmLWJsb2NrLmRuIC5yaHtjb2xvcjp2YXIoLS1mYWxsKX0KLnJmLWJsb2NrIC5yaXtmb250LXNpemU6MTFweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0taW5rLWRpbSl9Ci5yZi1ibG9jayAucmk6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLnJmLWJsb2NrIC5yaSBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTEuNXB4fQoucmYtYmxvY2sgLnJpIHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluay1mYWludCl9CgouZW1vdGlvbi1yb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweH0KLmVtb3Rpb24tZG9udXR7d2lkdGg6ODBweDtoZWlnaHQ6ODBweDtmbGV4LXNocmluazowfQouZW1vdGlvbi1sZWdlbmR7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweDtmb250LXNpemU6MTFweH0KLmVpe2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVzd3t3aWR0aDo2cHg7aGVpZ2h0OjZweDtib3JkZXItcmFkaXVzOjJweDtmbGV4LXNocmluazowfQouZW57ZmxleDoxO2NvbG9yOnZhcigtLWluay1kaW0pfQouZXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0taW5rKX0KCi50bC1jaGFydHtoZWlnaHQ6ODBweDttYXJnaW4tdG9wOjRweH0KCi5hcnRpY2xlc3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnRpY2xlewogIGRpc3BsYXk6ZmxleDtnYXA6OHB4O3BhZGRpbmc6OHB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydGljbGU6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDI1KTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyLXN0cm9uZyl9Ci5hLXNyY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDhweDtwYWRkaW5nLXRvcDoxcHh9Ci5hLXR4dHtmb250LXNpemU6MTEuNXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1pbmspfQoKLyog4pSA4pSAIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XIChiZWxvdyBtYXAgKyBwYW5lbCkg4pSA4pSAICovCi5uYXJyYXRpdmUtcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTUyMHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDMycHggMjRweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoyMHB4Owp9CgoubmFyLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmYWNlKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjsKfQoubmFyLWNhcmQtaGVhZGVyewogIHBhZGRpbmc6MTRweCAxOHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLm5hci1jYXJkLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLm5hci1jYXJkLW1ldGF7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLm5hci1jYXJkLWJvZHl7cGFkZGluZzoxNHB4IDE4cHh9CgovKiBSaXNpbmcvRmFsbGluZyBuYXJyYXRpdmVzICovCi5tb20taXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubW9tLWl0ZW06Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1yYW5re2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO3dpZHRoOjE0cHg7ZmxleC1zaHJpbms6MH0KLm1vbS1pbmZve2ZsZXg6MX0KLm1vbS1uYW1le2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5tb20tc3RhdGVze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0taW5rLWZhaW50KTttYXJnaW4tdG9wOjFweH0KLm1vbS1wY3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6NTAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGN0LnJpc2V7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tb20tcGN0LmZhbGx7Y29sb3I6dmFyKC0tZmFsbCl9Ci5tb20tdHJhY2t7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDttYXJnaW4tdG9wOjRweDtvdmVyZmxvdzpoaWRkZW59Ci5tb20tZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweH0KCi8qIFJlZ2lvbmFsIHNoaWZ0cyAqLwoucmVnLWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDoxMHB4O2FsaWduLWl0ZW1zOmNlbnRlcjsKICBwYWRkaW5nOjlweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzOwp9Ci5yZWctaXRlbTpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0ZW06aG92ZXJ7b3BhY2l0eTowLjh9Ci5yZWctYmFkZ2V7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIHBhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDEwNyw2MSwwLjA4KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjU1LDEwNyw2MSwwLjE1KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmxvd3tmbGV4OjE7Zm9udC1zaXplOjEycHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4O2ZsZXgtd3JhcDp3cmFwfQoucmVnLWZyb217Y29sb3I6dmFyKC0taW5rLWRpbSl9Ci5yZWctYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjY7Zm9udC1zaXplOjEzcHh9Ci5yZWctdG97Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5yZWctdGltZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWluay1mYWludCk7ZmxleC1zaHJpbms6MH0KCi8qIOKUgOKUgCBGQVZPUklURVMg4pSA4pSAICovCi5mYXZzLXNlY3Rpb257CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNTIwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzJweCAzMnB4Owp9Ci5zZWMtaGVhZHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1pbmstZmFpbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouZmF2LXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdi1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFye2hlaWdodDozcHh9Ci5mYXYtcm93Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlci1zdHJvbmcpO2JvcmRlci1yYWRpdXM6MnB4fQouZmF2LWNhcmR7CiAgZmxleDowIDAgMjAwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmYWNlKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxMXB4O3BhZGRpbmc6MTJweDtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4czsKfQouZmF2LWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnJnYmEoMjU1LDEwNyw2MSwwLjI1KTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDEwNyw2MSwwLjAyNSl9Ci5mYXYtaGVhZHtkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6YmFzZWxpbmU7bWFyZ2luLWJvdHRvbTo3cHh9Ci5mYXYtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHh9Ci5mYXYtc2NvcmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpfQouZmF2LXJvdzJ7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0taW5rLWRpbSk7bWFyZ2luLXRvcDozcHh9Ci5mYXYtcm93MiAudntjb2xvcjp2YXIoLS1pbmspO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4fQouZmF2LWVtcHR5e3BhZGRpbmc6MTJweDtjb2xvcjp2YXIoLS1pbmstZmFpbnQpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtc3R5bGU6aXRhbGljfQoKLyog4pSA4pSAIEZPT1RFUiDilIDilIAgKi8KLmZvb3R7CiAgdGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjFlbTsKICBjb2xvcjp2YXIoLS1pbmstZmFpbnQpO3BhZGRpbmc6MjRweCAzMnB4IDQwcHg7bWF4LXdpZHRoOjY0MHB4O21hcmdpbjowIGF1dG87CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxO2xpbmUtaGVpZ2h0OjEuODsKfQoKQGtleWZyYW1lcyBmYWRlVXB7ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNXB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoubWFwLWNhcmQsLnN0YXRlLXBhbmVsLC5uYXItY2FyZCwubmFycmF0aXZlLXN0cmlwe2FuaW1hdGlvbjpmYWRlVXAgMC41cyBjdWJpYy1iZXppZXIoLjIsLjgsLjIsMSkgYmFja3dhcmRzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjA2c30KLm5hci1jYXJkOm50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC4xMnN9CgpAbWVkaWEobWF4LXdpZHRoOjExMDBweCl7CiAgLmhlcm97Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAubWFpbi1ncmlke2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgLnN0YXRlLXBhbmVse2dyaWQtcm93OjI7bWF4LWhlaWdodDpub25lfQogIC5uYXJyYXRpdmUtcm93e2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9Cn0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjwhLS0gVE9QQkFSIC0tPgo8ZGl2IGNsYXNzPSJ0b3BiYXIiPgogIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLW1hcmsiPjwvZGl2PgogICAgPGRpdiBjbGFzcz0iYnJhbmQtdGV4dCI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYW1lIj5JbmRpYSBBdHRlbnRpb24gTWFwPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0ic3ViIj5OYXJyYXRpdmUgaW50ZWxsaWdlbmNlPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXJpZ2h0Ij4KICAgIDxkaXYgY2xhc3M9ImxpdmUtcGlsbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gSEVSTzogdGl0bGUgbGVmdCwgY29tcGFjdCBuYXJyYXRpdmUgc3RyaXAgcmlnaHQgLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIj4KICA8ZGl2IGNsYXNzPSJoZXJvLWxlZnQiPgogICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+SW5kaWEgJm1pZGRvdDsgUG9saXRpY2FsIG5hcnJhdGl2ZSBpbnRlbGxpZ2VuY2U8L2Rpdj4KICAgIDxoMT5XaGF0IGlzIEluZGlhPGJyLz48ZW0+Zm9jdXNlZCBvbjwvZW0+PGJyLz5yaWdodCBub3cuPC9oMT4KICAgIDxwIGNsYXNzPSJzdWIiPk9ic2VydmUgSW5kaWEncyBjb2xsZWN0aXZlIGF0dGVudGlvbiBhY3Jvc3MgMzAgc3RhdGVzIOKAlCB0aGVuIHVuZGVyc3RhbmQgaG93IGl0cyBuYXJyYXRpdmVzIGFyZSBzaGlmdGluZyBiZW5lYXRoIHRoZSBzdXJmYWNlLjwvcD4KICA8L2Rpdj4KCiAgPCEtLSBDT01QQUNUIE5BUlJBVElWRSBQVUxTRSAtLT4KICA8ZGl2IGNsYXNzPSJuYXJyYXRpdmUtc3RyaXAiPgogICAgPGRpdiBjbGFzcz0ic3RyaXAtaGVhZGVyIj4KICAgICAgPHNwYW4gY2xhc3M9InN0cmlwLXRpdGxlIj5OYXRpb25hbCBuYXJyYXRpdmUgcHVsc2U8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9InRpbWUtdGFicyI+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0idGltZS10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0idGltZS10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJ0aW1lLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic3RyaXAtYm9keSIgaWQ9InN0cmlwLWJvZHkiPgogICAgICA8IS0tIHBvcHVsYXRlZCBieSBKUyAtLT4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIE1BSU4gR1JJRDogbWFwIGxlZnQsIHN0YXRlIHBhbmVsIHJpZ2h0IC0tPgo8ZGl2IGNsYXNzPSJtYWluLWdyaWQiPgoKICA8IS0tIE1BUCAtLT4KICA8ZGl2IGNsYXNzPSJtYXAtY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJtYXAtY2FyZC1oZWFkZXIiPgogICAgICA8ZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC1jYXJkLXRpdGxlIj5BdHRlbnRpb24gaGVhdG1hcDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC1jYXJkLW1ldGEiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IExpdmUgJm1pZGRvdDsgU2lnbmFsLXdlaWdodGVkPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJsZWdlbmQiPgogICAgICAgIDxzcGFuPkxvdzwvc3Bhbj4KICAgICAgICA8ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj4KICAgICAgICA8c3Bhbj5IaWdoPC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLWNvbnRyb2xzIj4KICAgICAgPHNwYW4gY2xhc3M9Im1hcC1jb250cm9scy1sYWJlbCI+VmlldyBieTwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibGF5ZXItdGFicyI+CiAgICAgICAgPHNwYW4gY2xhc3M9ImxheWVyLXRhYiBhY3RpdmUiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uPC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsYXllci10YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb248L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9ImxheWVyLXRhYiIgZGF0YS1sYXllcj0idmVsb2NpdHkiPlZlbG9jaXR5PC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLXdyYXAiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtc3ZnLXdyYXAiPgogICAgICAgIDxzdmcgaWQ9ImluZGlhLW1hcCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgogICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iZ0ciIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjUwJSI+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0icmdiYSgyNTUsMTA3LDYxLDAuMDMpIi8+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICAgICAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgICAgICAgPC9kZWZzPgogICAgICAgICAgPHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI4MDAiIGZpbGw9InVybCgjZ0cpIi8+CiAgICAgICAgICA8ZyBpZD0ibWFwLXN0YXRlcyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1wdWxzZXMiPjwvZz4KICAgICAgICA8L3N2Zz4KICAgICAgICA8ZGl2IGNsYXNzPSJtYXAtdG9vbHRpcCIgaWQ9InRvb2x0aXAiPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFNUQVRFIFBBTkVMIC0tPgogIDxkaXYgY2xhc3M9InN0YXRlLXBhbmVsIiBpZD0ic3RhdGUtZGV0YWlsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLWVtcHR5Ij4KICAgICAgPHN2ZyB3aWR0aD0iMzYiIGhlaWdodD0iMzYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS4yIj48cGF0aCBkPSJNMjEgMTBjMCA3LTkgMTMtOSAxM3MtOS02LTktMTNhOSA5IDAgMCAxIDE4IDB6Ii8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMCIgcj0iMyIvPjwvc3ZnPgogICAgICA8ZGl2IGNsYXNzPSJ0Ij5TZWxlY3QgYSBzdGF0ZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzIj5DbGljayBhbnkgcmVnaW9uIG9uIHRoZSBtYXAgdG8gb3BlbiBpdHMgbmFycmF0aXZlIGludGVsbGlnZW5jZSBwYW5lbC48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKPC9kaXY+Cgo8IS0tIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XIC0tPgo8ZGl2IGNsYXNzPSJuYXJyYXRpdmUtcm93Ij4KCiAgPCEtLSBSSVNJTkcgTkFSUkFUSVZFUyAtLT4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYXItY2FyZC1oZWFkZXIiPgogICAgICA8ZGl2IGNsYXNzPSJuYXItY2FyZC10aXRsZSI+UmlzaW5nIG5hcnJhdGl2ZXM8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibmFyLWNhcmQtbWV0YSI+TGFzdCA3MiBob3VycyAmbWlkZG90OyBBbGwgc3RhdGVzPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5hci1jYXJkLWJvZHkiIGlkPSJyaXNpbmctbmFyIj48L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBERUNMSU5JTkcgTkFSUkFUSVZFUyAtLT4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYXItY2FyZC1oZWFkZXIiPgogICAgICA8ZGl2IGNsYXNzPSJuYXItY2FyZC10aXRsZSI+RGVjbGluaW5nIG5hcnJhdGl2ZXM8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibmFyLWNhcmQtbWV0YSI+TGFzdCA3MiBob3VycyAmbWlkZG90OyBBbGwgc3RhdGVzPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5hci1jYXJkLWJvZHkiIGlkPSJmYWxsaW5nLW5hciI+PC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gUkVHSU9OQUwgU0hJRlRTIC0tPgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5hci1jYXJkLWhlYWRlciI+CiAgICAgIDxkaXYgY2xhc3M9Im5hci1jYXJkLXRpdGxlIj5SZWdpb25hbCBuYXJyYXRpdmUgc2hpZnRzPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im5hci1jYXJkLW1ldGEiPlN0YXRlLWxldmVsIGV2b2x1dGlvbiAmbWlkZG90OyBMYXN0IDMwIGRheXM8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmFyLWNhcmQtYm9keSIgaWQ9InJlZ2lvbmFsLXNoaWZ0cyI+PC9kaXY+CiAgPC9kaXY+Cgo8L2Rpdj4KCjwhLS0gRkFWT1JJVEVTIC0tPgo8c2VjdGlvbiBjbGFzcz0iZmF2cy1zZWN0aW9uIj4KICA8ZGl2IGNsYXNzPSJzZWMtaGVhZCI+VHJhY2tlZCBzdGF0ZXM8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmYXYtcm93IiBpZD0iZmF2LXJvdyI+CiAgICA8ZGl2IGNsYXNzPSJmYXYtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBDbGljayB0aGUgYm9va21hcmsgaWNvbiBvbiBhbnkgc3RhdGUgcGFuZWwuPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjxkaXYgY2xhc3M9ImZvb3QiPgogIEluZGlhIEF0dGVudGlvbiBNYXAgaXMgYW4gb2JzZXJ2YXRpb25hbCBuYXJyYXRpdmUgaW50ZWxsaWdlbmNlIHBsYXRmb3JtLiBJdCB0cmFja3MgY29sbGVjdGl2ZSBhdHRlbnRpb24gcGF0dGVybnMgZnJvbSBwdWJsaWMgZGF0YSBhbmQgZG9lcyBub3QgaW5mZXIgcG9saXRpY2FsIHBvc2l0aW9ucywgcHJlZGljdCBlbGVjdGlvbnMsIG9yIGVuZG9yc2UgbmFycmF0aXZlcy4gQWxsIHNpZ25hbHMgcmVmbGVjdCB2b2x1bWUgYW5kIG1vbWVudHVtIOKAlCBub3QgZWRpdG9yaWFsIHNpZ25pZmljYW5jZS4KPC9kaXY+Cgo8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L25wbS90b3BvanNvbi1jbGllbnRAMy4xLjAvZGlzdC90b3BvanNvbi1jbGllbnQubWluLmpzIj48L3NjcmlwdD4KPHNjcmlwdD4KLy8g4pSA4pSAIEFQSSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKY29uc3QgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hBbGxTdGF0ZXMoKXsKICB0cnl7CiAgICBjb25zdCByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlcycpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIGNvbnN0IHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgaWYoIVNUQVRFX0RBVEFbcm93Lm5hbWVdKSBTVEFURV9EQVRBW3Jvdy5uYW1lXT17Li4uREVGQVVMVF9TVEFURX07CiAgICAgIE9iamVjdC5hc3NpZ24oU1RBVEVfREFUQVtyb3cubmFtZV0se2F0dGVudGlvbjpyb3cuYXR0ZW50aW9uLGRlbHRhOnJvdy5kZWx0YV8yNGgsdmVsb2NpdHk6cm93LnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246cm93LmRvbWluYW50X2Vtb3Rpb24sZG9taW5hbnRfbmFycmF0aXZlOnJvdy5kb21pbmFudF9uYXJyYXRpdmV9KTsKICAgIH0pOwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTmFycmF0aXZlTW9tZW50dW0oKTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbQVBJXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU3RhdGVEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgY29uc3Qgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChuYW1lKSk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgY29uc3QgZD1hd2FpdCByLmpzb24oKTsKICAgIFNUQVRFX0RBVEFbbmFtZV09ewogICAgICBhdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSwKICAgICAgZW1vdGlvbnM6ZC5lbW90aW9uc3x8REVGQVVMVF9TVEFURS5lbW90aW9ucywKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSwKICAgICAgc3VtbWFyeTpkLnN1bW1hcnl8fERFRkFVTFRfU1RBVEUuc3VtbWFyeSwKICAgICAgYXJ0aWNsZXM6ZC5hcnRpY2xlc3x8W10sdGltZWxpbmU6ZC50aW1lbGluZXx8REVGQVVMVF9TVEFURS50aW1lbGluZSwKICAgIH07CiAgICByZXR1cm4gU1RBVEVfREFUQVtuYW1lXTsKICB9Y2F0Y2goZSl7cmV0dXJuIGdldFN0YXRlRGF0YShuYW1lKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcHNob3QoKXsKICB0cnl7CiAgICBjb25zdCByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgY29uc3QgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIGNvbnN0IGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7CiAgICBpZihlbCYmZC50b3RhbF9zaWduYWxzKSBlbC50ZXh0Q29udGVudD1kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTsKICAgIGNvbnN0IG1ldGE9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1tZXRhJyk7CiAgICBpZihtZXRhJiZkLmFzX29mKSBtZXRhLnRleHRDb250ZW50PSczMCBzdGF0ZXMgwrcgVXBkYXRlZCAnK25ldyBEYXRlKGQuYXNfb2YpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nKSsnIMK3IFNpZ25hbC13ZWlnaHRlZCc7CiAgfWNhdGNoKGUpe30KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwc2hvdCgpXSk7CiAgdmFyIG49MDsKICB2YXIgZmFzdD1zZXRJbnRlcnZhbChhc3luYyBmdW5jdGlvbigpewogICAgbisrOwogICAgYXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTsKICAgIGF3YWl0IGZldGNoU25hcHNob3QoKTsKICAgIGlmKHNlbGVjdGVkU3RhdGUpIHJlbmRlclN0YXRlUGFuZWwoc2VsZWN0ZWRTdGF0ZSk7CiAgICBpZihuPj0xMil7CiAgICAgIGNsZWFySW50ZXJ2YWwoZmFzdCk7CiAgICAgIHNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXBzaG90KCk7aWYoc2VsZWN0ZWRTdGF0ZSlyZW5kZXJTdGF0ZVBhbmVsKHNlbGVjdGVkU3RhdGUpO30sMzAwMDAwKTsKICAgIH0KICB9LDE1MDAwKTsKfQoKLy8g4pSA4pSAIE5BUlJBVElWRSBEQVRBIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAp2YXIgTkFUSU9OQUxfU0hJRlRTPXsKICAnM20nOlsKICAgIHtmcm9tOidJbmZsYXRpb24nLGZyb21Ob3RlOidlYXNpbmcgbmF0aW9uYWxseScsdG86J0JvcmRlciBzZWN1cml0eScsdG9Ob3RlOidzdXJnaW5nIHBvc3QtaW5jaWRlbnQnfSwKICAgIHtmcm9tOidFbGVjdGlvbiByaGV0b3JpYycsZnJvbU5vdGU6J3Bvc3QtY3ljbGUgZmFkZScsdG86J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHRvTm90ZTonc3RlYWR5IG5hdGlvbmFsIHJpc2UnfSwKICAgIHtmcm9tOidGYXJtZXIgcHJvdGVzdHMnLGZyb21Ob3RlOidtb21lbnR1bSBsb3N0Jyx0bzonVW5lbXBsb3ltZW50IGFueGlldHknLHRvTm90ZToneW91dGgtc2lnbmFsIHN1cmdlJ30sCiAgXSwKICAnNm0nOlsKICAgIHtmcm9tOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZyb21Ob3RlOidwcmUtZWxlY3Rpb24gcGVhaycsdG86J0NvcnJ1cHRpb24gJiBzY2FtcycsdG9Ob3RlOidhY2NvdW50YWJpbGl0eSBjeWNsZSd9LAogICAge2Zyb206J1JlbGlnaW91cyBuYXRpb25hbGlzbScsZnJvbU5vdGU6J2hpZ2ggcGxhdGVhdScsdG86J0Vjb25vbWljIGFueGlldHknLHRvTm90ZTonY29zdC1vZi1saXZpbmcgc3VyZ2UnfSwKICAgIHtmcm9tOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZnJvbU5vdGU6J3JpYmJvbi1jdXR0aW5nIGRvbmUnLHRvOidMYXcgJiBvcmRlcicsdG9Ob3RlOidjcmltZSBuYXJyYXRpdmUgcmlzZSd9LAogIF0sCiAgJzF5JzpbCiAgICB7ZnJvbTonVW5lbXBsb3ltZW50Jyxmcm9tTm90ZTonMTItbW9udGggc3VzdGFpbmVkJyx0bzonTmF0aW9uYWxpc20nLHRvTm90ZTonZXZlbnQtZHJpdmVuIHNwaWtlcyd9LAogICAge2Zyb206J1BhbmRlbWljIHJlY292ZXJ5Jyxmcm9tTm90ZTonZmFkZWQgYnkgUTEnLHRvOidJbmZsYXRpb24nLHRvTm90ZTonZG9taW5hdGVkIG1pZC15ZWFyJ30sCiAgICB7ZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLGZyb21Ob3RlOidsYW5ndWFnZS1sZWQnLHRvOidTZWN1cml0eSAmIGJvcmRlcnMnLHRvTm90ZTonZ2VvcG9saXRpY2FsIGVzY2FsYXRpb24nfSwKICBdLAp9OwoKdmFyIFJFR0lPTkFMX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2Vla3MnfSwKICB7c3RhdGU6J0JpaGFyJyxmcm9tOidFbGVjdGlvbiByaGV0b3JpYycsdG86J1VuZW1wbG95bWVudCAmIGV4YW0gc2NhbXMnLHRpbWU6JzYgd2Vla3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2Vla3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdlZWtzJ30sCiAge3N0YXRlOidLYXJuYXRha2EnLGZyb206J01pbmluZyBjb250cm92ZXJzeScsdG86J0xhbmd1YWdlIHNpZ25hZ2UgcG9saXRpY3MnLHRpbWU6JzMgd2Vla3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyAmIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3ZWVrcyd9LAogIHtzdGF0ZTonUHVuamFiJyxmcm9tOidQb3dlciBjcmlzaXMnLHRvOidCb3JkZXIgc2VjdXJpdHkgwrcgRHJvbmVzJyx0aW1lOiczIHdlZWtzJ30sCl07Cgp2YXIgTU9DS19SSVNJTkc9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GQUxMSU5HPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbicsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBkYXRhPU5BVElPTkFMX1NISUZUU1twZXJpb2RdfHxOQVRJT05BTF9TSElGVFNbJzNtJ107CiAgdmFyIGh0bWw9Jyc7CiAgZGF0YS5mb3JFYWNoKGZ1bmN0aW9uKHMsaSl7CiAgICBpZihpPjApIGh0bWwrPSc8ZGl2IGNsYXNzPSJzdHJpcC1kaXZpZGVyIj48L2Rpdj4nOwogICAgaHRtbCs9CiAgICAgICc8ZGl2IGNsYXNzPSJzdHJpcC1yb3ciPicrCiAgICAgICAgJzxkaXYgY2xhc3M9InN0cmlwLWZyb20iPicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3RyaXAtbGFiZWwgZmFsbCI+RmFkaW5nPC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzdHJpcC10b3BpYyI+JytzLmZyb20rJzwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3RyaXAtbWV0YSI+JytzLmZyb21Ob3RlKyc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzdHJpcC1hcnJvdyI+4oaSPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3RyaXAtdG8iPicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3RyaXAtbGFiZWwgcmlzZSI+UmlzaW5nPC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzdHJpcC10b3BpYyI+JytzLnRvKyc8L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InN0cmlwLW1ldGEiPicrcy50b05vdGUrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogIH0pOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdHJpcC1ib2R5JykuaW5uZXJIVE1MPWh0bWw7Cn0KCmZ1bmN0aW9uIHJlbmRlck5hcnJhdGl2ZU1vbWVudHVtKCl7CiAgdmFyIG5hckNvdW50PXt9OwogIE9iamVjdC52YWx1ZXMoU1RBVEVfREFUQSkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pe25hckNvdW50W24ubmFtZV09KG5hckNvdW50W24ubmFtZV18fDApK24udmFsO30pOwogIH0pOwogIHZhciBzb3J0ZWQ9T2JqZWN0LmVudHJpZXMobmFyQ291bnQpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogIHZhciByaXNpbmc9c29ydGVkLnNsaWNlKDAsNSk7CiAgdmFyIGZhbGxpbmc9c29ydGVkLnNsaWNlKC01KS5yZXZlcnNlKCk7CgogIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1uYXInKTsKICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYWxsaW5nLW5hcicpOwogIHZhciBtYXg9cmlzaW5nLmxlbmd0aD9yaXNpbmdbMF1bMV06MTAwOwoKICBpZihyaXNpbmcubGVuZ3RoKXsKICAgIHJFbC5pbm5lckhUTUw9cmlzaW5nLm1hcChmdW5jdGlvbihuLGkpewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im1vbS1pdGVtIj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXJhbmsiPicrKGkrMSkrJzwvc3Bhbj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtb20taW5mbyI+PGRpdiBjbGFzcz0ibW9tLW5hbWUiPicrblswXSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPHNwYW4gY2xhc3M9Im1vbS1wY3QgcmlzZSI+4oaRPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im1vbS10cmFjayI+PGRpdiBjbGFzcz0ibW9tLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuWzFdL21heCoxMDApKyclO2JhY2tncm91bmQ6dmFyKC0tcmlzZSk7b3BhY2l0eTowLjU1Ij48L2Rpdj48L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfWVsc2V7CiAgICByRWwuaW5uZXJIVE1MPU1PQ0tfUklTSU5HLm1hcChmdW5jdGlvbihuLGkpewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im1vbS1pdGVtIj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXJhbmsiPicrKGkrMSkrJzwvc3Bhbj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtb20taW5mbyI+PGRpdiBjbGFzcz0ibW9tLW5hbWUiPicrbi5uYW1lKyc8L2Rpdj48ZGl2IGNsYXNzPSJtb20tc3RhdGVzIj4nK24uc3RhdGVzKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXBjdCByaXNlIj4nK24ucGN0Kyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibW9tLXRyYWNrIj48ZGl2IGNsYXNzPSJtb20tZmlsbCIgc3R5bGU9IndpZHRoOicrcGFyc2VJbnQobi5wY3QpKyclO2JhY2tncm91bmQ6dmFyKC0tcmlzZSk7b3BhY2l0eTowLjU1Ij48L2Rpdj48L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICBpZihmYWxsaW5nLmxlbmd0aCl7CiAgICBmRWwuaW5uZXJIVE1MPWZhbGxpbmcubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibW9tLWl0ZW0iPicrCiAgICAgICAgJzxzcGFuIGNsYXNzPSJtb20tcmFuayI+JysoaSsxKSsnPC9zcGFuPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im1vbS1pbmZvIj48ZGl2IGNsYXNzPSJtb20tbmFtZSI+JytuWzBdKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXBjdCBmYWxsIj7ihpM8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibW9tLXRyYWNrIj48ZGl2IGNsYXNzPSJtb20tZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG5bMV0vbWF4KjEwMCkrJyU7YmFja2dyb3VuZDp2YXIoLS1mYWxsKTtvcGFjaXR5OjAuNTUiPjwvZGl2PjwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9ZWxzZXsKICAgIGZFbC5pbm5lckhUTUw9TU9DS19GQUxMSU5HLm1hcChmdW5jdGlvbihuLGkpewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im1vbS1pdGVtIj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXJhbmsiPicrKGkrMSkrJzwvc3Bhbj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtb20taW5mbyI+PGRpdiBjbGFzcz0ibW9tLW5hbWUiPicrbi5uYW1lKyc8L2Rpdj48ZGl2IGNsYXNzPSJtb20tc3RhdGVzIj4nK24uc3RhdGVzKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0ibW9tLXBjdCBmYWxsIj4nK24ucGN0Kyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibW9tLXRyYWNrIj48ZGl2IGNsYXNzPSJtb20tZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5hYnMocGFyc2VJbnQobi5wY3QpKSsnJTtiYWNrZ3JvdW5kOnZhcigtLWZhbGwpO29wYWNpdHk6MC41NSI+PC9kaXY+PC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLXNoaWZ0cycpLmlubmVySFRNTD1SRUdJT05BTF9TSElGVFMubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJyZWctaXRlbSIgb25jbGljaz0ic2VsZWN0U3RhdGUoXCcnK3Muc3RhdGUrJ1wnKSI+JysKICAgICAgJzxzcGFuIGNsYXNzPSJyZWctYmFkZ2UiPicrcy5zdGF0ZSsnPC9zcGFuPicrCiAgICAgICc8ZGl2IGNsYXNzPSJyZWctZmxvdyI+JysKICAgICAgICAnPHNwYW4gY2xhc3M9InJlZy1mcm9tIj4nK3MuZnJvbSsnPC9zcGFuPicrCiAgICAgICAgJzxzcGFuIGNsYXNzPSJyZWctYXJyIj7ihpI8L3NwYW4+JysKICAgICAgICAnPHNwYW4gY2xhc3M9InJlZy10byI+JytzLnRvKyc8L3NwYW4+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8c3BhbiBjbGFzcz0icmVnLXRpbWUiPicrcy50aW1lKyc8L3NwYW4+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CgovLyB0aW1lIHRhYiBzd2l0Y2hlcgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcudGltZS10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHRhYil7CiAgdGFiLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnRpbWUtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0KXt0LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgdGFiLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogICAgcmVuZGVyU3RyaXAodGFiLmRhdGFzZXQucGVyaW9kKTsKICB9KTsKfSk7CgovLyDilIDilIAgU1RBVEUgREFUQSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKdmFyIFNUQVRFX0RBVEE9ewogICJCaWhhciI6e2F0dGVudGlvbjo4NyxkZWx0YTo2LHZlbG9jaXR5OjAuMzQsZW1vdGlvbnM6e2FueGlldHk6MzIsYW5nZXI6MjgsaG9wZToxNCxwcmlkZTo4LGZlYXI6MTh9LG5hcnJhdGl2ZXM6W3tuYW1lOiJVbmVtcGxveW1lbnQiLHZhbDo0MSxkaXI6InVwIn0se25hbWU6IkNvcnJ1cHRpb24iLHZhbDoyOCxkaXI6InVwIn0se25hbWU6IkNhc3RlIHBvbGl0aWNzIix2YWw6MTIsZGlyOiJmbGF0In0se25hbWU6IkVkdWNhdGlvbiIsdmFsOjExLGRpcjoidXAifSx7bmFtZToiTWlncmF0aW9uIix2YWw6OCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiRXhhbSBzY2FtIG91dHJhZ2UiLHBjdDoiKzQ3JSJ9LHt0OiJNaWdyYW50IHJldHVybiIscGN0OiIrMjIlIn1dLGZhbGxpbmc6W3t0OiJFbGVjdGlvbiBzcGVlY2hlcyIscGN0OiItMzElIn1dLGFydGljbGVzOlt7c3JjOiJEYWluaWsgQmhhc2thciIsdHh0OiJQYXRuYSBzdHVkZW50cyBzdGFnZSBwcm90ZXN0IGRlbWFuZGluZyBjYW5jZWxsYXRpb24gb2YgZXhhbSJ9LHtzcmM6IlRoZSBIaW5kdSIsdHh0OiJCaWhhcidzIHlvdXRoIHVuZW1wbG95bWVudCBjcmlzaXMgaW50ZW5zaWZpZXMgYXMgcHJpdmF0ZSBzZWN0b3IgaGlyaW5nIHNsb3dzIn0se3NyYzoiUFRJIix0eHQ6IlN0YXRlIGFubm91bmNlcyBoaWdoLWxldmVsIGlucXVpcnkgaW50byBhbGxlZ2VkIHBhcGVyIGxlYWsifV0sc3VtbWFyeToiQmloYXIncyBuYXJyYXRpdmUgc2hpZnRlZCBzaGFycGx5IHRvd2FyZCB1bmVtcGxveW1lbnQgYW5kIGV4YW0gYW54aWV0eS4gVGhlIGNvcnJ1cHRpb24tZWR1Y2F0aW9uIGludGVyc2VjdGlvbiBpcyBkb21pbmF0aW5nIGJvdGggSGluZGkgYW5kIEVuZ2xpc2ggcHJlc3MuIix0aW1lbGluZTpbNjIsNjQsNjYsNzEsNzQsNzgsODEsODddfSwKICAiTWFoYXJhc2h0cmEiOnthdHRlbnRpb246NjIsZGVsdGE6LTgsdmVsb2NpdHk6LTAuMTIsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MjIsaG9wZToxNixwcmlkZToyNCxmZWFyOjIwfSxuYXJyYXRpdmVzOlt7bmFtZToiTGFuZ3VhZ2UgcG9saXRpY3MiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MjEsZGlyOiJmbGF0In0se25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoxOCxkaXI6ImRvd24ifSx7bmFtZToiUmVnaW9uYWwgaWRlbnRpdHkiLHZhbDoxOSxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImRvd24ifV0scmlzaW5nOlt7dDoiTWFyYXRoaSBsYW5ndWFnZSByb3ciLHBjdDoiKzE5JSJ9XSxmYWxsaW5nOlt7dDoiT25pb24gcHJpY2VzIixwY3Q6Ii0yNCUifV0sYXJ0aWNsZXM6W3tzcmM6Ikxva21hdCIsdHh0OiJMYW5ndWFnZSBlbmZvcmNlbWVudCBkZWJhdGUgcmVpZ25pdGVzIGluIHN1YnVyYmFuIE11bWJhaSJ9LHtzcmM6IkluZGlhbiBFeHByZXNzIix0eHQ6Ik9uaW9uIHByaWNlcyBzdGFiaWxpemUgYXMgZmFybWVyIHByb3Rlc3QgYWN0aXZpdHkgc3Vic2lkZXMifV0sc3VtbWFyeToiTWFoYXJhc2h0cmEgY29vbGluZyBhcyBlY29ub21pYyBhbnhpZXRpZXMgZWFzZS4gTmFycmF0aXZlIHJvdGF0aW5nIHRvd2FyZCBsYW5ndWFnZSBhbmQgcmVnaW9uYWwgaWRlbnRpdHkuIix0aW1lbGluZTpbNzgsNzUsNzMsNzAsNjksNjYsNjQsNjJdfSwKICAiVXR0YXIgUHJhZGVzaCI6e2F0dGVudGlvbjo3OCxkZWx0YTozLHZlbG9jaXR5OjAuMDgsZW1vdGlvbnM6e2FueGlldHk6MjIsYW5nZXI6MjQsaG9wZToxOCxwcmlkZToyMixmZWFyOjE0fSxuYXJyYXRpdmVzOlt7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoyNixkaXI6InVwIn0se25hbWU6IlJlbGlnaW9uIix2YWw6MjIsZGlyOiJmbGF0In0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTgsZGlyOiJ1cCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkNhc3RlIHBvbGl0aWNzIix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkV4cHJlc3N3YXkgb3BlbmluZyIscGN0OiIrMjglIn1dLGZhbGxpbmc6W3t0OiJGZXN0aXZhbCBsb2dpc3RpY3MiLHBjdDoiLTE4JSJ9XSxhcnRpY2xlczpbe3NyYzoiRGFpbmlrIEphZ3JhbiIsdHh0OiJHYW5nYSBFeHByZXNzd2F5IGV4dGVuc2lvbiBpbmF1Z3VyYXRlZCJ9LHtzcmM6IkFtYXIgVWphbGEiLHR4dDoiQXlvZGh5YSB0b3VyaXNtIGNyb3NzZXMgbW9udGhseSByZWNvcmQifV0sc3VtbWFyeToiVVAgc2hvd3MgYSBzdGFibGUgaGlnaC1iYXNlbGluZSDigJQgbGF3ICYgb3JkZXIgYW5kIGluZnJhc3RydWN0dXJlIGFuY2hvciB0aGUgbmFycmF0aXZlLiBSZWxpZ2lvdXMgc2VudGltZW50IHN0ZWFkeS4iLHRpbWVsaW5lOls3Miw3Myw3NCw3NSw3Niw3Nyw3Niw3OF19LAogICJUYW1pbCBOYWR1Ijp7YXR0ZW50aW9uOjcxLGRlbHRhOjUsdmVsb2NpdHk6MC4yMSxlbW90aW9uczp7YW54aWV0eToxNCxhbmdlcjoyNixob3BlOjE4LHByaWRlOjMwLGZlYXI6MTJ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJMYW5ndWFnZSBwb2xpdGljcyIsdmFsOjMyLGRpcjoidXAifSx7bmFtZToiRmVkZXJhbGlzbSIsdmFsOjIxLGRpcjoidXAifSx7bmFtZToiRWR1Y2F0aW9uIix2YWw6MTYsZGlyOiJ1cCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IlJlZ2lvbmFsIGlkZW50aXR5Iix2YWw6MTUsZGlyOiJ1cCJ9XSxyaXNpbmc6W3t0OiJORVAgdGhyZWUtbGFuZ3VhZ2Ugcm93IixwY3Q6IiszOCUifSx7dDoiU3RhdGUgZnVuZHMgZGlzcHV0ZSIscGN0OiIrMjElIn1dLGZhbGxpbmc6W3t0OiJDeWNsb25lIGFmdGVybWF0aCIscGN0OiItMjklIn1dLGFydGljbGVzOlt7c3JjOiJUaGUgSGluZHUiLHR4dDoiVGhyZWUtbGFuZ3VhZ2UgZm9ybXVsYSBkZWJhdGUgZW50ZXJzIGZyZXNoIHBoYXNlIn0se3NyYzoiRGluYW1hbGFyIix0eHQ6IlN0dWRlbnQgZmVkZXJhdGlvbnMgcGFzcyByZXNvbHV0aW9ucyBhZ2FpbnN0IGxhbmd1YWdlIHBvbGljeSJ9XSxzdW1tYXJ5OiJUYW1pbCBOYWR1IHBpdm90ZWQgZmlybWx5IHRvIGxhbmd1YWdlIHBvbGl0aWNzIGFuZCBmZWRlcmFsaXNtLiBUaGUgdGhyZWUtbGFuZ3VhZ2UgZGViYXRlIGlzIHRoZSBzaW5nbGUgbGFyZ2VzdCBzaWduYWwgaW4gVGFtaWwtbGFuZ3VhZ2UgbWVkaWEuIix0aW1lbGluZTpbNjQsNjUsNjYsNjgsNjksNzAsNzAsNzFdfSwKICAiV2VzdCBCZW5nYWwiOnthdHRlbnRpb246NzQsZGVsdGE6NCx2ZWxvY2l0eTowLjE1LGVtb3Rpb25zOnthbnhpZXR5OjIwLGFuZ2VyOjI4LGhvcGU6MTQscHJpZGU6MTYsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IkxhdyAmIG9yZGVyIix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJSZWxpZ2lvbiIsdmFsOjE5LGRpcjoidXAifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjE4LGRpcjoidXAifSx7bmFtZToiQ29ycnVwdGlvbiIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTcsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkJvcmRlciBpbmZpbHRyYXRpb24gZGViYXRlIixwY3Q6IisyNCUifV0sZmFsbGluZzpbe3Q6IkJ5cG9sbCByZXN1bHRzIixwY3Q6Ii0yMiUifV0sYXJ0aWNsZXM6W3tzcmM6IkFuYW5kYWJhemFyIix0eHQ6IkJTRiBhbmQgc3RhdGUgcG9saWNlIGVzY2FsYXRlIGNvb3JkaW5hdGlvbiBuZWFyIEJvbmdhb24ifSx7c3JjOiJUZWxlZ3JhcGggSW5kaWEiLHR4dDoiU1NDIHJlY3J1aXRtZW50IGNhc2U6IENCSSBmaWxlcyBmcmVzaCBjaGFyZ2Ugc2hlZXQifV0sc3VtbWFyeToiQmVuZ2FsIGhlYXZ5IG9uIGxhdy1hbmQtb3JkZXIgYW5kIGJvcmRlciBuYXJyYXRpdmVzLiBTY2hvb2wgcmVjcnVpdG1lbnQgY2FzZSBjb250aW51ZXMgZ2VuZXJhdGluZyBjb3ZlcmFnZS4iLHRpbWVsaW5lOls2OCw2OSw3MCw3MSw3Miw3Myw3Myw3NF19LAogICJLYXJuYXRha2EiOnthdHRlbnRpb246NjgsZGVsdGE6Mix2ZWxvY2l0eTowLjA5LGVtb3Rpb25zOnthbnhpZXR5OjE2LGFuZ2VyOjIwLGhvcGU6MjIscHJpZGU6MjQsZmVhcjoxOH0sbmFycmF0aXZlczpbe25hbWU6Ikxhbmd1YWdlIHBvbGl0aWNzIix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJFY29ub215IC8gSVQiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IldhdGVyIGRpc3B1dGVzIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTksZGlyOiJ1cCJ9LHtuYW1lOiJSZWdpb25hbCBpZGVudGl0eSIsdmFsOjE3LGRpcjoidXAifV0scmlzaW5nOlt7dDoiS2FubmFkYSBzaWduYWdlIHJ1bGUiLHBjdDoiKzIyJSJ9XSxmYWxsaW5nOlt7dDoiTWluaW5nIGlucXVpcnkiLHBjdDoiLTEzJSJ9XSxhcnRpY2xlczpbe3NyYzoiRGVjY2FuIEhlcmFsZCIsdHh0OiJCZW5nYWx1cnUgY2l2aWMgYm9keSBzdGVwcyB1cCBLYW5uYWRhIHNpZ25hZ2UgZW5mb3JjZW1lbnQifV0sc3VtbWFyeToiS2FybmF0YWthIGJhbGFuY2luZyBlY29ub21pYyBvcHRpbWlzbSB3aXRoIHJlZ2lvbmFsLWlkZW50aXR5IHBvbGl0aWNzLiBMYW5ndWFnZSBlbmZvcmNlbWVudCBjb3ZlcmFnZSByaXNpbmcgc3RhdGV3aWRlLiIsdGltZWxpbmU6WzY0LDY1LDY1LDY2LDY3LDY3LDY4LDY4XX0sCiAgIktlcmFsYSI6e2F0dGVudGlvbjo1NCxkZWx0YToxLHZlbG9jaXR5OjAuMDQsZW1vdGlvbnM6e2FueGlldHk6MTQsYW5nZXI6MTgsaG9wZToyMixwcmlkZToyNixmZWFyOjIwfSxuYXJyYXRpdmVzOlt7bmFtZToiR292ZXJuYW5jZSIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJFZHVjYXRpb24iLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MTgsZGlyOiJ1cCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MjIsZGlyOiJkb3duIn0se25hbWU6IlJlbGlnaW9uIix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IldheWFuYWQgcmVoYWJpbGl0YXRpb24iLHBjdDoiKzE2JSJ9XSxmYWxsaW5nOlt7dDoiVG91cmlzbSBkZWJhdGUiLHBjdDoiLTklIn1dLGFydGljbGVzOlt7c3JjOiJNYXRocnViaHVtaSIsdHh0OiJXYXlhbmFkIHJlaGFiaWxpdGF0aW9uIHBoYXNlIHR3byBmYWNlcyBsYW5kIGFjcXVpc2l0aW9uIGRlbGF5cyJ9XSxzdW1tYXJ5OiJLZXJhbGEgbWFpbnRhaW5zIG1vZGVyYXRlIHN0YWJsZSBhdHRlbnRpb24uIFdheWFuYWQgcmVjb3Zlcnkgc3RlYWR5LiBFY29ub21pYyBhbnhpZXR5IGFyb3VuZCByZW1pdHRhbmNlcyByaXNpbmcgc2xvd2x5LiIsdGltZWxpbmU6WzUwLDUxLDUyLDUyLDUzLDUzLDU0LDU0XX0sCiAgIkRlbGhpIjp7YXR0ZW50aW9uOjgxLGRlbHRhOjksdmVsb2NpdHk6MC4yNyxlbW90aW9uczp7YW54aWV0eToyOCxhbmdlcjoyNixob3BlOjEyLHByaWRlOjE0LGZlYXI6MjB9LG5hcnJhdGl2ZXM6W3tuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MjYsZGlyOiJ1cCJ9LHtuYW1lOiJFbnZpcm9ubWVudCAvIEFpciIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoxOCxkaXI6InVwIn0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkNvcnJ1cHRpb24iLHZhbDoxNixkaXI6InVwIn1dLHJpc2luZzpbe3Q6IkFpciBxdWFsaXR5IGVtZXJnZW5jeSIscGN0OiIrNDElIn0se3Q6IkwtRyB2cyBDTSBzdGFuZG9mZiIscGN0OiIrMjMlIn1dLGZhbGxpbmc6W3t0OiJNZXRybyBmYXJlIGhpa2UiLHBjdDoiLTEyJSJ9XSxhcnRpY2xlczpbe3NyYzoiSW5kaWFuIEV4cHJlc3MiLHR4dDoiRGVsaGkgQVFJIHJlLWVudGVycyBzZXZlcmUgYmFuZDsgR1JBUCBTdGFnZSAzIGFjdGl2YXRlZCJ9LHtzcmM6IkhpbmR1c3RhbiBUaW1lcyIsdHh0OiJMLUcgd3JpdGVzIHRvIENNIGNpdGluZyBhZG1pbmlzdHJhdGl2ZSBkZWxheXMifV0sc3VtbWFyeToiRGVsaGkgY2xpbWJpbmcgc2hhcnBseSBvbiBvZmYtc2Vhc29uIGFpciBxdWFsaXR5IGVtZXJnZW5jeSBhbmQgcmVuZXdlZCBnb3Zlcm5hbmNlIHN0YW5kb2ZmLiBBbnhpZXR5IGRvbWluYW50IGZvciBzZWNvbmQgd2Vlay4iLHRpbWVsaW5lOls2OCw3MCw3Miw3NCw3Niw3OCw3OSw4MV19LAogICJHdWphcmF0Ijp7YXR0ZW50aW9uOjU5LGRlbHRhOi0yLHZlbG9jaXR5Oi0wLjA1LGVtb3Rpb25zOnthbnhpZXR5OjE0LGFuZ2VyOjE0LGhvcGU6MjQscHJpZGU6MzIsZmVhcjoxNn0sbmFycmF0aXZlczpbe25hbWU6IkVjb25vbXkiLHZhbDozMCxkaXI6InVwIn0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MjIsZGlyOiJmbGF0In0se25hbWU6IlJlbGlnaW9uIix2YWw6MTQsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjE2LGRpcjoidXAifV0scmlzaW5nOlt7dDoiU2VtaWNvbmR1Y3RvciBwbGFudCIscGN0OiIrMTglIn1dLGZhbGxpbmc6W3t0OiJTdGF0dWUgdG91cmlzbSIscGN0OiItMTQlIn1dLGFydGljbGVzOlt7c3JjOiJTYW5kZXNoIix0eHQ6IkRob2xlcmEgU0lSIGFkZHMgc2VtaWNvbmR1Y3RvciBhbmNob3IgdGVuYW50cyJ9XSxzdW1tYXJ5OiJHdWphcmF0IG5hcnJhdGl2ZSByZW1haW5zIGVjb25vbWljIGFuZCBpbmZyYXN0cnVjdHVyZS1sZWQuIFByaWRlIGlzIHRoZSBkb21pbmFudCBlbW90aW9uYWwgdG9uZS4iLHRpbWVsaW5lOls2Miw2MSw2MSw2MCw2MCw2MCw1OSw1OV19LAogICJSYWphc3RoYW4iOnthdHRlbnRpb246NTcsZGVsdGE6MSx2ZWxvY2l0eTowLjAzLGVtb3Rpb25zOnthbnhpZXR5OjE4LGFuZ2VyOjE4LGhvcGU6MTgscHJpZGU6MjAsZmVhcjoyNn0sbmFycmF0aXZlczpbe25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiSGVhdCB3YXZlIHdhcm5pbmdzIixwY3Q6IiszNCUifSx7dDoiV2VzdGVybiBib3JkZXIgYWxlcnRzIixwY3Q6IisxOSUifV0sZmFsbGluZzpbe3Q6IlRvdXJpc20gb2ZmLXNlYXNvbiIscGN0OiItMjIlIn1dLGFydGljbGVzOlt7c3JjOiJSYWphc3RoYW4gUGF0cmlrYSIsdHh0OiJCb3JkZXIgZGlzdHJpY3RzIHNlZSBmcmVzaCBzZWN1cml0eSBkcmlsbHMifV0sc3VtbWFyeToiUmFqYXN0aGFuIGF0dGVudGlvbiByaXNpbmcgb24gZW52aXJvbm1lbnQg4oCUIGhlYXQgd2F2ZSBjb3ZlcmFnZSBzcGlraW5nLiBCb3JkZXIgaXNzdWVzIGEgc3RlYWR5IHNlY29uZGFyeSBuYXJyYXRpdmUuIix0aW1lbGluZTpbNTUsNTUsNTYsNTYsNTYsNTcsNTcsNTddfSwKICAiTWFkaHlhIFByYWRlc2giOnthdHRlbnRpb246NTIsZGVsdGE6MCx2ZWxvY2l0eTowLjAxLGVtb3Rpb25zOnthbnhpZXR5OjE4LGFuZ2VyOjE2LGhvcGU6MjAscHJpZGU6MjIsZmVhcjoyNH0sbmFycmF0aXZlczpbe25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IlJlbGlnaW9uIix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjIwLGRpcjoidXAifV0scmlzaW5nOlt7dDoiU295YmVhbiBNU1AgZGViYXRlIixwY3Q6IisxNCUifV0sZmFsbGluZzpbe3Q6IkNhYmluZXQgZXhwYW5zaW9uIixwY3Q6Ii0xOCUifV0sYXJ0aWNsZXM6W3tzcmM6IlBhdHJpa2EiLHR4dDoiU295YmVhbiBncm93ZXJzIGluIE1hbHdhIGRlbWFuZCBNU1AgcmV2aWV3In1dLHN1bW1hcnk6Ik1QIHN0YWJsZSB3aXRoIGFncmljdWx0dXJlLWVjb25vbXkgbmFycmF0aXZlcyBkb21pbmF0aW5nLiBObyBzaGFycCBtb3ZlbWVudCB0aGlzIGN5Y2xlLiIsdGltZWxpbmU6WzUxLDUyLDUyLDUyLDUyLDUyLDUyLDUyXX0sCiAgIlB1bmphYiI6e2F0dGVudGlvbjo2NixkZWx0YTozLHZlbG9jaXR5OjAuMTEsZW1vdGlvbnM6e2FueGlldHk6MjIsYW5nZXI6MjYsaG9wZToxMixwcmlkZToyMixmZWFyOjE4fSxuYXJyYXRpdmVzOlt7bmFtZToiRmFybWVyIGlzc3VlcyIsdmFsOjI4LGRpcjoidXAifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoxOCxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxNCxkaXI6ImZsYXQifSx7bmFtZToiUmVsaWdpb24iLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiU3R1YmJsZSBwb2xpY3kiLHBjdDoiKzIxJSJ9LHt0OiJCb3JkZXIgZHJvbmUgc2lnaHRpbmdzIixwY3Q6IisxNyUifV0sZmFsbGluZzpbe3Q6IlBvd2VyIGNyaXNpcyIscGN0OiItMTQlIn1dLGFydGljbGVzOlt7c3JjOiJQdW5qYWJpIFRyaWJ1bmUiLHR4dDoiU3R1YmJsZSBtYW5hZ2VtZW50IHBsYW4gdW52ZWlsZWQgYWhlYWQgb2YgcGFkZHkgaGFydmVzdCJ9XSxzdW1tYXJ5OiJQdW5qYWIgcmlzaW5nIG9uIHR3aW4gdHJhY2tzOiBhZ3JpY3VsdHVyZSBwb2xpY3kgYW5kIGJvcmRlciBzZWN1cml0eS4gQW5nZXIgcmVtYWlucyBkb21pbmFudC4iLHRpbWVsaW5lOls2Miw2Myw2Myw2NCw2NCw2NSw2NSw2Nl19LAogICJIYXJ5YW5hIjp7YXR0ZW50aW9uOjYxLGRlbHRhOjIsdmVsb2NpdHk6MC4wNyxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoyMixob3BlOjE4LHByaWRlOjIyLGZlYXI6MjB9LG5hcnJhdGl2ZXM6W3tuYW1lOiJGYXJtZXIgaXNzdWVzIix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjIwLGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJDYXN0ZSBwb2xpdGljcyIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJTcG9ydHMiLHZhbDoyMCxkaXI6InVwIn1dLHJpc2luZzpbe3Q6Ik5DUiBwb2xsdXRpb24gc3BpbGxvdmVyIixwY3Q6IisxOSUifV0sZmFsbGluZzpbe3Q6IlJlYWwgZXN0YXRlIixwY3Q6Ii04JSJ9XSxhcnRpY2xlczpbe3NyYzoiRGFpbmlrIEphZ3JhbiIsdHh0OiJIYXJ5YW5hIHdyZXN0bGVycyByZWFjaCBuYXRpb25hbCBmaW5hbHMifV0sc3VtbWFyeToiSGFyeWFuYSBzaG93cyBiYWxhbmNlZCBuYXJyYXRpdmUgbWl4LiBTcG9ydHMgcHJpZGUgYW5kIGZhcm1lciBpc3N1ZXMgY28tYW5jaG9yIHB1YmxpYyBhdHRlbnRpb24uIix0aW1lbGluZTpbNTgsNTksNTksNjAsNjAsNjEsNjEsNjFdfSwKICAiVGVsYW5nYW5hIjp7YXR0ZW50aW9uOjYzLGRlbHRhOjQsdmVsb2NpdHk6MC4xMyxlbW90aW9uczp7YW54aWV0eToxNixhbmdlcjoxOCxob3BlOjIyLHByaWRlOjIyLGZlYXI6MjJ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJFY29ub215IC8gSVQiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IldhdGVyIGRpc3B1dGVzIix2YWw6MTgsZGlyOiJ1cCJ9LHtuYW1lOiJFZHVjYXRpb24iLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiUmVnaW9uYWwgaWRlbnRpdHkiLHZhbDoyMCxkaXI6InVwIn1dLHJpc2luZzpbe3Q6Ikh5ZGVyYWJhZCBjbHVzdGVyIGV4cGFuc2lvbiIscGN0OiIrMTYlIn1dLGZhbGxpbmc6W3t0OiJPbGQgY2l0eSByZWRldmVsb3BtZW50IixwY3Q6Ii0xMSUifV0sYXJ0aWNsZXM6W3tzcmM6IlNha3NoaSIsdHh0OiJUZWxhbmdhbmEgYWRkcyBpbnZlc3RtZW50IGNvbW1pdG1lbnRzIGF0IGdsb2JhbCBzdW1taXQifV0sc3VtbWFyeToiVGVsYW5nYW5hIHJpc2luZyBvbiBlY29ub21pYyBhbmQgZ292ZXJuYW5jZSB0cmFja3MuIEh5ZGVyYWJhZCB0ZWNoIG5hcnJhdGl2ZSBkb21pbmFudC4iLHRpbWVsaW5lOls1OCw1OSw2MCw2MCw2MSw2Miw2Miw2M119LAogICJBbmRocmEgUHJhZGVzaCI6e2F0dGVudGlvbjo1OCxkZWx0YToyLHZlbG9jaXR5OjAuMDgsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MTgsaG9wZToyMCxwcmlkZToyMixmZWFyOjIyfSxuYXJyYXRpdmVzOlt7bmFtZToiRWNvbm9teSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiQ2FwaXRhbCByb3ciLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IldhdGVyIGRpc3B1dGVzIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQW1hcmF2YXRpIHJlc3RhcnQiLHBjdDoiKzIyJSJ9XSxmYWxsaW5nOlt7dDoiTGlxdW9yIHBvbGljeSIscGN0OiItOSUifV0sYXJ0aWNsZXM6W3tzcmM6IkFuZGhyYSBKeW90aHkiLHR4dDoiQW1hcmF2YXRpIGNhcGl0YWwgY2l0eSBjb25zdHJ1Y3Rpb24gcmVzdGFydCBmb3JtYWxseSBhcHByb3ZlZCJ9XSxzdW1tYXJ5OiJBbmRocmEgcmlzaW5nIG9uIEFtYXJhdmF0aSByZXN0YXJ0IGFubm91bmNlbWVudC4gUHJpZGUgYW5kIGhvcGUgYXJlIHJpc2luZyBlbW90aW9uYWwgdG9uZXMuIix0aW1lbGluZTpbNTUsNTYsNTYsNTcsNTcsNTgsNTgsNThdfSwKICAiT2Rpc2hhIjp7YXR0ZW50aW9uOjQ5LGRlbHRhOi0xLHZlbG9jaXR5Oi0wLjAyLGVtb3Rpb25zOnthbnhpZXR5OjE2LGFuZ2VyOjE0LGhvcGU6MjQscHJpZGU6MjQsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IkVjb25vbXkiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IlRyaWJhbCBpc3N1ZXMiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQ3ljbG9uZSBwcmVwYXJlZG5lc3MiLHBjdDoiKzE4JSJ9XSxmYWxsaW5nOlt7dDoiTWluaW5nIGJpZCByb3VuZCIscGN0OiItMTIlIn1dLGFydGljbGVzOlt7c3JjOiJEaGFyaXRyaSIsdHh0OiJQcmUtbW9uc29vbiBjeWNsb25lIGRyaWxsIGFjcm9zcyBjb2FzdGFsIGRpc3RyaWN0cyJ9XSxzdW1tYXJ5OiJPZGlzaGEgcXVpZXQsIGRvbWluYXRlZCBieSBlbnZpcm9ubWVudGFsIHByZXBhcmVkbmVzcy4gQ29vbGluZyBtYXJnaW5hbGx5LiIsdGltZWxpbmU6WzUwLDUwLDUwLDUwLDQ5LDQ5LDQ5LDQ5XX0sCiAgIkpoYXJraGFuZCI6e2F0dGVudGlvbjo1NixkZWx0YTozLHZlbG9jaXR5OjAuMTIsZW1vdGlvbnM6e2FueGlldHk6MjAsYW5nZXI6MjIsaG9wZToxNCxwcmlkZToxNixmZWFyOjI4fSxuYXJyYXRpdmVzOlt7bmFtZToiVHJpYmFsIGlzc3VlcyIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiTWluaW5nIC8gRWNvbm9teSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiTWlncmF0aW9uIix2YWw6MTgsZGlyOiJ1cCJ9XSxyaXNpbmc6W3t0OiJGb3Jlc3QgcmlnaHRzIGRpc3B1dGUiLHBjdDoiKzI0JSJ9XSxmYWxsaW5nOlt7dDoiQ29hbCBibG9jayBhdWN0aW9uIixwY3Q6Ii0xMSUifV0sYXJ0aWNsZXM6W3tzcmM6IlByYWJoYXQgS2hhYmFyIix0eHQ6IlRyaWJhbCBvcmdhbml6YXRpb25zIHN0YWdlIHJhbGx5IG9uIGxhbmQgcmlnaHRzIn1dLHN1bW1hcnk6IkpoYXJraGFuZCBjbGltYmluZyBvbiB0cmliYWwgbGFuZCByaWdodHMgYW5kIG1pbmluZy1lY29ub215IGludGVyc2VjdGlvbnMuIix0aW1lbGluZTpbNTIsNTMsNTQsNTQsNTUsNTUsNTYsNTZdfSwKICAiQ2hoYXR0aXNnYXJoIjp7YXR0ZW50aW9uOjUxLGRlbHRhOjEsdmVsb2NpdHk6MC4wNCxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxOCxob3BlOjE4LHByaWRlOjIwLGZlYXI6MjZ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiVHJpYmFsIGlzc3VlcyIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQmFzdGFyIG9wZXJhdGlvbnMiLHBjdDoiKzE2JSJ9XSxmYWxsaW5nOlt7dDoiRm9yZXN0IGF1Y3Rpb24iLHBjdDoiLTklIn1dLGFydGljbGVzOlt7c3JjOiJQYXRyaWthIix0eHQ6IlNlY3VyaXR5IG9wZXJhdGlvbnMgaW4gQmFzdGFyIGxlYWQgdG8gcmVjb3JkIHN1cnJlbmRlciBudW1iZXJzIn1dLHN1bW1hcnk6IkNoaGF0dGlzZ2FyaCBhbmNob3JlZCBpbiBCYXN0YXIgc2VjdXJpdHkgb3BlcmF0aW9ucy4gRmVhciBkb21pbmFudCBpbiBzb3V0aGVybiBkaXN0cmljdHMuIix0aW1lbGluZTpbNTAsNTAsNTAsNTAsNTEsNTEsNTEsNTFdfSwKICAiQXNzYW0iOnthdHRlbnRpb246NjAsZGVsdGE6Myx2ZWxvY2l0eTowLjExLGVtb3Rpb25zOnthbnhpZXR5OjIyLGFuZ2VyOjIwLGhvcGU6MTgscHJpZGU6MjAsZmVhcjoyMH0sbmFycmF0aXZlczpbe25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IlJlbGlnaW9uIix2YWw6MjAsZGlyOiJ1cCJ9LHtuYW1lOiJFbnZpcm9ubWVudCAvIEZsb29kcyIsdmFsOjIwLGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJSZWdpb25hbCBpZGVudGl0eSIsdmFsOjIwLGRpcjoidXAifV0scmlzaW5nOlt7dDoiTlJDIHVwZGF0ZSBwdXNoIixwY3Q6IisxNyUifV0sZmFsbGluZzpbe3Q6IlRlYSBpbmR1c3RyeSIscGN0OiItOCUifV0sYXJ0aWNsZXM6W3tzcmM6IkFzb21peWEgUHJhdGlkaW4iLHR4dDoiU3RhdGUgYW5ub3VuY2VzIGZyZXNoIHB1c2ggb24gTlJDIHZlcmlmaWNhdGlvbiB0aW1lbGluZXMifV0sc3VtbWFyeToiQXNzYW0gcmlzaW5nIG9uIGRvY3VtZW50YXRpb24gcG9saXRpY3MgYW5kIGZsb29kIHByZXBhcmVkbmVzcy4iLHRpbWVsaW5lOls1Nyw1OCw1OCw1OSw1OSw2MCw2MCw2MF19LAogICJIaW1hY2hhbCBQcmFkZXNoIjp7YXR0ZW50aW9uOjM4LGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxNixhbmdlcjoxNCxob3BlOjI0LHByaWRlOjI4LGZlYXI6MTh9LG5hcnJhdGl2ZXM6W3tuYW1lOiJUb3VyaXNtIix2YWw6MjYsZGlyOiJmbGF0In0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkVkdWNhdGlvbiIsdmFsOjE4LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJDaGFyIERoYW0gcm9hZCB3b3JrIixwY3Q6IisxMSUifV0sZmFsbGluZzpbe3Q6IkhvdGVsIHJhdGUgY2FwIixwY3Q6Ii03JSJ9XSxhcnRpY2xlczpbe3NyYzoiQW1hciBVamFsYSIsdHh0OiJUb3VyaXN0IGFycml2YWxzIHRvIGhpbGwgc3RhdGlvbnMgcmlzZSBhaGVhZCBvZiBzdW1tZXIgcGVhayJ9XSxzdW1tYXJ5OiJIaW1hY2hhbCBpbiBsb3ctYXR0ZW50aW9uIHN1bW1lciBjYWRlbmNlLiBUb3VyaXNtIGVjb25vbXkgZG9taW5hdGVzLiIsdGltZWxpbmU6WzM4LDM4LDM4LDM4LDM4LDM4LDM4LDM4XX0sCiAgIlV0dGFyYWtoYW5kIjp7YXR0ZW50aW9uOjQxLGRlbHRhOjEsdmVsb2NpdHk6MC4wMyxlbW90aW9uczp7YW54aWV0eToxNixhbmdlcjoxNCxob3BlOjIyLHByaWRlOjI4LGZlYXI6MjB9LG5hcnJhdGl2ZXM6W3tuYW1lOiJSZWxpZ2lvbiAvIFBpbGdyaW1hZ2UiLHZhbDoyNixkaXI6InVwIn0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJUb3VyaXNtIix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiSW5mcmFzdHJ1Y3R1cmUiLHZhbDoxNixkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQ2hhciBEaGFtIHlhdHJhIHByZXAiLHBjdDoiKzE5JSJ9XSxmYWxsaW5nOlt7dDoiUG93ZXIgdGFyaWZmIGRlYmF0ZSIscGN0OiItOSUifV0sYXJ0aWNsZXM6W3tzcmM6IkhpbmR1c3RhbiBIaW5kaSIsdHh0OiJDaGFyIERoYW0geWF0cmEgcmVnaXN0cmF0aW9ucyBjcm9zcyBsYXN0IHllYXIncyByZWNvcmQifV0sc3VtbWFyeToiVXR0YXJha2hhbmQgc3RlYWR5LCByaXNpbmcgb24gcGlsZ3JpbWFnZSBhbmQgdG91cmlzbS4gUHJpZGUgZG9taW5hbnQuIix0aW1lbGluZTpbMzksNDAsNDAsNDAsNDEsNDEsNDEsNDFdfSwKICAiTWFuaXB1ciI6e2F0dGVudGlvbjo2NCxkZWx0YTo1LHZlbG9jaXR5OjAuMTksZW1vdGlvbnM6e2FueGlldHk6MjgsYW5nZXI6MjYsaG9wZToxMCxwcmlkZToxNCxmZWFyOjIyfSxuYXJyYXRpdmVzOlt7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDozMCxkaXI6InVwIn0se25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IklkZW50aXR5IC8gRXRobmljIix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTQsZGlyOiJmbGF0In0se25hbWU6Ik1pZ3JhdGlvbiIsdmFsOjEwLGRpcjoidXAifV0scmlzaW5nOlt7dDoiRXRobmljIHRlbnNpb25zIHJlc3VyZmFjZSIscGN0OiIrMjclIn0se3Q6IkFGU1BBIGRlYmF0ZSIscGN0OiIrMTQlIn1dLGZhbGxpbmc6W3t0OiJDYWJpbmV0IHNodWZmbGUgdGFsayIscGN0OiItMTMlIn1dLGFydGljbGVzOlt7c3JjOiJJbXBoYWwgRnJlZSBQcmVzcyIsdHh0OiJGcmVzaCB0ZW5zaW9ucyByZXBvcnRlZCBpbiB2YWxsZXktaGlsbHMgYm9yZGVyIHZpbGxhZ2VzIn1dLHN1bW1hcnk6Ik1hbmlwdXIgcmlzaW5nIHNoYXJwbHkgb24gZXRobmljLWlkZW50aXR5IG5hcnJhdGl2ZXMuIEFueGlldHkgYW5kIGFuZ2VyIGRvbWluYXRlLiIsdGltZWxpbmU6WzU1LDU3LDU5LDYwLDYxLDYyLDYzLDY0XX0sCiAgIkphbW11IGFuZCBLYXNobWlyIjp7YXR0ZW50aW9uOjcyLGRlbHRhOjQsdmVsb2NpdHk6MC4xNyxlbW90aW9uczp7YW54aWV0eToyMixhbmdlcjoyMixob3BlOjE4LHByaWRlOjE4LGZlYXI6MjB9LG5hcnJhdGl2ZXM6W3tuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjI2LGRpcjoidXAifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiVG91cmlzbSIsdmFsOjE4LGRpcjoidXAifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE2LGRpcjoiZmxhdCJ9LHtuYW1lOiJJZGVudGl0eSIsdmFsOjE2LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJUb3VyaXNtIGFycml2YWxzIGhpZ2giLHBjdDoiKzIxJSJ9LHt0OiJMb0MgaW5jaWRlbnRzIixwY3Q6IisxNSUifV0sZmFsbGluZzpbe3Q6IlN0YXRlaG9vZCBkZWJhdGUiLHBjdDoiLTklIn1dLGFydGljbGVzOlt7c3JjOiJHcmVhdGVyIEthc2htaXIiLHR4dDoiVG91cmlzdCBhcnJpdmFscyB0byBQYWhhbGdhbSwgR3VsbWFyZyBjcm9zcyBzZWFzb25hbCByZWNvcmQifSx7c3JjOiJEYWlseSBFeGNlbHNpb3IiLHR4dDoiTG9DIHJlbWFpbnMgYWN0aXZlOyBzZWN1cml0eSBmb3JjZXMgcmVzcG9uZCB0byBmcmVzaCBpbmNpZGVudCJ9XSxzdW1tYXJ5OiJKJksgb24gdHdpbiB0cmFja3Mg4oCUIHJlY29yZCB0b3VyaXNtIGFycml2YWxzIGFuZCByZW5ld2VkIExvQyBpbmNpZGVudHMuIix0aW1lbGluZTpbNjUsNjcsNjgsNjksNzAsNzEsNzIsNzJdfSwKICAiR29hIjp7YXR0ZW50aW9uOjM0LGRlbHRhOi0xLHZlbG9jaXR5Oi0wLjA0LGVtb3Rpb25zOnthbnhpZXR5OjE0LGFuZ2VyOjEyLGhvcGU6MjIscHJpZGU6MzAsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IlRvdXJpc20iLHZhbDozMCxkaXI6ImRvd24ifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE2LGRpcjoiZmxhdCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE0LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJDb2FzdGFsIHpvbmUgZW5mb3JjZW1lbnQiLHBjdDoiKzEyJSJ9XSxmYWxsaW5nOlt7dDoiVG91cmlzdCBhcnJpdmFscyIscGN0OiItMjElIn1dLGFydGljbGVzOlt7c3JjOiJIZXJhbGQgR29hIix0eHQ6IkNSWiBlbmZvcmNlbWVudCBkcml2ZSBiZWdpbnMgaW4gbm9ydGggR29hIn1dLHN1bW1hcnk6IkdvYSBpbiBvZmZzZWFzb24uIENvYXN0YWwgcmVndWxhdGlvbiBlbmZvcmNlbWVudCB0aGUgb25seSByaXNpbmcgc2lnbmFsLiIsdGltZWxpbmU6WzM2LDM2LDM1LDM1LDM0LDM0LDM0LDM0XX0sCiAgIlNpa2tpbSI6e2F0dGVudGlvbjoyMixkZWx0YTowLHZlbG9jaXR5OjAsZW1vdGlvbnM6e2FueGlldHk6MTQsYW5nZXI6MTAsaG9wZToyNixwcmlkZToyOCxmZWFyOjIyfSxuYXJyYXRpdmVzOlt7bmFtZToiRW52aXJvbm1lbnQiLHZhbDozMCxkaXI6InVwIn0se25hbWU6IlRvdXJpc20iLHZhbDoyMixkaXI6ImZsYXQifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkVjb25vbXkiLHZhbDoxNCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiR2xhY2llciBtb25pdG9yaW5nIixwY3Q6Iis5JSJ9XSxmYWxsaW5nOltdLGFydGljbGVzOlt7c3JjOiJTaWtraW0gRXhwcmVzcyIsdHh0OiJHbGFjaWVyIHN1cnZleSBzaG93cyBhY2NlbGVyYXRlZCByZXRyZWF0IGluIG5vcnRoIFNpa2tpbSJ9XSxzdW1tYXJ5OiJTaWtraW0gbWluaW1hbCBuYXRpb25hbCBhdHRlbnRpb24uIEVudmlyb25tZW50LWZvY3VzZWQgbmFycmF0aXZlcyBkb21pbmF0ZS4iLHRpbWVsaW5lOlsyMiwyMiwyMiwyMiwyMiwyMiwyMiwyMl19LAogICJOYWdhbGFuZCI6e2F0dGVudGlvbjoyOCxkZWx0YTowLHZlbG9jaXR5OjAuMDEsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MTQsaG9wZToyMCxwcmlkZToyNCxmZWFyOjI0fSxuYXJyYXRpdmVzOlt7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJJZGVudGl0eSIsdmFsOjI0LGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiTmFnYSBmcmFtZXdvcmsgYWdyZWVtZW50IixwY3Q6Iis4JSJ9XSxmYWxsaW5nOltdLGFydGljbGVzOlt7c3JjOiJNb3J1bmcgRXhwcmVzcyIsdHh0OiJGcmFtZXdvcmsgYWdyZWVtZW50IGltcGxlbWVudGF0aW9uIGRpc2N1c3Npb25zIHJlc3VtZSJ9XSxzdW1tYXJ5OiJOYWdhbGFuZCBxdWlldC4gSWRlbnRpdHkgYW5kIGZyYW1ld29yayBuYXJyYXRpdmVzIHN0ZWFkeSBidXQgbG93LXZvbHVtZS4iLHRpbWVsaW5lOlsyOCwyOCwyOCwyOCwyOCwyOCwyOCwyOF19LAogICJNaXpvcmFtIjp7YXR0ZW50aW9uOjE0LGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxMCxhbmdlcjo4LGhvcGU6MzAscHJpZGU6MzIsZmVhcjoyMH0sbmFycmF0aXZlczpbe25hbWU6IkdvdmVybmFuY2UiLHZhbDoyNCxkaXI6ImZsYXQifSx7bmFtZToiSWRlbnRpdHkiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkNyb3NzLWJvcmRlciByZWZ1Z2VlIHBvbGljeSIscGN0OiIrNiUifV0sZmFsbGluZzpbXSxhcnRpY2xlczpbe3NyYzoiVmFuZ2xhaW5pIix0eHQ6IlN0YXRlIHVwZGF0ZXMgcmVnaXN0cmF0aW9uIG5vcm1zIGZvciBjcm9zcy1ib3JkZXIgcmVmdWdlZXMifV0sc3VtbWFyeToiTWl6b3JhbSBxdWlldGVzdCBzdGF0ZS4gTWluaW1hbCBzaWduYWwgdm9sdW1lLiIsdGltZWxpbmU6WzE0LDE0LDE0LDE0LDE0LDE0LDE0LDE0XX0sCiAgIlRyaXB1cmEiOnthdHRlbnRpb246MzEsZGVsdGE6MSx2ZWxvY2l0eTowLjA0LGVtb3Rpb25zOnthbnhpZXR5OjE4LGFuZ2VyOjE0LGhvcGU6MjIscHJpZGU6MjIsZmVhcjoyNH0sbmFycmF0aXZlczpbe25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJJZGVudGl0eSIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE4LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJDcm9zcy1ib3JkZXIgdHJhZGUgcm91dGUiLHBjdDoiKzExJSJ9XSxmYWxsaW5nOlt7dDoiVHJpYmFsIGNvdW5jaWwgbWVldCIscGN0OiItNyUifV0sYXJ0aWNsZXM6W3tzcmM6IlRyaXB1cmEgVGltZXMiLHR4dDoiTmV3IGNyb3NzLWJvcmRlciB0cmFkZSBjb3JyaWRvciB3aXRoIEJhbmdsYWRlc2ggYW5ub3VuY2VkIn1dLHN1bW1hcnk6IlRyaXB1cmEgc2xvd2x5IHJpc2luZyBvbiBjcm9zcy1ib3JkZXIgZWNvbm9taWMgbmFycmF0aXZlcy4iLHRpbWVsaW5lOlszMCwzMCwzMCwzMSwzMSwzMSwzMSwzMV19LAogICJNZWdoYWxheWEiOnthdHRlbnRpb246MjYsZGVsdGE6MCx2ZWxvY2l0eTowLGVtb3Rpb25zOnthbnhpZXR5OjE0LGFuZ2VyOjEyLGhvcGU6MjQscHJpZGU6MjgsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IkVudmlyb25tZW50Iix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJUb3VyaXNtIix2YWw6MjIsZGlyOiJmbGF0In0se25hbWU6IklkZW50aXR5Iix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkVjb25vbXkiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE4LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJMaXZpbmcgcm9vdCBicmlkZ2UgVU5FU0NPIHB1c2giLHBjdDoiKzEwJSJ9XSxmYWxsaW5nOltdLGFydGljbGVzOlt7c3JjOiJTaGlsbG9uZyBUaW1lcyIsdHh0OiJTdGF0ZSBub21pbmF0ZXMgS2hhc2kgbGl2aW5nIHJvb3QgYnJpZGdlcyBmb3IgVU5FU0NPIGxpc3RpbmcifV0sc3VtbWFyeToiTWVnaGFsYXlhIGVudmlyb25tZW50IGFuZCB0b3VyaXNtLWxlZC4gTG93IG92ZXJhbGwgbmF0aW9uYWwgYXR0ZW50aW9uLiIsdGltZWxpbmU6WzI2LDI2LDI2LDI2LDI2LDI2LDI2LDI2XX0sCiAgIkFydW5hY2hhbCBQcmFkZXNoIjp7YXR0ZW50aW9uOjM2LGRlbHRhOjIsdmVsb2NpdHk6MC4wOSxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxNixob3BlOjE4LHByaWRlOjI0LGZlYXI6MjR9LG5hcnJhdGl2ZXM6W3tuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MzAsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiSWRlbnRpdHkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE0LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJCb3JkZXIgaW5mcmFzdHJ1Y3R1cmUgcHVzaCIscGN0OiIrMjIlIn1dLGZhbGxpbmc6W3t0OiJUb3VyaXNtIGNpcmN1aXQgbGF1bmNoIixwY3Q6Ii04JSJ9XSxhcnRpY2xlczpbe3NyYzoiQXJ1bmFjaGFsIFRpbWVzIix0eHQ6IkJvcmRlciBpbmZyYXN0cnVjdHVyZSBwcm9qZWN0cyBmYXN0LXRyYWNrZWQgYWNyb3NzIFRhd2FuZyBzZWN0b3IifV0sc3VtbWFyeToiQXJ1bmFjaGFsIHJpc2luZyBvbiBib3JkZXIgaW5mcmFzdHJ1Y3R1cmUgYW5kIG5hbWluZy1jb250cm92ZXJzeSBuYXJyYXRpdmVzLiIsdGltZWxpbmU6WzMzLDM0LDM0LDM1LDM1LDM2LDM2LDM2XX0sCn07Cgp2YXIgREVGQVVMVF9TVEFURT17CiAgYXR0ZW50aW9uOjIwLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7YW54aWV0eToxNSxhbmdlcjoxMixob3BlOjIyLHByaWRlOjI1LGZlYXI6MjZ9LAogIG5hcnJhdGl2ZXM6W3tuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MjUsZGlyOiJmbGF0In0se25hbWU6IkVjb25vbXkiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiSW5mcmFzdHJ1Y3R1cmUiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiVG91cmlzbSIsdmFsOjE5LGRpcjoiZmxhdCJ9XSwKICByaXNpbmc6W10sZmFsbGluZzpbXSwKICBhcnRpY2xlczpbe3NyYzoiUFRJIix0eHQ6IlN0YXRlIGdvdmVybmFuY2UgdXBkYXRlIHJlcG9ydGVkIGluIHJvdXRpbmUgY3ljbGUifV0sCiAgc3VtbWFyeToiTG93LWF0dGVudGlvbiByZWdpb24uIFJvdXRpbmUgZ292ZXJuYW5jZSBhbmQgZWNvbm9taWMgbmFycmF0aXZlcyB3aXRob3V0IHNpZ25pZmljYW50IG1vdmVtZW50LiIsCiAgdGltZWxpbmU6WzIwLDIwLDIwLDIwLDIwLDIwLDIwLDIwXQp9OwoKZnVuY3Rpb24gZ2V0U3RhdGVEYXRhKG4pe3JldHVybiBTVEFURV9EQVRBW25dfHxPYmplY3QuYXNzaWduKHt9LERFRkFVTFRfU1RBVEUpO30KCmZ1bmN0aW9uIGF0dGVudGlvbkNvbG9yKHMpewogIGlmKHM8MjApIHJldHVybiAnIzEyMjAzMCc7CiAgaWYoczwzNSkgcmV0dXJuICcjMWEzZjZhJzsKICBpZihzPDUwKSByZXR1cm4gJyMyNjY5OGEnOwogIGlmKHM8NjUpIHJldHVybiAnI2EwNmEyMCc7CiAgaWYoczw3OCkgcmV0dXJuICcjYzA0ZTE4JzsKICBpZihzPDg4KSByZXR1cm4gJyNkODMwMTgnOwogIHJldHVybiAnI2YwMTgyOCc7Cn0KZnVuY3Rpb24gZW1vdGlvbkNvbG9yKGUpewogIHZhciBtYXg9MCxkb209J3ByaWRlJzsKICBmb3IodmFyIGsgaW4gZSl7aWYoZVtrXT5tYXgpe21heD1lW2tdO2RvbT1rO319CiAgcmV0dXJuICh7YW54aWV0eTonIzljNWRlNScsYW5nZXI6JyNmZjRkNGQnLGhvcGU6JyM0YWRlODAnLHByaWRlOicjNGNjOWYwJyxmZWFyOicjZmZiODRkJ30pW2RvbV18fCcjNGNjOWYwJzsKfQpmdW5jdGlvbiB2ZWxvY2l0eUNvbG9yKHYpewogIGlmKHY+MC4yKSByZXR1cm4gJyNmMDE4MjgnOwogIGlmKHY+MC4xKSByZXR1cm4gJyNmZjZiM2QnOwogIGlmKHY+MC4wMikgcmV0dXJuICcjZmZiODRkJzsKICBpZih2PC0wLjA1KSByZXR1cm4gJyM0Y2M5ZjAnOwogIHJldHVybiAnIzFlMmMzYSc7Cn0KCnZhciBjdXJyZW50TGF5ZXI9J2F0dGVudGlvbicsc2VsZWN0ZWRTdGF0ZT1udWxsLGZhdm9yaXRlcz1uZXcgU2V0KCk7CgovLyDilIDilIAgTUFQIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiBidWlsZFByb2plY3Rpb24oZ2VvLHcsaCxwYWQpewogIHBhZD1wYWR8fDMwOwogIHZhciBtaW5Mb249NjguMSxtYXhMb249OTcuNCxtaW5MYXQ9Ni41LG1heExhdD0zNS43OwogIGZ1bmN0aW9uIG1ZKGxhdCl7cmV0dXJuIE1hdGgubG9nKE1hdGgudGFuKE1hdGguUEkvNCtsYXQqTWF0aC5QSS8zNjApKTt9CiAgdmFyIHlNaW49bVkobWluTGF0KSx5TWF4PW1ZKG1heExhdCksbG9uUj1tYXhMb24tbWluTG9uLHlSPXlNYXgteU1pbjsKICB2YXIgc2M9TWF0aC5taW4oKHctcGFkKjIpL2xvblIsKGgtcGFkKjIpL3lSKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLWxvblIqc2MpLzIsb3k9cGFkKyhoLXBhZCoyLXlSKnNjKS8yOwogIHJldHVybiBmdW5jdGlvbihsb24sbGF0KXtyZXR1cm4gW294Kyhsb24tbWluTG9uKSpzYywgb3krKHlNYXgtbVkobGF0KSkqc2NdO307Cn0KZnVuY3Rpb24gZ2VvVG9QYXRoKGdlb20scHJvail7CiAgdmFyIGQ9Jyc7CiAgZnVuY3Rpb24gcmluZyhjb29yZHMpe3ZhciBzPScnO2Nvb3Jkcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cHJvaihjWzBdLGNbMV0pO3MrPShpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KTtyZXR1cm4gcysnWic7fQogIGlmKGdlb20udHlwZT09PSdQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7CiAgZWxzZSBpZihnZW9tLnR5cGU9PT0nTXVsdGlQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHApe3AuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pO30pOwogIHJldHVybiBkOwp9CmZ1bmN0aW9uIGNlbnRyb2lkKGdlb20pewogIHZhciBwdHM9W107CiAgZnVuY3Rpb24gY29sKGMpe2lmKHR5cGVvZiBjWzBdPT09J251bWJlcicpIHB0cy5wdXNoKGMpOyBlbHNlIGMuZm9yRWFjaChjb2wpO30KICBjb2woZ2VvbS5jb29yZGluYXRlcyk7CiAgaWYoIXB0cy5sZW5ndGgpIHJldHVybiBbMCwwXTsKICByZXR1cm4gW3B0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzBdO30sMCkvcHRzLmxlbmd0aCxwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFsxXTt9LDApL3B0cy5sZW5ndGhdOwp9CmZ1bmN0aW9uIHN0YXRlTmFtZShwcm9wcyl7cmV0dXJuIHByb3BzLnN0X25tfHxwcm9wcy5OQU1FXzF8fHByb3BzLm5hbWV8fHByb3BzLk5BTUV8fCcnO30KCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaCgnaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L2doL3VkaXQtMDAxL2luZGlhLW1hcHMtZGF0YUBtYXN0ZXIvdG9wb2pzb24vaW5kaWEuanNvbicpOwogICAgdmFyIHRvcG89YXdhaXQgci5qc29uKCk7CiAgICB2YXIgZ2VvPXRvcG9qc29uLmZlYXR1cmUodG9wbyx0b3BvLm9iamVjdHMuc3RhdGVzKTsKICAgIGF3YWl0IG5ldyBQcm9taXNlKGZ1bmN0aW9uKHJlcyl7c2V0VGltZW91dChyZXMsMjAwKTt9KTsKICAgIHJlbmRlck1hcChnZW8pOwogICAgc2V0VGltZW91dChmdW5jdGlvbigpe3JlbmRlck1hcChnZW8pO30sOTAwKTsKICB9Y2F0Y2goZSl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLXN2Zy13cmFwJykuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjojNDQ0O3BhZGRpbmc6NDBweDt0ZXh0LWFsaWduOmNlbnRlcjtmb250LWZhbWlseTptb25vc3BhY2U7Zm9udC1zaXplOjEycHgiPk1hcCB1bmF2YWlsYWJsZTwvZGl2Pic7CiAgfQp9CgpmdW5jdGlvbiByZW5kZXJNYXAoc3RhdGVzKXsKICB2YXIgdz04MDAsaD04MDAscHJvaj1idWlsZFByb2plY3Rpb24oc3RhdGVzLHcsaCwzMCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJykscGc9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1wdWxzZXMnKTsKICBzZy5pbm5lckhUTUw9Jyc7cGcuaW5uZXJIVE1MPScnOwogIHN0YXRlcy5mZWF0dXJlcy5mb3JFYWNoKGZ1bmN0aW9uKGYpewogICAgaWYoIWYuZ2VvbWV0cnkpIHJldHVybjsKICAgIHZhciBuYW1lPXN0YXRlTmFtZShmLnByb3BlcnRpZXMpLGRhdGE9Z2V0U3RhdGVEYXRhKG5hbWUpOwogICAgdmFyIHBhdGg9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ3BhdGgnKTsKICAgIHBhdGguc2V0QXR0cmlidXRlKCdkJyxnZW9Ub1BhdGgoZi5nZW9tZXRyeSxwcm9qKSk7CiAgICBwYXRoLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aC5zZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScsbmFtZSk7CiAgICBwYXRoLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJywncmdiYSgyNTUsMjU1LDI1NSwwLjA5KScpOwogICAgcGF0aC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzAuNScpOwogICAgc2cuYXBwZW5kQ2hpbGQocGF0aCk7CiAgICB2YXIgY0xMPWNlbnRyb2lkKGYuZ2VvbWV0cnkpLGNwPXByb2ooY0xMWzBdLGNMTFsxXSk7CiAgICBpZihkYXRhLmF0dGVudGlvbj49NzApewogICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO3Jpbmcuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdwdWxzZS1yaW5nJyk7CiAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2UnLGF0dGVudGlvbkNvbG9yKGRhdGEuYXR0ZW50aW9uKSk7CiAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcxLjUnKTsKICAgICAgcmluZy5zdHlsZS5hbmltYXRpb25EZWxheT0oTWF0aC5yYW5kb20oKSoyLjIpKydzJzsKICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICB9CiAgfSk7CiAgYXBwbHlMYXllcigpOwogIGF0dGFjaEludGVyYWN0aW9ucygpOwp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHZhciBuYW1lPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWdldFN0YXRlRGF0YShuYW1lKSxmaWxsOwogICAgaWYoY3VycmVudExheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YXR0ZW50aW9uQ29sb3IoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihjdXJyZW50TGF5ZXI9PT0nZW1vdGlvbicpIGZpbGw9ZW1vdGlvbkNvbG9yKGQuZW1vdGlvbnMpOwogICAgZWxzZSBmaWxsPXZlbG9jaXR5Q29sb3IoZC52ZWxvY2l0eSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsZmlsbCk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxjdXJyZW50TGF5ZXI9PT0nYXR0ZW50aW9uJz8oMC40NStkLmF0dGVudGlvbi8yMjApOjAuNzgpOwogIH0pOwp9CgpmdW5jdGlvbiBhdHRhY2hJbnRlcmFjdGlvbnMoKXsKICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXsKICAgICAgdmFyIG5hbWU9cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9Z2V0U3RhdGVEYXRhKG5hbWUpOwogICAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgICB0aXAuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJ0dC1uYW1lIj4nK25hbWUrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InR0LXJvdyI+PHNwYW4+VG9wIG5hcnJhdGl2ZTwvc3Bhbj48c3Ryb25nPicrdG9wKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yb3ciPjxzcGFuPkF0dGVudGlvbjwvc3Bhbj48c3Ryb25nPicrZC5hdHRlbnRpb24rJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InR0LXJvdyI+PHNwYW4+MjRoPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrKGQuZGVsdGE+PTA/JyNmZjhhNWInOicjNGNjOWYwJykrJyI+JysoZC5kZWx0YT49MD8nKyc6JycpK2QuZGVsdGErJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtc3ZnLXdyYXAnKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE3NSkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTIwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0xOwogICAgfSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbGVhdmUnLGZ1bmN0aW9uKCl7dGlwLnN0eWxlLm9wYWNpdHk9MDt9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7c2VsZWN0U3RhdGUocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8g4pSA4pSAIFNUQVRFIFBBTkVMIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiBzZWxlY3RTdGF0ZShuYW1lKXsKICBzZWxlY3RlZFN0YXRlPW5hbWU7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXtwLmNsYXNzTGlzdC50b2dnbGUoJ3NlbGVjdGVkJyxwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk9PT1uYW1lKTt9KTsKICByZW5kZXJTdGF0ZVBhbmVsKG5hbWUpOwogIGZldGNoU3RhdGVEZXRhaWwobmFtZSkudGhlbihmdW5jdGlvbihkKXtpZihzZWxlY3RlZFN0YXRlPT09bmFtZSkgcmVuZGVyU3RhdGVQYW5lbChuYW1lKTt9KTsKfQoKZnVuY3Rpb24gcmVuZGVyU3RhdGVQYW5lbChuYW1lKXsKICB2YXIgZD1nZXRTdGF0ZURhdGEobmFtZSkscGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIHZhciBpc0Zhdj1mYXZvcml0ZXMuaGFzKG5hbWUpLGRTaWduPWQuZGVsdGE+PTA/JysnOicnLGRDbHM9ZC5kZWx0YT49MD8ndXAnOidkbic7CiAgdmFyIHBhbD17YW54aWV0eTonIzljNWRlNScsYW5nZXI6JyNmZjRkNGQnLGhvcGU6JyM0YWRlODAnLHByaWRlOicjNGNjOWYwJyxmZWFyOicjZmZiODRkJ307CiAgdmFyIGVMaXN0PU9iamVjdC5lbnRyaWVzKGQuZW1vdGlvbnMpLHRvdD1lTGlzdC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTQwLGN5PTQwLFI9MzUscmk9MjI7CiAgdmFyIGFyY3M9ZUxpc3QubWFwKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0sZnI9di90b3QsYTE9Y3VtQSxhMj1jdW1BK2ZyKk1hdGguUEkqMjsKICAgIGN1bUE9YTI7CiAgICB2YXIgbGc9KGEyLWExKT5NYXRoLlBJPzE6MDsKICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgIHZhciB4Mz1jeCtNYXRoLmNvcyhhMikqcmkseTM9Y3krTWF0aC5zaW4oYTIpKnJpLHg0PWN4K01hdGguY29zKGExKSpyaSx5ND1jeStNYXRoLnNpbihhMSkqcmk7CiAgICByZXR1cm4gJzxwYXRoIGQ9Ik0nK3gxLnRvRml4ZWQoMSkrJywnK3kxLnRvRml4ZWQoMSkrJyBBJytSKycsJytSKycgMCAnK2xnKycgMSAnK3gyLnRvRml4ZWQoMSkrJywnK3kyLnRvRml4ZWQoMSkrJyBMJyt4My50b0ZpeGVkKDEpKycsJyt5My50b0ZpeGVkKDEpKycgQScrcmkrJywnK3JpKycgMCAnK2xnKycgMCAnK3g0LnRvRml4ZWQoMSkrJywnK3k0LnRvRml4ZWQoMSkrJyBaIiBmaWxsPSInK3BhbFtrXSsnIiBvcGFjaXR5PSIwLjg4Ii8+JzsKICB9KS5qb2luKCcnKTsKICB2YXIgdGw9ZC50aW1lbGluZSx0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgdmFyIHR3PTI4MCx0aD02OCx0cD01OwogIHZhciBwdHM9dGwubWFwKGZ1bmN0aW9uKHYsaSl7cmV0dXJuIFt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICB2YXIgcEQ9cHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAoaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSkuam9pbignJyk7CiAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICB2YXIgYWM9YXR0ZW50aW9uQ29sb3IoZC5hdHRlbnRpb24pOwoKICBwYW5lbC5pbm5lckhUTUw9CiAgICAnPGRpdiBjbGFzcz0ic3RhdGUtaGVhZGVyIj4nKwogICAgICAnPGRpdj48ZGl2IGNsYXNzPSJzLWVrIj5TdGF0ZSBuYXJyYXRpdmUgcGFuZWw8L2Rpdj48ZGl2IGNsYXNzPSJzLW5hbWUiPicrbmFtZSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoaXNGYXY/J2FjdGl2ZSc6JycpKyciIG9uY2xpY2s9InRvZ2dsZUZhdihcJycrbmFtZSsnXCcpIj4nKwogICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoaXNGYXY/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAnPC9idXR0b24+JysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9InNjb3JlLXJvdyI+JysKICAgICAgJzxkaXY+PGRpdiBjbGFzcz0ic2NvcmUtbGFiZWwiPkF0dGVudGlvbiBpbmRleDwvZGl2PjxkaXYgY2xhc3M9InNjb3JlLXZhbCI+JytkLmF0dGVudGlvbisnPC9kaXY+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLWRlbHRhICcrZENscysnIj4nK2RTaWduK2QuZGVsdGErJyAvIDI0aDwvZGl2PicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJpbnNpZ2h0LWJveCI+JytkLnN1bW1hcnkrJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0ic3ViLXRpdGxlIj5Eb21pbmFudCBuYXJyYXRpdmVzPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJuYXJyYXRpdmVzIj4nKwogICAgICBkLm5hcnJhdGl2ZXMubWFwKGZ1bmN0aW9uKG4pewogICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFycmF0aXZlIj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9Im5hci1sYWJlbCI+JytuLm5hbWUrKG4uZGlyPT09J3VwJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojZmY4YTViO2ZvbnQtc2l6ZToxMHB4Ij7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojNGNjOWYwO2ZvbnQtc2l6ZToxMHB4Ij7ihpM8L3NwYW4+JzonJykrJzwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLXZhbCI+JytuLnZhbCsnJTwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLXRyYWNrIj48ZGl2IGNsYXNzPSJuYXItZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4udmFsKjIuNikrJyU7YmFja2dyb3VuZDonKyhuLmRpcj09PSd1cCc/JyNmZjZiM2QnOm4uZGlyPT09J2Rvd24nPycjNGNjOWYwJzonIzQ0NScpKyciPjwvZGl2PjwvZGl2PicrCiAgICAgICAgJzwvZGl2Pic7CiAgICAgIH0pLmpvaW4oJycpKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0ic3ViLXRpdGxlIj5OYXJyYXRpdmUgbW92ZW1lbnQ8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9InJmLWdyaWQiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJyZi1ibG9jayB1cCI+PGRpdiBjbGFzcz0icmgiPuKWsiBSaXNpbmc8L2Rpdj4nKwogICAgICAgIChkLnJpc2luZy5sZW5ndGg/ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0icmkiPjxzdHJvbmc+JytyLnQrJzwvc3Ryb25nPjxzcGFuPicrci5wY3QrJzwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpOic8ZGl2IGNsYXNzPSJyaSIgc3R5bGU9ImNvbG9yOnZhcigtLWluay1mYWludCkiPk5vIHJpc2luZyBzaWduYWxzPC9kaXY+JykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0icmYtYmxvY2sgZG4iPjxkaXYgY2xhc3M9InJoIj7ilrwgRmFsbGluZzwvZGl2PicrCiAgICAgICAgKGQuZmFsbGluZy5sZW5ndGg/ZC5mYWxsaW5nLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9InJpIj48c3Ryb25nPicrci50Kyc8L3N0cm9uZz48c3Bhbj4nK3IucGN0Kyc8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKTonPGRpdiBjbGFzcz0icmkiIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmstZmFpbnQpIj5ObyBmYWxsaW5nIHNpZ25hbHM8L2Rpdj4nKSsKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJzdWItdGl0bGUiPkVtb3Rpb25hbCB0b25lPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJlbW90aW9uLXJvdyI+JysKICAgICAgJzxzdmcgY2xhc3M9ImVtb3Rpb24tZG9udXQiIHZpZXdCb3g9IjAgMCA4MCA4MCI+JythcmNzKyc8L3N2Zz4nKwogICAgICAnPGRpdiBjbGFzcz0iZW1vdGlvbi1sZWdlbmQiPicrCiAgICAgICAgZUxpc3Quc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSkubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZWkiPjxzcGFuIGNsYXNzPSJlc3ciIHN0eWxlPSJiYWNrZ3JvdW5kOicrcGFsW2t2WzBdXSsnIj48L3NwYW4+PHNwYW4gY2xhc3M9ImVuIj4nK2t2WzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2t2WzBdLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImVwIj4nK01hdGgucm91bmQoa3ZbMV0qMTAwL3RvdCkrJyU8L3NwYW4+PC9kaXY+JzsKICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJzdWItdGl0bGUiPkF0dGVudGlvbiDCtyA4IGRheXM8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9InRsLWNoYXJ0Ij4nKwogICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgJyt0dysnICcrdGgrJyIgc3R5bGU9IndpZHRoOjEwMCU7aGVpZ2h0OjEwMCUiPicrCiAgICAgICAgJzxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0idGxnJytuYW1lLnJlcGxhY2UoL1xzKy9nLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMC4zIi8+JysKICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwIi8+JysKICAgICAgICAnPC9saW5lYXJHcmFkaWVudD48L2RlZnM+JysKICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbmFtZS5yZXBsYWNlKC9ccysvZywnJykrJykiLz4nKwogICAgICAgICc8cGF0aCBkPSInK3BEKyciIGZpbGw9Im5vbmUiIHN0cm9rZT0iJythYysnIiBzdHJva2Utd2lkdGg9IjEuNSIvPicrCiAgICAgICAgcHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAnPGNpcmNsZSBjeD0iJytwWzBdKyciIGN5PSInK3BbMV0rJyIgcj0iJysoaT09PXB0cy5sZW5ndGgtMT8yLjU6MS41KSsnIiBmaWxsPSInK2FjKyciLz4nO30pLmpvaW4oJycpKwogICAgICAnPC9zdmc+JysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9InN1Yi10aXRsZSI+UmVjZW50IHNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWluay1mYWludCk7Zm9udC13ZWlnaHQ6NDAwIj4nK2QuYXJ0aWNsZXMubGVuZ3RoKyc8L3NwYW4+PC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJhcnRpY2xlcyI+JysKICAgICAgZC5hcnRpY2xlcy5tYXAoZnVuY3Rpb24oYSl7cmV0dXJuICc8ZGl2IGNsYXNzPSJhcnRpY2xlIj48ZGl2IGNsYXNzPSJhLXNyYyI+JythLnNyYysnPC9kaXY+PGRpdiBjbGFzcz0iYS10eHQiPicrYS50eHQrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJykrCiAgICAnPC9kaXY+JzsKfQoKLy8g4pSA4pSAIEZBVk9SSVRFUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKZnVuY3Rpb24gdG9nZ2xlRmF2KG5hbWUpewogIGlmKGZhdm9yaXRlcy5oYXMobmFtZSkpIGZhdm9yaXRlcy5kZWxldGUobmFtZSk7IGVsc2UgZmF2b3JpdGVzLmFkZChuYW1lKTsKICByZW5kZXJTdGF0ZVBhbmVsKHNlbGVjdGVkU3RhdGUpOwogIHJlbmRlckZhdm9yaXRlcygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdm9yaXRlcygpewogIHZhciByb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Zhdi1yb3cnKTsKICBpZighZmF2b3JpdGVzLnNpemUpe3Jvdy5pbm5lckhUTUw9JzxkaXYgY2xhc3M9ImZhdi1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIENsaWNrIHRoZSBib29rbWFyayBpY29uIG9uIGFueSBzdGF0ZSBwYW5lbC48L2Rpdj4nO3JldHVybjt9CiAgcm93LmlubmVySFRNTD1BcnJheS5mcm9tKGZhdm9yaXRlcykubWFwKGZ1bmN0aW9uKG5hbWUpewogICAgdmFyIGQ9Z2V0U3RhdGVEYXRhKG5hbWUpLGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/JyNmZjhhNWInOicjNGNjOWYwJzsKICAgIHZhciB0b3A9ZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCc7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImZhdi1jYXJkIiBvbmNsaWNrPSJzZWxlY3RTdGF0ZShcJycrbmFtZSsnXCcpIj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmF2LWhlYWQiPjxzcGFuIGNsYXNzPSJmYXYtbmFtZSI+JytuYW1lKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImZhdi1zY29yZSI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYXYtcm93MiI+PHNwYW4+VG9wIG5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmF2LXJvdzIiPjxzcGFuPjI0aDwvc3Bhbj48c3BhbiBjbGFzcz0idiIgc3R5bGU9ImNvbG9yOicrZEMrJyI+JytkUytkLmRlbHRhKyc8L3NwYW4+PC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CgovLyBsYXllciBzd2l0Y2hlcgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubGF5ZXItdGFiJykuZm9yRWFjaChmdW5jdGlvbihjKXsKICBjLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmxheWVyLXRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7Y3VycmVudExheWVyPWMuZGF0YXNldC5sYXllcjthcHBseUxheWVyKCk7CiAgfSk7Cn0pOwoKLy8gY2xvY2sKZnVuY3Rpb24gdXBkYXRlQ2xvY2soKXsKICB2YXIgbm93PW5ldyBEYXRlKCksaXN0PW5ldyBEYXRlKG5vdy5nZXRUaW1lKCkrbm93LmdldFRpbWV6b25lT2Zmc2V0KCkqNjAwMDArMTk4MDAwMDApOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbG9jaycpLnRleHRDb250ZW50PVN0cmluZyhpc3QuZ2V0SG91cnMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldE1pbnV0ZXMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldFNlY29uZHMoKSkucGFkU3RhcnQoMiwnMCcpKycgSVNUJzsKfQpzZXRJbnRlcnZhbCh1cGRhdGVDbG9jaywxMDAwKTt1cGRhdGVDbG9jaygpOwoKLy8g4pSA4pSAIElOSVQg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACnJlbmRlclN0cmlwKCczbScpOwpyZW5kZXJOYXJyYXRpdmVNb21lbnR1bSgpOwpsb2FkTWFwKCk7CnNldFRpbWVvdXQoZnVuY3Rpb24oKXtzdGFydFBvbGxpbmcoKTt9LDEwMDApOwpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTVEFURV9EQVRBKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgbmFtZT10b3A/dG9wWzBdOidEZWxoaSc7CiAgaWYoZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJytuYW1lKyciXScpKSBzZWxlY3RTdGF0ZShuYW1lKTsKfSwyMjAwKTsKc2V0VGltZW91dChyZW5kZXJGYXZvcml0ZXMsMjIwMCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"

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
