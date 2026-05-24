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
    "Tamil Nadu":        ["tamil nadu", "tn", "chennai", "coimbatore", "madurai", "tirunelveli", "tamil", "dravidian", "dmk", "admk", "mk stalin"],
    "Uttar Pradesh":     ["uttar pradesh", "up", "lucknow", "varanasi", "noida", "kanpur", "ayodhya", "prayagraj", "agra", "meerut", "yogi", "yogi adityanath"],
    "Maharashtra":       ["maharashtra", "mumbai", "pune", "nagpur", "nashik", "thane", "aurangabad", "bombay", "shiv sena", "ncp", "devendra fadnavis", "uddhav"],
    "Bihar":             ["bihar", "patna", "gaya", "muzaffarpur", "bhagalpur", "nitish", "nitish kumar", "tejashwi", "rjd"],
    "West Bengal":       ["west bengal", "bengal", "kolkata", "calcutta", "howrah", "siliguri", "mamata", "mamata banerjee", "tmc", "trinamool"],
    "Karnataka":         ["karnataka", "bengaluru", "bangalore", "mysuru", "hubli", "mangaluru", "siddaramaiah", "dk shivakumar", "bs yediyurappa"],
    "Kerala":            ["kerala", "thiruvananthapuram", "kochi", "kozhikode", "wayanad", "pinarayi", "pinarayi vijayan", "cpim", "ldf", "udf"],
    "Gujarat":           ["gujarat", "ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar", "bhupendra patel"],
    "Rajasthan":         ["rajasthan", "jaipur", "jodhpur", "udaipur", "kota", "jaisalmer", "bhajan lal", "ashok gehlot"],
    "Madhya Pradesh":    ["madhya pradesh", "mp", "bhopal", "indore", "gwalior", "jabalpur", "mohan yadav"],
    "Telangana":         ["telangana", "hyderabad", "warangal", "secunderabad", "revanth reddy", "brs", "trs", "kcr"],
    "Andhra Pradesh":    ["andhra pradesh", "vijayawada", "visakhapatnam", "amaravati", "tirupati", "jagan", "jagan mohan reddy", "chandrababu", "tdp"],
    "Punjab":            ["punjab", "amritsar", "ludhiana", "chandigarh", "jalandhar", "bhagwant mann", "aap punjab"],
    "Haryana":           ["haryana", "gurugram", "gurgaon", "faridabad", "panipat", "nayab singh", "haryana govt"],
    "Odisha":            ["odisha", "orissa", "bhubaneswar", "cuttack", "puri", "mohan majhi", "naveen patnaik", "bjd"],
    "Jharkhand":         ["jharkhand", "ranchi", "jamshedpur", "dhanbad", "hemant soren", "jmm"],
    "Chhattisgarh":      ["chhattisgarh", "raipur", "bilaspur", "bastar", "vishnu deo sai"],
    "Assam":             ["assam", "guwahati", "dispur", "dibrugarh", "himanta", "himanta biswa", "nrc assam"],
    "Delhi":             ["delhi", "new delhi", "ncr", "arvind kejriwal", "aap delhi", "atishi", "rekha gupta"],
    "Jammu and Kashmir": ["kashmir", "jammu", "srinagar", "j&k", "pahalgam", "gulmarg", "omar abdullah", "nc kashmir", "article 370"],
    "Himachal Pradesh":  ["himachal", "shimla", "manali", "dharamshala", "sukhvinder", "sukhvinder sukhu"],
    "Uttarakhand":       ["uttarakhand", "dehradun", "haridwar", "char dham", "pushkar dhami"],
    "Goa":               ["goa", "panaji", "pramod sawant"],
    "Manipur":           ["manipur", "imphal", "manipuri", "meitei", "kuki", "churachandpur", "n biren singh"],
    "Nagaland":          ["nagaland", "kohima", "dimapur", "naga", "nscn", "neiphiu rio"],
    "Mizoram":           ["mizoram", "aizawl", "lunglei", "mizo", "lalduhoma", "zpm mizoram"],
    "Tripura":           ["tripura", "agartala", "tripuri", "manik saha", "tipra"],
    "Meghalaya":         ["meghalaya", "shillong", "tura", "khasi", "garo", "conrad sangma", "npp meghalaya"],
    "Arunachal Pradesh": ["arunachal", "itanagar", "tawang", "arunachal pradesh", "pema khandu", "china arunachal"],
    "Sikkim":            ["sikkim", "gangtok", "namchi", "ps golay", "skm sikkim"],
    "Andaman and Nicobar Islands": ["andaman", "nicobar", "port blair"],
    "Puducherry":        ["puducherry", "pondicherry", "n rangasamy"],
    "Dadra and Nagar Haveli and Daman and Diu": ["dadra", "daman", "diu"],
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
        title_key = sig.get("title", "")[:55].lower().strip()

        # Dedup by URL
        if url and url in self.seen_urls:
            return False

        # Dedup by title (same story from multiple sources)
        if title_key:
            state_title_key = f"{state}:{title_key}"
            if state_title_key in self.seen_urls:
                return False
            self.seen_urls.add(state_title_key)

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
    "Nagaland":          ["Nagaland Naga ceasefire Kohima", "Nagaland government NSCN", "Nagaland politics 2025"],
    "Mizoram":           ["Mizoram Aizawl governance", "Mizoram Myanmar refugee", "Mizoram ZPM government"],
    "Meghalaya":         ["Meghalaya Shillong NPP", "Meghalaya coal mining environment", "Meghalaya Conrad Sangma"],
    "Tripura":           ["Tripura Agartala BJP politics", "Tripura Bangladesh border", "Tripura Manik Saha"],
    "Manipur":           ["Manipur ethnic conflict Meitei Kuki", "Manipur Imphal violence", "Manipur Biren Singh"],
    "Arunachal Pradesh": ["Arunachal Pradesh China LAC border", "Arunachal Pradesh Pema Khandu", "Arunachal Pradesh 2025"],
    "Sikkim":            ["Sikkim Gangtok flood", "Sikkim SKM PS Golay", "Sikkim politics 2025"],
    "Assam":             ["Assam flood Brahmaputra 2025", "Assam Himanta Biswa politics", "Assam NRC citizenship"],
    "Gujarat":           ["Gujarat economy investment Ahmedabad", "Gujarat BJP Bhupendra Patel", "Gujarat news 2025"],
    "Goa":               ["Goa tourism mining politics", "Goa Pramod Sawant government", "Goa news 2025"],
    "Maharashtra":       ["Maharashtra politics Mumbai", "Maharashtra Fadnavis Shinde", "Maharashtra news 2025"],
    "Uttar Pradesh":     ["Uttar Pradesh Yogi politics", "UP Lucknow governance 2025", "Uttar Pradesh news"],
    "Karnataka":         ["Karnataka Siddaramaiah politics", "Karnataka Bengaluru 2025", "Karnataka government"],
    "West Bengal":       ["West Bengal Mamata politics", "Bengal Kolkata TMC 2025", "West Bengal news"],
    "Rajasthan":         ["Rajasthan Bhajan Lal politics", "Rajasthan Jaipur governance", "Rajasthan 2025"],
    "Madhya Pradesh":    ["Madhya Pradesh Mohan Yadav", "MP Bhopal politics 2025", "Madhya Pradesh news"],
    "Bihar":             ["Bihar Nitish Kumar politics", "Bihar Patna governance 2025", "Bihar news"],
    "Tamil Nadu":        ["Tamil Nadu MK Stalin politics", "Tamil Nadu Chennai governance", "TN DMK AIADMK 2025"],
    "Kerala":            ["Kerala Pinarayi Vijayan politics", "Kerala CPIM LDF 2025", "Kerala news"],
    "Telangana":         ["Telangana Revanth Reddy politics", "Telangana Hyderabad 2025", "Telangana Congress"],
    "Andhra Pradesh":    ["Andhra Pradesh Chandrababu TDP", "AP Jagan politics 2025", "Andhra Pradesh news"],
    "Delhi":             ["Delhi AAP BJP politics 2025", "Delhi governance Rekha Gupta", "Delhi news 2025"],
    "Punjab":            ["Punjab Bhagwant Mann AAP", "Punjab politics 2025", "Punjab news"],
    "Jharkhand":         ["Jharkhand Hemant Soren JMM", "Jharkhand politics 2025", "Jharkhand news"],
    "Odisha":            ["Odisha Mohan Majhi BJP politics", "Odisha Bhubaneswar 2025", "Odisha news"],
    "Jammu and Kashmir": ["Kashmir security Omar Abdullah", "J&K politics 2025", "Kashmir news"],
}

def build_rss_urls(state: str) -> list[str]:
    """Return 2-3 targeted RSS URLs for a state."""
    queries = STATE_QUERIES.get(state, [
        f"{state} politics government news",
        f"{state} latest news today",
    ])
    return [
        f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        for q in queries[:3]
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

        for entry in feed.entries[:20]:
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

# ── STATE BASELINE SIGNAL VOLUMES ────────────────────────────────────
# Expected daily signal count per state based on media ecosystem size
# Used for relative deviation scoring — not absolute volume
STATE_BASELINES: dict[str, float] = {
    "Uttar Pradesh": 18.0,  "Maharashtra": 16.0,   "Delhi": 15.0,
    "West Bengal": 14.0,    "Bihar": 12.0,          "Tamil Nadu": 12.0,
    "Karnataka": 11.0,      "Gujarat": 10.0,        "Rajasthan": 10.0,
    "Andhra Pradesh": 9.0,  "Madhya Pradesh": 9.0,  "Telangana": 9.0,
    "Kerala": 9.0,          "Punjab": 8.0,           "Haryana": 7.0,
    "Jharkhand": 6.0,       "Odisha": 6.0,           "Chhattisgarh": 5.0,
    "Assam": 6.0,           "Jammu and Kashmir": 7.0,"Uttarakhand": 4.0,
    "Himachal Pradesh": 3.0,"Goa": 3.0,              "Manipur": 3.0,
    "Tripura": 2.5,         "Meghalaya": 2.5,        "Nagaland": 2.0,
    "Mizoram": 2.0,         "Sikkim": 1.5,           "Arunachal Pradesh": 2.0,
    "Andaman and Nicobar Islands": 1.0, "Puducherry": 1.5,
    "Dadra and Nagar Haveli and Daman and Diu": 1.0,
}

def get_source_weight(source: str) -> float:
    """Regional sources carry more signal weight than national reposts."""
    s = source.lower()
    regional = ["morung","nagaland post","greater kashmir","chenab","sentinelassam",
        "shillong times","meghalaya","mizoram","tripura","manipur","northeast",
        "north east","seven sisters","deccan","mathrubhumi","eenadu","sakshi",
        "dainik bhaskar","dainik jagran","amar ujala","punjab kesari",
        "loksatta","sakal","divya bhaskar","rajasthan patrika","kerala kaumudi"]
    national = ["ndtv","times of india","hindustan times","india today","the hindu",
        "indian express","zee news","republic","wion","ani","pti","press trust",
        "news18","firstpost","the wire","scroll","the print","quint"]
    if any(kw in s for kw in regional):
        return 1.6
    if any(kw in s for kw in national):
        return 0.8
    return 1.0

async def recompute_state_score(state: str) -> dict:
    sigs = store.signals.get(state, [])
    now = datetime.now(timezone.utc)

    # Signal windows
    sigs_48h = [s for s in sigs if (now - s["published_at"]).total_seconds() < 172_800]
    sigs_prev = [s for s in sigs if 172_800 <= (now - s["published_at"]).total_seconds() < 345_600]

    # Source diversity → confidence
    sources = set(s.get("source", "") for s in sigs_48h)
    source_count = len(sources)
    if source_count >= 4:
        confidence, conf_weight = "HIGH", 1.0
    elif source_count >= 2:
        confidence, conf_weight = "MEDIUM", 0.75
    elif source_count == 1:
        confidence, conf_weight = "LOW", 0.45
    else:
        confidence, conf_weight = "LOW", 0.0

    # Source-weighted, duplicate-penalized signal volume
    seen_t: set[str] = set()
    weighted_volume = 0.0
    for s in sigs_48h:
        decay = 2 ** (-(now - s["published_at"]).total_seconds() / 129_600)
        src_w = get_source_weight(s.get("source", ""))
        intensity = s.get("intensity", 0.5)
        tk = s.get("title", "")[:50].lower().strip()
        if tk in seen_t:
            intensity *= 0.15  # heavy penalty for duplicate stories
        else:
            seen_t.add(tk)
        weighted_volume += decay * src_w * intensity

    # Baseline-normalized attention (deviation from expected, not raw volume)
    baseline = STATE_BASELINES.get(state, 5.0)
    raw_count = len(sigs_48h)
    deviation_ratio = raw_count / max(baseline, 0.1)
    normalized = weighted_volume * (deviation_ratio ** 0.5)
    attention = round(min(99, 100 * math.tanh(normalized / (baseline * 1.5))), 1)
    attention = round(attention * conf_weight, 1)

    # Momentum
    prev_count = len(sigs_prev)
    delta_24h = round(float(raw_count - prev_count), 1)
    velocity = round(delta_24h / max(1, prev_count), 3)
    if velocity > 0.5 and confidence in ("MEDIUM", "HIGH"):
        attention = round(min(99, attention * 1.2), 1)

    is_regional = deviation_ratio > 2.0 and raw_count < baseline * 1.5

    # Narratives
    nar_now: dict[str, int] = {}
    nar_prev_d: dict[str, int] = {}
    for s in sigs_48h:
        for n in s.get("narratives", []):
            nar_now[n] = nar_now.get(n, 0) + 1
    for s in sigs_prev:
        for n in s.get("narratives", []):
            nar_prev_d[n] = nar_prev_d.get(n, 0) + 1
    total_nar = max(1, sum(nar_now.values()))
    top_narratives = sorted(nar_now.items(), key=lambda kv: -kv[1])[:5]
    narrative_breakdown = []
    for n, c in top_narratives:
        prev = nar_prev_d.get(n, 0)
        val = round(c / total_nar * 100, 1)
        direction = "up" if (prev == 0 or c > prev * 1.1) else ("down" if c < prev * 0.9 else "flat")
        narrative_breakdown.append({"name": n, "val": val, "dir": direction})

    # Emotions
    emo_totals: dict[str, float] = {k: 0.0 for k in EMOTION_KEYWORDS}
    for s in sigs_48h:
        for k, v in s.get("emotions", {}).items():
            if k in emo_totals:
                emo_totals[k] += v
    total_emo = sum(emo_totals.values())
    emotions = {k: round(v / total_emo, 3) for k, v in emo_totals.items() if emo_totals[k] > 0} if total_emo > 0 else {}

    # Articles (deduped by source and title)
    seen_src2: set[str] = set()
    seen_t2: set[str] = set()
    articles = []
    for s in sorted(sigs_48h, key=lambda x: get_source_weight(x.get("source","")) * x.get("intensity", 0.5), reverse=True):
        src = s.get("source", "unknown")
        tk2 = s.get("title", "")[:60].lower().strip()
        if src in seen_src2 or tk2 in seen_t2:
            continue
        seen_src2.add(src)
        seen_t2.add(tk2)
        sig_emos = s.get("emotions", {})
        dom_emo = max(sig_emos.items(), key=lambda x: x[1])[0] if sig_emos else None
        articles.append({"src": src, "txt": s["title"], "url": s.get("source_url","#"),
                         "emotion": dom_emo, "narratives": s.get("narratives", [])[:2]})
        if len(articles) >= 10:
            break

    # Timeline
    hist = list(store.history.get(state, []))
    while len(hist) < 7:
        hist.insert(0, max(0.0, attention - 5))
    hist.append(attention)
    hist = hist[-8:]

    # Summary with confidence-aware prefix
    headlines = [s["title"] for s in sigs_48h[:15]]
    summary = await ai_summary(state, headlines, attention)
    if confidence == "LOW" and summary:
        summary = f"Limited signals from {state}. " + summary
    elif is_regional and summary:
        summary = f"Regional signal spike in {state}. " + summary

    dominant_emotion = max(emotions, key=lambda k: emotions[k]) if emotions else None
    dominant_narrative = top_narratives[0][0] if top_narratives else None

    return {
        "name": state, "attention": attention, "delta_24h": delta_24h,
        "velocity": velocity, "dominant_emotion": dominant_emotion,
        "dominant_narrative": dominant_narrative, "emotions": emotions,
        "narratives": narrative_breakdown, "rising": [], "falling": [],
        "summary": summary, "articles": articles, "timeline": hist,
        "signal_count": raw_count, "source_count": source_count,
        "confidence": confidence, "is_regional_story": is_regional,
        "deviation_ratio": round(deviation_ratio, 2),
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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjE0cHggMjBweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MTRweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsIHJnYmEoMjI0LDkwLDQwLDAuMDYpIDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMTAwJSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoOHB4KTsKICBtYXgtd2lkdGg6OTAwcHg7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMjVlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjEwcHg7Cn0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNHB4LDEuNHZ3LDE4cHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTtsZXR0ZXItc3BhY2luZzotMC4wMWVtOwp9Ci5zaS10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWFjY2VudCl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5tYXAtY2FyZDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNzAlIDUwJSBhdCAzNSUgMCUsIHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCA4MCUgMTAwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDMpIDAlLCB0cmFuc3BhcmVudCA2MCUpOwp9Ci5tYXAtdG9wewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTJweCAxOHB4IDA7Cn0KLm1hcC10aXRsZS1ibG9jayAubXR7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxN3B4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoubWFwLXRpdGxlLWJsb2NrIC5tc3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA2ZW07bWFyZ2luLXRvcDoycHh9Ci5sZWdlbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1kaW0pfQoubGVnZW5kLWJhcnsKICBoZWlnaHQ6M3B4O3dpZHRoOjgwcHg7Ym9yZGVyLXJhZGl1czoycHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQsIzBlMjAzNSwjMWE1NTgwIDI1JSwjOGE1YzE4IDU1JSwjYzAzODFhIDgwJSwjZTAxMDIwKTsKfQoubGF5ZXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7CiAgcGFkZGluZzoxMHB4IDIwcHggNnB4Owp9Ci5sYXllci1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5sdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjNweH0KLmx0YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjNweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQoubHRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDgpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpfQoubHRhYntkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NXB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OnZpc2libGV9Ci5sdGFiLWluZm97d2lkdGg6MTNweDtoZWlnaHQ6MTNweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4yKTtmb250LXNpemU6OHB4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc3R5bGU6aXRhbGljO2ZvbnQtd2VpZ2h0OjYwMDtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuMzUpO2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y3Vyc29yOmhlbHA7ZmxleC1zaHJpbms6MDt0cmFuc2l0aW9uOmFsbCAwLjE1cztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEwMH0KLmx0YWItaW5mbzpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjZW50KTtjb2xvcjp2YXIoLS1hY2NlbnQpfQojbHRhYi10b29sdGlwe3Bvc2l0aW9uOmZpeGVkO2JhY2tncm91bmQ6cmdiYSg4LDEyLDIwLDAuOTgpO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjEyKTtib3JkZXItcmFkaXVzOjhweDtwYWRkaW5nOjEwcHggMTNweDtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTFweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjY7d2lkdGg6MjMwcHg7d2hpdGUtc3BhY2U6bm9ybWFsO3RleHQtYWxpZ246bGVmdDtib3gtc2hhZG93OjAgOHB4IDMycHggcmdiYSgwLDAsMCwwLjYpO3BvaW50ZXItZXZlbnRzOm5vbmU7b3BhY2l0eTowO3RyYW5zaXRpb246b3BhY2l0eSAwLjE1czt6LWluZGV4Ojk5OTk5O2Rpc3BsYXk6bm9uZX0KI2x0YWItdG9vbHRpcC52aXNpYmxle29wYWNpdHk6MTtkaXNwbGF5OmJsb2NrfQoubHRhYjpob3Zlcntjb2xvcjp2YXIoLS1kaW0pfQoKLm1hcC1zdmctd3JhcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nOjEycHggMTZweCAxNnB4Owp9Ci5tYXAtaW5uZXJ7cG9zaXRpb246cmVsYXRpdmU7YXNwZWN0LXJhdGlvOjEvMTt3aWR0aDoxMDAlfQojaW5kaWEtbWFwe3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7ZGlzcGxheTpibG9jaztvdmVyZmxvdzp2aXNpYmxlfQoKLyogbWFwIHN0YXRlIHN0eWxlcyAqLwojaW5kaWEtbWFwIC5zdGF0ZXsKICBjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOmZpbHRlciAwLjI1cyBlYXNlLCBzdHJva2Utd2lkdGggMC4ycyBlYXNlLCBzdHJva2UgMC4ycyBlYXNlOwp9CiNpbmRpYS1tYXAgLnN0YXRlOmhvdmVyewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuNykgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MXB4ICFpbXBvcnRhbnQ7CiAgZmlsdGVyOmJyaWdodG5lc3MoMS4yNSkgZHJvcC1zaGFkb3coMCAwIDEwcHggcmdiYSgyNTUsMjU1LDI1NSwwLjIpKTsKfQojaW5kaWEtbWFwIC5zdGF0ZS5zZWxlY3RlZHsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjkpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjEuNHB4ICFpbXBvcnRhbnQ7CiAgZmlsdGVyOmJyaWdodG5lc3MoMS4zNSkgZHJvcC1zaGFkb3coMCAwIDE2cHggcmdiYSgyNTUsMjU1LDI1NSwwLjMpKTsKfQoKLyogYW5pbWF0ZWQgcHVsc2UgcmluZ3MgKi8KLnB1bHNlLXJpbmd7ZmlsbDpub25lO3BvaW50ZXItZXZlbnRzOm5vbmV9Ci5wdWxzZS1yaW5nLnAxe2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IGluZmluaXRlfQoucHVsc2UtcmluZy5wMnthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCAwLjlzIGluZmluaXRlfQpAa2V5ZnJhbWVzIHByewogIDAle3I6NDtvcGFjaXR5OjAuNztzdHJva2Utd2lkdGg6MS4yfQogIDEwMCV7cjoyNjtvcGFjaXR5OjA7c3Ryb2tlLXdpZHRoOjAuMn0KfQoKLyogYXRtb3NwaGVyaWMgZ2xvdyBiZWhpbmQgaG90IHN0YXRlcyAqLwouc3RhdGUtZ2xvd3twb2ludGVyLWV2ZW50czpub25lO2ZpbGw6bm9uZX0KQGtleWZyYW1lcyBnbG93UHVsc2V7MCUsMTAwJXtvcGFjaXR5OjAuMTJ9NTAle29wYWNpdHk6MC4yMn19CgoubWFwLXRvb2x0aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7cG9pbnRlci1ldmVudHM6bm9uZTsKICBiYWNrZ3JvdW5kOnJnYmEoNSw3LDEyLDAuOTUpO2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czo5cHg7CiAgcGFkZGluZzoxMnB4IDE0cHg7b3BhY2l0eTowO3RyYW5zaXRpb246b3BhY2l0eSAwLjEyczt6LWluZGV4Ojk5OTk7bWluLXdpZHRoOjE3MHB4Owp9Ci50dC1ue2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bWFyZ2luLWJvdHRvbTo4cHg7Y29sb3I6dmFyKC0taW5rKX0KLnR0LXJ7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4fQoudHQtciBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKX0KLnR0LW5hcnsKICBtYXJnaW4tdG9wOjhweDtwYWRkaW5nLXRvcDo4cHg7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTsKfQoudHQtbmFyIHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2Rpc3BsYXk6YmxvY2s7bWFyZ2luLWJvdHRvbToycHh9CgovKiBTVEFURSBQQU5FTCAqLwouc3RhdGUtcGFuZWx7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNnB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE2cHgpOwogIHBhZGRpbmc6MjBweDtvdmVyZmxvdy15OmF1dG87bWF4LWhlaWdodDo3ODBweDsKICBtaW4td2lkdGg6MDtvdmVyZmxvdy14OmhpZGRlbjsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjNweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7CiAgaGVpZ2h0OjEwMCU7bWluLWhlaWdodDozMjBweDt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjMycHggMjBweDsKfQoucGFuZWwtZW1wdHkgc3Zne29wYWNpdHk6MC4xNTttYXJnaW4tYm90dG9tOjE4cHh9Ci5wYW5lbC1lbXB0eSAucGUtdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjhweH0KLnBhbmVsLWVtcHR5IC5wZS1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW07bGluZS1oZWlnaHQ6MS43fQoKLyogc3RhdGUgcGFuZWwgaW50ZXJuYWxzICovCi5zcC1oZWFkewogIGRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIG1hcmdpbi1ib3R0b206MTZweDtwYWRkaW5nLWJvdHRvbToxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnNwLWVre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTtjb2xvcjp2YXIoLS1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NXB4fQouc3AtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI4cHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07bGluZS1oZWlnaHQ6MTtjb2xvcjp2YXIoLS1pbmspfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Y29sb3I6dmFyKC0tZmFpbnQpOwogIHdpZHRoOjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo2cHg7Y3Vyc29yOnBvaW50ZXI7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMThzO3BhZGRpbmc6MDtmbGV4LXNocmluazowOwp9Ci5mYXYtYnRuOmhvdmVye2NvbG9yOnZhcigtLWRpbSk7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5mYXYtYnRuLm9ue2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHh9CgovKiBuYXJyYXRpdmUgdGltZWxpbmUg4oCUIHRoZSBzaWduYXR1cmUgZmVhdHVyZSAqLwoubmFyLXRpbWVsaW5lewogIG1hcmdpbi1ib3R0b206MTZweDsKfQoubnQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLm50LWZsb3d7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nLWxlZnQ6MTZweDsKfQoubnQtZmxvdzo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDo1cHg7dG9wOjZweDtib3R0b206NnB4O3dpZHRoOjFweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx2YXIoLS1ib3JkZXIpKTtvcGFjaXR5OjAuNDsKfQoubnQtc3RlcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweDsKICBwYWRkaW5nOjVweCAwO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5udC1kb3R7CiAgd2lkdGg6MTBweDtoZWlnaHQ6MTBweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4LXNocmluazowOwogIHBvc2l0aW9uOmFic29sdXRlO2xlZnQ6LTE2cHg7dG9wOjdweDsKICBib3JkZXI6MS41cHggc29saWQgY3VycmVudENvbG9yO2JhY2tncm91bmQ6dmFyKC0tYmcpOwp9Ci5udC1zdGVwLnBhc3QgLm50LWRvdHtjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCl9Ci5udC1zdGVwLmN1cnJlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCAxMHB4IHJnYmEoMjI0LDkwLDQwLDAuNSl9Ci5udC1jb250ZW50e2ZsZXg6MX0KLm50LXRvcGlje2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4zfQoubnQtc3RlcC5wYXN0IC5udC10b3BpY3tjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZGltKX0KLm50LXdoZW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9CgovKiBpbnNpZ2h0IGJsb2NrICovCi5pbnNpZ2h0ewogIG1hcmdpbi1ib3R0b206MTRweDsKICBwYWRkaW5nOjEycHggMTRweCAxMnB4IDE2cHg7CiAgYm9yZGVyLWxlZnQ6MS41cHggc29saWQgdmFyKC0tYWNjZW50KTsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDMpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxMy41cHg7Zm9udC1zdHlsZTppdGFsaWM7CiAgY29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU1O2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLyogY29tcGFjdCBzY29yZSBzdHJpcCAqLwouc2NvcmUtc3RyaXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsKICBwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6N3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgbWFyZ2luLWJvdHRvbToxNHB4Owp9Ci5zcy1pdGVte2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjJweH0KLnNzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5zcy12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2NvbG9yOnZhcigtLWluayl9Ci5zcy1kZWx0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjJweCA3cHg7Ym9yZGVyLXJhZGl1czozcHh9Ci5zcy1kZWx0YS51cHtjb2xvcjojZTA2MDMwO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKX0KLnNzLWRlbHRhLmRue2NvbG9yOiMzYmI4ZDg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKX0KLnNzLWRpdmlkZXJ7d2lkdGg6MXB4O2hlaWdodDozMnB4O2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowfQouc3MtbmFye2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDB9Cgouc3Atc2VjdGlvbnttYXJnaW4tYm90dG9tOjE0cHh9Ci5zcC1zZWMtdGl0bGV7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206OXB4Owp9CgovKiBuYXJyYXRpdmVzICovCi5uYXItbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9Ci5uYXItaXRlbTJ7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgYXV0bztnYXA6NnB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLm5pLWxhYmVse2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKX0KLm5pLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5uaS10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi10b3A6LTNweH0KLm5pLWZpbGx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHg7dHJhbnNpdGlvbjp3aWR0aCAwLjdzfQoKLyogbW92ZW1lbnQgKi8KLm12LWdyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDo3cHh9Ci5tdi1ibG9ja3tiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6N3B4O3BhZGRpbmc6OXB4fQoubXYtaHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206N3B4fQoubXYtYmxvY2sudXAgLm12LWh7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tdi1ibG9jay5kbiAubXYtaHtjb2xvcjp2YXIoLS1mYWxsKX0KLm12LWl0e2ZvbnQtc2l6ZToxMC41cHg7cGFkZGluZzo0cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2NvbG9yOnZhcigtLWZhaW50KX0KLm12LWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tdi1pdCBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTFweH0KLm12LWl0IHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweH0KCi8qIGVtb3Rpb24gKi8KLmVtLXJvd3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQouZW0tZG9udXR7d2lkdGg6NzZweDtoZWlnaHQ6NzZweDtmbGV4LXNocmluazowfQouZW0tbGVne2ZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHh9Ci5lbS1pdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVtLXN3e3dpZHRoOjZweDtoZWlnaHQ6NnB4O2JvcmRlci1yYWRpdXM6MnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1ue2ZsZXg6MTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5lbS1we2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmspfQoKLyogdGltZWxpbmUgY2hhcnQgKi8KLnRsLXdyYXB7aGVpZ2h0OjcycHh9CgovKiBhcnRpY2xlcyAqLwouYXJ0LWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4fQouYXJ0LWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzo3cHggOXB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydC1pdGVtOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5hcnQtc3Jje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO3dpZHRoOjQ0cHg7cGFkZGluZy10b3A6MXB4fQouYXJ0LXR4dHtmb250LXNpemU6MTFweDtsaW5lLWhlaWdodDoxLjQ7Y29sb3I6dmFyKC0tZGltKX0KCi8qIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XICovCi5uYXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoxOHB4Owp9Ci5uYXItY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTRweCk7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNnB4IDIwcHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowOwp9Ci5uYy1ib2R5e3BhZGRpbmc6OHB4IDIwcHggMTZweDtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO30KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAzNnB4IDYwcHg7bWF4LXdpZHRoOjU4MHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZm9vdC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luLWJvdHRvbToxNHB4fQouZm9vdC1saW5le2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS44O21hcmdpbi1ib3R0b206MTJweH0KLmZvb3Qtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpfQoKLyogYW5pbWF0aW9ucyAqLwpAa2V5ZnJhbWVzIGZhZGVVcHtmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWSg2cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci5tYXAtY2FyZCwuc3RhdGUtcGFuZWwsLm5hci1jYXJkLC5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb246ZmFkZVVwIDAuNTVzIGN1YmljLWJlemllciguMiwuOCwuMiwxKSBiYWNrd2FyZHN9Ci5uYXItY2FyZDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMDdzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDMpe2FuaW1hdGlvbi1kZWxheTowLjE0c30KLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbi1kZWxheTowLjA1c30KCkBtZWRpYShtYXgtd2lkdGg6MTEwMHB4KXsKICAubWFpbntncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQogIC5zdGF0ZS1wYW5lbHttYXgtaGVpZ2h0Om5vbmV9CiAgLm5hci1yb3d7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KfQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PgoKPGRpdiBpZD0ibHRhYi10b29sdGlwIj48L2Rpdj4KPGRpdiBjbGFzcz0idG9wYmFyIj4KICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC1tYXJrIj48c3BhbiBjbGFzcz0iYnJhbmQtcHVsc2UtZG90Ij48L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC10ZXh0LWJsb2NrIj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLW5hbWUiPjxlbSBjbGFzcz0iYnJhbmQtcHVsc2Utd29yZCI+UHVsc2U8L2VtPiBvZiBJbmRpYTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXIiPgogICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgPHNwYW4gaWQ9ImxpdmUtY291bnQiPuKApjwvc3Bhbj4gc2lnbmFscwogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSIKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7CiAgYmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpOwogIHBhZGRpbmc6MCAzNnB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOwoiPgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLXNpZ25hbHMiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlNpZ25hbHMgdHJhY2tlZDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIj5MaXZlIGluZ2VzdGlvbjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1ob3R0ZXN0IiBzdHlsZT0iY3Vyc29yOnBvaW50ZXIiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWhvdHRlc3QtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj5DbGljayB0byBleHBsb3JlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWFuZ2VyLXN1YiI+T3V0cmFnZSAmIHByb3Rlc3Qgc2lnbmFsczwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlRvcCByaXNpbmcgbmFycmF0aXZlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1uYXJyYXRpdmUtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hcnJhdGl2ZS1zdWIiPk5hdGlvbmFsIHNpZ25hbCBzdXJnZTwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbGluZy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbGluZy1zdWIiPlNpZ25hbCBkZWNheTwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KLnN0YXQtY2VsbHsKICBmbGV4OjE7cGFkZGluZzoxMHB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5zdGF0LWNlbGw6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpfQouc3RhdC1kaXZ7d2lkdGg6MXB4O2JhY2tncm91bmQ6cmdiYSgxNjAsMTkwLDIzMCwwLjA3KTtmbGV4LXNocmluazowO21hcmdpbjo4cHggMH0KLnNjLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW4tdG9wOjFweH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KPC9zdHlsZT4KCgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJzaWduYXR1cmUtaW5zaWdodCIgc3R5bGU9Im1hcmdpbi10b3A6MDtmbGV4OjE7bWluLXdpZHRoOjAiPgogICAgICA8ZGl2IGNsYXNzPSJzaS1sYWJlbCI+V2hhdCBJbmRpYSBpcyB0YWxraW5nIGFib3V0IOKAlCByaWdodCBub3c8L2Rpdj4KICAgICAgPGRpdiBpZD0ic2lnLWluc2lnaHQiIHN0eWxlPSJtYXJnaW46MTJweCAwIDE0cHggMCI+CiAgICAgICAgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweCI+Q29sbGVjdGluZyBzaWduYWxzLi4uPC9zcGFuPgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBpZD0ic2lnLXRhZ3MiIHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjZweDttYXJnaW4tdG9wOjRweCI+PC9kaXY+CiAgICAgIDxkaXYgaWQ9InNpZy1tZXRhIiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KTttYXJnaW4tdG9wOjEycHg7bGV0dGVyLXNwYWNpbmc6MC4wNmVtIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZmxleDowIDAgMzYwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOyI+CiAgICAgIDwhLS0gaGVhZGVyIC0tPgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsiPgogICAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj5OYXJyYXRpdmUgc2hpZnRzPC9zcGFuPgogICAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MnB4OyI+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPCEtLSBzaGlmdHMgbGlzdCAtLT4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoxMHB4IDE0cHg7Z2FwOjZweDsiIGlkPSJzaGlmdC1saXN0Ij48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+CgoKPCEtLSBNQUlOOiBNQVAgKyBTVEFURSBQQU5FTCAtLT4KPGRpdiBjbGFzcz0ibWFpbiI+CgogIDxkaXYgY2xhc3M9Im1hcC1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im1hcC10b3AiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtdGl0bGUtYmxvY2siPgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij5JbmRpYSAmbWRhc2g7IGNvbGxlY3RpdmUgYXR0ZW50aW9uPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibXMiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IGxpdmUgc2lnbmFsIGNvbXBvc2l0ZTwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGVnZW5kIj48c3Bhbj5xdWlldDwvc3Bhbj48ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj48c3Bhbj5hY3RpdmU8L3NwYW4+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxheWVyLXJvdyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsYXllci1sYWJlbCI+Vmlldzwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibHRhYnMiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iYXR0ZW50aW9uIj5BdHRlbnRpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IldoaWNoIHN0YXRlcyBhcmUgcmVjZWl2aW5nIHRoZSBtb3N0IHB1YmxpYyBmb2N1cy4gSGlnaCBhdHRlbnRpb24gPSBjb25jZW50cmF0ZWQgbmV3cyBjb3ZlcmFnZSBhbmQgcG9saXRpY2FsIGFjdGl2aXR5LiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IlRoZSBkb21pbmFudCBlbW90aW9uYWwgdG9uZSDigJQgYW54aW91cywgYW5ncnksIGhvcGVmdWwsIHByb3VkIG9yIGZlYXJmdWwuIFJldmVhbHMgdGhlIHBzeWNob2xvZ2ljYWwgdW5kZXJjdXJyZW50IG9mIHBvbGl0aWNhbCBhdHRlbnRpb24uIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiIgZGF0YS1sYXllcj0idmVsb2NpdHkiPk1vbWVudHVtIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJJcyBhdHRlbnRpb24gcmlzaW5nIG9yIGZhbGxpbmc/IFJpc2luZyA9IG5hcnJhdGl2ZSBhY2NlbGVyYXRpbmcuIENvb2xpbmcgPSBsb3NpbmcgdHJhY3Rpb24uIFNob3dzIHN0YXRlcyBlbnRlcmluZyBvciBleGl0aW5nIGEgcG9saXRpY2FsIGN5Y2xlLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJtYXAtc3ZnLXdyYXAiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtaW5uZXIiPgogICAgICAgIDxzdmcgaWQ9ImluZGlhLW1hcCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgogICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iYW1iR2xvdyIgY3g9IjUwJSIgY3k9IjUwJSIgcj0iNTAlIj4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSJyZ2JhKDIyNCw5MCw0MCwwLjA0KSIvPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0idHJhbnNwYXJlbnQiLz4KICAgICAgICAgICAgPC9yYWRpYWxHcmFkaWVudD4KICAgICAgICAgICAgPGZpbHRlciBpZD0ic3RhdGVHbG93IiB4PSItMzAlIiB5PSItMzAlIiB3aWR0aD0iMTYwJSIgaGVpZ2h0PSIxNjAlIj4KICAgICAgICAgICAgICA8ZmVHYXVzc2lhbkJsdXIgaW49IlNvdXJjZUdyYXBoaWMiIHN0ZERldmlhdGlvbj0iOCIgcmVzdWx0PSJibHVyIi8+CiAgICAgICAgICAgICAgPGZlQ29tcG9zaXRlIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9ImJsdXIiIG9wZXJhdG9yPSJvdmVyIi8+CiAgICAgICAgICAgIDwvZmlsdGVyPgogICAgICAgICAgPC9kZWZzPgogICAgICAgICAgPHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI4MDAiIGZpbGw9InVybCgjYW1iR2xvdykiLz4KICAgICAgICAgIDxnIGlkPSJtYXAtZ2xvdyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1zdGF0ZXMiPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtcHVsc2VzIj48L2c+CiAgICAgICAgPC9zdmc+CiAgICAgICAgPGRpdiBjbGFzcz0ibWFwLXRvb2x0aXAiIGlkPSJ0b29sdGlwIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBTVEFURSBQQU5FTCAtLT4KICA8ZGl2IGNsYXNzPSJzdGF0ZS1wYW5lbCIgaWQ9InN0YXRlLWRldGFpbCI+CiAgICA8ZGl2IGNsYXNzPSJwYW5lbC1lbXB0eSI+CiAgICAgIDxzdmcgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEiPgogICAgICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0iTTEyIDh2NE0xMiAxNmguMDEiLz4KICAgICAgPC9zdmc+CiAgICAgIDxkaXYgY2xhc3M9InBlLXQiPlNlbGVjdCBhIHN0YXRlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InBlLXMiPkNsaWNrIGFueSByZWdpb24gb24gdGhlIG1hcDxici8+dG8gb3BlbiBpdHMgbmFycmF0aXZlIHBhbmVsLjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cgo8L2Rpdj4KCjwhLS0gTkFSUkFUSVZFIFJPVyAtLT4KPGRpdiBjbGFzcz0ibmFyLXJvdyI+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+PHNwYW4gY2xhc3M9Im5jLWRvdCByaXNlMiI+PC9zcGFuPjxzcGFuIGNsYXNzPSJuYy10aXRsZSI+UmlzaW5nIG5hcnJhdGl2ZXM8L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0icmlzaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+PHNwYW4gY2xhc3M9Im5jLWRvdCBmYWxsIj48L3NwYW4+PHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJkZWNsaW5pbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj48c3BhbiBjbGFzcz0ibmMtZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SZWdpb25hbCBzaGlmdHM8L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0icmVnaW9uYWwtbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIFJFUExBWSBJTkRJQSAtLT4KPHNlY3Rpb24gY2xhc3M9InJlcGxheS1zZWN0aW9uIj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktaGVhZGVyIj4KICAgIDxkaXY+PGRpdiBjbGFzcz0icmVwbGF5LWxhYmVsIj5SZXBsYXkgSW5kaWE8L2Rpdj48ZGl2IGNsYXNzPSJyZXBsYXktc3ViIj5XYXRjaCBob3cgY29sbGVjdGl2ZSBhdHRlbnRpb24gc2hpZnRlZCBvdmVyIHRpbWU8L2Rpdj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJlcGxheS1jb250cm9scyI+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biBhY3RpdmUiIGRhdGEtcGVyaW9kPSI3ZCI+NyBkYXlzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biIgZGF0YS1wZXJpb2Q9IjMwZCI+MzAgZGF5czwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4iIGRhdGEtcGVyaW9kPSI2bSI+NiBtb250aHM8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIiBkYXRhLXBlcmlvZD0iZWxlY3Rpb24iPkVsZWN0aW9uIDIwMjQ8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InJlcGxheS1zY3J1YmJlciI+CiAgICA8ZGl2IGNsYXNzPSJycC10cmFjayIgaWQ9InJwLXRyYWNrIj48ZGl2IGNsYXNzPSJycC1maWxsIiBpZD0icnAtZmlsbCI+PC9kaXY+PGRpdiBjbGFzcz0icnAtdGh1bWIiIGlkPSJycC10aHVtYiI+PC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJycC1kYXRlcyIgaWQ9InJwLWRhdGVzIj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktcGxheWJhY2siPgogICAgPGJ1dHRvbiBjbGFzcz0icnAtcGxheSIgaWQ9InJwLXBsYXktYnRuIiBvbmNsaWNrPSJ0b2dnbGVSZXBsYXkoKSI+CiAgICAgIDxzdmcgd2lkdGg9IjEwIiBoZWlnaHQ9IjEwIiB2aWV3Qm94PSIwIDAgMTAgMTAiIGZpbGw9ImN1cnJlbnRDb2xvciI+PHBvbHlnb24gcG9pbnRzPSIyLDEgOSw1IDIsOSIgaWQ9InJwLXBsYXktaWNvbiIvPjwvc3ZnPgogICAgPC9idXR0b24+CiAgICA8ZGl2IGNsYXNzPSJycC1jdXJyZW50LWRhdGUiIGlkPSJycC1jdXJyZW50LWRhdGUiPlNlbGVjdCBhIHBlcmlvZCBhbmQgcHJlc3MgcGxheTwvZGl2PgogICAgPGRpdiBjbGFzcz0icnAtc3BlZWQiPjxzcGFuIGNsYXNzPSJycC1zcGVlZC1sYWJlbCI+U3BlZWQ8L3NwYW4+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLXNwZCBhY3RpdmUiIGRhdGEtc3BkPSIxIj4xeDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQiIGRhdGEtc3BkPSIyIj4yeDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQiIGRhdGEtc3BkPSI0Ij40eDwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0icmVwbGF5LXNuYXBzaG90Ij48ZGl2IGNsYXNzPSJycC1zbmFwLWxhYmVsIj5OYXJyYXRpdmUgc25hcHNob3QgYXQgdGhpcyBtb21lbnQ8L2Rpdj48ZGl2IGNsYXNzPSJycC1zbmFwLXN0YXRlcyIgaWQ9InJwLXNuYXAtc3RhdGVzIj48ZGl2IGNsYXNzPSJycC1sb2ctZW1wdHkiPlByZXNzIHBsYXkgdG8gb2JzZXJ2ZSBJbmRpYSBpbiBtb3Rpb24uPC9kaXY+PC9kaXY+PC9kaXY+Cjwvc2VjdGlvbj4KPHN0eWxlPgoucmVwbGF5LXNlY3Rpb257cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxO21heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzZweCAzNnB4fQoucmVwbGF5LWhlYWRlcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1lbmQ7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbToyMHB4O2dhcDoyMHB4O2ZsZXgtd3JhcDp3cmFwfQoucmVwbGF5LWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjBweDtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0taW5rKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoucmVwbGF5LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6NHB4fQoucmVwbGF5LWNvbnRyb2xze2Rpc3BsYXk6ZmxleDtnYXA6NHB4O2ZsZXgtd3JhcDp3cmFwfQoucnAtYnRue2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7cGFkZGluZzo1cHggMTJweDtib3JkZXItcmFkaXVzOjRweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDp0cmFuc3BhcmVudDtjb2xvcjp2YXIoLS1mYWludCk7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xNXN9Ci5ycC1idG4uYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKX0KLnJlcGxheS1zY3J1YmJlcntiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjEycHg7cGFkZGluZzoxOHB4IDIwcHggMTRweDttYXJnaW4tYm90dG9tOjEycHh9Ci5ycC10cmFja3twb3NpdGlvbjpyZWxhdGl2ZTtoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjJweDtjdXJzb3I6cG9pbnRlcjttYXJnaW4tYm90dG9tOjEwcHh9Ci5ycC1maWxse3Bvc2l0aW9uOmFic29sdXRlO2xlZnQ6MDt0b3A6MDtib3R0b206MDt3aWR0aDowJTtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCxyZ2JhKDIyNCw5MCw0MCwwLjQpLHZhcigtLWFjY2VudCkpO2JvcmRlci1yYWRpdXM6MnB4fQoucnAtdGh1bWJ7cG9zaXRpb246YWJzb2x1dGU7dG9wOjUwJTt0cmFuc2Zvcm06dHJhbnNsYXRlKC01MCUsLTUwJSk7d2lkdGg6MTJweDtoZWlnaHQ6MTJweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7Ym9yZGVyOjJweCBzb2xpZCByZ2JhKDksMTMsMjEsMC44KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KTtsZWZ0OjAlO2N1cnNvcjpncmFifQoucnAtZGF0ZXN7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVwbGF5LXBsYXliYWNre2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7bWFyZ2luLWJvdHRvbToxNnB4fQoucnAtcGxheXt3aWR0aDoyOHB4O2hlaWdodDoyOHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMjUpO2NvbG9yOnZhcigtLWFjY2VudCk7Y3Vyc29yOnBvaW50ZXI7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQoucnAtY3VycmVudC1kYXRle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7ZmxleDoxfQoucnAtc3BlZWR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NHB4fQoucnAtc3BlZWQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXJpZ2h0OjJweH0KLnJwLXNwZHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7cGFkZGluZzozcHggOHB4O2JvcmRlci1yYWRpdXM6M3B4O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2NvbG9yOnZhcigtLWZhaW50KTtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE1c30KLnJwLXNwZC5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLWNvbG9yOnZhcigtLWJvcmRlcil9Ci5yZXBsYXktc25hcHNob3R7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MTZweCAyMHB4fQoucnAtc25hcC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTJweH0KLnJwLXNuYXAtc3RhdGVze2Rpc3BsYXk6ZmxleDtmbGV4LXdyYXA6d3JhcDtnYXA6OHB4fQoucnAtbG9nLWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KTtmb250LXN0eWxlOml0YWxpYztwYWRkaW5nOjRweCAwfQoucnAtc3RhdGUtY2FyZHtwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6NnB4O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7bWluLXdpZHRoOjE0MHB4fQoucnAtc3RhdGUtbmFtZXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDttYXJnaW4tYm90dG9tOjNweH0KLnJwLXN0YXRlLW5hcntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQoucnAtc3RhdGUtYXR0e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KX0KPC9zdHlsZT4KPCEtLSBGQVZTIC0tPgo8c2VjdGlvbiBjbGFzcz0iZmF2cyI+CiAgPGRpdiBjbGFzcz0iZmF2cy1sYWJlbCI+VHJhY2tlZCBzdGF0ZXM8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmYXZzLXJvdyIgaWQ9ImZhdi1yb3ciPgogICAgPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjxkaXYgY2xhc3M9ImZvb3QiPgogIDxkaXYgY2xhc3M9ImZvb3QtbmFtZSI+UHVsc2Ugb2YgSW5kaWE8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmb290LWxpbmUiPk9ic2VydmVzIGhvdyBwdWJsaWMgYXR0ZW50aW9uIHNoaWZ0cyBhY3Jvc3MgdGhlIGNvdW50cnkg4oCUIHVzaW5nIHNpZ25hbHMgZnJvbSBuZXdzLCBkaXNjb3Vyc2UsIGFuZCByZWdpb25hbCBkZXZlbG9wbWVudHMuPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1zdWIiPk5vdCBuZXdzLiBOb3QgcHJlZGljdGlvbi4gT2JzZXJ2YXRpb24uPC9kaXY+CjwvZGl2PgoKPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vdG9wb2pzb24tY2xpZW50QDMuMS4wL2Rpc3QvdG9wb2pzb24tY2xpZW50Lm1pbi5qcyI+PC9zY3JpcHQ+CjxzY3JpcHQ+CnZhciBBUElfQkFTRT0obG9jYXRpb24uaG9zdG5hbWU9PT0nbG9jYWxob3N0J3x8bG9jYXRpb24uaG9zdG5hbWU9PT0nMTI3LjAuMC4xJyk/J2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMCc6Jyc7CgovLyBBUEkKYXN5bmMgZnVuY3Rpb24gZmV0Y2hBbGxTdGF0ZXMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZXMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgcm93cz1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFyb3dzfHwhcm93cy5sZW5ndGgpIHJldHVybjsKICAgIHJvd3MuZm9yRWFjaChmdW5jdGlvbihyb3cpewogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhyb3cuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbUVtbz1yb3cuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxudWxsOwogICAgICB2YXIgZW50cnk9e2F0dGVudGlvbjpyb3cuYXR0ZW50aW9uLGRlbHRhOnJvdy5kZWx0YV8yNGgsdmVsb2NpdHk6cm93LnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tRW1vLGRvbWluYW50X25hcnJhdGl2ZTpyb3cuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgICBMSVZFW3Jvdy5uYW1lXT1lbnRyeTsKICAgICAgaWYoIVNEW3Jvdy5uYW1lXSkgU0Rbcm93Lm5hbWVdPU9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7CiAgICAgIE9iamVjdC5hc3NpZ24oU0Rbcm93Lm5hbWVdLGVudHJ5KTsKICAgIH0pOwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgc2V0VGltZW91dChyZW5kZXJNb21lbnR1bSwgNTAwKTsKICAgIGlmKFNFTCYmTElWRVtTRUxdJiZkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJykpIHJlbmRlclBhbmVsKFNFTCk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW0FQSV0nLGUubWVzc2FnZSk7fQp9CgpmdW5jdGlvbiBidWlsZExvY2FsSW5zaWdodCgpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwoKICAvLyBBZ2dyZWdhdGUgdG9wIG5hcnJhdGl2ZXMgYWNyb3NzIGFsbCBzdGF0ZXMKICB2YXIgbmFyPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5hcltuLm5hbWVdKSBuYXJbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuYXJbbi5uYW1lXVtuLmRpcl09KG5hcltuLm5hbWVdW24uZGlyXXx8MCkrbi52YWw7CiAgICAgIG5hcltuLm5hbWVdLnRvdGFsPShuYXJbbi5uYW1lXS50b3RhbHx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKCiAgLy8gVG9wIHJpc2luZyBhbmQgZmFsbGluZyAoZXhjbHVkZSB0aWVzIHdoZXJlIHNhbWUgbmFtZSByaXNlcyBhbmQgZmFsbHMpCiAgdmFyIHJpc2luZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPmt2WzFdLmRvd247fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udXAtYVsxXS51cDt9KS5zbGljZSgwLDMpOwogIHZhciBmYWxsaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS5kb3duLWFbMV0uZG93bjt9KS5zbGljZSgwLDIpOwogIHZhciB0b3AzPU9iamVjdC5lbnRyaWVzKG5hcikuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnRvdGFsLWFbMV0udG90YWw7fSkuc2xpY2UoMCwzKTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PWVudHJpZXMuc2xpY2UoKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgaG90dGVzdEVtbz1ob3R0ZXN0PyhMSVZFW2hvdHRlc3RbMF1dJiZMSVZFW2hvdHRlc3RbMF1dLmRvbWluYW50X2Vtb3Rpb24pfHwnJzonJyA7CgogIC8vIEJ1aWxkIGluc2lnaHQgdGV4dCDigJQgbW9yZSBhbmFseXRpY2FsLCBjb250ZXh0LWF3YXJlCiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICBpZighZWwpIHJldHVybjsKCiAgdmFyIGxpbmVzPVtdOwogIGlmKHJpc2luZy5sZW5ndGgmJmZhbGxpbmcubGVuZ3RoJiZyaXNpbmdbMF1bMF0hPT1mYWxsaW5nWzBdWzBdKXsKICAgIGxpbmVzLnB1c2goJzxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIHRoZSBkb21pbmFudCBzaWduYWwgYWNyb3NzIEluZGlhIHRvZGF5Jyk7CiAgICBpZihmYWxsaW5nWzBdKSBsaW5lcy5wdXNoKCcgYXMgPGVtPicrZmFsbGluZ1swXVswXSsnPC9lbT4gZmFkZXMgZnJvbSBuYXRpb25hbCBmb2N1cycpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBpcyB0aGUgbW9zdCBhY3RpdmUgc3RhdGUnKwogICAgICAoaG90dGVzdEVtbz8nIHdpdGggJytob3R0ZXN0RW1vKycgYXMgdGhlIHByaW1hcnkgc2lnbmFsIHRvbmUnOicnKSk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJy4gU2Vjb25kYXJ5IHN1cmdlOiA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHJpc2luZy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnU2lnbmFscyBhcmUgY29uY2VudHJhdGVkIGFyb3VuZCA8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBsZWFkcyBuYXRpb25hbCBhdHRlbnRpb24nKTsKICAgIGlmKHJpc2luZ1sxXSkgbGluZXMucHVzaCgnIGFsb25nc2lkZSA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHRvcDMubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ05hdGlvbmFsIHNpZ25hbHMgYXJlIGRpc3BlcnNlZC4gVG9wIG5hcnJhdGl2ZXM6ICcrdG9wMy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZW0+JytuWzBdKyc8L2VtPic7fSkuam9pbignLCAnKSk7CiAgfQoKICBpZihsaW5lcy5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2xpbmVzLmpvaW4oJycpKycuPC9kaXY+JzsKICB9CgogIC8vIFRhZ3MKICBpZih0RWwpewogICAgdmFyIHRhZ3M9W107CiAgICBmYWxsaW5nLnNsaWNlKDAsMSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoNTksMTg0LDIxNiwwLjMpO2NvbG9yOiMzYmI4ZDgiPuKGkyAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICByaXNpbmcuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7Y29sb3I6I2UwNWEyOCI+4oaRICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIGlmKHRhZ3MubGVuZ3RoKSB0RWwuaW5uZXJIVE1MPXRhZ3Muam9pbignJyk7CiAgfQoKICBpZihtZXRhRWwpewogICAgdmFyIHN0YXRlQ291bnQ9T2JqZWN0LnZhbHVlcyhMSVZFKS5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuYXR0ZW50aW9uPjI7fSkubGVuZ3RoOwogICAgbWV0YUVsLnRleHRDb250ZW50PSdPYnNlcnZpbmcgJytzdGF0ZUNvdW50KycgYWN0aXZlIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pOwogIH0KfQoKZnVuY3Rpb24gdXBkYXRlQWxsU3RyaXBzKCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CiAgdmFyIGhvdHRlc3Q9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKT4oYVsxXS5hdHRlbnRpb258fDApP2I6YTt9LGVudHJpZXNbMF0pOwogIHNldFRleHQoJ3NjLWhvdHRlc3QtdmFsJyxob3R0ZXN0WzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXN1YicsJ0F0dGVudGlvbiAnK2hvdHRlc3RbMV0uYXR0ZW50aW9uLnRvRml4ZWQoMSkpOwogIHZhciB0b3BBbmdlck5tPW51bGwsdG9wQW5nZXJQY3Q9MDsKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgdmFyIGU9a3ZbMV0uZW1vdGlvbnN8fHt9OwogICAgdmFyIGE9ZS5hbmdlcnx8MDsKICAgIGlmKGE+MCYmYTw9MSkgYT1NYXRoLnJvdW5kKGEqMTAwKTsKICAgIGlmKGE+dG9wQW5nZXJQY3Qpe3RvcEFuZ2VyUGN0PWE7dG9wQW5nZXJObT1rdlswXTt9CiAgfSk7CiAgaWYodG9wQW5nZXJObSYmdG9wQW5nZXJQY3Q+MCl7CiAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLHRvcEFuZ2VyTm0pOwogICAgc2V0VGV4dCgnc2MtYW5nZXItc3ViJywnQW5nZXIgJytNYXRoLnJvdW5kKHRvcEFuZ2VyUGN0KSsnJSBvZiBzaWduYWxzJyk7CiAgfSBlbHNlIHsKICAgIC8vIEZhbGwgYmFjayB0byBkb21pbmFudF9lbW90aW9uPWFuZ2VyCiAgICB2YXIgYW5nZXJEb209ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5kb21pbmFudF9lbW90aW9uPT09J2FuZ2VyJzt9KTsKICAgIGlmKGFuZ2VyRG9tLmxlbmd0aCl7CiAgICAgIHZhciB0b3BCeUF0dD1hbmdlckRvbS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICAgICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BCeUF0dFswXSk7CiAgICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsJ0RvbWluYW50IGVtb3Rpb246IGFuZ2VyJyk7CiAgICB9CiAgfQogIHZhciBjb29saW5nPWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLnZlbG9jaXR5fHwwKTwoYVsxXS52ZWxvY2l0eXx8MCk/YjphO30sZW50cmllc1swXSk7CiAgc2V0VGV4dCgnc2MtY29vbGluZy12YWwnLGNvb2xpbmdbMF0pO3NldFRleHQoJ3NjLWNvb2xpbmctc3ViJywnVmVsb2NpdHkgJytjb29saW5nWzFdLnZlbG9jaXR5LnRvRml4ZWQoMykpOwogIHZhciBuYz17fTtlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSluY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdPShuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdfHwwKSsxO30pOwogIHZhciB0bj1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgaWYodG4pe3NldFRleHQoJ3NjLW5hcnJhdGl2ZS12YWwnLHRuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3RuWzBdLnNsaWNlKDEpKTtzZXRUZXh0KCdzYy1uYXJyYXRpdmUtc3ViJywnRG9taW5hbnQgYWNyb3NzICcrdG5bMV0rJyBzdGF0ZXMnKTt9Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUvJytlbmNvZGVVUklDb21wb25lbnQobmFtZSkpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMoZC5lbW90aW9uc3x8e30pOwogICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgIFNEW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZSwKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSxzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OmQubmFycmF0aXZlSGlzdG9yeXx8REVGQVVMVC5uYXJyYXRpdmVIaXN0b3J5LHNpZ25hbF9jb3VudDpkLnNpZ25hbF9jb3VudHx8MH07CiAgICBpZighTElWRVtuYW1lXSlMSVZFW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmV9OwogICAgTElWRVtuYW1lXS5lbW90aW9ucz1lbW9zO0xJVkVbbmFtZV0uZG9taW5hbnRfZW1vdGlvbj1kb207CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbmFtZSxlLm1lc3NhZ2UpO3JldHVybiBTRFtuYW1lXXx8REVGQVVMVDt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICAvLyB0b3BiYXIKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGl2ZS1jb3VudCcpOwogICAgaWYoZWwmJmQudG90YWxfc2lnbmFscykgZWwudGV4dENvbnRlbnQ9ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk7CiAgICB2YXIgbWV0YT1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLW1ldGEnKTsKICAgIGlmKG1ldGEmJmQuYXNfb2YpIG1ldGEudGV4dENvbnRlbnQ9JzMwIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoZC5hc19vZikudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicpOwogICAgLy8gc3RhdHMgc3RyaXAKICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJywgZC50b3RhbF9zaWduYWxzP2QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOictJyk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICB9Y2F0Y2goZSl7fQp9CgpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZihlbCllbC50ZXh0Q29udGVudD12YWw7fQoKZnVuY3Rpb24gdXBkYXRlU3RyaXBOYXJyYXRpdmUoKXt1cGRhdGVBbGxTdHJpcHMoKTt9CmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXt9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hJbnNpZ2h0cygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2luc2lnaHRzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICB2YXIgc2lnPWQuc2lnbmF0dXJlOwogICAgaWYoc2lnKXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLnRhZ3MpdEVsLmlubmVySFRNTD1kLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICB9CiAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgaWYockVsJiZkLnJpc2luZyYmZC5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6I2UwNWEyOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICBpZihmRWwmJmQuZmFsbGluZyYmZC5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrbi5zdGF0ZXMuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMykrJyU7YmFja2dyb3VuZDojM2JiOGQ4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogICAgaWYoZ0VsJiZkLnJlZ2lvbmFsJiZkLnJlZ2lvbmFsLmxlbmd0aClnRWwuaW5uZXJIVE1MPWQucmVnaW9uYWwubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbiI+PHNwYW4gY2xhc3M9Im5pLW5hbWUiPicrci5yZWdpb24rJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3IuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+PGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK3IuaG90dGVzdF9zdGF0ZSsnIMK3ICcrci50b3BfbmFycmF0aXZlKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUubWVzc2FnZSk7fQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaEZ1bGxTbmFwc2hvdCgpewogIC8vIExvYWQgQUxMIHN0YXRlIGRhdGEgaW4gb25lIHJlcXVlc3QgZm9yIGluc3RhbnQgZmlyc3QtbG9hZAogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2Z1bGwtc25hcHNob3QnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQud2FybWluZ191cHx8IWQuc3RhdGVzfHwhZC5zdGF0ZXMubGVuZ3RoKSByZXR1cm4gZmFsc2U7CgogICAgLy8gUG9wdWxhdGUgU0QgYW5kIExJVkUgZnJvbSBmdWxsIHNuYXBzaG90CiAgICBkLnN0YXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgICBpZighcy5uYW1lKSByZXR1cm47CiAgICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKHMuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fHMuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgICAgdmFyIGVudHJ5PU9iamVjdC5hc3NpZ24oe30scyx7ZW1vdGlvbnM6ZW1vcyxkb21pbmFudF9lbW90aW9uOmRvbSxkZWx0YTpzLmRlbHRhXzI0aHx8MH0pOwogICAgICBTRFtzLm5hbWVdPWVudHJ5OwogICAgICBMSVZFW3MubmFtZV09e2F0dGVudGlvbjpzLmF0dGVudGlvbixkZWx0YTpzLmRlbHRhXzI0aHx8MCx2ZWxvY2l0eTpzLnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpzLmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgIH0pOwoKICAgIC8vIFVwZGF0ZSBzaWduYWxzIGNvdW50CiAgICBpZihkLnNuYXBzaG90JiZkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMpewogICAgICBzZXRUZXh0KCdzYy1zaWduYWxzLXZhbCcsZC5zbmFwc2hvdC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCkpOwogICAgfQoKICAgIC8vIFVwZGF0ZSBpbnNpZ2h0cyBmcm9tIGNhY2hlZCBkYXRhCiAgICBpZihkLmluc2lnaHRzJiZkLmluc2lnaHRzLnNpZ25hdHVyZSl7CiAgICAgIHZhciBzaWc9ZC5pbnNpZ2h0cy5zaWduYXR1cmU7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgaWYoZWwpZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij48ZW0+JytzaWcuZmFkaW5nLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3NpZy5mYWRpbmcuc2xpY2UoMSkrJzwvZW0+IGZhZGluZyBhcyA8ZW0+JytzaWcucmlzaW5nX3ByaW1hcnkrIjwvZW0+Iisoc2lnLnJpc2luZ19zZWNvbmRhcnk/IiBhbG9uZ3NpZGUgPGVtPiIrc2lnLnJpc2luZ19zZWNvbmRhcnkrIjwvZW0+IjoiIikrIiBhY3Jvc3MgdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gPHN0cm9uZyBzdHlsZT1cImNvbG9yOnZhcigtLWluaylcIj4iK3NpZy5ob3R0ZXN0X3N0YXRlKyI8L3N0cm9uZz4gZG9taW5hdGVzLjwvZGl2PiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC5pbnNpZ2h0cy50YWdzKXRFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgICBpZihyRWwmJmQuaW5zaWdodHMucmlzaW5nJiZkLmluc2lnaHRzLnJpc2luZy5sZW5ndGgpckVsLmlubmVySFRNTD1kLmluc2lnaHRzLnJpc2luZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgICAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICAgICAgaWYoZkVsJiZkLmluc2lnaHRzLmZhbGxpbmcmJmQuaW5zaWdodHMuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmluc2lnaHRzLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTtyZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6IzNiYjhkOCI+4oaTIGZhZGluZzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NHB4OyI+JytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiMzYmI4ZDg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB9CgogICAgLy8gUmVuZGVyIG1hcCBjb2xvcnMgYW5kIHN0cmlwcwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgLy8gTG9hZCBpbnNpZ2h0cyB0b28KICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIC8vIFVzZSBjYWNoZWQgbmFycmF0aXZlIGluc2lnaHQgaWYgYXZhaWxhYmxlCiAgICBpZihkLm5hcnJhdGl2ZV9pbnNpZ2h0JiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0Kyc8L2Rpdj4nOwogICAgICBpZih0RWwmJmQubmFycmF0aXZlX2luc2lnaHQudG9wX25hcnJhdGl2ZXMpewogICAgICAgIHRFbC5pbm5lckhUTUw9ZC5uYXJyYXRpdmVfaW5zaWdodC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICAgIHZhciBjb2w9aT09PTA/JyNlMDVhMjgnOidyZ2JhKDE2MCwxOTAsMjMwLDAuNiknOwogICAgICAgICAgdmFyIGFycj1pPT09MD8nXHUyMTkxICc6J1x1MDBiNyAnOwogICAgICAgICAgcmV0dXJuICc8c3BhbiBjbGFzcz1cInNpLXRhZ1wiIHN0eWxlPVwiYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMik7Y29sb3I6Jytjb2wrJ1wiPicrYXJyK24uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5zbGljZSgxKSsnPC9zcGFuPic7CiAgICAgICAgfSkuam9pbignJyk7CiAgICAgIH0KICAgIH0KICAgIHJldHVybiB0cnVlOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW2Z1bGwtc25hcHNob3RdJyxlLm1lc3NhZ2UpOwogICAgcmV0dXJuIGZhbHNlOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCl7CiAgdHJ5ewogICAgLy8gVHJ5IGNhY2hlZCB2ZXJzaW9uIGZyb20gZnVsbC1zbmFwc2hvdCBmaXJzdCAoYWxyZWFkeSBsb2FkZWQpCiAgICAvLyBUaGVuIGNhbGwgZGVkaWNhdGVkIGVuZHBvaW50IGZvciBmcmVzaCBBSSBhbmFseXNpcwogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvbmFycmF0aXZlLWluc2lnaHQnKTsKICAgIGlmKCFyLm9rKSByZXR1cm47CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFkLnRleHQpIHJldHVybjsKCiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKCiAgICBpZihlbCkgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2QudGV4dCsnPC9kaXY+JzsKCiAgICAvLyBUYWdzIGZyb20gdG9wIG5hcnJhdGl2ZXMKICAgIGlmKHRFbCYmZC50b3BfbmFycmF0aXZlcyYmZC50b3BfbmFycmF0aXZlcy5sZW5ndGgpewogICAgICB0RWwuaW5uZXJIVE1MPWQudG9wX25hcnJhdGl2ZXMubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgICAgdmFyIGNvbD1pPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC42KSc7CiAgICAgICAgdmFyIGFycm93PWk9PT0wPyfihpEgJzonwrcgJzsKICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKTtjb2xvcjonK2NvbCsnIj4nK2Fycm93K24uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5zbGljZSgxKSsnPC9zcGFuPic7CiAgICAgIH0pLmpvaW4oJycpOwogICAgfQoKICAgIGlmKG1ldGFFbCl7CiAgICAgIHZhciB0PW5ldyBEYXRlKGQuYXNfb2YpOwogICAgICBtZXRhRWwudGV4dENvbnRlbnQ9J1NpZ25hbCBhbmFseXNpcyDCtyAnK3QudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicse2hvdXI6JzItZGlnaXQnLG1pbnV0ZTonMi1kaWdpdCd9KSsoZC5mYWxsYmFjaz8nIMK3IHBhdHRlcm4tYmFzZWQnOicgwrcgQUkgc3ludGhlc2l6ZWQnKTsKICAgIH0KICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbbmFycmF0aXZlXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU3RhdGVDb250ZXh0KG5tKXsKICAvLyBGZXRjaCBjb250ZXh0dWFsIGJyaWVmIOKAlCBjb21iaW5lcyBHb29nbGUgTmV3cyArIHN0b3JlZCBzaWduYWxzICsgQUkKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS1jb250ZXh0LycrZW5jb2RlVVJJQ29tcG9uZW50KG5tKSk7CiAgICBpZighci5vaykgcmV0dXJuIG51bGw7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIHJldHVybiBkOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW2NvbnRleHRdJyxlLm1lc3NhZ2UpOwogICAgcmV0dXJuIG51bGw7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBzdGFydFBvbGxpbmcoKXsKICBhd2FpdCBQcm9taXNlLmFsbChbZmV0Y2hBbGxTdGF0ZXMoKSxmZXRjaFNuYXAoKV0pOwogIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbihlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUpO30pOwogIHZhciBuPTA7CiAgdmFyIHQ9c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXsKICAgIG4rKzthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpOwogICAgaWYoU0VMKSByZW5kZXJQYW5lbChTRUwpOwogICAgaWYobj49MTIpe2NsZWFySW50ZXJ2YWwodCk7c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpO2lmKFNFTClyZW5kZXJQYW5lbChTRUwpO30sMTIwMDAwKTsKICAgICAgc2V0SW50ZXJ2YWwoZmV0Y2hJbnNpZ2h0cywzNjAwMDAwKTt9CiAgfSwxNTAwMCk7Cn0KCi8vIE5BUlJBVElWRSBEQVRBCnZhciBTSElGVFM9ewogICczbSc6WwogICAge2ZhZGluZzonSW5mbGF0aW9uJyxmYWRpbmdOb3RlOidlYXNpbmcgbmF0aW9uYWxseScscmlzaW5nOidCb3JkZXIgc2VjdXJpdHknLHJpc2luZ05vdGU6J3Bvc3QtaW5jaWRlbnQgc3VyZ2UnfSwKICAgIHtmYWRpbmc6J0VsZWN0aW9uIHJoZXRvcmljJyxmYWRpbmdOb3RlOidwb3N0LWN5Y2xlIGZhZGUnLHJpc2luZzonR292ZXJuYW5jZSBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZTonc3RlYWR5IHJpc2UnfSwKICAgIHtmYWRpbmc6J0Zhcm1lciBwcm90ZXN0cycsZmFkaW5nTm90ZTonbW9tZW50dW0gbG9zdCcscmlzaW5nOidVbmVtcGxveW1lbnQgYW54aWV0eScscmlzaW5nTm90ZToneW91dGggc2lnbmFsIHN1cmdlJ30sCiAgXSwKICAnNm0nOlsKICAgIHtmYWRpbmc6J0Nhc3RlIG1vYmlsaXNhdGlvbicsZmFkaW5nTm90ZToncHJlLWVsZWN0aW9uIHBlYWsnLHJpc2luZzonQ29ycnVwdGlvbiBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZToncG9zdC1jeWNsZSBwdXNoJ30sCiAgICB7ZmFkaW5nOidSZWxpZ2lvdXMgbmF0aW9uYWxpc20nLGZhZGluZ05vdGU6J3BsYXRlYXUgcGhhc2UnLHJpc2luZzonRWNvbm9taWMgYW54aWV0eScscmlzaW5nTm90ZTonY29zdC1vZi1saXZpbmcnfSwKICAgIHtmYWRpbmc6J0luZnJhc3RydWN0dXJlIHByaWRlJyxmYWRpbmdOb3RlOidyaWJib24tY3V0dGluZyBkb25lJyxyaXNpbmc6J0xhdyAmIG9yZGVyJyxyaXNpbmdOb3RlOidjcmltZSBuYXJyYXRpdmUgcmlzZSd9LAogIF0sCiAgJzF5JzpbCiAgICB7ZmFkaW5nOidQYW5kZW1pYyByZWNvdmVyeScsZmFkaW5nTm90ZTonZmFkZWQgZWFybHkgeWVhcicscmlzaW5nOidJbmZsYXRpb24nLHJpc2luZ05vdGU6J2RvbWluYXRlZCBtaWQteWVhcid9LAogICAge2ZhZGluZzonUmVnaW9uYWwgaWRlbnRpdHknLGZhZGluZ05vdGU6J2xhbmd1YWdlLWxlZCBwZWFrJyxyaXNpbmc6J1NlY3VyaXR5ICYgYm9yZGVycycscmlzaW5nTm90ZTonZ2VvcG9saXRpY2FsIGVzY2FsYXRpb24nfSwKICAgIHtmYWRpbmc6J0dvdmVybmFuY2Ugb3B0aW1pc20nLGZhZGluZ05vdGU6J3BvbGljeSBob25leW1vb24gZW5kJyxyaXNpbmc6J0NvcnJ1cHRpb24gJiBzY2FtcycscmlzaW5nTm90ZTonYWNjb3VudGFiaWxpdHkgY3ljbGUnfSwKICBdLAp9Owp2YXIgUkVHX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidCaWhhcicsZnJvbTonRWxlY3Rpb24gcmhldG9yaWMnLHRvOidVbmVtcGxveW1lbnQgJiBleGFtIHNjYW1zJyx0aW1lOic2IHdrcyd9LAogIHtzdGF0ZTonV2VzdCBCZW5nYWwnLGZyb206J0J5cG9sbCBwb2xpdGljcycsdG86J0xhdyAmIG9yZGVyIMK3IEJvcmRlcicsdGltZTonNCB3a3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdrcyd9LAogIHtzdGF0ZTonS2FybmF0YWthJyxmcm9tOidNaW5pbmcgY29udHJvdmVyc3knLHRvOidMYW5ndWFnZSBzaWduYWdlIHBvbGl0aWNzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonRGVsaGknLGZyb206J01ldHJvIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3a3MnfSwKICB7c3RhdGU6J1B1bmphYicsZnJvbTonUG93ZXIgY3Jpc2lzJyx0bzonQm9yZGVyIHNlY3VyaXR5IMK3IERyb25lcycsdGltZTonMyB3a3MnfSwKXTsKdmFyIE1PQ0tfUj1bCiAge25hbWU6J0JvcmRlciBzZWN1cml0eScsc3RhdGVzOidKJksgwrcgUHVuamFiIMK3IFJhamFzdGhhbicscGN0OicrNDElJ30sCiAge25hbWU6J1VuZW1wbG95bWVudCcsc3RhdGVzOidCaWhhciDCtyBVUCDCtyBKaGFya2hhbmQnLHBjdDonKzI4JSd9LAogIHtuYW1lOidMYW5ndWFnZSBwb2xpdGljcycsc3RhdGVzOidUTiDCtyBLYXJuYXRha2EgwrcgTUgnLHBjdDonKzIyJSd9LAogIHtuYW1lOidFbnZpcm9ubWVudGFsIGNyaXNpcycsc3RhdGVzOidEZWxoaSDCtyBSYWphc3RoYW4gwrcgQVAnLHBjdDonKzE5JSd9LAogIHtuYW1lOidFdGhuaWMgdGVuc2lvbnMnLHN0YXRlczonTWFuaXB1ciDCtyBBc3NhbSDCtyBXQicscGN0OicrMTclJ30sCl07CnZhciBNT0NLX0Y9WwogIHtuYW1lOidFbGVjdGlvbiByaGV0b3JpYycsc3RhdGVzOidOYXRpb25hbCBwb3N0LWN5Y2xlJyxwY3Q6Jy0zOCUnfSwKICB7bmFtZTonSW5mbGF0aW9uIHByZXNzdXJlJyxzdGF0ZXM6J0Vhc2luZyBuYXRpb25hbGx5JyxwY3Q6Jy0yNCUnfSwKICB7bmFtZTonRmFybWVyIHByb3Rlc3RzJyxzdGF0ZXM6J01vbWVudHVtIGxvc3QnLHBjdDonLTE5JSd9LAogIHtuYW1lOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsc3RhdGVzOidSaWJib24tY3V0dGluZyBkb25lJyxwY3Q6Jy0xNCUnfSwKICB7bmFtZTonUmVsaWdpb3VzIGZlc3RpdmFscycsc3RhdGVzOidQb3N0LXNlYXNvbiBmYWRlJyxwY3Q6Jy0xMSUnfSwKXTsKCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGRhdGE9U0hJRlRTW3BlcmlvZF18fFNISUZUU1snM20nXTsKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NoaWZ0LWxpc3QnKTsKICBpZighZWwpIHJldHVybjsKICBlbC5pbm5lckhUTUw9ZGF0YS5tYXAoZnVuY3Rpb24ocyl7CiAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjA7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjhweDtvdmVyZmxvdzpoaWRkZW47Ij4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6NnB4IDEwcHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFsbCk7bWFyZ2luLWJvdHRvbTozcHg7Ij5mYWRpbmc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjI7Ij4nK3MuZmFkaW5nKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytzLmZhZGluZ05vdGUrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0id2lkdGg6MjhweDtmbGV4LXNocmluazowO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC40NTtmb250LXNpemU6MTNweDsiPuKGkjwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo4cHggMTBweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLXJpc2UpO21hcmdpbi1ib3R0b206M3B4OyI+cmlzaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLnJpc2luZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5yaXNpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHRhYil7CiAgdGFiLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odCl7dC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIHRhYi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtyZW5kZXJTdHJpcCh0YWIuZGF0YXNldC5wZXJpb2QpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHJlbmRlck1vbWVudHVtKCl7CiAgLy8gUmVhZCBmcm9tIFNEIChwb3B1bGF0ZWQgYnkgZmV0Y2hBbGxTdGF0ZXMgZnJvbSBsaXZlIEFQSSkKICB2YXIgbmM9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBuY1tuLm5hbWVdPShuY1tuLm5hbWVdfHwwKStuLnZhbDsKICAgIH0pOwogIH0pOwogIHZhciBzb3J0ZWQ9T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogIHZhciByaXNpbmc9c29ydGVkLnNsaWNlKDAsNSk7CiAgdmFyIGZhbGxpbmc9c29ydGVkLnNsaWNlKC01KS5yZXZlcnNlKCk7CiAgdmFyIG14PXJpc2luZy5sZW5ndGg/cmlzaW5nWzBdWzFdOjEwMDsKCiAgLy8gV3JpdGUgdG8gcmlzaW5nLWxpc3QgKG1hdGNoZXMgbmFyLXJvdyBIVE1MKQogIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgaWYockVsJiZyaXNpbmcubGVuZ3RoKXsKICAgIHJFbC5pbm5lckhUTUw9cmlzaW5nLm1hcChmdW5jdGlvbihuLGkpewogICAgICB2YXIgdz1NYXRoLm1pbigxMDAsblsxXS9teCoxMDApOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6I2UwNWEyOCI+4oaRIHJpc2luZzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiNlMDVhMjg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byBkZWNsaW5pbmctbGlzdAogIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgaWYoZkVsJiZmYWxsaW5nLmxlbmd0aCl7CiAgICBmRWwuaW5uZXJIVE1MPWZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pewogICAgICB2YXIgdz1NYXRoLm1pbigxMDAsblsxXS9teCoxMDApOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6IzNiYjhkOCI+4oaTIGZhZGluZzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiMzYmI4ZDg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byByZWdpb25hbC1saXN0IOKAlCB0b3Agc3RhdGUgcGVyIHJlZ2lvbiBmcm9tIExJVkUKICB2YXIgcmVnaW9ucz17CiAgICAnTm9ydGgnOlsnRGVsaGknLCdVdHRhciBQcmFkZXNoJywnUHVuamFiJywnSGFyeWFuYScsJ0hpbWFjaGFsIFByYWRlc2gnLCdVdHRhcmFraGFuZCcsJ0phbW11IGFuZCBLYXNobWlyJ10sCiAgICAnRWFzdCc6WydXZXN0IEJlbmdhbCcsJ0JpaGFyJywnSmhhcmtoYW5kJywnT2Rpc2hhJ10sCiAgICAnV2VzdCc6WydNYWhhcmFzaHRyYScsJ0d1amFyYXQnLCdSYWphc3RoYW4nLCdHb2EnXSwKICAgICdTb3V0aCc6WydUYW1pbCBOYWR1JywnS2FybmF0YWthJywnS2VyYWxhJywnQW5kaHJhIFByYWRlc2gnLCdUZWxhbmdhbmEnXSwKICAgICdORSc6WydBc3NhbScsJ01hbmlwdXInLCdOYWdhbGFuZCcsJ01pem9yYW0nLCdNZWdoYWxheWEnLCdUcmlwdXJhJywnQXJ1bmFjaGFsIFByYWRlc2gnLCdTaWtraW0nXSwKICAgICdDZW50cmFsJzpbJ01hZGh5YSBQcmFkZXNoJywnQ2hoYXR0aXNnYXJoJ10sCiAgfTsKICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgaWYoZ0VsKXsKICAgIHZhciByZWdJdGVtcz1PYmplY3QuZW50cmllcyhyZWdpb25zKS5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgcmVnaW9uPWt2WzBdLHN0YXRlcz1rdlsxXTsKICAgICAgdmFyIHRvcD1zdGF0ZXMubWFwKGZ1bmN0aW9uKHMpe3JldHVybiB7bmFtZTpzLGF0dDooTElWRVtzXSYmTElWRVtzXS5hdHRlbnRpb24pfHwwfTt9KQogICAgICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pWzBdOwogICAgICBpZighdG9wfHwhdG9wLmF0dCkgcmV0dXJuIG51bGw7CiAgICAgIHZhciBuYXI9KExJVkVbdG9wLm5hbWVdJiZMSVZFW3RvcC5uYW1lXS5kb21pbmFudF9uYXJyYXRpdmUpfHwn4oCUJzsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjhweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6YmFzZWxpbmU7bWFyZ2luLWJvdHRvbToycHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj4nK3JlZ2lvbisnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrdG9wLmF0dC50b0ZpeGVkKDEpKyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK3RvcC5uYW1lKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweDsiPicrbmFyKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmZpbHRlcihCb29sZWFuKS5qb2luKCcnKTsKICAgIGlmKHJlZ0l0ZW1zKSBnRWwuaW5uZXJIVE1MPXJlZ0l0ZW1zOwogIH0KfQoKCi8vIFNUQVRFIERBVEEKdmFyIFNEPXt9OwoKdmFyIExJVkU9e307CmZ1bmN0aW9uIG5vcm1hbGl6ZUVtb3Rpb25zKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybnt9O3ZhciB2YWxzPU9iamVjdC52YWx1ZXMoZSksdG90PXZhbHMucmVkdWNlKGZ1bmN0aW9uKHMsdil7cmV0dXJuIHMrdjt9LDApO2lmKHRvdDw9MClyZXR1cm57fTtpZih0b3Q8PTEuMDEpe3ZhciBvdXQ9e307T2JqZWN0LmtleXMoZSkuZm9yRWFjaChmdW5jdGlvbihrKXtvdXRba109TWF0aC5yb3VuZChlW2tdKjEwMCk7fSk7cmV0dXJuIG91dDt9cmV0dXJuIGU7fQpmdW5jdGlvbiBkb21pbmFudEVtb3Rpb24oZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJuIG51bGw7dmFyIG14PTAsZG9tPW51bGw7T2JqZWN0LmVudHJpZXMoZSkuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0+bXgpe214PWt2WzFdO2RvbT1rdlswXTt9fSk7cmV0dXJuIGRvbTt9CmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKCFlbClyZXR1cm47ZWwudGV4dENvbnRlbnQ9dmFsO2lmKHZhbCYmdmFsIT09Jy0nKXtlbC5jbGFzc0xpc3QucmVtb3ZlKCdsb2FkaW5nJyk7fX0KCnZhciBERUZBVUxUPXsKICBhdHRlbnRpb246MCxkZWx0YTowLHZlbG9jaXR5OjAsCiAgZW1vdGlvbnM6e30sZG9taW5hbnRfZW1vdGlvbjpudWxsLGRvbWluYW50X25hcnJhdGl2ZTpudWxsLAogIG5hcnJhdGl2ZXM6W10scmlzaW5nOltdLGZhbGxpbmc6W10sCiAgc3VtbWFyeTonJyxhcnRpY2xlczpbXSx0aW1lbGluZTpbXSwKICBuYXJyYXRpdmVIaXN0b3J5OltdLHNpZ25hbF9jb3VudDowLAp9OwoKZnVuY3Rpb24gZyhuKXtyZXR1cm4gU0Rbbl18fE9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7fQoKZnVuY3Rpb24gYUMocyl7CiAgLy8gRHluYW1pYyBzY2FsZTogYWx3YXlzIHNwcmVhZCBmdWxsIGNvbG9yIHJhbmdlIGFjcm9zcyBhY3R1YWwgZGF0YQogIC8vIEdldCBtaW4vbWF4IGZyb20gY3VycmVudCBTRCB0byBub3JtYWxpemUKICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSk7CiAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKTsKICB2YXIgbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogIC8vIE5vcm1hbGl6ZSAwLTEKICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHMtbW4pLyhteC1tbikpKTsKICAvLyBNYXAgdG8gY29sb3Igc3RvcHM6IGRhcmsgYmx1ZSDihpIgdGVhbCDihpIgYW1iZXIg4oaSIG9yYW5nZSDihpIgcmVkCiAgaWYobjwwLjEyKSByZXR1cm4gJyMwZDFlMzAnOwogIGlmKG48MC4yNSkgcmV0dXJuICcjMGUzZDZhJzsKICBpZihuPDAuMzgpIHJldHVybiAnIzBkNWY5MCc7CiAgaWYobjwwLjUwKSByZXR1cm4gJyMwZTdhYWEnOwogIGlmKG48MC42MikgcmV0dXJuICcjMWE5MDkwJzsKICBpZihuPDAuNzIpIHJldHVybiAnI2M4NzAxMCc7CiAgaWYobjwwLjgyKSByZXR1cm4gJyNkODQwMTAnOwogIGlmKG48MC45MikgcmV0dXJuICcjY2MxODA4JzsKICByZXR1cm4gJyNmZjAwMTAnOwp9CmZ1bmN0aW9uIGVDKGUpewogIHZhciBteD0wLGRvbT0ncHJpZGUnOwogIGZvcih2YXIgayBpbiBlKXtpZihlW2tdPm14KXtteD1lW2tdO2RvbT1rO319CiAgcmV0dXJuICh7YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ30pW2RvbV18fCcjMzNhYWNjJzsKfQpmdW5jdGlvbiB2Qyh2KXsKICBpZih2PjAuMikgcmV0dXJuICcjZGMwODE4JzsKICBpZih2PjAuMSkgcmV0dXJuICcjZTA1YTI4JzsKICBpZih2PjAuMDIpIHJldHVybiAnI2NjODgyMic7CiAgaWYodjwtMC4wNSkgcmV0dXJuICcjMjI5OWJiJzsKICByZXR1cm4gJyMxNTIwMzAnOwp9Cgp2YXIgbGF5ZXI9J2F0dGVudGlvbicsU0VMPW51bGwsRkFWUz1uZXcgU2V0KCk7CgovLyBNQVAKZnVuY3Rpb24gcHJval8odyxoLHBhZCl7CiAgcGFkPXBhZHx8MjA7CiAgdmFyIG1pbkxvbj02OC4xLG1heExvbj05Ny40LG1pbkxhdD02LjUsbWF4TGF0PTM3LjE7CiAgdmFyIHNjWD0ody1wYWQqMikvKG1heExvbi1taW5Mb24pOwogIHZhciBzY1k9KGgtcGFkKjIpLyhtYXhMYXQtbWluTGF0KTsKICB2YXIgc2M9TWF0aC5taW4oc2NYLHNjWSk7CiAgdmFyIG94PXBhZCsody1wYWQqMi0obWF4TG9uLW1pbkxvbikqc2MpLzI7CiAgdmFyIG95PXBhZCsoaC1wYWQqMi0obWF4TGF0LW1pbkxhdCkqc2MpLzI7CiAgcmV0dXJuIGZ1bmN0aW9uKGxvbixsYXQpe3JldHVybiBbb3grKGxvbi1taW5Mb24pKnNjLCBveSsobWF4TGF0LWxhdCkqc2NdO307Cn0KZnVuY3Rpb24gZ2VvMnBhdGgoZ2VvbSxwail7CiAgdmFyIGQ9Jyc7CiAgZnVuY3Rpb24gcmluZyhjcyl7dmFyIHM9Jyc7Y3MuZm9yRWFjaChmdW5jdGlvbihjLGkpe3ZhciBwPXBqKGNbMF0sY1sxXSk7cys9KGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pO3JldHVybiBzKydaJzt9CiAgaWYoZ2VvbS50eXBlPT09J1BvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTsKICBlbHNlIGlmKGdlb20udHlwZT09PSdNdWx0aVBvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocCl7cC5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7fSk7CiAgcmV0dXJuIGQ7Cn0KZnVuY3Rpb24gY3RyKGdlb20pewogIHZhciBwdHM9W107CiAgZnVuY3Rpb24gY29sKGMpe2lmKHR5cGVvZiBjWzBdPT09J251bWJlcicpIHB0cy5wdXNoKGMpO2Vsc2UgYy5mb3JFYWNoKGNvbCk7fQogIGNvbChnZW9tLmNvb3JkaW5hdGVzKTsKICBpZighcHRzLmxlbmd0aCkgcmV0dXJuIFswLDBdOwogIHJldHVybiBbcHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMF07fSwwKS9wdHMubGVuZ3RoLHB0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzFdO30sMCkvcHRzLmxlbmd0aF07Cn0KZnVuY3Rpb24gc05hbWUocHJvcHMpewogIHZhciByYXc9cHJvcHMuc3Rfbm18fHByb3BzLk5BTUVfMXx8cHJvcHMubmFtZXx8cHJvcHMuTkFNRXx8Jyc7CiAgdmFyIG1hcD17J0xhZGFraCc6J0phbW11IGFuZCBLYXNobWlyJywnSmFtbXUgJiBLYXNobWlyJzonSmFtbXUgYW5kIEthc2htaXInLCdVdHRhcmFuY2hhbCc6J1V0dGFyYWtoYW5kJywnQW5kYW1hbiBhbmQgTmljb2Jhcic6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ0FuZGFtYW4gJiBOaWNvYmFyIElzbGFuZCc6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ05DVCBvZiBEZWxoaSc6J0RlbGhpJywnUG9uZGljaGVycnknOidQdWR1Y2hlcnJ5JywnRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnLCdEYW1hbiBhbmQgRGl1JzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdSd9OwogIHJldHVybiBtYXBbcmF3XXx8cmF3Owp9Cgp2YXIgY2FjaGVkR2VvPW51bGw7Cgphc3luYyBmdW5jdGlvbiBsb2FkTWFwKGF0dGVtcHQpewogIGF0dGVtcHQgPSBhdHRlbXB0fHwxOwogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKCdodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvZ2gvdWRpdC0wMDEvaW5kaWEtbWFwcy1kYXRhQG1hc3Rlci90b3BvanNvbi9pbmRpYS5qc29uJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHRvcG89YXdhaXQgci5qc29uKCk7CiAgICBjYWNoZWRHZW89dG9wb2pzb24uZmVhdHVyZSh0b3BvLHRvcG8ub2JqZWN0cy5zdGF0ZXMpOwogICAgcmVuZGVyTWFwKGNhY2hlZEdlbyk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMTAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMzAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsNjAwMCk7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbbWFwXSBsb2FkIGZhaWxlZCBhdHRlbXB0ICcrYXR0ZW1wdCsnOicsZS5tZXNzYWdlKTsKICAgIGlmKGF0dGVtcHQ8NSl7CiAgICAgIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtsb2FkTWFwKGF0dGVtcHQrMSk7fSwgYXR0ZW1wdCoyMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHZhciBtaT1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJyk7CiAgICAgIGlmKG1pKSBtaS5pbm5lckhUTUw9JzxkaXYgc3R5bGU9ImNvbG9yOiMyYTNhNGE7cGFkZGluZzo0MHB4O3RleHQtYWxpZ246Y2VudGVyO2ZvbnQtZmFtaWx5Om1vbm9zcGFjZTtmb250LXNpemU6MTFweCI+TWFwIHVuYXZhaWxhYmxlIOKAlCByZWZyZXNoIHRvIHJldHJ5PC9kaXY+JzsKICAgIH0KICB9Cn0KCmZ1bmN0aW9uIHJlbmRlck1hcChzdGF0ZXMpewogIHZhciB3PTgwMCxoPTgwMCxwaj1wcm9qXyh3LGgsMjgpOwogIHZhciBzZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXN0YXRlcycpOwogIHZhciBwZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXB1bHNlcycpOwogIHZhciBnZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLWdsb3cnKTsKICBzZy5pbm5lckhUTUw9Jyc7cGcuaW5uZXJIVE1MPScnO2dnLmlubmVySFRNTD0nJzsKCiAgc3RhdGVzLmZlYXR1cmVzLmZvckVhY2goZnVuY3Rpb24oZil7CiAgICBpZighZi5nZW9tZXRyeSkgcmV0dXJuOwogICAgdmFyIG5tPXNOYW1lKGYucHJvcGVydGllcyksZD1nKG5tKTsKICAgIHZhciBwYXRoRWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ3BhdGgnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2QnLGdlbzJwYXRoKGYuZ2VvbWV0cnkscGopKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywnc3RhdGUnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScsbm0pOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJywncmdiYSgyNTUsMjU1LDI1NSwwLjA3KScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMC41Jyk7CiAgICBzZy5hcHBlbmRDaGlsZChwYXRoRWwpOwoKICAgIHZhciBjdD1jdHIoZi5nZW9tZXRyeSksY3A9cGooY3RbMF0sY3RbMV0pOwoKICAgIC8vIEF0bW9zcGhlcmljIGdsb3cgZm9yIGhpZ2gtYXR0ZW50aW9uIHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTY1KXsKICAgICAgdmFyIGdsb3dFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnZWxsaXBzZScpOwogICAgICB2YXIgZ2xvd1I9TWF0aC5taW4oNjAsMjArZC5hdHRlbnRpb24qMC41KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtnbG93RWwuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdyeCcsZ2xvd1IpO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J5JyxnbG93UiowLjcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWxsJyxhQyhkLmF0dGVudGlvbikpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdvcGFjaXR5JywnMC4wOCcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWx0ZXInLCd1cmwoI3N0YXRlR2xvdyknKTsKICAgICAgZ2xvd0VsLnN0eWxlLmFuaW1hdGlvbj0nZ2xvd1B1bHNlICcrKDIuNStNYXRoLnJhbmRvbSgpKSsncyBlYXNlLWluLW91dCAnKyhNYXRoLnJhbmRvbSgpKjIpKydzIGluZmluaXRlJzsKICAgICAgZ2cuYXBwZW5kQ2hpbGQoZ2xvd0VsKTsKICAgIH0KCiAgICAvLyBEdWFsIHB1bHNlIHJpbmdzIGZvciB2ZXJ5IGhvdCBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj03Mil7CiAgICAgIFswLDFdLmZvckVhY2goZnVuY3Rpb24oaSl7CiAgICAgICAgdmFyIHJpbmc9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2NpcmNsZScpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO3Jpbmcuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3B1bHNlLXJpbmcgcCcrKGkrMSkpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2UnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzEnKTsKICAgICAgICByaW5nLnN0eWxlLmFuaW1hdGlvbkRlbGF5PShNYXRoLnJhbmRvbSgpKjIuNSkrJ3MnOwogICAgICAgIHBnLmFwcGVuZENoaWxkKHJpbmcpOwogICAgICB9KTsKICAgIH0KICB9KTsKICBhcHBseUxheWVyKCk7CiAgYXR0YWNoSW50ZXJhY3Rpb25zKCk7Cn0KCi8vIFNpbmdsZSBzb3VyY2Ugb2YgdHJ1dGggZm9yIGVtb3Rpb24gY29sb3IKLy8gQm90aCBtYXAgYW5kIHBhbmVsIGNhbGwgdGhpcyDigJQgZ3VhcmFudGVlcyB0aGV5IGFsd2F5cyBtYXRjaApmdW5jdGlvbiBnZXRFZmZlY3RpdmVFbW90aW9uKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogIC8vIDEuIFRyeSBMSVZFLmRvbWluYW50X2Vtb3Rpb24gKHNldCBieSAvYXBpL3N0YXRlcykKICB2YXIgZG9tPWxpdmUuZG9taW5hbnRfZW1vdGlvbnx8ZC5kb21pbmFudF9lbW90aW9uOwoKICAvLyAyLiBUcnkgY29tcHV0aW5nIGZyb20gZW1vdGlvbnMgYnJlYWtkb3duCiAgaWYoIWRvbSl7CiAgICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogICAgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKTsKICB9CgogIC8vIDMuIEZhbGxiYWNrOiBpbmZlciBmcm9tIGRvbWluYW50IG5hcnJhdGl2ZSAoc2FtZSBsb2dpYyBldmVyeXdoZXJlKQogIGlmKCFkb20pewogICAgdmFyIG5wPShsaXZlLmRvbWluYW50X25hcnJhdGl2ZXx8ZC5kb21pbmFudF9uYXJyYXRpdmV8fCcnKS50b0xvd2VyQ2FzZSgpOwogICAgaWYobnAubWF0Y2goL2JvcmRlcnx0ZXJyb3J8c2VjdXJpdHl8Y29uZmxpY3R8YXR0YWNrfHdhcnxpbmZpbHRyYXQvKSkgZG9tPSdmZWFyJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL3NjYW18Y29ycnVwdHxwcm90ZXN0fGFycmVzdHx2aW9sZW5jZXxvdXRyYWdlfGNyaW1lLykpIGRvbT0nYW5nZXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvZGV2ZWxvcHxpbnZlc3R8Z3Jvd3RofGxhdW5jaHxpbmF1Z3VyfHJlZm9ybXxwcm9ncmVzc3xib29zdC8pKSBkb209J2hvcGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvY3VsdHVyZXxoZXJpdGFnZXxwcmlkZXx2aWN0b3J5fGNlbGVicmF0fG1lZGFsfGFjaGlldmVtZW50LykpIGRvbT0ncHJpZGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvZmxvb2R8ZHJvdWdodHx1bmVtcGxveW1lbnR8aW5mbGF0aW9ufHNob3J0YWdlfGNyaXNpc3xjb25jZXJuLykpIGRvbT0nYW54aWV0eSc7CiAgICBlbHNlIGlmKChsaXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDApPjUpIGRvbT0nYW54aWV0eSc7IC8vIGFjdGl2ZSBzdGF0ZSBkZWZhdWx0CiAgICBlbHNlIGRvbT0nYW54aWV0eSc7IC8vIGdsb2JhbCBkZWZhdWx0CiAgfQoKICByZXR1cm4gZG9tOwp9CgovLyBHZXQgZXN0aW1hdGVkIGVtb3Rpb24gYnJlYWtkb3duIChmb3IgcGFuZWwgZG9udXQgd2hlbiByZWFsIGRhdGEgbWlzc2luZykKZnVuY3Rpb24gZ2V0RW1vdGlvbkJyZWFrZG93bihubSl7CiAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogIHZhciBkPVNEW25tXXx8e307CiAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICBpZihPYmplY3Qua2V5cyhlbW9zKS5sZW5ndGgpIHJldHVybiB7ZW1vdGlvbnM6ZW1vcyxlc3RpbWF0ZWQ6ZmFsc2V9OwogIC8vIEJ1aWxkIHNrZXdlZCBkaXN0cmlidXRpb24gZnJvbSBlZmZlY3RpdmUgZW1vdGlvbgogIHZhciBkb209Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgdmFyIGJhc2U9e2FueGlldHk6MTMsYW5nZXI6MTMsaG9wZToxMyxwcmlkZToxMyxmZWFyOjEzfTsKICBiYXNlW2RvbV09NDg7CiAgcmV0dXJuIHtlbW90aW9uczpiYXNlLGVzdGltYXRlZDp0cnVlfTsKfQoKZnVuY3Rpb24gYXBwbHlMYXllcigpewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksZmlsbDsKICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRlPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICBmaWxsPWVNYXBbZGVdfHwnIzMzNDQ1NSc7CiAgICB9CiAgICBlbHNlIGZpbGw9dkMoZC52ZWxvY2l0eSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsZmlsbCk7CiAgICAoZnVuY3Rpb24oKXsKICAgICAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oeCl7cmV0dXJuIHguYXR0ZW50aW9ufHwwO30pOwogICAgICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpLG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAgICAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChkLmF0dGVudGlvbi1tbikvKG14LW1uKSkpOwogICAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxsYXllcj09PSdhdHRlbnRpb24nP01hdGgubWF4KDAuMywwLjMrbiowLjcpOjAuODUpOwogICAgfSkoKTsKICB9KTsKfQoKZnVuY3Rpb24gYXR0YWNoSW50ZXJhY3Rpb25zKCl7CiAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsZnVuY3Rpb24oZSl7CiAgICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk7CiAgICAgIHZhciBkPWcobm0pOwogICAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICAgIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICAgICAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBsYXRlc3Q9Jyc7CiAgICAgIGlmKGQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzLmxlbmd0aCkgbGF0ZXN0PWQubmFycmF0aXZlc1swXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2QubmFycmF0aXZlc1swXS5uYW1lLnNsaWNlKDEpOwogICAgICBlbHNlIGlmKGxpdmUuZG9taW5hbnRfbmFycmF0aXZlKSBsYXRlc3Q9bGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuc2xpY2UoMSk7CgogICAgICB2YXIgcm93cz0nJzsKICAgICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICAgICAgdmFyIGF0dD1saXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDA7CiAgICAgICAgdmFyIGRsdD1saXZlLmRlbHRhfHxkLmRlbHRhfHwwOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkF0dGVudGlvbjwvc3Bhbj48c3Ryb25nPicrYXR0LnRvRml4ZWQoMSkrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAoZGx0IT09MD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+MjRoIHNoaWZ0PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrKGRsdD4wPycjZTA1YTI4JzonIzNiYjhkOCcpKyciPicrKGRsdD4wPycrJzonJykrZGx0Kyc8L3N0cm9uZz48L2Rpdj4nOicnKSsKICAgICAgICAgIChsYXRlc3Q/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPlRvcCBuYXJyYXRpdmU8L3NwYW4+PHN0cm9uZz4nK2xhdGVzdCsnPC9zdHJvbmc+PC9kaXY+JzonJyk7CiAgICAgIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIGRvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgICBpZihkb21FbW8pewogICAgICAgICAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6ZC5lbW90aW9uc3x8e307CiAgICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Eb21pbmFudDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3BhbFtkb21FbW9dKyciPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW9zKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgICAgICAgIHZhciB0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICAgICAgICBpZih0b3Q+MCYmdG90PD0xLjAxKXtlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybltrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO3RvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTt9CiAgICAgICAgICByb3dzKz1lTC5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3BhbiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NHB4Ij48c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK3BhbFtrdlswXV0rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicra3ZbMF0rJzwvc3Bhbj48c3Ryb25nPicrTWF0aC5yb3VuZChrdlsxXSoxMDAvTWF0aC5tYXgoMSx0b3QpKSsnJTwvc3Ryb25nPjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgICAgfQogICAgICB9IGVsc2UgewogICAgICAgIHZhciB2ZWw9bGl2ZS52ZWxvY2l0eXx8ZC52ZWxvY2l0eXx8MDsKICAgICAgICB2YXIgdmVsRGlyPXZlbD4wLjE/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wMj8nUmlzaW5nJzp2ZWw8LTAuMDU/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgICAgIHZhciB2ZWxDb2w9dmVsPjAuMDI/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPk1vbWVudHVtPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RGlyZWN0aW9uPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrdmVsRGlyKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICB9CgogICAgICB0aXAuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJ0dC1uIj4nK25tKyc8L2Rpdj4nK3Jvd3MrKGxhdGVzdCYmbGF5ZXIhPT0nYXR0ZW50aW9uJz8nPGRpdiBjbGFzcz0idHQtbmFyIj48c3Ryb25nPk5hcnJhdGl2ZTwvc3Ryb25nPicrbGF0ZXN0Kyc8L2Rpdj4nOicnKTsKICAgICAgdmFyIHJlY3Q9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgICB0aXAuc3R5bGUubGVmdD1NYXRoLm1pbihlLmNsaWVudFgtcmVjdC5sZWZ0KzE0LHJlY3Qud2lkdGgtMTkwKSsncHgnOwogICAgICB0aXAuc3R5bGUudG9wPU1hdGgubWluKGUuY2xpZW50WS1yZWN0LnRvcCsxNCxyZWN0LmhlaWdodC0xNTApKydweCc7CiAgICAgIHRpcC5zdHlsZS5vcGFjaXR5PScxJzsKICAgIH0pOwpwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbGVhdmUnLGZ1bmN0aW9uKCl7dGlwLnN0eWxlLm9wYWNpdHk9MDt9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7c2VsZWN0XyhwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJykpO30pOwogIH0pOwp9CgovLyBTVEFURSBQQU5FTApmdW5jdGlvbiBzZWxlY3RfKG5tKXsKICBTRUw9bm07CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuY2xhc3NMaXN0LnRvZ2dsZSgnc2VsZWN0ZWQnLHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKT09PW5tKTsKICB9KTsKICAvLyBTaG93IGxvYWRpbmcgc3RhdGUgaW1tZWRpYXRlbHkgd2l0aCB3aGF0ZXZlciBMSVZFIGRhdGEgd2UgaGF2ZQogIHZhciBwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYocGFuZWwpewogICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgcGFuZWwuaW5uZXJIVE1MPQogICAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgICAnPGRpdj48ZGl2IGNsYXNzPSJzcC1layI+JysobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhGQVZTLmhhcyhubSk/J29uJzonJykrJyIgZGF0YS1ubT0iJytubSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYodGhpcy5kYXRhc2V0Lm5tKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAgICc8L2J1dHRvbj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MjBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4nKwogICAgICAgICdMb2FkaW5nIHNpZ25hbHMgZm9yICcrbm0rJy4uLicrCiAgICAgICAgKGxpdmUuYXR0ZW50aW9uPyc8YnI+PGJyPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Y29sb3I6dmFyKC0taW5rKSI+QXR0ZW50aW9uICcrbGl2ZS5hdHRlbnRpb24udG9GaXhlZCgxKSsnPC9zcGFuPic6JycpKwogICAgICAgIChsaXZlLmRvbWluYW50X2Vtb3Rpb24/Jzxicj48c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nK2xpdmUuZG9taW5hbnRfZW1vdGlvbisnIHNpZ25hbCBkb21pbmFudDwvc3Bhbj4nOicnKSsKICAgICAgJzwvZGl2Pic7CiAgfQogIC8vIEZldGNoIGZ1bGwgZGV0YWlsIHRoZW4gcmVuZGVyCiAgLy8gRmV0Y2ggZGV0YWlsIGFuZCBjb250ZXh0IGluIHBhcmFsbGVsCiAgUHJvbWlzZS5hbGwoWwogICAgZmV0Y2hEZXRhaWwobm0pLAogICAgbGF5ZXI9PT0nYXR0ZW50aW9uJz9mZXRjaFN0YXRlQ29udGV4dChubSk6UHJvbWlzZS5yZXNvbHZlKG51bGwpCiAgXSkudGhlbihmdW5jdGlvbihyZXN1bHRzKXsKICAgIGlmKFNFTCE9PW5tKSByZXR1cm47CiAgICB2YXIgY3R4PXJlc3VsdHNbMV07CiAgICByZW5kZXJQYW5lbChubSwgY3R4KTsKICAgIC8vIFVwZGF0ZSBtYXAgY29sb3IKICAgIHZhciBwYXRoPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9Iicrbm0rJyJdJyk7CiAgICBpZihwYXRoJiZsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgaWYoZU1hcFtkb21dKSBwYXRoLnNldEF0dHJpYnV0ZSgnZmlsbCcsZU1hcFtkb21dKTsKICAgIH0gZWxzZSB7CiAgICAgIGFwcGx5TGF5ZXIoKTsKICAgIH0KICB9KS5jYXRjaChmdW5jdGlvbihlKXsKICAgIGNvbnNvbGUud2FybignW3NlbGVjdF0nLGUpOwogICAgaWYoU0VMPT09bm0pIHJlbmRlclBhbmVsKG5tLCBudWxsKTsKICB9KTsKfQoKZnVuY3Rpb24gcmVuZGVyUGFuZWwobm0sIGN0eCl7CiAgdmFyIGQ9ZyhubSk7CiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZighcGFuZWwpIHJldHVybjsKICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgdmFyIGhlYWRlcj0KICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAnPGRpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1layIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsiPicrCiAgICAgICAgICAobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsKICAgICAgICAgIChkLmNvbmZpZGVuY2U/JzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07cGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6JysoZC5jb25maWRlbmNlPT09J0hJR0gnPydyZ2JhKDUxLDIwNCwxMDIsMC4xKSc6ZC5jb25maWRlbmNlPT09J01FRElVTSc/J3JnYmEoMjI0LDkwLDQwLDAuMSknOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpJykrJztjb2xvcjonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/JyMzM2NjNjYnOmQuY29uZmlkZW5jZT09PSdNRURJVU0nPycjZTA1YTI4JzoncmdiYSgyNTUsMjU1LDI1NSwwLjMpJykrJyI+JytkLmNvbmZpZGVuY2UrJyBTSUdOQUw8L3NwYW4+JzonJykrCiAgICAgICAgICAoZC5pc19yZWdpb25hbF9zdG9yeT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKTtjb2xvcjojM2JiOGQ4Ij5SRUdJT05BTCBTUElLRTwvc3Bhbj4nOicnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAnPC9idXR0b24+JysKICAgICc8L2Rpdj4nOwoKICB2YXIgYm9keT0nJzsKCiAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICB2YXIgZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8ndXAnOidkbic7CiAgICB2YXIgbmFycj1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHRsPShkLnRpbWVsaW5lJiZkLnRpbWVsaW5lLmxlbmd0aCk/ZC50aW1lbGluZTpbMCwwLDAsMCwwLDAsMCxkLmF0dGVudGlvbnx8MF07CiAgICB2YXIgdG1uPU1hdGgubWluLmFwcGx5KG51bGwsdGwpLHRteD1NYXRoLm1heC5hcHBseShudWxsLHRsKSx0cj1NYXRoLm1heCgxLHRteC10bW4pOwogICAgdmFyIHR3PTI2MCx0aD02Mix0cD01OwogICAgdmFyIHB0cz10bC5tYXAoZnVuY3Rpb24odixpKXtyZXR1cm5bdHArKGkvKHRsLmxlbmd0aC0xKSkqKHR3LXRwKjIpLHRwKygxLSh2LXRtbikvdHIpKih0aC10cCoyKV07fSk7CiAgICB2YXIgcEQ9cHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybihpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KS5qb2luKCcnKTsKICAgIHZhciBhRD1wRCsnIEwnK3B0c1twdHMubGVuZ3RoLTFdWzBdKycsJysodGgtdHApKycgTCcrcHRzWzBdWzBdKycsJysodGgtdHApKycgWic7CiAgICB2YXIgYWM9YUMoZC5hdHRlbnRpb258fDApOwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdIb3cgaW50ZW5zZWx5ICcrKG5tLnNwbGl0KCIgIilbMF0pKycgaXMgYmVpbmcgZGlzY3Vzc2VkIG5hdGlvbmFsbHkuIFNjb3JlIG9mICcrZC5hdHRlbnRpb24rJyBtZWFucyAnKyhkLmF0dGVudGlvbj42MD8ndmVyeSBoaWdoIOKAlCBkb21pbmF0ZXMgbmF0aW9uYWwgZGlzY291cnNlJzpkLmF0dGVudGlvbj4zNT8nZWxldmF0ZWQg4oCUIGNsZWFybHkgaW4gdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbic6ZC5hdHRlbnRpb24+MTU/J21vZGVyYXRlIOKAlCBzb21lIG5hdGlvbmFsIGNvdmVyYWdlJzpkLmF0dGVudGlvbj41Pydsb3cg4oCUIGxpbWl0ZWQgc2lnbmFscyc6J21pbmltYWwg4oCUIGZldyBzaWduYWxzIGRldGVjdGVkJykrJy4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0iaW5zaWdodCIgc3R5bGU9IicrKGN0eD8nJzonYm9yZGVyLWNvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4wNiknKSsnIj4nKwogICAgICAoY3R4JiZjdHguYnJpZWYKICAgICAgICA/IGN0eC5icmllZisoY3R4LnNvdXJjZT09PSJhaSI/Jyc6JycpCiAgICAgICAgOiAoZC5jb25maWRlbmNlPT09IkxPVyImJiFkLnN1bW1hcnkKICAgICAgICAgICAgPyAnTGltaXRlZCBzaWduYWxzIGZyb20gJytubSsnLiBNb25pdG9yaW5nIHJlZ2lvbmFsIHNvdXJjZXMuJwogICAgICAgICAgICA6IGQuc3VtbWFyeXx8J0NvbGxlY3Rpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJykpKwogICAgJzwvZGl2PicrCiAgICAoY3R4JiZjdHguZnJlc2hfaGVhZGxpbmVzJiZjdHguZnJlc2hfaGVhZGxpbmVzLmxlbmd0aD8KICAgICAgJzxkaXYgc3R5bGU9Im1hcmdpbi10b3A6MTBweDtwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6NnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NnB4Ij5MYXRlc3QgZnJvbSBHb29nbGUgTmV3czwvZGl2PicrCiAgICAgICAgY3R4LmZyZXNoX2hlYWRsaW5lcy5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihoKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLWRpbSk7cGFkZGluZzozcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpO2xpbmUtaGVpZ2h0OjEuNCI+JytoKyc8L2Rpdj4nO30pLmpvaW4oJycpKwogICAgICAnPC9kaXY+JzonJykrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKG5hcnJbMF0/bmFyclswXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25hcnJbMF0ubmFtZS5zbGljZSgxKTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAobmFyci5sZW5ndGg/CiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrbmFyci5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHZhciBubj1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPjxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25uKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4Ij7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojM2JiOGQ4O2ZvbnQtc2l6ZTo5cHgiPuKGkzwvc3Bhbj4nOicnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdmFsIj4nK3ZhbC50b0ZpeGVkKDEpKyclPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCx2YWwqMi41KSsnJTtiYWNrZ3JvdW5kOicrKG4uZGlyPT09J3VwJz8nI2UwNWEyOCc6bi5kaXI9PT0nZG93bic/JyMzYmI4ZDgnOicjMzM0NDU1JykrJyI+PC9kaXY+PC9kaXY+PC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpKyc8L2Rpdj4nOgogICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb3ctc2lnbmFsIHJlZ2lvbi4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLjwvZGl2PicpKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QXR0ZW50aW9uIOKAlCA4IGRheXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0bC13cmFwIj48c3ZnIHZpZXdCb3g9IjAgMCAnK3R3KycgJyt0aCsnIiBzdHlsZT0id2lkdGg6MTAwJTtoZWlnaHQ6MTAwJSI+JysKICAgICAgICAgICc8ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9InRsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwLjI1Ii8+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAiLz4nKwogICAgICAgICAgJzwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPicrCiAgICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnKSIgLz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrcEQrJyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSInK2FjKyciIHN0cm9rZS13aWR0aD0iMS4yIi8+JysKICAgICAgICAgIHB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gJzxjaXJjbGUgY3g9IicrcFswXSsnIiBjeT0iJytwWzFdKyciIHI9IicrKGk9PT1wdHMubGVuZ3RoLTE/Mi4yOjEuMikrJyIgZmlsbD0iJythYysnIi8+Jzt9KS5qb2luKCcnKSsKICAgICAgICAnPC9zdmc+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JysoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGg/ZC5hcnRpY2xlcy5sZW5ndGg6MCkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtbGlzdCI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLm1hcChmdW5jdGlvbihhKXtyZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIj48ZGl2IGNsYXNzPSJhcnQtc3JjIj4nKyhhLnNyY3x8JycpKyc8L2Rpdj48ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NnB4IDAiPk5vIHNpZ25hbHMgY29sbGVjdGVkIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAvLyBVc2Ugc2FtZSBmdW5jdGlvbnMgYXMgbWFwIOKAlCBndWFyYW50ZWVkIHRvIG1hdGNoCiAgICB2YXIgbWFwRG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgdmFyIGJyZWFrZG93bj1nZXRFbW90aW9uQnJlYWtkb3duKG5tKTsKICAgIHZhciBlbW90aW9ucz1icmVha2Rvd24uZW1vdGlvbnM7CiAgICB2YXIgaGFzRW1vcz0hYnJlYWtkb3duLmVzdGltYXRlZDsKICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW90aW9ucyk7CiAgICB2YXIgZVRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgIGlmKGVUb3Q+MCYmZVRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt9CiAgICB2YXIgdG90PU1hdGgubWF4KDEsZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogICAgZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICBpZighZUwubGVuZ3RoKXtwYW5lbC5pbm5lckhUTUw9aGVhZGVyKyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4Ij5ObyBlbW90aW9uIGRhdGEgeWV0LjwvZGl2Pic7cmV0dXJuO30KICAgIC8vIGRvbUVtbyA9IHNhbWUgYXMgbWFwIGNvbG9yIChmcm9tIGdldEVmZmVjdGl2ZUVtb3Rpb24pCiAgICB2YXIgZG9tRW1vPW1hcERvbUVtbzsKICAgIC8vIFJlb3JkZXIgZUwgc28gZG9taW5hbnQgc2hvd3MgZmlyc3QKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXsKICAgICAgaWYoYVswXT09PWRvbUVtbykgcmV0dXJuIC0xOwogICAgICBpZihiWzBdPT09ZG9tRW1vKSByZXR1cm4gMTsKICAgICAgcmV0dXJuIGJbMV0tYVsxXTsKICAgIH0pOwogICAgdmFyIGRvbVBjdD1NYXRoLnJvdW5kKChlTFswXT9lTFswXVsxXToyMCkqMTAwL3RvdCk7CiAgICB2YXIgbmFycjI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0b3BOYXJTdHI9bmFycjIuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCcgYW5kICcpOwogICAgdmFyIHdoYXRJdD17YW54aWV0eTonVW5jZXJ0YWludHkgYW5kIHVuZWFzZSBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gU2lnbmFsczogJyt0b3BOYXJTdHIrJy4nOicnKSxhbmdlcjonT3V0cmFnZSBhbmQgcHJlc3N1cmUgaW4gJytubSsodG9wTmFyU3RyPycuIERyaXZlbiBieTogJyt0b3BOYXJTdHIrJy4nOicnKSxob3BlOidPcHRpbWlzbSBhbmQgcHJvZ3Jlc3MgaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKSxwcmlkZTonSWRlbnRpdHkgYW5kIGFjaGlldmVtZW50IGluICcrbm0rKHRvcE5hclN0cj8nLiBBcm91bmQ6ICcrdG9wTmFyU3RyKycuJzonJyksZmVhcjonVGhyZWF0IHBlcmNlcHRpb24gaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKX07CiAgICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7Y3VtQT1hMjsKICAgICAgdmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICAgIH0pLmpvaW4oJycpOwogICAgdmFyIGVkZXNjPXthbnhpZXR5OidVbmNlcnRhaW50eSwgd29ycnknLGFuZ2VyOidPdXRyYWdlLCBwcm90ZXN0Jyxob3BlOidPcHRpbWlzbSwgcHJvZ3Jlc3MnLHByaWRlOidBY2hpZXZlbWVudCwgaWRlbnRpdHknLGZlYXI6J1RocmVhdCwgaW5zZWN1cml0eSd9OwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdUaGUgZW1vdGlvbmFsIHVuZGVyY3VycmVudCBvZiBzaWduYWxzIGZyb20gJytubSsnLiBXaGF0IHRvbmUgZG9taW5hdGVzIHRoZSBwb2xpdGljYWwgZGlzY291cnNlIOKAlCBvdXRyYWdlLCBob3BlLCBmZWFyLCBvciBhbnhpZXR5PycrCiAgICAnPC9kaXY+JysKICAgICghaGFzRW1vcz8nPGRpdiBzdHlsZT0icGFkZGluZzo2cHggMTFweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO21hcmdpbi1ib3R0b206MTBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KSI+RXN0aW1hdGVkIGZyb20gc2lnbmFsIGRpcmVjdGlvbiDigJQgbGltaXRlZCBkaXJlY3QgZW1vdGlvbiBkYXRhLjwvZGl2Pic6JycpKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrcGFsW2RvbUVtb10rJzE0O2JvcmRlcjoxcHggc29saWQgJytwYWxbZG9tRW1vXSsnMzM7bWFyZ2luLWJvdHRvbToxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6JytwYWxbZG9tRW1vXSsnO21hcmdpbi1ib3R0b206NnB4Ij5Eb21pbmFudCBlbW90aW9uPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyNnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHgiPicrZG9tUGN0KyclIMK3ICcrbm0rJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo4cHg7bGluZS1oZWlnaHQ6MS41O2ZvbnQtc3R5bGU6aXRhbGljIj4nK3doYXRJdFtkb21FbW9dKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RW1vdGlvbmFsIGJyZWFrZG93bjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE2cHg7Ij4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDc2IDc2IiBzdHlsZT0id2lkdGg6NzJweDtoZWlnaHQ6NzJweDtmbGV4LXNocmluazowIj4nK2FyY3MrJzwvc3ZnPicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjVweDsiPicrCiAgICAgICAgICAgIGVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgICAgICAgICAgdmFyIGs9a3ZbMF0sdj1rdlsxXSxwY3Q9TWF0aC5yb3VuZCh2KjEwMC90b3QpOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpjZW50ZXI7bWFyZ2luLWJvdHRvbToycHg7Ij4nKwogICAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OyI+PHNwYW4gc3R5bGU9IndpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6MnB4O2JhY2tncm91bmQ6JytwYWxba10rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjonKyhrPT09ZG9tRW1vPyd2YXIoLS1pbmspJzondmFyKC0tZGltKScpKyciPicray5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStrLnNsaWNlKDEpKyc8L3NwYW4+PC9kaXY+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWluaykiPicrcGN0KyclPC9zcGFuPicrCiAgICAgICAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrcGN0KyclO2JhY2tncm91bmQ6JytwYWxba10rJztvcGFjaXR5OjAuNztib3JkZXItcmFkaXVzOjFweCI+PC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAgIChrPT09ZG9tRW1vPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytlZGVzY1trXSsnPC9kaXY+JzonJykrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWwgaGVhZGxpbmVzPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NHB4OyI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLnNsaWNlKDAsNSkubWFwKGZ1bmN0aW9uKGEpewogICAgICAgICAgICAgIHZhciBlQ29sb3I9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjZweDtwYWRkaW5nOjZweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ij4nKwogICAgICAgICAgICAgICAgKGEuZW1vdGlvbj8nPHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytlQ29sb3JbYS5lbW90aW9uXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrO21hcmdpbi10b3A6NXB4O2ZsZXgtc2hyaW5rOjAiPjwvc3Bhbj4nOicnKSsKICAgICAgICAgICAgICAgICc8ZGl2PjxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS40Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHgiPicrKGEuc3JjfHwnJykrKGEuZW1vdGlvbj8nIMK3ICcrYS5lbW90aW9uOicnKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJyk6CiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo0cHggMCI+Tm8gc2lnbmFscyB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgewogICAgdmFyIHZlbD1kLnZlbG9jaXR5fHwwOwogICAgdmFyIHZlbERpcj12ZWw+MC4xNT8nUmlzaW5nIGZhc3QnOnZlbD4wLjA1PydSaXNpbmcnOnZlbDwtMC4xPydDb29saW5nIGZhc3QnOnZlbDwtMC4wMj8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICB2YXIgdmVsQ29sPXZlbD4wLjA1PycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgIHZhciB2ZWxEZXNjPXsnUmlzaW5nIGZhc3QnOidTaWduYWwgdm9sdW1lIHN1cmdpbmcuJywnUmlzaW5nJzonQXR0ZW50aW9uIGJ1aWxkaW5nLicsJ1N0YWJsZSc6J0JhbGFuY2VkIG1vbWVudHVtLicsJ0Nvb2xpbmcnOidBdHRlbnRpb24gZmFkaW5nLicsJ0Nvb2xpbmcgZmFzdCc6J1NoYXJwIHNpZ25hbCBkZWNheS4nfTsKICAgIHZhciBuYXJyMz1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHJpc2luZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSd1cCc7fSk7CiAgICB2YXIgZmFsbGluZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSdkb3duJzt9KTsKICAgIHZhciBjdHg9Jyc7CiAgICBpZih2ZWw+MC4wNSYmcmlzaW5nTmFycy5sZW5ndGgpIGN0eD0nRHJpdmVuIGJ5IHJpc2luZyBzaWduYWxzIGFyb3VuZCA8c3Ryb25nPicrcmlzaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+Lic7CiAgICBlbHNlIGlmKHZlbDwtMC4wNSYmZmFsbGluZ05hcnMubGVuZ3RoKSBjdHg9J1NpZ25hbHMgYXJvdW5kIDxzdHJvbmc+JytmYWxsaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+IGxvc2luZyB0cmFjdGlvbi4nOwogICAgZWxzZSBjdHg9J1NpZ25hbCB2b2x1bWUgJysodmVsPjAuMDI/J2J1aWxkaW5nJzonc3RhYmxlJykrJyBpbiAnK25tKycuJzsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnSXMgYXR0ZW50aW9uIGZvciAnK25tKycgZ3Jvd2luZyBvciBmYWRpbmc/IFJpc2luZyBtb21lbnR1bSBtZWFucyBhIG5hcnJhdGl2ZSBpcyBhY2NlbGVyYXRpbmcuIENvb2xpbmcgbWVhbnMgdGhlIHN0b3J5IGlzIGxvc2luZyB0cmFjdGlvbi4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrdmVsQ29sKycxNDtib3JkZXI6MXB4IHNvbGlkICcrdmVsQ29sKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3ZlbENvbCsnO21hcmdpbi1ib3R0b206NnB4Ij5TaWduYWwgbW9tZW50dW08L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo4cHg7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MzJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjonK3ZlbENvbCsnO2ZvbnQtd2VpZ2h0OjUwMCI+Jyt2ZWxEaXIrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC1zdHlsZTppdGFsaWM7bGluZS1oZWlnaHQ6MS41Ij4nK3ZlbERlc2NbdmVsRGlyXSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjttYXJnaW4tdG9wOjEwcHg7cGFkZGluZy10b3A6MTBweDtib3JkZXItdG9wOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIj4nK2N0eCsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlZlbG9jaXR5PC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIiBzdHlsZT0iZm9udC1zaXplOjE4cHg7Y29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCDOtDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrKGQuZGVsdGE+PTA/J3VwJzonZG4nKSsnIj4nKyhkLmRlbHRhPj0wPycrJzonJykrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAocmlzaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QWNjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICByaXNpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojZTA1YTI4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJykrCiAgICAgIChmYWxsaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RGVjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICBmYWxsaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoNTksMTg0LDIxNiwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojM2JiOGQ4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJyk7CiAgfQoKICBwYW5lbC5pbm5lckhUTUw9aGVhZGVyK2JvZHk7Cn0KCgpmdW5jdGlvbiB0b2dnbGVGYXYobm0pewogIGlmKEZBVlMuaGFzKG5tKSkgRkFWUy5kZWxldGUobm0pO2Vsc2UgRkFWUy5hZGQobm0pOwogIHJlbmRlclBhbmVsKFNFTCk7cmVuZGVyRmF2cygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdnMoKXsKICB2YXIgcm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXYtcm93Jyk7CiAgaWYoIUZBVlMuc2l6ZSl7cm93LmlubmVySFRNTD0nPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+JztyZXR1cm47fQogIHJvdy5pbm5lckhUTUw9QXJyYXkuZnJvbShGQVZTKS5tYXAoZnVuY3Rpb24obm0pewogICAgdmFyIGQ9ZyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8nI2UwNWEyOCc6JyMzYmI4ZDgnOwogICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZmF2LWNhcmQiIG9uY2xpY2s9InNlbGVjdF8oXCcnK25tKydcJykiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1oZWFkIj48c3BhbiBjbGFzcz0iZmMtbmFtZSI+JytubSsnPC9zcGFuPjxzcGFuIGNsYXNzPSJmYy1zYyI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPk5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj4yNGg8L3NwYW4+PHNwYW4gY2xhc3M9InYiIHN0eWxlPSJjb2xvcjonK2RDKyciPicrZFMrZC5kZWx0YSsnPC9zcGFuPjwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKGMpewogIGMuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7bGF5ZXI9Yy5kYXRhc2V0LmxheWVyO2FwcGx5TGF5ZXIoKTsKICB9KTsKfSk7CgpmdW5jdGlvbiB1cGRhdGVDbG9jaygpewogIHZhciBub3c9bmV3IERhdGUoKSxpc3Q9bmV3IERhdGUobm93LmdldFRpbWUoKStub3cuZ2V0VGltZXpvbmVPZmZzZXQoKSo2MDAwMCsxOTgwMDAwMCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Nsb2NrJykudGV4dENvbnRlbnQ9U3RyaW5nKGlzdC5nZXRIb3VycygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0TWludXRlcygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0U2Vjb25kcygpKS5wYWRTdGFydCgyLCcwJykrJyBJU1QnOwp9CnNldEludGVydmFsKHVwZGF0ZUNsb2NrLDEwMDApO3VwZGF0ZUNsb2NrKCk7CgovLyBJTklUIOKAlCB3YWl0IGZvciBET00KLy8gaSBidXR0b24gdG9vbHRpcCDigJQgdXNlcyBmaXhlZCBwb3NpdGlvbmluZyBzbyBpdCdzIG5ldmVyIGNsaXBwZWQKKGZ1bmN0aW9uKCl7CiAgdmFyIHRpcD1udWxsOwogIGZ1bmN0aW9uIHNob3dUaXAoZSl7CiAgICBpZighdGlwKXt0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2x0YWItdG9vbHRpcCcpO30KICAgIHZhciB0eHQ9dGhpcy5nZXRBdHRyaWJ1dGUoJ2RhdGEtdGlwJyk7CiAgICBpZighdHh0fHwhdGlwKSByZXR1cm47CiAgICB0aXAudGV4dENvbnRlbnQ9dHh0OwogICAgdGlwLmNsYXNzTGlzdC5hZGQoJ3Zpc2libGUnKTsKICAgIHZhciByZWN0PXRoaXMuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICB2YXIgdHc9MjQwOwogICAgdmFyIGxlZnQ9TWF0aC5taW4ocmVjdC5sZWZ0LHdpbmRvdy5pbm5lcldpZHRoLXR3LTEwKTsKICAgIHRpcC5zdHlsZS5sZWZ0PWxlZnQrJ3B4JzsKICAgIHRpcC5zdHlsZS50b3A9KHJlY3QudG9wLTEwLXRpcC5vZmZzZXRIZWlnaHR8fHJlY3QudG9wLTgwKSsncHgnOwogICAgLy8gUmVwb3NpdGlvbiBhZnRlciByZW5kZXIKICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmdW5jdGlvbigpewogICAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC10aXAub2Zmc2V0SGVpZ2h0LTgpKydweCc7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gaGlkZVRpcCgpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICBpZih0aXApIHRpcC5jbGFzc0xpc3QucmVtb3ZlKCd2aXNpYmxlJyk7CiAgfQogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3ZlcicsZnVuY3Rpb24oZSl7CiAgICBpZihlLnRhcmdldC5jbGFzc0xpc3QuY29udGFpbnMoJ2x0YWItaW5mbycpKSBzaG93VGlwLmNhbGwoZS50YXJnZXQsZSk7CiAgfSk7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VvdXQnLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgaGlkZVRpcCgpOwogIH0pOwp9KSgpOwoKZnVuY3Rpb24gaW5pdCgpewogIHJlbmRlclN0cmlwKCczbScpOwoKICAvLyBMb2FkIG1hcCB3aXRoIHJldHJ5CiAgdmFyIG1hcEF0dGVtcHRzPTA7CiAgZnVuY3Rpb24gdHJ5TG9hZE1hcCgpewogICAgaWYodHlwZW9mIHRvcG9qc29uPT09J3VuZGVmaW5lZCcpewogICAgICBpZihtYXBBdHRlbXB0cysrPDEwKXtzZXRUaW1lb3V0KHRyeUxvYWRNYXAsMzAwKTt9CiAgICAgIHJldHVybjsKICAgIH0KICAgIGxvYWRNYXAoKTsKICB9CiAgdHJ5TG9hZE1hcCgpOwoKICAvLyBMb2FkIGZ1bGwgY2FjaGVkIHNuYXBzaG90IGltbWVkaWF0ZWx5IGZvciBpbnN0YW50IGRhdGEKICBmZXRjaEZ1bGxTbmFwc2hvdCgpLnRoZW4oZnVuY3Rpb24ob2spewogICAgaWYob2spewogICAgICByZW5kZXJNb21lbnR1bSgpOwogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7c3RhcnRQb2xsaW5nKCk7fSwxMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHN0YXJ0UG9sbGluZygpOwogICAgfQogIH0pOwoKICAvLyBSZXRyeSBtYXAgaWYgc3RpbGwgZW1wdHkKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSwzMDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSw2MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw1MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDgwMDApOwp9CmlmKGRvY3VtZW50LnJlYWR5U3RhdGU9PT0nbG9hZGluZycpewogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ0RPTUNvbnRlbnRMb2FkZWQnLCBpbml0KTsKfSBlbHNlIHsKICAvLyBBbHJlYWR5IGxvYWRlZCDigJQgYnV0IHdhaXQgb25lIHRpY2sgdG8gZW5zdXJlIGFsbCBzY3JpcHRzIHBhcnNlZAogIHNldFRpbWVvdXQoaW5pdCwgMCk7Cn0KCi8vIFJFUExBWSBJTkRJQQp2YXIgUkVQTEFZX1BFUklPRFM9eyc3ZCc6e2RheXM6NyxsYWJlbDonUGFzdCA3IGRheXMnfSwnMzBkJzp7ZGF5czozMCxsYWJlbDonUGFzdCAzMCBkYXlzJ30sJzZtJzp7ZGF5czoxODAsbGFiZWw6J1Bhc3QgNiBtb250aHMnfSwnZWxlY3Rpb24nOntkYXlzOjkwLGxhYmVsOidFbGVjdGlvbiBzZWFzb24gMjAyNCd9fTsKdmFyIHJlcGxheVBlcmlvZD0nN2QnLHJlcGxheVBvcz0wLHJlcGxheVBsYXlpbmc9ZmFsc2UscmVwbGF5VGltZXI9bnVsbCxyZXBsYXlTcGVlZD0xLGxhc3RTbmFwUG9zPS0xOwpmdW5jdGlvbiBmbXREYXRlKGQpe3JldHVybiBkLnRvTG9jYWxlRGF0ZVN0cmluZygnZW4tSU4nLHtkYXk6J251bWVyaWMnLG1vbnRoOidzaG9ydCd9KTt9CmZ1bmN0aW9uIGluaXRSZXBsYXkoKXsKICB2YXIgcD1SRVBMQVlfUEVSSU9EU1tyZXBsYXlQZXJpb2RdLG5vdz1uZXcgRGF0ZSgpLHN0YXJ0PW5ldyBEYXRlKG5vdy1wLmRheXMqODY0MDAwMDApOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtZGF0ZXMnKTsKICBpZihlbCllbC5pbm5lckhUTUw9JzxzcGFuPicrZm10RGF0ZShzdGFydCkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjMzKSkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjY2KSkrJzwvc3Bhbj48c3Bhbj5Ub2RheTwvc3Bhbj4nOwogIHNldFJlcGxheVBvcygwKTsKfQpmdW5jdGlvbiBzZXRSZXBsYXlQb3MocG9zKXsKICByZXBsYXlQb3M9TWF0aC5tYXgoMCxNYXRoLm1pbigxLHBvcykpOwogIHZhciBmaWxsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1maWxsJyksdGh1bWI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRodW1iJyksZGF0ZUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1jdXJyZW50LWRhdGUnKTsKICBpZihmaWxsKWZpbGwuc3R5bGUud2lkdGg9KHJlcGxheVBvcyoxMDApKyclJzsKICBpZih0aHVtYil0aHVtYi5zdHlsZS5sZWZ0PShyZXBsYXlQb3MqMTAwKSsnJSc7CiAgdmFyIHA9UkVQTEFZX1BFUklPRFNbcmVwbGF5UGVyaW9kXSxub3c9bmV3IERhdGUoKSxzdGFydD1uZXcgRGF0ZShub3ctcC5kYXlzKjg2NDAwMDAwKSxjdXI9bmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3JlcGxheVBvcypwLmRheXMqODY0MDAwMDApOwogIGlmKGRhdGVFbClkYXRlRWwudGV4dENvbnRlbnQ9Zm10RGF0ZShjdXIpKycg4oCUICcrcC5sYWJlbDsKICB2YXIgc2NhbGU9MC4zNStyZXBsYXlQb3MqMC42NTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLHNhPShkLmF0dGVudGlvbnx8MCkqc2NhbGU7CiAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4gKHguYXR0ZW50aW9ufHwwKSpzY2FsZTt9KTsKICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxLG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzYS1tbikvKG14LW1uKSkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKHNhKSk7cC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsTWF0aC5tYXgoMC4yLDAuMituKjAuOCkpOwogIH0pOwogIGlmKE1hdGguYWJzKHJlcGxheVBvcy1sYXN0U25hcFBvcyk+MC4xMil7bGFzdFNuYXBQb3M9cmVwbGF5UG9zO3VwZGF0ZVJlcGxheVNuYXBzaG90KHJlcGxheVBvcyk7fQp9CmZ1bmN0aW9uIHVwZGF0ZVJlcGxheVNuYXBzaG90KHBvcyl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uYXR0ZW50aW9uPjA7fSkubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm57bmFtZTprdlswXSxhdHQ6TWF0aC5yb3VuZCgoa3ZbMV0uYXR0ZW50aW9ufHwwKSooMC4zNStwb3MqMC42NSkpLG5hcjooa3ZbMV0ubmFycmF0aXZlcyYma3ZbMV0ubmFycmF0aXZlc1swXT9rdlsxXS5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCcpfTt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pLnNsaWNlKDAsNik7CiAgdmFyIHNuYXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXNuYXAtc3RhdGVzJyk7CiAgaWYoIXNuYXApcmV0dXJuOwogIGlmKCF0b3AubGVuZ3RoKXtzbmFwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5ObyBzaWduYWwgZGF0YSB5ZXQuPC9kaXY+JztyZXR1cm47fQogIHNuYXAuaW5uZXJIVE1MPXRvcC5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuICc8ZGl2IGNsYXNzPSJycC1zdGF0ZS1jYXJkIj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1uYW1lIj4nK3MubmFtZSsnPC9kaXY+PGRpdiBjbGFzcz0icnAtc3RhdGUtbmFyIj4nK3MubmFyKyc8L2Rpdj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1hdHQiPkF0dGVudGlvbiAnK3MuYXR0Kyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwp9CmZ1bmN0aW9uIHRvZ2dsZVJlcGxheSgpewogIHJlcGxheVBsYXlpbmc9IXJlcGxheVBsYXlpbmc7CiAgdmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpOwogIGlmKHJlcGxheVBsYXlpbmcpe2lmKHJlcGxheVBvcz49MC45OSlzZXRSZXBsYXlQb3MoMCk7aWYoaWNvbilpY29uLnNldEF0dHJpYnV0ZSgncG9pbnRzJywnMywyIDcsMiA3LDggMyw4IE04LDIgMTIsMiAxMiw4IDgsOCcpO3J1blJlcGxheSgpO30KICBlbHNle2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO30KfQpmdW5jdGlvbiBydW5SZXBsYXkoKXsKICBjbGVhckludGVydmFsKHJlcGxheVRpbWVyKTsKICByZXBsYXlUaW1lcj1zZXRJbnRlcnZhbChmdW5jdGlvbigpewogICAgcmVwbGF5UG9zKz0wLjAwMypyZXBsYXlTcGVlZDsKICAgIGlmKHJlcGxheVBvcz49MSl7cmVwbGF5UG9zPTE7c2V0UmVwbGF5UG9zKDEpO3JlcGxheVBsYXlpbmc9ZmFsc2U7dmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpO2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO3JldHVybjt9CiAgICBzZXRSZXBsYXlQb3MocmVwbGF5UG9zKTsKICB9LDYwKTsKfQooZnVuY3Rpb24oKXt2YXIgdHJhY2s9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRyYWNrJyk7aWYoIXRyYWNrKXJldHVybjt2YXIgZHJhZz1mYWxzZTsKdHJhY2suYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vkb3duJyxmdW5jdGlvbihlKXtkcmFnPXRydWU7dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXtpZighZHJhZylyZXR1cm47dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2V1cCcsZnVuY3Rpb24oKXtpZihkcmFnKXtkcmFnPWZhbHNlO2lmKCFyZXBsYXlQbGF5aW5nKWFwcGx5TGF5ZXIoKTt9fSk7fSkoKTsKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYnRuKXtidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7ZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYil7Yi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTtidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVwbGF5UGVyaW9kPWJ0bi5kYXRhc2V0LnBlcmlvZDtyZXBsYXlQb3M9MDtsYXN0U25hcFBvcz0tMTtpbml0UmVwbGF5KCk7fSk7fSk7CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGJ0bil7YnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe2RvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGIpe2IuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7YnRuLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlcGxheVNwZWVkPXBhcnNlSW50KGJ0bi5kYXRhc2V0LnNwZCk7fSk7fSk7CmluaXRSZXBsYXkoKTsKc2V0VGltZW91dChmdW5jdGlvbigpewogIC8vIEF1dG8tc2VsZWN0IGhvdHRlc3Qgc3RhdGUgZnJvbSBMSVZFIGRhdGEKICB2YXIgc3JjPU9iamVjdC5rZXlzKExJVkUpLmxlbmd0aD9MSVZFOlNEOwogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoc3JjKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApewogICAgdmFyIGVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9IicrdG9wWzBdKyciXScpOwogICAgaWYoZWwpIHNlbGVjdF8odG9wWzBdKTsKICB9Cn0sMzAwMCk7CnNldFRpbWVvdXQocmVuZGVyRmF2cywyNDAwKTsKPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPgo="

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


@app.get("/api/full-snapshot")
async def full_snapshot():
    """Returns complete data for all states in one request.
    Cached and rebuilt every 15 minutes with the ingest cycle.
    Frontend uses this for instant first-load."""
    if not store.scores:
        return {"states": [], "snapshot": {}, "insights": {}, "warming_up": True}
    states_data = []
    for state, score in store.scores.items():
        if isinstance(score, dict) and score:
            states_data.append(score)
    return {
        "states": states_data,
        "snapshot": store.cache_snapshot or {},
        "insights": store.cache_insights or {},
        "narrative_insight": getattr(store, "cache_narrative_insight", None) or {},
        "as_of": store.cache_built_at.isoformat() if store.cache_built_at else None,
        "warming_up": False,
    }


@app.get("/api/narrative-insight")
async def narrative_insight():
    """AI-generated 24h national narrative insight from top signals across India."""
    if not store.scores:
        return {"insight": None, "error": "warming up"}

    # Gather top headlines from highest-attention states
    top_states = sorted(
        [(s, d) for s, d in store.scores.items() if isinstance(d, dict) and d.get("signal_count", 0) > 0],
        key=lambda x: x[1].get("attention", 0), reverse=True
    )[:8]

    if not top_states:
        return {"insight": None, "error": "no data"}

    # Build context from real signals
    context_lines = []
    all_narratives = {}
    for state, data in top_states:
        headlines = [a["txt"] for a in data.get("articles", [])[:3]]
        top_nar = data.get("dominant_narrative", "")
        confidence = data.get("confidence", "LOW")
        if headlines and confidence != "LOW":
            context_lines.append(f"{state} ({top_nar}): {'; '.join(headlines[:2])}")
        # Aggregate narratives
        for n in data.get("narratives", []):
            all_narratives[n["name"]] = all_narratives.get(n["name"], 0) + n["val"]

    top_narratives = sorted(all_narratives.items(), key=lambda x: x[1], reverse=True)[:6]
    narrative_str = ", ".join(f"{n[0]}" for n in top_narratives)

    if not context_lines:
        return {"insight": None, "error": "insufficient quality signals"}

    # Use AI to synthesize insight
    prompt = f"""You are an analyst for Pulse of India — a political attention observatory. 
Based on these real signals from across India in the last 24 hours, write ONE concise, analytical sentence (max 40 words) describing the dominant national narrative shift.

Top active states and their signals:
{chr(10).join(context_lines[:6])}

Top narratives nationally: {narrative_str}

Rules:
- Be specific and analytical, not generic
- Name actual narratives/themes, not just states  
- Sound like an intelligence briefing, not a news headline
- Do NOT start with "India's attention" or "Across India"
- Start with the dominant theme/narrative itself
- Example good format: "Border security concerns intensify as [specific context], with [state] signals showing [specific trend]."
- If data is insufficient, say: "Signals are dispersed — no single dominant narrative has emerged."

Write only the sentence, nothing else."""

    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        insight_text = msg.content[0].text.strip()
        # Cache it
        store.cache_narrative_insight = {
            "text": insight_text,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "top_narratives": [n[0] for n in top_narratives[:4]],
        }
        return store.cache_narrative_insight
    except Exception as e:
        # Fallback: construct from data without AI
        if top_narratives:
            top = top_narratives[0][0]
            second = top_narratives[1][0] if len(top_narratives) > 1 else None
            hottest = top_states[0][0] if top_states else "multiple states"
            text = f"{top.capitalize()} signals dominate national discourse"
            if second:
                text += f" alongside {second}"
            text += f", with {hottest} leading the conversation."
        else:
            text = "Signals are dispersed — monitoring across 30 states."
        return {"text": text, "as_of": datetime.now(timezone.utc).isoformat(),
                "top_narratives": [n[0] for n in top_narratives[:4]], "fallback": True}


@app.get("/api/state-context/{state_name}")
async def state_context(state_name: str):
    """
    Fresh contextual brief for a state — combines:
    1. Live signals from our store (what's spiking)
    2. Fresh Google News RSS search (broader context)
    3. AI synthesis into a readable 3-4 sentence brief
    
    Cached per state for 30 minutes.
    """
    import hashlib, time

    # Check cache (30 min)
    cache_key = f"ctx_{state_name}"
    cached = getattr(store, "context_cache", {}).get(cache_key)
    if cached and (time.time() - cached["ts"]) < 1800:
        return cached["data"]

    # Get our stored signals
    score = store.scores.get(state_name, {})
    stored_articles = score.get("articles", [])[:5]
    stored_narratives = score.get("narratives", [])[:3]
    confidence = score.get("confidence", "LOW")
    signal_count = score.get("signal_count", 0)

    # Fresh Google News search for broader context
    fresh_headlines = []
    try:
        import httpx, feedparser
        search_q = state_name.replace(" ", "+") + "+news+today"
        url = f"https://news.google.com/rss/search?q={search_q}&hl=en-IN&gl=IN&ceid=IN:en"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url)
            feed = feedparser.parse(r.text)
            seen = set()
            for entry in feed.entries[:15]:
                title = entry.get("title", "").strip()
                tk = title[:50].lower()
                if tk and tk not in seen:
                    seen.add(tk)
                    fresh_headlines.append(title)
                if len(fresh_headlines) >= 8:
                    break
    except Exception as e:
        print(f"[context] fresh search error: {e}")

    # Signal headlines
    signal_headlines = [a["txt"] for a in stored_articles if a.get("txt")]
    narrative_str = ", ".join(n["name"] for n in stored_narratives) if stored_narratives else "general news"

    # All headlines combined (deduplicated)
    all_headlines = []
    seen_h = set()
    for h in (signal_headlines + fresh_headlines):
        tk = h[:50].lower()
        if tk not in seen_h:
            seen_h.add(tk)
            all_headlines.append(h)

    if not all_headlines:
        result = {
            "brief": f"No recent signals collected for {state_name}. The state appears quiet in national discourse.",
            "source": "fallback",
            "signal_count": 0,
            "narratives": [],
        }
        return result

    # Build AI prompt
    prompt = f"""You are a political analyst writing for Pulse of India — an attention observatory.

Write a 3-4 sentence contextual brief about {state_name} based on these recent headlines.
The brief should explain: what is happening, why it matters politically, and what the public mood signals suggest.

Recent headlines from {state_name}:
{chr(10).join(f"- {h}" for h in all_headlines[:10])}

Signal analysis: {signal_count} signals detected. Top narratives: {narrative_str}. Signal confidence: {confidence}.

Guidelines:
- Be specific — name actual issues, people, events from the headlines
- Explain context so someone unfamiliar with the state understands why this matters
- If confidence is LOW, acknowledge limited data but still explain what we observe
- Keep it factual and observational — not opinionated
- 3-4 sentences maximum
- Start with the most significant development

Write only the brief, nothing else."""

    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        brief = msg.content[0].text.strip()
        source = "ai"
    except Exception as e:
        # Fallback: simple structured summary
        print(f"[context] AI error: {e}")
        if all_headlines:
            brief = f"Recent developments in {state_name}: {all_headlines[0]}. "
            if len(all_headlines) > 1:
                brief += f"Also in focus: {all_headlines[1]}. "
            if narrative_str != "general news":
                brief += f"Dominant signals point to {narrative_str}."
        else:
            brief = f"Limited signals from {state_name}. Monitoring regional and national sources."
        source = "pattern"

    result = {
        "brief": brief,
        "source": source,
        "signal_count": signal_count,
        "narratives": [n["name"] for n in stored_narratives],
        "headlines_used": len(all_headlines),
        "fresh_headlines": fresh_headlines[:5],
    }

    # Cache it
    if not hasattr(store, "context_cache"):
        store.context_cache = {}
    store.context_cache[cache_key] = {"ts": time.time(), "data": result}

    return result


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
