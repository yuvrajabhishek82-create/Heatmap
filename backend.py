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
import json
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

# PostgreSQL — optional, falls back to RAM if not configured
try:
    import asyncpg
    HAS_PG = True
except ImportError:
    HAS_PG = False
    print("[db] asyncpg not installed — running in RAM-only mode")

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

# ════════════════════════════════════════════════════════════════
# DATABASE LAYER — PostgreSQL (optional, falls back to RAM)
# Set DATABASE_URL env var on Render to enable persistence
# ════════════════════════════════════════════════════════════════

DB_URL = os.getenv("DATABASE_URL", "")

async def get_db() -> "asyncpg.Connection | None":
    if not HAS_PG or not DB_URL:
        return None
    try:
        return await asyncpg.connect(DB_URL)
    except Exception as e:
        print(f"[db] Connection failed: {e}")
        return None

async def init_db():
    """Create tables if they don't exist."""
    conn = await get_db()
    if not conn:
        print("[db] No database configured — using RAM only")
        return
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id          SERIAL PRIMARY KEY,
                state       TEXT NOT NULL,
                date        DATE NOT NULL,
                attention   REAL DEFAULT 0,
                velocity    REAL DEFAULT 0,
                delta       REAL DEFAULT 0,
                signal_count INT DEFAULT 0,
                source_count INT DEFAULT 0,
                confidence  TEXT DEFAULT 'LOW',
                dominant_emotion    TEXT,
                dominant_narrative  TEXT,
                emotions    JSONB DEFAULT '{}',
                narratives  JSONB DEFAULT '[]',
                top_articles JSONB DEFAULT '[]',
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(state, date)
            );

            CREATE TABLE IF NOT EXISTS signal_store (
                id          SERIAL PRIMARY KEY,
                state       TEXT NOT NULL,
                title       TEXT NOT NULL,
                source      TEXT,
                source_url  TEXT,
                published_at TIMESTAMPTZ NOT NULL,
                narratives  JSONB DEFAULT '[]',
                emotions    JSONB DEFAULT '{}',
                intensity   REAL DEFAULT 0.5,
                language    TEXT DEFAULT 'en',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_daily_state_date ON daily_snapshots(state, date DESC);
            CREATE INDEX IF NOT EXISTS idx_signal_state_pub ON signal_store(state, published_at DESC);
        """)
        print("[db] Tables ready")
    except Exception as e:
        print(f"[db] Init error: {e}")
    finally:
        await conn.close()

async def save_national_narrative_snapshot():
    """Save today's national narrative to DB — aggregated from all states."""
    conn = await get_db()
    if not conn:
        return
    try:
        today = datetime.now(timezone.utc).date()
        scored = [s for s in store.scores.values() if isinstance(s, dict) and s.get("signal_count",0)>0]
        if not scored:
            return
        # Aggregate narratives nationally
        nar_all: dict[str, float] = {}
        emo_all: dict[str, float] = {}
        for s in scored:
            for n in s.get("narratives", []):
                nar_all[n["name"]] = nar_all.get(n["name"], 0) + n.get("val", 0)
            for k, v in s.get("emotions", {}).items():
                emo_all[k] = emo_all.get(k, 0) + v
        top_nars = sorted(nar_all.items(), key=lambda x: x[1], reverse=True)[:5]
        top_emo = max(emo_all.items(), key=lambda x: x[1])[0] if emo_all else None
        hottest = max(scored, key=lambda s: s.get("attention", 0))

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS national_narratives (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL UNIQUE,
                top_narratives JSONB DEFAULT '[]',
                dominant_emotion TEXT,
                hottest_state TEXT,
                total_signals INT DEFAULT 0,
                summary TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            INSERT INTO national_narratives
                (date, top_narratives, dominant_emotion, hottest_state, total_signals)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT (date) DO UPDATE SET
                top_narratives=EXCLUDED.top_narratives,
                dominant_emotion=EXCLUDED.dominant_emotion,
                hottest_state=EXCLUDED.hottest_state,
                total_signals=EXCLUDED.total_signals
        """,
            today,
            json.dumps([{"name":n[0],"val":round(n[1],1)} for n in top_nars]),
            top_emo,
            hottest.get("name",""),
            sum(s.get("signal_count",0) for s in scored)
        )
    except Exception as e:
        print(f"[db] National narrative save error: {e}")
    finally:
        await conn.close()

async def save_daily_snapshot(state: str, score: dict):
    """Save today's state score to DB. Called after each ingest cycle."""
    conn = await get_db()
    if not conn:
        return
    try:
        today = datetime.now(timezone.utc).date()
        await conn.execute("""
            INSERT INTO daily_snapshots
                (state, date, attention, velocity, delta, signal_count, source_count,
                 confidence, dominant_emotion, dominant_narrative, emotions, narratives, top_articles)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (state, date) DO UPDATE SET
                attention=EXCLUDED.attention, velocity=EXCLUDED.velocity,
                delta=EXCLUDED.delta, signal_count=EXCLUDED.signal_count,
                source_count=EXCLUDED.source_count, confidence=EXCLUDED.confidence,
                dominant_emotion=EXCLUDED.dominant_emotion,
                dominant_narrative=EXCLUDED.dominant_narrative,
                emotions=EXCLUDED.emotions, narratives=EXCLUDED.narratives,
                top_articles=EXCLUDED.top_articles
        """,
            state, today,
            score.get("attention", 0), score.get("velocity", 0), score.get("delta_24h", 0),
            score.get("signal_count", 0), score.get("source_count", 0),
            score.get("confidence", "LOW"),
            score.get("dominant_emotion"), score.get("dominant_narrative"),
            json.dumps(score.get("emotions", {})),
            json.dumps(score.get("narratives", [])),
            json.dumps([{"src": a["src"], "txt": a["txt"]} for a in score.get("articles", [])[:5]])
        )
    except Exception as e:
        print(f"[db] Save snapshot error for {state}: {e}")
    finally:
        await conn.close()

async def save_signals_to_db(state: str, signals: list):
    """Persist raw signals to DB for historical replay."""
    conn = await get_db()
    if not conn:
        return
    try:
        rows = []
        for s in signals[-50:]:  # Save last 50 per state per cycle
            rows.append((
                state, s.get("title",""), s.get("source",""),
                s.get("source_url",""), s.get("published_at", datetime.now(timezone.utc)),
                json.dumps(s.get("narratives",[])), json.dumps(s.get("emotions",{})),
                s.get("intensity", 0.5), s.get("language","en")
            ))
        if rows:
            await conn.executemany("""
                INSERT INTO signal_store
                    (state, title, source, source_url, published_at, narratives, emotions, intensity, language)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT DO NOTHING
            """, rows)
    except Exception as e:
        print(f"[db] Save signals error: {e}")
    finally:
        await conn.close()

async def load_signals_from_db():
    """On startup: load last 48h of signals from DB into RAM store."""
    conn = await get_db()
    if not conn:
        return 0
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        rows = await conn.fetch("""
            SELECT state, title, source, source_url, published_at,
                   narratives, emotions, intensity, language
            FROM signal_store
            WHERE published_at > $1
            ORDER BY published_at DESC
        """, cutoff)
        loaded = 0
        for row in rows:
            sig = {
                "title": row["title"], "source": row["source"],
                "source_url": row["source_url"], "published_at": row["published_at"],
                "narratives": json.loads(row["narratives"]),
                "emotions": json.loads(row["emotions"]),
                "intensity": row["intensity"], "language": row["language"],
                "body": "",
            }
            state = row["state"]
            if state not in store.signals:
                store.signals[state] = []
            # Check not duplicate
            existing_urls = {s.get("source_url","") for s in store.signals[state]}
            if sig["source_url"] not in existing_urls:
                store.signals[state].append(sig)
                loaded += 1
        print(f"[db] Loaded {loaded} historical signals into RAM")
        return loaded
    except Exception as e:
        print(f"[db] Load signals error: {e}")
        return 0
    finally:
        await conn.close()

async def get_historical_snapshots(days: int = 7) -> dict:
    """Get daily attention scores for all states for the last N days."""
    conn = await get_db()
    if not conn:
        return {}
    try:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
        rows = await conn.fetch("""
            SELECT state, date, attention, velocity, dominant_emotion,
                   dominant_narrative, signal_count, confidence
            FROM daily_snapshots
            WHERE date >= $1
            ORDER BY state, date ASC
        """, cutoff)
        result = {}
        for row in rows:
            state = row["state"]
            if state not in result:
                result[state] = []
            result[state].append({
                "date": row["date"].isoformat(),
                "attention": row["attention"],
                "velocity": row["velocity"],
                "dominant_emotion": row["dominant_emotion"],
                "dominant_narrative": row["dominant_narrative"],
                "signal_count": row["signal_count"],
                "confidence": row["confidence"],
            })
        return result
    except Exception as e:
        print(f"[db] Historical fetch error: {e}")
        return {}
    finally:
        await conn.close()


INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu and Kashmir",
    "Puducherry", "Chandigarh", "Dadra and Nagar Haveli", "Lakshadweep",
    "Andaman and Nicobar Islands",
]

NARRATIVES = [
    "unemployment", "nationalism", "religion", "corruption", "economy",
    "inflation", "caste", "language politics", "regional identity",
    "education", "law & order", "governance", "elections", "security",
    "border issues", "environment", "farmer issues", "infrastructure",
    "tribal issues", "migration", "protest", "health",
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
    "Puducherry":        ["puducherry", "pondicherry", "n rangasamy", "pondy"],
    "Dadra and Nagar Haveli and Daman and Diu": ["dadra", "daman", "diu", "dnh"],
    "Lakshadweep":       ["lakshadweep", "laccadive", "kavaratti"],
    "Chandigarh":        ["chandigarh", "chandigarh ut"],
}

NARRATIVE_KEYWORDS: dict[str, list[str]] = {
    "unemployment":      ["unemployment", "jobless", "job loss", "recruitment exam cancelled", "vacancy", "layoffs", "no jobs", "retrenchment"],
    "corruption":        ["scam", "corruption", "bribe", "ed raid", "cbi probe", "paper leak", "embezzle", "disproportionate assets", "hawala", "money laundering"],
    "religion":          ["communal tension", "communal violence", "mosque demolition", "temple dispute", "religious conversion", "waqf", "cow vigilante", "religious riot", "mandir", "masjid row"],
    "economy":           ["gdp slowdown", "investment", "industry shutdown", "msme crisis", "export decline", "manufacturing"],
    "inflation":         ["inflation", "price rise", "petrol price hike", "onion price", "tomato price crisis", "cost of living", "cpi surge"],
    "caste":             ["caste violence", "caste discrimination", "dalit atrocity", "obc reservation", "quota protest", "sc/st", "upper caste", "jati"],
    "language politics": ["hindi imposition", "three-language policy", "kannada signage", "language dispute", "medium of instruction", "regional language"],
    "regional identity": ["sons of soil", "outsider", "marathi manoos", "bhumiputra", "domicile", "anti-outsider"],
    "education":         ["neet controversy", "jee protest", "paper leak", "exam scam", "student protest", "university shutdown", "fee hike", "student agitation"],
    "law & order":       ["murder", "rape", "gang rape", "lynching", "mob violence", "kidnapping", "encounter killing", "custodial death", "crime wave", "law and order breakdown"],
    "governance":        ["government collapse", "no confidence", "policy failure", "corruption charges", "misgovernance", "administrative failure", "chief minister resign", "political crisis", "constitutional crisis"],
    "elections":         ["election", "bypoll", "election commission", "poll violence", "vote buying", "candidate", "model code violation", "election result"],
    "border issues":     ["border tension", "loc violation", "infiltration", "drone attack", "bsf", "china border", "lac standoff", "myanmar border", "pakistan border"],
    "environment":       ["pollution crisis", "air quality hazardous", "flood", "drought", "heatwave deaths", "cyclone", "landslide", "earthquake", "environmental disaster"],
    "farmer issues":     ["farmer protest", "msp demand", "crop failure", "kisan andolan", "farm distress", "farmer suicide", "debt waiver", "agriculture crisis"],
    "infrastructure":    ["highway accident", "bridge collapse", "metro accident", "road death", "infrastructure failure", "power outage", "water crisis"],
    "nationalism":       ["anti-national", "sedition", "patriotic movement", "bharat mata", "independence day", "republic day"],
    "tribal issues":     ["tribal displacement", "adivasi protest", "vanvasi", "forest rights violation", "scheduled tribe atrocity", "tribal conflict"],
    "migration":         ["migrant workers", "mass migration", "exodus", "gulf return", "labour migration crisis", "displaced"],
    "security":          ["terror attack", "blast", "naxal attack", "maoist", "encounter", "ied blast", "security forces killed", "militant", "insurgent"],
    "protest":           ["protest", "agitation", "demonstration", "bandh", "strike", "rally", "march", "sit-in", "dharna", "blockade"],
    "health":            ["disease outbreak", "epidemic", "hospital crisis", "medicine shortage", "health emergency", "malnutrition", "encephalitis", "dengue outbreak"],
}

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anger":   ["outrage","fury","angry","furious","slams","blasts","condemns","protests erupt",
                "backlash","clashes","dispute","accuses","rejects","denounces",
                "criticizes","scam","corruption","fraud","violence erupts","riot","agitation",
                "demands resignation","opposes","uproar","controversy","tensions rise"],
    "anxiety": ["worry","concern","anxious","alarm","uncertain","crisis","panic","shortage",
                "inflation","unemployment","struggling","difficult","slowdown","recession",
                "debt","deficit","flood devastation","drought","disaster","tension","unrest",
                "instability","fear spreads","apprehensive","distress","economic anxiety"],
    "hope":    ["breakthrough","historic agreement","peace deal","major reform",
                "significant progress","transformative","milestone achieved",
                "economic revival","recovery signals","first time in history",
                "landmark decision","game changer","promising development"],
    "pride":   ["pride","historic victory","honor","celebrates achievement","proud moment",
                "awarded","international recognition","heritage","champion","gold medal",
                "landmark win","tribute","distinguished","celebrated achievement",
                "national pride","excellence recognised"],
    "fear":    ["terror attack","explosion","blast","naxal attack","militants strike",
                "security threat","unsafe","killing","crime wave","conflict escalates",
                "infiltration","casualties","ied","shoot","abduct","missing persons",
                "hostage","military crackdown","encounter","armed attack"],
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
        self.cache_built_at: datetime | None = None
        self.cache_snapshot: dict | None = None
        self.cache_insights: dict | None = None
        self.context_cache: dict = {}
        # 8-slot daily history per state (circular)
        self.history: dict[str, list[float]] = {s: [] for s in INDIAN_STATES}

    NOISE_KEYWORDS = [
        'trump', 'biden', 'ukraine', 'russia', 'gaza', 'israel', 'hamas',
        'elon musk', 'openai', 'chatgpt', 'north korea', 'nato',
        'white house', 'pentagon', 'federal reserve', 'wall street',
        'recipe', 'horoscope', 'astrology', 'box office collection',
        'bollywood gossip', 'celebrity wedding', 'music video', 'album release',
        'iphone launch', 'apple event', 'samsung launch', 'chatgpt update',
        'live score', 'match result', 'scorecard', 'fantasy team',
    ]
    NOISE_SOURCES = [
        'pinkvilla', 'bollywoodhungama', 'filmibeat', 'koimoi', 'buzzfeed',
    ]

    def add_signal(self, state: str, sig: dict) -> bool:
        """Returns True if the signal was new (not a dupe)."""
        title = sig.get("title", "").strip()
        if not title:
            return False

        # Filter irrelevant content at ingestion
        tl = title.lower()
        if any(kw in tl for kw in self.NOISE_KEYWORDS):
            return False
        src_url = sig.get("source_url", "").lower()
        if any(ns in src_url for ns in self.NOISE_SOURCES):
            return False

        url = sig.get("source_url", "")
        title_key = title[:55].lower().strip()

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
    "Puducherry":        ["Puducherry politics news", "Pondicherry governance"],
    "Chandigarh":        ["Chandigarh news politics", "Punjab Haryana capital"],
    "Dadra and Nagar Haveli": ["Dadra Nagar Haveli news", "DNH UT news"],
    "Lakshadweep":       ["Lakshadweep news", "Lakshadweep administration"],
    "Andaman and Nicobar Islands": ["Andaman Nicobar news", "Port Blair news"],
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

        # Reddit ingestion — state-specific subreddits
        reddit_sem = asyncio.Semaphore(5)
        async with httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "PulseOfIndia/1.0 signal-observer"},
            follow_redirects=True,
        ) as rclient:
            async def reddit_state(s):
                async with reddit_sem:
                    return await ingest_reddit_for_state(s, rclient)
            reddit_results = await asyncio.gather(
                *[reddit_state(s) for s in states], return_exceptions=True
            )
            reddit_total = sum(r for r in reddit_results if isinstance(r, int))
            total += reddit_total
            print(f"[ingest] Reddit state signals: +{reddit_total}")

        # Reddit national subreddits
        try:
            nr_national = await ingest_reddit_national()
            total += nr_national
            print(f"[ingest] Reddit national: +{nr_national} signals")
        except Exception as e:
            print(f"[reddit] national error: {e}")

    finally:
        store.ingest_running = False

    # YouTube signals — runs after ingest_running released
    if YOUTUBE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as yclient:
                yt_sem2 = asyncio.Semaphore(5)
                async def _yt(s):
                    async with yt_sem2:
                        return await ingest_youtube_for_state(s, yclient)
                yt_res = await asyncio.gather(
                    *[_yt(s) for s in states if s in STATE_YT_CHANNELS],
                    return_exceptions=True
                )
                yt_n = sum(r for r in yt_res if isinstance(r, int))
                yn_n = await ingest_youtube_national(yclient)
                total += yt_n + yn_n
                print(f"[ingest] YouTube: +{yt_n} state, +{yn_n} national")
        except Exception as e:
            print(f"[youtube] error: {e}")

    store.last_ingest = datetime.now(timezone.utc)
    print(f"[ingest] Done — {total} new signals total (RSS + Reddit + YouTube)")
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
    # Independent journalists — highest signal quality
    if "youtube" in s:
        # Trust already baked into intensity — return 1.0 here
        return 1.0
    # Reddit posts carry high weight — direct public discourse
    if "reddit" in s:
        return 1.5
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
    # Use higher denominator to prevent premature saturation
    # baseline * 6 means a state needs 4x its expected volume to hit ~80
    attention = round(min(95, 100 * math.tanh(normalized / (baseline * 6.0))), 1)
    attention = round(attention * conf_weight, 1)

    # Momentum — compare last 12h vs previous 12h for meaningful velocity
    sigs_12h_recent = [s for s in sigs_48h if (now - s["published_at"]).total_seconds() < 43200]
    sigs_12h_older  = [s for s in sigs_48h if 43200 <= (now - s["published_at"]).total_seconds() < 86400]
    recent_12 = len(sigs_12h_recent)
    older_12  = len(sigs_12h_older)
    prev_count = len(sigs_prev)
    delta_24h = round(float(raw_count - prev_count), 1)

    if older_12 > 0:
        raw_vel = (recent_12 - older_12) / max(older_12, 1)
    elif prev_count > 0:
        raw_vel = (raw_count - prev_count) / max(prev_count, 1)
    else:
        raw_vel = 0.0
    velocity = round(math.tanh(raw_vel * 2), 3)  # always -1 to +1
    if velocity > 0.3 and confidence in ("MEDIUM", "HIGH"):
        attention = round(min(95, attention * 1.15), 1)

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
    # Article significance scoring
    # Higher score = more important, more worth showing
    NATIONAL_SOURCES = {
        "ndtv", "the hindu", "hindustan times", "times of india", "india today",
        "the wire", "scroll", "firstpost", "news18", "republic", "zee news",
        "aaj tak", "abp news", "ani", "pti", "reuters", "bloomberg",
    }
    NOISE_TITLES = [
        "live update", "live blog", "breaking:", "watch:", "photos:", "video:",
        "top 10", "list of", "how to", "explainer:", "fact check",
    ]

    def article_score(sig):
        src = sig.get("source", "").lower()
        title = sig.get("title", "").lower()
        score = sig.get("intensity", 0.5) * get_source_weight(sig.get("source", ""))
        # Boost: regional/independent sources cover state-specific stories better
        if any(ns in src for ns in NATIONAL_SOURCES):
            score *= 0.7  # slight penalty — national might be generic
        # Boost: narratives that match state dominant narrative
        sig_nars = sig.get("narratives", [])
        if top_narratives and sig_nars and top_narratives[0][0] in sig_nars:
            score *= 1.4
        # Penalty: noise titles
        if any(n in title for n in NOISE_TITLES):
            score *= 0.3
        # Penalty: very short titles (likely metadata not news)
        if len(sig.get("title", "")) < 30:
            score *= 0.4
        # Boost: multiple emotion signals = stronger story
        if len(sig.get("emotions", {})) > 1:
            score *= 1.2
        return score

    # Sort by significance score
    ranked_sigs = sorted(sigs_48h, key=article_score, reverse=True)

    articles = []
    seen_t2 = set()
    for s in ranked_sigs:
        src = s.get("source", "unknown")
        title = s.get("title", "")
        if len(title) < 25:
            continue
        # Dedupe by title similarity (first 50 chars), not by source
        tk2 = title[:50].lower().strip()
        if tk2 in seen_t2:
            continue
        seen_t2.add(tk2)
        sig_emos = s.get("emotions", {})
        dom_emo = max(sig_emos.items(), key=lambda x: x[1])[0] if sig_emos else None
        # Determine display source — hide national media, show regional
        src_lower = src.lower()
        is_national = any(ns in src_lower for ns in NATIONAL_SOURCES)
        is_reddit = "reddit" in src_lower
        is_youtube = "youtube" in src_lower
        if is_youtube:
            display_src = ""  # never show YouTube
        elif is_reddit:
            display_src = "public discourse"
        elif is_national:
            display_src = ""  # hide national media name
        else:
            display_src = src.split("/")[0] if "/" in src else src
        articles.append({
            "src": display_src,
            "txt": title,
            "url": s.get("source_url", "#"),
            "emotion": dom_emo,
            "narratives": s.get("narratives", [])[:2],
        })
        if len(articles) >= 8:
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
    store.cache_built_at = datetime.now(timezone.utc)


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



# ── REDDIT SUBREDDIT MAPPING PER STATE ────────────────────────────────
# Reddit's public JSON API - no auth required for read access
STATE_SUBREDDITS: dict[str, list[str]] = {
    "Delhi":             ["delhi", "india", "IndiaSpeaks"],
    "Uttar Pradesh":     ["lucknow", "india", "IndiaSpeaks"],
    "Maharashtra":       ["mumbai", "pune", "nagpur", "india"],
    "Tamil Nadu":        ["TamilNadu", "Chennai", "india"],
    "Karnataka":         ["bangalore", "india", "IndiaSpeaks"],
    "Kerala":            ["Kerala", "india", "Keralites"],
    "West Bengal":       ["kolkata", "WestBengal", "india"],
    "Gujarat":           ["ahmedabad", "gujarat", "india"],
    "Rajasthan":         ["jaipur", "india", "IndiaSpeaks"],
    "Punjab":            ["punjab", "amritsar", "india"],
    "Andhra Pradesh":    ["AndhraPradesh", "hyderabad", "india"],
    "Telangana":         ["hyderabad", "Telangana", "india"],
    "Bihar":             ["bihar", "india", "IndiaSpeaks"],
    "Madhya Pradesh":    ["bhopal", "india", "IndiaSpeaks"],
    "Haryana":           ["india", "IndiaSpeaks"],
    "Odisha":            ["odisha", "india"],
    "Jharkhand":         ["india", "IndiaSpeaks"],
    "Chhattisgarh":      ["india", "IndiaSpeaks"],
    "Assam":             ["Assam", "india"],
    "Jammu and Kashmir": ["Kashmiri", "india", "IndiaSpeaks"],
    "Himachal Pradesh":  ["himachal", "india"],
    "Uttarakhand":       ["india", "IndiaSpeaks"],
    "Goa":               ["goa", "india"],
    "Manipur":           ["Manipur", "india", "NortheastIndia"],
    "Nagaland":          ["Nagaland", "india", "NortheastIndia"],
    "Mizoram":           ["Mizoram", "india", "NortheastIndia"],
    "Meghalaya":         ["Meghalaya", "india", "NortheastIndia"],
    "Tripura":           ["india", "NortheastIndia"],
    "Arunachal Pradesh": ["india", "NortheastIndia"],
    "Sikkim":            ["india", "NortheastIndia"],
}

NATIONAL_SUBREDDITS = ["india", "IndiaSpeaks", "unitedstatesofindia", "IndiaOpen"]

# Expected daily active posts per subreddit (for baseline normalization)
# Prevents large subreddits from dominating smaller regional ones
SUBREDDIT_BASELINES: dict[str, float] = {
    "india": 80.0, "IndiaSpeaks": 40.0, "unitedstatesofindia": 30.0, "IndiaOpen": 20.0,
    "bangalore": 50.0, "mumbai": 45.0, "delhi": 40.0, "pune": 30.0, "hyderabad": 30.0,
    "Chennai": 25.0, "kolkata": 20.0, "ahmedabad": 15.0, "jaipur": 12.0,
    "Kerala": 20.0, "Keralites": 15.0, "TamilNadu": 15.0, "WestBengal": 12.0,
    "AndhraPradesh": 10.0, "Telangana": 10.0, "gujarat": 10.0, "punjab": 10.0,
    "Assam": 8.0, "bihar": 8.0, "Kashmiri": 8.0, "lucknow": 8.0,
    "bhopal": 6.0, "amritsar": 6.0, "nagpur": 6.0, "himachal": 5.0,
    "odisha": 5.0, "goa": 5.0, "NortheastIndia": 5.0,
    "Manipur": 3.0, "Meghalaya": 2.5, "Nagaland": 2.0,
    "Mizoram": 2.0, "Sikkim": 1.5, "Assam": 6.0, "Tripura": 2.0,
}

async def fetch_reddit_posts(subreddit: str, client: httpx.AsyncClient, limit: int = 15) -> list[dict]:
    """Fetch top posts from a subreddit using Reddit's free JSON API."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        r = await client.get(url, headers={"User-Agent": "PulseOfIndia/1.0 signal-observer"})
        if r.status_code != 200:
            return []
        data = r.json()
        posts = []
        # Get baseline for this subreddit to normalize scores
        baseline = SUBREDDIT_BASELINES.get(subreddit, 10.0)
        for post in data.get("data", {}).get("children", []):
            p = post.get("data", {})
            if p.get("stickied") or p.get("score", 0) < 3:
                continue
            score = p.get("score", 0)
            # Normalize score relative to subreddit baseline
            # A post with 100 upvotes in r/Mizoram (baseline=2) is more significant
            # than 100 upvotes in r/bangalore (baseline=50)
            normalized_score = score / max(baseline, 1.0)
            posts.append({
                "title":           p.get("title", ""),
                "text":            p.get("selftext", "")[:300],
                "score":           score,
                "normalized_score": normalized_score,
                "comments":        p.get("num_comments", 0),
                "url":             f"https://reddit.com{p.get('permalink', '')}",
                "subreddit":       subreddit,
                "created":         p.get("created_utc", 0),
                "flair":           p.get("link_flair_text", ""),
            })
        return posts
    except Exception as e:
        print(f"[reddit] r/{subreddit} error: {e}")
        return []

async def ingest_reddit_for_state(state: str, client: httpx.AsyncClient) -> int:
    """Fetch Reddit posts for a state and add as signals."""
    added = 0
    subreddits = STATE_SUBREDDITS.get(state, ["india"])[:3]
    seen_titles: set[str] = set()

    for sub in subreddits:
        posts = await fetch_reddit_posts(sub, client, limit=20)
        for post in posts:
            title = post["title"].strip()
            if not title or len(title) < 10:
                continue

            # For national subreddits, only include if post mentions this state
            if sub in NATIONAL_SUBREDDITS:
                aliases = STATE_ALIASES.get(state, [state.lower()])
                text_lower = (title + " " + post["text"]).lower()
                if not any(a.lower() in text_lower for a in aliases):
                    continue

            tk = title[:55].lower()
            if tk in seen_titles:
                continue
            seen_titles.add(tk)

            text = title + " " + post["text"]
            narratives = classify_narratives(text)
            emotions   = classify_emotion(text)

            # Use normalized score (relative to subreddit baseline)
            # 5 posts in r/Mizoram (baseline=2) > 20 posts in r/bangalore (baseline=50)
            norm = post.get("normalized_score", 1.0)
            intensity = min(1.0, 0.35 + math.log1p(norm) * 0.3)

            try:
                pub = datetime.fromtimestamp(post["created"], tz=timezone.utc)
            except Exception:
                pub = datetime.now(timezone.utc)

            sig = {
                "title":      title,
                "source":     f"reddit/r/{sub}",
                "source_url": post["url"],
                "published_at": pub,
                "narratives": narratives,
                "emotions":   emotions,
                "intensity":  intensity,
                "body":       post["text"][:200],
                "language":   "en",
                "reddit_score": score,
                "reddit_comments": post["comments"],
            }
            if store.add_signal(state, sig):
                added += 1

    return added

async def ingest_reddit_national() -> int:
    """Fetch from national subreddits and geo-tag to relevant states."""
    added = 0
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        for sub in NATIONAL_SUBREDDITS:
            posts = await fetch_reddit_posts(sub, client, limit=25)
            for post in posts:
                title = post["title"].strip()
                if not title:
                    continue
                text = title + " " + post["text"]
                tagged = geotag_states(text)
                if not tagged:
                    continue

                narratives = classify_narratives(text)
                emotions   = classify_emotion(text)
                score      = post["score"]
                norm       = post.get("normalized_score", 1.0)
                intensity  = min(1.0, 0.35 + math.log1p(norm) * 0.3)

                try:
                    pub = datetime.fromtimestamp(post["created"], tz=timezone.utc)
                except Exception:
                    pub = datetime.now(timezone.utc)

                for state in tagged:
                    if state not in INDIAN_STATES:
                        continue
                    sig = {
                        "title": title, "source": f"reddit/r/{sub}",
                        "source_url": post["url"], "published_at": pub,
                        "narratives": narratives, "emotions": emotions,
                        "intensity": intensity, "body": post["text"][:200],
                        "language": "en", "reddit_score": score,
                    }
                    if store.add_signal(state, sig):
                        added += 1
    return added


# ════════════════════════════════════════════════════════════════
# YOUTUBE SIGNAL LAYER
# Free YouTube Data API v3 — 10k units/day quota
# Set YOUTUBE_API_KEY env var on Render
# ════════════════════════════════════════════════════════════════

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# ── CHANNEL TRUST WEIGHTS ──────────────────────────────────────
# Independent journalists and grassroots channels = higher weight
# National corporate media = lower weight
# Regional language channels = high weight (local signal)
CHANNEL_TRUST = {
    "independent":   1.8,   # indie journalists, fact-checkers
    "regional":      1.6,   # state/language-specific channels
    "grassroots":    1.5,   # ground reporters, local YouTubers
    "national_alt":  1.2,   # alternative national (The Wire etc)
    "national":      0.7,   # mainstream national channels
    "entertainment": 0.4,   # political comedy, satire
}

# ── NATIONAL CHANNELS (your list + additions) ──────────────────
NATIONAL_YT_CHANNELS = [
    # Independent — verified channel IDs
    {"id": "UCJqobJNbMB0DcMSvdQkfMzg", "name": "Newslaundry",        "type": "independent"},
    {"id": "UC3M7l8ved_rYQ45AVzS0RGA", "name": "Dhruv Rathee",        "type": "independent"},
    {"id": "UCGkJFMGBqUbNmXwqh0iAeow", "name": "Ravish Kumar",        "type": "independent"},
    {"id": "UCfIJut6tiwYV0sbHN3mFiIg", "name": "The Red Mic",         "type": "independent"},
    {"id": "UC3yY9qBsUTUiP-gBLhDVUoA", "name": "The Wire",            "type": "national_alt"},
    {"id": "UCBcRF18a7Qf58cCRy5xuWwQ", "name": "Newslaundry Hindi",   "type": "independent"},
    # National — verified
    {"id": "UCITjpxCxbDTFG2bOF08KOSA", "name": "NDTV",                "type": "national"},
    {"id": "UCt4t-jeY85JegMlZ-E5UWuQ", "name": "Aaj Tak",             "type": "national"},
    {"id": "UCYPvAwZP8pZhSMW8qs7cVCw", "name": "ABP News",            "type": "national"},
    {"id": "UCZFMm1mMw0F81Z37aaEzTUA", "name": "Zee News",            "type": "national"},
    {"id": "UCoHRYZ5cOvWTwRnPpWYB4Ng", "name": "India Today",         "type": "national"},
    {"id": "UCkKoYBnQHRMBGNI_o96T2dA", "name": "Republic TV",         "type": "national"},
    {"id": "UCqW4LFb93gLIg3StCTUhGEA", "name": "News18 India",        "type": "national"},
    {"id": "UCHqGVpJB_NKkPdJZlSBHBXQ", "name": "News Pinch",          "type": "independent"},
]

# ── STATE-SPECIFIC CHANNELS ────────────────────────────────────
STATE_YT_CHANNELS: dict[str, list[dict]] = {
    "Tamil Nadu": [
        {"id": "UCq4MIOdKWDEkuFBgKzHo6Ag", "name": "Thanthi TV",         "type": "regional"},
        {"id": "UCuuBMEMOoGGFaVKLp3Fa5bA", "name": "Polimer News",        "type": "regional"},
        {"id": "UCmfjoyV5BUBsQ9XSYC8ZRNA", "name": "Sun News",            "type": "regional"},
        {"id": "UCqV4m64mR0I0lCm6pgKm8tg", "name": "Puthiya Thalaimurai", "type": "regional"},
        {"id": "UCsBjURrPoezykLs9EqgamOA", "name": "ABP Nadu",            "type": "regional"},
    ],
    "Kerala": [
        {"id": "UCkFBr7BtHi5RQSV7uqFiHhQ", "name": "Manorama News",      "type": "regional"},
        {"id": "UCQmFbmAKGIHMBMGH5_26N2g", "name": "Asianet News",        "type": "regional"},
        {"id": "UCeiHRkFwfEGGKvblbLHHHLA", "name": "MediaOne",            "type": "regional"},
        {"id": "UCsTnQ1uAM9LkzZJHxOAfKtg", "name": "Mathrubhumi News",    "type": "regional"},
        {"id": "UC6tQvUiTK8zSb9UPWIh7bVA", "name": "Kairali TV",          "type": "regional"},
    ],
    "Karnataka": [
        {"id": "UCsMZFMTVz0rcXVJVMBgMdvg", "name": "TV9 Kannada",         "type": "regional"},
        {"id": "UC1PZFe3MEr9_7OsalJYGxhg", "name": "Suvarna News",        "type": "regional"},
        {"id": "UCkbPjMx0IHAs1_iVC2qKhcA", "name": "Public TV Kannada",   "type": "regional"},
        {"id": "UC3JwMmJMGWKPIAf67FsQ5Sg", "name": "News18 Kannada",      "type": "regional"},
    ],
    "Andhra Pradesh": [
        {"id": "UCbGa9MVQVPy7Y55Tr8hIMzg", "name": "TV9 Telugu",          "type": "regional"},
        {"id": "UCfygKMY9oJ0e49r0aKGzb0Q", "name": "ABN Andhra Jyothy",   "type": "regional"},
        {"id": "UCCdLJhUDmiMHwbAjEjnFp7Q", "name": "Sakshi TV",           "type": "regional"},
        {"id": "UCaIBD1fxbOoH5NVLJGhAlBg", "name": "NTV Telugu",          "type": "regional"},
    ],
    "Telangana": [
        {"id": "UCbGa9MVQVPy7Y55Tr8hIMzg", "name": "TV9 Telugu",          "type": "regional"},
        {"id": "UCe11HL_VBXZF0qQmK3Grfhg", "name": "V6 News",             "type": "regional"},
        {"id": "UCvQGKqd1K0kRwUKSFaegSQg", "name": "T News",              "type": "regional"},
    ],
    "Maharashtra": [
        {"id": "UCsBjURrPoezykLs9EqgamOA", "name": "ABP Majha",           "type": "regional"},
        {"id": "UCLfGnuTVqkGf_aLdBz7Qd7w", "name": "TV9 Marathi",         "type": "regional"},
        {"id": "UC7RXQFWU5R90fLvIpWCpg_A", "name": "Zee 24 Taas",         "type": "regional"},
        {"id": "UCItpmPXfElCFREvqBjaTNFg", "name": "News18 Lokmat",       "type": "regional"},
    ],
    "Gujarat": [
        {"id": "UCMQu1CKtsbhQFnCpYKnr7DA", "name": "Sandesh News",        "type": "regional"},
        {"id": "UC6mcOKNQzB2HrMDSIFVnBrg", "name": "VTV Gujarati",        "type": "regional"},
        {"id": "UCJ1mRNH9HjpzNPHMkMVa_9w", "name": "ABP Asmita",          "type": "regional"},
        {"id": "UCNbBFyZJqGMiOSvYGM8LAOQ", "name": "News18 Gujarat",      "type": "regional"},
    ],
    "Rajasthan": [
        {"id": "UCUMEQq7HwmkbNijZBJC5DqQ", "name": "ETV Rajasthan",       "type": "regional"},
        {"id": "UCFnbsUCJNAe5LT6jv-kijTQ", "name": "News18 Rajasthan",    "type": "regional"},
        {"id": "UCt6E2ooCm-iCERQUX3oKVEg", "name": "First India News",    "type": "regional"},
    ],
    "Uttar Pradesh": [
        {"id": "UC9rFWQDE_pEFk_xh-n7YCBQ", "name": "ABP Ganga",           "type": "regional"},
        {"id": "UCuLYM2K70M0gVBxIHyPFiXQ", "name": "Aaj Tak UP",          "type": "regional"},
        {"id": "UC6T5bxriFsqHEjBFhypWMfQ", "name": "News18 UP",           "type": "regional"},
    ],
    "Bihar": [
        {"id": "UCBwECRXNhRjt2V7WxWXYeYg", "name": "ETV Bihar",           "type": "regional"},
        {"id": "UCTRdAzFCnTVE5RtKiE9YZRA", "name": "News18 Bihar",        "type": "regional"},
        {"id": "UC9RkgwWFPOXZb7V0W7BVPwA", "name": "Mahua News",          "type": "regional"},
    ],
    "West Bengal": [
        {"id": "UC_1v_3fqxGNKW72vHdQRlJg", "name": "ABP Ananda",          "type": "regional"},
        {"id": "UCsdB3XM7S9YIJG9Z0bHUzfg", "name": "Zee 24 Ghanta",       "type": "regional"},
        {"id": "UCjfOPnFGhHM28aQZ41Kv3Mg", "name": "News18 Bangla",       "type": "regional"},
        {"id": "UCtbxlj9SJSQjJpjzPHg_PFQ", "name": "TV9 Bangla",          "type": "regional"},
    ],
    "Punjab": [
        {"id": "UCnHoMNrPKMI4hMxUbLfHGEg", "name": "PTC Punjab",          "type": "regional"},
        {"id": "UCyV_qc6s0OoMr3ZQdPwW_UA", "name": "News18 Punjab",       "type": "regional"},
        {"id": "UCz7Dqm4zUB0Bkl-Xdk3TfFg", "name": "ABP Sanjha",          "type": "regional"},
    ],
    "Haryana": [
        {"id": "UCFsHv_Kz7mHDhY-D6YENQYQ", "name": "News18 Haryana",      "type": "regional"},
        {"id": "UC0Gz_Vl5BFqBJFaqRGNGwOw", "name": "Haryana TV",          "type": "regional"},
    ],
    "Madhya Pradesh": [
        {"id": "UCHpEqHxelHiCxoMVRVtQkFg", "name": "Bansal News",         "type": "regional"},
        {"id": "UC6QVlUb0ZQMBE8mujqT6x9g", "name": "News18 MP",           "type": "regional"},
        {"id": "UCuNSPkm2KqkJuH7Iy6FRRgw", "name": "IBC24",               "type": "regional"},
    ],
    "Odisha": [
        {"id": "UCeUdWRPzBDq1DPu6ViIRNKg", "name": "OTV",                 "type": "regional"},
        {"id": "UC6tQvUiTK8zSb9UPWIh7bVA", "name": "Kanak News",          "type": "regional"},
    ],
    "Assam": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCNAzh1xJ0R3_NVH5BpOkNNg", "name": "DY365",               "type": "regional"},
        {"id": "UCh7mF2ql3hWJEPgB8c7YA-g", "name": "Pratidin Time",       "type": "regional"},
    ],
    "Manipur": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCNAzh1xJ0R3_NVH5BpOkNNg", "name": "DY365",               "type": "regional"},
        {"id": "UCt4t-jeY85JegMlZ-E5UWuQ", "name": "Aaj Tak",             "type": "national"},
    ],
    "Jammu and Kashmir": [
        {"id": "UCxHFGBSjBvHiGRyL5p82kLA", "name": "DD Kashir",           "type": "regional"},
        {"id": "UCkITAUCHKHBGMeGhPJlBERA", "name": "Kashmir Crown",       "type": "grassroots"},
    ],
    "Delhi": [
        {"id": "UCJqobJNbMB0DcMSvdQkfMzg", "name": "Newslaundry",         "type": "independent"},
        {"id": "UC3M7l8ved_rYQ45AVzS0RGA", "name": "Dhruv Rathee",         "type": "independent"},
        {"id": "UCt4t-jeY85JegMlZ-E5UWuQ", "name": "Aaj Tak",             "type": "national"},
        {"id": "UCGkJFMGBqUbNmXwqh0iAeow", "name": "Ravish Kumar",         "type": "independent"},
    ],
    "Himachal Pradesh": [
        {"id": "UCHFfmLPvNe5LjJhV-h8j2Nw", "name": "Himachal Abhi Abhi",  "type": "regional"},
        {"id": "UCFnbsUCJNAe5LT6jv-kijTQ", "name": "News18 Rajasthan",    "type": "regional"},
    ],
    "Uttarakhand": [
        {"id": "UCGylVDFBkCIBPHhMBe4HrIQ", "name": "Uttarakhand Tak",     "type": "regional"},
        {"id": "UC6T5bxriFsqHEjBFhypWMfQ", "name": "News18 UP",           "type": "regional"},
    ],
    "Goa": [
        {"id": "UClmBEAp7sPnlCWvFCG0hFEA", "name": "Goa365",              "type": "regional"},
        {"id": "UCsBjURrPoezykLs9EqgamOA", "name": "ABP Majha",           "type": "regional"},
    ],
    "Nagaland": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCNAzh1xJ0R3_NVH5BpOkNNg", "name": "DY365",               "type": "regional"},
    ],
    "Meghalaya": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCh7mF2ql3hWJEPgB8c7YA-g", "name": "Pratidin Time",       "type": "regional"},
    ],
    "Mizoram": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCNAzh1xJ0R3_NVH5BpOkNNg", "name": "DY365",               "type": "regional"},
    ],
    "Tripura": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
        {"id": "UCh7mF2ql3hWJEPgB8c7YA-g", "name": "Pratidin Time",       "type": "regional"},
    ],
    "Arunachal Pradesh": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
    ],
    "Sikkim": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",     "type": "regional"},
    ],
    "Chhattisgarh": [
        {"id": "UCuNSPkm2KqkJuH7Iy6FRRgw", "name": "IBC24",               "type": "regional"},
        {"id": "UC6QVlUb0ZQMBE8mujqT6x9g", "name": "News18 MP",           "type": "regional"},
    ],
    "Jharkhand": [
        {"id": "UCBwECRXNhRjt2V7WxWXYeYg", "name": "ETV Bihar",           "type": "regional"},
        {"id": "UCTRdAzFCnTVE5RtKiE9YZRA", "name": "News18 Bihar",        "type": "regional"},
    ],
}

# Expected subscriber baseline per channel type (for normalization)
CHANNEL_TYPE_BASELINE: dict[str, float] = {
    "independent":   500_000,
    "regional":    1_000_000,
    "grassroots":    100_000,
    "national_alt":  800_000,
    "national":    5_000_000,
    "entertainment": 300_000,
}

async def fetch_youtube_search(query: str, client: httpx.AsyncClient, max_results: int = 8) -> list[dict]:
    """Search YouTube for recent videos matching a query — more reliable than channel ID lookup."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        import urllib.parse
        encoded = urllib.parse.quote(query)
        # publishedAfter = 3 days ago for fresh results
        published_after = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"https://www.googleapis.com/youtube/v3/search"
            f"?key={YOUTUBE_API_KEY}&q={encoded}"
            f"&part=snippet&type=video&order=relevance"
            f"&relevanceLanguage=hi&regionCode=IN"
            f"&publishedAfter={published_after}"
            f"&maxResults={max_results}"
        )
        r = await client.get(url)
        if r.status_code != 200:
            print(f"[youtube] Search error {r.status_code}: {r.text[:100]}")
            return []
        data = r.json()
        videos = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            published = snippet.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                pub_dt = datetime.now(timezone.utc)
            videos.append({
                "video_id":     video_id,
                "title":        snippet.get("title", ""),
                "description":  snippet.get("description", "")[:200],
                "published_at": pub_dt,
                "channel_title": snippet.get("channelTitle", ""),
            })
        return videos
    except Exception as e:
        print(f"[youtube] Search error for '{query}': {e}")
        return []

async def fetch_youtube_channel_videos(channel_id: str, client: httpx.AsyncClient, max_results: int = 8) -> list[dict]:
    """Fetch recent videos from a YouTube channel by ID."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        published_after = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"https://www.googleapis.com/youtube/v3/search"
            f"?key={YOUTUBE_API_KEY}&channelId={channel_id}"
            f"&part=snippet&order=date&type=video"
            f"&publishedAfter={published_after}"
            f"&maxResults={max_results}"
        )
        r = await client.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        videos = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            published = snippet.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                pub_dt = datetime.now(timezone.utc)
            videos.append({
                "video_id":     video_id,
                "title":        snippet.get("title", ""),
                "description":  snippet.get("description", "")[:200],
                "published_at": pub_dt,
                "channel_title": snippet.get("channelTitle", ""),
            })
        return videos
    except Exception as e:
        print(f"[youtube] Channel {channel_id} error: {e}")
        return []

async def fetch_youtube_video_stats(video_ids: list[str], client: httpx.AsyncClient) -> dict:
    """Fetch view/like counts for videos."""
    if not YOUTUBE_API_KEY or not video_ids:
        return {}
    try:
        ids_str = ",".join(video_ids[:50])
        url = (
            f"https://www.googleapis.com/youtube/v3/videos"
            f"?key={YOUTUBE_API_KEY}&id={ids_str}&part=statistics"
        )
        r = await client.get(url)
        if r.status_code != 200:
            return {}
        data = r.json()
        stats = {}
        for item in data.get("items", []):
            vid_id = item["id"]
            s = item.get("statistics", {})
            stats[vid_id] = {
                "views":    int(s.get("viewCount", 0)),
                "likes":    int(s.get("likeCount", 0)),
                "comments": int(s.get("commentCount", 0)),
            }
        return stats
    except Exception as e:
        print(f"[youtube] Stats error: {e}")
        return {}

async def ingest_youtube_for_state(state: str, client: httpx.AsyncClient) -> int:
    """Ingest YouTube signals for a state — uses keyword search for reliability."""
    if not YOUTUBE_API_KEY:
        return 0

    added = 0
    all_videos = []
    # Default channel type for keyword-search results
    video_to_channel: dict[str, dict] = {}

    # Use STATE_QUERIES for keyword searches — guaranteed fresh results
    queries = STATE_QUERIES.get(state, [f"{state} news", f"{state} politics"])
    for q in queries[:2]:
        videos = await fetch_youtube_search(q, client, max_results=6)
        for v in videos:
            all_videos.append(v)
            video_to_channel[v["video_id"]] = {"name": v["channel_title"], "type": "regional"}

    # Also try channel-based for states with known good channels
    channels = STATE_YT_CHANNELS.get(state, [])
    for ch in channels[:2]:
        videos = await fetch_youtube_channel_videos(ch["id"], client, max_results=4)
        for v in videos:
            all_videos.append(v)
            video_to_channel[v["video_id"]] = ch

    if not all_videos:
        return 0

    # Batch fetch stats
    video_ids = [v["video_id"] for v in all_videos if v["video_id"]]
    stats = await fetch_youtube_video_stats(video_ids, client)

    seen_titles: set[str] = set()
    for video in all_videos:
        title = video["title"].strip()
        if not title or len(title) < 8:
            continue
        tk = title[:55].lower()
        if tk in seen_titles:
            continue
        seen_titles.add(tk)

        ch = video_to_channel.get(video["video_id"], {})
        ch_type = ch.get("type", "regional")
        trust_weight = CHANNEL_TRUST.get(ch_type, 1.0)

        # Get engagement stats
        s = stats.get(video["video_id"], {})
        views    = s.get("views", 0)
        likes    = s.get("likes", 0)
        comments = s.get("comments", 0)

        # Normalize by channel type baseline (same logic as Reddit)
        baseline = CHANNEL_TYPE_BASELINE.get(ch_type, 500_000)
        engagement = (views + likes * 5 + comments * 10)
        normalized_engagement = engagement / max(baseline / 100, 1)
        intensity = min(1.0, 0.3 + math.log1p(normalized_engagement) * 0.2) * trust_weight
        intensity = min(1.0, intensity)

        text = title + " " + video["description"]
        narratives = classify_narratives(text)
        emotions   = classify_emotion(text)

        sig = {
            "title":       title,
            "source":      f"youtube/{ch.get('name', 'unknown')}",
            "source_url":  f"https://youtube.com/watch?v={video['video_id']}",
            "published_at": video["published_at"],
            "narratives":  narratives,
            "emotions":    emotions,
            "intensity":   intensity,
            "body":        video["description"],
            "language":    "en",
            "yt_views":    views,
            "yt_channel_type": ch_type,
        }
        if store.add_signal(state, sig):
            added += 1

    return added

async def ingest_youtube_national(client: httpx.AsyncClient) -> int:
    """Ingest national YouTube via keyword search and channel IDs."""
    if not YOUTUBE_API_KEY:
        return 0
    added = 0
    # Keyword searches for national political topics
    national_queries = [
        "India politics news today",
        "BJP Congress Modi news",
        "India Parliament news",
        "India state election news",
    ]
    for q in national_queries:
        videos = await fetch_youtube_search(q, client, max_results=8)
        if not videos:
            continue
        video_ids = [v["video_id"] for v in videos if v["video_id"]]
        stats = await fetch_youtube_video_stats(video_ids, client)
        for video in videos:
            title = video["title"].strip()
            if not title:
                continue
            text = title + " " + video["description"]
            tagged = geotag_states(text)
            if not tagged:
                continue
            s_val = stats.get(video["video_id"], {})
            views = s_val.get("views", 0)
            normalized = (views + s_val.get("comments",0)*10) / 10000
            intensity = min(1.0, 0.3 + math.log1p(normalized)*0.2)
            narratives = classify_narratives(text)
            emotions = classify_emotion(text)
            for state in tagged:
                if state not in INDIAN_STATES:
                    continue
                sig = {
                    "title": title,
                    "source": f"youtube/{video['channel_title'] or 'news'}",
                    "source_url": f"https://youtube.com/watch?v={video['video_id']}",
                    "published_at": video["published_at"],
                    "narratives": narratives, "emotions": emotions,
                    "intensity": intensity, "body": video["description"],
                    "language": "en", "yt_channel_type": "national",
                }
                if store.add_signal(state, sig):
                    added += 1
        await asyncio.sleep(0.5)

    # Also try channel IDs
    for ch in NATIONAL_YT_CHANNELS:
        videos = await fetch_youtube_channel_videos(ch["id"], client, max_results=8)
        if not videos:
            continue
        video_ids = [v["video_id"] for v in videos if v["video_id"]]
        stats = await fetch_youtube_video_stats(video_ids, client)
        ch_type = ch.get("type", "national")
        trust_weight = CHANNEL_TRUST.get(ch_type, 0.7)

        for video in videos:
            title = video["title"].strip()
            if not title:
                continue
            text = title + " " + video["description"]
            tagged = geotag_states(text)
            if not tagged:
                continue

            s = stats.get(video["video_id"], {})
            views = s.get("views", 0)
            baseline = CHANNEL_TYPE_BASELINE.get(ch_type, 1_000_000)
            normalized = (views + s.get("comments",0)*10) / max(baseline/100, 1)
            intensity = min(1.0, 0.3 + math.log1p(normalized)*0.2) * trust_weight
            intensity = min(1.0, intensity)

            narratives = classify_narratives(text)
            emotions   = classify_emotion(text)

            for state in tagged:
                if state not in INDIAN_STATES:
                    continue
                sig = {
                    "title":       title,
                    "source":      f"youtube/{ch['name']}",
                    "source_url":  f"https://youtube.com/watch?v={video['video_id']}",
                    "published_at": video["published_at"],
                    "narratives":  narratives,
                    "emotions":    emotions,
                    "intensity":   intensity,
                    "body":        video["description"],
                    "language":    "en",
                    "yt_channel_type": ch_type,
                }
                if store.add_signal(state, sig):
                    added += 1
    return added

async def historical_backfill():
    """
    On first startup: fetch signals from past 7 days and populate DB.
    Uses date-ranged Google News queries per state.
    Runs only if DB has less than 100 historical signals.
    """
    conn = await get_db()
    if not conn:
        print("[backfill] No DB — skipping historical backfill")
        return
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM signal_store")
        if count > 100:
            print(f"[backfill] DB already has {count} signals — skipping backfill")
            return
        print(f"[backfill] DB has {count} signals — starting 7-day backfill...")
    except Exception as e:
        print(f"[backfill] Error checking DB: {e}")
        return
    finally:
        await conn.close()

    # Priority states for backfill (highest signal value)
    priority_states = [
        "Uttar Pradesh","Maharashtra","Delhi","West Bengal","Tamil Nadu",
        "Karnataka","Bihar","Gujarat","Rajasthan","Kerala","Telangana",
        "Andhra Pradesh","Punjab","Madhya Pradesh","Haryana","Assam",
        "Jharkhand","Odisha","Jammu and Kashmir","Manipur","Chhattisgarh",
        "Uttarakhand","Himachal Pradesh","Goa","Tripura","Meghalaya",
        "Nagaland","Mizoram","Arunachal Pradesh","Sikkim",
    ]

    total_added = 0
    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
        for state in priority_states:
            queries = STATE_QUERIES.get(state, [
                f"{state} politics government news",
                f"{state} latest news",
            ])
            for q in queries[:2]:
                url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
                try:
                    r = await client.get(url)
                    feed = feedparser.parse(r.text)
                    for entry in feed.entries[:20]:
                        title = entry.get("title","").strip()
                        if not title:
                            continue
                        try:
                            pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        except Exception:
                            pub = datetime.now(timezone.utc) - timedelta(days=3)

                        # Only take last 7 days
                        if (datetime.now(timezone.utc) - pub).days > 7:
                            continue

                        body = entry.get("summary","")
                        text = f"{title} {body}"
                        narratives = classify_narratives(text)
                        emotions = classify_emotion(text)
                        intensity = min(1.0, 0.3 + len(narratives) * 0.15)

                        sig = {
                            "title": title, "source": entry.get("source",{}).get("title","google_news"),
                            "source_url": entry.get("link",""), "published_at": pub,
                            "narratives": narratives, "emotions": emotions,
                            "intensity": intensity, "body": body[:200], "language": "en",
                        }
                        store.add_signal(state, sig)
                        total_added += 1
                except Exception as e:
                    print(f"[backfill] {state}: {e}")
                await asyncio.sleep(0.3)  # polite delay

    print(f"[backfill] Complete — added {total_added} historical signals across {len(priority_states)} states")

    # Now recompute all scores with the backfilled data
    await recompute_all_scores()
    # Save to DB
    await save_signals_to_db_bulk()
    print("[backfill] Historical data saved to DB")

async def save_signals_to_db_bulk():
    """Save all current RAM signals to DB."""
    for state, signals in store.signals.items():
        await save_signals_to_db(state, signals)


async def self_ping():
    """Ping own health endpoint every 10 min to keep Render free tier alive."""
    app_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not app_url:
        print("[ping] RENDER_EXTERNAL_URL not set — self-ping disabled")
        return
    await asyncio.sleep(60)  # wait for server to be ready
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(f"{app_url}/api/health")
            print("[ping] Self-ping OK")
        except Exception as e:
            print(f"[ping] Self-ping failed: {e}")
        await asyncio.sleep(600)  # every 10 minutes


async def continuous_ingest():
    """Run full ingest on startup then every 15 minutes."""
    while True:
        try:
            print(f"[continuous] Starting full ingest cycle")
            await run_ingest(states=INDIAN_STATES)
            # Persist to DB after each cycle
            if HAS_PG and DB_URL:
                for state, score in store.scores.items():
                    if isinstance(score, dict) and score.get("signal_count", 0) > 0:
                        await save_daily_snapshot(state, score)
                await save_signals_to_db_bulk()
                print("[continuous] Saved to DB")
                await save_national_narrative_snapshot()
        except Exception as e:
            print(f"[continuous] Error: {e}")
        await asyncio.sleep(900)  # 15 minutes

@app.on_event("startup")
async def startup():
    print("[startup] Pulse of India backend starting...")
    # 1. Init DB tables
    await init_db()
    # 2. Load last 48h of signals from DB into RAM (instant on restart)
    if HAS_PG and DB_URL:
        loaded = await load_signals_from_db()
        if loaded > 0:
            await recompute_all_scores()
            print(f"[startup] Restored {loaded} signals from DB — scores recomputed")
    # 3. Start continuous ingest (also triggers backfill on first run)
    asyncio.create_task(continuous_ingest())
    asyncio.create_task(startup_backfill())
    asyncio.create_task(self_ping())

async def startup_backfill():
    """Run backfill after a short delay to not block startup."""
    await asyncio.sleep(5)
    await historical_backfill()


# ── Serve frontend ──────────────────────────────────────────────────────────

# HTML served via base64 decode — avoids all string escaping issues
import base64 as _b64
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQouc2lnLWhvdmVyLXdyYXB7cG9zaXRpb246cmVsYXRpdmU7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtjdXJzb3I6ZGVmYXVsdH0KLnNpZy1ob3Zlci10aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7dG9wOmNhbGMoMTAwJSArIDEwcHgpO3JpZ2h0OjA7CiAgYmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45Nyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxNHB4O3doaXRlLXNwYWNlOm5vd3JhcDsKICBwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt2aXNpYmlsaXR5OmhpZGRlbjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xOHMsdmlzaWJpbGl0eSAwLjE4czsKICB6LWluZGV4Ojk5OTk7Cn0KLnNpZy1ob3Zlci13cmFwOmhvdmVyIC5zaWctaG92ZXItdGlwe29wYWNpdHk6MTt2aXNpYmlsaXR5OnZpc2libGV9Ci5zaWctaG92ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NXB4O29wYWNpdHk6MC43fQouc2lnLWhvdmVyLXNvdXJjZXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA0ZW19Ci5saXZlLWluZGljYXRvcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo3cHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtOwp9Ci5saXZlLWRvdHt3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiM0YWRlODA7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoNzQsMjIyLDEyOCwwLjcpO2FuaW1hdGlvbjpsZCAyLjVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGxkezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjM1O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KLmNsb2Nre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNGVtfQoKLyogSEVSTyAqLwouaGVyb3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgcGFkZGluZzo3MnB4IDM2cHggMDsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87Cn0KLmhlcm8tZXllYnJvd3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMzJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MjRweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4fQouaGVyby1leWVicm93OjpiZWZvcmV7Y29udGVudDonJzt3aWR0aDoxNnB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1mYWludCk7b3BhY2l0eTowLjV9Ci5oZXJvLWJyYW5kLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOm5vcm1hbDtmb250LXNpemU6Y2xhbXAoMzZweCw0LjJ2dyw2NHB4KTtsaW5lLWhlaWdodDoxO2xldHRlci1zcGFjaW5nOi0wLjAzZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW46MH0KLmhlcm8tYnJhbmQtbmFtZSBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjojZThjNGEwO2FuaW1hdGlvbjpwdWxzZU5hbWVHbG93IDVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlTmFtZUdsb3d7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC43Mn19Ci5oZXJvLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNXB4LDEuNXZ3LDIwcHgpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbjowIDAgMTJweCAwO21heC13aWR0aDo0ODBweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLWRlc2N7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEzcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjY7bWF4LXdpZHRoOjQwMHB4O21hcmdpbjowIDAgNnB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1zdWItbGluZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNik7bWFyZ2luOjAgMCAyMHB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1wdWxzZS1zaWduYWx7cG9zaXRpb246cmVsYXRpdmU7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtmbGV4LXNocmluazowfQouaHBzLWNvcmV7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuOTthbmltYXRpb246aHBzQ29yZSA0cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBocHNDb3JlezAlLDEwMCV7b3BhY2l0eTowLjk7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNDt0cmFuc2Zvcm06c2NhbGUoMC43NSl9fQouaHBzLXJpbmd7cG9zaXRpb246YWJzb2x1dGU7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1hY2NlbnQpO2FuaW1hdGlvbjpocHNSaW5nIDRzIGVhc2Utb3V0IGluZmluaXRlfQouaHBzLXJpbmcucjF7aW5zZXQ6MXB4O2FuaW1hdGlvbi1kZWxheTowc30uaHBzLXJpbmcucjJ7aW5zZXQ6LTNweDthbmltYXRpb24tZGVsYXk6MS40cztib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zNSl9CkBrZXlmcmFtZXMgaHBzUmluZ3swJXtvcGFjaXR5OjAuNjt0cmFuc2Zvcm06c2NhbGUoMC43KX0xMDAle29wYWNpdHk6MDt0cmFuc2Zvcm06c2NhbGUoMS42KX19CgovKiBTSUdOQVRVUkUgSU5TSUdIVCAqLwouc2lnbmF0dXJlLWluc2lnaHR7CiAgbWFyZ2luLXRvcDowOwogIHBhZGRpbmc6MTRweCAyMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNikgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIG1heC13aWR0aDo5MDBweDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47Cn0KLnNpZ25hdHVyZS1pbnNpZ2h0OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgdmFyKC0tYWNjZW50KSwgdHJhbnNwYXJlbnQpOwp9Ci5zaS1sYWJlbHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouc2ktdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE0cHgsMS40dncsMThweCk7CiAgZm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluayk7bGluZS1oZWlnaHQ6MS41O2xldHRlci1zcGFjaW5nOi0wLjAxZW07Cn0KLnNpLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tYWNjZW50KX0KLnNpLXN1YnsKICBtYXJnaW4tdG9wOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTsKICBsZXR0ZXItc3BhY2luZzowLjA0ZW07ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDtmbGV4LXdyYXA6d3JhcDsKfQouc2ktdGFnewogIHBhZGRpbmc6MnB4IDhweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTsKICBmb250LXNpemU6OS41cHg7Cn0KCi8qIE5BUlJBVElWRSBTVFJJUCAqLwoKLnN0cmlwLXRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjRweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQouc3RyaXAtdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xMil9Ci5zdHJpcC10YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbHsKICBmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1iZzEpO3BhZGRpbmc6MDsKfQouc3RyaXAtY29sLWhlYWR7CiAgcGFkZGluZzoxMHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwp9Ci5zdHJpcC1jb2wtaGVhZC5mYWRle2NvbG9yOnZhcigtLWZhbGwpfQouc3RyaXAtY29sLWhlYWQucmlzZTJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5zdHJpcC1jb2wtaGVhZC5zaGlmdHtjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sLWJvZHl7cGFkZGluZzoxMnB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OHB4fQouc3RyaXAtaXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjhweDsKfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5zdHJpcC1ub3Rle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5zdHJpcC1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTtmb250LXNpemU6MTRweDtmbGV4LXNocmluazowfQoKLyogTUFJTiBMQVlPVVQgKi8KLm1haW57CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM2MHB4OwogIGdhcDoxNHB4OwogIG1pbi13aWR0aDowOwp9CgovKiBNQVAgKi8KLm1hcC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBvdmVyZmxvdzpoaWRkZW47cG9zaXRpb246cmVsYXRpdmU7Cn0KLm1hcC1jYXJkOjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA3MCUgNTAlIGF0IDM1JSAwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNSkgMCUsIHRyYW5zcGFyZW50IDYwJSksCiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA1MCUgNDAlIGF0IDgwJSAxMDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMCUsIHRyYW5zcGFyZW50IDYwJSk7Cn0KLm1hcC10b3B7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47CiAgcGFkZGluZzoxMnB4IDE4cHggMDsKfQoubWFwLXRpdGxlLWJsb2NrIC5tdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE3cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtdGl0bGUtYmxvY2sgLm1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5sZWdlbmQtYmFyewogIGhlaWdodDozcHg7d2lkdGg6ODBweDtib3JkZXItcmFkaXVzOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCwjMGUyMDM1LCMxYTU1ODAgMjUlLCM4YTVjMTggNTUlLCNjMDM4MWEgODAlLCNlMDEwMjApOwp9Ci5sYXllci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsKICBwYWRkaW5nOjEwcHggMjBweCA2cHg7Cn0KLmxheWVyLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLmx0YWJze2Rpc3BsYXk6ZmxleDtnYXA6M3B4fQoubHRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6M3B4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5sdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wOCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5sdGFie2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo1cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6dmlzaWJsZX0KLmx0YWItaW5mb3t3aWR0aDoxM3B4O2hlaWdodDoxM3B4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjIpO2ZvbnQtc2l6ZTo4cHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zdHlsZTppdGFsaWM7Zm9udC13ZWlnaHQ6NjAwO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4zNSk7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjdXJzb3I6aGVscDtmbGV4LXNocmluazowO3RyYW5zaXRpb246YWxsIDAuMTVzO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTAwfQoubHRhYi1pbmZvOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2NlbnQpO2NvbG9yOnZhcigtLWFjY2VudCl9CiNsdGFiLXRvb2x0aXB7cG9zaXRpb246Zml4ZWQ7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMTIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjt3aWR0aDoyMzBweDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNik7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzO3otaW5kZXg6OTk5OTk7ZGlzcGxheTpub25lfQojbHRhYi10b29sdGlwLnZpc2libGV7b3BhY2l0eToxO2Rpc3BsYXk6YmxvY2t9Ci5sdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9CgoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmc6MTJweCAxNnB4IDE2cHg7Cn0KLm1hcC1pbm5lcntwb3NpdGlvbjpyZWxhdGl2ZTthc3BlY3QtcmF0aW86MS8xO3dpZHRoOjEwMCV9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrO292ZXJmbG93OnZpc2libGV9CgovKiBtYXAgc3RhdGUgc3R5bGVzICovCiNpbmRpYS1tYXAgLnN0YXRlewogIGN1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246ZmlsdGVyIDAuMjVzIGVhc2UsIHN0cm9rZS13aWR0aCAwLjJzIGVhc2UsIHN0cm9rZSAwLjJzIGVhc2U7Cn0KI2luZGlhLW1hcCAuc3RhdGU6aG92ZXJ7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC43KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxcHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjI1KSBkcm9wLXNoYWRvdygwIDAgMTBweCByZ2JhKDI1NSwyNTUsMjU1LDAuMikpOwp9CiNpbmRpYS1tYXAgLnN0YXRlLnNlbGVjdGVkewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuOSkgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MS40cHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjM1KSBkcm9wLXNoYWRvdygwIDAgMTZweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpOwp9CgovKiBhbmltYXRlZCBwdWxzZSByaW5ncyAqLwoucHVsc2UtcmluZ3tmaWxsOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZX0KLnB1bHNlLXJpbmcucDF7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5wdWxzZS1yaW5nLnAye2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IDAuOXMgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHJ7CiAgMCV7cjo0O29wYWNpdHk6MC43O3N0cm9rZS13aWR0aDoxLjJ9CiAgMTAwJXtyOjI2O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4yfQp9CgovKiBhdG1vc3BoZXJpYyBnbG93IGJlaGluZCBob3Qgc3RhdGVzICovCi5zdGF0ZS1nbG93e3BvaW50ZXItZXZlbnRzOm5vbmU7ZmlsbDpub25lfQpAa2V5ZnJhbWVzIGdsb3dQdWxzZXswJSwxMDAle29wYWNpdHk6MC4xMn01MCV7b3BhY2l0eTowLjIyfX0KCi5tYXAtdG9vbHRpcHsKICBwb3NpdGlvbjphYnNvbHV0ZTtwb2ludGVyLWV2ZW50czpub25lOwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC45NSk7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjlweDsKICBwYWRkaW5nOjEycHggMTRweDtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTJzO3otaW5kZXg6OTk5OTttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmcjtnYXA6MThweDsKfQoubmFyLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE0cHgpO292ZXJmbG93OmhpZGRlbjsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOwp9Ci5uYy1oZWFkewogIHBhZGRpbmc6MTZweCAyMHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsKfQoubmMtYm9keXtwYWRkaW5nOjhweCAyMHB4IDE2cHg7ZmxleDoxO292ZXJmbG93LXk6YXV0bzt9Ci5uYy10aXRsZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKX0KLm5jLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDVlbTttYXJnaW4tdG9wOjJweH0KLm5jLWJvZHl7cGFkZGluZzoxM3B4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MH0KCi5tb20taXR7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4OwogIHBhZGRpbmc6N3B4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubW9tLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tb20tcmt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7d2lkdGg6MTNweDtmbGV4LXNocmluazowfQoubW9tLWluZntmbGV4OjF9Ci5tb20tbm17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5tb20tc3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KLm1vbS1wY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2ZvbnQtd2VpZ2h0OjQwMDtmbGV4LXNocmluazowfQoubW9tLXBjLnJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tb20tcGMuZntjb2xvcjp2YXIoLS1mYWxsKX0KLm1vbS10cntoZWlnaHQ6MS41cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O21hcmdpbjozcHggMCAwO292ZXJmbG93OmhpZGRlbn0KLm1vbS1mbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweH0KCi5yZWctaXR7CiAgZGlzcGxheTpmbGV4O2dhcDo5cHg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDsKICBwYWRkaW5nOjhweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzOwp9Ci5yZWctaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLnJlZy1pdDpob3ZlcntvcGFjaXR5OjAuNzV9Ci5yZWctYmFkZ2V7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjA3ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIHBhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNCk7CiAgY29sb3I6dmFyKC0tYWNjZW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4O3doaXRlLXNwYWNlOm5vd3JhcDsKfQoucmVnLWZse2ZsZXg6MTtmb250LXNpemU6MTEuNXB4O2xpbmUtaGVpZ2h0OjEuNX0KLnJlZy1mcm9te2NvbG9yOnZhcigtLWZhaW50KX0KLnJlZy1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTttYXJnaW46MCA0cHh9Ci5yZWctdG97Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5yZWctdG17Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4fQoKLyogRkFWUyAqLwouZmF2c3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAzNnB4IDQwcHg7Cn0KLmZhdnMtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQouZmF2cy1yb3d7ZGlzcGxheTpmbGV4O2dhcDoxMHB4O292ZXJmbG93LXg6YXV0bztwYWRkaW5nLWJvdHRvbTozcHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXJ7aGVpZ2h0OjJweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MXB4fQouZmF2LWNhcmR7CiAgZmxleDowIDAgMTkwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MTJweDtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4czsKfQouZmF2LWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMjIpO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMil9Ci5mYy1oZWFke2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjdweH0KLmZjLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjp2YXIoLS1pbmspfQouZmMtc2N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouZmMtcm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDozcHh9Ci5mYy1yb3cgLnZ7Y29sb3I6dmFyKC0tZGltKTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHh9Ci5mYXZzLWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LXN0eWxlOml0YWxpYztwYWRkaW5nOjRweCAwfQoKLyogRk9PVCAqLwouZm9vdHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjQ4cHggMzZweCA2MHB4O21heC13aWR0aDo1ODBweDttYXJnaW46MCBhdXRvO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmZvb3QtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbi1ib3R0b206MTRweH0KLmZvb3QtbGluZXtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjEuODttYXJnaW4tYm90dG9tOjEycHh9Ci5mb290LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KX0KCi8qIGFuaW1hdGlvbnMgKi8KQGtleWZyYW1lcyBmYWRlVXB7ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoubWFwLWNhcmQsLnN0YXRlLXBhbmVsLC5uYXItY2FyZCwuc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uOmZhZGVVcCAwLjU1cyBjdWJpYy1iZXppZXIoLjIsLjgsLjIsMSkgYmFja3dhcmRzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjA3c30KLm5hci1jYXJkOm50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC4xNHN9Ci5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb24tZGVsYXk6MC4wNXN9CgpAbWVkaWEobWF4LXdpZHRoOjExMDBweCl7CiAgLm1haW57Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAuc3RhdGUtcGFuZWx7bWF4LWhlaWdodDpub25lfQogIC5uYXItcm93e2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9Cn0KCi8qIOKUgOKUgCBXSEFUIElORElBIElTIFJFQUNUSU5HIFRPIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwoud2lyLXNlY3Rpb257CiAgZmxleDoxO21pbi13aWR0aDowOwogIHBhZGRpbmc6MDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLndpci1oZWFkZXJ7CiAgcGFkZGluZzoxOHB4IDIycHggMTRweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Cn0KLndpci10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuM2VtOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC44NTsKfQoud2lyLWxpdmV7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMWVtOwp9Ci53aXItbGl2ZS1kb3R7CiAgd2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojMzlmZjE0OwogIGJveC1zaGFkb3c6MCAwIDZweCByZ2JhKDU3LDI1NSwyMCwwLjYpOwogIGFuaW1hdGlvbjp3aXJMaXZlUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyB3aXJMaXZlUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC4zfX0KLndpci1zaWduYWxze2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47ZmxleDoxO292ZXJmbG93OmhpZGRlbn0KLndpci1zaWduYWx7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjA7CiAgcGFkZGluZzoxM3B4IDIycHg7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzNSk7CiAgb3BhY2l0eTowOwogIGFuaW1hdGlvbjp3aXJGYWRlSW4gMC42cyBlYXNlIGZvcndhcmRzOwogIHBvc2l0aW9uOnJlbGF0aXZlO2N1cnNvcjpkZWZhdWx0OwogIHRyYW5zaXRpb246YmFja2dyb3VuZCAwLjE1czsKfQoud2lyLXNpZ25hbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci53aXItc2lnbmFsOmxhc3QtY2hpbGR7Ym9yZGVyLWJvdHRvbTpub25lfQpAa2V5ZnJhbWVzIHdpckZhZGVJbntmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWCgtNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoud2lyLXNpZ25hbC1iYXJ7CiAgd2lkdGg6MnB4O2JvcmRlci1yYWRpdXM6MXB4O2ZsZXgtc2hyaW5rOjA7CiAgbWFyZ2luLXJpZ2h0OjE0cHg7bWFyZ2luLXRvcDo0cHg7CiAgYWxpZ24tc2VsZjpzdHJldGNoO21pbi1oZWlnaHQ6MTZweDsKICBvcGFjaXR5OjAuNjsKfQoud2lyLXNpZ25hbC1jb250ZW50e2ZsZXg6MTttaW4td2lkdGg6MH0KLndpci1zaWduYWwtdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE0LjVweDtmb250LXdlaWdodDozMDA7CiAgY29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQoud2lyLXNpZ25hbC10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOmluaGVyaXQ7b3BhY2l0eTowLjh9Ci53aXItc2lnbmFsLW1ldGF7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6NHB4Owp9Ci53aXItc2lnbmFsLXRhZ3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6N3B4O2xldHRlci1zcGFjaW5nOjAuMTRlbTsKICB0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7b3BhY2l0eTowLjQ1Owp9Ci53aXItc2lnbmFsLWxvY3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci53aXItbG9hZGluZ3sKICBkaXNwbGF5OmZsZXg7Z2FwOjZweDtwYWRkaW5nOjIwcHggMjJweDthbGlnbi1pdGVtczpjZW50ZXI7Cn0KLndpci1kb3R7d2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjp3aXJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZX0KLndpci1kb3Q6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjJzfQoud2lyLWRvdDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuNHN9CkBrZXlmcmFtZXMgd2lyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KCi8qIOKUgOKUgCBTVEFUUyBTVFJJUCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KI3N0YXRzLXN0cmlwe292ZXJmbG93OmhpZGRlbn0KLnN0YXQtY2VsbHsKICBmbGV4OjE7bWluLXdpZHRoOjA7cGFkZGluZzoxNHB4IDE4cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICBqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDcpOwp9Ci5zdGF0LWNlbGw6bGFzdC1jaGlsZHtib3JkZXItcmlnaHQ6bm9uZX0KLnN0YXQtZGl2e3dpZHRoOjFweDtiYWNrZ3JvdW5kOnJnYmEoMTYwLDE5MCwyMzAsMC4wNyk7ZmxleC1zaHJpbms6MDttYXJnaW46OHB4IDB9Ci5zYy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7d2hpdGUtc3BhY2U6bm93cmFwfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjZ2dywyMnB4KTtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjE7d2hpdGUtc3BhY2U6bm93cmFwO292ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzfQouc2MtdmFsLXNte2ZvbnQtc2l6ZTpjbGFtcCgxM3B4LDEuMnZ3LDE2cHgpIWltcG9ydGFudH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4O3doaXRlLXNwYWNlOm5vd3JhcDtvdmVyZmxvdzpoaWRkZW47dGV4dC1vdmVyZmxvdzplbGxpcHNpc30KLnNjLWhvdmVyYWJsZXtwb3NpdGlvbjpyZWxhdGl2ZTtjdXJzb3I6ZGVmYXVsdH0KLnNjLXRvb2x0aXB7CiAgZGlzcGxheTpub25lO3Bvc2l0aW9uOmFic29sdXRlO2JvdHRvbTpjYWxjKDEwMCUgKyA4cHgpO2xlZnQ6NTAlOwogIHRyYW5zZm9ybTp0cmFuc2xhdGVYKC01MCUpOwogIGJhY2tncm91bmQ6cmdiYSg4LDEyLDIwLDAuOTcpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4IDE0cHg7d2lkdGg6MjAwcHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU7CiAgei1pbmRleDo5OTk5O3BvaW50ZXItZXZlbnRzOm5vbmU7d2hpdGUtc3BhY2U6bm9ybWFsO3RleHQtYWxpZ246bGVmdDsKICBib3gtc2hhZG93OjAgOHB4IDI0cHggcmdiYSgwLDAsMCwwLjUpOwp9Ci5zYy1ob3ZlcmFibGU6aG92ZXIgLnNjLXRvb2x0aXB7ZGlzcGxheTpibG9ja30KLnNjLXRpcC10aXRsZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NnB4fQouc2MtdGlwLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjZweDttYXJnaW4tYm90dG9tOjRweDtmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pfQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PgoKPGRpdiBpZD0ibHRhYi10b29sdGlwIj48L2Rpdj4KCjwhLS0gTE9BREVSIC0tPgo8ZGl2IGlkPSJhcHAtbG9hZGVyIiBzdHlsZT0icG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDt6LWluZGV4Ojk5OTk4O2JhY2tncm91bmQ6IzA2MDkxMDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246b3BhY2l0eSAwLjhzIGVhc2UsdmlzaWJpbGl0eSAwLjhzIGVhc2U7Ij4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt3aWR0aDo2NHB4O2hlaWdodDo2NHB4O21hcmdpbi1ib3R0b206MzZweCI+CiAgICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDoyNHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6I2UwNWEyODthbmltYXRpb246bGRyUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNSk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuNXMiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6LTEwcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjA3KTthbmltYXRpb246bGRyUmluZyAycyBlYXNlLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MXMiPjwvZGl2PgogIDwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidQbGF5ZmFpciBEaXNwbGF5JyxHZW9yZ2lhLHNlcmlmO2ZvbnQtc2l6ZTpjbGFtcCgyOHB4LDV2dyw0MnB4KTtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtjb2xvcjojZjBlY2U0O2xpbmUtaGVpZ2h0OjE7bWFyZ2luLWJvdHRvbToxMHB4Ij4KICAgIDxlbSBzdHlsZT0iY29sb3I6I2U4YzRhMDtmb250LXN0eWxlOml0YWxpYyI+UHVsc2U8L2VtPiBvZiBJbmRpYQogIDwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4O2xldHRlci1zcGFjaW5nOjAuMjhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjQpO21hcmdpbi1ib3R0b206MjhweCI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lczwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjI1KTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4Ij4KICAgIDxzcGFuPk5vdCBuZXdzPC9zcGFuPjxzcGFuIHN0eWxlPSJvcGFjaXR5OjAuMyI+wrc8L3NwYW4+PHNwYW4+Tm90IHByZWRpY3Rpb248L3NwYW4+PHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPkp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KTthbmltYXRpb246bGRyR2xvdyAycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+b2JzZXJ2YXRpb248L3NwYW4+PC9zcGFuPgogIDwvZGl2PgogIDxkaXYgc3R5bGU9Im1hcmdpbi10b3A6NDhweDtkaXNwbGF5OmZsZXg7Z2FwOjZweCI+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+PC9zcGFuPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuMnMiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjRzIj48L3NwYW4+CiAgPC9kaXY+CjwvZGl2Pgo8c3R5bGU+CkBrZXlmcmFtZXMgbGRyUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjE7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19CkBrZXlmcmFtZXMgbGRyUmluZ3swJXt0cmFuc2Zvcm06c2NhbGUoMC44KTtvcGFjaXR5OjAuNn0xMDAle3RyYW5zZm9ybTpzY2FsZSgxLjUpO29wYWNpdHk6MH19CkBrZXlmcmFtZXMgbGRyR2xvd3swJSwxMDAle3RleHQtc2hhZG93OjAgMCAxMHB4IHJnYmEoNTcsMjU1LDIwLDAuNSl9NTAle3RleHQtc2hhZG93OjAgMCAyMnB4IHJnYmEoNTcsMjU1LDIwLDAuOSksMCAwIDQwcHggcmdiYSg1NywyNTUsMjAsMC4zKX19CkBrZXlmcmFtZXMgbGRyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KPC9zdHlsZT4KCjxkaXYgY2xhc3M9InRvcGJhciI+CiAgPGRpdiBjbGFzcz0iYnJhbmQiPgogICAgPGRpdiBjbGFzcz0iYnJhbmQtbWFyayI+PHNwYW4gY2xhc3M9ImJyYW5kLXB1bHNlLWRvdCI+PC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0iYnJhbmQtdGV4dC1ibG9jayI+CiAgICAgIDxzcGFuIGNsYXNzPSJicmFuZC1uYW1lIj48ZW0gY2xhc3M9ImJyYW5kLXB1bHNlLXdvcmQiPlB1bHNlPC9lbT4gb2YgSW5kaWE8L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJicmFuZC10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvc3Bhbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InRvcGJhci1yIj4KICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICAgIDxzcGFuIGlkPSJsaXZlLWNvdW50Ij7igKY8L3NwYW4+IHNpZ25hbHMKICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci10aXAiPgogICAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci1sYWJlbCI+T2JzZXJ2ZWQgZnJvbTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci1zb3VyY2VzIj5yZWdpb25hbCBtZWRpYSDCtyBwdWJsaWMgZGlzY3Vzc2lvbiDCtyBpbmRlcGVuZGVudCByZXBvcnRpbmcgwrcgc29jaWFsIHNpZ25hbHM8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gSEVSTyAtLT4KPHNlY3Rpb24gY2xhc3M9Imhlcm8iIHN0eWxlPSJwYWRkaW5nLXRvcDo4MHB4O3BhZGRpbmctYm90dG9tOjI0cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuIj4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTt3aWR0aDo2MDBweDtoZWlnaHQ6MzUwcHg7dG9wOi02MHB4O2xlZnQ6LTgwcHg7YmFja2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSBhdCA0MCUgNTAlLHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLHRyYW5zcGFyZW50IDY1JSk7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7YW5pbWF0aW9uOmFtYmllbnRTaGlmdCAxMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUgYWx0ZXJuYXRlIj48L2Rpdj4KICA8c3R5bGU+QGtleWZyYW1lcyBhbWJpZW50U2hpZnR7MCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMCl9MTAwJXt0cmFuc2Zvcm06dHJhbnNsYXRlWCgyNHB4KSB0cmFuc2xhdGVZKC0xMnB4KX19PC9zdHlsZT4KICA8ZGl2IGNsYXNzPSJoZXJvLWV5ZWJyb3ciIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPkNvbGxlY3RpdmUgYXR0ZW50aW9uICZtaWRkb3Q7IEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1icmFuZC1ibG9jayIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE4cHg7bWFyZ2luLWJvdHRvbToxNnB4O3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MSI+CiAgICA8ZGl2IGNsYXNzPSJoZXJvLXB1bHNlLXNpZ25hbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJocHMtY29yZSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMiI+PC9zcGFuPgogICAgPC9kaXY+CiAgICA8aDEgY2xhc3M9Imhlcm8tYnJhbmQtbmFtZSI+PGVtPlB1bHNlPC9lbT4gb2YgSW5kaWE8L2gxPgogIDwvZGl2PgogIDxwIGNsYXNzPSJoZXJvLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9wPgogIDxwIGNsYXNzPSJoZXJvLWRlc2MiPk9ic2VydmUgaG93IEluZGlhJ3MgbmFycmF0aXZlcyBhbmQgcHVibGljIGF0dGVudGlvbiBzaGlmdCBpbiByZWFsIHRpbWUuPC9wPgogIDxwIGNsYXNzPSJoZXJvLXN1Yi1saW5lIj5PYnNlcnZpbmcgSW5kaWEgaW4gbW90aW9uLjwvcD4KCiAgPCEtLSBMSVZFIFNUQVRTIFNUUklQIC0tPgo8ZGl2IGlkPSJzdGF0cy1zdHJpcCIgc3R5bGU9InBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MjtiYWNrZ3JvdW5kOnJnYmEoOSwxMywyMSwwLjkpO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4wOCk7cGFkZGluZzowIDM2cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7Ij4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5TaWduYWxzIHRyYWNrZWQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwgc2MtdmFsLXNtIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLXNpZ25hbHMtc3ViIj5sb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSIgc3R5bGU9ImN1cnNvcjpwb2ludGVyIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1ob3R0ZXN0LXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1ob3R0ZXN0LXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1hbmdlci1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1hbmdlci10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCByaXNpbmc8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1yaXNpbmctdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1yaXNpbmctc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtcmlzaW5nLXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5Ub3AgbmFycmF0aXZlPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIHNjLXZhbC1zbSIgaWQ9InNjLW5hci12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hci1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1uYXItdGlwIj48L2Rpdj48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIHNjLWhvdmVyYWJsZSI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkxlYXN0IGFjdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWNvb2wtdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29sLXN1YiI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLWNvb2wtdGlwIj48L2Rpdj48L2Rpdj4KPC9kaXY+CgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJ3aXItc2VjdGlvbiI+CiAgICAgIDxkaXYgY2xhc3M9Indpci1oZWFkZXIiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci10aXRsZSI+V2hhdCBJbmRpYSBpcyByZWFjdGluZyB0bzwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Indpci1saXZlIj48c3BhbiBjbGFzcz0id2lyLWxpdmUtZG90Ij48L3NwYW4+bGl2ZSBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ3aXItc2lnbmFscyIgaWQ9Indpci1zaWduYWxzIj4KICAgICAgICA8ZGl2IGNsYXNzPSJ3aXItbG9hZGluZyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+UmVjZW50IG5hcnJhdGl2ZSBzaGlmdHM8L3NwYW4+CiAgICAgIDwvZGl2PgogICAgICA8IS0tIHNoaWZ0cyBsaXN0IC0tPgogICAgICA8ZGl2IHN0eWxlPSJmbGV4OjE7b3ZlcmZsb3c6aGlkZGVuO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47anVzdGlmeS1jb250ZW50OmNlbnRlcjtwYWRkaW5nOjEwcHggMTRweDtnYXA6NnB4OyIgaWQ9InNoaWZ0LWxpc3QiPjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCgo8IS0tIE1BSU46IE1BUCArIFNUQVRFIFBBTkVMIC0tPgo8ZGl2IGNsYXNzPSJtYWluIj4KCiAgPGRpdiBjbGFzcz0ibWFwLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibWFwLXRvcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC10aXRsZS1ibG9jayI+CiAgICAgICAgPGRpdiBjbGFzcz0ibXQiPkluZGlhICZtZGFzaDsgY29sbGVjdGl2ZSBhdHRlbnRpb248L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtcyIgaWQ9Im1hcC1tZXRhIj4zMCBzdGF0ZXMgJm1pZGRvdDsgbGl2ZSBzaWduYWwgY29tcG9zaXRlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJsZWdlbmQiPjxzcGFuPnF1aWV0PC9zcGFuPjxkaXYgY2xhc3M9ImxlZ2VuZC1iYXIiPjwvZGl2PjxzcGFuPmFjdGl2ZTwvc3Bhbj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibGF5ZXItcm93Ij4KICAgICAgPHNwYW4gY2xhc3M9ImxheWVyLWxhYmVsIj5WaWV3PC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJsdGFicyI+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iZW1vdGlvbiI+RW1vdGlvbiA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iVGhlIGRvbWluYW50IGVtb3Rpb25hbCB0b25lIOKAlCBhbnhpb3VzLCBhbmdyeSwgaG9wZWZ1bCwgcHJvdWQgb3IgZmVhcmZ1bC4gUmV2ZWFscyB0aGUgcHN5Y2hvbG9naWNhbCB1bmRlcmN1cnJlbnQgb2YgcG9saXRpY2FsIGF0dGVudGlvbi4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJ2ZWxvY2l0eSI+TW9tZW50dW0gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IklzIGF0dGVudGlvbiByaXNpbmcgb3IgZmFsbGluZz8gUmlzaW5nID0gbmFycmF0aXZlIGFjY2VsZXJhdGluZy4gQ29vbGluZyA9IGxvc2luZyB0cmFjdGlvbi4gU2hvd3Mgc3RhdGVzIGVudGVyaW5nIG9yIGV4aXRpbmcgYSBwb2xpdGljYWwgY3ljbGUuIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im1hcC1zdmctd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC1pbm5lciI+CiAgICAgICAgPHN2ZyBpZD0iaW5kaWEtbWFwIiB2aWV3Qm94PSIwIDAgODAwIDgwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgbWVldCI+CiAgICAgICAgICA8ZGVmcz4KICAgICAgICAgICAgPHJhZGlhbEdyYWRpZW50IGlkPSJhbWJHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9InJnYmEoMjI0LDkwLDQwLDAuMDQpIi8+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICAgICAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgICAgICAgICA8ZmlsdGVyIGlkPSJzdGF0ZUdsb3ciIHg9Ii0zMCUiIHk9Ii0zMCUiIHdpZHRoPSIxNjAlIiBoZWlnaHQ9IjE2MCUiPgogICAgICAgICAgICAgIDxmZUdhdXNzaWFuQmx1ciBpbj0iU291cmNlR3JhcGhpYyIgc3RkRGV2aWF0aW9uPSI4IiByZXN1bHQ9ImJsdXIiLz4KICAgICAgICAgICAgICA8ZmVDb21wb3NpdGUgaW49IlNvdXJjZUdyYXBoaWMiIGluMj0iYmx1ciIgb3BlcmF0b3I9Im92ZXIiLz4KICAgICAgICAgICAgPC9maWx0ZXI+CiAgICAgICAgICA8L2RlZnM+CiAgICAgICAgICA8cmVjdCB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgZmlsbD0idXJsKCNhbWJHbG93KSIvPgogICAgICAgICAgPGcgaWQ9Im1hcC1nbG93Ij48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXN0YXRlcyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1wdWxzZXMiPjwvZz4KICAgICAgICA8L3N2Zz4KICAgICAgICA8ZGl2IGNsYXNzPSJtYXAtdG9vbHRpcCIgaWQ9InRvb2x0aXAiPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFNUQVRFIFBBTkVMIC0tPgogIDxkaXYgY2xhc3M9InN0YXRlLXBhbmVsIiBpZD0ic3RhdGUtZGV0YWlsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLWVtcHR5Ij4KICAgICAgPHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMSI+CiAgICAgICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgOHY0TTEyIDE2aC4wMSIvPgogICAgICA8L3N2Zz4KICAgICAgPGRpdiBjbGFzcz0icGUtdCI+U2VsZWN0IGEgc3RhdGU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0icGUtcyI+Q2xpY2sgYW55IHJlZ2lvbiBvbiB0aGUgbWFwPGJyLz50byBvcGVuIGl0cyBuYXJyYXRpdmUgcGFuZWwuPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCjwvZGl2PgoKPCEtLSBOQVJSQVRJVkUgUk9XIC0tPgo8ZGl2IGNsYXNzPSJuYXItcm93IiBpZD0ibmFyLXJvdyIgc3R5bGU9Im9wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC41cyBlYXNlIj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SaXNpbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1sZWZ0OmF1dG8iPmdhaW5pbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyaXNpbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgZmFsbCI+PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0ibmMtdGl0bGUiPkRlY2xpbmluZyBuYXJyYXRpdmVzPC9zcGFuPgogICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWxlZnQ6YXV0byI+bG9zaW5nIHRyYWN0aW9uPC9zcGFuPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0iZGVjbGluaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSBGQVZTIC0tPgo8c2VjdGlvbiBjbGFzcz0iZmF2cyI+CiAgPGRpdiBjbGFzcz0iZmF2cy1sYWJlbCI+VHJhY2tlZCBzdGF0ZXM8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmYXZzLXJvdyIgaWQ9ImZhdi1yb3ciPgogICAgPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjxkaXYgY2xhc3M9ImZvb3QiPgogIDxkaXYgY2xhc3M9ImZvb3QtbmFtZSI+UHVsc2Ugb2YgSW5kaWE8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmb290LWxpbmUiPk9ic2VydmVzIGhvdyBwdWJsaWMgYXR0ZW50aW9uIHNoaWZ0cyBhY3Jvc3MgdGhlIGNvdW50cnkg4oCUIHVzaW5nIHNpZ25hbHMgZnJvbSBuZXdzLCBkaXNjb3Vyc2UsIGFuZCByZWdpb25hbCBkZXZlbG9wbWVudHMuPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1zdWIiPk5vdCBuZXdzLiBOb3QgcHJlZGljdGlvbi4gSnVzdCA8c3BhbiBzdHlsZT0iY29sb3I6IzM5ZmYxNDt0ZXh0LXNoYWRvdzowIDAgOHB4IHJnYmEoNTcsMjU1LDIwLDAuNCkiPm9ic2VydmF0aW9uPC9zcGFuPi48L2Rpdj4KPC9kaXY+Cgo8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L25wbS90b3BvanNvbi1jbGllbnRAMy4xLjAvZGlzdC90b3BvanNvbi1jbGllbnQubWluLmpzIj48L3NjcmlwdD4KPHNjcmlwdD4KdmFyIEFQSV9CQVNFPShsb2NhdGlvbi5ob3N0bmFtZT09PSdsb2NhbGhvc3QnfHxsb2NhdGlvbi5ob3N0bmFtZT09PScxMjcuMC4wLjEnKT8naHR0cDovL2xvY2FsaG9zdDo4MDAwJzonJzsKCi8vIEFQSQphc3luYyBmdW5jdGlvbiBmZXRjaEFsbFN0YXRlcygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlcycpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciByb3dzPWF3YWl0IHIuanNvbigpOwogICAgaWYoIXJvd3N8fCFyb3dzLmxlbmd0aCkgcmV0dXJuOwogICAgcm93cy5mb3JFYWNoKGZ1bmN0aW9uKHJvdyl7CiAgICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKHJvdy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tRW1vPXJvdy5kb21pbmFudF9lbW90aW9ufHxkb21pbmFudEVtb3Rpb24oZW1vcyl8fG51bGw7CiAgICAgIHZhciBlbnRyeT17YXR0ZW50aW9uOnJvdy5hdHRlbnRpb24sZGVsdGE6cm93LmRlbHRhXzI0aCx2ZWxvY2l0eTpyb3cudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb21FbW8sZG9taW5hbnRfbmFycmF0aXZlOnJvdy5kb21pbmFudF9uYXJyYXRpdmUsZW1vdGlvbnM6ZW1vc307CiAgICAgIExJVkVbcm93Lm5hbWVdPWVudHJ5OwogICAgICBpZighU0Rbcm93Lm5hbWVdKSBTRFtyb3cubmFtZV09T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTsKICAgICAgT2JqZWN0LmFzc2lnbihTRFtyb3cubmFtZV0sZW50cnkpOwogICAgfSk7CiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICByZW5kZXJTdHJpcCgiM20iKTsKICAgIGJ1aWxkV0lSU2lnbmFscygpOwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgc2V0VGltZW91dChyZW5kZXJNb21lbnR1bSwgNTAwKTsKICAgIGlmKFNFTCYmTElWRVtTRUxdJiZkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJykpIHJlbmRlclBhbmVsKFNFTCk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW0FQSV0nLGUubWVzc2FnZSk7fQp9CgpmdW5jdGlvbiBidWlsZExvY2FsSW5zaWdodCgpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwoKICAvLyBBZ2dyZWdhdGUgdG9wIG5hcnJhdGl2ZXMgYWNyb3NzIGFsbCBzdGF0ZXMKICB2YXIgbmFyPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5hcltuLm5hbWVdKSBuYXJbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuYXJbbi5uYW1lXVtuLmRpcl09KG5hcltuLm5hbWVdW24uZGlyXXx8MCkrbi52YWw7CiAgICAgIG5hcltuLm5hbWVdLnRvdGFsPShuYXJbbi5uYW1lXS50b3RhbHx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKCiAgLy8gVG9wIHJpc2luZyBhbmQgZmFsbGluZyAoZXhjbHVkZSB0aWVzIHdoZXJlIHNhbWUgbmFtZSByaXNlcyBhbmQgZmFsbHMpCiAgdmFyIHJpc2luZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPmt2WzFdLmRvd247fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udXAtYVsxXS51cDt9KS5zbGljZSgwLDMpOwogIHZhciBmYWxsaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS5kb3duLWFbMV0uZG93bjt9KS5zbGljZSgwLDIpOwogIHZhciB0b3AzPU9iamVjdC5lbnRyaWVzKG5hcikuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnRvdGFsLWFbMV0udG90YWw7fSkuc2xpY2UoMCwzKTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PWVudHJpZXMuc2xpY2UoKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgaG90dGVzdEVtbz1ob3R0ZXN0PyhMSVZFW2hvdHRlc3RbMF1dJiZMSVZFW2hvdHRlc3RbMF1dLmRvbWluYW50X2Vtb3Rpb24pfHwnJzonJyA7CgogIC8vIEJ1aWxkIGluc2lnaHQgdGV4dCDigJQgbW9yZSBhbmFseXRpY2FsLCBjb250ZXh0LWF3YXJlCiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICBpZighZWwpIHJldHVybjsKCiAgdmFyIGxpbmVzPVtdOwogIGlmKHJpc2luZy5sZW5ndGgmJmZhbGxpbmcubGVuZ3RoJiZyaXNpbmdbMF1bMF0hPT1mYWxsaW5nWzBdWzBdKXsKICAgIGxpbmVzLnB1c2goJzxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIHRoZSBkb21pbmFudCBzaWduYWwgYWNyb3NzIEluZGlhIHRvZGF5Jyk7CiAgICBpZihmYWxsaW5nWzBdKSBsaW5lcy5wdXNoKCcgYXMgPGVtPicrZmFsbGluZ1swXVswXSsnPC9lbT4gZmFkZXMgZnJvbSBuYXRpb25hbCBmb2N1cycpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBpcyB0aGUgbW9zdCBhY3RpdmUgc3RhdGUnKwogICAgICAoaG90dGVzdEVtbz8nIHdpdGggJytob3R0ZXN0RW1vKycgYXMgdGhlIHByaW1hcnkgc2lnbmFsIHRvbmUnOicnKSk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJy4gU2Vjb25kYXJ5IHN1cmdlOiA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHJpc2luZy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnU2lnbmFscyBhcmUgY29uY2VudHJhdGVkIGFyb3VuZCA8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBsZWFkcyBuYXRpb25hbCBhdHRlbnRpb24nKTsKICAgIGlmKHJpc2luZ1sxXSkgbGluZXMucHVzaCgnIGFsb25nc2lkZSA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHRvcDMubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ05hdGlvbmFsIHNpZ25hbHMgYXJlIGRpc3BlcnNlZC4gVG9wIG5hcnJhdGl2ZXM6ICcrdG9wMy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZW0+JytuWzBdKyc8L2VtPic7fSkuam9pbignLCAnKSk7CiAgfQoKICBpZihsaW5lcy5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2xpbmVzLmpvaW4oJycpKycuPC9kaXY+JzsKICB9CgogIC8vIFRhZ3MKICBpZih0RWwpewogICAgdmFyIHRhZ3M9W107CiAgICBmYWxsaW5nLnNsaWNlKDAsMSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoNTksMTg0LDIxNiwwLjMpO2NvbG9yOiMzYmI4ZDgiPuKGkyAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICByaXNpbmcuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7Y29sb3I6I2UwNWEyOCI+4oaRICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIGlmKHRhZ3MubGVuZ3RoKSB0RWwuaW5uZXJIVE1MPXRhZ3Muam9pbignJyk7CiAgfQoKICBpZihtZXRhRWwpewogICAgdmFyIHN0YXRlQ291bnQ9T2JqZWN0LnZhbHVlcyhMSVZFKS5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuYXR0ZW50aW9uPjI7fSkubGVuZ3RoOwogICAgbWV0YUVsLnRleHRDb250ZW50PSdPYnNlcnZpbmcgJytzdGF0ZUNvdW50KycgYWN0aXZlIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pOwogIH0KfQoKZnVuY3Rpb24gdXBkYXRlQWxsU3RyaXBzKCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CgogIC8vIE1lcmdlIFNEIGRhdGEgZm9yIG5hcnJhdGl2ZXMvc291cmNlX2NvdW50L2NvbmZpZGVuY2UKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgaWYoU0Rba3ZbMF1dKXsKICAgICAgaWYoU0Rba3ZbMF1dLm5hcnJhdGl2ZXMpIGt2WzFdLm5hcnJhdGl2ZXM9U0Rba3ZbMF1dLm5hcnJhdGl2ZXM7CiAgICAgIGlmKFNEW2t2WzBdXS5zb3VyY2VfY291bnQpIGt2WzFdLnNvdXJjZV9jb3VudD1TRFtrdlswXV0uc291cmNlX2NvdW50OwogICAgICBpZihTRFtrdlswXV0uY29uZmlkZW5jZSkga3ZbMV0uY29uZmlkZW5jZT1TRFtrdlswXV0uY29uZmlkZW5jZTsKICAgIH0KICB9KTsKCiAgLy8gVG9vbHRpcCBoZWxwZXIKICBmdW5jdGlvbiB0aXAoaWQsdGl0bGUsbmFycyl7CiAgICB2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIXQpcmV0dXJuOwogICAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0aXRsZSsnPC9kaXY+JysobmFyc3x8W10pLnNsaWNlKDAsMykubWFwKGZ1bmN0aW9uKG4pewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9InNjLXRpcC1yb3ciPsK3ICcrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gU2lnbmFscyB0cmFja2VkCiAgdmFyIHRvdD1PYmplY3QudmFsdWVzKFNEKS5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcysodi5zaWduYWxfY291bnR8fDApO30sMCk7CiAgdmFyIGxjPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7aWYobGMpbGMudGV4dENvbnRlbnQ9dG90LnRvTG9jYWxlU3RyaW5nKCdlbi1JTicpOwogIHZhciBzdj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Mtc2lnbmFscy12YWwnKTtpZihzdilzdi50ZXh0Q29udGVudD10b3QudG9Mb2NhbGVTdHJpbmcoJ2VuLUlOJyk7CiAgdmFyIGFjdGl2ZUNvdW50PWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4oa3ZbMV0uYXR0ZW50aW9ufHwwKT4yO30pLmxlbmd0aDsKICBzZXRUZXh0KCdzYy1zaWduYWxzLXN1YicsJ2Fjcm9zcyAnK2FjdGl2ZUNvdW50KycgYWN0aXZlIHN0YXRlcycpOwoKICAvLyBDb250ZXh0dWFsIHNpZ25pZmljYW5jZSB3ZWlnaHRzIOKAlCBzYW1lIHNpZ25hbHMgbWVhbiBtb3JlIGluIGhpZ2gtaW1wb3J0YW5jZSBzdGF0ZXMKICB2YXIgU0lHPXsKICAgICdKYW1tdSBhbmQgS2FzaG1pcic6Mi4yLCdNYW5pcHVyJzoyLjAsJ0RlbGhpJzoxLjksJ1V0dGFyIFByYWRlc2gnOjEuNywKICAgICdXZXN0IEJlbmdhbCc6MS41LCdQdW5qYWInOjEuNSwnTWFoYXJhc2h0cmEnOjEuNCwnQmloYXInOjEuMywKICAgICdBc3NhbSc6MS4zLCdBcnVuYWNoYWwgUHJhZGVzaCc6MS40LCdDaGhhdHRpc2dhcmgnOjEuMiwnS2VyYWxhJzoxLjIsCiAgICAnS2FybmF0YWthJzoxLjIsJ1RhbWlsIE5hZHUnOjEuMiwnUmFqYXN0aGFuJzoxLjIsJ01hZGh5YSBQcmFkZXNoJzoxLjIsCiAgICAnR3VqYXJhdCc6MS4yLCdIYXJ5YW5hJzoxLjIsJ1RlbGFuZ2FuYSc6MS4xLCdBbmRocmEgUHJhZGVzaCc6MS4xLAogICAgJ09kaXNoYSc6MS4xLCdKaGFya2hhbmQnOjEuMSwnTmFnYWxhbmQnOjEuMSwnVHJpcHVyYSc6MS4xLAogIH07CiAgdmFyIE5BUl9TSUc9eydib3JkZXIgaXNzdWVzJzoxLjgsJ2xhdyAmIG9yZGVyJzoxLjYsJ3NlY3VyaXR5JzoxLjYsCiAgICAnZWxlY3Rpb25zJzoxLjUsJ2NvbW11bmFsJzoxLjcsJ2NvcnJ1cHRpb24nOjEuNCwncHJvdGVzdCc6MS40LAogICAgJ2dvdmVybmFuY2UnOjEuMywnbmF0aW9uYWxpc20nOjEuMywncmVsaWdpb24nOjEuNCwnZWNvbm9teSc6MS4yfTsKCiAgLy8gQ29udGV4dHVhbCBzY29yZSA9IGF0dGVudGlvbiDDlyBzdGF0ZSBzaWduaWZpY2FuY2Ugw5cgbmFycmF0aXZlIHNpZ25pZmljYW5jZQogIGZ1bmN0aW9uIGN0eFNjb3JlKGt2KXsKICAgIHZhciBhdHQ9a3ZbMV0uYXR0ZW50aW9ufHwwOwogICAgdmFyIHNTaWc9U0lHW2t2WzBdXXx8MS4wOwogICAgdmFyIG5hclNpZz1OQVJfU0lHW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZXx8JyddfHwxLjA7CiAgICB2YXIgY29uZj17J0hJR0gnOjEuMCwnTUVESVVNJzowLjg1LCdMT1cnOjAuNn1ba3ZbMV0uY29uZmlkZW5jZXx8J0xPVyddfHwwLjY7CiAgICByZXR1cm4gYXR0ICogKDErKHNTaWctMSkqMC40KSAqICgxKyhuYXJTaWctMSkqMC4yKSAqIGNvbmY7CiAgfQoKICAvLyBIaWdoZXN0IGF0dGVudGlvbiDigJQgYnkgY29udGV4dHVhbCBzY29yZSwgbm90IHJhdyBhdHRlbnRpb24KICB2YXIgaG90dGVzdD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBjdHhTY29yZShiKS1jdHhTY29yZShhKTt9KVswXTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsaG90dGVzdFswXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCdBdHRlbnRpb24gJysoKGhvdHRlc3RbMV0uYXR0ZW50aW9ufHwwKS50b0ZpeGVkP2hvdHRlc3RbMV0uYXR0ZW50aW9uLnRvRml4ZWQoMSk6aG90dGVzdFsxXS5hdHRlbnRpb24pKTsKICB0aXAoJ3NjLWhvdHRlc3QtdGlwJywnV2h5ICcraG90dGVzdFswXSsnPycsaG90dGVzdFsxXS5uYXJyYXRpdmVzKTsKCiAgLy8gUGVhayBhbmdlciBzdGF0ZSDigJQgY29udGV4dHVhbGx5IHdlaWdodGVkCiAgdmFyIGFuZ2VyRG9tPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcic7fSk7CiAgaWYoYW5nZXJEb20ubGVuZ3RoKXsKICAgIHZhciB0b3BBbmdlcj1hbmdlckRvbS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGN0eFNjb3JlKGIpLWN0eFNjb3JlKGEpO30pWzBdOwogICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlclswXSk7CiAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLHRvcEFuZ2VyWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J2FuZ2VyIHNpZ25hbHMnKTsKICAgIHRpcCgnc2MtYW5nZXItdGlwJywnQW5nZXIgaW4gJyt0b3BBbmdlclswXSx0b3BBbmdlclsxXS5uYXJyYXRpdmVzKTsKICB9CgogIC8vIEZhc3Rlc3QgcmlzaW5nIOKAlCBieSB2ZWxvY2l0eQogIHZhciBub3JtVj1mdW5jdGlvbih2KXtpZighdilyZXR1cm4gMDt2YXIgYT1NYXRoLmFicyh2KTtpZihhPjEpdj12L01hdGgubWF4KGEsNTApO3JldHVybiBNYXRoLm1heCgtMSxNYXRoLm1pbigxLHYpKTt9OwogIHZhciByaXNpbmc9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBub3JtVihrdlsxXS52ZWxvY2l0eXx8MCk+MC4wMTt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gbm9ybVYoYlsxXS52ZWxvY2l0eXx8MCktbm9ybVYoYVsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYocmlzaW5nKXsKICAgIHNldFRleHQoJ3NjLXJpc2luZy12YWwnLHJpc2luZ1swXSk7CiAgICBzZXRUZXh0KCdzYy1yaXNpbmctc3ViJyxyaXNpbmdbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnc2lnbmFsIHJpc2luZycpOwogICAgdGlwKCdzYy1yaXNpbmctdGlwJywnV2h5ICcrcmlzaW5nWzBdKycgaXMgcmlzaW5nJyxyaXNpbmdbMV0ubmFycmF0aXZlcyk7CiAgfSBlbHNlIHsKICAgIC8vIFNob3cgaGlnaGVzdCB2ZWxvY2l0eSBldmVuIGlmIGFsbCBwb3NpdGl2ZQogICAgdmFyIHRvcFZlbD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLnZlbG9jaXR5fHwwKS0oYVsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgICBpZih0b3BWZWwpe3NldFRleHQoJ3NjLXJpc2luZy12YWwnLHRvcFZlbFswXSk7c2V0VGV4dCgnc2MtcmlzaW5nLXN1YicsdG9wVmVsWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J21vc3QgbW9tZW50dW0nKTt9CiAgfQoKICAvLyBUb3AgbmFycmF0aXZlCiAgdmFyIG5jPXt9OwogIGVudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICAoa3ZbMV0ubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7bmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7fSk7CiAgICBpZigha3ZbMV0ubmFycmF0aXZlcyYma3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlKSBuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdPShuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdfHwwKSsxOwogIH0pOwogIHZhciB0Tj1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgaWYodE4pewogICAgc2V0VGV4dCgnc2MtbmFyLXZhbCcsdE5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdE5bMF0uc2xpY2UoMSkpOwogICAgdmFyIG5TdGF0ZXM9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5uYXJyYXRpdmVzfHxbXSkuc29tZShmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lPT09dE5bMF07fSk7fSkuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiBrdlswXS5zcGxpdCgnICcpWzBdO30pOwogICAgc2V0VGV4dCgnc2MtbmFyLXN1YicsblN0YXRlcy5qb2luKCcsICcpfHwnbmF0aW9uYWxseScpOwogICAgdmFyIHRUPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzYy1uYXItdGlwJyk7CiAgICBpZih0VCl0VC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0TlswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0TlswXS5zbGljZSgxKSsnIOKAlCBpbjwvZGl2PicrCiAgICAgIGVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4oa3ZbMV0ubmFycmF0aXZlc3x8W10pLnNvbWUoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZT09PXROWzBdO30pO30pLnNsaWNlKDAsMykKICAgICAgLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJzYy10aXAtcm93Ij7CtyAnK2t2WzBdKyc8L2Rpdj4nO30pLmpvaW4oJycpOwogIH0KCiAgLy8gTGVhc3QgYWN0aXZlIChsb3dlc3QgdmVsb2NpdHkpCiAgdmFyIGNvb2w9ZW50cmllcy5zbGljZSgpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gbm9ybVYoYVsxXS52ZWxvY2l0eXx8MCktbm9ybVYoYlsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYoY29vbCl7CiAgICBzZXRUZXh0KCdzYy1jb29sLXZhbCcsY29vbFswXSk7CiAgICB2YXIgY1Y9bm9ybVYoY29vbFsxXS52ZWxvY2l0eXx8MCk7CiAgICBzZXRUZXh0KCdzYy1jb29sLXN1YicsKGNvb2xbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnJykrKGNWPC0wLjA1PycgwrcgcmV0cmVhdGluZyc6JyDCtyBsZWFzdCBtb21lbnR1bScpKTsKICAgIHRpcCgnc2MtY29vbC10aXAnLCdMb3dlc3QgbW9tZW50dW06ICcrY29vbFswXSxjb29sWzFdLm5hcnJhdGl2ZXMpOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUvJytlbmNvZGVVUklDb21wb25lbnQobmFtZSkpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMoZC5lbW90aW9uc3x8e30pOwogICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgIFNEW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZSwKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSxzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OmQubmFycmF0aXZlSGlzdG9yeXx8REVGQVVMVC5uYXJyYXRpdmVIaXN0b3J5LHNpZ25hbF9jb3VudDpkLnNpZ25hbF9jb3VudHx8MH07CiAgICBpZighTElWRVtuYW1lXSlMSVZFW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmV9OwogICAgTElWRVtuYW1lXS5lbW90aW9ucz1lbW9zO0xJVkVbbmFtZV0uZG9taW5hbnRfZW1vdGlvbj1kb207CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbmFtZSxlLm1lc3NhZ2UpO3JldHVybiBTRFtuYW1lXXx8REVGQVVMVDt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICAvLyB0b3BiYXIKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGl2ZS1jb3VudCcpOwogICAgaWYoZWwmJmQudG90YWxfc2lnbmFscykgZWwudGV4dENvbnRlbnQ9ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk7CiAgICB2YXIgbWV0YT1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLW1ldGEnKTsKICAgIGlmKG1ldGEmJmQuYXNfb2YpIG1ldGEudGV4dENvbnRlbnQ9JzMwIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoZC5hc19vZikudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicpOwogICAgLy8gc3RhdHMgc3RyaXAKICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJywgZC50b3RhbF9zaWduYWxzP2QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOictJyk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICB9Y2F0Y2goZSl7fQp9CgpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZihlbCllbC50ZXh0Q29udGVudD12YWw7fQoKZnVuY3Rpb24gdXBkYXRlU3RyaXBOYXJyYXRpdmUoKXt1cGRhdGVBbGxTdHJpcHMoKTt9CmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXt9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hJbnNpZ2h0cygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2luc2lnaHRzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yfHwhZC5yaXNpbmcpIHJldHVybjsgIC8vIGd1YXJkOiBza2lwIGlmIGRhdGEgaW5jb21wbGV0ZQogICAgdmFyIHNpZz1kLnNpZ25hdHVyZTsKICAgIGlmKHNpZyl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgaWYoZWwpZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij48ZW0+JytzaWcuZmFkaW5nLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3NpZy5mYWRpbmcuc2xpY2UoMSkrJzwvZW0+IGZhZGluZyBhcyA8ZW0+JytzaWcucmlzaW5nX3ByaW1hcnkrIjwvZW0+Iisoc2lnLnJpc2luZ19zZWNvbmRhcnk/IiBhbG9uZ3NpZGUgPGVtPiIrc2lnLnJpc2luZ19zZWNvbmRhcnkrIjwvZW0+IjoiIikrIiBhY3Jvc3MgdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gPHN0cm9uZyBzdHlsZT1cImNvbG9yOnZhcigtLWluaylcIj4iK3NpZy5ob3R0ZXN0X3N0YXRlKyI8L3N0cm9uZz4gZG9taW5hdGVzLjwvZGl2PiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC50YWdzKXRFbC5pbm5lckhUTUw9ZC50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgfQogICAgc2V0VGltZW91dChmdW5jdGlvbigpewogICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgIGlmKHJFbCYmZC5yaXNpbmcmJmQucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQucmlzaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytuLnN0YXRlcy5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKSsnJTtiYWNrZ3JvdW5kOiNlMDVhMjgiPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgaWYoZkVsJiZkLmZhbGxpbmcmJmQuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6IzNiYjhkOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIC8vIEZhZGUgaW4gbmFyLXJvdyBvbmx5IHdoZW4gZGF0YSBpcyByZWFkeSDigJQgcHJldmVudHMgZmxhc2ggb2YgYnJva2VuIHRleHQKICB2YXIgbmFyUm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduYXItcm93Jyk7CiAgaWYobmFyUm93JiYocmlzaW5nLmxlbmd0aHx8ZmFsbGluZy5sZW5ndGgpKSBuYXJSb3cuc3R5bGUub3BhY2l0eT0nMSc7CgogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgIGlmKGdFbCYmZC5yZWdpb25hbCYmZC5yZWdpb25hbC5sZW5ndGgpZ0VsLmlubmVySFRNTD1kLnJlZ2lvbmFsLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW4iPjxzcGFuIGNsYXNzPSJuaS1uYW1lIj4nK3IucmVnaW9uKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+JytyLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PjxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytyLmhvdHRlc3Rfc3RhdGUrJyDCtyAnK3IudG9wX25hcnJhdGl2ZSsnPC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIC8vIFJldmVhbCBuYXItcm93IGFmdGVyIGRhdGEgd3JpdHRlbgogICAgdmFyIG5yPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduYXItcm93Jyk7aWYobnIpbnIuc3R5bGUub3BhY2l0eT0nMSc7CiAgICB9LDUwKTsKCiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoRnVsbFNuYXBzaG90KCl7CiAgLy8gTG9hZCBBTEwgc3RhdGUgZGF0YSBpbiBvbmUgcmVxdWVzdCBmb3IgaW5zdGFudCBmaXJzdC1sb2FkCiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvZnVsbC1zbmFwc2hvdCcpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC53YXJtaW5nX3VwfHwhZC5zdGF0ZXN8fCFkLnN0YXRlcy5sZW5ndGgpIHJldHVybiBmYWxzZTsKCiAgICAvLyBQb3B1bGF0ZSBTRCBhbmQgTElWRSBmcm9tIGZ1bGwgc25hcHNob3QKICAgIGQuc3RhdGVzLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAgIGlmKCFzLm5hbWUpIHJldHVybjsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8cy5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgICB2YXIgZW50cnk9T2JqZWN0LmFzc2lnbih7fSxzLHtlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRlbHRhOnMuZGVsdGFfMjRofHwwfSk7CiAgICAgIFNEW3MubmFtZV09ZW50cnk7CiAgICAgIExJVkVbcy5uYW1lXT17YXR0ZW50aW9uOnMuYXR0ZW50aW9uLGRlbHRhOnMuZGVsdGFfMjRofHwwLHZlbG9jaXR5OnMudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOnMuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgfSk7CgogICAgLy8gVXBkYXRlIHNpZ25hbHMgY291bnQKICAgIGlmKGQuc25hcHNob3QmJmQuc25hcHNob3QudG90YWxfc2lnbmFscyl7CiAgICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJyxkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKSk7CiAgICB9CgogICAgLy8gVXBkYXRlIGluc2lnaHRzIGZyb20gY2FjaGVkIGRhdGEKICAgIGlmKGQuaW5zaWdodHMmJmQuaW5zaWdodHMuc2lnbmF0dXJlKXsKICAgICAgdmFyIHNpZz1kLmluc2lnaHRzLnNpZ25hdHVyZTsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLmluc2lnaHRzLnRhZ3MpdEVsLmlubmVySFRNTD1kLmluc2lnaHRzLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICAgIGlmKHJFbCYmZC5pbnNpZ2h0cy5yaXNpbmcmJmQuaW5zaWdodHMucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMucmlzaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgICBpZihmRWwmJmQuaW5zaWdodHMuZmFsbGluZyYmZC5pbnNpZ2h0cy5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIH0KCiAgICAvLyBSZW5kZXIgbWFwIGNvbG9ycyBhbmQgc3RyaXBzCiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICByZW5kZXJTdHJpcCgiM20iKTsKICAgIC8vIExvYWQgaW5zaWdodHMgdG9vCiAgICBidWlsZExvY2FsSW5zaWdodCgpOwogICAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7CiAgICAvLyBVc2UgY2FjaGVkIG5hcnJhdGl2ZSBpbnNpZ2h0IGlmIGF2YWlsYWJsZQogICAgaWYoZC5uYXJyYXRpdmVfaW5zaWdodCYmZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0KXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwogICAgICBpZihlbCkgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2QubmFycmF0aXZlX2luc2lnaHQudGV4dCsnPC9kaXY+JzsKICAgICAgaWYodEVsJiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRvcF9uYXJyYXRpdmVzKXsKICAgICAgfQogICAgfQogICAgcmV0dXJuIHRydWU7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbZnVsbC1zbmFwc2hvdF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gZmFsc2U7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKXsKICB0cnl7CiAgICAvLyBUcnkgY2FjaGVkIHZlcnNpb24gZnJvbSBmdWxsLXNuYXBzaG90IGZpcnN0IChhbHJlYWR5IGxvYWRlZCkKICAgIC8vIFRoZW4gY2FsbCBkZWRpY2F0ZWQgZW5kcG9pbnQgZm9yIGZyZXNoIEFJIGFuYWx5c2lzCiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9uYXJyYXRpdmUtaW5zaWdodCcpOwogICAgaWYoIXIub2spIHJldHVybjsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoIWQudGV4dCkgcmV0dXJuOwoKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwoKICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC50ZXh0Kyc8L2Rpdj4nOwoKICAgIC8vIFRhZ3MgZnJvbSB0b3AgbmFycmF0aXZlcwogICAgaWYodEVsJiZkLnRvcF9uYXJyYXRpdmVzJiZkLnRvcF9uYXJyYXRpdmVzLmxlbmd0aCl7CiAgICAgIHRFbC5pbm5lckhUTUw9ZC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICB2YXIgY29sPWk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjYpJzsKICAgICAgICB2YXIgYXJyb3c9aT09PTA/J+KGkSAnOifCtyAnOwogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpO2NvbG9yOicrY29sKyciPicrYXJyb3crbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9CgogICAgaWYobWV0YUVsKXsKICAgICAgdmFyIHQ9bmV3IERhdGUoZC5hc19vZik7CiAgICAgIG1ldGFFbC50ZXh0Q29udGVudD0nU2lnbmFsIGFuYWx5c2lzIMK3ICcrdC50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pKyhkLmZhbGxiYWNrPycgwrcgcGF0dGVybi1iYXNlZCc6JyDCtyBBSSBzeW50aGVzaXplZCcpOwogICAgfQogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tuYXJyYXRpdmVdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlKTt9KTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDEyMDAwMCk7CiAgICAgIHNldEludGVydmFsKGZldGNoSW5zaWdodHMsMzYwMDAwMCk7fQogIH0sMTUwMDApOwp9CgovLyBOQVJSQVRJVkUgREFUQQp2YXIgU0hJRlRTPXsKICAnM20nOlsKICAgIHtmYWRpbmc6J0luZmxhdGlvbicsZmFkaW5nTm90ZTonZWFzaW5nIG5hdGlvbmFsbHknLHJpc2luZzonQm9yZGVyIHNlY3VyaXR5JyxyaXNpbmdOb3RlOidwb3N0LWluY2lkZW50IHN1cmdlJ30sCiAgICB7ZmFkaW5nOidFbGVjdGlvbiByaGV0b3JpYycsZmFkaW5nTm90ZToncG9zdC1jeWNsZSBmYWRlJyxyaXNpbmc6J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3N0ZWFkeSByaXNlJ30sCiAgICB7ZmFkaW5nOidGYXJtZXIgcHJvdGVzdHMnLGZhZGluZ05vdGU6J21vbWVudHVtIGxvc3QnLHJpc2luZzonVW5lbXBsb3ltZW50IGFueGlldHknLHJpc2luZ05vdGU6J3lvdXRoIHNpZ25hbCBzdXJnZSd9LAogIF0sCiAgJzZtJzpbCiAgICB7ZmFkaW5nOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZhZGluZ05vdGU6J3ByZS1lbGVjdGlvbiBwZWFrJyxyaXNpbmc6J0NvcnJ1cHRpb24gYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3Bvc3QtY3ljbGUgcHVzaCd9LAogICAge2ZhZGluZzonUmVsaWdpb3VzIG5hdGlvbmFsaXNtJyxmYWRpbmdOb3RlOidwbGF0ZWF1IHBoYXNlJyxyaXNpbmc6J0Vjb25vbWljIGFueGlldHknLHJpc2luZ05vdGU6J2Nvc3Qtb2YtbGl2aW5nJ30sCiAgICB7ZmFkaW5nOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZmFkaW5nTm90ZToncmliYm9uLWN1dHRpbmcgZG9uZScscmlzaW5nOidMYXcgJiBvcmRlcicscmlzaW5nTm90ZTonY3JpbWUgbmFycmF0aXZlIHJpc2UnfSwKICBdLAogICcxeSc6WwogICAge2ZhZGluZzonUGFuZGVtaWMgcmVjb3ZlcnknLGZhZGluZ05vdGU6J2ZhZGVkIGVhcmx5IHllYXInLHJpc2luZzonSW5mbGF0aW9uJyxyaXNpbmdOb3RlOidkb21pbmF0ZWQgbWlkLXllYXInfSwKICAgIHtmYWRpbmc6J1JlZ2lvbmFsIGlkZW50aXR5JyxmYWRpbmdOb3RlOidsYW5ndWFnZS1sZWQgcGVhaycscmlzaW5nOidTZWN1cml0eSAmIGJvcmRlcnMnLHJpc2luZ05vdGU6J2dlb3BvbGl0aWNhbCBlc2NhbGF0aW9uJ30sCiAgICB7ZmFkaW5nOidHb3Zlcm5hbmNlIG9wdGltaXNtJyxmYWRpbmdOb3RlOidwb2xpY3kgaG9uZXltb29uIGVuZCcscmlzaW5nOidDb3JydXB0aW9uICYgc2NhbXMnLHJpc2luZ05vdGU6J2FjY291bnRhYmlsaXR5IGN5Y2xlJ30sCiAgXSwKfTsKdmFyIFJFR19TSElGVFM9WwogIHtzdGF0ZTonVGFtaWwgTmFkdScsZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLHRvOidGZWRlcmFsIHJlc291cmNlIGRpc3B1dGVzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonQmloYXInLGZyb206J0VsZWN0aW9uIHJoZXRvcmljJyx0bzonVW5lbXBsb3ltZW50ICYgZXhhbSBzY2FtcycsdGltZTonNiB3a3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2tzJ30sCiAge3N0YXRlOidSYWphc3RoYW4nLGZyb206J0Zhcm1lciBwcm90ZXN0cycsdG86J0hlYXQgd2F2ZSDCtyBFbnZpcm9ubWVudCcsdGltZTonMiB3a3MnfSwKICB7c3RhdGU6J0thcm5hdGFrYScsZnJvbTonTWluaW5nIGNvbnRyb3ZlcnN5Jyx0bzonTGFuZ3VhZ2Ugc2lnbmFnZSBwb2xpdGljcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyBpbmZyYXN0cnVjdHVyZScsdG86J0FpciBxdWFsaXR5IGNyaXNpcycsdGltZTonMTAgZGF5cyd9LAogIHtzdGF0ZTonTWFuaXB1cicsZnJvbTonR292ZXJuYW5jZSAmIGNhYmluZXQnLHRvOidFdGhuaWMgdGVuc2lvbnMgwrcgQUZTUEEnLHRpbWU6JzUgd2tzJ30sCiAge3N0YXRlOidQdW5qYWInLGZyb206J1Bvd2VyIGNyaXNpcycsdG86J0JvcmRlciBzZWN1cml0eSDCtyBEcm9uZXMnLHRpbWU6JzMgd2tzJ30sCl07CnZhciBNT0NLX1I9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbiBwcmVzc3VyZScsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2hpZnQtbGlzdCcpOwogIGlmKCFlbCkgcmV0dXJuOwogIHZhciBuYz17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIGlmKCFuY1tuLm5hbWVdKW5jW24ubmFtZV09e3VwOjAsZG93bjowLHN0YXRlczpbXX07CiAgICAgIG5jW24ubmFtZV1bbi5kaXI9PT0ndXAnPyd1cCc6J2Rvd24nXSs9KG4udmFsfHwwKTsKICAgICAgaWYobmNbbi5uYW1lXS5zdGF0ZXMuaW5kZXhPZihzLm5hbWV8fCcnKTwwKW5jW24ubmFtZV0uc3RhdGVzLnB1c2gocy5uYW1lfHwnJyk7CiAgICB9KTsKICB9KTsKICB2YXIgYWxsPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0udXArYlsxXS5kb3duKS0oYVsxXS51cCthWzFdLmRvd24pO30pOwogIHZhciByaXNpbmc9YWxsLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPj1rdlsxXS5kb3duO30pLnNsaWNlKDAsMyk7CiAgdmFyIGZhZGluZz1hbGwuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KTsKICBpZighZmFkaW5nLmxlbmd0aCkgZmFkaW5nPWFsbC5zbGljZSgtMyk7CiAgZmFkaW5nPWZhZGluZy5zbGljZSgwLDMpOwogIGlmKCFhbGwubGVuZ3RoKXsKICAgIGVsLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkNvbGxlY3Rpbmcgc2lnbmFsIGRhdGEuLi48L2Rpdj4nOwogICAgcmV0dXJuOwogIH0KICB2YXIgcm93cz1bXTsKICBmb3IodmFyIGk9MDtpPE1hdGgubWF4KHJpc2luZy5sZW5ndGgsZmFkaW5nLmxlbmd0aCwxKTtpKyspewogICAgdmFyIHI9cmlzaW5nW2ldLGY9ZmFkaW5nW2ldOwogICAgcm93cy5wdXNoKAogICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O292ZXJmbG93OmhpZGRlbjttYXJnaW4tYm90dG9tOjZweCI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjhweCAxMHB4O2JvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjojM2JiOGQ4O21hcmdpbi1ib3R0b206M3B4Ij5GQURJTkc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMCI+JysoZj9mWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2ZbMF0uc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj4nKwogICAgICAgIChmJiZmWzFdLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nK2ZbMV0uc3RhdGVzLnNsaWNlKDAsMikuam9pbignLCAnKSsnPC9kaXY+JzonJykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzowIDhweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zaXplOjE0cHgiPuKGkjwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo4cHggMTBweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjojZTA1YTI4O21hcmdpbi1ib3R0b206M3B4Ij5SSVNJTkc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMCI+Jysocj9yWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3JbMF0uc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj4nKwogICAgICAgIChyJiZyWzFdLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nK3JbMV0uc3RhdGVzLnNsaWNlKDAsMikuam9pbignLCAnKSsnPC9kaXY+JzonJykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JwogICAgKTsKICB9CiAgZWwuaW5uZXJIVE1MPXJvd3Muam9pbignJyk7Cn0KZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odGFiKXsKICB0YWIuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0KXt0LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgdGFiLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlbmRlclN0cmlwKHRhYi5kYXRhc2V0LnBlcmlvZCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gcmVuZGVyTW9tZW50dW0oKXsKICAvLyBEb24ndCByZW5kZXIgdW50aWwgU0QgaGFzIGRhdGEg4oCUIHByZXZlbnRzIGZsYXNoIG9mIGJyb2tlbi9wYXJ0aWFsIGNvbnRlbnQKICBpZighT2JqZWN0LmtleXMoU0QpLmxlbmd0aCkgcmV0dXJuOwogIC8vIFJlYWQgZnJvbSBTRCAocG9wdWxhdGVkIGJ5IGZldGNoQWxsU3RhdGVzIGZyb20gbGl2ZSBBUEkpCiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgbmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpOwogIHZhciBmYWxsaW5nPXNvcnRlZC5zbGljZSgtNSkucmV2ZXJzZSgpOwogIHZhciBteD1yaXNpbmcubGVuZ3RoP3Jpc2luZ1swXVsxXToxMDA7CgogIC8vIFdyaXRlIHRvIHJpc2luZy1saXN0IChtYXRjaGVzIG5hci1yb3cgSFRNTCkKICAvLyBEZWZlciB3cml0ZSB0byBhdm9pZCBmbGFzaCBvZiB1bnN0eWxlZCBjb250ZW50CiAgdmFyIF9ybVJpc2luZz1yaXNpbmcsX3JtRmFsbGluZz1mYWxsaW5nLF9ybU14PW14OwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogIHZhciByaXNpbmc9X3JtUmlzaW5nLGZhbGxpbmc9X3JtRmFsbGluZyxteD1fcm1NeDsKICBpZihyRWwmJnJpc2luZy5sZW5ndGgpewogICAgckVsLmlubmVySFRNTD1yaXNpbmcubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIGRlY2xpbmluZy1saXN0CiAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICBpZihmRWwmJmZhbGxpbmcubGVuZ3RoKXsKICAgIGZFbC5pbm5lckhUTUw9ZmFsbGluZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIHJlZ2lvbmFsLWxpc3Qg4oCUIHRvcCBzdGF0ZSBwZXIgcmVnaW9uIGZyb20gTElWRQogIHZhciByZWdpb25zPXsKICAgICdOb3J0aCc6WydEZWxoaScsJ1V0dGFyIFByYWRlc2gnLCdQdW5qYWInLCdIYXJ5YW5hJywnSGltYWNoYWwgUHJhZGVzaCcsJ1V0dGFyYWtoYW5kJywnSmFtbXUgYW5kIEthc2htaXInXSwKICAgICdFYXN0JzpbJ1dlc3QgQmVuZ2FsJywnQmloYXInLCdKaGFya2hhbmQnLCdPZGlzaGEnXSwKICAgICdXZXN0JzpbJ01haGFyYXNodHJhJywnR3VqYXJhdCcsJ1JhamFzdGhhbicsJ0dvYSddLAogICAgJ1NvdXRoJzpbJ1RhbWlsIE5hZHUnLCdLYXJuYXRha2EnLCdLZXJhbGEnLCdBbmRocmEgUHJhZGVzaCcsJ1RlbGFuZ2FuYSddLAogICAgJ05FJzpbJ0Fzc2FtJywnTWFuaXB1cicsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ1RyaXB1cmEnLCdBcnVuYWNoYWwgUHJhZGVzaCcsJ1Npa2tpbSddLAogICAgJ0NlbnRyYWwnOlsnTWFkaHlhIFByYWRlc2gnLCdDaGhhdHRpc2dhcmgnXSwKICB9OwogIC8vIEZhZGUgaW4gbmFyLXJvdyBvbmx5IHdoZW4gZGF0YSBpcyByZWFkeSDigJQgcHJldmVudHMgZmxhc2ggb2YgYnJva2VuIHRleHQKICB2YXIgbmFyUm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduYXItcm93Jyk7CiAgaWYobmFyUm93JiYocmlzaW5nLmxlbmd0aHx8ZmFsbGluZy5sZW5ndGgpKSBuYXJSb3cuc3R5bGUub3BhY2l0eT0nMSc7CgogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICBpZihnRWwpewogICAgdmFyIHJlZ0l0ZW1zPU9iamVjdC5lbnRyaWVzKHJlZ2lvbnMpLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciByZWdpb249a3ZbMF0sc3RhdGVzPWt2WzFdOwogICAgICB2YXIgdG9wPXN0YXRlcy5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuIHtuYW1lOnMsYXR0OihMSVZFW3NdJiZMSVZFW3NdLmF0dGVudGlvbil8fDB9O30pCiAgICAgICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSlbMF07CiAgICAgIGlmKCF0b3B8fCF0b3AuYXR0KSByZXR1cm4gbnVsbDsKICAgICAgdmFyIG5hcj0oTElWRVt0b3AubmFtZV0mJkxJVkVbdG9wLm5hbWVdLmRvbWluYW50X25hcnJhdGl2ZSl8fCfigJQnOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6OHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCkiPicrcmVnaW9uKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+Jyt0b3AuYXR0LnRvRml4ZWQoMSkrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrdG9wLm5hbWUrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4OyI+JytuYXIrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuZmlsdGVyKEJvb2xlYW4pLmpvaW4oJycpOwogICAgaWYocmVnSXRlbXMpIGdFbC5pbm5lckhUTUw9cmVnSXRlbXM7CiAgfQogIH0sNTApOwp9CgoKLy8gU1RBVEUgREFUQQp2YXIgU0Q9e307Cgp2YXIgTElWRT17fTsKZnVuY3Rpb24gbm9ybWFsaXplRW1vdGlvbnMoZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJue307dmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhlKSx0b3Q9dmFscy5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcyt2O30sMCk7aWYodG90PD0wKXJldHVybnt9O2lmKHRvdDw9MS4wMSl7dmFyIG91dD17fTtPYmplY3Qua2V5cyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGspe291dFtrXT1NYXRoLnJvdW5kKGVba10qMTAwKTt9KTtyZXR1cm4gb3V0O31yZXR1cm4gZTt9CmZ1bmN0aW9uIGRvbWluYW50RW1vdGlvbihlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm4gbnVsbDt2YXIgbXg9MCxkb209bnVsbDtPYmplY3QuZW50cmllcyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXT5teCl7bXg9a3ZbMV07ZG9tPWt2WzBdO319KTtyZXR1cm4gZG9tO30KZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIWVsKXJldHVybjtlbC50ZXh0Q29udGVudD12YWw7aWYodmFsJiZ2YWwhPT0nLScpe2VsLmNsYXNzTGlzdC5yZW1vdmUoJ2xvYWRpbmcnKTt9fQoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjowLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7fSxkb21pbmFudF9lbW90aW9uOm51bGwsZG9taW5hbnRfbmFycmF0aXZlOm51bGwsCiAgbmFycmF0aXZlczpbXSxyaXNpbmc6W10sZmFsbGluZzpbXSwKICBzdW1tYXJ5OicnLGFydGljbGVzOltdLHRpbWVsaW5lOltdLAogIG5hcnJhdGl2ZUhpc3Rvcnk6W10sc2lnbmFsX2NvdW50OjAsCn07CgpmdW5jdGlvbiBnKG4pe3JldHVybiBTRFtuXXx8T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTt9CgpmdW5jdGlvbiBhQyhzKXsKICAvLyBEeW5hbWljIHNjYWxlOiBhbHdheXMgc3ByZWFkIGZ1bGwgY29sb3IgcmFuZ2UgYWNyb3NzIGFjdHVhbCBkYXRhCiAgLy8gR2V0IG1pbi9tYXggZnJvbSBjdXJyZW50IFNEIHRvIG5vcm1hbGl6ZQogIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKGQpe3JldHVybiBkLmF0dGVudGlvbnx8MDt9KTsKICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpOwogIHZhciBteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgLy8gTm9ybWFsaXplIDAtMQogIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwocy1tbikvKG14LW1uKSkpOwogIC8vIE1hcCB0byBjb2xvciBzdG9wczogZGFyayBibHVlIOKGkiB0ZWFsIOKGkiBhbWJlciDihpIgb3JhbmdlIOKGkiByZWQKICBpZihuPDAuMTIpIHJldHVybiAnIzBkMWUzMCc7CiAgaWYobjwwLjI1KSByZXR1cm4gJyMwZTNkNmEnOwogIGlmKG48MC4zOCkgcmV0dXJuICcjMGQ1ZjkwJzsKICBpZihuPDAuNTApIHJldHVybiAnIzBlN2FhYSc7CiAgaWYobjwwLjYyKSByZXR1cm4gJyMxYTkwOTAnOwogIGlmKG48MC43MikgcmV0dXJuICcjYzg3MDEwJzsKICBpZihuPDAuODIpIHJldHVybiAnI2Q4NDAxMCc7CiAgaWYobjwwLjkyKSByZXR1cm4gJyNjYzE4MDgnOwogIHJldHVybiAnI2ZmMDAxMCc7Cn0KZnVuY3Rpb24gZUMoZSl7CiAgdmFyIG14PTAsZG9tPSdwcmlkZSc7CiAgZm9yKHZhciBrIGluIGUpe2lmKGVba10+bXgpe214PWVba107ZG9tPWs7fX0KICByZXR1cm4gKHthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfSlbZG9tXXx8JyMzM2FhY2MnOwp9CmZ1bmN0aW9uIHZDKHYpewogIC8vIFBlcmNlbnRpbGUtY2xpcHBlZCBub3JtYWxpemF0aW9uIOKAlCBvdXRsaWVycyBkb24ndCBkb21pbmF0ZQogIGlmKCF2Qy5fdHN8fERhdGUubm93KCktdkMuX3RzPjQwMDApewogICAgdmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKGQpe3JldHVybiBkLnZlbG9jaXR5fHwwO30pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYS1iO30pOwogICAgaWYodmFscy5sZW5ndGg+Mil7CiAgICAgIC8vIFVzZSBwMTUgdG8gcDg1IHRvIGNsaXAgb3V0bGllcnMKICAgICAgdmFyIGxvPU1hdGguZmxvb3IodmFscy5sZW5ndGgqMC4xNSksIGhpPU1hdGguY2VpbCh2YWxzLmxlbmd0aCowLjg1KS0xOwogICAgICB2Qy5fbWluPXZhbHNbbG9dOyB2Qy5fbWF4PXZhbHNbaGldOwogICAgfSBlbHNlIGlmKHZhbHMubGVuZ3RoKXsKICAgICAgdkMuX21pbj12YWxzWzBdOyB2Qy5fbWF4PXZhbHNbdmFscy5sZW5ndGgtMV07CiAgICB9IGVsc2UgeyB2Qy5fbWluPTA7IHZDLl9tYXg9MTsgfQogICAgaWYodkMuX21heDw9dkMuX21pbikgdkMuX21heD12Qy5fbWluKzAuMDE7CiAgICB2Qy5fdHM9RGF0ZS5ub3coKTsKICB9CiAgdmFyIG49KHYtdkMuX21pbikvKHZDLl9tYXgtdkMuX21pbik7CiAgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsbikpOwogIC8vIENvb2wgKGJsdWUpIOKGkiBuZXV0cmFsIChzbGF0ZSkg4oaSIHdhcm0gKGFtYmVyL3JlZCkKICBpZihuPDAuMzMpewogICAgdmFyIHQ9bi8wLjMzOwogICAgcmV0dXJuICdyZ2IoJytNYXRoLnJvdW5kKDIwKzE0KnQpKycsJytNYXRoLnJvdW5kKDEwMCs1Myp0KSsnLCcrTWF0aC5yb3VuZCgxODAtMzAqdCkrJyknOwogIH0gZWxzZSBpZihuPDAuNjYpewogICAgdmFyIHQ9KG4tMC4zMykvMC4zMzsKICAgIHJldHVybiAncmdiKCcrTWF0aC5yb3VuZCgzNCsxMDYqdCkrJywnK01hdGgucm91bmQoMTUzLTkzKnQpKycsJytNYXRoLnJvdW5kKDE1MC0xMzAqdCkrJyknOwogIH0gZWxzZSB7CiAgICB2YXIgdD0obi0wLjY2KS8wLjM0OwogICAgcmV0dXJuICdyZ2IoJytNYXRoLnJvdW5kKDE0MCsxMTUqdCkrJywnK01hdGgucm91bmQoNjAtNjAqdCkrJywnK01hdGgucm91bmQoMjApKycpJzsKICB9Cn0KCnZhciBsYXllcj0nZW1vdGlvbicsU0VMPW51bGwsRkFWUz1uZXcgU2V0KCk7CgovLyBNQVAKZnVuY3Rpb24gcHJval8odyxoLHBhZCl7CiAgcGFkPXBhZHx8MjA7CiAgdmFyIG1pbkxvbj02OC4xLG1heExvbj05Ny40LG1pbkxhdD02LjUsbWF4TGF0PTM3LjE7CiAgdmFyIHNjWD0ody1wYWQqMikvKG1heExvbi1taW5Mb24pOwogIHZhciBzY1k9KGgtcGFkKjIpLyhtYXhMYXQtbWluTGF0KTsKICB2YXIgc2M9TWF0aC5taW4oc2NYLHNjWSk7CiAgdmFyIG94PXBhZCsody1wYWQqMi0obWF4TG9uLW1pbkxvbikqc2MpLzI7CiAgdmFyIG95PXBhZCsoaC1wYWQqMi0obWF4TGF0LW1pbkxhdCkqc2MpLzI7CiAgcmV0dXJuIGZ1bmN0aW9uKGxvbixsYXQpe3JldHVybiBbb3grKGxvbi1taW5Mb24pKnNjLCBveSsobWF4TGF0LWxhdCkqc2NdO307Cn0KZnVuY3Rpb24gZ2VvMnBhdGgoZ2VvbSxwail7CiAgdmFyIGQ9Jyc7CiAgZnVuY3Rpb24gcmluZyhjcyl7dmFyIHM9Jyc7Y3MuZm9yRWFjaChmdW5jdGlvbihjLGkpe3ZhciBwPXBqKGNbMF0sY1sxXSk7cys9KGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pO3JldHVybiBzKydaJzt9CiAgaWYoZ2VvbS50eXBlPT09J1BvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTsKICBlbHNlIGlmKGdlb20udHlwZT09PSdNdWx0aVBvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocCl7cC5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7fSk7CiAgcmV0dXJuIGQ7Cn0KZnVuY3Rpb24gY3RyKGdlb20pewogIHZhciBwdHM9W107CiAgZnVuY3Rpb24gY29sKGMpe2lmKHR5cGVvZiBjWzBdPT09J251bWJlcicpIHB0cy5wdXNoKGMpO2Vsc2UgYy5mb3JFYWNoKGNvbCk7fQogIGNvbChnZW9tLmNvb3JkaW5hdGVzKTsKICBpZighcHRzLmxlbmd0aCkgcmV0dXJuIFswLDBdOwogIHJldHVybiBbcHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMF07fSwwKS9wdHMubGVuZ3RoLHB0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzFdO30sMCkvcHRzLmxlbmd0aF07Cn0KZnVuY3Rpb24gc05hbWUocHJvcHMpewogIHZhciByYXc9cHJvcHMuc3Rfbm18fHByb3BzLk5BTUVfMXx8cHJvcHMubmFtZXx8cHJvcHMuTkFNRXx8Jyc7CiAgdmFyIG1hcD17J0xhZGFraCc6J0phbW11IGFuZCBLYXNobWlyJywnSmFtbXUgJiBLYXNobWlyJzonSmFtbXUgYW5kIEthc2htaXInLCdVdHRhcmFuY2hhbCc6J1V0dGFyYWtoYW5kJywnQW5kYW1hbiBhbmQgTmljb2Jhcic6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ0FuZGFtYW4gJiBOaWNvYmFyIElzbGFuZCc6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ05DVCBvZiBEZWxoaSc6J0RlbGhpJywnUG9uZGljaGVycnknOidQdWR1Y2hlcnJ5JywnRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnLCdEYW1hbiBhbmQgRGl1JzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdSd9OwogIHJldHVybiBtYXBbcmF3XXx8cmF3Owp9Cgp2YXIgY2FjaGVkR2VvPW51bGw7Cgphc3luYyBmdW5jdGlvbiBsb2FkTWFwKGF0dGVtcHQpewogIGF0dGVtcHQgPSBhdHRlbXB0fHwxOwogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKCdodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvZ2gvdWRpdC0wMDEvaW5kaWEtbWFwcy1kYXRhQG1hc3Rlci90b3BvanNvbi9pbmRpYS5qc29uJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHRvcG89YXdhaXQgci5qc29uKCk7CiAgICBjYWNoZWRHZW89dG9wb2pzb24uZmVhdHVyZSh0b3BvLHRvcG8ub2JqZWN0cy5zdGF0ZXMpOwogICAgcmVuZGVyTWFwKGNhY2hlZEdlbyk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMTAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMzAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsNjAwMCk7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbbWFwXSBsb2FkIGZhaWxlZCBhdHRlbXB0ICcrYXR0ZW1wdCsnOicsZS5tZXNzYWdlKTsKICAgIGlmKGF0dGVtcHQ8NSl7CiAgICAgIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtsb2FkTWFwKGF0dGVtcHQrMSk7fSwgYXR0ZW1wdCoyMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHZhciBtaT1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJyk7CiAgICAgIGlmKG1pKSBtaS5pbm5lckhUTUw9JzxkaXYgc3R5bGU9ImNvbG9yOiMyYTNhNGE7cGFkZGluZzo0MHB4O3RleHQtYWxpZ246Y2VudGVyO2ZvbnQtZmFtaWx5Om1vbm9zcGFjZTtmb250LXNpemU6MTFweCI+TWFwIHVuYXZhaWxhYmxlIOKAlCByZWZyZXNoIHRvIHJldHJ5PC9kaXY+JzsKICAgIH0KICB9Cn0KCmZ1bmN0aW9uIHJlbmRlck1hcChzdGF0ZXMpewogIHZhciB3PTgwMCxoPTgwMCxwaj1wcm9qXyh3LGgsMjgpOwogIHZhciBzZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXN0YXRlcycpOwogIHZhciBwZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXB1bHNlcycpOwogIHZhciBnZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLWdsb3cnKTsKICBzZy5pbm5lckhUTUw9Jyc7cGcuaW5uZXJIVE1MPScnO2dnLmlubmVySFRNTD0nJzsKCiAgc3RhdGVzLmZlYXR1cmVzLmZvckVhY2goZnVuY3Rpb24oZil7CiAgICBpZighZi5nZW9tZXRyeSkgcmV0dXJuOwogICAgdmFyIG5tPXNOYW1lKGYucHJvcGVydGllcyksZD1nKG5tKTsKICAgIHZhciBwYXRoRWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ3BhdGgnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2QnLGdlbzJwYXRoKGYuZ2VvbWV0cnkscGopKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywnc3RhdGUnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScsbm0pOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJywncmdiYSgyNTUsMjU1LDI1NSwwLjA3KScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMC41Jyk7CiAgICBzZy5hcHBlbmRDaGlsZChwYXRoRWwpOwoKICAgIHZhciBjdD1jdHIoZi5nZW9tZXRyeSksY3A9cGooY3RbMF0sY3RbMV0pOwoKICAgIC8vIEF0bW9zcGhlcmljIGdsb3cgZm9yIGhpZ2gtYXR0ZW50aW9uIHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTY1KXsKICAgICAgdmFyIGdsb3dFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnZWxsaXBzZScpOwogICAgICB2YXIgZ2xvd1I9TWF0aC5taW4oNjAsMjArZC5hdHRlbnRpb24qMC41KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtnbG93RWwuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdyeCcsZ2xvd1IpO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J5JyxnbG93UiowLjcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWxsJyxhQyhkLmF0dGVudGlvbikpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdvcGFjaXR5JywnMC4wOCcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWx0ZXInLCd1cmwoI3N0YXRlR2xvdyknKTsKICAgICAgZ2xvd0VsLnN0eWxlLmFuaW1hdGlvbj0nZ2xvd1B1bHNlICcrKDIuNStNYXRoLnJhbmRvbSgpKSsncyBlYXNlLWluLW91dCAnKyhNYXRoLnJhbmRvbSgpKjIpKydzIGluZmluaXRlJzsKICAgICAgZ2cuYXBwZW5kQ2hpbGQoZ2xvd0VsKTsKICAgIH0KCiAgICAvLyBEdWFsIHB1bHNlIHJpbmdzIGZvciB2ZXJ5IGhvdCBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj03Mil7CiAgICAgIFswLDFdLmZvckVhY2goZnVuY3Rpb24oaSl7CiAgICAgICAgdmFyIHJpbmc9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2NpcmNsZScpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO3Jpbmcuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3B1bHNlLXJpbmcgcCcrKGkrMSkpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2UnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzEnKTsKICAgICAgICByaW5nLnN0eWxlLmFuaW1hdGlvbkRlbGF5PShNYXRoLnJhbmRvbSgpKjIuNSkrJ3MnOwogICAgICAgIHBnLmFwcGVuZENoaWxkKHJpbmcpOwogICAgICB9KTsKICAgIH0KICB9KTsKICBhcHBseUxheWVyKCk7CiAgYXR0YWNoSW50ZXJhY3Rpb25zKCk7Cn0KCi8vIFNpbmdsZSBzb3VyY2Ugb2YgdHJ1dGggZm9yIGVtb3Rpb24gY29sb3IKLy8gQm90aCBtYXAgYW5kIHBhbmVsIGNhbGwgdGhpcyDigJQgZ3VhcmFudGVlcyB0aGV5IGFsd2F5cyBtYXRjaApmdW5jdGlvbiBnZXRFZmZlY3RpdmVFbW90aW9uKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogIC8vIDEuIFRyeSBMSVZFLmRvbWluYW50X2Vtb3Rpb24gKHNldCBieSAvYXBpL3N0YXRlcykKICB2YXIgZG9tPWxpdmUuZG9taW5hbnRfZW1vdGlvbnx8ZC5kb21pbmFudF9lbW90aW9uOwoKICAvLyAyLiBUcnkgY29tcHV0aW5nIGZyb20gZW1vdGlvbnMgYnJlYWtkb3duCiAgaWYoIWRvbSl7CiAgICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogICAgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKTsKICB9CgogIC8vIDMuIEZhbGxiYWNrOiBpbmZlciBmcm9tIGRvbWluYW50IG5hcnJhdGl2ZSAoc2FtZSBsb2dpYyBldmVyeXdoZXJlKQogIGlmKCFkb20pewogICAgdmFyIG5wPShsaXZlLmRvbWluYW50X25hcnJhdGl2ZXx8ZC5kb21pbmFudF9uYXJyYXRpdmV8fCcnKS50b0xvd2VyQ2FzZSgpOwogICAgaWYobnAubWF0Y2goL2JvcmRlcnx0ZXJyb3J8c2VjdXJpdHl8Y29uZmxpY3R8YXR0YWNrfHdhcnxpbmZpbHRyYXQvKSkgZG9tPSdmZWFyJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL3NjYW18Y29ycnVwdHxwcm90ZXN0fGFycmVzdHx2aW9sZW5jZXxvdXRyYWdlfGNyaW1lLykpIGRvbT0nYW5nZXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvZGV2ZWxvcHxpbnZlc3R8Z3Jvd3RofGxhdW5jaHxpbmF1Z3VyfHJlZm9ybXxwcm9ncmVzc3xib29zdC8pKSBkb209J2hvcGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvY3VsdHVyZXxoZXJpdGFnZXxwcmlkZXx2aWN0b3J5fGNlbGVicmF0fG1lZGFsfGFjaGlldmVtZW50LykpIGRvbT0ncHJpZGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvZmxvb2R8ZHJvdWdodHx1bmVtcGxveW1lbnR8aW5mbGF0aW9ufHNob3J0YWdlfGNyaXNpc3xjb25jZXJuLykpIGRvbT0nYW54aWV0eSc7CiAgICBlbHNlIGlmKChsaXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDApPjUpIGRvbT0nYW54aWV0eSc7IC8vIGFjdGl2ZSBzdGF0ZSBkZWZhdWx0CiAgICBlbHNlIGRvbT0nYW54aWV0eSc7IC8vIGdsb2JhbCBkZWZhdWx0CiAgfQoKICByZXR1cm4gZG9tOwp9CgovLyBHZXQgZXN0aW1hdGVkIGVtb3Rpb24gYnJlYWtkb3duIChmb3IgcGFuZWwgZG9udXQgd2hlbiByZWFsIGRhdGEgbWlzc2luZykKZnVuY3Rpb24gZ2V0RW1vdGlvbkJyZWFrZG93bihubSl7CiAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogIHZhciBkPVNEW25tXXx8e307CiAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICBpZihPYmplY3Qua2V5cyhlbW9zKS5sZW5ndGgpIHJldHVybiB7ZW1vdGlvbnM6ZW1vcyxlc3RpbWF0ZWQ6ZmFsc2V9OwogIC8vIEJ1aWxkIHNrZXdlZCBkaXN0cmlidXRpb24gZnJvbSBlZmZlY3RpdmUgZW1vdGlvbgogIHZhciBkb209Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgdmFyIGJhc2U9e2FueGlldHk6MTMsYW5nZXI6MTMsaG9wZToxMyxwcmlkZToxMyxmZWFyOjEzfTsKICBiYXNlW2RvbV09NDg7CiAgcmV0dXJuIHtlbW90aW9uczpiYXNlLGVzdGltYXRlZDp0cnVlfTsKfQoKZnVuY3Rpb24gYXBwbHlMYXllcigpewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksZmlsbDsKICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRlPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICBmaWxsPWVNYXBbZGVdfHwnIzMzNDQ1NSc7CiAgICB9CiAgICBlbHNlIGZpbGw9dkMoZC52ZWxvY2l0eSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsZmlsbCk7CiAgICAoZnVuY3Rpb24oKXsKICAgICAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oeCl7cmV0dXJuIHguYXR0ZW50aW9ufHwwO30pOwogICAgICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpLG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAgICAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChkLmF0dGVudGlvbi1tbikvKG14LW1uKSkpOwogICAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxsYXllcj09PSdhdHRlbnRpb24nP01hdGgubWF4KDAuMywwLjMrbiowLjcpOjAuODUpOwogICAgfSkoKTsKICB9KTsKfQoKZnVuY3Rpb24gYXR0YWNoSW50ZXJhY3Rpb25zKCl7CiAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsZnVuY3Rpb24oZSl7CiAgICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk7CiAgICAgIHZhciBkPWcobm0pOwogICAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICAgIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICAgICAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBsYXRlc3Q9Jyc7CiAgICAgIGlmKGQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzLmxlbmd0aCkgbGF0ZXN0PWQubmFycmF0aXZlc1swXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2QubmFycmF0aXZlc1swXS5uYW1lLnNsaWNlKDEpOwogICAgICBlbHNlIGlmKGxpdmUuZG9taW5hbnRfbmFycmF0aXZlKSBsYXRlc3Q9bGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuc2xpY2UoMSk7CgogICAgICB2YXIgcm93cz0nJzsKICAgICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICAgICAgdmFyIGF0dD1saXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDA7CiAgICAgICAgdmFyIGRsdD1saXZlLmRlbHRhfHxkLmRlbHRhfHwwOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkF0dGVudGlvbjwvc3Bhbj48c3Ryb25nPicrYXR0LnRvRml4ZWQoMSkrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAoZGx0IT09MD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+MjRoIHNoaWZ0PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrKGRsdD4wPycjZTA1YTI4JzonIzNiYjhkOCcpKyciPicrKGRsdD4wPycrJzonJykrZGx0Kyc8L3N0cm9uZz48L2Rpdj4nOicnKSsKICAgICAgICAgIChsYXRlc3Q/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPlRvcCBuYXJyYXRpdmU8L3NwYW4+PHN0cm9uZz4nK2xhdGVzdCsnPC9zdHJvbmc+PC9kaXY+JzonJyk7CiAgICAgIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIGRvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgICBpZihkb21FbW8pewogICAgICAgICAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6ZC5lbW90aW9uc3x8e307CiAgICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Eb21pbmFudDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3BhbFtkb21FbW9dKyciPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW9zKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgICAgICAgIHZhciB0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICAgICAgICBpZih0b3Q+MCYmdG90PD0xLjAxKXtlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybltrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO3RvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTt9CiAgICAgICAgICByb3dzKz1lTC5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3BhbiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NHB4Ij48c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK3BhbFtrdlswXV0rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicra3ZbMF0rJzwvc3Bhbj48c3Ryb25nPicrTWF0aC5yb3VuZChrdlsxXSoxMDAvTWF0aC5tYXgoMSx0b3QpKSsnJTwvc3Ryb25nPjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgICAgfQogICAgICB9IGVsc2UgewogICAgICAgIHZhciB2ZWw9bGl2ZS52ZWxvY2l0eXx8ZC52ZWxvY2l0eXx8MDsKICAgICAgICB2YXIgdmVsRGlyPXZlbD4wLjE/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wMj8nUmlzaW5nJzp2ZWw8LTAuMDU/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgICAgIHZhciB2ZWxDb2w9dmVsPjAuMDI/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPk1vbWVudHVtPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RGlyZWN0aW9uPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrdmVsRGlyKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICB9CgogICAgICB0aXAuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJ0dC1uIj4nK25tKyc8L2Rpdj4nK3Jvd3MrKGxhdGVzdCYmbGF5ZXIhPT0nYXR0ZW50aW9uJz8nPGRpdiBjbGFzcz0idHQtbmFyIj48c3Ryb25nPk5hcnJhdGl2ZTwvc3Ryb25nPicrbGF0ZXN0Kyc8L2Rpdj4nOicnKTsKICAgICAgdmFyIHJlY3Q9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgICB0aXAuc3R5bGUubGVmdD1NYXRoLm1pbihlLmNsaWVudFgtcmVjdC5sZWZ0KzE0LHJlY3Qud2lkdGgtMTkwKSsncHgnOwogICAgICB0aXAuc3R5bGUudG9wPU1hdGgubWluKGUuY2xpZW50WS1yZWN0LnRvcCsxNCxyZWN0LmhlaWdodC0xNTApKydweCc7CiAgICAgIHRpcC5zdHlsZS5vcGFjaXR5PScxJzsKICAgIH0pOwpwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbGVhdmUnLGZ1bmN0aW9uKCl7dGlwLnN0eWxlLm9wYWNpdHk9MDt9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7c2VsZWN0XyhwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJykpO30pOwogIH0pOwp9CgovLyBTVEFURSBQQU5FTApmdW5jdGlvbiBzZWxlY3RfKG5tKXsKICBTRUw9bm07CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuY2xhc3NMaXN0LnRvZ2dsZSgnc2VsZWN0ZWQnLHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKT09PW5tKTsKICB9KTsKICAvLyBTaG93IGxvYWRpbmcgc3RhdGUgaW1tZWRpYXRlbHkgd2l0aCB3aGF0ZXZlciBMSVZFIGRhdGEgd2UgaGF2ZQogIHZhciBwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYocGFuZWwpewogICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgcGFuZWwuaW5uZXJIVE1MPQogICAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgICAnPGRpdj48ZGl2IGNsYXNzPSJzcC1layI+JysobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhGQVZTLmhhcyhubSk/J29uJzonJykrJyIgZGF0YS1ubT0iJytubSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYodGhpcy5kYXRhc2V0Lm5tKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAgICc8L2J1dHRvbj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MjBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4nKwogICAgICAgICdMb2FkaW5nIHNpZ25hbHMgZm9yICcrbm0rJy4uLicrCiAgICAgICAgKGxpdmUuYXR0ZW50aW9uPyc8YnI+PGJyPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Y29sb3I6dmFyKC0taW5rKSI+QXR0ZW50aW9uICcrbGl2ZS5hdHRlbnRpb24udG9GaXhlZCgxKSsnPC9zcGFuPic6JycpKwogICAgICAgIChsaXZlLmRvbWluYW50X2Vtb3Rpb24/Jzxicj48c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nK2xpdmUuZG9taW5hbnRfZW1vdGlvbisnIHNpZ25hbCBkb21pbmFudDwvc3Bhbj4nOicnKSsKICAgICAgJzwvZGl2Pic7CiAgfQogIC8vIEZldGNoIGZ1bGwgZGV0YWlsIHRoZW4gcmVuZGVyCiAgZmV0Y2hEZXRhaWwobm0pLnRoZW4oZnVuY3Rpb24oKXsKICAgIGlmKFNFTD09PW5tKXsKICAgICAgcmVuZGVyUGFuZWwobm0pOwogICAgICAvLyBVcGRhdGUganVzdCB0aGlzIHN0YXRlJ3MgbWFwIGNvbG9yIHRvIG1hdGNoIHRoZSBwYW5lbAogICAgICB2YXIgcGF0aD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK25tKyciXScpOwogICAgICBpZihwYXRoJiZsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICB2YXIgZG9tPWxpdmUuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGxpdmUuZW1vdGlvbnN8fHt9KTsKICAgICAgICBpZihkb20mJmVNYXBbZG9tXSkgcGF0aC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGVNYXBbZG9tXSk7CiAgICAgIH0gZWxzZSB7CiAgICAgICAgYXBwbHlMYXllcigpOwogICAgICB9CiAgICB9CiAgfSkuY2F0Y2goZnVuY3Rpb24oZSl7CiAgICBjb25zb2xlLndhcm4oJ1tzZWxlY3RdJyxlKTsKICAgIGlmKFNFTD09PW5tKSByZW5kZXJQYW5lbChubSk7CiAgfSk7Cn0KCmZ1bmN0aW9uIHJlbmRlclBhbmVsKG5tKXsKICB2YXIgZD1nKG5tKTsKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKCFwYW5lbCkgcmV0dXJuOwogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICB2YXIgaGVhZGVyPQogICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICc8ZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLWVrIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+JysKICAgICAgICAgIChsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKwogICAgICAgICAgKGQuY29uZmlkZW5jZT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/J3JnYmEoNTEsMjA0LDEwMiwwLjEpJzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8ncmdiYSgyMjQsOTAsNDAsMC4xKSc6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknKSsnO2NvbG9yOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8nIzMzY2M2Nic6ZC5jb25maWRlbmNlPT09J01FRElVTSc/JyNlMDVhMjgnOidyZ2JhKDI1NSwyNTUsMjU1LDAuMyknKSsnIj4nK2QuY29uZmlkZW5jZSsnIFNJR05BTDwvc3Bhbj4nOicnKSsKICAgICAgICAgIChkLmlzX3JlZ2lvbmFsX3N0b3J5Pyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpO2NvbG9yOiMzYmI4ZDgiPlJFR0lPTkFMIFNQSUtFPC9zcGFuPic6JycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICc8L2J1dHRvbj4nKwogICAgJzwvZGl2Pic7CgogIHZhciBib2R5PScnOwoKICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgIHZhciBkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPyd1cCc6J2RuJzsKICAgIHZhciBuYXJyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdGw9KGQudGltZWxpbmUmJmQudGltZWxpbmUubGVuZ3RoKT9kLnRpbWVsaW5lOlswLDAsMCwwLDAsMCwwLGQuYXR0ZW50aW9ufHwwXTsKICAgIHZhciB0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgICB2YXIgdHc9MjYwLHRoPTYyLHRwPTU7CiAgICB2YXIgcHRzPXRsLm1hcChmdW5jdGlvbih2LGkpe3JldHVyblt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICAgIHZhciBwRD1wdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuKGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pLmpvaW4oJycpOwogICAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICAgIHZhciBhYz1hQyhkLmF0dGVudGlvbnx8MCk7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ0hvdyBpbnRlbnNlbHkgJysobm0uc3BsaXQoJyAnKVswXSkrJyBpcyBiZWluZyBkaXNjdXNzZWQgbmF0aW9uYWxseS4gU2NvcmUgb2YgJytkLmF0dGVudGlvbisnIG1lYW5zICcrKGQuYXR0ZW50aW9uPjYwPyd2ZXJ5IGhpZ2gg4oCUIHRoaXMgc3RhdGUgZG9taW5hdGVzIG5hdGlvbmFsIGRpc2NvdXJzZSc6ZC5hdHRlbnRpb24+MzU/J2VsZXZhdGVkIOKAlCBjbGVhcmx5IGluIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24nOmQuYXR0ZW50aW9uPjE1Pydtb2RlcmF0ZSDigJQgc29tZSBuYXRpb25hbCBjb3ZlcmFnZSc6ZC5hdHRlbnRpb24+NT8nbG93IOKAlCBsaW1pdGVkIG5hdGlvbmFsIGF0dGVudGlvbic6J21pbmltYWwg4oCUIGZldyBzaWduYWxzIGRldGVjdGVkJykrJy4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0iaW5zaWdodCIgc3R5bGU9IicrKGQuY29uZmlkZW5jZT09PSJMT1ciPydib3JkZXItY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjA2KTtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTppdGFsaWMnOicnKSsnIj4nKygoZC5jb25maWRlbmNlPT09IkxPVyImJiFkLnN1bW1hcnkpPydMaW1pdGVkIHNpZ25hbHMgZGV0ZWN0ZWQgZm9yICcrbm0rJy4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLic6ZC5zdW1tYXJ5fHwnQ29sbGVjdGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKSsnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj4yNGggc2hpZnQ8L2Rpdj48ZGl2IGNsYXNzPSJzcy1kZWx0YSAnK2RDKyciPicrZFMrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VG9wIG5hcnJhdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNzLW5hciI+JysobmFyclswXT9uYXJyWzBdLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbmFyclswXS5uYW1lLnNsaWNlKDEpOifigJQnKSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPk5hcnJhdGl2ZSBicmVha2Rvd248L2Rpdj4nKwogICAgICAgIChuYXJyLmxlbmd0aD8KICAgICAgICAgICc8ZGl2IGNsYXNzPSJuYXItbGlzdCI+JytuYXJyLm1hcChmdW5jdGlvbihuKXsKICAgICAgICAgICAgdmFyIG5uPW4ubmFtZT9uLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpOm4ubmFtZTsKICAgICAgICAgICAgdmFyIHZhbD10eXBlb2Ygbi52YWw9PT0nbnVtYmVyJz9uLnZhbDowOwogICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtMiI+PGRpdiBjbGFzcz0ibmktbGFiZWwiPicrbm4rKG4uZGlyPT09J3VwJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojZTA1YTI4O2ZvbnQtc2l6ZTo5cHgiIHRpdGxlPSJnYWluaW5nIHRyYWN0aW9uIj7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojM2JiOGQ4O2ZvbnQtc2l6ZTo5cHgiIHRpdGxlPSJyZXRyZWF0aW5nIj7ihpM8L3NwYW4+JzonJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXZhbCI+Jyt2YWwudG9GaXhlZCgxKSsnJTwvZGl2PicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsdmFsKjIuNSkrJyU7YmFja2dyb3VuZDonKyhuLmRpcj09PSd1cCc/JyNlMDVhMjgnOm4uZGlyPT09J2Rvd24nPycjM2JiOGQ4JzonIzMzNDQ1NScpKyciPjwvZGl2PjwvZGl2PjwvZGl2Pic7CiAgICAgICAgICB9KS5qb2luKCcnKSsnPC9kaXY+JzoKICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG93LXNpZ25hbCByZWdpb24uIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy48L2Rpdj4nKSsKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkF0dGVudGlvbiDigJQgOCBkYXlzPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0idGwtd3JhcCI+PHN2ZyB2aWV3Qm94PSIwIDAgJyt0dysnICcrdGgrJyIgc3R5bGU9IndpZHRoOjEwMCU7aGVpZ2h0OjEwMCUiPicrCiAgICAgICAgICAnPGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJ0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJyIgeDE9IjAiIHgyPSIwIiB5MT0iMCIgeTI9IjEiPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMC4yNSIvPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwIi8+JysKICAgICAgICAgICc8L2xpbmVhckdyYWRpZW50PjwvZGVmcz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrYUQrJyIgZmlsbD0idXJsKCN0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJykiIC8+JysKICAgICAgICAgICc8cGF0aCBkPSInK3BEKyciIGZpbGw9Im5vbmUiIHN0cm9rZT0iJythYysnIiBzdHJva2Utd2lkdGg9IjEuMiIvPicrCiAgICAgICAgICBwdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuICc8Y2lyY2xlIGN4PSInK3BbMF0rJyIgY3k9IicrcFsxXSsnIiByPSInKyhpPT09cHRzLmxlbmd0aC0xPzIuMjoxLjIpKyciIGZpbGw9IicrYWMrJyIvPic7fSkuam9pbignJykrCiAgICAgICAgJzwvc3ZnPjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWxzIDxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoP2QuYXJ0aWNsZXMubGVuZ3RoOjApKyc8L3NwYW4+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0iYXJ0LWxpc3QiPicrCiAgICAgICAgICAoKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoKT8KICAgICAgICAgICAgZC5hcnRpY2xlcy5tYXAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIHR4dD1hLnR4dHx8YS50aXRsZXx8Jyc7CiAgICAgICAgICAgICAgdmFyIHNyYz1hLnNyY3x8Jyc7CiAgICAgICAgICAgICAgLy8gU2tpcCBlbXB0eSBvciB2ZXJ5IHNob3J0IHRpdGxlcwogICAgICAgICAgICAgIGlmKHR4dC5sZW5ndGg8MjUpIHJldHVybiBudWxsOwogICAgICAgICAgICAgIC8vIFNraXAgWW91VHViZSBlbnRpcmVseQogICAgICAgICAgICAgIGlmKHNyYy5pbmRleE9mKCd5b3V0dWJlJyk+PTApIHJldHVybiBudWxsOwogICAgICAgICAgICAgIC8vIFNraXAgbm9pc2Uga2V5d29yZHMKICAgICAgICAgICAgICB2YXIgdGw9dHh0LnRvTG93ZXJDYXNlKCk7CiAgICAgICAgICAgICAgdmFyIG5vaXNlS3c9Wyd0cnVtcCcsJ3VrcmFpbmUnLCdydXNzaWEnLCdnYXphJywncmVjaXBlJywnaG9yb3Njb3BlJywnY2VsZWJyaXR5JywnYm94IG9mZmljZScsJ211c2ljIHZpZGVvJywnbGl2ZSBzY29yZScsJ2NyaWNrZXQgc2NvcmUnLCd3YXRjaDonLCdwaG90b3M6JywnYnJlYWtpbmc6J107CiAgICAgICAgICAgICAgaWYobm9pc2VLdy5zb21lKGZ1bmN0aW9uKGspe3JldHVybiB0bC5pbmRleE9mKGspPj0wO30pKSByZXR1cm4gbnVsbDsKICAgICAgICAgICAgICAvLyBTb3VyY2UgbGFiZWwg4oCUIGJhY2tlbmQgYWxyZWFkeSBjbGVhbmVkIHRoaXMKICAgICAgICAgICAgICAvLyBJZiBlbXB0eSAobmF0aW9uYWwgbWVkaWEgaGlkZGVuKSwgc2hvdyBubyBzb3VyY2UgbGFiZWwKICAgICAgICAgICAgICB2YXIgc3JjSHRtbD1zcmM/JzxkaXYgY2xhc3M9ImFydC1zcmMiPicrc3JjKyc8L2Rpdj4nOicnOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0iYXJ0LWl0ZW0iPicrc3JjSHRtbCsnPGRpdiBjbGFzcz0iYXJ0LXR4dCI+Jyt0eHQrJzwvZGl2PjwvZGl2Pic7CiAgICAgICAgICAgIH0pLmZpbHRlcihCb29sZWFuKS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjZweCAwIj5ObyBzaWduYWxzIGNvbGxlY3RlZCB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgLy8gVXNlIHNhbWUgZnVuY3Rpb25zIGFzIG1hcCDigJQgZ3VhcmFudGVlZCB0byBtYXRjaAogICAgdmFyIG1hcERvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgIHZhciBicmVha2Rvd249Z2V0RW1vdGlvbkJyZWFrZG93bihubSk7CiAgICB2YXIgZW1vdGlvbnM9YnJlYWtkb3duLmVtb3Rpb25zOwogICAgdmFyIGhhc0Vtb3M9IWJyZWFrZG93bi5lc3RpbWF0ZWQ7CiAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vdGlvbnMpOwogICAgdmFyIGVUb3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICBpZihlVG90PjAmJmVUb3Q8PTEuMDEpe2VMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7fQogICAgdmFyIHRvdD1NYXRoLm1heCgxLGVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApKTsKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgaWYoIWVMLmxlbmd0aCl7cGFuZWwuaW5uZXJIVE1MPWhlYWRlcisnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweCI+Tm8gZW1vdGlvbiBkYXRhIHlldC48L2Rpdj4nO3JldHVybjt9CiAgICAvLyBkb21FbW8gPSBzYW1lIGFzIG1hcCBjb2xvciAoZnJvbSBnZXRFZmZlY3RpdmVFbW90aW9uKQogICAgdmFyIGRvbUVtbz1tYXBEb21FbW87CgogICAgLy8gQ29udGV4dHVhbCBlbW90aW9uIHJlYXNvbiDigJQgc3RhdGUtc3BlY2lmaWMsIG5hcnJhdGl2ZS1kcml2ZW4KICAgIC8vIFN0YXRlLXNwZWNpZmljIG92ZXJyaWRlcyBmb3IgY29udGV4dHVhbCBhY2N1cmFjeQogICAgdmFyIF9zdGF0ZUVtb0NvbnRleHQ9ewogICAgICAnSmFtbXUgYW5kIEthc2htaXInOiB7YW5nZXI6J1NlY3VyaXR5IGluY2lkZW50cyBhbmQgcG9saXRpY2FsIHRlbnNpb25zIHJ1bm5pbmcgaGlnaCcsIGZlYXI6J09uZ29pbmcgc2VjdXJpdHkgc2l0dWF0aW9uIGtlZXBpbmcgYW54aWV0eSBlbGV2YXRlZCcsIGFueGlldHk6J1VuY2VydGFpbnR5IGFyb3VuZCBwb2xpdGljYWwgc3RhdHVzIGFuZCBzZWN1cml0eSd9LAogICAgICAnTWFuaXB1cic6ICAgICAgICAgICB7YW5nZXI6J0V0aG5pYyBjb25mbGljdCBhbmQgdmlvbGVuY2UgZHJpdmluZyBpbnRlbnNlIHB1YmxpYyBmcnVzdHJhdGlvbicsIGZlYXI6J09uZ29pbmcgY29tbXVuYWwgY29uZmxpY3QgbWFraW5nIGNvbW11bml0aWVzIGZlYXJmdWwnLCBhbnhpZXR5OidQcm9sb25nZWQgZXRobmljIHRlbnNpb25zIGNyZWF0aW5nIGRlZXAgdW5jZXJ0YWludHknfSwKICAgICAgJ1B1bmphYic6ICAgICAgICAgICAge2FuZ2VyOidQb2xpdGljYWwgZGlzcHV0ZXMgYW5kIGFncmFyaWFuIGRpc3RyZXNzIGZ1ZWxsaW5nIGZydXN0cmF0aW9uJywgYW54aWV0eTonRWNvbm9taWMgcHJlc3N1cmVzIG9uIGZhcm1pbmcgY29tbXVuaXRpZXMgY3JlYXRpbmcgY29uY2Vybid9LAogICAgICAnV2VzdCBCZW5nYWwnOiAgICAgICB7YW5nZXI6J1BvbGl0aWNhbCB2aW9sZW5jZSBhbmQgcGFydHkgcml2YWxyeSBnZW5lcmF0aW5nIG91dHJhZ2UnLCBhbnhpZXR5OidQb2xpdGljYWwgdW5jZXJ0YWludHkgYW5kIGdvdmVybmFuY2UgY29uY2VybnMnfSwKICAgICAgJ1V0dGFyIFByYWRlc2gnOiAgICAge2FuZ2VyOidMYXcgYW5kIG9yZGVyIGluY2lkZW50cyBhbmQgcG9saXRpY2FsIGRpc3B1dGVzIGRyaXZpbmcgYW5nZXInLCBhbnhpZXR5OidFY29ub21pYyBjb25jZXJucyBhbmQgZ292ZXJuYW5jZSBnYXBzIGNyZWF0aW5nIHVuZWFzZSd9LAogICAgICAnRGVsaGknOiAgICAgICAgICAgICB7YW5nZXI6J0dvdmVybmFuY2UgZGlzcHV0ZXMgYW5kIHBvbGl0aWNhbCBjbGFzaGVzIGRyaXZpbmcgZnJ1c3RyYXRpb24nLCBhbnhpZXR5OidBaXIgcXVhbGl0eSwgZ292ZXJuYW5jZSBhbmQgcG9saXRpY2FsIHVuY2VydGFpbnR5J30sCiAgICB9OwogICAgLy8gVXNlIHN0YXRlLXNwZWNpZmljIGNvbnRleHQgaWYgYXZhaWxhYmxlIGZvciBkb21pbmFudCBlbW90aW9uCiAgICB2YXIgX3N0YXRlU3BlY2lmaWM9KF9zdGF0ZUVtb0NvbnRleHRbbm1dfHx7fSlbZG9tRW1vXXx8bnVsbDsKCiAgICB2YXIgX2Vtb1JlYXNvbnM9ewogICAgICBhbmdlcjp7CiAgICAgICAgJ2JvcmRlciBpc3N1ZXMnOiAgICdCb3JkZXIgdGVuc2lvbnMgYW5kIHNlY3VyaXR5IGluY2lkZW50cyBmdWVsbGluZyBwdWJsaWMgZnJ1c3RyYXRpb24nLAogICAgICAgICdsYXcgJiBvcmRlcic6ICAgICAnQ3JpbWUgYW5kIGxhdyBlbmZvcmNlbWVudCBpbmNpZGVudHMgZ2VuZXJhdGluZyBzdHJvbmcgcHVibGljIGFuZ2VyJywKICAgICAgICAnY29ycnVwdGlvbic6ICAgICAgJ1NjYW0gZXhwb3N1cmUgYW5kIGdvdmVybmFuY2UgZmFpbHVyZXMgZnVlbGxpbmcgb3V0cmFnZScsCiAgICAgICAgJ2VsZWN0aW9ucyc6ICAgICAgICdFbGVjdG9yYWwgZGlzcHV0ZXMgYW5kIHBvbGl0aWNhbCByaXZhbHJpZXMgaW50ZW5zaWZ5aW5nIHB1YmxpYyBhbmdlcicsCiAgICAgICAgJ3Byb3Rlc3QnOiAgICAgICAgICdBY3RpdmUgc3RyZWV0IHByb3Rlc3RzIGFuZCBhZ2l0YXRpb25zIGRyaXZpbmcgZGlzY291cnNlJywKICAgICAgICAnZ292ZXJuYW5jZSc6ICAgICAgJ0FkbWluaXN0cmF0aXZlIGZhaWx1cmVzIGFuZCBwb2xpY3kgZGlzcHV0ZXMgZHJhd2luZyBhbmdlcicsCiAgICAgICAgJ2Nhc3RlJzogICAgICAgICAgICdDYXN0ZSBkaXNjcmltaW5hdGlvbiBpbmNpZGVudHMgc3Rva2luZyBjb21tdW5pdHkgdGVuc2lvbnMnLAogICAgICAgICdyZWxpZ2lvbic6ICAgICAgICAnQ29tbXVuYWwgdGVuc2lvbnMgZ2VuZXJhdGluZyBzdHJvbmcgZW1vdGlvbmFsIHJlYWN0aW9ucycsCiAgICAgICAgJ2Zhcm1lciBpc3N1ZXMnOiAgICdBZ3JhcmlhbiBkaXN0cmVzcyBkcml2aW5nIGZhcm1lciBhZ2l0YXRpb24nLAogICAgICAgICdzZWN1cml0eSc6ICAgICAgICAnU2VjdXJpdHkgaW5jaWRlbnRzIGZ1ZWxsaW5nIGZlYXIgYW5kIGFuZ2VyJywKICAgICAgfSwKICAgICAgYW54aWV0eTp7CiAgICAgICAgJ2Vjb25vbXknOiAgICAgICAgICdFY29ub21pYyB1bmNlcnRhaW50eSBjcmVhdGluZyB3aWRlc3ByZWFkIGFwcHJlaGVuc2lvbicsCiAgICAgICAgJ2luZmxhdGlvbic6ICAgICAgICdSaXNpbmcgcHJpY2VzIGVyb2RpbmcgaG91c2Vob2xkIGNvbmZpZGVuY2UnLAogICAgICAgICd1bmVtcGxveW1lbnQnOiAgICAnSm9iIG1hcmtldCBjb25jZXJucyBnZW5lcmF0aW5nIGFueGlldHkgYWNyb3NzIHRoZSBzdGF0ZScsCiAgICAgICAgJ2JvcmRlciBpc3N1ZXMnOiAgICdCb3JkZXIgdGVuc2lvbnMgY3JlYXRpbmcgc2VjdXJpdHkgYW54aWV0eScsCiAgICAgICAgJ2Vudmlyb25tZW50JzogICAgICdFbnZpcm9ubWVudGFsIGNyaXNpcyB0cmlnZ2VyaW5nIHB1YmxpYyBjb25jZXJuJywKICAgICAgICAnZmFybWVyIGlzc3Vlcyc6ICAgJ0Nyb3AgZGlzdHJlc3MgYW5kIG1vbnNvb24gdW5jZXJ0YWludHkgY3JlYXRpbmcgYW54aWV0eScsCiAgICAgICAgJ2hlYWx0aCc6ICAgICAgICAgICdIZWFsdGggZW1lcmdlbmN5IHNpZ25hbHMgZWxldmF0aW5nIHB1YmxpYyBjb25jZXJuJywKICAgICAgICAnZ292ZXJuYW5jZSc6ICAgICAgJ1BvbGljeSB1bmNlcnRhaW50eSBnZW5lcmF0aW5nIGluc3RpdHV0aW9uYWwgYW54aWV0eScsCiAgICAgICAgJ3NlY3VyaXR5JzogICAgICAgICdTZWN1cml0eSBzaXR1YXRpb24gY3JlYXRpbmcgdW5kZXJseWluZyBmZWFyJywKICAgICAgfSwKICAgICAgaG9wZTp7CiAgICAgICAgJ2VsZWN0aW9ucyc6ICAgICAgICdFbGVjdG9yYWwgbW9tZW50dW0gZ2VuZXJhdGluZyBvcHRpbWlzbSBmb3IgcG9saXRpY2FsIGNoYW5nZScsCiAgICAgICAgJ2Vjb25vbXknOiAgICAgICAgICdFY29ub21pYyBpbmRpY2F0b3JzIHNob3dpbmcgZWFybHkgcmVjb3Zlcnkgc2lnbmFscycsCiAgICAgICAgJ2dvdmVybmFuY2UnOiAgICAgICdQb2xpY3kgYW5ub3VuY2VtZW50cyBjcmVhdGluZyBjYXV0aW91cyBvcHRpbWlzbScsCiAgICAgICAgJ2luZnJhc3RydWN0dXJlJzogICdJbmZyYXN0cnVjdHVyZSBkZXZlbG9wbWVudCBnZW5lcmF0aW5nIGRldmVsb3BtZW50IGhvcGVzJywKICAgICAgICAnZWR1Y2F0aW9uJzogICAgICAgJ0VkdWNhdGlvbiByZWZvcm1zIGJ1aWxkaW5nIGV4cGVjdGF0aW9ucyBmb3IgY2hhbmdlJywKICAgICAgfSwKICAgICAgZmVhcjp7CiAgICAgICAgJ3NlY3VyaXR5JzogICAgICAgICdTZWN1cml0eSBpbmNpZGVudHMgY3JlYXRpbmcgZmVhciBhY3Jvc3MgY29tbXVuaXRpZXMnLAogICAgICAgICdib3JkZXIgaXNzdWVzJzogICAnQm9yZGVyIHNpdHVhdGlvbiBnZW5lcmF0aW5nIGZlYXIgb2YgZXNjYWxhdGlvbicsCiAgICAgICAgJ2xhdyAmIG9yZGVyJzogICAgICdDcmltZSBwYXR0ZXJucyBjcmVhdGluZyBwdWJsaWMgc2FmZXR5IGNvbmNlcm5zJywKICAgICAgICAnaGVhbHRoJzogICAgICAgICAgJ0Rpc2Vhc2Ugc2lnbmFscyBnZW5lcmF0aW5nIHB1YmxpYyBoZWFsdGggYW54aWV0eScsCiAgICAgICAgJ2Vudmlyb25tZW50JzogICAgICdFbnZpcm9ubWVudGFsIHRocmVhdHMgY3JlYXRpbmcgZmVhciBvZiBkaXNhc3RlcicsCiAgICAgICAgJ3JlbGlnaW9uJzogICAgICAgICdDb21tdW5hbCB0ZW5zaW9ucyBjcmVhdGluZyBmZWFyIG9mIHZpb2xlbmNlJywKICAgICAgfSwKICAgICAgcHJpZGU6ewogICAgICAgICduYXRpb25hbGlzbSc6ICAgICAnTmF0aW9uYWwgc2VudGltZW50IGFuZCBwYXRyaW90aWMgZGlzY291cnNlIGF0IGhpZ2ggaW50ZW5zaXR5JywKICAgICAgICAnZWxlY3Rpb25zJzogICAgICAgJ0VsZWN0b3JhbCBtb21lbnR1bSBnZW5lcmF0aW5nIHN0cm9uZyBjb21tdW5pdHkgcHJpZGUnLAogICAgICAgICdyZWxpZ2lvbic6ICAgICAgICAnQ3VsdHVyYWwgYW5kIHJlbGlnaW91cyBjZWxlYnJhdGlvbnMgZHJpdmluZyBwcmlkZSBzaWduYWxzJywKICAgICAgICAnaW5mcmFzdHJ1Y3R1cmUnOiAgJ0RldmVsb3BtZW50IG1pbGVzdG9uZXMgZ2VuZXJhdGluZyByZWdpb25hbCBwcmlkZScsCiAgICAgIH0sCiAgICB9OwogICAgdmFyIF9lbW9DdHg9Jyc7CiAgICB2YXIgX3NkPVNEW25tXXx8e307CiAgICB2YXIgX2RvbU5hcj1fc2QuZG9taW5hbnRfbmFycmF0aXZlfHwnJzsKICAgIC8vIFVzZSBzdGF0ZS1zcGVjaWZpYyBjb250ZXh0IGZpcnN0LCB0aGVuIG5hcnJhdGl2ZS1iYXNlZCwgdGhlbiBnZW5lcmljCiAgICBpZihfc3RhdGVTcGVjaWZpYyl7CiAgICAgIF9lbW9DdHg9X3N0YXRlU3BlY2lmaWM7CiAgICB9IGVsc2UgaWYoZG9tRW1vJiZfZW1vUmVhc29uc1tkb21FbW9dKXsKICAgICAgX2Vtb0N0eD1fZW1vUmVhc29uc1tkb21FbW9dW19kb21OYXJdfHxfZW1vUmVhc29uc1tkb21FbW9dW09iamVjdC5rZXlzKF9lbW9SZWFzb25zW2RvbUVtb10pWzBdXXx8Jyc7CiAgICB9CiAgICAvLyBGYWxsYmFjayBjb250ZXh0IGZyb20gc2lnbmFsIGFydGljbGVzCiAgICBpZighX2Vtb0N0eCYmX3NkLmFydGljbGVzJiZfc2QuYXJ0aWNsZXMubGVuZ3RoKXsKICAgICAgdmFyIF90b3BBcnQ9X3NkLmFydGljbGVzWzBdOwogICAgICBpZihfdG9wQXJ0JiZfdG9wQXJ0LnR4dCkgX2Vtb0N0eD0nU2lnbmFscyBjb25jZW50cmF0ZWQgYXJvdW5kOiAnK190b3BBcnQudHh0LnNsaWNlKDAsODApOwogICAgfQoKICAgIC8vIFJlb3JkZXIgZUwgc28gZG9taW5hbnQgc2hvd3MgZmlyc3QKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXsKICAgICAgaWYoYVswXT09PWRvbUVtbykgcmV0dXJuIC0xOwogICAgICBpZihiWzBdPT09ZG9tRW1vKSByZXR1cm4gMTsKICAgICAgcmV0dXJuIGJbMV0tYVsxXTsKICAgIH0pOwogICAgdmFyIGRvbVBjdD1NYXRoLnJvdW5kKChlTFswXT9lTFswXVsxXToyMCkqMTAwL3RvdCk7CiAgICB2YXIgbmFycjI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0b3BOYXJTdHI9bmFycjIuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCcgYW5kICcpOwogICAgdmFyIHdoYXRJdD17YW54aWV0eTonQSBkaWZmdXNlIHVuZWFzZSBpcyBydW5uaW5nIHRocm91Z2ggc2lnbmFscyBmcm9tICcrbm0rKHRvcE5hclN0cj8nLCBjb25jZW50cmF0ZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFNpZ25hbHMgYXQgdGhpcyBzdGFnZSB0ZW5kIHRvIGJlIGxvY2FsbHkgYWJzb3JiZWQgYmVmb3JlIHdpZGVuaW5nLic6Jy4nICApLGFuZ2VyOidGcnVzdHJhdGlvbiBzaWduYWxzIGFyZSBlbGV2YXRlZCBpbiAnK25tKyh0b3BOYXJTdHI/JywgcGFydGljdWxhcmx5IGFyb3VuZCAnK3RvcE5hclN0cisnLiBUaGUgdG9uZSBzdWdnZXN0cyBwcmVzc3VyZSBidWlsZGluZyByYXRoZXIgdGhhbiBhIHNpbmdsZSBldmVudC4nOicuIFRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgaXMgbm90aWNlYWJseSB0ZW5zZS4nKSxob3BlOidBbiB1bnVzdWFsbHkgb3B0aW1pc3RpYyBzaWduYWwgcmVnaXN0ZXIgZnJvbSAnK25tKyh0b3BOYXJTdHI/Jywgb3JpZW50ZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFdvcnRoIHdhdGNoaW5nIOKAlCBwb3NpdGl2ZSBzaWduYWxzIGF0IHRoaXMgZGVuc2l0eSBhcmUgcmVsYXRpdmVseSByYXJlLic6Jy4gQSBzaWduYWwgd29ydGggbW9uaXRvcmluZy4nKSxwcmlkZTonU3Ryb25nIGlkZW50aXR5IHNpZ25hbHMgaW4gJytubSsodG9wTmFyU3RyPycsIGNlbnRyZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFJlZ2lvbmFsbHkgY29uY2VudHJhdGVkIGFuZCBlbW90aW9uYWxseSBkZW5zZS4nOicuIExvY2FsbHkgY29uY2VudHJhdGVkLCBlbW90aW9uYWxseSBzdHJvbmcuJyksZmVhcjonQXBwcmVoZW5zaW9uIHNpZ25hbHMgaW4gJytubSsodG9wTmFyU3RyPycsIGFyb3VuZCAnK3RvcE5hclN0cisnLiBUaGVzZSB0ZW5kIHRvIGludGVuc2lmeSBiZWZvcmUgYWNoaWV2aW5nIHdpZGVyIHZpc2liaWxpdHkuJzonLiBUaGUgcmVnaXN0ZXIgY2FycmllcyBhbiBlZGdlIHRoYXQgdGVuZHMgdG8gcHJlY2VkZSBsYXJnZXIgY3ljbGVzLicpfTsKICAgIHZhciBjdW1BPS1NYXRoLlBJLzIsY3g9MzgsY3k9MzgsUj0zMyxyaT0yMDsKICAgIHZhciBhcmNzPWVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0sZnI9di90b3QsYTE9Y3VtQSxhMj1jdW1BK2ZyKk1hdGguUEkqMjtjdW1BPWEyOwogICAgICB2YXIgbGc9KGEyLWExKT5NYXRoLlBJPzE6MDsKICAgICAgdmFyIHgxPWN4K01hdGguY29zKGExKSpSLHkxPWN5K01hdGguc2luKGExKSpSLHgyPWN4K01hdGguY29zKGEyKSpSLHkyPWN5K01hdGguc2luKGEyKSpSOwogICAgICB2YXIgeDM9Y3grTWF0aC5jb3MoYTIpKnJpLHkzPWN5K01hdGguc2luKGEyKSpyaSx4ND1jeCtNYXRoLmNvcyhhMSkqcmkseTQ9Y3krTWF0aC5zaW4oYTEpKnJpOwogICAgICByZXR1cm4gJzxwYXRoIGQ9Ik0nK3gxLnRvRml4ZWQoMSkrJywnK3kxLnRvRml4ZWQoMSkrJyBBJytSKycsJytSKycgMCAnK2xnKycgMSAnK3gyLnRvRml4ZWQoMSkrJywnK3kyLnRvRml4ZWQoMSkrJyBMJyt4My50b0ZpeGVkKDEpKycsJyt5My50b0ZpeGVkKDEpKycgQScrcmkrJywnK3JpKycgMCAnK2xnKycgMCAnK3g0LnRvRml4ZWQoMSkrJywnK3k0LnRvRml4ZWQoMSkrJyBaIiBmaWxsPSInK3BhbFtrXSsnIiBvcGFjaXR5PSIwLjkiLz4nOwogICAgfSkuam9pbignJyk7CiAgICB2YXIgZWRlc2M9e2FueGlldHk6J0RpZmZ1c2UgdW5lYXNlLCB3b3JyeSBzaWduYWxzJyxhbmdlcjonRnJ1c3RyYXRpb24sIHByZXNzdXJlIHNpZ25hbHMnLGhvcGU6J09wdGltaXNtLCBmb3J3YXJkIG1vbWVudHVtJyxwcmlkZTonSWRlbnRpdHksIHJlZ2lvbmFsIGFzc2VydGlvbicsZmVhcjonQXBwcmVoZW5zaW9uLCB0aHJlYXQgcGVyY2VwdGlvbid9OwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdUaGUgZW1vdGlvbmFsIHJlZ2lzdGVyIG9mIHNpZ25hbHMgZnJvbSAnK25tKycg4oCUIHdoYXQgdG9uZSBydW5zIHRocm91Z2ggdGhlIGRpc2NvdXJzZSBhbmQgaG93IGNvbmNlbnRyYXRlZCBpdCBpcy4nKwogICAgJzwvZGl2PicrCiAgICAoIWhhc0Vtb3M/JzxkaXYgc3R5bGU9InBhZGRpbmc6NnB4IDExcHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTttYXJnaW4tYm90dG9tOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCkiPkVzdGltYXRlZCBmcm9tIHNpZ25hbCBkaXJlY3Rpb24g4oCUIGxpbWl0ZWQgZGlyZWN0IGVtb3Rpb24gZGF0YS48L2Rpdj4nOicnKSsKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MTRweDtib3JkZXItcmFkaXVzOjEwcHg7YmFja2dyb3VuZDonK3BhbFtkb21FbW9dKycxNDtib3JkZXI6MXB4IHNvbGlkICcrcGFsW2RvbUVtb10rJzMzO21hcmdpbi1ib3R0b206MTJweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOicrcGFsW2RvbUVtb10rJzttYXJnaW4tYm90dG9tOjZweCI+RG9taW5hbnQgZW1vdGlvbjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjZweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+Jytkb21FbW8uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4Ij4nK2RvbVBjdCsnJSDCtyAnK25tKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6OHB4O2xpbmUtaGVpZ2h0OjEuNTtmb250LXN0eWxlOml0YWxpYyI+JysoX2Vtb0N0eHx8d2hhdEl0W2RvbUVtb118fCcnKSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkVtb3Rpb25hbCBicmVha2Rvd248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OyI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCA3NiA3NiIgc3R5bGU9IndpZHRoOjcycHg7aGVpZ2h0OjcycHg7ZmxleC1zaHJpbms6MCI+JythcmNzKyc8L3N2Zz4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHg7Ij4nKwogICAgICAgICAgICBlTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICAgICAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0scGN0PU1hdGgucm91bmQodioxMDAvdG90KTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6Y2VudGVyO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweDsiPjxzcGFuIHN0eWxlPSJ3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjJweDtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMS41cHg7Y29sb3I6Jysoaz09PWRvbUVtbz8ndmFyKC0taW5rKSc6J3ZhcigtLWRpbSknKSsnIj4nK2suY2hhckF0KDApLnRvVXBwZXJDYXNlKCkray5zbGljZSgxKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2YXIoLS1pbmspIj4nK3BjdCsnJTwvc3Bhbj4nKwogICAgICAgICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3BjdCsnJTtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7b3BhY2l0eTowLjc7Ym9yZGVyLXJhZGl1czoxcHgiPjwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgICAoaz09PWRvbUVtbz8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrZWRlc2Nba10rJzwvZGl2Pic6JycpKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFsIGhlYWRsaW5lczwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweDsiPicrCiAgICAgICAgICAoKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoKT8KICAgICAgICAgICAgZC5hcnRpY2xlcy5zbGljZSgwLDUpLm1hcChmdW5jdGlvbihhKXsKICAgICAgICAgICAgICB2YXIgZUNvbG9yPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDo2cHg7cGFkZGluZzo2cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpOyI+JysKICAgICAgICAgICAgICAgIChhLmVtb3Rpb24/JzxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrZUNvbG9yW2EuZW1vdGlvbl0rJztkaXNwbGF5OmlubGluZS1ibG9jazttYXJnaW4tdG9wOjVweDtmbGV4LXNocmluazowIj48L3NwYW4+JzonJykrCiAgICAgICAgICAgICAgICAnPGRpdj48ZGl2IHN0eWxlPSJmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNCI+JysoYS50eHR8fGEudGl0bGV8fCcnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nKyhhLnNyY3x8JycpKyhhLmVtb3Rpb24/JyDCtyAnK2EuZW1vdGlvbjonJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NHB4IDAiPk5vIHNpZ25hbHMgeWV0LjwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKCiAgfSBlbHNlIHsKICAgIHZhciB2ZWw9ZC52ZWxvY2l0eXx8MDsKICAgIHZhciB2ZWxEaXI9dmVsPjAuMTU/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wNT8nUmlzaW5nJzp2ZWw8LTAuMT8nQ29vbGluZyBmYXN0Jzp2ZWw8LTAuMDI/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgdmFyIHZlbENvbD12ZWw+MC4wNT8nI2UwNWEyOCc6dmVsPC0wLjAyPycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICB2YXIgbmFycjM9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciByaXNpbmdOYXJzPW5hcnIzLmZpbHRlcihmdW5jdGlvbihuKXtyZXR1cm4gbi5kaXI9PT0ndXAnO30pOwogICAgdmFyIGZhbGxpbmdOYXJzPW5hcnIzLmZpbHRlcihmdW5jdGlvbihuKXtyZXR1cm4gbi5kaXI9PT0nZG93bic7fSk7CiAgICB2YXIgdG9wTmFyPW5hcnIzLmxlbmd0aD9uYXJyM1swXS5uYW1lOicnOwogICAgdmFyIGRvbUVtb01vbT1kLmRvbWluYW50X2Vtb3Rpb258fCcnOwogICAgdmFyIHNpZ0NvdW50PWQuc2lnbmFsX2NvdW50fHwwOwogICAgdmFyIHNyY0NvdW50PWQuc291cmNlX2NvdW50fHwxOwogICAgdmFyIHRvcEFydHM9KGQuYXJ0aWNsZXN8fFtdKS5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihhKXtyZXR1cm4gYS50eHR8fGEudGl0bGV8fCcnfSkuZmlsdGVyKEJvb2xlYW4pOwoKICAgIC8vIEJ1aWxkIGEgcmljaCwgc3RhdGUtc3BlY2lmaWMgaW50ZXJwcmV0YXRpb24gb2Ygd2hhdCB0aGUgbW9tZW50dW0gbWVhbnMKICAgIGZ1bmN0aW9uIGJ1aWxkTW9tZW50dW1TdG9yeSgpewogICAgICB2YXIgbGluZXM9W107CgogICAgICAvLyBMaW5lIDE6IFdoYXQgaXMgZHJpdmluZyB0aGUgbW92ZW1lbnQg4oCUIHRoZSBXSFkKICAgICAgaWYodmVsPjAuMDUpewogICAgICAgIGlmKHJpc2luZ05hcnMubGVuZ3RoKXsKICAgICAgICAgIHZhciBuYXJOYW1lcz1yaXNpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGVtPicrbi5uYW1lKyc8L2VtPic7fSkuam9pbignIGFuZCAnKTsKICAgICAgICAgIGxpbmVzLnB1c2gobm0rJyBpcyBhdHRyYWN0aW5nIGFjY2VsZXJhdGluZyBhdHRlbnRpb24gYXJvdW5kICcrbmFyTmFtZXMrJy4nKwogICAgICAgICAgICAoc3JjQ291bnQ+Mz8nIFNpZ25hbHMgYXJlIGFycml2aW5nIGZyb20gJytzcmNDb3VudCsnIGRpc3RpbmN0IHNvdXJjZSB0eXBlcyDigJQgcmVnaW9uYWwgcHJlc3MsIHB1YmxpYyBkaXNjb3Vyc2UsIGFuZCBicm9hZGVyIG1lZGlhIOKAlCBzdWdnZXN0aW5nIHRoaXMgaXMgbm90IGEgbG9jYWxpc2VkIGZsYXJlIGJ1dCBhIHdpZGVuaW5nIHN0b3J5Lic6CiAgICAgICAgICAgICcgQ292ZXJhZ2UgaXMgc3RpbGwgY29uc29saWRhdGluZyBhY3Jvc3Mgc291cmNlcy4nKSk7CiAgICAgICAgfSBlbHNlIHsKICAgICAgICAgIGxpbmVzLnB1c2goJ1NpZ25hbCB2b2x1bWUgaW4gJytubSsnIGlzIGNsaW1iaW5nIOKAlCAnK3NpZ0NvdW50Kycgc2lnbmFscyB0cmFja2VkIGluIHRoZSBsYXN0IDQ4IGhvdXJzLCB1cCBzaWduaWZpY2FudGx5IGZyb20gdGhlIHByZXZpb3VzIHdpbmRvdy4nKTsKICAgICAgICB9CiAgICAgIH0gZWxzZSBpZih2ZWw8LTAuMDUpewogICAgICAgIGlmKGZhbGxpbmdOYXJzLmxlbmd0aCl7CiAgICAgICAgICB2YXIgbmFyTmFtZXM9ZmFsbGluZ05hcnMuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZW0+JytuLm5hbWUrJzwvZW0+Jzt9KS5qb2luKCcgYW5kICcpOwogICAgICAgICAgbGluZXMucHVzaCgnQ292ZXJhZ2Ugb2YgJytuYXJOYW1lcysnIGluICcrbm0rJyBpcyBjb250cmFjdGluZy4gVGhlIGRpc2NvdXJzZSBjeWNsZSBhcm91bmQgdGhpcyBuYXJyYXRpdmUgYXBwZWFycyB0byBoYXZlIHBhc3NlZCBpdHMgcGVhayDigJQgc2lnbmFsIGludGVuc2l0eSBpcyBkZWNsaW5pbmcgYW5kIHNvdXJjZXMgYXJlIG1vdmluZyBvbi4nKTsKICAgICAgICB9IGVsc2UgewogICAgICAgICAgbGluZXMucHVzaChubSsnIGlzIGVudGVyaW5nIGEgcXVpZXRlciBwaGFzZS4gQWZ0ZXIgcmVjZW50IGFjdGl2aXR5LCBzaWduYWwgdm9sdW1lIGlzIHJldHJlYXRpbmcg4oCUIG5hdGlvbmFsIGF0dGVudGlvbiBpcyBsaWtlbHkgc2hpZnRpbmcgdG8gb3RoZXIgc3Rvcmllcy4nKTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgbGluZXMucHVzaChubSsnIGlzIGhvbGRpbmcgYSBzdGVhZHkgc2lnbmFsIGJhc2VsaW5lLiAnK3NpZ0NvdW50Kycgc2lnbmFscyB0cmFja2VkIOKAlCBjb25zaXN0ZW50IHByZXNlbmNlIGluIG5hdGlvbmFsIGRpc2NvdXJzZSB3aXRob3V0IGEgZG9taW5hbnQgYWNjZWxlcmF0aW9uIGV2ZW50LicpOwogICAgICB9CgogICAgICAvLyBMaW5lIDI6IFdoYXQgdGhlIGVtb3Rpb25hbCByZWdpc3RlciB0ZWxscyB1cyBhYm91dCB0aGUgV0hZCiAgICAgIGlmKGRvbUVtb01vbSYmdmVsPjAuMDIpewogICAgICAgIHZhciBlbW9DdHg9ewogICAgICAgICAgYW5nZXI6ICdUaGUgZG9taW5hbnQgZW1vdGlvbmFsIHJlZ2lzdGVyIGlzIGFuZ2VyIOKAlCB0aGUgbW9tZW50dW0gaGVyZSBpcyBkcml2ZW4gYnkgcHVibGljIGZydXN0cmF0aW9uLCBub3Qgcm91dGluZSBjb3ZlcmFnZS4gVGhpcyBwYXR0ZXJuIHR5cGljYWxseSBpbmRpY2F0ZXMgYSBnb3Zlcm5hbmNlIG9yIGxhdy1hbmQtb3JkZXIgdHJpZ2dlciB0aGF0IGlzIGdlbmVyYXRpbmcgcmVhY3RpdmUgZGlzY291cnNlLicsCiAgICAgICAgICBhbnhpZXR5OiAnVGhlIHVuZGVybHlpbmcgc2lnbmFsIHRvbmUgaXMgYW54aW91cyDigJQgbW9tZW50dW0gaXMgYnVpbGRpbmcgYXJvdW5kIHVuY2VydGFpbnR5IHJhdGhlciB0aGFuIGEgc2luZ2xlIGV2ZW50LiBFY29ub21pYyBwcmVzc3VyZSwgcG9saWN5IGFtYmlndWl0eSwgb3IgYW4gdW5yZXNvbHZlZCBjcmlzaXMgaXMgbGlrZWx5IHN1c3RhaW5pbmcgdGhlIGF0dGVudGlvbi4nLAogICAgICAgICAgaG9wZTogJ1RoZSBzaWduYWwgdG9uZSBza2V3cyBvcHRpbWlzdGljIOKAlCBtb21lbnR1bSBpcyBiZWluZyBkcml2ZW4gYnkgYSBkZXZlbG9wbWVudCwgYW5ub3VuY2VtZW50LCBvciBpbml0aWF0aXZlIHRoYXQgaXMgZ2VuZXJhdGluZyBwb3NpdGl2ZSByZWdpb25hbCBhdHRlbnRpb24uJywKICAgICAgICAgIGZlYXI6ICdGZWFyIGlzIHRoZSBkb21pbmFudCBzaWduYWwgcmVnaXN0ZXIg4oCUIG1vbWVudHVtIGlzIGJ1aWxkaW5nIGFyb3VuZCBhIHNlY3VyaXR5LCBzYWZldHksIG9yIHRocmVhdC1yZWxhdGVkIHN0b3J5LiBUaGUgYWNjZWxlcmF0aW9uIGhlcmUgd2FycmFudHMgY2xvc2Ugd2F0Y2hpbmcuJywKICAgICAgICAgIHByaWRlOiAnUHJpZGUgc2lnbmFscyBhcmUgZHJpdmluZyB0aGUgbW9tZW50dW0g4oCUIGFuIGFjaGlldmVtZW50LCByZWNvZ25pdGlvbiwgb3IgY3VsdHVyYWwgZXZlbnQgaXMgZ2VuZXJhdGluZyBzdXN0YWluZWQgcG9zaXRpdmUgYXR0ZW50aW9uIGluICcrbm0rJy4nCiAgICAgICAgfTsKICAgICAgICBpZihlbW9DdHhbZG9tRW1vTW9tXSkgbGluZXMucHVzaChlbW9DdHhbZG9tRW1vTW9tXSk7CiAgICAgIH0KCiAgICAgIC8vIExpbmUgMzogV2hhdCB0byB3YXRjaCDigJQgZm9yd2FyZC1sb29raW5nIGludGVycHJldGF0aW9uCiAgICAgIGlmKHZlbD4wLjE1KXsKICAgICAgICBsaW5lcy5wdXNoKCdBdCB0aGlzIGFjY2VsZXJhdGlvbiByYXRlLCAnK25tKycgaXMgbGlrZWx5IHRvIGJlY29tZSBhIG5hdGlvbmFsIGF0dGVudGlvbiBmb2NhbCBwb2ludCB3aXRoaW4gdGhlIG5leHQgMjTigJM0OCBob3Vycy4gU3RhdGVzIHJlYWNoaW5nIHRoaXMgdmVsb2NpdHkgdGhyZXNob2xkIHR5cGljYWxseSBhdHRyYWN0IG1haW5zdHJlYW0gbWVkaWEgYW1wbGlmaWNhdGlvbiBzaG9ydGx5IGFmdGVyLicpOwogICAgICB9IGVsc2UgaWYodmVsPjAuMDUpewogICAgICAgIGxpbmVzLnB1c2goJ0lmIHNpZ25hbCBtb21lbnR1bSBob2xkcywgdGhpcyBzdG9yeSBoYXMgdGhlIHRyYWplY3RvcnkgdG8gYnJlYWsgaW50byBicm9hZGVyIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gTW9uaXRvcmluZyB0aGUgbmV4dCBpbmdlc3QgY3ljbGUgd2lsbCBpbmRpY2F0ZSB3aGV0aGVyIHRoZSBhY2NlbGVyYXRpb24gaXMgc3VzdGFpbmVkIG9yIHBsYXRlYXVpbmcuJyk7CiAgICAgIH0gZWxzZSBpZih2ZWw8LTAuMSl7CiAgICAgICAgbGluZXMucHVzaCgnVGhlIGF0dGVudGlvbiBjeWNsZSBmb3IgJytubSsnIGFwcGVhcnMgdG8gYmUgY29tcGxldGluZy4gVGhpcyBpcyB0eXBpY2FsIHBvc3QtcGVhayBiZWhhdmlvdXIg4oCUIHVubGVzcyBhIG5ldyB0cmlnZ2VyIGVtZXJnZXMsIHNpZ25hbCB2b2x1bWUgd2lsbCBsaWtlbHkgc3RhYmlsaXNlIGF0IGJhc2VsaW5lIHdpdGhpbiB0aGUgbmV4dCBjeWNsZS4nKTsKICAgICAgfSBlbHNlIGlmKHZlbDwtMC4wMil7CiAgICAgICAgbGluZXMucHVzaCgnTW9tZW50dW0gaXMgcmV0cmVhdGluZywgYnV0IG5vdCBjb2xsYXBzZWQuIEEgc2Vjb25kYXJ5IHRyaWdnZXIgY291bGQgcmUtaWduaXRlIGNvdmVyYWdlIOKAlCB3b3J0aCB3YXRjaGluZyBmb3IgZm9sbG93LXVwIGRldmVsb3BtZW50cy4nKTsKICAgICAgfQoKICAgICAgcmV0dXJuIGxpbmVzLmpvaW4oJyAnKTsKICAgIH0KICAgIHZhciBjdHg9YnVpbGRNb21lbnR1bVN0b3J5KCk7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ1NpZ25hbCB2ZWxvY2l0eSBmb3IgJytubSsnIOKAlCB3aGV0aGVyIGF0dGVudGlvbiBpcyBidWlsZGluZywgaG9sZGluZywgb3IgYmVnaW5uaW5nIHRvIHJldHJlYXQgZnJvbSB0aGUgY3VycmVudCBjeWNsZS4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrdmVsQ29sKycxNDtib3JkZXI6MXB4IHNvbGlkICcrdmVsQ29sKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3ZlbENvbCsnO21hcmdpbi1ib3R0b206NnB4Ij5TaWduYWwgbW9tZW50dW08L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo4cHg7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MzJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjonK3ZlbENvbCsnO2ZvbnQtd2VpZ2h0OjUwMCI+Jyt2ZWxEaXIrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42NTttYXJnaW4tdG9wOjZweCI+JytjdHgrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic2NvcmUtc3RyaXAiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5WZWxvY2l0eTwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCIgc3R5bGU9ImZvbnQtc2l6ZToxOHB4O2NvbG9yOicrdmVsQ29sKyciPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj4yNGggzrQ8L2Rpdj48ZGl2IGNsYXNzPSJzcy1kZWx0YSAnKyhkLmRlbHRhPj0wPyd1cCc6J2RuJykrJyI+JysoZC5kZWx0YT49MD8nKyc6JycpKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLW5hciI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgKHJpc2luZ05hcnMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkFjY2VsZXJhdGluZzwvZGl2PicrCiAgICAgICAgcmlzaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDUpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xMikiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspIj4nK3IubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyLm5hbWUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6I2UwNWEyOCI+JytyLnZhbC50b0ZpeGVkKDEpKyclPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJykrJzwvZGl2Pic6JycpKwogICAgICAoZmFsbGluZ05hcnMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkRlY2VsZXJhdGluZzwvZGl2PicrCiAgICAgICAgZmFsbGluZ05hcnMubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6N3B4IDEwcHg7bWFyZ2luLWJvdHRvbTo0cHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDU5LDE4NCwyMTYsMC4xMikiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspIj4nK3IubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyLm5hbWUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6IzNiYjhkOCI+JytyLnZhbC50b0ZpeGVkKDEpKyclPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJykrJzwvZGl2Pic6JycpOwogIH0KCiAgcGFuZWwuaW5uZXJIVE1MPWhlYWRlcitib2R5Owp9CgoKZnVuY3Rpb24gdG9nZ2xlRmF2KG5tKXsKICBpZihGQVZTLmhhcyhubSkpIEZBVlMuZGVsZXRlKG5tKTtlbHNlIEZBVlMuYWRkKG5tKTsKICByZW5kZXJQYW5lbChTRUwpO3JlbmRlckZhdnMoKTsKfQpmdW5jdGlvbiByZW5kZXJGYXZzKCl7CiAgdmFyIHJvdz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmF2LXJvdycpOwogIGlmKCFGQVZTLnNpemUpe3Jvdy5pbm5lckhUTUw9JzxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2Pic7cmV0dXJuO30KICByb3cuaW5uZXJIVE1MPUFycmF5LmZyb20oRkFWUykubWFwKGZ1bmN0aW9uKG5tKXsKICAgIHZhciBkPWcobm0pLGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/JyNlMDVhMjgnOicjM2JiOGQ4JzsKICAgIHZhciB0b3A9ZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCc7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImZhdi1jYXJkIiBvbmNsaWNrPSJzZWxlY3RfKFwnJytubSsnXCcpIj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtaGVhZCI+PHNwYW4gY2xhc3M9ImZjLW5hbWUiPicrbm0rJzwvc3Bhbj48c3BhbiBjbGFzcz0iZmMtc2MiPicrZC5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj5OYXJyYXRpdmU8L3NwYW4+PHNwYW4gY2xhc3M9InYiPicrdG9wKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+MjRoPC9zcGFuPjxzcGFuIGNsYXNzPSJ2IiBzdHlsZT0iY29sb3I6JytkQysnIj4nK2RTK2QuZGVsdGErJzwvc3Bhbj48L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KCmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbihjKXsKICBjLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHgpe3guY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICBjLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO2xheWVyPWMuZGF0YXNldC5sYXllcjthcHBseUxheWVyKCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gdXBkYXRlQ2xvY2soKXsKICB2YXIgbm93PW5ldyBEYXRlKCksaXN0PW5ldyBEYXRlKG5vdy5nZXRUaW1lKCkrbm93LmdldFRpbWV6b25lT2Zmc2V0KCkqNjAwMDArMTk4MDAwMDApOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbG9jaycpLnRleHRDb250ZW50PVN0cmluZyhpc3QuZ2V0SG91cnMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldE1pbnV0ZXMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldFNlY29uZHMoKSkucGFkU3RhcnQoMiwnMCcpKycgSVNUJzsKfQpzZXRJbnRlcnZhbCh1cGRhdGVDbG9jaywxMDAwKTt1cGRhdGVDbG9jaygpOwoKZnVuY3Rpb24gYnVpbGRXSVJTaWduYWxzKCl7CiAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgdmFyIHNyYz1PYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg/TElWRTpTRDsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhzcmMpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLmF0dGVudGlvbnx8MCk+Mzt9KTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKICBlbnRyaWVzLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSk7CgogIHZhciB1c2VkTmFycmF0aXZlcz1bXSx1c2VkU3RhdGVzPVtdOwogIHZhciBzaWduYWxzPVtdOwogIGZ1bmN0aW9uIHVzZWQobmFyLHN0YXRlKXtyZXR1cm4gdXNlZE5hcnJhdGl2ZXMuaW5kZXhPZihuYXIpPj0wfHx1c2VkU3RhdGVzLmluZGV4T2Yoc3RhdGUpPj0wO30KICBmdW5jdGlvbiB1c2UobmFyLHN0YXRlKXtpZihuYXIpdXNlZE5hcnJhdGl2ZXMucHVzaChuYXIpO2lmKHN0YXRlKXVzZWRTdGF0ZXMucHVzaChzdGF0ZSk7fQoKICAvLyAxLiBEb21pbmFudCBzaWduYWwg4oCUIGRpcmVjdCwgZ3JvdW5kZWQKICB2YXIgdG9wPWVudHJpZXNbMF07CiAgaWYodG9wKXsKICAgIHZhciBuYXI9dG9wWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J3BvbGl0aWNhbCBhY3Rpdml0eSc7CiAgICB2YXIgZW1vPXRvcFsxXS5kb21pbmFudF9lbW90aW9uOwogICAgdmFyIGNvbD1lbW8/cGFsW2Vtb106J3ZhcigtLWFjY2VudCknOwogICAgdmFyIHZlbD10b3BbMV0udmVsb2NpdHl8fDA7CiAgICB2YXIgdGFpbD12ZWw+MC4wOD8nLCBhbmQgdGhlIHNpZ25hbCBpcyBzdGlsbCBidWlsZGluZyc6dmVsPC0wLjA0PycsIHRob3VnaCBtb21lbnR1bSBpcyBiZWdpbm5pbmcgdG8gZWFzZSc6Jyc7CiAgICB2YXIgZW1vQ3R4PXthbmdlcjonIOKAlCB3aXRoIGZydXN0cmF0aW9uIGFzIHRoZSBwcmV2YWlsaW5nIHRvbmUnLGFueGlldHk6JyDigJQgdW5kZXJjdXJyZW50IG9mIGFueGlldHkgcnVubmluZyB0aHJvdWdoIHNpZ25hbHMnLGZlYXI6JyDigJQgc2lnbmFscyBjYXJyeWluZyBhbiBlZGdlIG9mIGFwcHJlaGVuc2lvbicsaG9wZTonIOKAlCBhIHJlbGF0aXZlbHkgb3B0aW1pc3RpYyByZWdpc3RlcicscHJpZGU6Jyd9OwogICAgc2lnbmFscy5wdXNoKHtjb2w6Y29sLHRhZzonaGlnaGVzdCBzaWduYWwnLGxvYzp0b3BbMF0sCiAgICAgIHRleHQ6JzxzdHJvbmc+Jyt0b3BbMF0rJzwvc3Ryb25nPiBpcyBnZW5lcmF0aW5nIHRoZSBtb3N0IGF0dGVudGlvbiBuYXRpb25hbGx5IGFyb3VuZCA8ZW0+JytuYXIrJzwvZW0+Jyt0YWlsKyhlbW8/ZW1vQ3R4W2Vtb118fCcnOicnKSxkZWxheTowfSk7CiAgICB1c2UobmFyLHRvcFswXSk7CiAgfQoKICAvLyAyLiBFYXJseSBtb3ZlciDigJQgc29tZXRoaW5nIGJ1aWxkaW5nIGJlZm9yZSBpdCBnb2VzIG5hdGlvbmFsCiAgdmFyIGVhcmx5PWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybihrdlsxXS52ZWxvY2l0eXx8MCk+MC4wNSYmKGt2WzFdLmF0dGVudGlvbnx8MCk8MzUmJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSk7CiAgfSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLnZlbG9jaXR5fHwwKS0oYVsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYoZWFybHkpewogICAgdmFyIGVOYXI9ZWFybHlbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnbG9jYWwgZGV2ZWxvcG1lbnRzJzsKICAgIHZhciBlRW1vPWVhcmx5WzFdLmRvbWluYW50X2Vtb3Rpb247CiAgICBzaWduYWxzLnB1c2goe2NvbDplRW1vP3BhbFtlRW1vXTonI2UwNzgyMCcsdGFnOididWlsZGluZyBzaWduYWwnLGxvYzplYXJseVswXSwKICAgICAgdGV4dDonPGVtPicrZU5hci5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStlTmFyLnNsaWNlKDEpKyc8L2VtPiBzaWduYWxzIGFyZSBnYWluaW5nIHRyYWN0aW9uIGluIDxzdHJvbmc+JytlYXJseVswXSsnPC9zdHJvbmc+IOKAlCBlYXJsaWVyIHRoYW4gbW9zdCBjeWNsZXMgYXQgdGhpcyBzdGFnZScsZGVsYXk6MTYwfSk7CiAgICB1c2UoZU5hcixlYXJseVswXSk7CiAgfQoKICAvLyAzLiBFbW90aW9uYWwgY29uY2VudHJhdGlvbiDigJQgdG9uZSByZWFkLCBub3QgYSBoZWFkbGluZQogIHZhciBlbW9Gb2N1cz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7CiAgICByZXR1cm4ga3ZbMV0uZG9taW5hbnRfZW1vdGlvbiYmIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKSYmKGt2WzFdLmF0dGVudGlvbnx8MCk+NDsKICB9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKGVtb0ZvY3VzKXsKICAgIHZhciBlZk5hcj1lbW9Gb2N1c1sxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdkZXZlbG9wbWVudHMnOwogICAgdmFyIGVmRW1vPWVtb0ZvY3VzWzFdLmRvbWluYW50X2Vtb3Rpb247CiAgICB2YXIgZWZDb2w9cGFsW2VmRW1vXXx8JyM1NTY2NzcnOwogICAgdmFyIGVmUmVhZD17CiAgICAgIGFuZ2VyOidTaWduYWxzIGZyb20gPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiBjYXJyeSBhIG5vdGljZWFibHkgZnJ1c3RyYXRlZCB0b25lIOKAlCB3b3J0aCB3YXRjaGluZycsCiAgICAgIGFueGlldHk6J1RoZXJlIGlzIGEgcXVpZXQgdW5lYXNlIGluIDxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4g4oCUIHNpZ25hbHMgc3VnZ2VzdCB0aGlzIGhhcyBub3QgcGVha2VkIHlldCcsCiAgICAgIGZlYXI6J1NpZ25hbHMgaW4gPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiBjYXJyeSBhbiBlZGdlIOKAlCB0aGUgZW1vdGlvbmFsIHJlZ2lzdGVyIGlzIGFwcHJlaGVuc2l2ZScsCiAgICAgIGhvcGU6J1NvbWV3aGF0IHVudXN1YWxseSwgPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gaXMgc2hvd2luZyBhbiBvcHRpbWlzdGljIHNpZ25hbCByZWdpc3RlciBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+JywKICAgICAgcHJpZGU6JzxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IHNpZ25hbHMgYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiBoYXZlIGEgc3Ryb25nIGlkZW50aXR5IHRvbmUg4oCUIGxvY2FsbHkgY29uY2VudHJhdGVkJwogICAgfTsKICAgIHNpZ25hbHMucHVzaCh7Y29sOmVmQ29sLHRhZzonZW1vdGlvbmFsIHRvbmUnLGxvYzplbW9Gb2N1c1swXSwKICAgICAgdGV4dDplZlJlYWRbZWZFbW9dfHwnU2lnbmFscyBmcm9tIDxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4gYXJlIHdvcnRoIHdhdGNoaW5nJyxkZWxheTozMjB9KTsKICAgIHVzZShlZk5hcixlbW9Gb2N1c1swXSk7CiAgfQoKICAvLyA0LiBDb29saW5nIOKAlCBjeWNsZSBjb21wbGV0aW5nCiAgdmFyIGNvb2xpbmc9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3YpewogICAgcmV0dXJuKGt2WzFdLnZlbG9jaXR5fHwwKTwtMC4wNCYmIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKSYmKGt2WzFdLmF0dGVudGlvbnx8MCk+NTsKICB9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGFbMV0udmVsb2NpdHl8fDApLShiWzFdLnZlbG9jaXR5fHwwKTt9KVswXTsKICBpZihjb29saW5nKXsKICAgIHZhciBjTmFyPWNvb2xpbmdbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwncmVjZW50IGZvY3VzJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOicjM2JiOGQ4Jyx0YWc6J3NpZ25hbCByZXRyZWF0aW5nJyxsb2M6Y29vbGluZ1swXSwKICAgICAgdGV4dDonPGVtPicrY05hci5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStjTmFyLnNsaWNlKDEpKyc8L2VtPiBpbiA8c3Ryb25nPicrY29vbGluZ1swXSsnPC9zdHJvbmc+IGFwcGVhcnMgdG8gYmUgbG9zaW5nIHNpZ25hbCBzdHJlbmd0aCDigJQgdGhlIGN5Y2xlIG1heSBiZSBydW5uaW5nIGl0cyBjb3Vyc2UnLGRlbGF5OjQ2MH0pOwogICAgdXNlKGNOYXIsY29vbGluZ1swXSk7CiAgfQoKICAvLyA1LiBOb3J0aGVhc3Qg4oCUIHNpbXBseSBvYnNlcnZhdGlvbmFsLCBubyBkcmFtYXRpc2F0aW9uCiAgdmFyIG5lU3RhdGVzPVsnTWFuaXB1cicsJ0Fzc2FtJywnTmFnYWxhbmQnLCdNaXpvcmFtJywnTWVnaGFsYXlhJywnQXJ1bmFjaGFsIFByYWRlc2gnLCdUcmlwdXJhJ107CiAgdmFyIG5lQWN0aXZlPW5lU3RhdGVzLmZpbHRlcihmdW5jdGlvbihzKXtyZXR1cm4gc3JjW3NdJiYoc3JjW3NdLmF0dGVudGlvbnx8MCk+MiYmdXNlZFN0YXRlcy5pbmRleE9mKHMpPDA7fSk7CiAgaWYobmVBY3RpdmUubGVuZ3RoPj0yKXsKICAgIHZhciBuZU5hcj0oc3JjW25lQWN0aXZlWzBdXSYmc3JjW25lQWN0aXZlWzBdXS5kb21pbmFudF9uYXJyYXRpdmUpfHwncmVnaW9uYWwgZGV2ZWxvcG1lbnRzJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOidyZ2JhKDE2MCwxOTAsMjMwLDAuNDUpJyx0YWc6J3JlZ2lvbmFsIHNpZ25hbCcsbG9jOidOb3J0aGVhc3QnLAogICAgICB0ZXh0Om5lQWN0aXZlLmxlbmd0aCsnIG5vcnRoZWFzdGVybiBzdGF0ZXMgYXJlIHNob3dpbmcgY29uY2VudHJhdGVkIHNpZ25hbHMgYXJvdW5kIDxlbT4nK25lTmFyKyc8L2VtPiDigJQgYSBwYXR0ZXJuIHRoYXQgdGVuZHMgdG8gcHJlY2VkZSB3aWRlciBuYXRpb25hbCBhdHRlbnRpb24nLGRlbGF5OjU4MH0pOwogIH0KCiAgaWYoIXNpZ25hbHMubGVuZ3RoKSByZXR1cm47CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd3aXItc2lnbmFscycpOwogIGlmKCFlbCkgcmV0dXJuOwogIGVsLmlubmVySFRNTD1zaWduYWxzLm1hcChmdW5jdGlvbihzKXsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbCIgc3R5bGU9ImFuaW1hdGlvbi1kZWxheTonK3MuZGVsYXkrJ21zIj4nKwogICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1iYXIiIHN0eWxlPSJiYWNrZ3JvdW5kOicrcy5jb2wrJyI+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9Indpci1zaWduYWwtY29udGVudCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC10ZXh0Ij4nK3MudGV4dCsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1tZXRhIj4nKwogICAgICAgICAgJzxzcGFuIGNsYXNzPSJ3aXItc2lnbmFsLXRhZyIgc3R5bGU9ImNvbG9yOicrcy5jb2wrJyI+JytzLnRhZysnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gY2xhc3M9Indpci1zaWduYWwtbG9jIj4nK3MubG9jKyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKCi8vIElOSVQg4oCUIHdhaXQgZm9yIERPTQovLyBpIGJ1dHRvbiB0b29sdGlwIOKAlCB1c2VzIGZpeGVkIHBvc2l0aW9uaW5nIHNvIGl0J3MgbmV2ZXIgY2xpcHBlZAooZnVuY3Rpb24oKXsKICB2YXIgdGlwPW51bGw7CiAgZnVuY3Rpb24gc2hvd1RpcChlKXsKICAgIGlmKCF0aXApe3RpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbHRhYi10b29sdGlwJyk7fQogICAgdmFyIHR4dD10aGlzLmdldEF0dHJpYnV0ZSgnZGF0YS10aXAnKTsKICAgIGlmKCF0eHR8fCF0aXApIHJldHVybjsKICAgIHRpcC50ZXh0Q29udGVudD10eHQ7CiAgICB0aXAuY2xhc3NMaXN0LmFkZCgndmlzaWJsZScpOwogICAgdmFyIHJlY3Q9dGhpcy5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgIHZhciB0dz0yNDA7CiAgICB2YXIgbGVmdD1NYXRoLm1pbihyZWN0LmxlZnQsd2luZG93LmlubmVyV2lkdGgtdHctMTApOwogICAgdGlwLnN0eWxlLmxlZnQ9bGVmdCsncHgnOwogICAgdGlwLnN0eWxlLnRvcD0ocmVjdC50b3AtMTAtdGlwLm9mZnNldEhlaWdodHx8cmVjdC50b3AtODApKydweCc7CiAgICAvLyBSZXBvc2l0aW9uIGFmdGVyIHJlbmRlcgogICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGZ1bmN0aW9uKCl7CiAgICAgIHRpcC5zdHlsZS50b3A9KHJlY3QudG9wLXRpcC5vZmZzZXRIZWlnaHQtOCkrJ3B4JzsKICAgIH0pOwogIH0KICBmdW5jdGlvbiBoaWRlVGlwKCl7CiAgICBpZighdGlwKXt0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2x0YWItdG9vbHRpcCcpO30KICAgIGlmKHRpcCkgdGlwLmNsYXNzTGlzdC5yZW1vdmUoJ3Zpc2libGUnKTsKICB9CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VvdmVyJyxmdW5jdGlvbihlKXsKICAgIGlmKGUudGFyZ2V0LmNsYXNzTGlzdC5jb250YWlucygnbHRhYi1pbmZvJykpIHNob3dUaXAuY2FsbChlLnRhcmdldCxlKTsKICB9KTsKICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW91dCcsZnVuY3Rpb24oZSl7CiAgICBpZihlLnRhcmdldC5jbGFzc0xpc3QuY29udGFpbnMoJ2x0YWItaW5mbycpKSBoaWRlVGlwKCk7CiAgfSk7Cn0pKCk7CgpmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7CiAgdmFyIGxkcj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpOwogIGlmKCFsZHJ8fGxkci5fZGlzbWlzc2VkKSByZXR1cm47CiAgbGRyLl9kaXNtaXNzZWQ9dHJ1ZTsKICAvLyBTbW9vdGggZmFkZSBvdXQgb3ZlciAxcwogIGxkci5zdHlsZS50cmFuc2l0aW9uPSdvcGFjaXR5IDFzIGVhc2UnOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICBpZihsZHIpeyBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsgbGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnOyB9CiAgICAvLyBBcHBseSBlbW90aW9uIGxheWVyIGFmdGVyIGxvYWRlciBjbG9zZXMgc28gbWFwIHJlbmRlcnMgY29ycmVjdGx5CiAgICBsYXllcj0nZW1vdGlvbic7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24odCl7dC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIHZhciBlbW9UYWI9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLmx0YWJbZGF0YS1sYXllcj0iZW1vdGlvbiJdJyk7CiAgICBpZihlbW9UYWIpIGVtb1RhYi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgYnVpbGRXSVJTaWduYWxzKCk7CiAgICByZW5kZXJTdHJpcCgnM20nKTsKICB9LCAxMDAwKTsKfQoKCgpmdW5jdGlvbiBpbml0KCl7CiAgcmVuZGVyU3RyaXAoJzNtJyk7CgogIC8vIExvYWQgbWFwIHdpdGggcmV0cnkKICB2YXIgbWFwQXR0ZW1wdHM9MDsKICBmdW5jdGlvbiB0cnlMb2FkTWFwKCl7CiAgICBpZih0eXBlb2YgdG9wb2pzb249PT0ndW5kZWZpbmVkJyl7CiAgICAgIGlmKG1hcEF0dGVtcHRzKys8MTApe3NldFRpbWVvdXQodHJ5TG9hZE1hcCwzMDApO30KICAgICAgcmV0dXJuOwogICAgfQogICAgbG9hZE1hcCgpOwogIH0KICB0cnlMb2FkTWFwKCk7CgogIC8vIExvYWQgZnVsbCBjYWNoZWQgc25hcHNob3Qg4oCUIGxvYWRlciBzaG93cyBtaW5pbXVtIDNzLCBtYXggN3MKICB2YXIgX2xvYWRlck1pblRpbWU9MzAwMDsKICB2YXIgX2xvYWRlclN0YXJ0PURhdGUubm93KCk7CiAgdmFyIF9kYXRhUmVhZHk9ZmFsc2U7CiAgdmFyIF9taW5UaW1lRG9uZT1mYWxzZTsKCiAgZnVuY3Rpb24gdHJ5RGlzbWlzcygpewogICAgaWYoX2RhdGFSZWFkeSYmX21pblRpbWVEb25lKSBkaXNtaXNzTG9hZGVyKCk7CiAgfQoKICAvLyBNaW5pbXVtIDMgc2Vjb25kcyBsb2FkZXIgdGltZQogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAgIF9taW5UaW1lRG9uZT10cnVlOwogICAgdHJ5RGlzbWlzcygpOwogIH0sIF9sb2FkZXJNaW5UaW1lKTsKCiAgLy8gRmV0Y2ggZGF0YQogIGZldGNoRnVsbFNuYXBzaG90KCkudGhlbihmdW5jdGlvbihvayl7CiAgICBpZihvayl7CiAgICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICAgIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtzdGFydFBvbGxpbmcoKTt9LDEwMDApOwogICAgfSBlbHNlIHsKICAgICAgc3RhcnRQb2xsaW5nKCk7CiAgICB9CiAgICBfZGF0YVJlYWR5PXRydWU7CiAgICB0cnlEaXNtaXNzKCk7CiAgfSkuY2F0Y2goZnVuY3Rpb24oKXsKICAgIF9kYXRhUmVhZHk9dHJ1ZTsKICAgIHRyeURpc21pc3MoKTsKICB9KTsKCiAgLy8gU2FmZXR5IGZhbGxiYWNrIOKAlCBkaXNtaXNzIGFmdGVyIDdzIHJlZ2FyZGxlc3MKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7IGlmKCFkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpLl9kaXNtaXNzZWQpIGRpc21pc3NMb2FkZXIoKTsgfSwgNzAwMCk7CgogIC8vIFJldHJ5IG1hcCBpZiBzdGlsbCBlbXB0eQogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDMwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDYwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDUwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKS5jYXRjaChmdW5jdGlvbigpe30pO30sODAwMCk7Cn0KaWYoZG9jdW1lbnQucmVhZHlTdGF0ZT09PSdsb2FkaW5nJyl7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsIGluaXQpOwp9IGVsc2UgewogIC8vIEFscmVhZHkgbG9hZGVkIOKAlCBidXQgd2FpdCBvbmUgdGljayB0byBlbnN1cmUgYWxsIHNjcmlwdHMgcGFyc2VkCiAgc2V0VGltZW91dChpbml0LCAwKTsKfQoKCnNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAvLyBBdXRvLXNlbGVjdCBob3R0ZXN0IHN0YXRlIGZyb20gTElWRSBkYXRhCiAgdmFyIHNyYz1PYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg/TElWRTpTRDsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKHNyYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKXsKICAgIHZhciBlbD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK3RvcFswXSsnIl0nKTsKICAgIGlmKGVsKSBzZWxlY3RfKHRvcFswXSk7CiAgfQp9LDMwMDApOwpzZXRUaW1lb3V0KHJlbmRlckZhdnMsMjQwMCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"

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
    # National top narrative for comparison
    top_all = sorted(nar_all.items(), key=lambda x: x[1], reverse=True)
    nat_top = top_all[0][0] if top_all else None

    regional = []
    for region, states in regions.items():
        rs = [s for s in scored if s.get("name") in states]
        if not rs:
            continue
        na: dict[str, float] = {}
        emo_agg: dict[str, float] = {}
        for s in rs:
            for n in s.get("narratives", []):
                na[n["name"]] = na.get(n["name"], 0) + n.get("val", 0)
            for k, v in s.get("emotions", {}).items():
                emo_agg[k] = emo_agg.get(k, 0) + v
        if not na:
            continue
        top_nars = sorted(na.items(), key=lambda x: x[1], reverse=True)[:3]
        top_n = top_nars[0]
        hs = max(rs, key=lambda s: s.get("attention", 0))
        dom_emo = max(emo_agg.items(), key=lambda x: x[1])[0] if emo_agg else None
        avg_att = round(sum(s.get("attention",0) for s in rs) / max(len(rs),1), 1)
        # Is region's top narrative different from national?
        unique = top_n[0] != nat_top
        regional.append({
            "region":          region,
            "top_narrative":   top_n[0],
            "narratives":      [n[0] for n in top_nars],
            "hottest_state":   hs["name"],
            "attention":       round(hs.get("attention", 0), 1),
            "avg_attention":   avg_att,
            "dominant_emotion": dom_emo,
            "unique_focus":    unique,
            "state_count":     len(rs),
        })
    # Sort by avg attention descending
    regional.sort(key=lambda r: r["avg_attention"], reverse=True)

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
    prompt = f"""You are a political analyst writing a state intelligence brief for a public attention observatory.

Write a 3-4 sentence brief about {state_name} that gives the reader full context on what is happening politically and socially right now.

Use these recent headlines as your source material:
{chr(10).join(f"- {h}" for h in all_headlines[:10])}

Background: {signal_count} public attention signals detected. Dominant themes: {narrative_str}.

Guidelines:
- Sound like an experienced analyst, not a news ticker
- Name actual people, events, and issues — be specific
- Give enough context that someone unfamiliar with this state understands why this matters
- Connect the immediate news to the broader political backdrop if relevant
- 3-4 sentences, no bullet points, no headers
- Do not mention signals, data, or sources — just write the brief
- Start directly with the most significant development

Write only the brief text."""

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


@app.get("/api/history")
async def history(days: int = 7):
    """Historical daily snapshots for all states — powers Replay India."""
    if not HAS_PG or not DB_URL:
        return {"error": "no_database", "message": "Database not configured"}
    data = await get_historical_snapshots(days=min(days, 30))
    return {"states": data, "days": days, "has_data": bool(data)}


@app.get("/api/national-history")
async def national_history(days: int = 7):
    """Daily national narrative for the past N days — powers Replay India."""
    conn = await get_db()
    if not conn:
        return {"error": "no_database", "data": []}
    try:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=min(days, 365))
        # Try the national_narratives table first
        try:
            rows = await conn.fetch("""
                SELECT date, top_narratives, dominant_emotion, hottest_state, total_signals
                FROM national_narratives
                WHERE date >= $1
                ORDER BY date ASC
            """, cutoff)
            if rows:
                return {
                    "data": [{"date": r["date"].isoformat(),
                               "narratives": json.loads(r["top_narratives"]),
                               "emotion": r["dominant_emotion"],
                               "hottest_state": r["hottest_state"],
                               "total_signals": r["total_signals"]} for r in rows],
                    "days": days
                }
        except Exception:
            pass
        # Fallback: derive from daily_snapshots
        rows = await conn.fetch("""
            SELECT date,
                   array_agg(dominant_narrative) as narratives,
                   array_agg(dominant_emotion) as emotions,
                   MAX(attention) as max_att,
                   string_agg(state, ',' ORDER BY attention DESC) as states_ordered
            FROM daily_snapshots
            WHERE date >= $1 AND dominant_narrative IS NOT NULL
            GROUP BY date ORDER BY date ASC
        """, cutoff)
        result = []
        for r in rows:
            nars = [n for n in (r["narratives"] or []) if n]
            nar_counts = {}
            for n in nars: nar_counts[n] = nar_counts.get(n, 0) + 1
            top = sorted(nar_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            emos = [e for e in (r["emotions"] or []) if e]
            emo_counts = {}
            for e in emos: emo_counts[e] = emo_counts.get(e, 0) + 1
            top_emo = max(emo_counts.items(), key=lambda x: x[1])[0] if emo_counts else None
            states = (r["states_ordered"] or "").split(",")
            result.append({
                "date": r["date"].isoformat(),
                "narratives": [{"name": n[0], "val": n[1]} for n in top],
                "emotion": top_emo,
                "hottest_state": states[0] if states else None,
            })
        return {"data": result, "days": days}
    except Exception as e:
        return {"error": str(e), "data": []}
    finally:
        await conn.close()


@app.get("/api/pulse-snapshots")
async def pulse_snapshots():
    """
    6 snapshots of the last 24 hours, one every 4 hours.
    Each snapshot: dominant narrative, hottest state, why it matters, emotion.
    """
    now = datetime.now(timezone.utc)
    snapshots = []

    for i in range(5, -1, -1):  # 6 windows: 20-24h ago, 16-20h, 12-16h, 8-12h, 4-8h, 0-4h
        window_end   = now - timedelta(hours=i * 4)
        window_start = now - timedelta(hours=(i + 1) * 4)
        label = "Just now" if i == 0 else f"{(i)*4}–{(i+1)*4}h ago"

        # Collect signals in this window from RAM
        nar_scores:   dict[str, float] = {}
        emo_scores:   dict[str, float] = {}
        state_scores: dict[str, float] = {}
        articles:     list[str] = []
        sig_count = 0

        for state, sigs in store.signals.items():
            for s in sigs:
                pub = s.get("published_at")
                if not pub or not (window_start <= pub < window_end):
                    continue
                sig_count += 1
                intensity = s.get("intensity", 0.5)
                # Narratives
                for n in s.get("narratives", []):
                    nar_scores[n] = nar_scores.get(n, 0) + intensity
                # Emotions
                for k, v in s.get("emotions", {}).items():
                    emo_scores[k] = emo_scores.get(k, 0) + v * intensity
                # State attention
                state_scores[state] = state_scores.get(state, 0) + intensity
                # Collect headlines
                title = s.get("title", "")
                if title and len(articles) < 5:
                    articles.append(title)

        if sig_count == 0:
            continue

        top_nars   = sorted(nar_scores.items(),   key=lambda x: x[1], reverse=True)[:3]
        top_states = sorted(state_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top_emo    = max(emo_scores.items(), key=lambda x: x[1])[0] if emo_scores else None

        primary_nar   = top_nars[0][0] if top_nars else "general"
        hottest_state = top_states[0][0] if top_states else None
        second_state  = top_states[1][0] if len(top_states) > 1 else None

        # Build a one-sentence insight
        insight = ""
        if primary_nar and hottest_state:
            insight = f"{primary_nar.capitalize()} signals dominated"
            if second_state:
                insight += f", with {hottest_state} and {second_state} generating the most activity"
            else:
                insight += f", concentrated in {hottest_state}"
            if top_emo:
                insight += f". Public tone: {top_emo}."
            else:
                insight += "."

        snapshots.append({
            "label":          label,
            "window_start":   window_start.strftime("%H:%M"),
            "window_end":     window_end.strftime("%H:%M"),
            "primary_narrative": primary_nar,
            "narratives":     [n[0] for n in top_nars],
            "hottest_state":  hottest_state,
            "second_state":   second_state,
            "dominant_emotion": top_emo,
            "signal_count":   sig_count,
            "insight":        insight,
            "headlines":      articles[:3],
        })

    return {"snapshots": snapshots}


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
