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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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
        self.cache_built_at: datetime | None = None
        self.cache_snapshot: dict | None = None
        self.cache_insights: dict | None = None
        self.context_cache: dict = {}
        # 8-slot daily history per state (circular)
        self.history: dict[str, list[float]] = {s: [] for s in INDIAN_STATES}

    NOISE_KEYWORDS = [
        'trump', 'biden', 'ukraine', 'russia', 'gaza', 'israel', 'elon musk',
        'openai', 'chatgpt', 'north korea', 'nato summit', 'us congress',
        'white house', 'pentagon', 'federal reserve', 'wall street',
        'recipe', 'horoscope', 'ipl score', 'box office collection',
    ]

    def add_signal(self, state: str, sig: dict) -> bool:
        """Returns True if the signal was new (not a dupe)."""
        url = sig.get("source_url", "")
 
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
    attention = round(min(99, 100 * math.tanh(normalized / (baseline * 1.5))), 1)
    attention = round(attention * conf_weight, 1)

    # Momentum — compare last 12h vs previous 12h for meaningful velocity
    cutoff_12h = 43200  # 12 hours in seconds
    cutoff_24h = 86400  # 24 hours in seconds
    sigs_recent = [s for s in sigs_48h if (now - s["published_at"]).total_seconds() < cutoff_12h]
    sigs_older  = [s for s in sigs_48h if cutoff_12h <= (now - s["published_at"]).total_seconds() < cutoff_24h]
    recent_count = len(sigs_recent)
    older_count  = len(sigs_older)
    prev_count   = len(sigs_prev)

    delta_24h = round(float(raw_count - prev_count), 1)

    # Velocity = change in signal rate (recent 12h vs older 12h)
    if older_count > 0:
        raw_velocity = (recent_count - older_count) / max(older_count, 1)
    elif prev_count > 0:
        raw_velocity = (raw_count - prev_count) / max(prev_count, 1)
    else:
        raw_velocity = 0.0

    # Normalize with tanh — output is -1 to +1
    velocity = round(math.tanh(raw_velocity * 2), 3)
    if velocity > 0.3 and confidence in ("MEDIUM", "HIGH"):
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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5JbmRpYSBBdHRlbnRpb24gTWFwPC90aXRsZT4KPGxpbmsgcmVsPSJwcmVjb25uZWN0IiBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tIj4KPGxpbmsgcmVsPSJwcmVjb25uZWN0IiBocmVmPSJodHRwczovL2ZvbnRzLmdzdGF0aWMuY29tIiBjcm9zc29yaWdpbj4KPGxpbmsgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbS9jc3MyP2ZhbWlseT1GcmF1bmNlczpvcHN6LGl0YWwsd2dodEA5Li4xNDQsMCwzMDA7OS4uMTQ0LDAsNDAwOzkuLjE0NCwxLDMwMDs5Li4xNDQsMSw0MDAmZmFtaWx5PUpldEJyYWlucytNb25vOndnaHRAMzAwOzQwMCZmYW1pbHk9SW50ZXIrVGlnaHQ6d2dodEAzMDA7NDAwOzUwMCZkaXNwbGF5PXN3YXAiIHJlbD0ic3R5bGVzaGVldCI+CjxzdHlsZT4KOnJvb3R7CiAgLS1iZzojMDUwNzBjOwogIC0tYmcxOiMwOTBkMTU7CiAgLS1iZzI6IzBkMTIyMDsKICAtLXN1cmY6cmdiYSgxNCwyMCwzNCwwLjYpOwogIC0tYm9yZGVyOnJnYmEoMTYwLDE5MCwyMzAsMC4wNik7CiAgLS1ib3JkZXIyOnJnYmEoMTYwLDE5MCwyMzAsMC4xMyk7CiAgLS1pbms6I2RkZTZmNTsKICAtLWRpbTojN2E4ODk5OwogIC0tZmFpbnQ6IzNlNGQ2MDsKICAtLWFjY2VudDojZTA1YTI4OwogIC0tYWNjZW50RGltOnJnYmEoMjI0LDkwLDQwLDAuMTUpOwogIC0tcmlzZTojZTA1YTI4OwogIC0tZmFsbDojM2JiOGQ4OwogIC0tc2VyaWY6J0ZyYXVuY2VzJyxHZW9yZ2lhLHNlcmlmOwogIC0tc2FuczonSW50ZXIgVGlnaHQnLHN5c3RlbS11aSxzYW5zLXNlcmlmOwogIC0tbW9ubzonSmV0QnJhaW5zIE1vbm8nLG1vbm9zcGFjZTsKfQoqe2JveC1zaXppbmc6Ym9yZGVyLWJveDttYXJnaW46MDtwYWRkaW5nOjB9Cmh0bWwsYm9keXtiYWNrZ3JvdW5kOnZhcigtLWJnKTtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO292ZXJmbG93LXg6aGlkZGVuO3Njcm9sbC1iZWhhdmlvcjpzbW9vdGh9CgovKiBhdG1vc3BoZXJpYyBiYWNrZ3JvdW5kICovCmJvZHl7CiAgYmFja2dyb3VuZDoKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDgwJSA1MCUgYXQgNTAlIC0xMCUsIHJnYmEoMjI0LDkwLDQwLDAuMDU1KSAwJSwgdHJhbnNwYXJlbnQgNjAlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDUwJSA0MCUgYXQgMTAlIDYwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDI1KSAwJSwgdHJhbnNwYXJlbnQgNTUlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDYwJSA1MCUgYXQgOTAlIDEwMCUsIHJnYmEoMTQwLDgwLDIwMCwwLjAyKSAwJSwgdHJhbnNwYXJlbnQgNTUlKSwKICAgIHZhcigtLWJnKTsKICBtaW4taGVpZ2h0OjEwMHZoOwp9CmJvZHk6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmZpeGVkO2luc2V0OjA7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7CiAgYmFja2dyb3VuZC1pbWFnZTp1cmwoImRhdGE6aW1hZ2Uvc3ZnK3htbDt1dGY4LDxzdmcgeG1sbnM9J2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJyB3aWR0aD0nMjAwJyBoZWlnaHQ9JzIwMCc+PGZpbHRlciBpZD0nbic+PGZlVHVyYnVsZW5jZSB0eXBlPSdmcmFjdGFsTm9pc2UnIGJhc2VGcmVxdWVuY3k9JzAuODUnIG51bU9jdGF2ZXM9JzInLz48ZmVDb2xvck1hdHJpeCB2YWx1ZXM9JzAgMCAwIDAgMC44NSAwIDAgMCAwIDAuOSAwIDAgMCAwIDEgMCAwIDAgMC4wNCAwJy8+PC9maWx0ZXI+PHJlY3Qgd2lkdGg9JzEwMCUnIGhlaWdodD0nMTAwJScgZmlsdGVyPSd1cmwoJTIzbiknLz48L3N2Zz4iKTsKICBvcGFjaXR5OjAuNDU7bWl4LWJsZW5kLW1vZGU6c29mdC1saWdodDsKfQoKLyogVE9QQkFSICovCi50b3BiYXJ7CiAgcG9zaXRpb246Zml4ZWQ7dG9wOjA7bGVmdDowO3JpZ2h0OjA7ei1pbmRleDoxMDA7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjE0cHggMzZweDsKICBiYWNrZ3JvdW5kOnJnYmEoNSw3LDEyLDAuNzUpOwogIGJhY2tkcm9wLWZpbHRlcjpibHVyKDI4cHgpIHNhdHVyYXRlKDEzMCUpOwogIGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLmJyYW5ke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjExcHg7dGV4dC1kZWNvcmF0aW9uOm5vbmV9Ci5icmFuZC1tYXJrewogIHdpZHRoOjI2cHg7aGVpZ2h0OjI2cHg7Ym9yZGVyLXJhZGl1czo1cHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTM1ZGVnLCNlMDVhMjggMCUsI2MwMzA1MCAxMDAlKTsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47CiAgYm94LXNoYWRvdzowIDAgMTRweCByZ2JhKDIyNCw5MCw0MCwwLjI1KSxpbnNldCAwIDAgMCAxcHggcmdiYSgyNTUsMjU1LDI1NSwwLjEpOwp9Ci5icmFuZC1tYXJrOjphZnRlcnsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDsKICBib3JkZXI6MS41cHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjgpO2JvcmRlci1yYWRpdXM6MnB4OwogIGNsaXAtcGF0aDpwb2x5Z29uKDUwJSAwJSwxMDAlIDM4JSw4MiUgMTAwJSwxOCUgMTAwJSwwJSAzOCUpOwp9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoudG9wYmFyLXJ7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MjBweH0KLmxpdmUtaW5kaWNhdG9yewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdweDsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA1ZW07Cn0KLmxpdmUtZG90e3dpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6IzRhZGU4MDtib3gtc2hhZG93OjAgMCA4cHggcmdiYSg3NCwyMjIsMTI4LDAuNyk7YW5pbWF0aW9uOmxkIDIuNXMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgbGR7MCUsMTAwJXtvcGFjaXR5OjE7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuMzU7dHJhbnNmb3JtOnNjYWxlKDAuOCl9fQouY2xvY2t7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW19CgovKiBIRVJPICovCi5oZXJvewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBwYWRkaW5nOjcycHggMzZweCAwOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKfQouaGVyby1leWVicm93ewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtsZXR0ZXItc3BhY2luZzowLjNlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjIycHg7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweDsKfQouaGVyby1leWVicm93OjpiZWZvcmV7Y29udGVudDonJzt3aWR0aDoxOHB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC43fQouaGVyby10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6aXRhbGljOwogIGZvbnQtc2l6ZTpjbGFtcCgyOHB4LDMuMnZ3LDUycHgpO2xpbmUtaGVpZ2h0OjEuMDU7CiAgbGV0dGVyLXNwYWNpbmc6LTAuMDNlbTttYXgtd2lkdGg6ODIwcHg7Y29sb3I6dmFyKC0taW5rKTsKfQouaGVyby10aXRsZSBzcGFue2ZvbnQtc3R5bGU6bm9ybWFsO2NvbG9yOnJnYmEoMjIxLDIzMCwyNDUsMC41NSl9CgovKiBTSUdOQVRVUkUgSU5TSUdIVCAqLwouc2lnbmF0dXJlLWluc2lnaHR7CiAgbWFyZ2luLXRvcDowOwogIHBhZGRpbmc6MTRweCAyMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNikgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIG1heC13aWR0aDo5MDBweDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47Cn0KLnNpZ25hdHVyZS1pbnNpZ2h0OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgdmFyKC0tYWNjZW50KSwgdHJhbnNwYXJlbnQpOwp9Ci5zaS1sYWJlbHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouc2ktdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE0cHgsMS40dncsMThweCk7CiAgZm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluayk7bGluZS1oZWlnaHQ6MS41O2xldHRlci1zcGFjaW5nOi0wLjAxZW07Cn0KLnNpLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tYWNjZW50KX0KLnNpLXN1YnsKICBtYXJnaW4tdG9wOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTsKICBsZXR0ZXItc3BhY2luZzowLjA0ZW07ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDtmbGV4LXdyYXA6d3JhcDsKfQouc2ktdGFnewogIHBhZGRpbmc6MnB4IDhweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTsKICBmb250LXNpemU6OS41cHg7Cn0KCi8qIE5BUlJBVElWRSBTVFJJUCAqLwoKLnN0cmlwLXRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjRweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQouc3RyaXAtdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xMil9Ci5zdHJpcC10YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbHsKICBmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1iZzEpO3BhZGRpbmc6MDsKfQouc3RyaXAtY29sLWhlYWR7CiAgcGFkZGluZzoxMHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwp9Ci5zdHJpcC1jb2wtaGVhZC5mYWRle2NvbG9yOnZhcigtLWZhbGwpfQouc3RyaXAtY29sLWhlYWQucmlzZTJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5zdHJpcC1jb2wtaGVhZC5zaGlmdHtjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sLWJvZHl7cGFkZGluZzoxMnB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OHB4fQouc3RyaXAtaXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjhweDsKfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5zdHJpcC1ub3Rle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5zdHJpcC1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTtmb250LXNpemU6MTRweDtmbGV4LXNocmluazowfQoKLyogTUFJTiBMQVlPVVQgKi8KLm1haW57CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM2MHB4OwogIGdhcDoxNHB4OwogIG1pbi13aWR0aDowOwp9CgovKiBNQVAgKi8KLm1hcC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBvdmVyZmxvdzpoaWRkZW47cG9zaXRpb246cmVsYXRpdmU7Cn0KLm1hcC1jYXJkOjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA3MCUgNTAlIGF0IDM1JSAwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNSkgMCUsIHRyYW5zcGFyZW50IDYwJSksCiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA1MCUgNDAlIGF0IDgwJSAxMDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMCUsIHRyYW5zcGFyZW50IDYwJSk7Cn0KLm1hcC10b3B7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47CiAgcGFkZGluZzoxMnB4IDE4cHggMDsKfQoubWFwLXRpdGxlLWJsb2NrIC5tdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE3cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtdGl0bGUtYmxvY2sgLm1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5sZWdlbmQtYmFyewogIGhlaWdodDozcHg7d2lkdGg6ODBweDtib3JkZXItcmFkaXVzOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCwjMGUyMDM1LCMxYTU1ODAgMjUlLCM4YTVjMTggNTUlLCNjMDM4MWEgODAlLCNlMDEwMjApOwp9Ci5sYXllci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsKICBwYWRkaW5nOjEwcHggMjBweCA2cHg7Cn0KLmxheWVyLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLmx0YWJze2Rpc3BsYXk6ZmxleDtnYXA6M3B4fQoubHRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6M3B4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5sdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wOCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5sdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9CgoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmc6MTJweCAxNnB4IDE2cHg7Cn0KLm1hcC1pbm5lcntwb3NpdGlvbjpyZWxhdGl2ZTthc3BlY3QtcmF0aW86MS8xO3dpZHRoOjEwMCV9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrO292ZXJmbG93OnZpc2libGV9CgovKiBtYXAgc3RhdGUgc3R5bGVzICovCiNpbmRpYS1tYXAgLnN0YXRlewogIGN1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246ZmlsdGVyIDAuMjVzIGVhc2UsIHN0cm9rZS13aWR0aCAwLjJzIGVhc2UsIHN0cm9rZSAwLjJzIGVhc2U7Cn0KI2luZGlhLW1hcCAuc3RhdGU6aG92ZXJ7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC43KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxcHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjI1KSBkcm9wLXNoYWRvdygwIDAgMTBweCByZ2JhKDI1NSwyNTUsMjU1LDAuMikpOwp9CiNpbmRpYS1tYXAgLnN0YXRlLnNlbGVjdGVkewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuOSkgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MS40cHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjM1KSBkcm9wLXNoYWRvdygwIDAgMTZweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpOwp9CgovKiBhbmltYXRlZCBwdWxzZSByaW5ncyAqLwoucHVsc2UtcmluZ3tmaWxsOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZX0KLnB1bHNlLXJpbmcucDF7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5wdWxzZS1yaW5nLnAye2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IDAuOXMgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHJ7CiAgMCV7cjo0O29wYWNpdHk6MC43O3N0cm9rZS13aWR0aDoxLjJ9CiAgMTAwJXtyOjI2O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4yfQp9CgovKiBhdG1vc3BoZXJpYyBnbG93IGJlaGluZCBob3Qgc3RhdGVzICovCi5zdGF0ZS1nbG93e3BvaW50ZXItZXZlbnRzOm5vbmU7ZmlsbDpub25lfQpAa2V5ZnJhbWVzIGdsb3dQdWxzZXswJSwxMDAle29wYWNpdHk6MC4xMn01MCV7b3BhY2l0eTowLjIyfX0KCi5tYXAtdG9vbHRpcHsKICBwb3NpdGlvbjphYnNvbHV0ZTtwb2ludGVyLWV2ZW50czpub25lOwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC45NSk7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjlweDsKICBwYWRkaW5nOjEycHggMTRweDtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTJzO3otaW5kZXg6MjA7bWluLXdpZHRoOjE3MHB4Owp9Ci50dC1ue2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bWFyZ2luLWJvdHRvbTo4cHg7Y29sb3I6dmFyKC0taW5rKX0KLnR0LXJ7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4fQoudHQtciBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKX0KLnR0LW5hcnsKICBtYXJnaW4tdG9wOjhweDtwYWRkaW5nLXRvcDo4cHg7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTsKfQoudHQtbmFyIHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2Rpc3BsYXk6YmxvY2s7bWFyZ2luLWJvdHRvbToycHh9CgovKiBTVEFURSBQQU5FTCAqLwouc3RhdGUtcGFuZWx7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNnB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE2cHgpOwogIHBhZGRpbmc6MjBweDtvdmVyZmxvdy15OmF1dG87bWF4LWhlaWdodDo3ODBweDsKICBtaW4td2lkdGg6MDtvdmVyZmxvdy14OmhpZGRlbjsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjNweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7CiAgaGVpZ2h0OjEwMCU7bWluLWhlaWdodDozMjBweDt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjMycHggMjBweDsKfQoucGFuZWwtZW1wdHkgc3Zne29wYWNpdHk6MC4xNTttYXJnaW4tYm90dG9tOjE4cHh9Ci5wYW5lbC1lbXB0eSAucGUtdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjhweH0KLnBhbmVsLWVtcHR5IC5wZS1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW07bGluZS1oZWlnaHQ6MS43fQoKLyogc3RhdGUgcGFuZWwgaW50ZXJuYWxzICovCi5zcC1oZWFkewogIGRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIG1hcmdpbi1ib3R0b206MTZweDtwYWRkaW5nLWJvdHRvbToxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnNwLWVre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTtjb2xvcjp2YXIoLS1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NXB4fQouc3AtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI4cHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07bGluZS1oZWlnaHQ6MTtjb2xvcjp2YXIoLS1pbmspfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Y29sb3I6dmFyKC0tZmFpbnQpOwogIHdpZHRoOjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo2cHg7Y3Vyc29yOnBvaW50ZXI7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMThzO3BhZGRpbmc6MDtmbGV4LXNocmluazowOwp9Ci5mYXYtYnRuOmhvdmVye2NvbG9yOnZhcigtLWRpbSk7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5mYXYtYnRuLm9ue2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHh9CgovKiBuYXJyYXRpdmUgdGltZWxpbmUg4oCUIHRoZSBzaWduYXR1cmUgZmVhdHVyZSAqLwoubmFyLXRpbWVsaW5lewogIG1hcmdpbi1ib3R0b206MTZweDsKfQoubnQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLm50LWZsb3d7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nLWxlZnQ6MTZweDsKfQoubnQtZmxvdzo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDo1cHg7dG9wOjZweDtib3R0b206NnB4O3dpZHRoOjFweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx2YXIoLS1ib3JkZXIpKTtvcGFjaXR5OjAuNDsKfQoubnQtc3RlcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweDsKICBwYWRkaW5nOjVweCAwO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5udC1kb3R7CiAgd2lkdGg6MTBweDtoZWlnaHQ6MTBweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4LXNocmluazowOwogIHBvc2l0aW9uOmFic29sdXRlO2xlZnQ6LTE2cHg7dG9wOjdweDsKICBib3JkZXI6MS41cHggc29saWQgY3VycmVudENvbG9yO2JhY2tncm91bmQ6dmFyKC0tYmcpOwp9Ci5udC1zdGVwLnBhc3QgLm50LWRvdHtjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCl9Ci5udC1zdGVwLmN1cnJlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCAxMHB4IHJnYmEoMjI0LDkwLDQwLDAuNSl9Ci5udC1jb250ZW50e2ZsZXg6MX0KLm50LXRvcGlje2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4zfQoubnQtc3RlcC5wYXN0IC5udC10b3BpY3tjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZGltKX0KLm50LXdoZW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9CgovKiBpbnNpZ2h0IGJsb2NrICovCi5pbnNpZ2h0ewogIG1hcmdpbi1ib3R0b206MTRweDsKICBwYWRkaW5nOjEycHggMTRweCAxMnB4IDE2cHg7CiAgYm9yZGVyLWxlZnQ6MS41cHggc29saWQgdmFyKC0tYWNjZW50KTsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDMpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxMy41cHg7Zm9udC1zdHlsZTppdGFsaWM7CiAgY29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU1O2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLyogY29tcGFjdCBzY29yZSBzdHJpcCAqLwouc2NvcmUtc3RyaXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsKICBwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6N3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgbWFyZ2luLWJvdHRvbToxNHB4Owp9Ci5zcy1pdGVte2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjJweH0KLnNzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5zcy12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2NvbG9yOnZhcigtLWluayl9Ci5zcy1kZWx0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjJweCA3cHg7Ym9yZGVyLXJhZGl1czozcHh9Ci5zcy1kZWx0YS51cHtjb2xvcjojZTA2MDMwO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKX0KLnNzLWRlbHRhLmRue2NvbG9yOiMzYmI4ZDg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKX0KLnNzLWRpdmlkZXJ7d2lkdGg6MXB4O2hlaWdodDozMnB4O2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowfQouc3MtbmFye2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDB9Cgouc3Atc2VjdGlvbnttYXJnaW4tYm90dG9tOjE0cHh9Ci5zcC1zZWMtdGl0bGV7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206OXB4Owp9CgovKiBuYXJyYXRpdmVzICovCi5uYXItbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9Ci5uYXItaXRlbTJ7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgYXV0bztnYXA6NnB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLm5pLWxhYmVse2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKX0KLm5pLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5uaS10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi10b3A6LTNweH0KLm5pLWZpbGx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHg7dHJhbnNpdGlvbjp3aWR0aCAwLjdzfQoKLyogbW92ZW1lbnQgKi8KLm12LWdyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDo3cHh9Ci5tdi1ibG9ja3tiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6N3B4O3BhZGRpbmc6OXB4fQoubXYtaHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206N3B4fQoubXYtYmxvY2sudXAgLm12LWh7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tdi1ibG9jay5kbiAubXYtaHtjb2xvcjp2YXIoLS1mYWxsKX0KLm12LWl0e2ZvbnQtc2l6ZToxMC41cHg7cGFkZGluZzo0cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2NvbG9yOnZhcigtLWZhaW50KX0KLm12LWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tdi1pdCBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTFweH0KLm12LWl0IHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweH0KCi8qIGVtb3Rpb24gKi8KLmVtLXJvd3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQouZW0tZG9udXR7d2lkdGg6NzZweDtoZWlnaHQ6NzZweDtmbGV4LXNocmluazowfQouZW0tbGVne2ZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHh9Ci5lbS1pdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVtLXN3e3dpZHRoOjZweDtoZWlnaHQ6NnB4O2JvcmRlci1yYWRpdXM6MnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1ue2ZsZXg6MTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5lbS1we2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmspfQoKLyogdGltZWxpbmUgY2hhcnQgKi8KLnRsLXdyYXB7aGVpZ2h0OjcycHh9CgovKiBhcnRpY2xlcyAqLwouYXJ0LWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4fQouYXJ0LWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzo3cHggOXB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydC1pdGVtOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5hcnQtc3Jje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO3dpZHRoOjQ0cHg7cGFkZGluZy10b3A6MXB4fQouYXJ0LXR4dHtmb250LXNpemU6MTFweDtsaW5lLWhlaWdodDoxLjQ7Y29sb3I6dmFyKC0tZGltKX0KCi8qIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XICovCi5uYXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoxOHB4Owp9Ci5uYXItY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTRweCk7b3ZlcmZsb3c6aGlkZGVuOwp9Ci5uYy1oZWFkewogIHBhZGRpbmc6MTRweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290ewogIHRleHQtYWxpZ246Y2VudGVyO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzoyMHB4IDM2cHggNDBweDttYXgtd2lkdGg6NTgwcHg7bWFyZ2luOjAgYXV0bzsKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7bGluZS1oZWlnaHQ6MS45Owp9CgovKiBhbmltYXRpb25zICovCkBrZXlmcmFtZXMgZmFkZVVwe2Zyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVZKDZweCl9dG97b3BhY2l0eToxO3RyYW5zZm9ybTpub25lfX0KLm1hcC1jYXJkLC5zdGF0ZS1wYW5lbCwubmFyLWNhcmQsLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbjpmYWRlVXAgMC41NXMgY3ViaWMtYmV6aWVyKC4yLC44LC4yLDEpIGJhY2t3YXJkc30KLm5hci1jYXJkOm50aC1jaGlsZCgyKXthbmltYXRpb24tZGVsYXk6MC4wN3N9Ci5uYXItY2FyZDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuMTRzfQouc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uLWRlbGF5OjAuMDVzfQoKQG1lZGlhKG1heC13aWR0aDoxMTAwcHgpewogIC5tYWlue2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgLnN0YXRlLXBhbmVse21heC1oZWlnaHQ6bm9uZX0KICAubmFyLXJvd3tncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQp9Cgouc2MtdmFsLXNte2ZvbnQtc2l6ZTpjbGFtcCgxM3B4LDEuM3Z3LDE2cHgpIWltcG9ydGFudH0KLnNjLWhvdmVyYWJsZXtwb3NpdGlvbjpyZWxhdGl2ZTtjdXJzb3I6ZGVmYXVsdH0KLnNjLXRvb2x0aXB7ZGlzcGxheTpub25lO3Bvc2l0aW9uOmFic29sdXRlO2JvdHRvbTpjYWxjKDEwMCUgKyA4cHgpO2xlZnQ6NTAlO3RyYW5zZm9ybTp0cmFuc2xhdGVYKC01MCUpO2JhY2tncm91bmQ6cmdiYSg4LDEyLDIwLDAuOTcpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4IDE0cHg7d2lkdGg6MjIwcHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU7ei1pbmRleDo5OTk5O3BvaW50ZXItZXZlbnRzOm5vbmU7d2hpdGUtc3BhY2U6bm9ybWFsO3RleHQtYWxpZ246bGVmdDtib3gtc2hhZG93OjAgOHB4IDI0cHggcmdiYSgwLDAsMCwwLjUpfQouc2MtaG92ZXJhYmxlOmhvdmVyIC5zYy10b29sdGlwe2Rpc3BsYXk6YmxvY2t9Ci5zYy10aXAtdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjZweH0KLnNjLXRpcC1yb3d7ZGlzcGxheTpmbGV4O2dhcDo2cHg7bWFyZ2luLWJvdHRvbTo0cHg7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKX0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgY2xhc3M9InRvcGJhciI+CiAgPGRpdiBjbGFzcz0iYnJhbmQiPgogICAgPGRpdiBjbGFzcz0iYnJhbmQtbWFyayI+PC9kaXY+CiAgICA8c3BhbiBjbGFzcz0iYnJhbmQtbmFtZSI+SW5kaWEgQXR0ZW50aW9uIE1hcDwvc3Bhbj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJ0b3BiYXItciI+CiAgICA8ZGl2IGNsYXNzPSJsaXZlLWluZGljYXRvciI+CiAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7YmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpO3BhZGRpbmc6MCAzNnB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOyI+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+U2lnbmFscyB0cmFja2VkPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIHNjLXZhbC1zbSIgaWQ9InNjLXNpZ25hbHMtdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1zaWduYWxzLXN1YiI+bG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIiBvbmNsaWNrPSJzZWxlY3RIb3R0ZXN0KCkiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkhpZ2hlc3QgYXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtaG90dGVzdC10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWFuZ2VyLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtYW5nZXItdGlwIj48L2Rpdj48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIHNjLWhvdmVyYWJsZSI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgcmlzaW5nPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtcmlzaW5nLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtcmlzaW5nLXN1YiI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLXJpc2luZy10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIG5hcnJhdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCBzYy12YWwtc20iIGlkPSJzYy1uYXItdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1uYXItc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtbmFyLXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5MZWFzdCBhY3RpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29sLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbC1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1jb29sLXRpcCI+PC9kaXY+PC9kaXY+CjwvZGl2CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLWhvdHRlc3QiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciIgb25jbGljaz0ic2VsZWN0SG90dGVzdCgpIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5IaWdoZXN0IGF0dGVudGlvbjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPkNsaWNrIHRvIGV4cGxvcmU8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1hbmdlci12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj5PdXRyYWdlICYgcHJvdGVzdCBzaWduYWxzPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIHJpc2luZyBuYXJyYXRpdmU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLW5hcnJhdGl2ZS12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtbmFycmF0aXZlLXN1YiI+TmF0aW9uYWwgc2lnbmFsIHN1cmdlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCBjb29saW5nPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29saW5nLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29saW5nLXN1YiI+U2lnbmFsIGRlY2F5PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPHN0eWxlPgouc3RhdC1jZWxsewogIGZsZXg6MTtwYWRkaW5nOjEwcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2FwOjJweDsKICB0cmFuc2l0aW9uOmJhY2tncm91bmQgMC4xNXM7Cn0KLnN0YXQtY2VsbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci5zdGF0LWRpdnt3aWR0aDoxcHg7YmFja2dyb3VuZDpyZ2JhKDE2MCwxOTAsMjMwLDAuMDcpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luOjhweCAwfQouc2MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO21hcmdpbi10b3A6MXB4fQouc2Mtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQo8L3N0eWxlPgoKPCEtLSBIRVJPIC0tPgo8c2VjdGlvbiBjbGFzcz0iaGVybyIgc3R5bGU9InBhZGRpbmctYm90dG9tOjAiPgogIDxkaXYgY2xhc3M9Imhlcm8tZXllYnJvdyI+UG9saXRpY2FsIG5hcnJhdGl2ZSBpbnRlbGxpZ2VuY2UgJm1pZGRvdDsgSW5kaWE8L2Rpdj4KICA8aDEgY2xhc3M9Imhlcm8tdGl0bGUiPgogICAgT2JzZXJ2aW5nIEluZGlhJ3M8YnIvPgogICAgPHNwYW4+Y29sbGVjdGl2ZTwvc3Bhbj4gcG9saXRpY2FsPGJyLz4KICAgIGF0dGVudGlvbiDigJQgbGl2ZS4KICA8L2gxPgoKICA8IS0tIFNJR05BVFVSRSBJTlNJR0hUICsgTkFSUkFUSVZFIFNUUklQIHNpZGUgYnkgc2lkZSAtLT4KICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjE4cHg7YWxpZ24taXRlbXM6c3RyZXRjaDttYXJnaW4tdG9wOjE2cHg7bWFyZ2luLWJvdHRvbTowO21heC13aWR0aDoxNDgwcHg7bWFyZ2luLWxlZnQ6YXV0bzttYXJnaW4tcmlnaHQ6YXV0bztwYWRkaW5nOjAgMzZweDsiPgogICAgPGRpdiBjbGFzcz0ic2lnbmF0dXJlLWluc2lnaHQiIHN0eWxlPSJtYXJnaW4tdG9wOjA7ZmxleDoxO21pbi13aWR0aDowIj4KICAgICAgPGRpdiBjbGFzcz0ic2ktbGFiZWwiPkN1cnJlbnQgbmF0aW9uYWwgbmFycmF0aXZlIHNoaWZ0PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InNpLXRleHQiIGlkPSJzaWctaW5zaWdodCI+SW5kaWEncyBhdHRlbnRpb24gaXMgbW92aW5nIGZyb20gPGVtPmluZmxhdGlvbiBhbmQgY29zdC1vZi1saXZpbmcgYW54aWV0eTwvZW0+IHRvd2FyZCA8ZW0+Z292ZXJuYW5jZSBhY2NvdW50YWJpbGl0eSBhbmQgYm9yZGVyIHNlY3VyaXR5PC9lbT4g4oCUIGEgc2hpZnQgYWNjZWxlcmF0aW5nIGFjcm9zcyB0aGUgbm9ydGhlcm4gYmVsdC48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ic2ktc3ViIj4KICAgICAgICA8c3BhbiBjbGFzcz0ic2ktdGFnIiBpZD0ic2lnLXRhZzEiPuKGkyBJbmZsYXRpb24gwrcgZmFkaW5nPC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJzaS10YWciIGlkPSJzaWctdGFnMiI+4oaRIEdvdmVybmFuY2UgwrcgcmlzaW5nPC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJzaS10YWciIGlkPSJzaWctdGFnMyI+4oaRIEJvcmRlciBzZWN1cml0eSDCtyBzdXJnaW5nPC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZmxleDowIDAgMzYwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOyI+CiAgICAgIDwhLS0gaGVhZGVyIC0tPgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsiPgogICAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj5OYXJyYXRpdmUgc2hpZnRzPC9zcGFuPgogICAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MnB4OyI+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPCEtLSBzaGlmdHMgbGlzdCAtLT4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoxMHB4IDE0cHg7Z2FwOjZweDsiIGlkPSJzaGlmdC1saXN0Ij48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+CgoKPCEtLSBNQUlOOiBNQVAgKyBTVEFURSBQQU5FTCAtLT4KPGRpdiBjbGFzcz0ibWFpbiI+CgogIDxkaXYgY2xhc3M9Im1hcC1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im1hcC10b3AiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtdGl0bGUtYmxvY2siPgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij5JbmRpYSAmbWRhc2g7IGNvbGxlY3RpdmUgYXR0ZW50aW9uPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibXMiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IGxpdmUgc2lnbmFsIGNvbXBvc2l0ZTwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGVnZW5kIj48c3Bhbj5xdWlldDwvc3Bhbj48ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj48c3Bhbj5hY3RpdmU8L3NwYW4+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxheWVyLXJvdyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsYXllci1sYWJlbCI+Vmlldzwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibHRhYnMiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iYXR0ZW50aW9uIj5BdHRlbnRpb248L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb248L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9InZlbG9jaXR5Ij5Nb21lbnR1bTwvc3Bhbj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im1hcC1zdmctd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC1pbm5lciI+CiAgICAgICAgPHN2ZyBpZD0iaW5kaWEtbWFwIiB2aWV3Qm94PSIwIDAgODAwIDgwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgbWVldCI+CiAgICAgICAgICA8ZGVmcz4KICAgICAgICAgICAgPHJhZGlhbEdyYWRpZW50IGlkPSJhbWJHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9InJnYmEoMjI0LDkwLDQwLDAuMDQpIi8+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICAgICAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgICAgICAgICA8ZmlsdGVyIGlkPSJzdGF0ZUdsb3ciIHg9Ii0zMCUiIHk9Ii0zMCUiIHdpZHRoPSIxNjAlIiBoZWlnaHQ9IjE2MCUiPgogICAgICAgICAgICAgIDxmZUdhdXNzaWFuQmx1ciBpbj0iU291cmNlR3JhcGhpYyIgc3RkRGV2aWF0aW9uPSI4IiByZXN1bHQ9ImJsdXIiLz4KICAgICAgICAgICAgICA8ZmVDb21wb3NpdGUgaW49IlNvdXJjZUdyYXBoaWMiIGluMj0iYmx1ciIgb3BlcmF0b3I9Im92ZXIiLz4KICAgICAgICAgICAgPC9maWx0ZXI+CiAgICAgICAgICA8L2RlZnM+CiAgICAgICAgICA8cmVjdCB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgZmlsbD0idXJsKCNhbWJHbG93KSIvPgogICAgICAgICAgPGcgaWQ9Im1hcC1nbG93Ij48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXN0YXRlcyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1wdWxzZXMiPjwvZz4KICAgICAgICA8L3N2Zz4KICAgICAgICA8ZGl2IGNsYXNzPSJtYXAtdG9vbHRpcCIgaWQ9InRvb2x0aXAiPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFNUQVRFIFBBTkVMIC0tPgogIDxkaXYgY2xhc3M9InN0YXRlLXBhbmVsIiBpZD0ic3RhdGUtZGV0YWlsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLWVtcHR5Ij4KICAgICAgPHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMSI+CiAgICAgICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgOHY0TTEyIDE2aC4wMSIvPgogICAgICA8L3N2Zz4KICAgICAgPGRpdiBjbGFzcz0icGUtdCI+U2VsZWN0IGEgc3RhdGU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0icGUtcyI+Q2xpY2sgYW55IHJlZ2lvbiBvbiB0aGUgbWFwPGJyLz50byBvcGVuIGl0cyBuYXJyYXRpdmUgcGFuZWwuPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCjwvZGl2PgoKPCEtLSBOQVJSQVRJVkUgUk9XIC0tPgo8ZGl2IGNsYXNzPSJuYXItcm93Ij4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj4KICAgICAgPGRpdiBjbGFzcz0ibmMtdGl0bGUiPlJpc2luZyBuYXJyYXRpdmVzPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im5jLXN1YiI+NzIgaG91cnMgJm1pZGRvdDsgYWxsIHN0YXRlczwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0icmlzaW5nLW5hciI+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJuYy1zdWIiPjcyIGhvdXJzICZtaWRkb3Q7IGFsbCBzdGF0ZXM8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9ImZhbGxpbmctbmFyIj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj4KICAgICAgPGRpdiBjbGFzcz0ibmMtdGl0bGUiPlJlZ2lvbmFsIHNoaWZ0czwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJuYy1zdWIiPlN0YXRlLWxldmVsIGV2b2x1dGlvbiAmbWlkZG90OyAzMCBkYXlzPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyZWdpb25hbC1zaGlmdHMiPjwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICBJbmRpYSBBdHRlbnRpb24gTWFwIGlzIGFuIG9ic2VydmF0aW9uYWwgcGxhdGZvcm0uIEl0IHRyYWNrcyBjb2xsZWN0aXZlIGF0dGVudGlvbiBwYXR0ZXJucyBmcm9tIHB1YmxpYyBkYXRhIGFuZCBkb2VzIG5vdCBpbmZlciBwb2xpdGljYWwgcG9zaXRpb25zLCBwcmVkaWN0IGVsZWN0aW9ucywgb3IgZW5kb3JzZSBuYXJyYXRpdmVzLgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgaWYoIVNEW3Jvdy5uYW1lXSkgU0Rbcm93Lm5hbWVdPU9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7CiAgICAgIE9iamVjdC5hc3NpZ24oU0Rbcm93Lm5hbWVdLHthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOnJvdy5kb21pbmFudF9lbW90aW9uLGRvbWluYW50X25hcnJhdGl2ZTpyb3cuZG9taW5hbnRfbmFycmF0aXZlfSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVTdHJpcE5hcnJhdGl2ZSgpOwogICAgdXBkYXRlU3RyaXBBbmdlcigpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tBUEldJyxlLm1lc3NhZ2UpO30KfQphc3luYyBmdW5jdGlvbiBmZXRjaERldGFpbChuYW1lKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChuYW1lKSk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBTRFtuYW1lXT17CiAgICAgIGF0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LAogICAgICBlbW90aW9uczpkLmVtb3Rpb25zfHxERUZBVUxULmVtb3Rpb25zLAogICAgICBuYXJyYXRpdmVzOihkLm5hcnJhdGl2ZXN8fFtdKS5tYXAoZnVuY3Rpb24obil7cmV0dXJue25hbWU6bi5uYW1lLHZhbDpuLnZhbCxkaXI6bi5kaXJ8fCdmbGF0J307fSksCiAgICAgIHJpc2luZzpkLnJpc2luZ3x8W10sZmFsbGluZzpkLmZhbGxpbmd8fFtdLAogICAgICBzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OkRFRkFVTFQubmFycmF0aXZlSGlzdG9yeSwKICAgIH07CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe3JldHVybiBnKG5hbWUpO30KfQphc3luYyBmdW5jdGlvbiBmZXRjaFNuYXAoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zbmFwc2hvdC9kYWlseScpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC5lcnJvcikgcmV0dXJuOwogICAgLy8gdG9wYmFyCiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xpdmUtY291bnQnKTsKICAgIGlmKGVsJiZkLnRvdGFsX3NpZ25hbHMpIGVsLnRleHRDb250ZW50PWQudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOwogICAgdmFyIG1ldGE9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1tZXRhJyk7CiAgICBpZihtZXRhJiZkLmFzX29mKSBtZXRhLnRleHRDb250ZW50PSczMCBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKGQuYXNfb2YpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nKTsKICAgIC8vIHN0YXRzIHN0cmlwCiAgICBzZXRUZXh0KCdzYy1zaWduYWxzLXZhbCcsIGQudG90YWxfc2lnbmFscz9kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTon4oCUJyk7CiAgICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsIGQuaG90dGVzdF9zdGF0ZXx8J+KAlCcpOwogICAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCBkLmhvdHRlc3Rfc2NvcmU/J0F0dGVudGlvbiAnK2QuaG90dGVzdF9zY29yZTonQ2xpY2sgdG8gZXhwbG9yZScpOwogICAgc2V0VGV4dCgnc2MtY29vbGluZy12YWwnLCBkLmZhc3Rlc3RfY29vbGluZ3x8J+KAlCcpOwogICAgc2V0VGV4dCgnc2MtY29vbGluZy1zdWInLCAnU2lnbmFsIHZlbG9jaXR5IOKGkycpOwogICAgLy8gdG9wIG5hcnJhdGl2ZSBmcm9tIHN0YXRlIGRhdGEKICAgIHVwZGF0ZVN0cmlwTmFycmF0aXZlKCk7CiAgICB1cGRhdGVTdHJpcEFuZ2VyKCk7CiAgfWNhdGNoKGUpe30KfQoKZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoZWwpZWwudGV4dENvbnRlbnQ9dmFsO30KCmZ1bmN0aW9uIHVwZGF0ZVN0cmlwTmFycmF0aXZlKCl7CiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7KHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7aWYobi5kaXI9PT0ndXAnKSBuY1tuLm5hbWVdPShuY1tuLm5hbWVdfHwwKStuLnZhbDt9KTt9KTsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXTsKICBpZih0b3ApewogICAgdmFyIG5tPXRvcFswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0b3BbMF0uc2xpY2UoMSk7CiAgICBzZXRUZXh0KCdzYy1uYXJyYXRpdmUtdmFsJyxubSk7CiAgICBzZXRUZXh0KCdzYy1uYXJyYXRpdmUtc3ViJywnUmlzaW5nIGFjcm9zcyBtdWx0aXBsZSBzdGF0ZXMnKTsKICB9Cn0KCmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXsKICB2YXIgdG9wQW5nZXI9bnVsbCx0b3BWYWw9MDsKICBPYmplY3QuZW50cmllcyhTRCkuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICB2YXIgbm09a3ZbMF0scz1rdlsxXTsKICAgIHZhciBhbmdlcj1zLmVtb3Rpb25zJiZzLmVtb3Rpb25zLmFuZ2VyP3MuZW1vdGlvbnMuYW5nZXI6MDsKICAgIC8vIG5vcm1hbGl6ZSBpZiBkZWNpbWFsCiAgICBpZihhbmdlcjwxKSBhbmdlcj1hbmdlcioxMDA7CiAgICBpZihhbmdlcj50b3BWYWwpe3RvcFZhbD1hbmdlcjt0b3BBbmdlcj1ubTt9CiAgfSk7CiAgaWYodG9wQW5nZXIpewogICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlcik7CiAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdBbmdlciAnK01hdGgucm91bmQodG9wVmFsKSsnJSBvZiBzaWduYWxzJyk7CiAgfQp9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDMwMDAwMCk7fQogIH0sMTUwMDApOwp9Cgp2YXIgUkVHX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidCaWhhcicsZnJvbTonRWxlY3Rpb24gcmhldG9yaWMnLHRvOidVbmVtcGxveW1lbnQgJiBleGFtIHNjYW1zJyx0aW1lOic2IHdrcyd9LAogIHtzdGF0ZTonV2VzdCBCZW5nYWwnLGZyb206J0J5cG9sbCBwb2xpdGljcycsdG86J0xhdyAmIG9yZGVyIMK3IEJvcmRlcicsdGltZTonNCB3a3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdrcyd9LAogIHtzdGF0ZTonS2FybmF0YWthJyxmcm9tOidNaW5pbmcgY29udHJvdmVyc3knLHRvOidMYW5ndWFnZSBzaWduYWdlIHBvbGl0aWNzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonRGVsaGknLGZyb206J01ldHJvIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3a3MnfSwKICB7c3RhdGU6J1B1bmphYicsZnJvbTonUG93ZXIgY3Jpc2lzJyx0bzonQm9yZGVyIHNlY3VyaXR5IMK3IERyb25lcycsdGltZTonMyB3a3MnfSwKXTsKdmFyIE1PQ0tfUj1bCiAge25hbWU6J0JvcmRlciBzZWN1cml0eScsc3RhdGVzOidKJksgwrcgUHVuamFiIMK3IFJhamFzdGhhbicscGN0OicrNDElJ30sCiAge25hbWU6J1VuZW1wbG95bWVudCcsc3RhdGVzOidCaWhhciDCtyBVUCDCtyBKaGFya2hhbmQnLHBjdDonKzI4JSd9LAogIHtuYW1lOidMYW5ndWFnZSBwb2xpdGljcycsc3RhdGVzOidUTiDCtyBLYXJuYXRha2EgwrcgTUgnLHBjdDonKzIyJSd9LAogIHtuYW1lOidFbnZpcm9ubWVudGFsIGNyaXNpcycsc3RhdGVzOidEZWxoaSDCtyBSYWphc3RoYW4gwrcgQVAnLHBjdDonKzE5JSd9LAogIHtuYW1lOidFdGhuaWMgdGVuc2lvbnMnLHN0YXRlczonTWFuaXB1ciDCtyBBc3NhbSDCtyBXQicscGN0OicrMTclJ30sCl07CnZhciBNT0NLX0Y9WwogIHtuYW1lOidFbGVjdGlvbiByaGV0b3JpYycsc3RhdGVzOidOYXRpb25hbCBwb3N0LWN5Y2xlJyxwY3Q6Jy0zOCUnfSwKICB7bmFtZTonSW5mbGF0aW9uIHByZXNzdXJlJyxzdGF0ZXM6J0Vhc2luZyBuYXRpb25hbGx5JyxwY3Q6Jy0yNCUnfSwKICB7bmFtZTonRmFybWVyIHByb3Rlc3RzJyxzdGF0ZXM6J01vbWVudHVtIGxvc3QnLHBjdDonLTE5JSd9LAogIHtuYW1lOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsc3RhdGVzOidSaWJib24tY3V0dGluZyBkb25lJyxwY3Q6Jy0xNCUnfSwKICB7bmFtZTonUmVsaWdpb3VzIGZlc3RpdmFscycsc3RhdGVzOidQb3N0LXNlYXNvbiBmYWRlJyxwY3Q6Jy0xMSUnfSwKXTsKCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5jW24ubmFtZV0pbmNbbi5uYW1lXT17dXA6MCxkb3duOjAsc3RhdGVzOltdfTsKICAgICAgbmNbbi5uYW1lXVtuLmRpcj09PSd1cCc/J3VwJzonZG93biddKz0obi52YWx8fDApOwogICAgICBpZihuY1tuLm5hbWVdLnN0YXRlcy5pbmRleE9mKHMubmFtZXx8JycpPDApbmNbbi5uYW1lXS5zdGF0ZXMucHVzaChzLm5hbWV8fCcnKTsKICAgIH0pOwogIH0pOwogIHZhciBhbGw9T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYlsxXS51cCtiWzFdLmRvd24pLShhWzFdLnVwK2FbMV0uZG93bik7fSk7CiAgdmFyIHJpc2luZz1hbGwuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0udXA+PWt2WzFdLmRvd247fSkuc2xpY2UoMCwzKTsKICB2YXIgZmFkaW5nPWFsbC5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5kb3duPmt2WzFdLnVwO30pOwogIGlmKCFmYWRpbmcubGVuZ3RoKSBmYWRpbmc9YWxsLnNsaWNlKC0zKTsKICBmYWRpbmc9ZmFkaW5nLnNsaWNlKDAsMyk7CiAgaWYoIWFsbC5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+Q29sbGVjdGluZyBzaWduYWwgZGF0YS4uLjwvZGl2Pic7CiAgICByZXR1cm47CiAgfQogIHZhciByb3dzPVtdOwogIGZvcih2YXIgaT0wO2k8TWF0aC5tYXgocmlzaW5nLmxlbmd0aCxmYWRpbmcubGVuZ3RoLDEpO2krKyl7CiAgICB2YXIgcj1yaXNpbmdbaV0sZj1mYWRpbmdbaV07CiAgICByb3dzLnB1c2goCiAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo4cHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi1ib3R0b206NnB4Ij4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6OHB4IDEwcHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjdweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOiMzYmI4ZDg7bWFyZ2luLWJvdHRvbTozcHgiPkZBRElORzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC13ZWlnaHQ6NTAwIj4nKyhmP2ZbMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZlswXS5zbGljZSgxKTon4oCUJykrJzwvZGl2PicrCiAgICAgICAgKGYmJmZbMV0uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHgiPicrZlsxXS5zdGF0ZXMuc2xpY2UoMCwyKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjAgOHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LXNpemU6MTRweCI+4oaSPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjhweCAxMHB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjdweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOiNlMDVhMjg7bWFyZ2luLWJvdHRvbTozcHgiPlJJU0lORzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwIj4nKyhyP3JbMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrclswXS5zbGljZSgxKTon4oCUJykrJzwvZGl2PicrCiAgICAgICAgKHImJnJbMV0uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHgiPicrclsxXS5zdGF0ZXMuc2xpY2UoMCwyKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsKICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nCiAgICApOwogIH0KICBlbC5pbm5lckhUTUw9cm93cy5qb2luKCcnKTsKfQoKZnVuY3Rpb24gcmVuZGVyTW9tZW50dW0oKXsKICB2YXIgbmM9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXtuY1tuLm5hbWVdPShuY1tuLm5hbWVdfHwwKStuLnZhbDt9KTt9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpLGZhbGxpbmc9c29ydGVkLnNsaWNlKC01KS5yZXZlcnNlKCk7CiAgdmFyIG14PXJpc2luZy5sZW5ndGg/cmlzaW5nWzBdWzFdOjEwMDsKCiAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLW5hcicpLGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmFsbGluZy1uYXInKTsKCiAgaWYocmlzaW5nLmxlbmd0aCl7CiAgICByRWwuaW5uZXJIVE1MPXJpc2luZy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJtb20taXQiPjxzcGFuIGNsYXNzPSJtb20tcmsiPicrKGkrMSkrJzwvc3Bhbj48ZGl2IGNsYXNzPSJtb20taW5mIj48ZGl2IGNsYXNzPSJtb20tbm0iPicrblswXSsnPC9kaXY+PC9kaXY+PHNwYW4gY2xhc3M9Im1vbS1wYyByIj7ihpE8L3NwYW4+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibW9tLXRyIj48ZGl2IGNsYXNzPSJtb20tZmwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuWzFdL214KjEwMCkrJyU7YmFja2dyb3VuZDp2YXIoLS1yaXNlKTtvcGFjaXR5OjAuNSI+PC9kaXY+PC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH1lbHNlewogICAgckVsLmlubmVySFRNTD1NT0NLX1IubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibW9tLWl0Ij48c3BhbiBjbGFzcz0ibW9tLXJrIj4nKyhpKzEpKyc8L3NwYW4+PGRpdiBjbGFzcz0ibW9tLWluZiI+PGRpdiBjbGFzcz0ibW9tLW5tIj4nK24ubmFtZSsnPC9kaXY+PGRpdiBjbGFzcz0ibW9tLXN0Ij4nK24uc3RhdGVzKyc8L2Rpdj48L2Rpdj48c3BhbiBjbGFzcz0ibW9tLXBjIHIiPicrbi5wY3QrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtb20tdHIiPjxkaXYgY2xhc3M9Im1vbS1mbCIgc3R5bGU9IndpZHRoOicrcGFyc2VJbnQobi5wY3QpKyclO2JhY2tncm91bmQ6dmFyKC0tcmlzZSk7b3BhY2l0eTowLjUiPjwvZGl2PjwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CiAgaWYoZmFsbGluZy5sZW5ndGgpewogICAgZkVsLmlubmVySFRNTD1mYWxsaW5nLm1hcChmdW5jdGlvbihuLGkpewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im1vbS1pdCI+PHNwYW4gY2xhc3M9Im1vbS1yayI+JysoaSsxKSsnPC9zcGFuPjxkaXYgY2xhc3M9Im1vbS1pbmYiPjxkaXYgY2xhc3M9Im1vbS1ubSI+JytuWzBdKyc8L2Rpdj48L2Rpdj48c3BhbiBjbGFzcz0ibW9tLXBjIGYiPuKGkzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtb20tdHIiPjxkaXYgY2xhc3M9Im1vbS1mbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKSsnJTtiYWNrZ3JvdW5kOnZhcigtLWZhbGwpO29wYWNpdHk6MC41Ij48L2Rpdj48L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfWVsc2V7CiAgICBmRWwuaW5uZXJIVE1MPU1PQ0tfRi5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJtb20taXQiPjxzcGFuIGNsYXNzPSJtb20tcmsiPicrKGkrMSkrJzwvc3Bhbj48ZGl2IGNsYXNzPSJtb20taW5mIj48ZGl2IGNsYXNzPSJtb20tbm0iPicrbi5uYW1lKyc8L2Rpdj48ZGl2IGNsYXNzPSJtb20tc3QiPicrbi5zdGF0ZXMrJzwvZGl2PjwvZGl2PjxzcGFuIGNsYXNzPSJtb20tcGMgZiI+JytuLnBjdCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im1vbS10ciI+PGRpdiBjbGFzcz0ibW9tLWZsIiBzdHlsZT0id2lkdGg6JytNYXRoLmFicyhwYXJzZUludChuLnBjdCkpKyclO2JhY2tncm91bmQ6dmFyKC0tZmFsbCk7b3BhY2l0eTowLjUiPjwvZGl2PjwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLXNoaWZ0cycpLmlubmVySFRNTD1SRUdfU0hJRlRTLm1hcChmdW5jdGlvbihzKXsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0icmVnLWl0IiBvbmNsaWNrPSJzZWxlY3RfKFwnJytzLnN0YXRlKydcJykiPicrCiAgICAgICc8c3BhbiBjbGFzcz0icmVnLWJhZGdlIj4nK3Muc3RhdGUrJzwvc3Bhbj4nKwogICAgICAnPGRpdiBjbGFzcz0icmVnLWZsIj48c3BhbiBjbGFzcz0icmVnLWZyb20iPicrcy5mcm9tKyc8L3NwYW4+PHNwYW4gY2xhc3M9InJlZy1hcnIiPuKGkjwvc3Bhbj48c3BhbiBjbGFzcz0icmVnLXRvIj4nK3MudG8rJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPHNwYW4gY2xhc3M9InJlZy10bSI+JytzLnRpbWUrJzwvc3Bhbj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KCi8vIFNUQVRFIERBVEEKdmFyIFNEPXsKICAiQmloYXIiOnthdHRlbnRpb246ODcsZGVsdGE6Nix2ZWxvY2l0eTowLjM0LGVtb3Rpb25zOnthbnhpZXR5OjMyLGFuZ2VyOjI4LGhvcGU6MTQscHJpZGU6OCxmZWFyOjE4fSwKICAgIG5hcnJhdGl2ZXM6W3tuYW1lOiJVbmVtcGxveW1lbnQiLHZhbDo0MSxkaXI6InVwIn0se25hbWU6IkNvcnJ1cHRpb24iLHZhbDoyOCxkaXI6InVwIn0se25hbWU6IkNhc3RlIHBvbGl0aWNzIix2YWw6MTIsZGlyOiJmbGF0In0se25hbWU6IkVkdWNhdGlvbiIsdmFsOjExLGRpcjoidXAifSx7bmFtZToiTWlncmF0aW9uIix2YWw6OCxkaXI6ImZsYXQifV0sCiAgICByaXNpbmc6W3t0OiJFeGFtIHNjYW0gb3V0cmFnZSIscGN0OiIrNDclIn0se3Q6Ik1pZ3JhbnQgcmV0dXJuIixwY3Q6IisyMiUifV0sZmFsbGluZzpbe3Q6IkVsZWN0aW9uIHNwZWVjaGVzIixwY3Q6Ii0zMSUifV0sCiAgICBhcnRpY2xlczpbe3NyYzoiRGFpbmlrIEJoYXNrYXIiLHR4dDoiUGF0bmEgc3R1ZGVudHMgc3RhZ2UgcHJvdGVzdCBkZW1hbmRpbmcgY2FuY2VsbGF0aW9uIG9mIGV4YW0ifSx7c3JjOiJUaGUgSGluZHUiLHR4dDoiQmloYXIncyB5b3V0aCB1bmVtcGxveW1lbnQgY3Jpc2lzIGludGVuc2lmaWVzIn0se3NyYzoiUFRJIix0eHQ6IlN0YXRlIGFubm91bmNlcyBpbnF1aXJ5IGludG8gYWxsZWdlZCBwYXBlciBsZWFrIn1dLAogICAgc3VtbWFyeToiQmloYXIncyBmb2N1cyBoYXMgcGl2b3RlZCBmcm9tIGVsZWN0b3JhbCBub2lzZSB0byBkZWVwIHN0cnVjdHVyYWwgYW54aWV0aWVzIOKAlCB1bmVtcGxveW1lbnQsIGV4YW0gZmFpbHVyZXMsIGFuZCBtaWdyYXRpb24uIFRoZSB5b3V0aCBuYXJyYXRpdmUgaXMgbm93IGRvbWluYW50LiIsCiAgICB0aW1lbGluZTpbNjIsNjQsNjYsNzEsNzQsNzgsODEsODddLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJFbGVjdGlvbiByaGV0b3JpYyIsd2hlbjoiNiBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkV4YW0gc2NhbSBvdXRyYWdlIix3aGVuOiIzIG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiWW91dGggdW5lbXBsb3ltZW50Iix3aGVuOiI2IHdlZWtzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6Ik1pZ3JhdGlvbiBhbnhpZXR5Iix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIk1haGFyYXNodHJhIjp7YXR0ZW50aW9uOjYyLGRlbHRhOi04LHZlbG9jaXR5Oi0wLjEyLGVtb3Rpb25zOnthbnhpZXR5OjE4LGFuZ2VyOjIyLGhvcGU6MTYscHJpZGU6MjQsZmVhcjoyMH0sCiAgICBuYXJyYXRpdmVzOlt7bmFtZToiTGFuZ3VhZ2UgcG9saXRpY3MiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MjEsZGlyOiJmbGF0In0se25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoxOCxkaXI6ImRvd24ifSx7bmFtZToiUmVnaW9uYWwgaWRlbnRpdHkiLHZhbDoxOSxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImRvd24ifV0sCiAgICByaXNpbmc6W3t0OiJNYXJhdGhpIGxhbmd1YWdlIHJvdyIscGN0OiIrMTklIn1dLGZhbGxpbmc6W3t0OiJPbmlvbiBwcmljZXMiLHBjdDoiLTI0JSJ9XSwKICAgIGFydGljbGVzOlt7c3JjOiJMb2ttYXQiLHR4dDoiTGFuZ3VhZ2UgZW5mb3JjZW1lbnQgZGViYXRlIHJlaWduaXRlcyBpbiBzdWJ1cmJhbiBNdW1iYWkifSx7c3JjOiJJbmRpYW4gRXhwcmVzcyIsdHh0OiJPbmlvbiBwcmljZXMgc3RhYmlsaXplIGFzIGZhcm1lciBwcm90ZXN0IGFjdGl2aXR5IHN1YnNpZGVzIn1dLAogICAgc3VtbWFyeToiTWFoYXJhc2h0cmEgaXMgY29vbGluZyBhcyBlY29ub21pYyBhbnhpZXRpZXMgZWFzZS4gVGhlIG5hcnJhdGl2ZSBpcyByb3RhdGluZyB0b3dhcmQgbGFuZ3VhZ2UgaWRlbnRpdHkgYW5kIHJlZ2lvbmFsIHBvbGl0aWNzLiIsCiAgICB0aW1lbGluZTpbNzgsNzUsNzMsNzAsNjksNjYsNjQsNjJdLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJDb2FsaXRpb24gcG9saXRpY3MiLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJGYXJtZXIgZGlzdHJlc3MiLHdoZW46IjQgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJPbmlvbiBwcmljZSBjcmlzaXMiLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6Ikxhbmd1YWdlICYgaWRlbnRpdHkiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiVXR0YXIgUHJhZGVzaCI6e2F0dGVudGlvbjo3OCxkZWx0YTozLHZlbG9jaXR5OjAuMDgsZW1vdGlvbnM6e2FueGlldHk6MjIsYW5nZXI6MjQsaG9wZToxOCxwcmlkZToyMixmZWFyOjE0fSwKICAgIG5hcnJhdGl2ZXM6W3tuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjI2LGRpcjoidXAifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMixkaXI6ImZsYXQifSx7bmFtZToiSW5mcmFzdHJ1Y3R1cmUiLHZhbDoxOCxkaXI6InVwIn0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiQ2FzdGUgcG9saXRpY3MiLHZhbDoxOCxkaXI6ImZsYXQifV0sCiAgICByaXNpbmc6W3t0OiJFeHByZXNzd2F5IG9wZW5pbmciLHBjdDoiKzI4JSJ9XSxmYWxsaW5nOlt7dDoiRmVzdGl2YWwgbG9naXN0aWNzIixwY3Q6Ii0xOCUifV0sCiAgICBhcnRpY2xlczpbe3NyYzoiRGFpbmlrIEphZ3JhbiIsdHh0OiJHYW5nYSBFeHByZXNzd2F5IGV4dGVuc2lvbiBpbmF1Z3VyYXRlZCJ9LHtzcmM6IkFtYXIgVWphbGEiLHR4dDoiQXlvZGh5YSB0b3VyaXNtIGNyb3NzZXMgbW9udGhseSByZWNvcmQifV0sCiAgICBzdW1tYXJ5OiJVUCBtYWludGFpbnMgYSBoaWdoLWJhc2VsaW5lIHBhdHRlcm4uIExhdyAmIG9yZGVyIGFuZCBpbmZyYXN0cnVjdHVyZSBkb21pbmF0ZSB0aGUgbmFycmF0aXZlIOKAlCBhIHN0YWJsZSBkdWFsaXR5IHRoYXQgaGFzIHBlcnNpc3RlZCBmb3IgbW9udGhzLiIsCiAgICB0aW1lbGluZTpbNzIsNzMsNzQsNzUsNzYsNzcsNzYsNzhdLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJFbGVjdGlvbiBjYW1wYWlnbmluZyIsd2hlbjoiOSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlJlbGlnaW91cyBpbmZyYXN0cnVjdHVyZSIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkxhdyAmIG9yZGVyIHB1c2giLHdoZW46IjMgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IkluZnJhc3RydWN0dXJlIHByaWRlIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIlRhbWlsIE5hZHUiOnthdHRlbnRpb246NzEsZGVsdGE6NSx2ZWxvY2l0eTowLjIxLGVtb3Rpb25zOnthbnhpZXR5OjE0LGFuZ2VyOjI2LGhvcGU6MTgscHJpZGU6MzAsZmVhcjoxMn0sCiAgICBuYXJyYXRpdmVzOlt7bmFtZToiTGFuZ3VhZ2UgcG9saXRpY3MiLHZhbDozMixkaXI6InVwIn0se25hbWU6IkZlZGVyYWxpc20iLHZhbDoyMSxkaXI6InVwIn0se25hbWU6IkVkdWNhdGlvbiIsdmFsOjE2LGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjE2LGRpcjoiZmxhdCJ9LHtuYW1lOiJSZWdpb25hbCBpZGVudGl0eSIsdmFsOjE1LGRpcjoidXAifV0sCiAgICByaXNpbmc6W3t0OiJORVAgdGhyZWUtbGFuZ3VhZ2Ugcm93IixwY3Q6IiszOCUifSx7dDoiU3RhdGUgZnVuZHMgZGlzcHV0ZSIscGN0OiIrMjElIn1dLGZhbGxpbmc6W3t0OiJDeWNsb25lIGFmdGVybWF0aCIscGN0OiItMjklIn1dLAogICAgYXJ0aWNsZXM6W3tzcmM6IlRoZSBIaW5kdSIsdHh0OiJUaHJlZS1sYW5ndWFnZSBmb3JtdWxhIGRlYmF0ZSBlbnRlcnMgZnJlc2ggcGhhc2UifSx7c3JjOiJEaW5hbWFsYXIiLHR4dDoiU3R1ZGVudCBmZWRlcmF0aW9ucyBwYXNzIHJlc29sdXRpb25zIGFnYWluc3QgbGFuZ3VhZ2UgcG9saWN5In1dLAogICAgc3VtbWFyeToiVGFtaWwgTmFkdSBoYXMgcGl2b3RlZCBkZWNpc2l2ZWx5IHRvIGxhbmd1YWdlIHBvbGl0aWNzIGFuZCBmZWRlcmFsaXNtLiBUaGUgdGhyZWUtbGFuZ3VhZ2UgZGViYXRlIGlzIG5vdyB0aGUgZGVmaW5pbmcgbmFycmF0aXZlIGluIGJvdGggVGFtaWwgYW5kIEVuZ2xpc2ggbWVkaWEuIiwKICAgIHRpbWVsaW5lOls2NCw2NSw2Niw2OCw2OSw3MCw3MCw3MV0sCiAgICBuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IkN5Y2xvbmUgYWZ0ZXJtYXRoIix3aGVuOiI0IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiUmVnaW9uYWwgZWNvbm9taWMgZ3Jvd3RoIix3aGVuOiIzIG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiTGFuZ3VhZ2UgcG9saWN5IHJlc2lzdGFuY2UiLHdoZW46IjYgd2Vla3MgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiRmVkZXJhbCByZXNvdXJjZSBkaXNwdXRlcyIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAogICJXZXN0IEJlbmdhbCI6e2F0dGVudGlvbjo3NCxkZWx0YTo0LHZlbG9jaXR5OjAuMTUsZW1vdGlvbnM6e2FueGlldHk6MjAsYW5nZXI6MjgsaG9wZToxNCxwcmlkZToxNixmZWFyOjIyfSwKICAgIG5hcnJhdGl2ZXM6W3tuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiUmVsaWdpb24iLHZhbDoxOSxkaXI6InVwIn0se25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoxOCxkaXI6InVwIn0se25hbWU6IkNvcnJ1cHRpb24iLHZhbDoyMixkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE3LGRpcjoiZmxhdCJ9XSwKICAgIHJpc2luZzpbe3Q6IkJvcmRlciBpbmZpbHRyYXRpb24gZGViYXRlIixwY3Q6IisyNCUifV0sZmFsbGluZzpbe3Q6IkJ5cG9sbCByZXN1bHRzIixwY3Q6Ii0yMiUifV0sCiAgICBhcnRpY2xlczpbe3NyYzoiQW5hbmRhYmF6YXIiLHR4dDoiQlNGIGFuZCBzdGF0ZSBwb2xpY2UgZXNjYWxhdGUgY29vcmRpbmF0aW9uIG5lYXIgQm9uZ2FvbiJ9LHtzcmM6IlRlbGVncmFwaCBJbmRpYSIsdHh0OiJTU0MgcmVjcnVpdG1lbnQgY2FzZTogQ0JJIGZpbGVzIGZyZXNoIGNoYXJnZSBzaGVldCJ9XSwKICAgIHN1bW1hcnk6IkJlbmdhbCdzIGF0dGVudGlvbiBpcyBoZWF2eSBvbiBsYXctYW5kLW9yZGVyIGFuZCBib3JkZXIgbmFycmF0aXZlcy4gVGhlIHNjaG9vbCByZWNydWl0bWVudCBjYXNlIGNvbnRpbnVlcyB0byBnZW5lcmF0ZSBjeWNsZXMgb2Ygb3V0cmFnZS4iLAogICAgdGltZWxpbmU6WzY4LDY5LDcwLDcxLDcyLDczLDczLDc0XSwKICAgIG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiQnlwb2xsIHBvbGl0aWNzIix3aGVuOiI1IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiQ29ycnVwdGlvbiAmIHJlY3J1aXRtZW50Iix3aGVuOiIzIG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiQm9yZGVyICYgc2VjdXJpdHkiLHdoZW46IjYgd2Vla3MgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiTGF3ICYgb3JkZXIgZG9taW5hbmNlIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIkthcm5hdGFrYSI6e2F0dGVudGlvbjo2OCxkZWx0YToyLHZlbG9jaXR5OjAuMDksZW1vdGlvbnM6e2FueGlldHk6MTYsYW5nZXI6MjAsaG9wZToyMixwcmlkZToyNCxmZWFyOjE4fSwKICAgIG5hcnJhdGl2ZXM6W3tuYW1lOiJMYW5ndWFnZSBwb2xpdGljcyIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiRWNvbm9teSAvIElUIix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJXYXRlciBkaXNwdXRlcyIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE5LGRpcjoidXAifSx7bmFtZToiUmVnaW9uYWwgaWRlbnRpdHkiLHZhbDoxNyxkaXI6InVwIn1dLAogICAgcmlzaW5nOlt7dDoiS2FubmFkYSBzaWduYWdlIHJ1bGUiLHBjdDoiKzIyJSJ9XSxmYWxsaW5nOlt7dDoiTWluaW5nIGlucXVpcnkiLHBjdDoiLTEzJSJ9XSwKICAgIGFydGljbGVzOlt7c3JjOiJEZWNjYW4gSGVyYWxkIix0eHQ6IkJlbmdhbHVydSBjaXZpYyBib2R5IHN0ZXBzIHVwIEthbm5hZGEgc2lnbmFnZSBlbmZvcmNlbWVudCJ9XSwKICAgIHN1bW1hcnk6Ikthcm5hdGFrYSBiYWxhbmNlcyBlY29ub21pYyBvcHRpbWlzbSBmcm9tIHRoZSBJVCBjb3JyaWRvciB3aXRoIHJpc2luZyByZWdpb25hbC1pZGVudGl0eSBwb2xpdGljcy4gQm90aCBuYXJyYXRpdmVzIGFyZSBhY2NlbGVyYXRpbmcgc2ltdWx0YW5lb3VzbHkuIiwKICAgIHRpbWVsaW5lOls2NCw2NSw2NSw2Niw2Nyw2Nyw2OCw2OF0sCiAgICBuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6Ik1pbmluZyBjb250cm92ZXJzeSIsd2hlbjoiNiBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IklUIHNlY3RvciBncm93dGgiLHdoZW46IjMgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJXYXRlciBkaXNwdXRlIHJldml2YWwiLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6Ikxhbmd1YWdlICYgcmVnaW9uYWwgaWRlbnRpdHkiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiS2VyYWxhIjp7YXR0ZW50aW9uOjU0LGRlbHRhOjEsdmVsb2NpdHk6MC4wNCxlbW90aW9uczp7YW54aWV0eToxNCxhbmdlcjoxOCxob3BlOjIyLHByaWRlOjI2LGZlYXI6MjB9LAogICAgbmFycmF0aXZlczpbe25hbWU6IkdvdmVybmFuY2UiLHZhbDoyMixkaXI6ImZsYXQifSx7bmFtZToiRWR1Y2F0aW9uIix2YWw6MjAsZGlyOiJ1cCJ9LHtuYW1lOiJFbnZpcm9ubWVudCIsdmFsOjE4LGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjIyLGRpcjoiZG93biJ9LHtuYW1lOiJSZWxpZ2lvbiIsdmFsOjE4LGRpcjoiZmxhdCJ9XSwKICAgIHJpc2luZzpbe3Q6IldheWFuYWQgcmVoYWJpbGl0YXRpb24iLHBjdDoiKzE2JSJ9XSxmYWxsaW5nOlt7dDoiVG91cmlzbSBkZWJhdGUiLHBjdDoiLTklIn1dLAogICAgYXJ0aWNsZXM6W3tzcmM6Ik1hdGhydWJodW1pIix0eHQ6IldheWFuYWQgcmVoYWJpbGl0YXRpb24gcGhhc2UgdHdvIGZhY2VzIGxhbmQgYWNxdWlzaXRpb24gZGVsYXlzIn1dLAogICAgc3VtbWFyeToiS2VyYWxhIG1haW50YWlucyBhIG1vZGVyYXRlLCBtZWFzdXJlZCBhdHRlbnRpb24gcHJvZmlsZS4gVGhlIFdheWFuYWQgcmVjb3Zlcnkgb3BlcmF0aW9uIGFuY2hvcnMgZW52aXJvbm1lbnRhbCBjb25zY2lvdXNuZXNzOyByZW1pdHRhbmNlIGFueGlldHkgaXMgdGhlIHVuZGVyY3VycmVudC4iLAogICAgdGltZWxpbmU6WzUwLDUxLDUyLDUyLDUzLDUzLDU0LDU0XSwKICAgIG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiVG91cmlzbSBvcHRpbWlzbSIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkZsb29kICYgZGlzYXN0ZXIgcmVzcG9uc2UiLHdoZW46IjMgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJXYXlhbmFkIHJlY292ZXJ5Iix3aGVuOiIyIG1vbnRocyBhZ28iLGNsczoicmVjZW50In0se3RvcGljOiJFY29ub215ICYgcmVtaXR0YW5jZXMiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiRGVsaGkiOnthdHRlbnRpb246ODEsZGVsdGE6OSx2ZWxvY2l0eTowLjI3LGVtb3Rpb25zOnthbnhpZXR5OjI4LGFuZ2VyOjI2LGhvcGU6MTIscHJpZGU6MTQsZmVhcjoyMH0sCiAgICBuYXJyYXRpdmVzOlt7bmFtZToiR292ZXJuYW5jZSIsdmFsOjI2LGRpcjoidXAifSx7bmFtZToiRW52aXJvbm1lbnQgLyBBaXIiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IkxhdyAmIG9yZGVyIix2YWw6MTgsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE2LGRpcjoiZmxhdCJ9LHtuYW1lOiJDb3JydXB0aW9uIix2YWw6MTYsZGlyOiJ1cCJ9XSwKICAgIHJpc2luZzpbe3Q6IkFpciBxdWFsaXR5IGVtZXJnZW5jeSIscGN0OiIrNDElIn0se3Q6IkwtRyB2cyBDTSBzdGFuZG9mZiIscGN0OiIrMjMlIn1dLGZhbGxpbmc6W3t0OiJNZXRybyBmYXJlIGhpa2UiLHBjdDoiLTEyJSJ9XSwKICAgIGFydGljbGVzOlt7c3JjOiJJbmRpYW4gRXhwcmVzcyIsdHh0OiJEZWxoaSBBUUkgcmUtZW50ZXJzIHNldmVyZSBiYW5kOyBHUkFQIFN0YWdlIDMgYWN0aXZhdGVkIn0se3NyYzoiSGluZHVzdGFuIFRpbWVzIix0eHQ6IkwtRyB3cml0ZXMgdG8gQ00gY2l0aW5nIGFkbWluaXN0cmF0aXZlIGRlbGF5cyJ9XSwKICAgIHN1bW1hcnk6IkRlbGhpIGlzIGNsaW1iaW5nIHNoYXJwbHkg4oCUIGFuIG9mZi1zZWFzb24gYWlyIHF1YWxpdHkgZW1lcmdlbmN5IGNvbGxpZGluZyB3aXRoIGEgcmVuZXdlZCBnb3Zlcm5hbmNlIHN0YW5kb2ZmLiBBbnhpZXR5IGlzIHRoZSBkb21pbmFudCByZWdpc3RlciBmb3IgdGhlIHNlY29uZCBjb25zZWN1dGl2ZSB3ZWVrLiIsCiAgICB0aW1lbGluZTpbNjgsNzAsNzIsNzQsNzYsNzgsNzksODFdLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJJbmZyYXN0cnVjdHVyZSAmIE1ldHJvIix3aGVuOiI2IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiR292ZXJuYW5jZSBzdGFuZG9mZiIsd2hlbjoiNCBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkFpciBxdWFsaXR5IGNyaXNpcyIsd2hlbjoiMyB3ZWVrcyBhZ28iLGNsczoicmVjZW50In0se3RvcGljOiJFbnZpcm9ubWVudCAmIGdvdmVybmFuY2UiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiR3VqYXJhdCI6e2F0dGVudGlvbjo1OSxkZWx0YTotMix2ZWxvY2l0eTotMC4wNSxlbW90aW9uczp7YW54aWV0eToxNCxhbmdlcjoxNCxob3BlOjI0LHByaWRlOjMyLGZlYXI6MTZ9LAogICAgbmFycmF0aXZlczpbe25hbWU6IkVjb25vbXkiLHZhbDozMCxkaXI6InVwIn0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MjIsZGlyOiJmbGF0In0se25hbWU6IlJlbGlnaW9uIix2YWw6MTQsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjE2LGRpcjoidXAifV0sCiAgICByaXNpbmc6W3t0OiJTZW1pY29uZHVjdG9yIHBsYW50IixwY3Q6IisxOCUifV0sZmFsbGluZzpbe3Q6IlN0YXR1ZSB0b3VyaXNtIixwY3Q6Ii0xNCUifV0sCiAgICBhcnRpY2xlczpbe3NyYzoiU2FuZGVzaCIsdHh0OiJEaG9sZXJhIFNJUiBhZGRzIHNlbWljb25kdWN0b3IgYW5jaG9yIHRlbmFudHMifV0sCiAgICBzdW1tYXJ5OiJHdWphcmF0J3MgbmFycmF0aXZlIHJlbWFpbnMgZWNvbm9taWMgYW5kIGluZnJhc3RydWN0dXJlLWxlZC4gUHJpZGUgaXMgdGhlIGRvbWluYW50IGVtb3Rpb25hbCByZWdpc3RlciDigJQgdW5jaGFyYWN0ZXJpc3RpYyByZWxhdGl2ZSB0byBuYXRpb25hbCBhdmVyYWdlLiIsCiAgICB0aW1lbGluZTpbNjIsNjEsNjEsNjAsNjAsNjAsNTksNTldLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJSZWxpZ2lvdXMgaW5mcmFzdHJ1Y3R1cmUiLHdoZW46IjcgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJFY29ub21pYyBpbnZlc3RtZW50IHB1c2giLHdoZW46IjQgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJNYW51ZmFjdHVyaW5nICYgZXhwb3J0cyIsd2hlbjoiMiBtb250aHMgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiU2VtaWNvbmR1Y3RvciAmIHRlY2giLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiUmFqYXN0aGFuIjp7YXR0ZW50aW9uOjU3LGRlbHRhOjEsdmVsb2NpdHk6MC4wMyxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxOCxob3BlOjE4LHByaWRlOjIwLGZlYXI6MjZ9LAogICAgbmFycmF0aXZlczpbe25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMCxkaXI6ImZsYXQifV0sCiAgICByaXNpbmc6W3t0OiJIZWF0IHdhdmUgd2FybmluZ3MiLHBjdDoiKzM0JSJ9LHt0OiJXZXN0ZXJuIGJvcmRlciBhbGVydHMiLHBjdDoiKzE5JSJ9XSxmYWxsaW5nOlt7dDoiVG91cmlzbSBvZmYtc2Vhc29uIixwY3Q6Ii0yMiUifV0sCiAgICBhcnRpY2xlczpbe3NyYzoiUmFqYXN0aGFuIFBhdHJpa2EiLHR4dDoiQm9yZGVyIGRpc3RyaWN0cyBzZWUgZnJlc2ggc2VjdXJpdHkgZHJpbGxzIn1dLAogICAgc3VtbWFyeToiUmFqYXN0aGFuJ3MgYXR0ZW50aW9uIGlzIHJpc2luZyBvbiBlbnZpcm9ubWVudCDigJQgaGVhdCB3YXZlIGNvdmVyYWdlIHNwaWtpbmcgc2hhcnBseS4gQm9yZGVyIHNlY3VyaXR5IGlzIGEgc3RlYWR5IHNlY29uZGFyeSBjdXJyZW50IGluIHdlc3Rlcm4gZGlzdHJpY3RzLiIsCiAgICB0aW1lbGluZTpbNTUsNTUsNTYsNTYsNTYsNTcsNTcsNTddLAogICAgbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJTdGF0ZSBlbGVjdGlvbiBhZnRlcm1hdGgiLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJGYXJtZXIgZGlzdHJlc3MiLHdoZW46IjQgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJUb3VyaXNtICYgZWNvbm9teSIsd2hlbjoiMiBtb250aHMgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiSGVhdCB3YXZlICYgYm9yZGVyIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIk1hZGh5YSBQcmFkZXNoIjp7YXR0ZW50aW9uOjUyLGRlbHRhOjAsdmVsb2NpdHk6MC4wMSxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxNixob3BlOjIwLHByaWRlOjIyLGZlYXI6MjR9LG5hcnJhdGl2ZXM6W3tuYW1lOiJGYXJtZXIgaXNzdWVzIix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJSZWxpZ2lvbiIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiSW5mcmFzdHJ1Y3R1cmUiLHZhbDoyMCxkaXI6InVwIn1dLHJpc2luZzpbe3Q6IlNveWJlYW4gTVNQIGRlYmF0ZSIscGN0OiIrMTQlIn1dLGZhbGxpbmc6W3t0OiJDYWJpbmV0IGV4cGFuc2lvbiIscGN0OiItMTglIn1dLGFydGljbGVzOlt7c3JjOiJQYXRyaWthIix0eHQ6IlNveWJlYW4gZ3Jvd2VycyBpbiBNYWx3YSBkZW1hbmQgTVNQIHJldmlldyJ9XSxzdW1tYXJ5OiJNUCBzdGFibGUgd2l0aCBhZ3JpY3VsdHVyZS1lY29ub215IG5hcnJhdGl2ZXMgZG9taW5hdGluZy4gTm8gc2hhcnAgbW92ZW1lbnQgdGhpcyBjeWNsZS4iLHRpbWVsaW5lOls1MSw1Miw1Miw1Miw1Miw1Miw1Miw1Ml0sbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJTdGF0ZSBwb2xsIGNhbXBhaWducyIsd2hlbjoiOCBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlJlbGlnaW91cyBzZW50aW1lbnQiLHdoZW46IjUgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJBZ3JpY3VsdHVyYWwgZGlzdHJlc3MiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJJbmZyYXN0cnVjdHVyZSBwdXNoIix3aGVuOiJFbWVyZ2luZyIsY2xzOiJyZWNlbnQifV19LAogICJQdW5qYWIiOnthdHRlbnRpb246NjYsZGVsdGE6Myx2ZWxvY2l0eTowLjExLGVtb3Rpb25zOnthbnhpZXR5OjIyLGFuZ2VyOjI2LGhvcGU6MTIscHJpZGU6MjIsZmVhcjoxOH0sbmFycmF0aXZlczpbe25hbWU6IkZhcm1lciBpc3N1ZXMiLHZhbDoyOCxkaXI6InVwIn0se25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkxhdyAmIG9yZGVyIix2YWw6MTgsZGlyOiJ1cCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTQsZGlyOiJmbGF0In0se25hbWU6IlJlbGlnaW9uIix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IlN0dWJibGUgcG9saWN5IixwY3Q6IisyMSUifSx7dDoiQm9yZGVyIGRyb25lIHNpZ2h0aW5ncyIscGN0OiIrMTclIn1dLGZhbGxpbmc6W3t0OiJQb3dlciBjcmlzaXMiLHBjdDoiLTE0JSJ9XSxhcnRpY2xlczpbe3NyYzoiUHVuamFiaSBUcmlidW5lIix0eHQ6IlN0dWJibGUgbWFuYWdlbWVudCBwbGFuIHVudmVpbGVkIGFoZWFkIG9mIHBhZGR5IGhhcnZlc3QifV0sc3VtbWFyeToiUHVuamFiIHJpc2luZyBvbiB0d2luIHRyYWNrczogYWdyaWN1bHR1cmUgcG9saWN5IGFuZCBib3JkZXIgc2VjdXJpdHkuIEFuZ2VyIHJlbWFpbnMgdGhlIGRvbWluYW50IGVtb3Rpb25hbCB0b25lLiIsdGltZWxpbmU6WzYyLDYzLDYzLDY0LDY0LDY1LDY1LDY2XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IkZhcm0gbGF3IGFmdGVybWF0aCIsd2hlbjoiOSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlBvd2VyICYgZWxlY3RyaWNpdHkiLHdoZW46IjQgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJTdHViYmxlIGJ1cm5pbmcgcG9saWN5Iix3aGVuOiI2IHdlZWtzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IkJvcmRlciAmIGRyb25lcyIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAogICJIYXJ5YW5hIjp7YXR0ZW50aW9uOjYxLGRlbHRhOjIsdmVsb2NpdHk6MC4wNyxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoyMixob3BlOjE4LHByaWRlOjIyLGZlYXI6MjB9LG5hcnJhdGl2ZXM6W3tuYW1lOiJGYXJtZXIgaXNzdWVzIix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjIwLGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJDYXN0ZSBwb2xpdGljcyIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJTcG9ydHMiLHZhbDoyMCxkaXI6InVwIn1dLHJpc2luZzpbe3Q6Ik5DUiBwb2xsdXRpb24gc3BpbGxvdmVyIixwY3Q6IisxOSUifV0sZmFsbGluZzpbe3Q6IlJlYWwgZXN0YXRlIixwY3Q6Ii04JSJ9XSxhcnRpY2xlczpbe3NyYzoiRGFpbmlrIEphZ3JhbiIsdHh0OiJIYXJ5YW5hIHdyZXN0bGVycyByZWFjaCBuYXRpb25hbCBmaW5hbHMifV0sc3VtbWFyeToiSGFyeWFuYSBzaG93cyBhIGJhbGFuY2VkIG5hcnJhdGl2ZSBtaXguIFNwb3J0cyBwcmlkZSBhbmQgZmFybWVyIGlzc3VlcyBjby1hbmNob3IgcHVibGljIGF0dGVudGlvbiBpbiBhbiB1bnVzdWFsIHBhaXJpbmcuIix0aW1lbGluZTpbNTgsNTksNTksNjAsNjAsNjEsNjEsNjFdLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiRWxlY3Rpb24gY2FtcGFpZ25zIix3aGVuOiI3IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiV3Jlc3RsZXIgcHJvdGVzdCIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkZhcm1lciAmIGNhc3RlIGlzc3VlcyIsd2hlbjoiMyBtb250aHMgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiUG9sbHV0aW9uICYgc3BvcnRzIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIlRlbGFuZ2FuYSI6e2F0dGVudGlvbjo2MyxkZWx0YTo0LHZlbG9jaXR5OjAuMTMsZW1vdGlvbnM6e2FueGlldHk6MTYsYW5nZXI6MTgsaG9wZToyMixwcmlkZToyMixmZWFyOjIyfSxuYXJyYXRpdmVzOlt7bmFtZToiRWNvbm9teSAvIElUIix2YWw6MjQsZGlyOiJ1cCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MjAsZGlyOiJ1cCJ9LHtuYW1lOiJXYXRlciBkaXNwdXRlcyIsdmFsOjE4LGRpcjoidXAifSx7bmFtZToiRWR1Y2F0aW9uIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IlJlZ2lvbmFsIGlkZW50aXR5Iix2YWw6MjAsZGlyOiJ1cCJ9XSxyaXNpbmc6W3t0OiJIeWRlcmFiYWQgY2x1c3RlciBleHBhbnNpb24iLHBjdDoiKzE2JSJ9XSxmYWxsaW5nOlt7dDoiT2xkIGNpdHkgcmVkZXZlbG9wbWVudCIscGN0OiItMTElIn1dLGFydGljbGVzOlt7c3JjOiJTYWtzaGkiLHR4dDoiVGVsYW5nYW5hIGFkZHMgaW52ZXN0bWVudCBjb21taXRtZW50cyBhdCBnbG9iYWwgc3VtbWl0In1dLHN1bW1hcnk6IlRlbGFuZ2FuYSByaXNpbmcgb24gZWNvbm9taWMgYW5kIGdvdmVybmFuY2UgdHJhY2tzIHNpbXVsdGFuZW91c2x5LiBIeWRlcmFiYWQncyB0ZWNoIGNsdXN0ZXIgaXMgdGhlIGRvbWluYW50IHByaWRlIG5hcnJhdGl2ZS4iLHRpbWVsaW5lOls1OCw1OSw2MCw2MCw2MSw2Miw2Miw2M10sbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJTdGF0ZSBmb3JtYXRpb24gYW5uaXZlcnNhcnkiLHdoZW46IjYgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJHb3Zlcm5tZW50IGZvcm1hdGlvbiIsd2hlbjoiNCBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkVjb25vbWljIGludmVzdG1lbnQiLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IklUICYgZ292ZXJuYW5jZSIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAogICJBbmRocmEgUHJhZGVzaCI6e2F0dGVudGlvbjo1OCxkZWx0YToyLHZlbG9jaXR5OjAuMDgsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MTgsaG9wZToyMCxwcmlkZToyMixmZWFyOjIyfSxuYXJyYXRpdmVzOlt7bmFtZToiRWNvbm9teSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiQ2FwaXRhbCByb3ciLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IldhdGVyIGRpc3B1dGVzIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQW1hcmF2YXRpIHJlc3RhcnQiLHBjdDoiKzIyJSJ9XSxmYWxsaW5nOlt7dDoiTGlxdW9yIHBvbGljeSIscGN0OiItOSUifV0sYXJ0aWNsZXM6W3tzcmM6IkFuZGhyYSBKeW90aHkiLHR4dDoiQW1hcmF2YXRpIGNhcGl0YWwgY2l0eSBjb25zdHJ1Y3Rpb24gcmVzdGFydCBmb3JtYWxseSBhcHByb3ZlZCJ9XSxzdW1tYXJ5OiJBbmRocmEgcmlzaW5nIG9uIEFtYXJhdmF0aSByZXN0YXJ0LiBQcmlkZSBhbmQgaG9wZSBhcmUgdGhlIHJpc2luZyBlbW90aW9uYWwgdG9uZXMgYWZ0ZXIgeWVhcnMgb2YgY2FwaXRhbCBjaXR5IHVuY2VydGFpbnR5LiIsdGltZWxpbmU6WzU1LDU2LDU2LDU3LDU3LDU4LDU4LDU4XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IkNhcGl0YWwgY2l0eSBkaXNwdXRlIix3aGVuOiIxIHllYXIgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkdvdmVybm1lbnQgdHJhbnNpdGlvbiIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkFtYXJhdmF0aSByZXN0YXJ0Iix3aGVuOiIyIG1vbnRocyBhZ28iLGNsczoicmVjZW50In0se3RvcGljOiJEZXZlbG9wbWVudCAmIHByaWRlIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIk9kaXNoYSI6e2F0dGVudGlvbjo0OSxkZWx0YTotMSx2ZWxvY2l0eTotMC4wMixlbW90aW9uczp7YW54aWV0eToxNixhbmdlcjoxNCxob3BlOjI0LHByaWRlOjI0LGZlYXI6MjJ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJFY29ub215Iix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJFbnZpcm9ubWVudCIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiSW5mcmFzdHJ1Y3R1cmUiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJUcmliYWwgaXNzdWVzIix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkN5Y2xvbmUgcHJlcGFyZWRuZXNzIixwY3Q6IisxOCUifV0sZmFsbGluZzpbe3Q6Ik1pbmluZyBiaWQgcm91bmQiLHBjdDoiLTEyJSJ9XSxhcnRpY2xlczpbe3NyYzoiRGhhcml0cmkiLHR4dDoiUHJlLW1vbnNvb24gY3ljbG9uZSBkcmlsbCBhY3Jvc3MgY29hc3RhbCBkaXN0cmljdHMifV0sc3VtbWFyeToiT2Rpc2hhIHF1aWV0LCBkb21pbmF0ZWQgYnkgZW52aXJvbm1lbnRhbCBwcmVwYXJlZG5lc3MuIFRoZSBjeWNsb25lIGRyaWxsIHNlYXNvbiBpcyB0aGUgcHJpbWFyeSBjb3ZlcmFnZSBhbmNob3IuIix0aW1lbGluZTpbNTAsNTAsNTAsNTAsNDksNDksNDksNDldLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiU3RhdGUgZWxlY3Rpb24iLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJHb3Zlcm5tZW50IGZvcm1hdGlvbiIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkluZnJhc3RydWN0dXJlIHB1c2giLHdoZW46IjMgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IkVudmlyb25tZW50ICYgY3ljbG9uZSBwcmVwIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XX0sCiAgIkpoYXJraGFuZCI6e2F0dGVudGlvbjo1NixkZWx0YTozLHZlbG9jaXR5OjAuMTIsZW1vdGlvbnM6e2FueGlldHk6MjAsYW5nZXI6MjIsaG9wZToxNCxwcmlkZToxNixmZWFyOjI4fSxuYXJyYXRpdmVzOlt7bmFtZToiVHJpYmFsIGlzc3VlcyIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiTWluaW5nIC8gRWNvbm9teSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiTWlncmF0aW9uIix2YWw6MTgsZGlyOiJ1cCJ9XSxyaXNpbmc6W3t0OiJGb3Jlc3QgcmlnaHRzIGRpc3B1dGUiLHBjdDoiKzI0JSJ9XSxmYWxsaW5nOlt7dDoiQ29hbCBibG9jayBhdWN0aW9uIixwY3Q6Ii0xMSUifV0sYXJ0aWNsZXM6W3tzcmM6IlByYWJoYXQgS2hhYmFyIix0eHQ6IlRyaWJhbCBvcmdhbml6YXRpb25zIHN0YWdlIHJhbGx5IG9uIGxhbmQgcmlnaHRzIn1dLHN1bW1hcnk6IkpoYXJraGFuZCBjbGltYmluZyBvbiB0cmliYWwgbGFuZCByaWdodHMgYW5kIG1pbmluZyBpbnRlcnNlY3Rpb25zLiBGZWFyIGlzIHRoZSBkb21pbmFudCBlbW90aW9uYWwgcmVnaXN0ZXIg4oCUIGRyaXZlbiBieSBkaXNwbGFjZW1lbnQgbmFycmF0aXZlcy4iLHRpbWVsaW5lOls1Miw1Myw1NCw1NCw1NSw1NSw1Niw1Nl0sbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJTdGF0ZSBlbGVjdGlvbnMiLHdoZW46IjYgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJNaW5pbmcgJiBlY29ub215Iix3aGVuOiI0IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiVHJpYmFsIGxhbmQgY29uZmxpY3QiLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IkZvcmVzdCByaWdodHMgJiBkaXNwbGFjZW1lbnQiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiQ2hoYXR0aXNnYXJoIjp7YXR0ZW50aW9uOjUxLGRlbHRhOjEsdmVsb2NpdHk6MC4wNCxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxOCxob3BlOjE4LHByaWRlOjIwLGZlYXI6MjZ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJMYXcgJiBvcmRlciIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiVHJpYmFsIGlzc3VlcyIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiQmFzdGFyIG9wZXJhdGlvbnMiLHBjdDoiKzE2JSJ9XSxmYWxsaW5nOlt7dDoiRm9yZXN0IGF1Y3Rpb24iLHBjdDoiLTklIn1dLGFydGljbGVzOlt7c3JjOiJQYXRyaWthIix0eHQ6IlNlY3VyaXR5IG9wZXJhdGlvbnMgaW4gQmFzdGFyIGxlYWQgdG8gcmVjb3JkIHN1cnJlbmRlciBudW1iZXJzIn1dLHN1bW1hcnk6IkNoaGF0dGlzZ2FyaCBhbmNob3JlZCBpbiBCYXN0YXIgc2VjdXJpdHkgb3BlcmF0aW9ucy4gRmVhciBpcyB0aGUgZG9taW5hbnQgZW1vdGlvbmFsIHJlZ2lzdGVyIGluIHRoZSBzb3V0aGVybiBkaXN0cmljdHMuIix0aW1lbGluZTpbNTAsNTAsNTAsNTAsNTEsNTEsNTEsNTFdLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiU3RhdGUgZWxlY3Rpb25zIix3aGVuOiI4IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiQW50aS1NYW9pc3Qgb3BlcmF0aW9ucyIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlRyaWJhbCAmIGZvcmVzdCBwb2xpY3kiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJEZXZlbG9wbWVudCBwdXNoIix3aGVuOiJFbWVyZ2luZyIsY2xzOiJyZWNlbnQifV19LAogICJBc3NhbSI6e2F0dGVudGlvbjo2MCxkZWx0YTozLHZlbG9jaXR5OjAuMTEsZW1vdGlvbnM6e2FueGlldHk6MjIsYW5nZXI6MjAsaG9wZToxOCxwcmlkZToyMCxmZWFyOjIwfSxuYXJyYXRpdmVzOlt7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiUmVsaWdpb24iLHZhbDoyMCxkaXI6InVwIn0se25hbWU6IkVudmlyb25tZW50IC8gRmxvb2RzIix2YWw6MjAsZGlyOiJ1cCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IlJlZ2lvbmFsIGlkZW50aXR5Iix2YWw6MjAsZGlyOiJ1cCJ9XSxyaXNpbmc6W3t0OiJOUkMgdXBkYXRlIHB1c2giLHBjdDoiKzE3JSJ9XSxmYWxsaW5nOlt7dDoiVGVhIGluZHVzdHJ5IixwY3Q6Ii04JSJ9XSxhcnRpY2xlczpbe3NyYzoiQXNvbWl5YSBQcmF0aWRpbiIsdHh0OiJTdGF0ZSBhbm5vdW5jZXMgZnJlc2ggcHVzaCBvbiBOUkMgdmVyaWZpY2F0aW9uIHRpbWVsaW5lcyJ9XSxzdW1tYXJ5OiJBc3NhbSByaXNpbmcgb24gZG9jdW1lbnRhdGlvbiBwb2xpdGljcyBhbmQgcHJlLW1vbnNvb24gZmxvb2QgcHJlcGFyZWRuZXNzLiBOYXJyYXRpdmVzIHNwbGl0IGJldHdlZW4gaWRlbnRpdHkgYW5kIGVudmlyb25tZW50LiIsdGltZWxpbmU6WzU3LDU4LDU4LDU5LDU5LDYwLDYwLDYwXSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IlRlYSBpbmR1c3RyeSAmIGVjb25vbXkiLHdoZW46IjYgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJDaXRpemVuc2hpcCBkb2N1bWVudGF0aW9uIix3aGVuOiI0IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiTlJDIHB1c2giLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IkJvcmRlciAmIGZsb29kcyIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAogICJIaW1hY2hhbCBQcmFkZXNoIjp7YXR0ZW50aW9uOjM4LGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxNixhbmdlcjoxNCxob3BlOjI0LHByaWRlOjI4LGZlYXI6MTh9LG5hcnJhdGl2ZXM6W3tuYW1lOiJUb3VyaXNtIix2YWw6MjYsZGlyOiJmbGF0In0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkVkdWNhdGlvbiIsdmFsOjE4LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJDaGFyIERoYW0gcm9hZCB3b3JrIixwY3Q6IisxMSUifV0sZmFsbGluZzpbe3Q6IkhvdGVsIHJhdGUgY2FwIixwY3Q6Ii03JSJ9XSxhcnRpY2xlczpbe3NyYzoiQW1hciBVamFsYSIsdHh0OiJUb3VyaXN0IGFycml2YWxzIHRvIGhpbGwgc3RhdGlvbnMgcmlzZSBhaGVhZCBvZiBzdW1tZXIgcGVhayJ9XSxzdW1tYXJ5OiJIaW1hY2hhbCBpbiBsb3ctYXR0ZW50aW9uIHN1bW1lciBjYWRlbmNlLiBUb3VyaXNtIGVjb25vbXkgZG9taW5hdGVzOyBlbnZpcm9ubWVudGFsIGNvbmNlcm5zIGFyZSB0aGUgdW5kZXJjdXJyZW50LiIsdGltZWxpbmU6WzM4LDM4LDM4LDM4LDM4LDM4LDM4LDM4XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IkRpc2FzdGVyIHJlY29uc3RydWN0aW9uIix3aGVuOiI4IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiVG91cmlzbSByZXZpdmFsIix3aGVuOiI0IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiSW5mcmFzdHJ1Y3R1cmUgZGV2ZWxvcG1lbnQiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJFbnZpcm9ubWVudCBtb25pdG9yaW5nIix3aGVuOiJFbWVyZ2luZyIsY2xzOiJyZWNlbnQifV19LAogICJVdHRhcmFraGFuZCI6e2F0dGVudGlvbjo0MSxkZWx0YToxLHZlbG9jaXR5OjAuMDMsZW1vdGlvbnM6e2FueGlldHk6MTYsYW5nZXI6MTQsaG9wZToyMixwcmlkZToyOCxmZWFyOjIwfSxuYXJyYXRpdmVzOlt7bmFtZToiUmVsaWdpb24gLyBQaWxncmltYWdlIix2YWw6MjYsZGlyOiJ1cCJ9LHtuYW1lOiJFbnZpcm9ubWVudCIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiVG91cmlzbSIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTYsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkNoYXIgRGhhbSB5YXRyYSBwcmVwIixwY3Q6IisxOSUifV0sZmFsbGluZzpbe3Q6IlBvd2VyIHRhcmlmZiBkZWJhdGUiLHBjdDoiLTklIn1dLGFydGljbGVzOlt7c3JjOiJIaW5kdXN0YW4gSGluZGkiLHR4dDoiQ2hhciBEaGFtIHlhdHJhIHJlZ2lzdHJhdGlvbnMgY3Jvc3MgbGFzdCB5ZWFyJ3MgcmVjb3JkIn1dLHN1bW1hcnk6IlV0dGFyYWtoYW5kIHN0ZWFkeSBvbiBwaWxncmltYWdlIGFuZCB0b3VyaXNtLiBQcmlkZSBpcyB0aGUgZG9taW5hbnQgcmVnaXN0ZXIg4oCUIENoYXIgRGhhbSBzZWFzb24gZHJpdmVzIGNvbGxlY3RpdmUgb3B0aW1pc20uIix0aW1lbGluZTpbMzksNDAsNDAsNDAsNDEsNDEsNDEsNDFdLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiRGlzYXN0ZXIgcmVjb3ZlcnkiLHdoZW46IjcgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJJbmZyYXN0cnVjdHVyZSBwdXNoIix3aGVuOiI0IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiUGlsZ3JpbWFnZSByZXZpdmFsIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9LHt0b3BpYzoiRW52aXJvbm1lbnQgY29uY2VybnMiLHdoZW46IkVtZXJnaW5nIixjbHM6InJlY2VudCJ9XX0sCiAgIk1hbmlwdXIiOnthdHRlbnRpb246NjQsZGVsdGE6NSx2ZWxvY2l0eTowLjE5LGVtb3Rpb25zOnthbnhpZXR5OjI4LGFuZ2VyOjI2LGhvcGU6MTAscHJpZGU6MTQsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IkxhdyAmIG9yZGVyIix2YWw6MzAsZGlyOiJ1cCJ9LHtuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MjIsZGlyOiJ1cCJ9LHtuYW1lOiJJZGVudGl0eSAvIEV0aG5pYyIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE0LGRpcjoiZmxhdCJ9LHtuYW1lOiJNaWdyYXRpb24iLHZhbDoxMCxkaXI6InVwIn1dLHJpc2luZzpbe3Q6IkV0aG5pYyB0ZW5zaW9ucyByZXN1cmZhY2UiLHBjdDoiKzI3JSJ9LHt0OiJBRlNQQSBkZWJhdGUiLHBjdDoiKzE0JSJ9XSxmYWxsaW5nOlt7dDoiQ2FiaW5ldCBzaHVmZmxlIHRhbGsiLHBjdDoiLTEzJSJ9XSxhcnRpY2xlczpbe3NyYzoiSW1waGFsIEZyZWUgUHJlc3MiLHR4dDoiRnJlc2ggdGVuc2lvbnMgcmVwb3J0ZWQgaW4gdmFsbGV5LWhpbGxzIGJvcmRlciB2aWxsYWdlcyJ9XSxzdW1tYXJ5OiJNYW5pcHVyIHJpc2luZyBzaGFycGx5IG9uIGV0aG5pYy1pZGVudGl0eSBuYXJyYXRpdmVzLiBUaGUgYW54aWV0eS1hbmdlciBlbW90aW9uYWwgcHJvZmlsZSBoYXMgaGVsZCBmb3IgdGhyZWUgY29uc2VjdXRpdmUgbW9udGhzLiIsdGltZWxpbmU6WzU1LDU3LDU5LDYwLDYxLDYyLDYzLDY0XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IlBvc3QtY29uZmxpY3QgcmVjb3ZlcnkiLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJFdGhuaWMgY29uZmxpY3QgZXNjYWxhdGlvbiIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkFGU1BBICYgZ292ZXJuYW5jZSIsd2hlbjoiMiBtb250aHMgYWdvIixjbHM6InJlY2VudCJ9LHt0b3BpYzoiRXRobmljIHRlbnNpb25zIMK3IEJvcmRlciIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAogICJKYW1tdSBhbmQgS2FzaG1pciI6e2F0dGVudGlvbjo3MixkZWx0YTo0LHZlbG9jaXR5OjAuMTcsZW1vdGlvbnM6e2FueGlldHk6MjIsYW5nZXI6MjIsaG9wZToxOCxwcmlkZToxOCxmZWFyOjIwfSxuYXJyYXRpdmVzOlt7bmFtZToiTGF3ICYgb3JkZXIiLHZhbDoyNixkaXI6InVwIn0se25hbWU6IkJvcmRlciBpc3N1ZXMiLHZhbDoyNCxkaXI6InVwIn0se25hbWU6IlRvdXJpc20iLHZhbDoxOCxkaXI6InVwIn0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiSWRlbnRpdHkiLHZhbDoxNixkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiVG91cmlzbSBhcnJpdmFscyBoaWdoIixwY3Q6IisyMSUifSx7dDoiTG9DIGluY2lkZW50cyIscGN0OiIrMTUlIn1dLGZhbGxpbmc6W3t0OiJTdGF0ZWhvb2QgZGViYXRlIixwY3Q6Ii05JSJ9XSxhcnRpY2xlczpbe3NyYzoiR3JlYXRlciBLYXNobWlyIix0eHQ6IlRvdXJpc3QgYXJyaXZhbHMgdG8gUGFoYWxnYW0sIEd1bG1hcmcgY3Jvc3Mgc2Vhc29uYWwgcmVjb3JkIn0se3NyYzoiRGFpbHkgRXhjZWxzaW9yIix0eHQ6IkxvQyByZW1haW5zIGFjdGl2ZTsgc2VjdXJpdHkgZm9yY2VzIHJlc3BvbmQgdG8gaW5jaWRlbnQifV0sc3VtbWFyeToiSiZLIG9uIHR3aW4gdHJhY2tzIOKAlCByZWNvcmQgdG91cmlzbSBhcnJpdmFscyBhbmQgcmVuZXdlZCBMb0MgdGVuc2lvbnMuIEFuIHVudXN1YWwgY29leGlzdGVuY2Ugb2YgaG9wZSBhbmQgYW54aWV0eSBpbiB0aGUgc2FtZSBuZXdzIGN5Y2xlLiIsdGltZWxpbmU6WzY1LDY3LDY4LDY5LDcwLDcxLDcyLDcyXSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IkFydGljbGUgMzcwIGFmdGVybWF0aCIsd2hlbjoiMSB5ZWFyIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJHb3Zlcm5hbmNlIG5vcm1hbGlzYXRpb24iLHdoZW46IjYgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJUb3VyaXNtIHJldml2YWwiLHdoZW46IjMgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6IlRvdXJpc20gJiBMb0MgdGVuc2lvbnMiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In1dfSwKICAiR29hIjp7YXR0ZW50aW9uOjM0LGRlbHRhOi0xLHZlbG9jaXR5Oi0wLjA0LGVtb3Rpb25zOnthbnhpZXR5OjE0LGFuZ2VyOjEyLGhvcGU6MjIscHJpZGU6MzAsZmVhcjoyMn0sbmFycmF0aXZlczpbe25hbWU6IlRvdXJpc20iLHZhbDozMCxkaXI6ImRvd24ifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoyMixkaXI6InVwIn0se25hbWU6IkVjb25vbXkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE2LGRpcjoiZmxhdCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjE0LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJDb2FzdGFsIHpvbmUgZW5mb3JjZW1lbnQiLHBjdDoiKzEyJSJ9XSxmYWxsaW5nOlt7dDoiVG91cmlzdCBhcnJpdmFscyIscGN0OiItMjElIn1dLGFydGljbGVzOlt7c3JjOiJIZXJhbGQgR29hIix0eHQ6IkNSWiBlbmZvcmNlbWVudCBkcml2ZSBiZWdpbnMgaW4gbm9ydGggR29hIn1dLHN1bW1hcnk6IkdvYSBpbiBvZmZzZWFzb24gcXVpZXQuIENvYXN0YWwgcmVndWxhdGlvbiBlbmZvcmNlbWVudCBpcyB0aGUgb25seSByaXNpbmcgc2lnbmFsIGluIGFuIG90aGVyd2lzZSBzdWJkdWVkIGN5Y2xlLiIsdGltZWxpbmU6WzM2LDM2LDM1LDM1LDM0LDM0LDM0LDM0XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IlRvdXJpc20gYm9vbSIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkluZnJhc3RydWN0dXJlIGRlYmF0ZSIsd2hlbjoiMyBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkVudmlyb25tZW50IGVuZm9yY2VtZW50Iix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9LHt0b3BpYzoiT2Zmc2Vhc29uIHF1aWV0Iix3aGVuOiJOb3ciLGNsczoicmVjZW50In1dfSwKICAiU2lra2ltIjp7YXR0ZW50aW9uOjIyLGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxNCxhbmdlcjoxMCxob3BlOjI2LHByaWRlOjI4LGZlYXI6MjJ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJFbnZpcm9ubWVudCIsdmFsOjMwLGRpcjoidXAifSx7bmFtZToiVG91cmlzbSIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiRWNvbm9teSIsdmFsOjE0LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJHbGFjaWVyIG1vbml0b3JpbmciLHBjdDoiKzklIn1dLGZhbGxpbmc6W10sYXJ0aWNsZXM6W3tzcmM6IlNpa2tpbSBFeHByZXNzIix0eHQ6IkdsYWNpZXIgc3VydmV5IHNob3dzIGFjY2VsZXJhdGVkIHJldHJlYXQgaW4gbm9ydGggU2lra2ltIn1dLHN1bW1hcnk6IlNpa2tpbSBkcmF3cyBtaW5pbWFsIG5hdGlvbmFsIGF0dGVudGlvbi4gRW52aXJvbm1lbnQtZm9jdXNlZCBuYXJyYXRpdmVzIGRvbWluYXRlIGEgc21hbGwgYnV0IGNvbnNpc3RlbnQgc2lnbmFsIHZvbHVtZS4iLHRpbWVsaW5lOlsyMiwyMiwyMiwyMiwyMiwyMiwyMiwyMl0sbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJGbG9vZCBkaXNhc3RlciIsd2hlbjoiOCBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlJlY29uc3RydWN0aW9uIix3aGVuOiI1IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiR2xhY2llciAmIGVudmlyb25tZW50Iix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9LHt0b3BpYzoiVG91cmlzbSByZXZpdmFsIix3aGVuOiJFbWVyZ2luZyIsY2xzOiJyZWNlbnQifV19LAogICJOYWdhbGFuZCI6e2F0dGVudGlvbjoyOCxkZWx0YTowLHZlbG9jaXR5OjAuMDEsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MTQsaG9wZToyMCxwcmlkZToyNCxmZWFyOjI0fSxuYXJyYXRpdmVzOlt7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJJZGVudGl0eSIsdmFsOjI0LGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiTmFnYSBmcmFtZXdvcmsgYWdyZWVtZW50IixwY3Q6Iis4JSJ9XSxmYWxsaW5nOltdLGFydGljbGVzOlt7c3JjOiJNb3J1bmcgRXhwcmVzcyIsdHh0OiJGcmFtZXdvcmsgYWdyZWVtZW50IGltcGxlbWVudGF0aW9uIGRpc2N1c3Npb25zIHJlc3VtZSJ9XSxzdW1tYXJ5OiJOYWdhbGFuZCBxdWlldGx5IHBlcnNpc3RlbnQuIElkZW50aXR5IGFuZCBmcmFtZXdvcmsgbmFycmF0aXZlcyBob2xkIHN0ZWFkeSBhdCBsb3cgdm9sdW1lLiIsdGltZWxpbmU6WzI4LDI4LDI4LDI4LDI4LDI4LDI4LDI4XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IlBlYWNlIHByb2Nlc3MiLHdoZW46IjEgeWVhciBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiU3RhbGxlZCBmcmFtZXdvcmsgdGFsa3MiLHdoZW46IjYgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJJZGVudGl0eSBwb2xpdGljcyIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifSx7dG9waWM6IkZyYW1ld29yayByZXZpdmFsIix3aGVuOiJFbWVyZ2luZyIsY2xzOiJyZWNlbnQifV19LAogICJNaXpvcmFtIjp7YXR0ZW50aW9uOjE0LGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxMCxhbmdlcjo4LGhvcGU6MzAscHJpZGU6MzIsZmVhcjoyMH0sbmFycmF0aXZlczpbe25hbWU6IkdvdmVybmFuY2UiLHZhbDoyNCxkaXI6ImZsYXQifSx7bmFtZToiSWRlbnRpdHkiLHZhbDoyMCxkaXI6ImZsYXQifSx7bmFtZToiRWNvbm9teSIsdmFsOjE4LGRpcjoiZmxhdCJ9LHtuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkNyb3NzLWJvcmRlciByZWZ1Z2VlIHBvbGljeSIscGN0OiIrNiUifV0sZmFsbGluZzpbXSxhcnRpY2xlczpbe3NyYzoiVmFuZ2xhaW5pIix0eHQ6IlN0YXRlIHVwZGF0ZXMgcmVnaXN0cmF0aW9uIG5vcm1zIGZvciBjcm9zcy1ib3JkZXIgcmVmdWdlZXMifV0sc3VtbWFyeToiTWl6b3JhbSBpcyB0aGUgcXVpZXRlc3Qgc3RhdGUgaW4gSW5kaWEncyBjdXJyZW50IGF0dGVudGlvbiBsYW5kc2NhcGUuIENyb3NzLWJvcmRlciBodW1hbml0YXJpYW4gbmFycmF0aXZlcyBhcmUgdGhlIG9ubHkgc2lnbmFsLiIsdGltZWxpbmU6WzE0LDE0LDE0LDE0LDE0LDE0LDE0LDE0XSxuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6Ik15YW5tYXIgcmVmdWdlZSBjcmlzaXMiLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJHb3Zlcm5tZW50IGZvcm1hdGlvbiIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkNyb3NzLWJvcmRlciBwb2xpY3kiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJFY29ub21pYyBkZXZlbG9wbWVudCIsd2hlbjoiRW1lcmdpbmciLGNsczoicmVjZW50In1dfSwKICAiVHJpcHVyYSI6e2F0dGVudGlvbjozMSxkZWx0YToxLHZlbG9jaXR5OjAuMDQsZW1vdGlvbnM6e2FueGlldHk6MTgsYW5nZXI6MTQsaG9wZToyMixwcmlkZToyMixmZWFyOjI0fSxuYXJyYXRpdmVzOlt7bmFtZToiQm9yZGVyIGlzc3VlcyIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiRWNvbm9teSIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJHb3Zlcm5hbmNlIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IklkZW50aXR5Iix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTgsZGlyOiJmbGF0In1dLHJpc2luZzpbe3Q6IkNyb3NzLWJvcmRlciB0cmFkZSByb3V0ZSIscGN0OiIrMTElIn1dLGZhbGxpbmc6W3t0OiJUcmliYWwgY291bmNpbCBtZWV0IixwY3Q6Ii03JSJ9XSxhcnRpY2xlczpbe3NyYzoiVHJpcHVyYSBUaW1lcyIsdHh0OiJOZXcgY3Jvc3MtYm9yZGVyIHRyYWRlIGNvcnJpZG9yIHdpdGggQmFuZ2xhZGVzaCBhbm5vdW5jZWQifV0sc3VtbWFyeToiVHJpcHVyYSBzbG93bHkgcmlzaW5nIG9uIGNyb3NzLWJvcmRlciBlY29ub21pYyBuYXJyYXRpdmVzLiBUaGUgQmFuZ2xhZGVzaCB0cmFkZSBjb3JyaWRvciBpcyB0aGUgZGVmaW5pbmcgc3RvcnkuIix0aW1lbGluZTpbMzAsMzAsMzAsMzEsMzEsMzEsMzEsMzFdLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiUG9saXRpY2FsIHRyYW5zaXRpb24iLHdoZW46IjggbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJJbmZyYXN0cnVjdHVyZSBkZXZlbG9wbWVudCIsd2hlbjoiNCBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkJhbmdsYWRlc2ggdHJhZGUiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJCb3JkZXIgJiBpZGVudGl0eSIsd2hlbjoiRW1lcmdpbmciLGNsczoicmVjZW50In1dfSwKICAiTWVnaGFsYXlhIjp7YXR0ZW50aW9uOjI2LGRlbHRhOjAsdmVsb2NpdHk6MCxlbW90aW9uczp7YW54aWV0eToxNCxhbmdlcjoxMixob3BlOjI0LHByaWRlOjI4LGZlYXI6MjJ9LG5hcnJhdGl2ZXM6W3tuYW1lOiJFbnZpcm9ubWVudCIsdmFsOjI0LGRpcjoidXAifSx7bmFtZToiVG91cmlzbSIsdmFsOjIyLGRpcjoiZmxhdCJ9LHtuYW1lOiJJZGVudGl0eSIsdmFsOjIwLGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MTYsZGlyOiJmbGF0In0se25hbWU6IkdvdmVybmFuY2UiLHZhbDoxOCxkaXI6ImZsYXQifV0scmlzaW5nOlt7dDoiTGl2aW5nIHJvb3QgYnJpZGdlIFVORVNDTyBwdXNoIixwY3Q6IisxMCUifV0sZmFsbGluZzpbXSxhcnRpY2xlczpbe3NyYzoiU2hpbGxvbmcgVGltZXMiLHR4dDoiU3RhdGUgbm9taW5hdGVzIEtoYXNpIGxpdmluZyByb290IGJyaWRnZXMgZm9yIFVORVNDTyBsaXN0aW5nIn1dLHN1bW1hcnk6Ik1lZ2hhbGF5YSBlbnZpcm9ubWVudCBhbmQgdG91cmlzbS1sZWQuIFRoZSBVTkVTQ08gbm9taW5hdGlvbiBpcyB0aGUgZmxhZ3NoaXAgcHJpZGUgc3RvcnkuIix0aW1lbGluZTpbMjYsMjYsMjYsMjYsMjYsMjYsMjYsMjZdLG5hcnJhdGl2ZUhpc3Rvcnk6W3t0b3BpYzoiQ29hbCBtaW5pbmcgYmFuIix3aGVuOiI5IG1vbnRocyBhZ28iLGNsczoicGFzdCJ9LHt0b3BpYzoiVG91cmlzbSBkZXZlbG9wbWVudCIsd2hlbjoiNSBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IkVudmlyb25tZW50YWwgaGVyaXRhZ2UiLHdoZW46Ik5vdyIsY2xzOiJjdXJyZW50In0se3RvcGljOiJVTkVTQ08gY2FtcGFpZ24iLHdoZW46IkVtZXJnaW5nIixjbHM6InJlY2VudCJ9XX0sCiAgIkFydW5hY2hhbCBQcmFkZXNoIjp7YXR0ZW50aW9uOjM2LGRlbHRhOjIsdmVsb2NpdHk6MC4wOSxlbW90aW9uczp7YW54aWV0eToxOCxhbmdlcjoxNixob3BlOjE4LHByaWRlOjI0LGZlYXI6MjR9LG5hcnJhdGl2ZXM6W3tuYW1lOiJCb3JkZXIgaXNzdWVzIix2YWw6MzAsZGlyOiJ1cCJ9LHtuYW1lOiJJbmZyYXN0cnVjdHVyZSIsdmFsOjIyLGRpcjoidXAifSx7bmFtZToiSWRlbnRpdHkiLHZhbDoxOCxkaXI6ImZsYXQifSx7bmFtZToiRW52aXJvbm1lbnQiLHZhbDoxNixkaXI6ImZsYXQifSx7bmFtZToiR292ZXJuYW5jZSIsdmFsOjE0LGRpcjoiZmxhdCJ9XSxyaXNpbmc6W3t0OiJCb3JkZXIgaW5mcmFzdHJ1Y3R1cmUgcHVzaCIscGN0OiIrMjIlIn1dLGZhbGxpbmc6W3t0OiJUb3VyaXNtIGNpcmN1aXQgbGF1bmNoIixwY3Q6Ii04JSJ9XSxhcnRpY2xlczpbe3NyYzoiQXJ1bmFjaGFsIFRpbWVzIix0eHQ6IkJvcmRlciBpbmZyYXN0cnVjdHVyZSBwcm9qZWN0cyBmYXN0LXRyYWNrZWQgYWNyb3NzIFRhd2FuZyBzZWN0b3IifV0sc3VtbWFyeToiQXJ1bmFjaGFsIHJpc2luZyBvbiBib3JkZXIgaW5mcmFzdHJ1Y3R1cmUgYW5kIG5hbWluZy1jb250cm92ZXJzeSBuYXJyYXRpdmVzLiBCb3RoIGFyZSBjb25uZWN0ZWQgdG8gdGhlIHNhbWUgZ2VvcG9saXRpY2FsIHVuZGVyY3VycmVudC4iLHRpbWVsaW5lOlszMywzNCwzNCwzNSwzNSwzNiwzNiwzNl0sbmFycmF0aXZlSGlzdG9yeTpbe3RvcGljOiJDaGluZXNlIG5hbWluZyBjb250cm92ZXJzeSIsd2hlbjoiNiBtb250aHMgYWdvIixjbHM6InBhc3QifSx7dG9waWM6IlRvdXJpc20gZGV2ZWxvcG1lbnQiLHdoZW46IjQgbW9udGhzIGFnbyIsY2xzOiJwYXN0In0se3RvcGljOiJCb3JkZXIgaW5mcmFzdHJ1Y3R1cmUiLHdoZW46IjIgbW9udGhzIGFnbyIsY2xzOiJyZWNlbnQifSx7dG9waWM6Ikdlb3BvbGl0aWNhbCBpZGVudGl0eSIsd2hlbjoiTm93IixjbHM6ImN1cnJlbnQifV19LAp9OwoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjoyMCxkZWx0YTowLHZlbG9jaXR5OjAsCiAgZW1vdGlvbnM6e2FueGlldHk6MTUsYW5nZXI6MTIsaG9wZToyMixwcmlkZToyNSxmZWFyOjI2fSwKICBuYXJyYXRpdmVzOlt7bmFtZToiR292ZXJuYW5jZSIsdmFsOjI1LGRpcjoiZmxhdCJ9LHtuYW1lOiJFY29ub215Iix2YWw6MjAsZGlyOiJmbGF0In0se25hbWU6IkVudmlyb25tZW50Iix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IkluZnJhc3RydWN0dXJlIix2YWw6MTgsZGlyOiJmbGF0In0se25hbWU6IlRvdXJpc20iLHZhbDoxOSxkaXI6ImZsYXQifV0sCiAgcmlzaW5nOltdLGZhbGxpbmc6W10sCiAgYXJ0aWNsZXM6W3tzcmM6IlBUSSIsdHh0OiJTdGF0ZSBnb3Zlcm5hbmNlIHVwZGF0ZSByZXBvcnRlZCBpbiByb3V0aW5lIGN5Y2xlIn1dLAogIHN1bW1hcnk6Ikxvdy1hdHRlbnRpb24gcmVnaW9uLiBSb3V0aW5lIGdvdmVybmFuY2UgYW5kIGVjb25vbWljIG5hcnJhdGl2ZXMgd2l0aG91dCBzaWduaWZpY2FudCBtb3ZlbWVudCBpbiB0aGlzIGN5Y2xlLiIsCiAgdGltZWxpbmU6WzIwLDIwLDIwLDIwLDIwLDIwLDIwLDIwXSwKICBuYXJyYXRpdmVIaXN0b3J5Olt7dG9waWM6IlN0ZWFkeSBnb3Zlcm5hbmNlIix3aGVuOiJPbmdvaW5nIixjbHM6InBhc3QifSx7dG9waWM6IkVjb25vbWljIGJhc2VsaW5lIix3aGVuOiJOb3ciLGNsczoiY3VycmVudCJ9XQp9OwoKZnVuY3Rpb24gZyhuKXtyZXR1cm4gU0Rbbl18fE9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7fQoKZnVuY3Rpb24gYUMocyl7CiAgLy8gRHluYW1pYyBzY2FsZTogYWx3YXlzIHNwcmVhZCBmdWxsIGNvbG9yIHJhbmdlIGFjcm9zcyBhY3R1YWwgZGF0YQogIC8vIEdldCBtaW4vbWF4IGZyb20gY3VycmVudCBTRCB0byBub3JtYWxpemUKICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSk7CiAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKTsKICB2YXIgbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogIC8vIE5vcm1hbGl6ZSAwLTEKICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHMtbW4pLyhteC1tbikpKTsKICAvLyBNYXAgdG8gY29sb3Igc3RvcHM6IGRhcmsgYmx1ZSDihpIgdGVhbCDihpIgYW1iZXIg4oaSIG9yYW5nZSDihpIgcmVkCiAgaWYobjwwLjEyKSByZXR1cm4gJyMwZDFlMzAnOwogIGlmKG48MC4yNSkgcmV0dXJuICcjMGUzZDZhJzsKICBpZihuPDAuMzgpIHJldHVybiAnIzBkNWY5MCc7CiAgaWYobjwwLjUwKSByZXR1cm4gJyMwZTdhYWEnOwogIGlmKG48MC42MikgcmV0dXJuICcjMWE5MDkwJzsKICBpZihuPDAuNzIpIHJldHVybiAnI2M4NzAxMCc7CiAgaWYobjwwLjgyKSByZXR1cm4gJyNkODQwMTAnOwogIGlmKG48MC45MikgcmV0dXJuICcjY2MxODA4JzsKICByZXR1cm4gJyNmZjAwMTAnOwp9CmZ1bmN0aW9uIGVDKGUpewogIHZhciBteD0wLGRvbT0ncHJpZGUnOwogIGZvcih2YXIgayBpbiBlKXtpZihlW2tdPm14KXtteD1lW2tdO2RvbT1rO319CiAgcmV0dXJuICh7YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ30pW2RvbV18fCcjMzNhYWNjJzsKfQpmdW5jdGlvbiB2Qyh2KXsKICBpZih2PjAuMikgcmV0dXJuICcjZGMwODE4JzsKICBpZih2PjAuMSkgcmV0dXJuICcjZTA1YTI4JzsKICBpZih2PjAuMDIpIHJldHVybiAnI2NjODgyMic7CiAgaWYodjwtMC4wNSkgcmV0dXJuICcjMjI5OWJiJzsKICByZXR1cm4gJyMxNTIwMzAnOwp9Cgp2YXIgbGF5ZXI9J2F0dGVudGlvbicsU0VMPW51bGwsRkFWUz1uZXcgU2V0KCk7CgovLyBNQVAKZnVuY3Rpb24gcHJval8odyxoLHBhZCl7CiAgcGFkPXBhZHx8MjA7CiAgdmFyIG1pbkxvbj02OC4xLG1heExvbj05Ny40LG1pbkxhdD02LjUsbWF4TGF0PTM3LjE7CiAgdmFyIHNjWD0ody1wYWQqMikvKG1heExvbi1taW5Mb24pOwogIHZhciBzY1k9KGgtcGFkKjIpLyhtYXhMYXQtbWluTGF0KTsKICB2YXIgc2M9TWF0aC5taW4oc2NYLHNjWSk7CiAgdmFyIG94PXBhZCsody1wYWQqMi0obWF4TG9uLW1pbkxvbikqc2MpLzI7CiAgdmFyIG95PXBhZCsoaC1wYWQqMi0obWF4TGF0LW1pbkxhdCkqc2MpLzI7CiAgcmV0dXJuIGZ1bmN0aW9uKGxvbixsYXQpe3JldHVybiBbb3grKGxvbi1taW5Mb24pKnNjLCBveSsobWF4TGF0LWxhdCkqc2NdO307Cn0KZnVuY3Rpb24gZ2VvMnBhdGgoZ2VvbSxwail7CiAgdmFyIGQ9Jyc7CiAgZnVuY3Rpb24gcmluZyhjcyl7dmFyIHM9Jyc7Y3MuZm9yRWFjaChmdW5jdGlvbihjLGkpe3ZhciBwPXBqKGNbMF0sY1sxXSk7cys9KGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pO3JldHVybiBzKydaJzt9CiAgaWYoZ2VvbS50eXBlPT09J1BvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTsKICBlbHNlIGlmKGdlb20udHlwZT09PSdNdWx0aVBvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocCl7cC5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7fSk7CiAgcmV0dXJuIGQ7Cn0KZnVuY3Rpb24gY3RyKGdlb20pewogIHZhciBwdHM9W107CiAgZnVuY3Rpb24gY29sKGMpe2lmKHR5cGVvZiBjWzBdPT09J251bWJlcicpIHB0cy5wdXNoKGMpO2Vsc2UgYy5mb3JFYWNoKGNvbCk7fQogIGNvbChnZW9tLmNvb3JkaW5hdGVzKTsKICBpZighcHRzLmxlbmd0aCkgcmV0dXJuIFswLDBdOwogIHJldHVybiBbcHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMF07fSwwKS9wdHMubGVuZ3RoLHB0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzFdO30sMCkvcHRzLmxlbmd0aF07Cn0KZnVuY3Rpb24gc05hbWUocHJvcHMpewogIHZhciByYXc9cHJvcHMuc3Rfbm18fHByb3BzLk5BTUVfMXx8cHJvcHMubmFtZXx8cHJvcHMuTkFNRXx8Jyc7CiAgLy8gTm9ybWFsaXplIGNvbW1vbiBtaXNtYXRjaGVzIGJldHdlZW4gVG9wb0pTT04gYW5kIG91ciBTRCBrZXlzCiAgdmFyIG1hcD17CiAgICAnVXR0YXJhbmNoYWwnOidVdHRhcmFraGFuZCcsCiAgICAnVXR0YXJha2hhbmQnOidVdHRhcmFraGFuZCcsCiAgICAnSmFtbXUgJiBLYXNobWlyJzonSmFtbXUgYW5kIEthc2htaXInLAogICAgJ0phbW11IGFuZCBLYXNobWlyJzonSmFtbXUgYW5kIEthc2htaXInLAogICAgJ0RhZHJhIGFuZCBOYWdhciBIYXZlbGknOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1JywKICAgICdEYW1hbiBhbmQgRGl1JzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsCiAgfTsKICByZXR1cm4gbWFwW3Jhd118fHJhdzsKfQoKdmFyIGNhY2hlZEdlbz1udWxsOwoKYXN5bmMgZnVuY3Rpb24gbG9hZE1hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKCdodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvZ2gvdWRpdC0wMDEvaW5kaWEtbWFwcy1kYXRhQG1hc3Rlci90b3BvanNvbi9pbmRpYS5qc29uJyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICBhd2FpdCBuZXcgUHJvbWlzZShmdW5jdGlvbihyZXMpe3NldFRpbWVvdXQocmVzLDIwMCk7fSk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIC8vIEFmdGVyIGxpdmUgZGF0YSBhcnJpdmVzLCByZS1hcHBseSBjb2xvcnMgV0lUSE9VVCByZS1yZW5kZXJpbmcgZ2VvbWV0cnkKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwgMTIwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsIDMwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLCA2MDAwKTsKICB9Y2F0Y2goZSl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJykuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjojMmEzYTRhO3BhZGRpbmc6NDBweDt0ZXh0LWFsaWduOmNlbnRlcjtmb250LWZhbWlseTptb25vc3BhY2U7Zm9udC1zaXplOjExcHg7aGVpZ2h0OjEwMCU7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyIj5NYXAgdW5hdmFpbGFibGU8L2Rpdj4nOwogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKZnVuY3Rpb24gYXBwbHlMYXllcigpewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksZmlsbDsKICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihsYXllcj09PSdlbW90aW9uJykgZmlsbD1lQyhkLmVtb3Rpb25zKTsKICAgIGVsc2UgZmlsbD12QyhkLnZlbG9jaXR5KTsKICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsJyxmaWxsKTsKICAgIChmdW5jdGlvbigpewogICAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4geC5hdHRlbnRpb258fDA7fSk7CiAgICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogICAgICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKGQuYXR0ZW50aW9uLW1uKS8obXgtbW4pKSk7CiAgICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsLW9wYWNpdHknLGxheWVyPT09J2F0dGVudGlvbic/TWF0aC5tYXgoMC4zLDAuMytuKjAuNyk6MC44NSk7CiAgICB9KSgpOwogIH0pOwp9CgpmdW5jdGlvbiBhdHRhY2hJbnRlcmFjdGlvbnMoKXsKICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXsKICAgICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pOwogICAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgICB2YXIgaGlzdD1kLm5hcnJhdGl2ZUhpc3Rvcnk7CiAgICAgIHZhciBsYXRlc3Q9aGlzdCYmaGlzdC5sZW5ndGg/aGlzdFtoaXN0Lmxlbmd0aC0xXS50b3BpYzon4oCUJzsKICAgICAgLy8gRHluYW1pYyB0b29sdGlwIGNvbnRlbnQgYmFzZWQgb24gYWN0aXZlIGxheWVyCiAgICAgIHZhciBsYXllclJvd3M9Jyc7CiAgICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgICAgIGxheWVyUm93cz0KICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5BdHRlbnRpb24gaW5kZXg8L3NwYW4+PHN0cm9uZz4nK2QuYXR0ZW50aW9uKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgKGQuZGVsdGEhPT0wPyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj4yNGggc2hpZnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JysoZC5kZWx0YT4wPycjZTA1YTI4JzonIzNiYjhkOCcpKyciPicrKGQuZGVsdGE+MD8nKyc6JycpK2QuZGVsdGErJzwvc3Ryb25nPjwvZGl2Pic6JycpKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPlRvcCBuYXJyYXRpdmU8L3NwYW4+PHN0cm9uZz4nK3RvcCsnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfSBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICB2YXIgZUxpc3Q9T2JqZWN0LmVudHJpZXMoZC5lbW90aW9ucykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICAgICAgdmFyIHJhd1Q9ZUxpc3QucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICAgICAgaWYocmF3VD4wJiZyYXdUPD0xLjAxKXtlTGlzdD1lTGlzdC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiBba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt9CiAgICAgICAgdmFyIHRvdD1NYXRoLm1heCgxLGVMaXN0LnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApKTsKICAgICAgICB2YXIgZG9tRW1vPWVMaXN0WzBdOwogICAgICAgIGxheWVyUm93cz0KICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Eb21pbmFudCBlbW90aW9uPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrcGFsW2RvbUVtb1swXV0rJyI+Jytkb21FbW9bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vWzBdLnNsaWNlKDEpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgZUxpc3Quc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3YpewogICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo0cHgiPjxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrcGFsW2t2WzBdXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JytrdlswXSsnPC9zcGFuPjxzdHJvbmc+JytNYXRoLnJvdW5kKGt2WzFdKjEwMC90b3QpKyclPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpOwogICAgICB9IGVsc2UgewogICAgICAgIHZhciB2RGlyPWQudmVsb2NpdHk+MC4wNT8nUmlzaW5nJzpkLnZlbG9jaXR5PC0wLjA1PydDb29saW5nJzonU3RhYmxlJzsKICAgICAgICB2YXIgdkNvbD1kLnZlbG9jaXR5PjAuMDU/JyNlMDVhMjgnOmQudmVsb2NpdHk8LTAuMDU/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgICAgICBsYXllclJvd3M9CiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+TW9tZW50dW08L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2Q29sKyciPicrKGQudmVsb2NpdHk+MD8nKyc6JycpK2QudmVsb2NpdHkudG9GaXhlZCgyKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5EaXJlY3Rpb248L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2Q29sKyciPicrKGQudmVsb2NpdHk+MC4xPydSaXNpbmcgZmFzdCc6ZC52ZWxvY2l0eT4wLjAyPydSaXNpbmcnOmQudmVsb2NpdHk8LTAuMDU/J0Nvb2xpbmcnOidTdGFibGUnKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj4yNGggc2lnbmFsczwvc3Bhbj48c3Ryb25nPicrKGQuZGVsdGE+PTA/JysnOicnKStkLmRlbHRhKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICB9CiAgICAgIHRpcC5pbm5lckhUTUw9CiAgICAgICAgJzxkaXYgY2xhc3M9InR0LW4iPicrbm0rJzwvZGl2PicrCiAgICAgICAgbGF5ZXJSb3dzKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1uYXIiPjxzdHJvbmc+Q3VycmVudCBuYXJyYXRpdmU8L3N0cm9uZz4nK2xhdGVzdCsnPC9kaXY+JzsKICAgICAgdmFyIHJlY3Q9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgICB0aXAuc3R5bGUubGVmdD1NYXRoLm1pbihlLmNsaWVudFgtcmVjdC5sZWZ0KzE0LHJlY3Qud2lkdGgtMTgwKSsncHgnOwogICAgICB0aXAuc3R5bGUudG9wPU1hdGgubWluKGUuY2xpZW50WS1yZWN0LnRvcCsxNCxyZWN0LmhlaWdodC0xNDApKydweCc7CiAgICAgIHRpcC5zdHlsZS5vcGFjaXR5PTE7CiAgICB9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VsZWF2ZScsZnVuY3Rpb24oKXt0aXAuc3R5bGUub3BhY2l0eT0wO30pOwogICAgcC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXtzZWxlY3RfKHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSk7fSk7CiAgfSk7Cn0KCi8vIFNUQVRFIFBBTkVMCmZ1bmN0aW9uIHNlbGVjdF8obm0pewogIFNFTD1ubTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApe3AuY2xhc3NMaXN0LnRvZ2dsZSgnc2VsZWN0ZWQnLHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKT09PW5tKTt9KTsKICByZW5kZXJQYW5lbChubSk7CiAgZmV0Y2hEZXRhaWwobm0pLnRoZW4oZnVuY3Rpb24oZCl7aWYoU0VMPT09bm0pIHJlbmRlclBhbmVsKG5tKTt9KTsKfQoKZnVuY3Rpb24gcmVuZGVyUGFuZWwobm0pewogIHZhciBkPWcobm0pLHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICB2YXIgaXNGYXY9RkFWUy5oYXMobm0pLGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/J3VwJzonZG4nOwogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogIHZhciBlbW90aW9ucz1kLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhkLmVtb3Rpb25zKS5sZW5ndGg/ZC5lbW90aW9uczp7YW54aWV0eToyMCxhbmdlcjoxNSxob3BlOjI1LHByaWRlOjI1LGZlYXI6MTV9OwogIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW90aW9ucyk7CiAgLy8gTm9ybWFsaXplOiBBUEkgbWF5IHJldHVybiAwLTEgZmxvYXRzIE9SIDAtMTAwIGludGVnZXJzCiAgdmFyIHJhd1RvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICBpZihyYXdUb3Q+MCAmJiByYXdUb3Q8PTEuMDEpeyBlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiBba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTsgfQogIHZhciB0b3Q9TWF0aC5tYXgoMSxlTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKSk7CiAgdmFyIGN1bUE9LU1hdGguUEkvMixjeD0zOCxjeT0zOCxSPTMzLHJpPTIwOwogIHZhciBhcmNzPWVMLm1hcChmdW5jdGlvbihrdil7CiAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7CiAgICBjdW1BPWEyO3ZhciBsZz0oYTItYTEpPk1hdGguUEk/MTowOwogICAgdmFyIHgxPWN4K01hdGguY29zKGExKSpSLHkxPWN5K01hdGguc2luKGExKSpSLHgyPWN4K01hdGguY29zKGEyKSpSLHkyPWN5K01hdGguc2luKGEyKSpSOwogICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgIHJldHVybiAnPHBhdGggZD0iTScreDEudG9GaXhlZCgxKSsnLCcreTEudG9GaXhlZCgxKSsnIEEnK1IrJywnK1IrJyAwICcrbGcrJyAxICcreDIudG9GaXhlZCgxKSsnLCcreTIudG9GaXhlZCgxKSsnIEwnK3gzLnRvRml4ZWQoMSkrJywnK3kzLnRvRml4ZWQoMSkrJyBBJytyaSsnLCcrcmkrJyAwICcrbGcrJyAwICcreDQudG9GaXhlZCgxKSsnLCcreTQudG9GaXhlZCgxKSsnIFoiIGZpbGw9IicrcGFsW2tdKyciIG9wYWNpdHk9IjAuOSIvPic7CiAgfSkuam9pbignJyk7CgogIHZhciB0bD1kLnRpbWVsaW5lLHRtbj1NYXRoLm1pbi5hcHBseShudWxsLHRsKSx0bXg9TWF0aC5tYXguYXBwbHkobnVsbCx0bCksdHI9TWF0aC5tYXgoMSx0bXgtdG1uKTsKICB2YXIgdHc9MjYwLHRoPTYyLHRwPTU7CiAgdmFyIHB0cz10bC5tYXAoZnVuY3Rpb24odixpKXtyZXR1cm4gW3RwKyhpLyh0bC5sZW5ndGgtMSkpKih0dy10cCoyKSx0cCsoMS0odi10bW4pL3RyKSoodGgtdHAqMildO30pOwogIHZhciBwRD1wdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuIChpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KS5qb2luKCcnKTsKICB2YXIgYUQ9cEQrJyBMJytwdHNbcHRzLmxlbmd0aC0xXVswXSsnLCcrKHRoLXRwKSsnIEwnK3B0c1swXVswXSsnLCcrKHRoLXRwKSsnIFonOwogIHZhciBhYz1hQyhkLmF0dGVudGlvbik7CgogIHZhciBoaXN0PWQubmFycmF0aXZlSGlzdG9yeXx8W107CgogIHBhbmVsLmlubmVySFRNTD0KICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAnPGRpdj48ZGl2IGNsYXNzPSJzcC1layI+TmFycmF0aXZlIHBhbmVsPC9kaXY+PGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoaXNGYXY/J29uJzonJykrJyIgb25jbGljaz0idG9nZ2xlRmF2KFwnJytubSsnXCcpIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhpc0Zhdj8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICc8L2J1dHRvbj4nKwogICAgJzwvZGl2PicrCgogICAgLy8gTmFycmF0aXZlIGhpc3RvcnkgdGltZWxpbmUg4oCUIHNpZ25hdHVyZSBmZWF0dXJlCiAgICAoaGlzdC5sZW5ndGg/CiAgICAgICc8ZGl2IGNsYXNzPSJuYXItdGltZWxpbmUiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im50LWxhYmVsIj5OYXJyYXRpdmUgZXZvbHV0aW9uPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ibnQtZmxvdyI+JysKICAgICAgICAgIGhpc3QubWFwKGZ1bmN0aW9uKGgpewogICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im50LXN0ZXAgJytoLmNscysnIj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJudC1kb3QiPjwvZGl2PicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im50LWNvbnRlbnQiPicrCiAgICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibnQtdG9waWMiPicraC50b3BpYysnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJudC13aGVuIj4nK2gud2hlbisnPC9kaXY+JysKICAgICAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic6JycpKwoKICAgICc8ZGl2IGNsYXNzPSJpbnNpZ2h0Ij4nK2Quc3VtbWFyeSsnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic2NvcmUtc3RyaXAiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIj4nK2QuYXR0ZW50aW9uKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj4yNGggc2hpZnQ8L2Rpdj48ZGl2IGNsYXNzPSJzcy1kZWx0YSAnK2RDKyciPicrZFMrZC5kZWx0YSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VG9wIG5hcnJhdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNzLW5hciI+JysoZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnKSsnPC9kaXY+PC9kaXY+JysKICAgICc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgKGQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzLmxlbmd0aD8KICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrCiAgICAgICAgZC5uYXJyYXRpdmVzLm1hcChmdW5jdGlvbihuKXsKICAgICAgICAgIHZhciBubT1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICB2YXIgdmFsPXR5cGVvZiBuLnZhbD09PSdudW1iZXInP24udmFsOjA7CiAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtMiI+JysKICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25tKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4Ij7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojM2JiOGQ4O2ZvbnQtc2l6ZTo5cHgiPuKGkzwvc3Bhbj4nOicnKSsnPC9kaXY+JysKICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXZhbCI+Jyt2YWwudG9GaXhlZCgxKSsnJTwvZGl2PicrCiAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLHZhbCoyLjUpKyclO2JhY2tncm91bmQ6Jysobi5kaXI9PT0ndXAnPycjZTA1YTI4JzpuLmRpcj09PSdkb3duJz8nIzNiYjhkOCc6JyMzMzQ0NTUnKSsnIj48L2Rpdj48L2Rpdj4nKwogICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgJzwvZGl2Pic6CiAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwO2xpbmUtaGVpZ2h0OjEuNiI+TG93LXNpZ25hbCByZWdpb24uIE5hdGlvbmFsIHByZXNzIGNvdmVyYWdlIGlzIGxpbWl0ZWQgZm9yIHRoaXMgc3RhdGUg4oCUIHJlZ2lvbmFsIGxhbmd1YWdlIHNvdXJjZXMgYXJlIGJlaW5nIG1vbml0b3JlZC48L2Rpdj4nKSsKICAgICc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5Nb3ZlbWVudDwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJtdi1ncmlkIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJtdi1ibG9jayB1cCI+PGRpdiBjbGFzcz0ibXYtaCI+UmlzaW5nPC9kaXY+JysKICAgICAgICAgIChkLnJpc2luZy5sZW5ndGg/ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0ibXYtaXQiPjxzdHJvbmc+JytyLnQrJzwvc3Ryb25nPjxzcGFuPicrci5wY3QrJzwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpOic8ZGl2IGNsYXNzPSJtdi1pdCIgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+U3RhYmxlPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Im12LWJsb2NrIGRuIj48ZGl2IGNsYXNzPSJtdi1oIj5GYWxsaW5nPC9kaXY+JysKICAgICAgICAgIChkLmZhbGxpbmcubGVuZ3RoP2QuZmFsbGluZy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IGNsYXNzPSJtdi1pdCI+PHN0cm9uZz4nK3IudCsnPC9zdHJvbmc+PHNwYW4+JytyLnBjdCsnPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJyk6JzxkaXYgY2xhc3M9Im12LWl0IiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj5TdGFibGU8L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JysKCiAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RW1vdGlvbmFsIHJlZ2lzdGVyPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImVtLXJvdyI+JysKICAgICAgICAnPHN2ZyBjbGFzcz0iZW0tZG9udXQiIHZpZXdCb3g9IjAgMCA3NiA3NiI+JythcmNzKyc8L3N2Zz4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJlbS1sZWciPicrCiAgICAgICAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KS5tYXAoZnVuY3Rpb24oa3YpewogICAgICAgICAgICB2YXIgaz1rdlswXSx2PWt2WzFdOwogICAgICAgICAgICB2YXIgZGVzYz17YW54aWV0eTonVW5jZXJ0YWludHkgJiB3b3JyeScsYW5nZXI6J091dHJhZ2UgJiBwcm90ZXN0Jyxob3BlOidPcHRpbWlzbSAmIGdyb3d0aCcscHJpZGU6J0FjaGlldmVtZW50ICYgaWRlbnRpdHknLGZlYXI6J1RocmVhdCBwZXJjZXB0aW9uJ307CiAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZW0taXRlbSIgc3R5bGU9Im1hcmdpbi1ib3R0b206MXB4Ij4nKwogICAgICAgICAgICAgICc8c3BhbiBjbGFzcz0iZW0tc3ciIHN0eWxlPSJiYWNrZ3JvdW5kOicrcGFsW2tdKyciPjwvc3Bhbj4nKwogICAgICAgICAgICAgICc8c3BhbiBjbGFzcz0iZW0tbiI+JytrLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2suc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgICAgICc8c3BhbiBjbGFzcz0iZW0tcCI+JytNYXRoLnJvdW5kKHYqMTAwL3RvdCkrJyU8L3NwYW4+JysKICAgICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAgICh2PT09ZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF1bMV0/CiAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzoxcHggMCA0cHggMTJweDtib3JkZXItbGVmdDoxcHggc29saWQgJytwYWxba10rJzttYXJnaW4tbGVmdDozcHg7bWFyZ2luLWJvdHRvbTozcHg7Ij4nK2Rlc2Nba10rJzwvZGl2Pic6CiAgICAgICAgICAgICcnKTsKICAgICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BdHRlbnRpb24g4oCUIDggZGF5czwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJ0bC13cmFwIj4nKwogICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAnK3R3KycgJyt0aCsnIiBzdHlsZT0id2lkdGg6MTAwJTtoZWlnaHQ6MTAwJSI+JysKICAgICAgICAgICc8ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9InRsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwLjI1Ii8+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAiLz4nKwogICAgICAgICAgJzwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPicrCiAgICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnKSIvPicrCiAgICAgICAgICAnPHBhdGggZD0iJytwRCsnIiBmaWxsPSJub25lIiBzdHJva2U9IicrYWMrJyIgc3Ryb2tlLXdpZHRoPSIxLjIiLz4nKwogICAgICAgICAgcHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAnPGNpcmNsZSBjeD0iJytwWzBdKyciIGN5PSInK3BbMV0rJyIgcj0iJysoaT09PXB0cy5sZW5ndGgtMT8yLjI6MS4yKSsnIiBmaWxsPSInK2FjKyciLz4nO30pLmpvaW4oJycpKwogICAgICAgICc8L3N2Zz4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nKwoKICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWxzIDxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrZC5hcnRpY2xlcy5sZW5ndGgrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iYXJ0LWxpc3QiPicrCiAgICAgICAgZC5hcnRpY2xlcy5tYXAoZnVuY3Rpb24oYSl7cmV0dXJuICc8ZGl2IGNsYXNzPSJhcnQtaXRlbSI+PGRpdiBjbGFzcz0iYXJ0LXNyYyI+JythLnNyYysnPC9kaXY+PGRpdiBjbGFzcz0iYXJ0LXR4dCI+JythLnR4dCsnPC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKSsKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JzsKfQoKZnVuY3Rpb24gdG9nZ2xlRmF2KG5tKXsKICBpZihGQVZTLmhhcyhubSkpIEZBVlMuZGVsZXRlKG5tKTtlbHNlIEZBVlMuYWRkKG5tKTsKICByZW5kZXJQYW5lbChTRUwpO3JlbmRlckZhdnMoKTsKfQpmdW5jdGlvbiByZW5kZXJGYXZzKCl7CiAgdmFyIHJvdz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmF2LXJvdycpOwogIGlmKCFGQVZTLnNpemUpe3Jvdy5pbm5lckhUTUw9JzxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2Pic7cmV0dXJuO30KICByb3cuaW5uZXJIVE1MPUFycmF5LmZyb20oRkFWUykubWFwKGZ1bmN0aW9uKG5tKXsKICAgIHZhciBkPWcobm0pLGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/JyNlMDVhMjgnOicjM2JiOGQ4JzsKICAgIHZhciB0b3A9ZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCc7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImZhdi1jYXJkIiBvbmNsaWNrPSJzZWxlY3RfKFwnJytubSsnXCcpIj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtaGVhZCI+PHNwYW4gY2xhc3M9ImZjLW5hbWUiPicrbm0rJzwvc3Bhbj48c3BhbiBjbGFzcz0iZmMtc2MiPicrZC5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj5OYXJyYXRpdmU8L3NwYW4+PHNwYW4gY2xhc3M9InYiPicrdG9wKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+MjRoPC9zcGFuPjxzcGFuIGNsYXNzPSJ2IiBzdHlsZT0iY29sb3I6JytkQysnIj4nK2RTK2QuZGVsdGErJzwvc3Bhbj48L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KCmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbihjKXsKICBjLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHgpe3guY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICBjLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO2xheWVyPWMuZGF0YXNldC5sYXllcjthcHBseUxheWVyKCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gdXBkYXRlQ2xvY2soKXsKICB2YXIgbm93PW5ldyBEYXRlKCksaXN0PW5ldyBEYXRlKG5vdy5nZXRUaW1lKCkrbm93LmdldFRpbWV6b25lT2Zmc2V0KCkqNjAwMDArMTk4MDAwMDApOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbG9jaycpLnRleHRDb250ZW50PVN0cmluZyhpc3QuZ2V0SG91cnMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldE1pbnV0ZXMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldFNlY29uZHMoKSkucGFkU3RhcnQoMiwnMCcpKycgSVNUJzsKfQpzZXRJbnRlcnZhbCh1cGRhdGVDbG9jaywxMDAwKTt1cGRhdGVDbG9jaygpOwoKLy8gSU5JVApyZW5kZXJTdHJpcCgnM20nKTsKcmVuZGVyTW9tZW50dW0oKTsKbG9hZE1hcCgpOwpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7c3RhcnRQb2xsaW5nKCk7fSw4MDApOwpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wJiZkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK3RvcFswXSsnIl0nKSkgc2VsZWN0Xyh0b3BbMF0pOwp9LDI0MDApOwpzZXRUaW1lb3V0KHJlbmRlckZhdnMsMjQwMCk7CgovLyDilIDilIAgQ09MT1IgVVRJTElUSUVTIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiBsZXJwQ29sb3IoYSxiLHQpe2Z1bmN0aW9uIGgoeCl7cmV0dXJuIHBhcnNlSW50KHgsMTYpO312YXIgYXI9aChhLnNsaWNlKDEsMykpLGFnPWgoYS5zbGljZSgzLDUpKSxhYj1oKGEuc2xpY2UoNSw3KSk7dmFyIGJyPWgoYi5zbGljZSgxLDMpKSxiZz1oKGIuc2xpY2UoMyw1KSksYmI9aChiLnNsaWNlKDUsNykpO2Z1bmN0aW9uIHAobil7dmFyIHM9TWF0aC5yb3VuZChuKS50b1N0cmluZygxNik7cmV0dXJuIHMubGVuZ3RoPDI/JzAnK3M6czt9cmV0dXJuICcjJytwKGFyKyhici1hcikqdCkrcChhZysoYmctYWcpKnQpK3AoYWIrKGJiLWFiKSp0KTt9CmZ1bmN0aW9uIGNvbG9yU2NhbGUobixzdG9wcyl7Zm9yKHZhciBpPTA7aTxzdG9wcy5sZW5ndGgtMTtpKyspe2lmKG4+PXN0b3BzW2ldWzBdJiZuPD1zdG9wc1tpKzFdWzBdKXt2YXIgdD0obi1zdG9wc1tpXVswXSkvKHN0b3BzW2krMV1bMF0tc3RvcHNbaV1bMF0pO3JldHVybiBsZXJwQ29sb3Ioc3RvcHNbaV1bMV0sc3RvcHNbaSsxXVsxXSx0KTt9fXJldHVybiBzdG9wc1tzdG9wcy5sZW5ndGgtMV1bMV07fQp2YXIgX2FOb3JtPXttbjowLG14OjEsdHM6MH07CmZ1bmN0aW9uIG5vcm1WKHYpe2lmKCF2KXJldHVybiAwO3ZhciBhPU1hdGguYWJzKHYpO2lmKGE+MSl2PXYvTWF0aC5tYXgoYSw1MCk7cmV0dXJuIE1hdGgubWF4KC0xLE1hdGgubWluKDEsdikpO30KZnVuY3Rpb24gYUMocyl7aWYoRGF0ZS5ub3coKS1fYU5vcm0udHM+NTAwMCl7dmFyIHNjPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSkuZmlsdGVyKGZ1bmN0aW9uKHYpe3JldHVybiB2PjA7fSk7X2FOb3JtLm1uPXNjLmxlbmd0aD9NYXRoLm1pbi5hcHBseShudWxsLHNjKTowO19hTm9ybS5teD1zYy5sZW5ndGg/KE1hdGgubWF4LmFwcGx5KG51bGwsc2MpfHwxKToxO19hTm9ybS50cz1EYXRlLm5vdygpO312YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHMtX2FOb3JtLm1uKS9NYXRoLm1heChfYU5vcm0ubXgtX2FOb3JtLm1uLDEpKSk7cmV0dXJuIGNvbG9yU2NhbGUobixbWzAsJyMwYTE2MjgnXSxbMC4xNSwnIzBkM2E2ZSddLFswLjMsJyMwYTVmOGEnXSxbMC40NSwnIzBkOGE3YSddLFswLjU4LCcjMmE3YTRhJ10sWzAuNywnI2IwODAxMCddLFswLjgsJyNkMDYwMTAnXSxbMC45LCcjY2MyODA4J10sWzEsJyNmZjEwMjAnXV0pO30KZnVuY3Rpb24gdkModil7dj1ub3JtVih2KTtpZighdkMuX3R8fERhdGUubm93KCktdkMuX3Q+NTAwMCl7dmFyIG5vcm1zPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gbm9ybVYoZC52ZWxvY2l0eXx8MCk7fSk7dkMuX3A9TWF0aC5tYXguYXBwbHkobnVsbCxub3Jtcy5maWx0ZXIoZnVuY3Rpb24oeCl7cmV0dXJuIHg+MDt9KS5jb25jYXQoWzAuMV0pKTt2Qy5fbj1NYXRoLmFicyhNYXRoLm1pbi5hcHBseShudWxsLG5vcm1zLmZpbHRlcihmdW5jdGlvbih4KXtyZXR1cm4geDwwO30pLmNvbmNhdChbLTAuMV0pKSk7dkMuX3Q9RGF0ZS5ub3coKTt9aWYodj4wLjAwNSlyZXR1cm4gY29sb3JTY2FsZShNYXRoLm1pbigxLHYvKHZDLl9wfHwwLjEpKSxbWzAsJyMyYTI4MTgnXSxbMC4zLCcjOGE2MDEwJ10sWzAuNiwnI2QwNTAxMCddLFsxLCcjZTgxMDEwJ11dKTtpZih2PC0wLjAwNSlyZXR1cm4gY29sb3JTY2FsZShNYXRoLm1pbigxLE1hdGguYWJzKHYpLyh2Qy5fbnx8MC4xKSksW1swLCcjMTgyMDI4J10sWzAuMywnIzFhNTA3MCddLFswLjYsJyMxMDYwYTAnXSxbMSwnIzA4MjhjMCddXSk7cmV0dXJuICcjMjUyZTNhJzt9CgovLyDilIDilIAgTE9BREVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7dmFyIGw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2FwcC1sb2FkZXInKTtpZighbClyZXR1cm47bC5zdHlsZS5vcGFjaXR5PScwJztsLnN0eWxlLnZpc2liaWxpdHk9J2hpZGRlbic7c2V0VGltZW91dChmdW5jdGlvbigpe2lmKGwpbC5zdHlsZS5kaXNwbGF5PSdub25lJzt9LDkwMCk7fQoKLy8g4pSA4pSAIFNUQVRTIFNUUklQIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiB1cGRhdGVBbGxTdHJpcHMoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKFNEW2t2WzBdXSl7aWYoU0Rba3ZbMF1dLm5hcnJhdGl2ZXMpa3ZbMV0ubmFycmF0aXZlcz1TRFtrdlswXV0ubmFycmF0aXZlcztpZihTRFtrdlswXV0uc291cmNlX2NvdW50KWt2WzFdLnNvdXJjZV9jb3VudD1TRFtrdlswXV0uc291cmNlX2NvdW50O2lmKFNEW2t2WzBdXS5jb25maWRlbmNlKWt2WzFdLmNvbmZpZGVuY2U9U0Rba3ZbMF1dLmNvbmZpZGVuY2U7fWt2WzFdLl92ZWw9bm9ybVYoa3ZbMV0udmVsb2NpdHl8fDApO30pOwogIGZ1bmN0aW9uIHNTKGt2KXt2YXIgZD1rdlsxXTtyZXR1cm4oZC5hdHRlbnRpb258fDApKyhkLl92ZWx8fDApKjE1K01hdGgubWluKChkLnNvdXJjZV9jb3VudHx8MSksNSkqMisoeydISUdIJzozLCdNRURJVU0nOjEsJ0xPVyc6LTJ9W2QuY29uZmlkZW5jZXx8J0xPVyddfHwwKTt9CiAgdmFyIHNjb3JlZD1lbnRyaWVzLm1hcChmdW5jdGlvbihrdil7cmV0dXJue25hbWU6a3ZbMF0sZDprdlsxXSxzY29yZTpzUyhrdil9O30pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5zY29yZS1hLnNjb3JlO30pOwogIGZ1bmN0aW9uIHRpcChpZCx0aXRsZSxuYXJzKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIXQpcmV0dXJuO3QuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzYy10aXAtdGl0bGUiPicrdGl0bGUrJzwvZGl2PicrKG5hcnN8fFtdKS5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9InNjLXRpcC1yb3ciPsK3ICcrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsnPC9kaXY+Jzt9KS5qb2luKCcnKTt9CiAgdmFyIGg9c2NvcmVkWzBdO2lmKGgpe3NldFRleHQoJ3NjLWhvdHRlc3QtdmFsJyxoLm5hbWUpO3NldFRleHQoJ3NjLWhvdHRlc3Qtc3ViJywnQXR0ZW50aW9uICcrTWF0aC5yb3VuZChoLmQuYXR0ZW50aW9ufHwwKSk7dGlwKCdzYy1ob3R0ZXN0LXRpcCcsJ1doeSAnK2gubmFtZSsnPycsaC5kLm5hcnJhdGl2ZXMpO30KICB2YXIgYW5nPXNjb3JlZC5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuZC5kb21pbmFudF9lbW90aW9uPT09J2FuZ2VyJyYmKHMuZC5hdHRlbnRpb258fDApPjM7fSlbMF07CiAgaWYoYW5nKXtzZXRUZXh0KCdzYy1hbmdlci12YWwnLGFuZy5uYW1lKTtzZXRUZXh0KCdzYy1hbmdlci1zdWInLGFuZy5kLmRvbWluYW50X25hcnJhdGl2ZXx8J3NpZ25hbHMnKTt0aXAoJ3NjLWFuZ2VyLXRpcCcsJ0FuZ2VyIGluICcrYW5nLm5hbWUsYW5nLmQubmFycmF0aXZlcyk7fQogIHZhciByaXM9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5fdmVsfHwwKT4wLjAxO30pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYlsxXS5fdmVsfHwwKS0oYVsxXS5fdmVsfHwwKTt9KVswXTsKICBpZihyaXMpe3NldFRleHQoJ3NjLXJpc2luZy12YWwnLHJpc1swXSk7c2V0VGV4dCgnc2MtcmlzaW5nLXN1YicscmlzWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J3NpZ25hbCcpO3RpcCgnc2MtcmlzaW5nLXRpcCcsJ1doeSAnK3Jpc1swXSsnIGlzIHJpc2luZycscmlzWzFdLm5hcnJhdGl2ZXMpO30KICB2YXIgbmM9e307ZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXsoa3ZbMV0ubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7bmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7fSk7fSk7CiAgdmFyIHROPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXTsKICBpZih0Til7c2V0VGV4dCgnc2MtbmFyLXZhbCcsdE5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdE5bMF0uc2xpY2UoMSkpO3ZhciBuUz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLm5hcnJhdGl2ZXN8fFtdKS5zb21lKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU9PT10TlswXTt9KTt9KS5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihrdil7cmV0dXJuIGt2WzBdLnNwbGl0KCcgJylbMF07fSk7c2V0VGV4dCgnc2MtbmFyLXN1YicsblMuam9pbignLCAnKXx8J25hdGlvbmFsbHknKTt2YXIgdFQ9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NjLW5hci10aXAnKTtpZih0VCl0VC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0TlswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0TlswXS5zbGljZSgxKSsnIOKAlCBpbjwvZGl2PicrZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5uYXJyYXRpdmVzfHxbXSkuc29tZShmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lPT09dE5bMF07fSk7fSkuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0ic2MtdGlwLXJvdyI+wrcgJytrdlswXSsnPC9kaXY+Jzt9KS5qb2luKCcnKTt9CiAgdmFyIGNvb2w9ZW50cmllcy5zbGljZSgpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYVsxXS5fdmVsfHwwKS0oYlsxXS5fdmVsfHwwKTt9KVswXTsKICBpZihjb29sKXtzZXRUZXh0KCdzYy1jb29sLXZhbCcsY29vbFswXSk7c2V0VGV4dCgnc2MtY29vbC1zdWInLChjb29sWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8JycpKyhjb29sWzFdLl92ZWw8LTAuMDU/JyDCtyByZXRyZWF0aW5nJzonIMK3IGxlYXN0IG1vbWVudHVtJykpO3RpcCgnc2MtY29vbC10aXAnLCdMb3dlc3QgbW9tZW50dW06ICcrY29vbFswXSxjb29sWzFdLm5hcnJhdGl2ZXMpO30KICB2YXIgdG90PU9iamVjdC52YWx1ZXMoU0QpLnJlZHVjZShmdW5jdGlvbihzLHYpe3JldHVybiBzKyh2LnNpZ25hbF9jb3VudHx8MCk7fSwwKTsKICB2YXIgbGM9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xpdmUtY291bnQnKTtpZihsYylsYy50ZXh0Q29udGVudD10b3QudG9Mb2NhbGVTdHJpbmcoJ2VuLUlOJyk7CiAgdmFyIHN2PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzYy1zaWduYWxzLXZhbCcpO2lmKHN2KXN2LnRleHRDb250ZW50PXRvdC50b0xvY2FsZVN0cmluZygnZW4tSU4nKTsKICBzZXRUZXh0KCdzYy1zaWduYWxzLXN1YicsJ2Fjcm9zcyAnK09iamVjdC5rZXlzKExJVkUpLmZpbHRlcihmdW5jdGlvbihrKXtyZXR1cm4oTElWRVtrXS5hdHRlbnRpb258fDApPjI7fSkubGVuZ3RoKycgYWN0aXZlIHN0YXRlcycpOwp9CgovLyDilIDilIAgV0lSIFNJR05BTFMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmZ1bmN0aW9uIGJ1aWxkV0lSU2lnbmFscygpewogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLmF0dGVudGlvbnx8MCk+Mzt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBzaWduYWxzPVtdLHVzZWROPVtdLHVzZWRTPVtdOwogIGZ1bmN0aW9uIHVzZWQobixzKXtyZXR1cm4gdXNlZE4uaW5kZXhPZihuKT49MHx8dXNlZFMuaW5kZXhPZihzKT49MDt9CiAgZnVuY3Rpb24gdXNlKG4scyl7aWYobil1c2VkTi5wdXNoKG4pO2lmKHMpdXNlZFMucHVzaChzKTt9CiAgdmFyIHRvcD1lbnRyaWVzWzBdOwogIGlmKHRvcCl7dmFyIG5hcj10b3BbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwncmVnaW9uYWwgYWN0aXZpdHknO3ZhciBlbW89dG9wWzFdLmRvbWluYW50X2Vtb3Rpb247dmFyIGNvbD1lbW8/cGFsW2Vtb106J3ZhcigtLWFjY2VudCknO3NpZ25hbHMucHVzaCh7Y29sOmNvbCx0YWc6J3ByaW1hcnkgc2lnbmFsJyxsb2M6dG9wWzBdLHRleHQ6J0xvY2FsaXplZCBhdHRlbnRpb24gYXJvdW5kIDxlbT4nK25hcisnPC9lbT4gY29uY2VudHJhdGluZyBpbiA8c3Ryb25nPicrdG9wWzBdKyc8L3N0cm9uZz4nLGRlbGF5OjB9KTt1c2UobmFyLHRvcFswXSk7fQogIHZhciBlYXJseT1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIG5vcm1WKGt2WzFdLnZlbG9jaXR5fHwwKT4wLjA0JiYoa3ZbMV0uYXR0ZW50aW9ufHwwKTw0MCYmIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKTt9KVswXTsKICBpZihlYXJseSl7c2lnbmFscy5wdXNoKHtjb2w6JyNlMDc4MjAnLHRhZzonZWFybHkgbW92ZW1lbnQnLGxvYzplYXJseVswXSx0ZXh0OidSZWdpb25hbCBkaXNjb3Vyc2UgYXJvdW5kIDxlbT4nKyhlYXJseVsxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdyZWdpb25hbCBhY3Rpdml0eScpKyc8L2VtPiBzaG93aW5nIGVhcmx5IGFjY2VsZXJhdGlvbiBpbiA8c3Ryb25nPicrZWFybHlbMF0rJzwvc3Ryb25nPicsZGVsYXk6MTYwfSk7dXNlKGVhcmx5WzFdLmRvbWluYW50X25hcnJhdGl2ZSxlYXJseVswXSk7fQogIHZhciBhbmdlcj1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvbWluYW50X2Vtb3Rpb249PT0nYW5nZXInJiYhdXNlZChrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUsa3ZbMF0pJiYoa3ZbMV0uYXR0ZW50aW9ufHwwKT40O30pWzBdOwogIGlmKGFuZ2VyKXtzaWduYWxzLnB1c2goe2NvbDpwYWwuYW5nZXIsdGFnOidlbW90aW9uYWwgc2lnbmFsJyxsb2M6YW5nZXJbMF0sdGV4dDonRnJ1c3RyYXRpb24gc2lnbmFscyBlbGV2YXRlZCBpbiA8c3Ryb25nPicrYW5nZXJbMF0rJzwvc3Ryb25nPiBhcm91bmQgPGVtPicrKGFuZ2VyWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J2dvdmVybmFuY2UnKSsnPC9lbT4nLGRlbGF5OjMyMH0pO3VzZShhbmdlclsxXS5kb21pbmFudF9uYXJyYXRpdmUsYW5nZXJbMF0pO30KICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3dpci1zaWduYWxzJyk7CiAgaWYoIWVsfHwhc2lnbmFscy5sZW5ndGgpIHJldHVybjsKICBlbC5pbm5lckhUTUw9c2lnbmFscy5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsIiBzdHlsZT0iYW5pbWF0aW9uLWRlbGF5Oicrcy5kZWxheSsnbXMiPjxkaXYgY2xhc3M9Indpci1zaWduYWwtYmFyIiBzdHlsZT0iYmFja2dyb3VuZDonK3MuY29sKyciPjwvZGl2PjxkaXYgY2xhc3M9Indpci1zaWduYWwtY29udGVudCI+PGRpdiBjbGFzcz0id2lyLXNpZ25hbC10ZXh0Ij4nK3MudGV4dCsnPC9kaXY+PGRpdiBjbGFzcz0id2lyLXNpZ25hbC1tZXRhIj48c3BhbiBjbGFzcz0id2lyLXNpZ25hbC10YWciIHN0eWxlPSJjb2xvcjonK3MuY29sKyciPicrcy50YWcrJzwvc3Bhbj48c3BhbiBjbGFzcz0id2lyLXNpZ25hbC1sb2MiPicrcy5sb2MrJzwvc3Bhbj48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwp9CgovLyDilIDilIAgU1RBVEUgQ09OVEVYVCArIERFVEFJTCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTdGF0ZUNvbnRleHQobm0pe3RyeXt2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS1jb250ZXh0LycrZW5jb2RlVVJJQ29tcG9uZW50KG5tKSk7aWYoIXIub2spcmV0dXJuIG51bGw7cmV0dXJuIGF3YWl0IHIuanNvbigpO31jYXRjaChlKXtyZXR1cm4gbnVsbDt9fQphc3luYyBmdW5jdGlvbiBmZXRjaERldGFpbChubSl7dHJ5e3ZhciBjPW5ldyBBYm9ydENvbnRyb2xsZXIoKTt2YXIgdGlkPXNldFRpbWVvdXQoZnVuY3Rpb24oKXtjLmFib3J0KCk7fSw1MDAwKTt2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChubSkse3NpZ25hbDpjLnNpZ25hbH0pO2NsZWFyVGltZW91dCh0aWQpO2lmKCFyLm9rKXJldHVybiBmYWxzZTt2YXIgZD1hd2FpdCByLmpzb24oKTtpZihkJiZkLm5hbWUpe3ZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKGQuZW1vdGlvbnN8fHt9KTt2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8ZC5kb21pbmFudF9lbW90aW9ufHxudWxsO1NEW25tXT1PYmplY3QuYXNzaWduKHt9LGQse2Vtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZGVsdGE6ZC5kZWx0YV8yNGh8fDB9KTtMSVZFW25tXT1PYmplY3QuYXNzaWduKExJVkVbbm1dfHx7fSx7YXR0ZW50aW9uOmQuYXR0ZW50aW9uLHZlbG9jaXR5OmQudmVsb2NpdHksZGVsdGE6ZC5kZWx0YV8yNGh8fDAsZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3MsbmFycmF0aXZlczpkLm5hcnJhdGl2ZXMsc2lnbmFsX2NvdW50OmQuc2lnbmFsX2NvdW50LHNvdXJjZV9jb3VudDpkLnNvdXJjZV9jb3VudCxjb25maWRlbmNlOmQuY29uZmlkZW5jZX0pO31yZXR1cm4gdHJ1ZTt9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbZmV0Y2hEZXRhaWxdJyxubSxlLm1lc3NhZ2UpO3JldHVybiBmYWxzZTt9fQo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="

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




@app.get("/api/state/{state_name}")
async def get_state(state_name: str):
    """Full detail for a single state — called when user clicks a state on the map."""
    score = store.scores.get(state_name)
    if not score:
        raise HTTPException(status_code=404, detail=f"State '{state_name}' not found")
    return score


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
