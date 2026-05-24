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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQouc2lnLWhvdmVyLXdyYXB7cG9zaXRpb246cmVsYXRpdmU7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtjdXJzb3I6ZGVmYXVsdH0KLnNpZy1ob3Zlci10aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7dG9wOmNhbGMoMTAwJSArIDEwcHgpO3JpZ2h0OjA7CiAgYmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45Nyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxNHB4O3doaXRlLXNwYWNlOm5vd3JhcDsKICBwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt2aXNpYmlsaXR5OmhpZGRlbjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xOHMsdmlzaWJpbGl0eSAwLjE4czsKICB6LWluZGV4Ojk5OTk7Cn0KLnNpZy1ob3Zlci13cmFwOmhvdmVyIC5zaWctaG92ZXItdGlwe29wYWNpdHk6MTt2aXNpYmlsaXR5OnZpc2libGV9Ci5zaWctaG92ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NXB4O29wYWNpdHk6MC43fQouc2lnLWhvdmVyLXNvdXJjZXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA0ZW19Ci5saXZlLWluZGljYXRvcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo3cHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtOwp9Ci5saXZlLWRvdHt3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiM0YWRlODA7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoNzQsMjIyLDEyOCwwLjcpO2FuaW1hdGlvbjpsZCAyLjVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGxkezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjM1O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KLmNsb2Nre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNGVtfQoKLyogSEVSTyAqLwouaGVyb3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgcGFkZGluZzo3MnB4IDM2cHggMDsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87Cn0KLmhlcm8tZXllYnJvd3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMzJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MjRweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4fQouaGVyby1leWVicm93OjpiZWZvcmV7Y29udGVudDonJzt3aWR0aDoxNnB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1mYWludCk7b3BhY2l0eTowLjV9Ci5oZXJvLWJyYW5kLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOm5vcm1hbDtmb250LXNpemU6Y2xhbXAoMzZweCw0LjJ2dyw2NHB4KTtsaW5lLWhlaWdodDoxO2xldHRlci1zcGFjaW5nOi0wLjAzZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW46MH0KLmhlcm8tYnJhbmQtbmFtZSBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjojZThjNGEwO2FuaW1hdGlvbjpwdWxzZU5hbWVHbG93IDVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlTmFtZUdsb3d7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC43Mn19Ci5oZXJvLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNXB4LDEuNXZ3LDIwcHgpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbjowIDAgMTJweCAwO21heC13aWR0aDo0ODBweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLWRlc2N7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEzcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjY7bWF4LXdpZHRoOjQwMHB4O21hcmdpbjowIDAgNnB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1zdWItbGluZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNik7bWFyZ2luOjAgMCAyMHB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1wdWxzZS1zaWduYWx7cG9zaXRpb246cmVsYXRpdmU7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtmbGV4LXNocmluazowfQouaHBzLWNvcmV7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuOTthbmltYXRpb246aHBzQ29yZSA0cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBocHNDb3JlezAlLDEwMCV7b3BhY2l0eTowLjk7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNDt0cmFuc2Zvcm06c2NhbGUoMC43NSl9fQouaHBzLXJpbmd7cG9zaXRpb246YWJzb2x1dGU7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1hY2NlbnQpO2FuaW1hdGlvbjpocHNSaW5nIDRzIGVhc2Utb3V0IGluZmluaXRlfQouaHBzLXJpbmcucjF7aW5zZXQ6MXB4O2FuaW1hdGlvbi1kZWxheTowc30uaHBzLXJpbmcucjJ7aW5zZXQ6LTNweDthbmltYXRpb24tZGVsYXk6MS40cztib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zNSl9CkBrZXlmcmFtZXMgaHBzUmluZ3swJXtvcGFjaXR5OjAuNjt0cmFuc2Zvcm06c2NhbGUoMC43KX0xMDAle29wYWNpdHk6MDt0cmFuc2Zvcm06c2NhbGUoMS42KX19CgovKiBTSUdOQVRVUkUgSU5TSUdIVCAqLwouc2lnbmF0dXJlLWluc2lnaHR7CiAgbWFyZ2luLXRvcDowOwogIHBhZGRpbmc6MTRweCAyMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNikgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIG1heC13aWR0aDo5MDBweDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47Cn0KLnNpZ25hdHVyZS1pbnNpZ2h0OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgdmFyKC0tYWNjZW50KSwgdHJhbnNwYXJlbnQpOwp9Ci5zaS1sYWJlbHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouc2ktdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE0cHgsMS40dncsMThweCk7CiAgZm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluayk7bGluZS1oZWlnaHQ6MS41O2xldHRlci1zcGFjaW5nOi0wLjAxZW07Cn0KLnNpLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tYWNjZW50KX0KLnNpLXN1YnsKICBtYXJnaW4tdG9wOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTsKICBsZXR0ZXItc3BhY2luZzowLjA0ZW07ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDtmbGV4LXdyYXA6d3JhcDsKfQouc2ktdGFnewogIHBhZGRpbmc6MnB4IDhweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTsKICBmb250LXNpemU6OS41cHg7Cn0KCi8qIE5BUlJBVElWRSBTVFJJUCAqLwoKLnN0cmlwLXRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjRweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQouc3RyaXAtdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xMil9Ci5zdHJpcC10YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbHsKICBmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1iZzEpO3BhZGRpbmc6MDsKfQouc3RyaXAtY29sLWhlYWR7CiAgcGFkZGluZzoxMHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwp9Ci5zdHJpcC1jb2wtaGVhZC5mYWRle2NvbG9yOnZhcigtLWZhbGwpfQouc3RyaXAtY29sLWhlYWQucmlzZTJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5zdHJpcC1jb2wtaGVhZC5zaGlmdHtjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sLWJvZHl7cGFkZGluZzoxMnB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OHB4fQouc3RyaXAtaXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjhweDsKfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5zdHJpcC1ub3Rle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5zdHJpcC1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTtmb250LXNpemU6MTRweDtmbGV4LXNocmluazowfQoKLyogTUFJTiBMQVlPVVQgKi8KLm1haW57CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM2MHB4OwogIGdhcDoxNHB4OwogIG1pbi13aWR0aDowOwp9CgovKiBNQVAgKi8KLm1hcC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBvdmVyZmxvdzpoaWRkZW47cG9zaXRpb246cmVsYXRpdmU7Cn0KLm1hcC1jYXJkOjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA3MCUgNTAlIGF0IDM1JSAwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNSkgMCUsIHRyYW5zcGFyZW50IDYwJSksCiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA1MCUgNDAlIGF0IDgwJSAxMDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMCUsIHRyYW5zcGFyZW50IDYwJSk7Cn0KLm1hcC10b3B7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47CiAgcGFkZGluZzoxMnB4IDE4cHggMDsKfQoubWFwLXRpdGxlLWJsb2NrIC5tdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE3cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtdGl0bGUtYmxvY2sgLm1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5sZWdlbmQtYmFyewogIGhlaWdodDozcHg7d2lkdGg6ODBweDtib3JkZXItcmFkaXVzOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCwjMGUyMDM1LCMxYTU1ODAgMjUlLCM4YTVjMTggNTUlLCNjMDM4MWEgODAlLCNlMDEwMjApOwp9Ci5sYXllci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsKICBwYWRkaW5nOjEwcHggMjBweCA2cHg7Cn0KLmxheWVyLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLmx0YWJze2Rpc3BsYXk6ZmxleDtnYXA6M3B4fQoubHRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6M3B4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5sdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wOCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5sdGFie2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo1cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6dmlzaWJsZX0KLmx0YWItaW5mb3t3aWR0aDoxM3B4O2hlaWdodDoxM3B4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjIpO2ZvbnQtc2l6ZTo4cHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zdHlsZTppdGFsaWM7Zm9udC13ZWlnaHQ6NjAwO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4zNSk7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjdXJzb3I6aGVscDtmbGV4LXNocmluazowO3RyYW5zaXRpb246YWxsIDAuMTVzO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTAwfQoubHRhYi1pbmZvOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2NlbnQpO2NvbG9yOnZhcigtLWFjY2VudCl9CiNsdGFiLXRvb2x0aXB7cG9zaXRpb246Zml4ZWQ7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMTIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjt3aWR0aDoyMzBweDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNik7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzO3otaW5kZXg6OTk5OTk7ZGlzcGxheTpub25lfQojbHRhYi10b29sdGlwLnZpc2libGV7b3BhY2l0eToxO2Rpc3BsYXk6YmxvY2t9Ci5sdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9CgoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmc6MTJweCAxNnB4IDE2cHg7Cn0KLm1hcC1pbm5lcntwb3NpdGlvbjpyZWxhdGl2ZTthc3BlY3QtcmF0aW86MS8xO3dpZHRoOjEwMCV9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrO292ZXJmbG93OnZpc2libGV9CgovKiBtYXAgc3RhdGUgc3R5bGVzICovCiNpbmRpYS1tYXAgLnN0YXRlewogIGN1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246ZmlsdGVyIDAuMjVzIGVhc2UsIHN0cm9rZS13aWR0aCAwLjJzIGVhc2UsIHN0cm9rZSAwLjJzIGVhc2U7Cn0KI2luZGlhLW1hcCAuc3RhdGU6aG92ZXJ7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC43KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxcHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjI1KSBkcm9wLXNoYWRvdygwIDAgMTBweCByZ2JhKDI1NSwyNTUsMjU1LDAuMikpOwp9CiNpbmRpYS1tYXAgLnN0YXRlLnNlbGVjdGVkewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuOSkgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MS40cHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjM1KSBkcm9wLXNoYWRvdygwIDAgMTZweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpOwp9CgovKiBhbmltYXRlZCBwdWxzZSByaW5ncyAqLwoucHVsc2UtcmluZ3tmaWxsOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZX0KLnB1bHNlLXJpbmcucDF7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5wdWxzZS1yaW5nLnAye2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IDAuOXMgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHJ7CiAgMCV7cjo0O29wYWNpdHk6MC43O3N0cm9rZS13aWR0aDoxLjJ9CiAgMTAwJXtyOjI2O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4yfQp9CgovKiBhdG1vc3BoZXJpYyBnbG93IGJlaGluZCBob3Qgc3RhdGVzICovCi5zdGF0ZS1nbG93e3BvaW50ZXItZXZlbnRzOm5vbmU7ZmlsbDpub25lfQpAa2V5ZnJhbWVzIGdsb3dQdWxzZXswJSwxMDAle29wYWNpdHk6MC4xMn01MCV7b3BhY2l0eTowLjIyfX0KCi5tYXAtdG9vbHRpcHsKICBwb3NpdGlvbjphYnNvbHV0ZTtwb2ludGVyLWV2ZW50czpub25lOwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC45NSk7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjlweDsKICBwYWRkaW5nOjEycHggMTRweDtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTJzO3otaW5kZXg6OTk5OTttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmcjtnYXA6MThweDsKfQoubmFyLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE0cHgpO292ZXJmbG93OmhpZGRlbjsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOwp9Ci5uYy1oZWFkewogIHBhZGRpbmc6MTZweCAyMHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsKfQoubmMtYm9keXtwYWRkaW5nOjhweCAyMHB4IDE2cHg7ZmxleDoxO292ZXJmbG93LXk6YXV0bzt9Ci5uYy10aXRsZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKX0KLm5jLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDVlbTttYXJnaW4tdG9wOjJweH0KLm5jLWJvZHl7cGFkZGluZzoxM3B4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MH0KCi5tb20taXR7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4OwogIHBhZGRpbmc6N3B4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubW9tLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tb20tcmt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7d2lkdGg6MTNweDtmbGV4LXNocmluazowfQoubW9tLWluZntmbGV4OjF9Ci5tb20tbm17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5tb20tc3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KLm1vbS1wY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2ZvbnQtd2VpZ2h0OjQwMDtmbGV4LXNocmluazowfQoubW9tLXBjLnJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tb20tcGMuZntjb2xvcjp2YXIoLS1mYWxsKX0KLm1vbS10cntoZWlnaHQ6MS41cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O21hcmdpbjozcHggMCAwO292ZXJmbG93OmhpZGRlbn0KLm1vbS1mbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweH0KCi5yZWctaXR7CiAgZGlzcGxheTpmbGV4O2dhcDo5cHg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDsKICBwYWRkaW5nOjhweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzOwp9Ci5yZWctaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLnJlZy1pdDpob3ZlcntvcGFjaXR5OjAuNzV9Ci5yZWctYmFkZ2V7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjA3ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIHBhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNCk7CiAgY29sb3I6dmFyKC0tYWNjZW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4O3doaXRlLXNwYWNlOm5vd3JhcDsKfQoucmVnLWZse2ZsZXg6MTtmb250LXNpemU6MTEuNXB4O2xpbmUtaGVpZ2h0OjEuNX0KLnJlZy1mcm9te2NvbG9yOnZhcigtLWZhaW50KX0KLnJlZy1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTttYXJnaW46MCA0cHh9Ci5yZWctdG97Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5yZWctdG17Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4fQoKLyogRkFWUyAqLwouZmF2c3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAzNnB4IDQwcHg7Cn0KLmZhdnMtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQouZmF2cy1yb3d7ZGlzcGxheTpmbGV4O2dhcDoxMHB4O292ZXJmbG93LXg6YXV0bztwYWRkaW5nLWJvdHRvbTozcHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXJ7aGVpZ2h0OjJweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MXB4fQouZmF2LWNhcmR7CiAgZmxleDowIDAgMTkwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MTJweDtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4czsKfQouZmF2LWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMjIpO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMil9Ci5mYy1oZWFke2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjdweH0KLmZjLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjp2YXIoLS1pbmspfQouZmMtc2N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouZmMtcm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDozcHh9Ci5mYy1yb3cgLnZ7Y29sb3I6dmFyKC0tZGltKTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHh9Ci5mYXZzLWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LXN0eWxlOml0YWxpYztwYWRkaW5nOjRweCAwfQoKLyogRk9PVCAqLwouZm9vdHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjQ4cHggMzZweCA2MHB4O21heC13aWR0aDo1ODBweDttYXJnaW46MCBhdXRvO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmZvb3QtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbi1ib3R0b206MTRweH0KLmZvb3QtbGluZXtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjEuODttYXJnaW4tYm90dG9tOjEycHh9Ci5mb290LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KX0KCi8qIGFuaW1hdGlvbnMgKi8KQGtleWZyYW1lcyBmYWRlVXB7ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoubWFwLWNhcmQsLnN0YXRlLXBhbmVsLC5uYXItY2FyZCwuc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uOmZhZGVVcCAwLjU1cyBjdWJpYy1iZXppZXIoLjIsLjgsLjIsMSkgYmFja3dhcmRzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjA3c30KLm5hci1jYXJkOm50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC4xNHN9Ci5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb24tZGVsYXk6MC4wNXN9CgpAbWVkaWEobWF4LXdpZHRoOjExMDBweCl7CiAgLm1haW57Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAuc3RhdGUtcGFuZWx7bWF4LWhlaWdodDpub25lfQogIC5uYXItcm93e2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9Cn0KCi8qIOKUgOKUgCBXSEFUIElORElBIElTIFJFQUNUSU5HIFRPIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwoud2lyLXNlY3Rpb257CiAgZmxleDoxO21pbi13aWR0aDowOwogIHBhZGRpbmc6MDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLndpci1oZWFkZXJ7CiAgcGFkZGluZzoxOHB4IDIycHggMTRweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Cn0KLndpci10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuM2VtOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC44NTsKfQoud2lyLWxpdmV7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMWVtOwp9Ci53aXItbGl2ZS1kb3R7CiAgd2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojMzlmZjE0OwogIGJveC1zaGFkb3c6MCAwIDZweCByZ2JhKDU3LDI1NSwyMCwwLjYpOwogIGFuaW1hdGlvbjp3aXJMaXZlUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyB3aXJMaXZlUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC4zfX0KLndpci1zaWduYWxze2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47ZmxleDoxO292ZXJmbG93OmhpZGRlbn0KLndpci1zaWduYWx7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjA7CiAgcGFkZGluZzoxM3B4IDIycHg7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzNSk7CiAgb3BhY2l0eTowOwogIGFuaW1hdGlvbjp3aXJGYWRlSW4gMC42cyBlYXNlIGZvcndhcmRzOwogIHBvc2l0aW9uOnJlbGF0aXZlO2N1cnNvcjpkZWZhdWx0OwogIHRyYW5zaXRpb246YmFja2dyb3VuZCAwLjE1czsKfQoud2lyLXNpZ25hbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci53aXItc2lnbmFsOmxhc3QtY2hpbGR7Ym9yZGVyLWJvdHRvbTpub25lfQpAa2V5ZnJhbWVzIHdpckZhZGVJbntmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWCgtNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoud2lyLXNpZ25hbC1iYXJ7CiAgd2lkdGg6MnB4O2JvcmRlci1yYWRpdXM6MXB4O2ZsZXgtc2hyaW5rOjA7CiAgbWFyZ2luLXJpZ2h0OjE0cHg7bWFyZ2luLXRvcDo0cHg7CiAgYWxpZ24tc2VsZjpzdHJldGNoO21pbi1oZWlnaHQ6MTZweDsKICBvcGFjaXR5OjAuNjsKfQoud2lyLXNpZ25hbC1jb250ZW50e2ZsZXg6MTttaW4td2lkdGg6MH0KLndpci1zaWduYWwtdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE0LjVweDtmb250LXdlaWdodDozMDA7CiAgY29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQoud2lyLXNpZ25hbC10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOmluaGVyaXQ7b3BhY2l0eTowLjh9Ci53aXItc2lnbmFsLW1ldGF7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6NHB4Owp9Ci53aXItc2lnbmFsLXRhZ3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6N3B4O2xldHRlci1zcGFjaW5nOjAuMTRlbTsKICB0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7b3BhY2l0eTowLjQ1Owp9Ci53aXItc2lnbmFsLWxvY3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci53aXItbG9hZGluZ3sKICBkaXNwbGF5OmZsZXg7Z2FwOjZweDtwYWRkaW5nOjIwcHggMjJweDthbGlnbi1pdGVtczpjZW50ZXI7Cn0KLndpci1kb3R7d2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjp3aXJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZX0KLndpci1kb3Q6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjJzfQoud2lyLWRvdDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuNHN9CkBrZXlmcmFtZXMgd2lyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KCi5uYy1oZWFke3BhZGRpbmc6MTRweCAxOHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O2ZsZXgtc2hyaW5rOjB9Ci5uYy1oaW50e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1sZWZ0OmF1dG99Ci5uYy1sb2FkaW5ne2NvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwfQoucDI0LXNlY3Rpb257cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxO21heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzZweCA0OHB4fQoucDI0LWhlYWRlcnttYXJnaW4tYm90dG9tOjIycHh9Ci5wMjQtdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMHB4O2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1pbmspO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5wMjQtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDo0cHh9Ci5wMjQtY2FyZHN7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczpyZXBlYXQoMywxZnIpO2dhcDoxNHB4fQoucDI0LWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtncmlkLWNvbHVtbjoxLy0xO3BhZGRpbmc6MjBweCAwfQoucDI0LWNhcmR7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O3BhZGRpbmc6MThweCAyMHB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjEwcHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVufQoucDI0LWNhcmQtdGltZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQoucDI0LWNhcmQtbmFye2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjN9Ci5wMjQtY2FyZC1pbnNpZ2h0e2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMS41cHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42fQoucDI0LWNhcmQtc3RhdGV7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6MnB4fQoucDI0LWNhcmQtc3RhdGUtbmFtZXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnAyNC1jYXJkLXN0YXRlLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQoucDI0LWNhcmQtZm9vdGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Ym9yZGVyLXRvcDoxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtwYWRkaW5nLXRvcDo4cHg7bWFyZ2luLXRvcDoycHh9Ci5wMjQtY2FyZC1lbW97Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnAyNC1jYXJkLXNpZ3N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5wMjQtY2FyZC1uYXJze2Rpc3BsYXk6ZmxleDtmbGV4LXdyYXA6d3JhcDtnYXA6NHB4fQoucDI0LWNhcmQtbmFyLXRhZ3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbC1zbXtmb250LXNpemU6Y2xhbXAoMTNweCwxLjN2dywxNnB4KSFpbXBvcnRhbnR9Ci5zYy1ob3ZlcmFibGV7cG9zaXRpb246cmVsYXRpdmU7Y3Vyc29yOmRlZmF1bHR9Ci5zYy10b29sdGlwe2Rpc3BsYXk6bm9uZTtwb3NpdGlvbjphYnNvbHV0ZTtib3R0b206Y2FsYygxMDAlICsgOHB4KTtsZWZ0OjUwJTt0cmFuc2Zvcm06dHJhbnNsYXRlWCgtNTAlKTtiYWNrZ3JvdW5kOnJnYmEoOCwxMiwyMCwwLjk3KTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MTJweCAxNHB4O3dpZHRoOjIyMHB4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS41O3otaW5kZXg6OTk5OTtwb2ludGVyLWV2ZW50czpub25lO3doaXRlLXNwYWNlOm5vcm1hbDt0ZXh0LWFsaWduOmxlZnQ7Ym94LXNoYWRvdzowIDhweCAyNHB4IHJnYmEoMCwwLDAsMC41KX0KLnNjLWhvdmVyYWJsZTpob3ZlciAuc2MtdG9vbHRpcHtkaXNwbGF5OmJsb2NrfQouc2MtdGlwLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWFjY2VudCk7bWFyZ2luLWJvdHRvbTo2cHh9Ci5zYy10aXAtcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtnYXA6NnB4O21hcmdpbi1ib3R0b206NHB4O2ZvbnQtc2l6ZToxMXB4fQouc2MtdGlwLXJvdyBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDB9Ci5uYXItaXRlbXtwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpfQoubmFyLWl0ZW06bGFzdC1jaGlsZHtib3JkZXItYm90dG9tOm5vbmV9Ci5uaS1uYW1le2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwO3dvcmQtYnJlYWs6YnJlYWstd29yZDtsaW5lLWhlaWdodDoxLjQ7bWFyZ2luLWJvdHRvbTozcHh9Ci5uaS1zdGF0ZXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo1cHg7d29yZC1icmVhazpicmVhay13b3JkfQoubmktdHJhY2t7aGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHh9Ci5uaS1maWxse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4O3RyYW5zaXRpb246d2lkdGggMC41cyBlYXNlfQoKLnNoaWZ0LXNlY3Rpb257cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxO21heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOjAgMzZweCAzNnB4fQouc2hpZnQtaGVhZGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbToxNnB4fQouc2hpZnQtdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxOHB4O2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1pbmspfQouc2hpZnQtdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjRweH0KLnNoaWZ0LXRhYntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMTJlbTtwYWRkaW5nOjRweCAxMHB4O2JvcmRlci1yYWRpdXM6NHB4O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOm5vbmU7Y29sb3I6dmFyKC0tZmFpbnQpO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQouc2hpZnQtdGFiLmFjdGl2ZXtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjMpO2NvbG9yOnZhcigtLWFjY2VudCl9Ci5zaGlmdC1jYXJkc3tkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoxNHB4fQouc2hpZnQtY2FyZHtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjEycHg7cGFkZGluZzoxNnB4IDE4cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweH0KLnNoaWZ0LWNhcmQtZmFkaW5ne2ZsZXg6MX0KLnNoaWZ0LWNhcmQtZmFkaW5nIC5zYy1sYmx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6IzNiYjhkODttYXJnaW4tYm90dG9tOjRweH0KLnNoaWZ0LWNhcmQtcmlzaW5ne2ZsZXg6MX0KLnNoaWZ0LWNhcmQtcmlzaW5nIC5zYy1sYmx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6I2UwNWEyODttYXJnaW4tYm90dG9tOjRweH0KLnNoaWZ0LWNhcmQtbmFtZXtmb250LXNpemU6MTRweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDtsaW5lLWhlaWdodDoxLjN9Ci5zaGlmdC1jYXJkLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6M3B4fQouc2hpZnQtYXJyb3d7Y29sb3I6dmFyKC0tYm9yZGVyMik7Zm9udC1zaXplOjE2cHg7ZmxleC1zaHJpbms6MH0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9Imx0YWItdG9vbHRpcCI+PC9kaXY+Cgo8IS0tIExPQURFUiAtLT4KPGRpdiBpZD0iYXBwLWxvYWRlciIgc3R5bGU9InBvc2l0aW9uOmZpeGVkO2luc2V0OjA7ei1pbmRleDo5OTk5ODtiYWNrZ3JvdW5kOiMwNjA5MTA7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOm9wYWNpdHkgMC44cyBlYXNlLHZpc2liaWxpdHkgMC44cyBlYXNlOyI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7d2lkdGg6NjRweDtoZWlnaHQ6NjRweDttYXJnaW4tYm90dG9tOjM2cHgiPgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiNlMDVhMjg7YW5pbWF0aW9uOmxkclB1bHNlIDJzIGVhc2UtaW4tb3V0IGluZmluaXRlIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0OjE2cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjpsZHJSaW5nIDJzIGVhc2Utb3V0IGluZmluaXRlIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTUpO2FuaW1hdGlvbjpsZHJSaW5nIDJzIGVhc2Utb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjVzIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0Oi0xMHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4wNyk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjFzIj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonUGxheWZhaXIgRGlzcGxheScsR2VvcmdpYSxzZXJpZjtmb250LXNpemU6Y2xhbXAoMjhweCw1dncsNDJweCk7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6I2YwZWNlNDtsaW5lLWhlaWdodDoxO21hcmdpbi1ib3R0b206MTBweCI+CiAgICA8ZW0gc3R5bGU9ImNvbG9yOiNlOGM0YTA7Zm9udC1zdHlsZTppdGFsaWMiPlB1bHNlPC9lbT4gb2YgSW5kaWEKICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonQ291cmllciBOZXcnLG1vbm9zcGFjZTtmb250LXNpemU6MTFweDtsZXR0ZXItc3BhY2luZzowLjI4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC40KTttYXJnaW4tYm90dG9tOjI4cHgiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXM8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonQ291cmllciBOZXcnLG1vbm9zcGFjZTtmb250LXNpemU6MTBweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4yNSk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweCI+CiAgICA8c3Bhbj5Ob3QgbmV3czwvc3Bhbj48c3BhbiBzdHlsZT0ib3BhY2l0eTowLjMiPsK3PC9zcGFuPjxzcGFuPk5vdCBwcmVkaWN0aW9uPC9zcGFuPjxzcGFuIHN0eWxlPSJvcGFjaXR5OjAuMyI+wrc8L3NwYW4+CiAgICA8c3Bhbj5KdXN0IDxzcGFuIHN0eWxlPSJjb2xvcjojMzlmZjE0O3RleHQtc2hhZG93OjAgMCAxMHB4IHJnYmEoNTcsMjU1LDIwLDAuNSk7YW5pbWF0aW9uOmxkckdsb3cgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPm9ic2VydmF0aW9uPC9zcGFuPjwvc3Bhbj4KICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJtYXJnaW4tdG9wOjQ4cHg7ZGlzcGxheTpmbGV4O2dhcDo2cHgiPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjJzIj48L3NwYW4+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MC40cyI+PC9zcGFuPgogIDwvZGl2Pgo8L2Rpdj4KPHN0eWxlPgpAa2V5ZnJhbWVzIGxkclB1bHNlezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjU7dHJhbnNmb3JtOnNjYWxlKDAuOCl9fQpAa2V5ZnJhbWVzIGxkclJpbmd7MCV7dHJhbnNmb3JtOnNjYWxlKDAuOCk7b3BhY2l0eTowLjZ9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS41KTtvcGFjaXR5OjB9fQpAa2V5ZnJhbWVzIGxkckdsb3d7MCUsMTAwJXt0ZXh0LXNoYWRvdzowIDAgMTBweCByZ2JhKDU3LDI1NSwyMCwwLjUpfTUwJXt0ZXh0LXNoYWRvdzowIDAgMjJweCByZ2JhKDU3LDI1NSwyMCwwLjkpLDAgMCA0MHB4IHJnYmEoNTcsMjU1LDIwLDAuMyl9fQpAa2V5ZnJhbWVzIGxkckRvdHswJSw4MCUsMTAwJXt0cmFuc2Zvcm06c2NhbGUoMC42KTtvcGFjaXR5OjAuM300MCV7dHJhbnNmb3JtOnNjYWxlKDEpO29wYWNpdHk6MX19Cjwvc3R5bGU+Cgo8ZGl2IGNsYXNzPSJ0b3BiYXIiPgogIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLW1hcmsiPjxzcGFuIGNsYXNzPSJicmFuZC1wdWxzZS1kb3QiPjwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLXRleHQtYmxvY2siPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtbmFtZSI+PGVtIGNsYXNzPSJicmFuZC1wdWxzZS13b3JkIj5QdWxzZTwvZW0+IG9mIEluZGlhPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtdGFnbGluZSI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3NwYW4+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJ0b3BiYXItciI+CiAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9ImxpdmUtaW5kaWNhdG9yIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItdGlwIj4KICAgICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItbGFiZWwiPk9ic2VydmVkIGZyb208L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItc291cmNlcyI+cmVnaW9uYWwgbWVkaWEgwrcgcHVibGljIGRpc2N1c3Npb24gwrcgaW5kZXBlbmRlbnQgcmVwb3J0aW5nIMK3IHNvY2lhbCBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgoKICA8IS0tIExJVkUgU1RBVFMgU1RSSVAgLS0+CjxkaXYgaWQ9InN0YXRzLXN0cmlwIiBzdHlsZT0iCiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoyOwogIGJhY2tncm91bmQ6cmdiYSg5LDEzLDIxLDAuOSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjA4KTsKICBwYWRkaW5nOjAgMzZweDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6c3RyZXRjaDsKIj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1zaWduYWxzIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5TaWduYWxzIHRyYWNrZWQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCBzYy12YWwtc20iIGlkPSJzYy1zaWduYWxzLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1zaWduYWxzLXN1YiI+bG9hZGluZy4uLjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIiBvbmNsaWNrPSJzZWxlY3RIb3R0ZXN0KCkiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkhpZ2hlc3QgYXR0ZW50aW9uPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1ob3R0ZXN0LXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1ob3R0ZXN0LXN1YiI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtaG90dGVzdC10aXAiPjxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+V2h5IHRoaXMgc3RhdGU/PC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlBlYWsgYW5nZXIgc3RhdGU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWFuZ2VyLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1hbmdlci1zdWIiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLWFuZ2VyLXRpcCI+PGRpdiBjbGFzcz0ic2MtdGlwLXRpdGxlIj5BbmdlciBzaWduYWxzPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgcmlzaW5nPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1yaXNpbmctdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLXJpc2luZy1zdWIiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLXJpc2luZy10aXAiPjxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+UmlzaW5nIHNpZ25hbHM8L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIHNjLWhvdmVyYWJsZSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIHJpc2luZyBuYXJyYXRpdmU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCBzYy12YWwtc20iIGlkPSJzYy1uYXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hci1zdWIiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLW5hci10aXAiPjxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+QWN0aXZlIGluPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbC12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbC1zdWIiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLWNvb2wtdGlwIj48ZGl2IGNsYXNzPSJzYy10aXAtdGl0bGUiPkNvb2xpbmcgc2lnbmFsczwvZGl2PjwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCiAgPCEtLSBTSUdOQVRVUkUgSU5TSUdIVCArIE5BUlJBVElWRSBTVFJJUCBzaWRlIGJ5IHNpZGUgLS0+CiAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoxOHB4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7bWFyZ2luLXRvcDoxNnB4O21hcmdpbi1ib3R0b206MDttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbi1sZWZ0OmF1dG87bWFyZ2luLXJpZ2h0OmF1dG87cGFkZGluZzowIDM2cHg7Ij4KICAgIDxkaXYgY2xhc3M9Indpci1zZWN0aW9uIj4KICAgICAgPGRpdiBjbGFzcz0id2lyLWhlYWRlciI+CiAgICAgICAgPGRpdiBjbGFzcz0id2lyLXRpdGxlIj5XaGF0IEluZGlhIGlzIHJlYWN0aW5nIHRvPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0id2lyLWxpdmUiPjxzcGFuIGNsYXNzPSJ3aXItbGl2ZS1kb3QiPjwvc3Bhbj5saXZlIHNpZ25hbHM8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Indpci1zaWduYWxzIiBpZD0id2lyLXNpZ25hbHMiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci1sb2FkaW5nIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj48c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZmxleDowIDAgMzYwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOyI+CiAgICAgIDwhLS0gaGVhZGVyIC0tPgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsiPgogICAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj5OYXJyYXRpdmUgc2hpZnRzPC9zcGFuPgogICAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MnB4OyI+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPCEtLSBzaGlmdHMgbGlzdCAtLT4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoxMHB4IDE0cHg7Z2FwOjZweDsiIGlkPSJzaGlmdC1saXN0Ij48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+CgoKPCEtLSBNQUlOOiBNQVAgKyBTVEFURSBQQU5FTCAtLT4KPGRpdiBjbGFzcz0ibWFpbiI+CgogIDxkaXYgY2xhc3M9Im1hcC1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im1hcC10b3AiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtdGl0bGUtYmxvY2siPgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij5JbmRpYSAmbWRhc2g7IGNvbGxlY3RpdmUgYXR0ZW50aW9uPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibXMiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IGxpdmUgc2lnbmFsIGNvbXBvc2l0ZTwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGVnZW5kIj48c3Bhbj5xdWlldDwvc3Bhbj48ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj48c3Bhbj5hY3RpdmU8L3NwYW4+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxheWVyLXJvdyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsYXllci1sYWJlbCI+Vmlldzwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibHRhYnMiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iYXR0ZW50aW9uIj5BdHRlbnRpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IldoaWNoIHN0YXRlcyBhcmUgcmVjZWl2aW5nIHRoZSBtb3N0IHB1YmxpYyBmb2N1cy4gSGlnaCBhdHRlbnRpb24gPSBjb25jZW50cmF0ZWQgbmV3cyBjb3ZlcmFnZSBhbmQgcG9saXRpY2FsIGFjdGl2aXR5LiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IlRoZSBkb21pbmFudCBlbW90aW9uYWwgdG9uZSDigJQgYW54aW91cywgYW5ncnksIGhvcGVmdWwsIHByb3VkIG9yIGZlYXJmdWwuIFJldmVhbHMgdGhlIHBzeWNob2xvZ2ljYWwgdW5kZXJjdXJyZW50IG9mIHBvbGl0aWNhbCBhdHRlbnRpb24uIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiIgZGF0YS1sYXllcj0idmVsb2NpdHkiPk1vbWVudHVtIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJJcyBhdHRlbnRpb24gcmlzaW5nIG9yIGZhbGxpbmc/IFJpc2luZyA9IG5hcnJhdGl2ZSBhY2NlbGVyYXRpbmcuIENvb2xpbmcgPSBsb3NpbmcgdHJhY3Rpb24uIFNob3dzIHN0YXRlcyBlbnRlcmluZyBvciBleGl0aW5nIGEgcG9saXRpY2FsIGN5Y2xlLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJtYXAtc3ZnLXdyYXAiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtaW5uZXIiPgogICAgICAgIDxzdmcgaWQ9ImluZGlhLW1hcCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgogICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iYW1iR2xvdyIgY3g9IjUwJSIgY3k9IjUwJSIgcj0iNTAlIj4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSJyZ2JhKDIyNCw5MCw0MCwwLjA0KSIvPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0idHJhbnNwYXJlbnQiLz4KICAgICAgICAgICAgPC9yYWRpYWxHcmFkaWVudD4KICAgICAgICAgICAgPGZpbHRlciBpZD0ic3RhdGVHbG93IiB4PSItMzAlIiB5PSItMzAlIiB3aWR0aD0iMTYwJSIgaGVpZ2h0PSIxNjAlIj4KICAgICAgICAgICAgICA8ZmVHYXVzc2lhbkJsdXIgaW49IlNvdXJjZUdyYXBoaWMiIHN0ZERldmlhdGlvbj0iOCIgcmVzdWx0PSJibHVyIi8+CiAgICAgICAgICAgICAgPGZlQ29tcG9zaXRlIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9ImJsdXIiIG9wZXJhdG9yPSJvdmVyIi8+CiAgICAgICAgICAgIDwvZmlsdGVyPgogICAgICAgICAgPC9kZWZzPgogICAgICAgICAgPHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI4MDAiIGZpbGw9InVybCgjYW1iR2xvdykiLz4KICAgICAgICAgIDxnIGlkPSJtYXAtZ2xvdyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1zdGF0ZXMiPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtcHVsc2VzIj48L2c+CiAgICAgICAgPC9zdmc+CiAgICAgICAgPGRpdiBjbGFzcz0ibWFwLXRvb2x0aXAiIGlkPSJ0b29sdGlwIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBTVEFURSBQQU5FTCAtLT4KICA8ZGl2IGNsYXNzPSJzdGF0ZS1wYW5lbCIgaWQ9InN0YXRlLWRldGFpbCI+CiAgICA8ZGl2IGNsYXNzPSJwYW5lbC1lbXB0eSI+CiAgICAgIDxzdmcgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEiPgogICAgICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0iTTEyIDh2NE0xMiAxNmguMDEiLz4KICAgICAgPC9zdmc+CiAgICAgIDxkaXYgY2xhc3M9InBlLXQiPlNlbGVjdCBhIHN0YXRlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InBlLXMiPkNsaWNrIGFueSByZWdpb24gb24gdGhlIG1hcDxici8+dG8gb3BlbiBpdHMgbmFycmF0aXZlIHBhbmVsLjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cgo8L2Rpdj4KCjwhLS0gTkFSUkFUSVZFIFJPVyAtLT4KPGRpdiBjbGFzcz0ibmFyLXJvdyI+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SaXNpbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLWhpbnQiPmdhaW5pbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyaXNpbmctbGlzdCI+PGRpdiBjbGFzcz0ibmMtbG9hZGluZyI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPgogICAgICA8c3BhbiBjbGFzcz0ibmMtZG90IGZhbGwiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLWhpbnQiPmxvc2luZyB0cmFjdGlvbjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9ImRlY2xpbmluZy1saXN0Ij48ZGl2IGNsYXNzPSJuYy1sb2FkaW5nIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKCiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIEp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDhweCByZ2JhKDU3LDI1NSwyMCwwLjQpIj5vYnNlcnZhdGlvbjwvc3Bhbj4uPC9kaXY+CjwvZGl2PgoKPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vdG9wb2pzb24tY2xpZW50QDMuMS4wL2Rpc3QvdG9wb2pzb24tY2xpZW50Lm1pbi5qcyI+PC9zY3JpcHQ+CjxzY3JpcHQ+CnZhciBBUElfQkFTRT0obG9jYXRpb24uaG9zdG5hbWU9PT0nbG9jYWxob3N0J3x8bG9jYXRpb24uaG9zdG5hbWU9PT0nMTI3LjAuMC4xJyk/J2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMCc6Jyc7CgovLyBBUEkKYXN5bmMgZnVuY3Rpb24gZmV0Y2hBbGxTdGF0ZXMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZXMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgcm93cz1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFyb3dzfHwhcm93cy5sZW5ndGgpIHJldHVybjsKICAgIHJvd3MuZm9yRWFjaChmdW5jdGlvbihyb3cpewogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhyb3cuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbUVtbz1yb3cuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxudWxsOwogICAgICB2YXIgZW50cnk9e2F0dGVudGlvbjpyb3cuYXR0ZW50aW9uLGRlbHRhOnJvdy5kZWx0YV8yNGgsdmVsb2NpdHk6cm93LnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tRW1vLGRvbWluYW50X25hcnJhdGl2ZTpyb3cuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgICBMSVZFW3Jvdy5uYW1lXT1lbnRyeTsKICAgICAgaWYoIVNEW3Jvdy5uYW1lXSkgU0Rbcm93Lm5hbWVdPU9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7CiAgICAgIE9iamVjdC5hc3NpZ24oU0Rbcm93Lm5hbWVdLGVudHJ5KTsKICAgIH0pOwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgYnVpbGRXSVJTaWduYWxzKCk7CiAgICBidWlsZExvY2FsSW5zaWdodCgpOwogICAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7CiAgICBzZXRUaW1lb3V0KHJlbmRlck1vbWVudHVtLCA1MDApOwogICAgaWYoU0VMJiZMSVZFW1NFTF0mJmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKSkgcmVuZGVyUGFuZWwoU0VMKTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbQVBJXScsZS5tZXNzYWdlKTt9Cn0KCmZ1bmN0aW9uIGJ1aWxkTG9jYWxJbnNpZ2h0KCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CgogIC8vIEFnZ3JlZ2F0ZSB0b3AgbmFycmF0aXZlcyBhY3Jvc3MgYWxsIHN0YXRlcwogIHZhciBuYXI9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBpZighbmFyW24ubmFtZV0pIG5hcltuLm5hbWVdPXt1cDowLGRvd246MCxmbGF0OjAsdG90YWw6MH07CiAgICAgIG5hcltuLm5hbWVdW24uZGlyXT0obmFyW24ubmFtZV1bbi5kaXJdfHwwKStuLnZhbDsKICAgICAgbmFyW24ubmFtZV0udG90YWw9KG5hcltuLm5hbWVdLnRvdGFsfHwwKStuLnZhbDsKICAgIH0pOwogIH0pOwoKICAvLyBUb3AgcmlzaW5nIGFuZCBmYWxsaW5nIChleGNsdWRlIHRpZXMgd2hlcmUgc2FtZSBuYW1lIHJpc2VzIGFuZCBmYWxscykKICB2YXIgcmlzaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0udXA+a3ZbMV0uZG93bjt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS51cC1hWzFdLnVwO30pLnNsaWNlKDAsMyk7CiAgdmFyIGZhbGxpbmc9T2JqZWN0LmVudHJpZXMobmFyKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5kb3duPmt2WzFdLnVwO30pCiAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLmRvd24tYVsxXS5kb3duO30pLnNsaWNlKDAsMik7CiAgdmFyIHRvcDM9T2JqZWN0LmVudHJpZXMobmFyKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udG90YWwtYVsxXS50b3RhbDt9KS5zbGljZSgwLDMpOwoKICAvLyBIb3R0ZXN0IHN0YXRlCiAgdmFyIGhvdHRlc3Q9ZW50cmllcy5zbGljZSgpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIHZhciBob3R0ZXN0RW1vPWhvdHRlc3Q/KExJVkVbaG90dGVzdFswXV0mJkxJVkVbaG90dGVzdFswXV0uZG9taW5hbnRfZW1vdGlvbil8fCcnOicnIDsKCiAgLy8gQnVpbGQgaW5zaWdodCB0ZXh0IOKAlCBtb3JlIGFuYWx5dGljYWwsIGNvbnRleHQtYXdhcmUKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwogIGlmKCFlbCkgcmV0dXJuOwoKICB2YXIgbGluZXM9W107CiAgaWYocmlzaW5nLmxlbmd0aCYmZmFsbGluZy5sZW5ndGgmJnJpc2luZ1swXVswXSE9PWZhbGxpbmdbMF1bMF0pewogICAgbGluZXMucHVzaCgnPGVtPicrcmlzaW5nWzBdWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3Jpc2luZ1swXVswXS5zbGljZSgxKSsnPC9lbT4gaXMgdGhlIGRvbWluYW50IHNpZ25hbCBhY3Jvc3MgSW5kaWEgdG9kYXknKTsKICAgIGlmKGZhbGxpbmdbMF0pIGxpbmVzLnB1c2goJyBhcyA8ZW0+JytmYWxsaW5nWzBdWzBdKyc8L2VtPiBmYWRlcyBmcm9tIG5hdGlvbmFsIGZvY3VzJyk7CiAgICBpZihob3R0ZXN0KSBsaW5lcy5wdXNoKCcuIDxzdHJvbmcgc3R5bGU9ImNvbG9yOnZhcigtLWluaykiPicraG90dGVzdFswXSsnPC9zdHJvbmc+IGlzIHRoZSBtb3N0IGFjdGl2ZSBzdGF0ZScrCiAgICAgIChob3R0ZXN0RW1vPycgd2l0aCAnK2hvdHRlc3RFbW8rJyBhcyB0aGUgcHJpbWFyeSBzaWduYWwgdG9uZSc6JycpKTsKICAgIGlmKHJpc2luZ1sxXSkgbGluZXMucHVzaCgnLiBTZWNvbmRhcnkgc3VyZ2U6IDxlbT4nK3Jpc2luZ1sxXVswXSsnPC9lbT4nKTsKICB9IGVsc2UgaWYocmlzaW5nLmxlbmd0aCl7CiAgICBsaW5lcy5wdXNoKCdTaWduYWxzIGFyZSBjb25jZW50cmF0ZWQgYXJvdW5kIDxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+Jyk7CiAgICBpZihob3R0ZXN0KSBsaW5lcy5wdXNoKCcuIDxzdHJvbmcgc3R5bGU9ImNvbG9yOnZhcigtLWluaykiPicraG90dGVzdFswXSsnPC9zdHJvbmc+IGxlYWRzIG5hdGlvbmFsIGF0dGVudGlvbicpOwogICAgaWYocmlzaW5nWzFdKSBsaW5lcy5wdXNoKCcgYWxvbmdzaWRlIDxlbT4nK3Jpc2luZ1sxXVswXSsnPC9lbT4nKTsKICB9IGVsc2UgaWYodG9wMy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnTmF0aW9uYWwgc2lnbmFscyBhcmUgZGlzcGVyc2VkLiBUb3AgbmFycmF0aXZlczogJyt0b3AzLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxlbT4nK25bMF0rJzwvZW0+Jzt9KS5qb2luKCcsICcpKTsKICB9CgogIGlmKGxpbmVzLmxlbmd0aCl7CiAgICBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrbGluZXMuam9pbignJykrJy48L2Rpdj4nOwogIH0KCiAgLy8gVGFncwogIGlmKHRFbCl7CiAgICB2YXIgdGFncz1bXTsKICAgIGZhbGxpbmcuc2xpY2UoMCwxKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICB0YWdzLnB1c2goJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSg1OSwxODQsMjE2LDAuMyk7Y29sb3I6IzNiYjhkOCI+4oaTICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIHJpc2luZy5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICB0YWdzLnB1c2goJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtjb2xvcjojZTA1YTI4Ij7ihpEgJytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKTsKICAgIH0pOwogICAgaWYodGFncy5sZW5ndGgpIHRFbC5pbm5lckhUTUw9dGFncy5qb2luKCcnKTsKICB9CgogIGlmKG1ldGFFbCl7CiAgICB2YXIgc3RhdGVDb3VudD1PYmplY3QudmFsdWVzKExJVkUpLmZpbHRlcihmdW5jdGlvbihzKXtyZXR1cm4gcy5hdHRlbnRpb24+Mjt9KS5sZW5ndGg7CiAgICBtZXRhRWwudGV4dENvbnRlbnQ9J09ic2VydmluZyAnK3N0YXRlQ291bnQrJyBhY3RpdmUgc3RhdGVzIMK3IHVwZGF0ZWQgJytuZXcgRGF0ZSgpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nLHtob3VyOicyLWRpZ2l0JyxtaW51dGU6JzItZGlnaXQnfSk7CiAgfQp9CgpmdW5jdGlvbiB1cGRhdGVBbGxTdHJpcHMoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKICAvLyBNZXJnZSBTRCBuYXJyYXRpdmUgZGF0YSBpbnRvIGVudHJpZXMKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgaWYoU0Rba3ZbMF1dJiZTRFtrdlswXV0ubmFycmF0aXZlcykga3ZbMV0ubmFycmF0aXZlcz1TRFtrdlswXV0ubmFycmF0aXZlczsKICAgIGlmKFNEW2t2WzBdXSYmU0Rba3ZbMF1dLnNvdXJjZV9jb3VudCkga3ZbMV0uc291cmNlX2NvdW50PVNEW2t2WzBdXS5zb3VyY2VfY291bnQ7CiAgICBpZihTRFtrdlswXV0mJlNEW2t2WzBdXS5jb25maWRlbmNlKSBrdlsxXS5jb25maWRlbmNlPVNEW2t2WzBdXS5jb25maWRlbmNlOwogIH0pOwoKICAvLyBTbWFydGVyIHJhbmtpbmc6IHdlaWdodGVkIHNjb3JlID0gYXR0ZW50aW9uICsgdmVsb2NpdHkgYm9udXMgKyBzb3VyY2UgZGl2ZXJzaXR5IGJvbnVzCiAgLy8gQnJlYWtzIHRpZXMgYnkgcHJpb3JpdGl6aW5nIHN0YXRlcyB3aXRoIGRpdmVyc2Ugc291cmNlcyAobm90IGp1c3Qgc2lnbmFsIHZvbHVtZSkKICBmdW5jdGlvbiBzbWFydFNjb3JlKGt2KXsKICAgIHZhciBkPWt2WzFdOwogICAgdmFyIGF0dD1kLmF0dGVudGlvbnx8MDsKICAgIHZhciB2ZWw9KGQudmVsb2NpdHl8fDApKjE1OyAvLyBtb21lbnR1bSBib251cwogICAgdmFyIHNyYz1NYXRoLm1pbigoZC5zb3VyY2VfY291bnR8fDEpLDUpKjI7IC8vIHNvdXJjZSBkaXZlcnNpdHkgYm9udXMgKG1heCA1IHNvdXJjZXMpCiAgICB2YXIgY29uZj17J0hJR0gnOjMsJ01FRElVTSc6MSwnTE9XJzotMn1bZC5jb25maWRlbmNlfHwnTE9XJ118fDA7CiAgICByZXR1cm4gYXR0K3ZlbCtzcmMrY29uZjsKICB9CgogIHZhciBzY29yZWQ9ZW50cmllcy5tYXAoZnVuY3Rpb24oa3Ype3JldHVybntuYW1lOmt2WzBdLGQ6a3ZbMV0sc2NvcmU6c21hcnRTY29yZShrdil9O30pOwogIHNjb3JlZC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuc2NvcmUtYS5zY29yZTt9KTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PXNjb3JlZFswXTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsIGhvdHRlc3QubmFtZSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCAnQXR0ZW50aW9uICcrTWF0aC5yb3VuZChob3R0ZXN0LmQuYXR0ZW50aW9ufHwwKSsoaG90dGVzdC5kLnNvdXJjZV9jb3VudD4yPycgwrcgJytob3R0ZXN0LmQuc291cmNlX2NvdW50Kycgc291cmNlcyc6JycpKTsKICAoZnVuY3Rpb24oKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2MtaG90dGVzdC10aXAnKTtpZighdCkgcmV0dXJuOwogIHZhciBuYXJzPShob3R0ZXN0LmQubmFycmF0aXZlc3x8W10pLnNsaWNlKDAsMyk7CiAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+V2h5ICcraG90dGVzdC5uYW1lKyc/PC9kaXY+JytuYXJzLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9InNjLXRpcC1yb3ciPsK3ICcrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsobi5kaXI9PT0ndXAnPycgPHNwYW4gc3R5bGU9ImNvbG9yOiNlMDVhMjgiPuKGkTwvc3Bhbj4nOicnKSsnPC9kaXY+Jzt9KS5qb2luKCcnKTt9KSgpOwoKICAvLyBQZWFrIGFuZ2VyIOKAlCBoaWdoZXN0IGF0dGVudGlvbiBhbW9uZyBhbmdlciBzdGF0ZXMsIHdpdGggc291cmNlIGRpdmVyc2l0eSB0aWVicmVhawogIHZhciBhbmdlclN0YXRlcz1zY29yZWQuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzLmQuZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcicmJihzLmQuYXR0ZW50aW9ufHwwKT4zO30pOwogIGlmKGFuZ2VyU3RhdGVzLmxlbmd0aCl7CiAgICB2YXIgdG9wQW5nZXI9YW5nZXJTdGF0ZXNbMF07CiAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLCB0b3BBbmdlci5uYW1lKTsKICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsICh0b3BBbmdlci5kLmRvbWluYW50X25hcnJhdGl2ZXx8J3NpZ25hbHMnKSk7CiAgICAoZnVuY3Rpb24oKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2MtYW5nZXItdGlwJyk7aWYoIXQpIHJldHVybjsKICAgIHZhciBuYXJzPSh0b3BBbmdlci5kLm5hcnJhdGl2ZXN8fFtdKS5zbGljZSgwLDMpOwogICAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+QW5nZXIgaW4gJyt0b3BBbmdlci5uYW1lKyc8L2Rpdj4nK25hcnMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ic2MtdGlwLXJvdyI+wrcgJytuLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpKyc8L2Rpdj4nO30pLmpvaW4oJycpO30pKCk7CiAgfQoKICAvLyBGYXN0ZXN0IHJpc2luZyDigJQgdmVsb2NpdHkgd2VpZ2h0ZWQgYnkgc291cmNlIGNvdW50IChsb2NhbCBwcm90ZXN0IHZzIGludGVybmF0aW9uYWwgY292ZXJhZ2UpCiAgdmFyIHJpc2luZz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLnZlbG9jaXR5fHwwKT4wO30pCiAgICAubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm57bmFtZTprdlswXSxkOmt2WzFdLAogICAgICB2ZWxTY29yZTooa3ZbMV0udmVsb2NpdHl8fDApKigoa3ZbMV0uc291cmNlX2NvdW50fHwxKT4yPzEuNDoxLjApfTt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi52ZWxTY29yZS1hLnZlbFNjb3JlO30pWzBdOwogIGlmKHJpc2luZyl7CiAgICBzZXRUZXh0KCdzYy1yaXNpbmctdmFsJywgcmlzaW5nLm5hbWUpOwogICAgc2V0VGV4dCgnc2MtcmlzaW5nLXN1YicsIChyaXNpbmcuZC5kb21pbmFudF9uYXJyYXRpdmV8fCdzaWduYWwnKSsocmlzaW5nLmQuc291cmNlX2NvdW50PjI/JyDCtyBtdWx0aS1zb3VyY2UnOicnKSk7CiAgICAoZnVuY3Rpb24oKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2MtcmlzaW5nLXRpcCcpO2lmKCF0KSByZXR1cm47CiAgICB2YXIgbmFycz0ocmlzaW5nLmQubmFycmF0aXZlc3x8W10pLmZpbHRlcihmdW5jdGlvbihuKXtyZXR1cm4gbi5kaXI9PT0ndXAnO30pLnNsaWNlKDAsMyk7CiAgICB0LmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2MtdGlwLXRpdGxlIj5XaHkgJytyaXNpbmcubmFtZSsnIGlzIHJpc2luZzwvZGl2PicrKG5hcnMubGVuZ3RoP25hcnMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ic2MtdGlwLXJvdyI+wrcgJytuLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpKycgPHNwYW4gc3R5bGU9ImNvbG9yOiNlMDVhMjgiPuKGkTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpOic8ZGl2IGNsYXNzPSJzYy10aXAtcm93Ij7CtyBTaWduYWwgdm9sdW1lIGluY3JlYXNpbmc8L2Rpdj4nKTt9KSgpOwogIH0KCiAgLy8gVG9wIG5hcnJhdGl2ZSDigJQgbW9zdCBzaWduYWxzIGFjcm9zcyBhbGwgc3RhdGVzCiAgdmFyIG5hckNvdW50cz17fTsKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgKGt2WzFdLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBuYXJDb3VudHNbbi5uYW1lXT0obmFyQ291bnRzW24ubmFtZV18fDApK24udmFsOwogICAgfSk7CiAgfSk7CiAgdmFyIHRvcE5hcj1PYmplY3QuZW50cmllcyhuYXJDb3VudHMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pWzBdOwogIGlmKHRvcE5hcil7CiAgICBzZXRUZXh0KCdzYy1uYXItdmFsJywgdG9wTmFyWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3RvcE5hclswXS5zbGljZSgxKSk7CiAgICAvLyBGaW5kIHdoaWNoIHN0YXRlcyBkcml2ZSBpdAogICAgdmFyIG5hclN0YXRlcz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7CiAgICAgIHJldHVybihrdlsxXS5uYXJyYXRpdmVzfHxbXSkuc29tZShmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lPT09dG9wTmFyWzBdJiZuLmRpcj09PSd1cCc7fSk7CiAgICB9KS5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihrdil7cmV0dXJuIGt2WzBdLnNwbGl0KCcgJylbMF07fSk7CiAgICBzZXRUZXh0KCdzYy1uYXItc3ViJywgbmFyU3RhdGVzLmxlbmd0aD9uYXJTdGF0ZXMuam9pbignLCAnKTonbmF0aW9uYWxseScpOwogICAgKGZ1bmN0aW9uKCl7dmFyIHQ9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NjLW5hci10aXAnKTtpZighdCkgcmV0dXJuOwogICAgdmFyIHRvcFN0YXRlcz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLm5hcnJhdGl2ZXN8fFtdKS5zb21lKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU9PT10b3BOYXJbMF07fSk7fSkKICAgICAgLnNvcnQoZnVuY3Rpb24oYSxiKXt2YXIgYXY9KGFbMV0ubmFycmF0aXZlc3x8W10pLmZpbmQoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZT09PXRvcE5hclswXTt9KTt2YXIgYnY9KGJbMV0ubmFycmF0aXZlc3x8W10pLmZpbmQoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZT09PXRvcE5hclswXTt9KTtyZXR1cm4oKGJ2JiZidi52YWwpfHwwKS0oKGF2JiZhdi52YWwpfHwwKTt9KS5zbGljZSgwLDMpOwogICAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0b3BOYXJbMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdG9wTmFyWzBdLnNsaWNlKDEpKycg4oCUIGFjdGl2ZSBpbjwvZGl2PicrdG9wU3RhdGVzLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJzYy10aXAtcm93Ij7CtyAnK2t2WzBdKyc8L2Rpdj4nO30pLmpvaW4oJycpO30pKCk7CiAgfQoKICAvLyBGYXN0ZXN0IGNvb2xpbmcg4oCUIHVzZSBzbWFydCBzY29yZSB0b28KICB2YXIgY29vbGluZzI9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS52ZWxvY2l0eXx8MCk8LTAuMDE7fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGFbMV0udmVsb2NpdHl8fDApLShiWzFdLnZlbG9jaXR5fHwwKTt9KVswXTsKICBpZihjb29saW5nMil7CiAgICBzZXRUZXh0KCdzYy1jb29sLXZhbCcsIGNvb2xpbmcyWzBdKTsKICAgIHZhciBjTmFyPWNvb2xpbmcyWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8KGNvb2xpbmcyWzFdLm5hcnJhdGl2ZXMmJmNvb2xpbmcyWzFdLm5hcnJhdGl2ZXNbMF0mJmNvb2xpbmcyWzFdLm5hcnJhdGl2ZXNbMF0ubmFtZSl8fCcnOwogICAgc2V0VGV4dCgnc2MtY29vbC1zdWInLCBjTmFyP2NOYXIrJyDCtyByZXRyZWF0aW5nJzonU2lnbmFsIHJldHJlYXRpbmcnKTsKICB9CgogIC8vIFNpZ25hbCBjb3VudCDigJQgdXBkYXRlIGJvdGggdG9wYmFyIGFuZCBzdGF0cyBzdHJpcAogIHZhciB0b3RhbD1PYmplY3QudmFsdWVzKFNEKS5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcysodi5zaWduYWxfY291bnR8fDApO30sMCk7CiAgdmFyIGxjPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7CiAgaWYobGMpIGxjLnRleHRDb250ZW50PXRvdGFsLnRvTG9jYWxlU3RyaW5nKCdlbi1JTicpOwogIC8vIFN0YXRzIHN0cmlwIHNpZ25hbCBjb3VudAogIHZhciBzY1NpZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Mtc2lnbmFscy12YWwnKTsKICBpZihzY1NpZykgc2NTaWcudGV4dENvbnRlbnQ9dG90YWwudG9Mb2NhbGVTdHJpbmcoJ2VuLUlOJyk7CiAgc2V0VGV4dCgnc2Mtc2lnbmFscy1zdWInLCdhY3Jvc3MgJytPYmplY3Qua2V5cyhMSVZFKS5maWx0ZXIoZnVuY3Rpb24oayl7cmV0dXJuKExJVkVba10uYXR0ZW50aW9ufHwwKT4yO30pLmxlbmd0aCsnIGFjdGl2ZSBzdGF0ZXMnKTsKfQoKCmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKGVsKWVsLnRleHRDb250ZW50PXZhbDt9CgpmdW5jdGlvbiB1cGRhdGVTdHJpcE5hcnJhdGl2ZSgpe3VwZGF0ZUFsbFN0cmlwcygpO30KZnVuY3Rpb24gdXBkYXRlU3RyaXBBbmdlcigpe30KCmZ1bmN0aW9uIHNlbGVjdEhvdHRlc3QoKXsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKFNEKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApIHNlbGVjdF8odG9wWzBdKTsKfQphc3luYyBmdW5jdGlvbiBmZXRjaEluc2lnaHRzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvaW5zaWdodHMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIHZhciBzaWc9ZC5zaWduYXR1cmU7CiAgICBpZihzaWcpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQudGFncyl0RWwuaW5uZXJIVE1MPWQudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgIH0KICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICBpZihyRWwmJmQucmlzaW5nJiZkLnJpc2luZy5sZW5ndGgpckVsLmlubmVySFRNTD1kLnJpc2luZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuIHJlbmRlck5hckNhcmQobiwncmlzaW5nJyk7fSkuam9pbignJyk7OwogICAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICAgIGlmKGZFbCYmZC5mYWxsaW5nJiZkLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gcmVuZGVyTmFyQ2FyZChuLCdkZWNsaW5pbmcnKTt9KS5qb2luKCcnKTs7CiAgICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgICBpZihnRWwmJmQucmVnaW9uYWwmJmQucmVnaW9uYWwubGVuZ3RoKWdFbC5pbm5lckhUTUw9ZC5yZWdpb25hbC5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuIj48c3BhbiBjbGFzcz0ibmktbmFtZSI+JytyLnJlZ2lvbisnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrci5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj48ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrci5ob3R0ZXN0X3N0YXRlKycgwrcgJytyLnRvcF9uYXJyYXRpdmUrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoRnVsbFNuYXBzaG90KCl7CiAgLy8gTG9hZCBBTEwgc3RhdGUgZGF0YSBpbiBvbmUgcmVxdWVzdCBmb3IgaW5zdGFudCBmaXJzdC1sb2FkCiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvZnVsbC1zbmFwc2hvdCcpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC53YXJtaW5nX3VwfHwhZC5zdGF0ZXN8fCFkLnN0YXRlcy5sZW5ndGgpIHJldHVybiBmYWxzZTsKCiAgICAvLyBQb3B1bGF0ZSBTRCBhbmQgTElWRSBmcm9tIGZ1bGwgc25hcHNob3QKICAgIGQuc3RhdGVzLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAgIGlmKCFzLm5hbWUpIHJldHVybjsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8cy5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgICB2YXIgZW50cnk9T2JqZWN0LmFzc2lnbih7fSxzLHtlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRlbHRhOnMuZGVsdGFfMjRofHwwfSk7CiAgICAgIFNEW3MubmFtZV09ZW50cnk7CiAgICAgIExJVkVbcy5uYW1lXT17YXR0ZW50aW9uOnMuYXR0ZW50aW9uLGRlbHRhOnMuZGVsdGFfMjRofHwwLHZlbG9jaXR5OnMudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOnMuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgfSk7CgogICAgLy8gVXBkYXRlIHNpZ25hbHMgY291bnQKICAgIGlmKGQuc25hcHNob3QmJmQuc25hcHNob3QudG90YWxfc2lnbmFscyl7CiAgICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJyxkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKSk7CiAgICB9CgogICAgLy8gVXBkYXRlIGluc2lnaHRzIGZyb20gY2FjaGVkIGRhdGEKICAgIGlmKGQuaW5zaWdodHMmJmQuaW5zaWdodHMuc2lnbmF0dXJlKXsKICAgICAgdmFyIHNpZz1kLmluc2lnaHRzLnNpZ25hdHVyZTsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLmluc2lnaHRzLnRhZ3MpdEVsLmlubmVySFRNTD1kLmluc2lnaHRzLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICAgIGlmKHJFbCYmZC5pbnNpZ2h0cy5yaXNpbmcmJmQuaW5zaWdodHMucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMucmlzaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgICBpZihmRWwmJmQuaW5zaWdodHMuZmFsbGluZyYmZC5pbnNpZ2h0cy5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIH0KCiAgICAvLyBSZW5kZXIgbWFwIGNvbG9ycyBhbmQgc3RyaXBzCiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICAvLyBMb2FkIGluc2lnaHRzIHRvbwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgLy8gVXNlIGNhY2hlZCBuYXJyYXRpdmUgaW5zaWdodCBpZiBhdmFpbGFibGUKICAgIGlmKGQubmFycmF0aXZlX2luc2lnaHQmJmQubmFycmF0aXZlX2luc2lnaHQudGV4dCl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICAgICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQrJzwvZGl2Pic7CiAgICAgIGlmKHRFbCYmZC5uYXJyYXRpdmVfaW5zaWdodC50b3BfbmFycmF0aXZlcyl7CiAgICAgIH0KICAgIH0KICAgIHJldHVybiB0cnVlOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW2Z1bGwtc25hcHNob3RdJyxlLm1lc3NhZ2UpOwogICAgcmV0dXJuIGZhbHNlOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCl7CiAgdHJ5ewogICAgLy8gVHJ5IGNhY2hlZCB2ZXJzaW9uIGZyb20gZnVsbC1zbmFwc2hvdCBmaXJzdCAoYWxyZWFkeSBsb2FkZWQpCiAgICAvLyBUaGVuIGNhbGwgZGVkaWNhdGVkIGVuZHBvaW50IGZvciBmcmVzaCBBSSBhbmFseXNpcwogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvbmFycmF0aXZlLWluc2lnaHQnKTsKICAgIGlmKCFyLm9rKSByZXR1cm47CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFkLnRleHQpIHJldHVybjsKCiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKCiAgICBpZihlbCkgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2QudGV4dCsnPC9kaXY+JzsKCiAgICAvLyBUYWdzIGZyb20gdG9wIG5hcnJhdGl2ZXMKICAgIGlmKHRFbCYmZC50b3BfbmFycmF0aXZlcyYmZC50b3BfbmFycmF0aXZlcy5sZW5ndGgpewogICAgICB0RWwuaW5uZXJIVE1MPWQudG9wX25hcnJhdGl2ZXMubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgICAgdmFyIGNvbD1pPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC42KSc7CiAgICAgICAgdmFyIGFycm93PWk9PT0wPyfihpEgJzonwrcgJzsKICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKTtjb2xvcjonK2NvbCsnIj4nK2Fycm93K24uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5zbGljZSgxKSsnPC9zcGFuPic7CiAgICAgIH0pLmpvaW4oJycpOwogICAgfQoKICAgIGlmKG1ldGFFbCl7CiAgICAgIHZhciB0PW5ldyBEYXRlKGQuYXNfb2YpOwogICAgICBtZXRhRWwudGV4dENvbnRlbnQ9J1NpZ25hbCBhbmFseXNpcyDCtyAnK3QudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicse2hvdXI6JzItZGlnaXQnLG1pbnV0ZTonMi1kaWdpdCd9KSsoZC5mYWxsYmFjaz8nIMK3IHBhdHRlcm4tYmFzZWQnOicgwrcgQUkgc3ludGhlc2l6ZWQnKTsKICAgIH0KICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbbmFycmF0aXZlXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIHN0YXJ0UG9sbGluZygpewogIGF3YWl0IFByb21pc2UuYWxsKFtmZXRjaEFsbFN0YXRlcygpLGZldGNoU25hcCgpXSk7CiAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZSk7fSk7CiAgdmFyIG49MDsKICB2YXIgdD1zZXRJbnRlcnZhbChhc3luYyBmdW5jdGlvbigpewogICAgbisrO2F3YWl0IGZldGNoQWxsU3RhdGVzKCk7YXdhaXQgZmV0Y2hTbmFwKCk7CiAgICBpZihTRUwpIHJlbmRlclBhbmVsKFNFTCk7CiAgICBpZihuPj0xMil7Y2xlYXJJbnRlcnZhbCh0KTtzZXRJbnRlcnZhbChhc3luYyBmdW5jdGlvbigpe2F3YWl0IGZldGNoQWxsU3RhdGVzKCk7YXdhaXQgZmV0Y2hTbmFwKCk7aWYoU0VMKXJlbmRlclBhbmVsKFNFTCk7fSwxMjAwMDApOwogICAgICBzZXRJbnRlcnZhbChmZXRjaEluc2lnaHRzLDM2MDAwMDApO30KICB9LDE1MDAwKTsKfQoKLy8gTkFSUkFUSVZFIERBVEEKdmFyIFNISUZUUz17CiAgJzNtJzpbCiAgICB7ZmFkaW5nOidJbmZsYXRpb24nLGZhZGluZ05vdGU6J2Vhc2luZyBuYXRpb25hbGx5JyxyaXNpbmc6J0JvcmRlciBzZWN1cml0eScscmlzaW5nTm90ZToncG9zdC1pbmNpZGVudCBzdXJnZSd9LAogICAge2ZhZGluZzonRWxlY3Rpb24gcmhldG9yaWMnLGZhZGluZ05vdGU6J3Bvc3QtY3ljbGUgZmFkZScscmlzaW5nOidHb3Zlcm5hbmNlIGFjY291bnRhYmlsaXR5JyxyaXNpbmdOb3RlOidzdGVhZHkgcmlzZSd9LAogICAge2ZhZGluZzonRmFybWVyIHByb3Rlc3RzJyxmYWRpbmdOb3RlOidtb21lbnR1bSBsb3N0JyxyaXNpbmc6J1VuZW1wbG95bWVudCBhbnhpZXR5JyxyaXNpbmdOb3RlOid5b3V0aCBzaWduYWwgc3VyZ2UnfSwKICBdLAogICc2bSc6WwogICAge2ZhZGluZzonQ2FzdGUgbW9iaWxpc2F0aW9uJyxmYWRpbmdOb3RlOidwcmUtZWxlY3Rpb24gcGVhaycscmlzaW5nOidDb3JydXB0aW9uIGFjY291bnRhYmlsaXR5JyxyaXNpbmdOb3RlOidwb3N0LWN5Y2xlIHB1c2gnfSwKICAgIHtmYWRpbmc6J1JlbGlnaW91cyBuYXRpb25hbGlzbScsZmFkaW5nTm90ZToncGxhdGVhdSBwaGFzZScscmlzaW5nOidFY29ub21pYyBhbnhpZXR5JyxyaXNpbmdOb3RlOidjb3N0LW9mLWxpdmluZyd9LAogICAge2ZhZGluZzonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLGZhZGluZ05vdGU6J3JpYmJvbi1jdXR0aW5nIGRvbmUnLHJpc2luZzonTGF3ICYgb3JkZXInLHJpc2luZ05vdGU6J2NyaW1lIG5hcnJhdGl2ZSByaXNlJ30sCiAgXSwKICAnMXknOlsKICAgIHtmYWRpbmc6J1BhbmRlbWljIHJlY292ZXJ5JyxmYWRpbmdOb3RlOidmYWRlZCBlYXJseSB5ZWFyJyxyaXNpbmc6J0luZmxhdGlvbicscmlzaW5nTm90ZTonZG9taW5hdGVkIG1pZC15ZWFyJ30sCiAgICB7ZmFkaW5nOidSZWdpb25hbCBpZGVudGl0eScsZmFkaW5nTm90ZTonbGFuZ3VhZ2UtbGVkIHBlYWsnLHJpc2luZzonU2VjdXJpdHkgJiBib3JkZXJzJyxyaXNpbmdOb3RlOidnZW9wb2xpdGljYWwgZXNjYWxhdGlvbid9LAogICAge2ZhZGluZzonR292ZXJuYW5jZSBvcHRpbWlzbScsZmFkaW5nTm90ZToncG9saWN5IGhvbmV5bW9vbiBlbmQnLHJpc2luZzonQ29ycnVwdGlvbiAmIHNjYW1zJyxyaXNpbmdOb3RlOidhY2NvdW50YWJpbGl0eSBjeWNsZSd9LAogIF0sCn07CnZhciBSRUdfU0hJRlRTPVsKICB7c3RhdGU6J1RhbWlsIE5hZHUnLGZyb206J1JlZ2lvbmFsIGlkZW50aXR5Jyx0bzonRmVkZXJhbCByZXNvdXJjZSBkaXNwdXRlcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0JpaGFyJyxmcm9tOidFbGVjdGlvbiByaGV0b3JpYycsdG86J1VuZW1wbG95bWVudCAmIGV4YW0gc2NhbXMnLHRpbWU6JzYgd2tzJ30sCiAge3N0YXRlOidXZXN0IEJlbmdhbCcsZnJvbTonQnlwb2xsIHBvbGl0aWNzJyx0bzonTGF3ICYgb3JkZXIgwrcgQm9yZGVyJyx0aW1lOic0IHdrcyd9LAogIHtzdGF0ZTonUmFqYXN0aGFuJyxmcm9tOidGYXJtZXIgcHJvdGVzdHMnLHRvOidIZWF0IHdhdmUgwrcgRW52aXJvbm1lbnQnLHRpbWU6JzIgd2tzJ30sCiAge3N0YXRlOidLYXJuYXRha2EnLGZyb206J01pbmluZyBjb250cm92ZXJzeScsdG86J0xhbmd1YWdlIHNpZ25hZ2UgcG9saXRpY3MnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidEZWxoaScsZnJvbTonTWV0cm8gaW5mcmFzdHJ1Y3R1cmUnLHRvOidBaXIgcXVhbGl0eSBjcmlzaXMnLHRpbWU6JzEwIGRheXMnfSwKICB7c3RhdGU6J01hbmlwdXInLGZyb206J0dvdmVybmFuY2UgJiBjYWJpbmV0Jyx0bzonRXRobmljIHRlbnNpb25zIMK3IEFGU1BBJyx0aW1lOic1IHdrcyd9LAogIHtzdGF0ZTonUHVuamFiJyxmcm9tOidQb3dlciBjcmlzaXMnLHRvOidCb3JkZXIgc2VjdXJpdHkgwrcgRHJvbmVzJyx0aW1lOiczIHdrcyd9LApdOwp2YXIgTU9DS19SPVsKICB7bmFtZTonQm9yZGVyIHNlY3VyaXR5JyxzdGF0ZXM6J0omSyDCtyBQdW5qYWIgwrcgUmFqYXN0aGFuJyxwY3Q6Jys0MSUnfSwKICB7bmFtZTonVW5lbXBsb3ltZW50JyxzdGF0ZXM6J0JpaGFyIMK3IFVQIMK3IEpoYXJraGFuZCcscGN0OicrMjglJ30sCiAge25hbWU6J0xhbmd1YWdlIHBvbGl0aWNzJyxzdGF0ZXM6J1ROIMK3IEthcm5hdGFrYSDCtyBNSCcscGN0OicrMjIlJ30sCiAge25hbWU6J0Vudmlyb25tZW50YWwgY3Jpc2lzJyxzdGF0ZXM6J0RlbGhpIMK3IFJhamFzdGhhbiDCtyBBUCcscGN0OicrMTklJ30sCiAge25hbWU6J0V0aG5pYyB0ZW5zaW9ucycsc3RhdGVzOidNYW5pcHVyIMK3IEFzc2FtIMK3IFdCJyxwY3Q6JysxNyUnfSwKXTsKdmFyIE1PQ0tfRj1bCiAge25hbWU6J0VsZWN0aW9uIHJoZXRvcmljJyxzdGF0ZXM6J05hdGlvbmFsIHBvc3QtY3ljbGUnLHBjdDonLTM4JSd9LAogIHtuYW1lOidJbmZsYXRpb24gcHJlc3N1cmUnLHN0YXRlczonRWFzaW5nIG5hdGlvbmFsbHknLHBjdDonLTI0JSd9LAogIHtuYW1lOidGYXJtZXIgcHJvdGVzdHMnLHN0YXRlczonTW9tZW50dW0gbG9zdCcscGN0OictMTklJ30sCiAge25hbWU6J0luZnJhc3RydWN0dXJlIHByaWRlJyxzdGF0ZXM6J1JpYmJvbi1jdXR0aW5nIGRvbmUnLHBjdDonLTE0JSd9LAogIHtuYW1lOidSZWxpZ2lvdXMgZmVzdGl2YWxzJyxzdGF0ZXM6J1Bvc3Qtc2Vhc29uIGZhZGUnLHBjdDonLTExJSd9LApdOwoKZnVuY3Rpb24gcmVuZGVyU3RyaXAocGVyaW9kKXsKICB2YXIgZGF0YT1TSElGVFNbcGVyaW9kXXx8U0hJRlRTWyczbSddOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2hpZnQtbGlzdCcpOwogIGlmKCFlbCkgcmV0dXJuOwogIGVsLmlubmVySFRNTD1kYXRhLm1hcChmdW5jdGlvbihzKXsKICAgIHJldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O292ZXJmbG93OmhpZGRlbjsiPicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo2cHggMTBweDtib3JkZXItcmlnaHQ6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWxsKTttYXJnaW4tYm90dG9tOjNweDsiPmZhZGluZzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuMjsiPicrcy5mYWRpbmcrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK3MuZmFkaW5nTm90ZSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJ3aWR0aDoyOHB4O2ZsZXgtc2hyaW5rOjA7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjQ1O2ZvbnQtc2l6ZToxM3B4OyI+4oaSPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjhweCAxMHB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tcmlzZSk7bWFyZ2luLWJvdHRvbTozcHg7Ij5yaXNpbmc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjI7Ij4nK3MucmlzaW5nKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytzLnJpc2luZ05vdGUrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odGFiKXsKICB0YWIuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0KXt0LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgdGFiLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogICAgcmVuZGVyU3RyaXAodGFiLmRhdGFzZXQucGVyaW9kfHwnM20nKTsKICB9KTsKfSk7CgpmdW5jdGlvbiByZW5kZXJNb21lbnR1bSgpewogIC8vIFJlYWQgZnJvbSBTRCAocG9wdWxhdGVkIGJ5IGZldGNoQWxsU3RhdGVzIGZyb20gbGl2ZSBBUEkpCiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgbmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpOwogIHZhciBmYWxsaW5nPXNvcnRlZC5zbGljZSgtNSkucmV2ZXJzZSgpOwogIHZhciBteD1yaXNpbmcubGVuZ3RoP3Jpc2luZ1swXVsxXToxMDA7CgogIC8vIFdyaXRlIHRvIHJpc2luZy1saXN0IChtYXRjaGVzIG5hci1yb3cgSFRNTCkKICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogIGlmKHJFbCYmcmlzaW5nLmxlbmd0aCl7CiAgICByRWwuaW5uZXJIVE1MPXJpc2luZy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gZGVjbGluaW5nLWxpc3QKICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogIGlmKGZFbCYmZmFsbGluZy5sZW5ndGgpewogICAgZkVsLmlubmVySFRNTD1mYWxsaW5nLm1hcChmdW5jdGlvbihuKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gcmVnaW9uYWwtbGlzdCDigJQgdG9wIHN0YXRlIHBlciByZWdpb24gZnJvbSBMSVZFCiAgdmFyIHJlZ2lvbnM9ewogICAgJ05vcnRoJzpbJ0RlbGhpJywnVXR0YXIgUHJhZGVzaCcsJ1B1bmphYicsJ0hhcnlhbmEnLCdIaW1hY2hhbCBQcmFkZXNoJywnVXR0YXJha2hhbmQnLCdKYW1tdSBhbmQgS2FzaG1pciddLAogICAgJ0Vhc3QnOlsnV2VzdCBCZW5nYWwnLCdCaWhhcicsJ0poYXJraGFuZCcsJ09kaXNoYSddLAogICAgJ1dlc3QnOlsnTWFoYXJhc2h0cmEnLCdHdWphcmF0JywnUmFqYXN0aGFuJywnR29hJ10sCiAgICAnU291dGgnOlsnVGFtaWwgTmFkdScsJ0thcm5hdGFrYScsJ0tlcmFsYScsJ0FuZGhyYSBQcmFkZXNoJywnVGVsYW5nYW5hJ10sCiAgICAnTkUnOlsnQXNzYW0nLCdNYW5pcHVyJywnTmFnYWxhbmQnLCdNaXpvcmFtJywnTWVnaGFsYXlhJywnVHJpcHVyYScsJ0FydW5hY2hhbCBQcmFkZXNoJywnU2lra2ltJ10sCiAgICAnQ2VudHJhbCc6WydNYWRoeWEgUHJhZGVzaCcsJ0NoaGF0dGlzZ2FyaCddLAogIH07CiAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogIGlmKGdFbCl7CiAgICB2YXIgcmVnSXRlbXM9T2JqZWN0LmVudHJpZXMocmVnaW9ucykubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgdmFyIHJlZ2lvbj1rdlswXSxzdGF0ZXM9a3ZbMV07CiAgICAgIHZhciB0b3A9c3RhdGVzLm1hcChmdW5jdGlvbihzKXtyZXR1cm4ge25hbWU6cyxhdHQ6KExJVkVbc10mJkxJVkVbc10uYXR0ZW50aW9uKXx8MH07fSkKICAgICAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiLmF0dC1hLmF0dDt9KVswXTsKICAgICAgaWYoIXRvcHx8IXRvcC5hdHQpIHJldHVybiBudWxsOwogICAgICB2YXIgbmFyPShMSVZFW3RvcC5uYW1lXSYmTElWRVt0b3AubmFtZV0uZG9taW5hbnRfbmFycmF0aXZlKXx8J+KAlCc7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzo4cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjEyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+JytyZWdpb24rJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3RvcC5hdHQudG9GaXhlZCgxKSsnPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+Jyt0b3AubmFtZSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHg7Ij4nK25hcisnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5maWx0ZXIoQm9vbGVhbikuam9pbignJyk7CiAgICBpZihyZWdJdGVtcykgZ0VsLmlubmVySFRNTD1yZWdJdGVtczsKICB9Cn0KCgovLyBTVEFURSBEQVRBCnZhciBTRD17fTsKCnZhciBMSVZFPXt9OwpmdW5jdGlvbiBub3JtYWxpemVFbW90aW9ucyhlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm57fTt2YXIgdmFscz1PYmplY3QudmFsdWVzKGUpLHRvdD12YWxzLnJlZHVjZShmdW5jdGlvbihzLHYpe3JldHVybiBzK3Y7fSwwKTtpZih0b3Q8PTApcmV0dXJue307aWYodG90PD0xLjAxKXt2YXIgb3V0PXt9O09iamVjdC5rZXlzKGUpLmZvckVhY2goZnVuY3Rpb24oayl7b3V0W2tdPU1hdGgucm91bmQoZVtrXSoxMDApO30pO3JldHVybiBvdXQ7fXJldHVybiBlO30KZnVuY3Rpb24gZG9taW5hbnRFbW90aW9uKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybiBudWxsO3ZhciBteD0wLGRvbT1udWxsO09iamVjdC5lbnRyaWVzKGUpLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdPm14KXtteD1rdlsxXTtkb209a3ZbMF07fX0pO3JldHVybiBkb207fQpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZighZWwpcmV0dXJuO2VsLnRleHRDb250ZW50PXZhbDtpZih2YWwmJnZhbCE9PSctJyl7ZWwuY2xhc3NMaXN0LnJlbW92ZSgnbG9hZGluZycpO319Cgp2YXIgREVGQVVMVD17CiAgYXR0ZW50aW9uOjAsZGVsdGE6MCx2ZWxvY2l0eTowLAogIGVtb3Rpb25zOnt9LGRvbWluYW50X2Vtb3Rpb246bnVsbCxkb21pbmFudF9uYXJyYXRpdmU6bnVsbCwKICBuYXJyYXRpdmVzOltdLHJpc2luZzpbXSxmYWxsaW5nOltdLAogIHN1bW1hcnk6JycsYXJ0aWNsZXM6W10sdGltZWxpbmU6W10sCiAgbmFycmF0aXZlSGlzdG9yeTpbXSxzaWduYWxfY291bnQ6MCwKfTsKCmZ1bmN0aW9uIGcobil7cmV0dXJuIFNEW25dfHxPYmplY3QuYXNzaWduKHt9LERFRkFVTFQpO30KCi8vIOKUgOKUgCBDT0xPUiBVVElMSVRJRVMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmZ1bmN0aW9uIGxlcnBDb2xvcihhLGIsdCl7CiAgLy8gTGluZWFyIGludGVycG9sYXRlIGJldHdlZW4gdHdvIGhleCBjb2xvcnMKICB2YXIgYXI9cGFyc2VJbnQoYS5zbGljZSgxLDMpLDE2KSxhZz1wYXJzZUludChhLnNsaWNlKDMsNSksMTYpLGFiPXBhcnNlSW50KGEuc2xpY2UoNSw3KSwxNik7CiAgdmFyIGJyPXBhcnNlSW50KGIuc2xpY2UoMSwzKSwxNiksYmc9cGFyc2VJbnQoYi5zbGljZSgzLDUpLDE2KSxiYj1wYXJzZUludChiLnNsaWNlKDUsNyksMTYpOwogIHZhciByPU1hdGgucm91bmQoYXIrKGJyLWFyKSp0KTsKICB2YXIgZz1NYXRoLnJvdW5kKGFnKyhiZy1hZykqdCk7CiAgdmFyIGJ2PU1hdGgucm91bmQoYWIrKGJiLWFiKSp0KTsKICByZXR1cm4gJyMnKygnMCcrci50b1N0cmluZygxNikpLnNsaWNlKC0yKSsoJzAnK2cudG9TdHJpbmcoMTYpKS5zbGljZSgtMikrKCcwJytidi50b1N0cmluZygxNikpLnNsaWNlKC0yKTsKfQoKZnVuY3Rpb24gY29sb3JTY2FsZShuLCBzdG9wcyl7CiAgLy8gbiA9IDAtMSwgc3RvcHMgPSBbW3BvcywnI2hleCddLC4uLl0KICBmb3IodmFyIGk9MDtpPHN0b3BzLmxlbmd0aC0xO2krKyl7CiAgICBpZihuPj1zdG9wc1tpXVswXSYmbjw9c3RvcHNbaSsxXVswXSl7CiAgICAgIHZhciB0PShuLXN0b3BzW2ldWzBdKS8oc3RvcHNbaSsxXVswXS1zdG9wc1tpXVswXSk7CiAgICAgIHJldHVybiBsZXJwQ29sb3Ioc3RvcHNbaV1bMV0sc3RvcHNbaSsxXVsxXSx0KTsKICAgIH0KICB9CiAgcmV0dXJuIHN0b3BzW3N0b3BzLmxlbmd0aC0xXVsxXTsKfQoKLy8gQXR0ZW50aW9uIGNvbG9yIOKAlCBzbW9vdGggZ3JhZGllbnQsIGFsd2F5cyBub3JtYWxpemVkIHRvIGFjdHVhbCBkYXRhIHJhbmdlCnZhciBfYU5vcm09e21uOjAsbXg6MSx0czowfTsKZnVuY3Rpb24gYUMocyl7CiAgdmFyIG5vdz1EYXRlLm5vdygpOwogIGlmKG5vdy1fYU5vcm0udHM+MzAwMCl7CiAgICB2YXIgc2M9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKGQpe3JldHVybiBkLmF0dGVudGlvbnx8MDt9KS5maWx0ZXIoZnVuY3Rpb24odil7cmV0dXJuIHY+MDt9KTsKICAgIGlmKHNjLmxlbmd0aCl7CiAgICAgIF9hTm9ybS5tbj1NYXRoLm1pbi5hcHBseShudWxsLHNjKTsKICAgICAgX2FOb3JtLm14PU1hdGgubWF4LmFwcGx5KG51bGwsc2MpfHwxOwogICAgfQogICAgX2FOb3JtLnRzPW5vdzsKICB9CiAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzLV9hTm9ybS5tbikvTWF0aC5tYXgoX2FOb3JtLm14LV9hTm9ybS5tbiwxKSkpOwogIHJldHVybiBjb2xvclNjYWxlKG4sWwogICAgWzAuMDAsJyMwYTE2MjgnXSwgIC8vIGRlZXAgbmF2eSDigJQgbWluaW1hbCBzaWduYWwKICAgIFswLjE1LCcjMGQzYTZlJ10sICAvLyBuYXZ5CiAgICBbMC4zMCwnIzBhNWY4YSddLCAgLy8gc3RlZWwgYmx1ZQogICAgWzAuNDUsJyMwZDhhN2EnXSwgIC8vIHRlYWwKICAgIFswLjU4LCcjMmE3YTRhJ10sICAvLyBzYWdlIGdyZWVuCiAgICBbMC43MCwnI2IwODAxMCddLCAgLy8gZ29sZAogICAgWzAuODAsJyNkMDYwMTAnXSwgIC8vIGFtYmVyCiAgICBbMC45MCwnI2NjMjgwOCddLCAgLy8gY3JpbXNvbgogICAgWzEuMDAsJyNmZjEwMjAnXSwgIC8vIHJlZCDigJQgcGVhayBzaWduYWwKICBdKTsKfQoKZnVuY3Rpb24gZUMoZSl7CiAgdmFyIG14PTAsZG9tPSdwcmlkZSc7CiAgZm9yKHZhciBrIGluIGUpe2lmKGVba10+bXgpe214PWVba107ZG9tPWs7fX0KICByZXR1cm4gKHthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfSlbZG9tXXx8JyMzM2FhY2MnOwp9CmZ1bmN0aW9uIG5vcm1WKHYpewogIC8vIE5vcm1hbGl6ZSB2ZWxvY2l0eSByZWdhcmRsZXNzIG9mIHNjYWxlCiAgLy8gT2xkIGRhdGE6IHZlbG9jaXR5IGlzIHJhdyBkZWx0YSAobGFyZ2UsIGUuZy4gMTExKQogIC8vIE5ldyBkYXRhOiB2ZWxvY2l0eSBpcyB0YW5oLW5vcm1hbGl6ZWQgKC0xIHRvICsxKQogIGlmKCF2KSByZXR1cm4gMDsKICB2YXIgYWJzPU1hdGguYWJzKHYpOwogIGlmKGFicz4xKSB2PXYvTWF0aC5tYXgoYWJzLDUwKTsgLy8gY29tcHJlc3MgbGFyZ2UgdmFsdWVzCiAgcmV0dXJuIE1hdGgubWF4KC0xLE1hdGgubWluKDEsdikpOwp9CgpmdW5jdGlvbiB2Qyh2KXsKICB2PW5vcm1WKHYpOwogIC8vIE5vdyB2IGlzIGFsd2F5cyAtMSB0byArMQogIC8vIFVzZSByZWxhdGl2ZSByYW5raW5nIHdpdGhpbiBjdXJyZW50IGRhdGEgZm9yIGJldHRlciBzcHJlYWQKICBpZighdkMuX3JuZ3x8IXZDLl9tYXhQb3N8fERhdGUubm93KCktdkMuX3RzPjMwMDApewogICAgdmFyIG5vcm1zPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gbm9ybVYoZC52ZWxvY2l0eXx8MCk7fSk7CiAgICB2YXIgcG9zPW5vcm1zLmZpbHRlcihmdW5jdGlvbih4KXtyZXR1cm4geD4wO30pOwogICAgdmFyIG5lZz1ub3Jtcy5maWx0ZXIoZnVuY3Rpb24oeCl7cmV0dXJuIHg8MDt9KTsKICAgIHZDLl9tYXhQb3M9cG9zLmxlbmd0aD9NYXRoLm1heC5hcHBseShudWxsLHBvcyk6MC4xOwogICAgdkMuX21heE5lZz1uZWcubGVuZ3RoP01hdGguYWJzKE1hdGgubWluLmFwcGx5KG51bGwsbmVnKSk6MC4xOwogICAgdkMuX3JuZz10cnVlOyB2Qy5fdHM9RGF0ZS5ub3coKTsKICB9CiAgaWYodj4wLjAwNSl7CiAgICB2YXIgbj1NYXRoLm1pbigxLHYvKHZDLl9tYXhQb3N8fDAuMSkpOwogICAgcmV0dXJuIGNvbG9yU2NhbGUobixbCiAgICAgIFswLjAwLCcjMmEyODE4J10sICAvLyBiYXJlbHkgd2FybQogICAgICBbMC4yNSwnIzhhNjAxMCddLCAgLy8gZGFyayBnb2xkCiAgICAgIFswLjU1LCcjYzg3MDIwJ10sICAvLyBhbWJlcgogICAgICBbMC44MCwnI2Q4NDAxMCddLCAgLy8gb3JhbmdlCiAgICAgIFsxLjAwLCcjZTgxMDEwJ10sICAvLyByZWQg4oCUIHN1cmdpbmcKICAgIF0pOwogIH0gZWxzZSBpZih2PC0wLjAwNSl7CiAgICB2YXIgbj1NYXRoLm1pbigxLE1hdGguYWJzKHYpLyh2Qy5fbWF4TmVnfHwwLjEpKTsKICAgIHJldHVybiBjb2xvclNjYWxlKG4sWwogICAgICBbMC4wMCwnIzE4MjAyOCddLCAgLy8gYmFyZWx5IGNvb2wKICAgICAgWzAuMjUsJyMxYTUwNzAnXSwgIC8vIGRhcmsgdGVhbAogICAgICBbMC41NSwnIzEwNjBhMCddLCAgLy8gYmx1ZQogICAgICBbMS4wMCwnIzA4MjhjMCddLCAgLy8gZGVlcCBibHVlIOKAlCBjb29saW5nIGZhc3QKICAgIF0pOwogIH0gZWxzZSB7CiAgICByZXR1cm4gJyMyNTJlM2EnOyAvLyBzdGFibGUg4oCUIG5ldXRyYWwgc2xhdGUKICB9Cn0KCnZhciBsYXllcj0nYXR0ZW50aW9uJyxTRUw9bnVsbCxGQVZTPW5ldyBTZXQoKTsKCi8vIE1BUApmdW5jdGlvbiBwcm9qXyh3LGgscGFkKXsKICBwYWQ9cGFkfHwyMDsKICB2YXIgbWluTG9uPTY4LjEsbWF4TG9uPTk3LjQsbWluTGF0PTYuNSxtYXhMYXQ9MzcuMTsKICB2YXIgc2NYPSh3LXBhZCoyKS8obWF4TG9uLW1pbkxvbik7CiAgdmFyIHNjWT0oaC1wYWQqMikvKG1heExhdC1taW5MYXQpOwogIHZhciBzYz1NYXRoLm1pbihzY1gsc2NZKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLShtYXhMb24tbWluTG9uKSpzYykvMjsKICB2YXIgb3k9cGFkKyhoLXBhZCoyLShtYXhMYXQtbWluTGF0KSpzYykvMjsKICByZXR1cm4gZnVuY3Rpb24obG9uLGxhdCl7cmV0dXJuIFtveCsobG9uLW1pbkxvbikqc2MsIG95KyhtYXhMYXQtbGF0KSpzY107fTsKfQpmdW5jdGlvbiBnZW8ycGF0aChnZW9tLHBqKXsKICB2YXIgZD0nJzsKICBmdW5jdGlvbiByaW5nKGNzKXt2YXIgcz0nJztjcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cGooY1swXSxjWzFdKTtzKz0oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSk7cmV0dXJuIHMrJ1onO30KICBpZihnZW9tLnR5cGU9PT0nUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pOwogIGVsc2UgaWYoZ2VvbS50eXBlPT09J011bHRpUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihwKXtwLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTt9KTsKICByZXR1cm4gZDsKfQpmdW5jdGlvbiBjdHIoZ2VvbSl7CiAgdmFyIHB0cz1bXTsKICBmdW5jdGlvbiBjb2woYyl7aWYodHlwZW9mIGNbMF09PT0nbnVtYmVyJykgcHRzLnB1c2goYyk7ZWxzZSBjLmZvckVhY2goY29sKTt9CiAgY29sKGdlb20uY29vcmRpbmF0ZXMpOwogIGlmKCFwdHMubGVuZ3RoKSByZXR1cm4gWzAsMF07CiAgcmV0dXJuIFtwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFswXTt9LDApL3B0cy5sZW5ndGgscHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMV07fSwwKS9wdHMubGVuZ3RoXTsKfQpmdW5jdGlvbiBzTmFtZShwcm9wcyl7CiAgdmFyIHJhdz1wcm9wcy5zdF9ubXx8cHJvcHMuTkFNRV8xfHxwcm9wcy5uYW1lfHxwcm9wcy5OQU1FfHwnJzsKICB2YXIgbWFwPXsnTGFkYWtoJzonSmFtbXUgYW5kIEthc2htaXInLCdKYW1tdSAmIEthc2htaXInOidKYW1tdSBhbmQgS2FzaG1pcicsJ1V0dGFyYW5jaGFsJzonVXR0YXJha2hhbmQnLCdBbmRhbWFuIGFuZCBOaWNvYmFyJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnQW5kYW1hbiAmIE5pY29iYXIgSXNsYW5kJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnTkNUIG9mIERlbGhpJzonRGVsaGknLCdQb25kaWNoZXJyeSc6J1B1ZHVjaGVycnknLCdEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpJzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsJ0RhbWFuIGFuZCBEaXUnOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1J307CiAgcmV0dXJuIG1hcFtyYXddfHxyYXc7Cn0KCnZhciBjYWNoZWRHZW89bnVsbDsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoYXR0ZW1wdCl7CiAgYXR0ZW1wdCA9IGF0dGVtcHR8fDE7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goJ2h0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9naC91ZGl0LTAwMS9pbmRpYS1tYXBzLWRhdGFAbWFzdGVyL3RvcG9qc29uL2luZGlhLmpzb24nKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwxMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwzMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciw2MDAwKTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1ttYXBdIGxvYWQgZmFpbGVkIGF0dGVtcHQgJythdHRlbXB0Kyc6JyxlLm1lc3NhZ2UpOwogICAgaWYoYXR0ZW1wdDw1KXsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe2xvYWRNYXAoYXR0ZW1wdCsxKTt9LCBhdHRlbXB0KjIwMDApOwogICAgfSBlbHNlIHsKICAgICAgdmFyIG1pPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKTsKICAgICAgaWYobWkpIG1pLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6IzJhM2E0YTtwYWRkaW5nOjQwcHg7dGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6bW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4Ij5NYXAgdW5hdmFpbGFibGUg4oCUIHJlZnJlc2ggdG8gcmV0cnk8L2Rpdj4nOwogICAgfQogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKLy8gU2luZ2xlIHNvdXJjZSBvZiB0cnV0aCBmb3IgZW1vdGlvbiBjb2xvcgovLyBCb3RoIG1hcCBhbmQgcGFuZWwgY2FsbCB0aGlzIOKAlCBndWFyYW50ZWVzIHRoZXkgYWx3YXlzIG1hdGNoCmZ1bmN0aW9uIGdldEVmZmVjdGl2ZUVtb3Rpb24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgLy8gMS4gVHJ5IExJVkUuZG9taW5hbnRfZW1vdGlvbiAoc2V0IGJ5IC9hcGkvc3RhdGVzKQogIHZhciBkb209bGl2ZS5kb21pbmFudF9lbW90aW9ufHxkLmRvbWluYW50X2Vtb3Rpb247CgogIC8vIDIuIFRyeSBjb21wdXRpbmcgZnJvbSBlbW90aW9ucyBicmVha2Rvd24KICBpZighZG9tKXsKICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgICBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpOwogIH0KCiAgLy8gMy4gRmFsbGJhY2s6IGluZmVyIGZyb20gZG9taW5hbnQgbmFycmF0aXZlIChzYW1lIGxvZ2ljIGV2ZXJ5d2hlcmUpCiAgaWYoIWRvbSl7CiAgICB2YXIgbnA9KGxpdmUuZG9taW5hbnRfbmFycmF0aXZlfHxkLmRvbWluYW50X25hcnJhdGl2ZXx8JycpLnRvTG93ZXJDYXNlKCk7CiAgICBpZihucC5tYXRjaCgvYm9yZGVyfHRlcnJvcnxzZWN1cml0eXxjb25mbGljdHxhdHRhY2t8d2FyfGluZmlsdHJhdC8pKSBkb209J2ZlYXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvc2NhbXxjb3JydXB0fHByb3Rlc3R8YXJyZXN0fHZpb2xlbmNlfG91dHJhZ2V8Y3JpbWUvKSkgZG9tPSdhbmdlcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9kZXZlbG9wfGludmVzdHxncm93dGh8bGF1bmNofGluYXVndXJ8cmVmb3JtfHByb2dyZXNzfGJvb3N0LykpIGRvbT0naG9wZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9jdWx0dXJlfGhlcml0YWdlfHByaWRlfHZpY3Rvcnl8Y2VsZWJyYXR8bWVkYWx8YWNoaWV2ZW1lbnQvKSkgZG9tPSdwcmlkZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9mbG9vZHxkcm91Z2h0fHVuZW1wbG95bWVudHxpbmZsYXRpb258c2hvcnRhZ2V8Y3Jpc2lzfGNvbmNlcm4vKSkgZG9tPSdhbnhpZXR5JzsKICAgIGVsc2UgaWYoKGxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MCk+NSkgZG9tPSdhbnhpZXR5JzsgLy8gYWN0aXZlIHN0YXRlIGRlZmF1bHQKICAgIGVsc2UgZG9tPSdhbnhpZXR5JzsgLy8gZ2xvYmFsIGRlZmF1bHQKICB9CgogIHJldHVybiBkb207Cn0KCi8vIEdldCBlc3RpbWF0ZWQgZW1vdGlvbiBicmVha2Rvd24gKGZvciBwYW5lbCBkb251dCB3aGVuIHJlYWwgZGF0YSBtaXNzaW5nKQpmdW5jdGlvbiBnZXRFbW90aW9uQnJlYWtkb3duKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogIGlmKE9iamVjdC5rZXlzKGVtb3MpLmxlbmd0aCkgcmV0dXJuIHtlbW90aW9uczplbW9zLGVzdGltYXRlZDpmYWxzZX07CiAgLy8gQnVpbGQgc2tld2VkIGRpc3RyaWJ1dGlvbiBmcm9tIGVmZmVjdGl2ZSBlbW90aW9uCiAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICB2YXIgYmFzZT17YW54aWV0eToxMyxhbmdlcjoxMyxob3BlOjEzLHByaWRlOjEzLGZlYXI6MTN9OwogIGJhc2VbZG9tXT00ODsKICByZXR1cm4ge2Vtb3Rpb25zOmJhc2UsZXN0aW1hdGVkOnRydWV9Owp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgLy8gUHJlLWNvbXB1dGUgYXR0ZW50aW9uIHJhbmdlIG9uY2UgcGVyIHJlbmRlcgogIHZhciBhdHRTY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiB4LmF0dGVudGlvbnx8MDt9KS5maWx0ZXIoZnVuY3Rpb24odil7cmV0dXJuIHY+MDt9KTsKICB2YXIgYXR0TW49YXR0U2NvcmVzLmxlbmd0aD9NYXRoLm1pbi5hcHBseShudWxsLGF0dFNjb3Jlcyk6MDsKICB2YXIgYXR0TXg9YXR0U2NvcmVzLmxlbmd0aD8oTWF0aC5tYXguYXBwbHkobnVsbCxhdHRTY29yZXMpfHwxKToxOwogIF9hTm9ybS5tbj1hdHRNbjtfYU5vcm0ubXg9YXR0TXg7X2FOb3JtLnRzPURhdGUubm93KCk7IC8vIGtlZXAgY2FjaGUgd2FybQoKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLGZpbGwsb3BhY2l0eTsKICAgIHZhciBhdHROb3JtPU1hdGgubWF4KDAsTWF0aC5taW4oMSwoZC5hdHRlbnRpb24tYXR0TW4pL01hdGgubWF4KGF0dE14LWF0dE1uLDEpKSk7CgogICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICAgIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgICBvcGFjaXR5PU1hdGgubWF4KDAuMjUsMC4zK2F0dE5vcm0qMC43KTsgLy8gZGltIGxvdywgYnJpZ2h0IGhpZ2gKICAgIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRlPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICBmaWxsPWVNYXBbZGVdfHwnIzMzNDQ1NSc7CiAgICAgIC8vIFZhcnkgb3BhY2l0eSBieSBzaWduYWwgc3RyZW5ndGggc28gZG9taW5hbnQtZW1vdGlvbiBzdGF0ZXMgcG9wCiAgICAgIHZhciBjb25mPWQuY29uZmlkZW5jZT09PSdISUdIJz8xLjA6ZC5jb25maWRlbmNlPT09J01FRElVTSc/MC43OjAuNDsKICAgICAgb3BhY2l0eT1NYXRoLm1heCgwLjI1LDAuMzUrYXR0Tm9ybSowLjUpKmNvbmY7CiAgICB9IGVsc2UgewogICAgICBmaWxsPXZDKGQudmVsb2NpdHl8fDApOwogICAgICAvLyBWYXJ5IG9wYWNpdHkgYnkgbm9ybWFsaXplZCB2ZWxvY2l0eSBtYWduaXR1ZGUKICAgICAgdmFyIHZlbE5vcm09TWF0aC5taW4oMSxNYXRoLmFicyhub3JtVihkLnZlbG9jaXR5fHwwKSkvKHZDLl9tYXhQb3N8fHZDLl9tYXhOZWd8fDAuMSkpOwogICAgICBvcGFjaXR5PU1hdGgubWF4KDAuMzUsMC4zNSt2ZWxOb3JtKjAuNjUpOwogICAgfQogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGZpbGwpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsb3BhY2l0eSk7CiAgfSk7Cn0KCmZ1bmN0aW9uIGF0dGFjaEludGVyYWN0aW9ucygpewogIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpewogICAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpOwogICAgICB2YXIgZD1nKG5tKTsKICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgICAgIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgbGF0ZXN0PScnOwogICAgICBpZihkLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlcy5sZW5ndGgpIGxhdGVzdD1kLm5hcnJhdGl2ZXNbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkLm5hcnJhdGl2ZXNbMF0ubmFtZS5zbGljZSgxKTsKICAgICAgZWxzZSBpZihsaXZlLmRvbWluYW50X25hcnJhdGl2ZSkgbGF0ZXN0PWxpdmUuZG9taW5hbnRfbmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2xpdmUuZG9taW5hbnRfbmFycmF0aXZlLnNsaWNlKDEpOwoKICAgICAgdmFyIHJvd3M9Jyc7CiAgICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgICAgIHZhciBhdHQ9bGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwOwogICAgICAgIHZhciBkbHQ9bGl2ZS5kZWx0YXx8ZC5kZWx0YXx8MDsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5BdHRlbnRpb248L3NwYW4+PHN0cm9uZz4nK2F0dC50b0ZpeGVkKDEpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgKGRsdCE9PTA/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaGlmdDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonKyhkbHQ+MD8nI2UwNWEyOCc6JyMzYmI4ZDgnKSsnIj4nKyhkbHQ+MD8nKyc6JycpK2RsdCsnPC9zdHJvbmc+PC9kaXY+JzonJykrCiAgICAgICAgICAobGF0ZXN0Pyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Ub3AgbmFycmF0aXZlPC9zcGFuPjxzdHJvbmc+JytsYXRlc3QrJzwvc3Ryb25nPjwvZGl2Pic6JycpOwogICAgICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBkb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgICAgaWYoZG9tRW1vKXsKICAgICAgICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOmQuZW1vdGlvbnN8fHt9OwogICAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RG9taW5hbnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JytwYWxbZG9tRW1vXSsnIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgICAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vcykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICAgICAgICB2YXIgdG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgICAgICAgaWYodG90PjAmJnRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7fQogICAgICAgICAgcm93cys9ZUwuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4gc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweCI+PHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytwYWxba3ZbMF1dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nK2t2WzBdKyc8L3NwYW4+PHN0cm9uZz4nK01hdGgucm91bmQoa3ZbMV0qMTAwL01hdGgubWF4KDEsdG90KSkrJyU8L3N0cm9uZz48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICAgIH0KICAgICAgfSBlbHNlIHsKICAgICAgICB2YXIgdmVsPWxpdmUudmVsb2NpdHl8fGQudmVsb2NpdHl8fDA7CiAgICAgICAgdmFyIHZlbERpcj12ZWw+MC4xPydSaXNpbmcgZmFzdCc6dmVsPjAuMDI/J1Jpc2luZyc6dmVsPC0wLjA1PydDb29saW5nJzonU3RhYmxlJzsKICAgICAgICB2YXIgdmVsQ29sPXZlbD4wLjAyPycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Nb21lbnR1bTwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRpcmVjdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nK3ZlbERpcisnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfQoKICAgICAgdGlwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0idHQtbiI+JytubSsnPC9kaXY+Jytyb3dzKyhsYXRlc3QmJmxheWVyIT09J2F0dGVudGlvbic/JzxkaXYgY2xhc3M9InR0LW5hciI+PHN0cm9uZz5OYXJyYXRpdmU8L3N0cm9uZz4nK2xhdGVzdCsnPC9kaXY+JzonJyk7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE5MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTUwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0nMSc7CiAgICB9KTsKcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJyxmdW5jdGlvbigpe3RpcC5zdHlsZS5vcGFjaXR5PTA7fSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe3NlbGVjdF8ocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8gU1RBVEUgUEFORUwKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTdGF0ZUNvbnRleHQobm0pewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlLWNvbnRleHQvJytlbmNvZGVVUklDb21wb25lbnQobm0pKTsKICAgIGlmKCFyLm9rKSByZXR1cm4gbnVsbDsKICAgIHJldHVybiBhd2FpdCByLmpzb24oKTsKICB9Y2F0Y2goZSl7IHJldHVybiBudWxsOyB9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoRGV0YWlsKG5tKXsKICB0cnl7CiAgICB2YXIgY29udHJvbGxlcj1uZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgICB2YXIgdGlkPXNldFRpbWVvdXQoZnVuY3Rpb24oKXtjb250cm9sbGVyLmFib3J0KCk7fSw1MDAwKTsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlLycrZW5jb2RlVVJJQ29tcG9uZW50KG5tKSx7c2lnbmFsOmNvbnRyb2xsZXIuc2lnbmFsfSk7CiAgICBjbGVhclRpbWVvdXQodGlkKTsKICAgIGlmKCFyLm9rKSByZXR1cm4gZmFsc2U7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQmJmQubmFtZSl7CiAgICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKGQuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgICAgU0Rbbm1dPU9iamVjdC5hc3NpZ24oe30sZCx7ZW1vdGlvbnM6ZW1vcyxkb21pbmFudF9lbW90aW9uOmRvbSxkZWx0YTpkLmRlbHRhXzI0aHx8MH0pOwogICAgICBMSVZFW25tXT1PYmplY3QuYXNzaWduKExJVkVbbm1dfHx7fSx7CiAgICAgICAgYXR0ZW50aW9uOmQuYXR0ZW50aW9uLHZlbG9jaXR5OmQudmVsb2NpdHksZGVsdGE6ZC5kZWx0YV8yNGh8fDAsCiAgICAgICAgZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlLAogICAgICAgIGVtb3Rpb25zOmVtb3MsbmFycmF0aXZlczpkLm5hcnJhdGl2ZXMsc2lnbmFsX2NvdW50OmQuc2lnbmFsX2NvdW50LAogICAgICAgIHNvdXJjZV9jb3VudDpkLnNvdXJjZV9jb3VudCxjb25maWRlbmNlOmQuY29uZmlkZW5jZQogICAgICB9KTsKICAgIH0KICAgIHJldHVybiB0cnVlOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbm0sZS5tZXNzYWdlKTsKICAgIHJldHVybiBmYWxzZTsKICB9Cn0KCmZ1bmN0aW9uIHNlbGVjdF8obm0pewogIFNFTD1ubTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5jbGFzc0xpc3QudG9nZ2xlKCdzZWxlY3RlZCcscC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpPT09bm0pOwogIH0pOwogIC8vIFNob3cgbG9hZGluZyBzdGF0ZSBpbW1lZGlhdGVseSB3aXRoIHdoYXRldmVyIExJVkUgZGF0YSB3ZSBoYXZlCiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZihwYW5lbCl7CiAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICBwYW5lbC5pbm5lckhUTUw9CiAgICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAgICc8ZGl2PjxkaXYgY2xhc3M9InNwLWVrIj4nKyhsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICAgJzwvYnV0dG9uPicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA4ZW0iPicrCiAgICAgICAgJ0xvYWRpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJysKICAgICAgICAobGl2ZS5hdHRlbnRpb24/Jzxicj48YnI+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtjb2xvcjp2YXIoLS1pbmspIj5BdHRlbnRpb24gJytsaXZlLmF0dGVudGlvbi50b0ZpeGVkKDEpKyc8L3NwYW4+JzonJykrCiAgICAgICAgKGxpdmUuZG9taW5hbnRfZW1vdGlvbj8nPGJyPjxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrbGl2ZS5kb21pbmFudF9lbW90aW9uKycgc2lnbmFsIGRvbWluYW50PC9zcGFuPic6JycpKwogICAgICAnPC9kaXY+JzsKICB9CiAgLy8gRmV0Y2ggZnVsbCBkZXRhaWwgd2l0aCB0aW1lb3V0IOKAlCBmYWxsIGJhY2sgdG8gTElWRSBkYXRhIGlmIHNsb3cKICB2YXIgZGV0YWlsVGltZW91dD1zZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICAvLyBBZnRlciAzcywgcmVuZGVyIHdpdGggd2hhdGV2ZXIgd2UgaGF2ZSByYXRoZXIgdGhhbiBrZWVwIHVzZXIgd2FpdGluZwogICAgaWYoU0VMPT09bm0mJiFTRFtubV0pewogICAgICBjb25zb2xlLndhcm4oJ1tzZWxlY3RdIHRpbWVvdXQg4oCUIHJlbmRlcmluZyBmcm9tIExJVkUgZGF0YScpOwogICAgICByZW5kZXJQYW5lbChubSxudWxsKTsKICAgIH0KICB9LDMwMDApOwoKICAvLyBBbHNvIGZldGNoIGN0eCBmb3IgYXR0ZW50aW9uIGxheWVyCiAgdmFyIGN0eFByb21pc2U9KGxheWVyPT09J2F0dGVudGlvbicpP2ZldGNoU3RhdGVDb250ZXh0KG5tKTpQcm9taXNlLnJlc29sdmUobnVsbCk7CgogIFByb21pc2UuYWxsKFtmZXRjaERldGFpbChubSksY3R4UHJvbWlzZV0pLnRoZW4oZnVuY3Rpb24ocmVzdWx0cyl7CiAgICBjbGVhclRpbWVvdXQoZGV0YWlsVGltZW91dCk7CiAgICBpZihTRUwhPT1ubSkgcmV0dXJuOwogICAgdmFyIGN0eD1yZXN1bHRzWzFdOwogICAgcmVuZGVyUGFuZWwobm0sY3R4KTsKICAgIHZhciBwYXRoPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9Iicrbm0rJyJdJyk7CiAgICBpZihwYXRoJiZsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgaWYoZU1hcFtkb21dKSBwYXRoLnNldEF0dHJpYnV0ZSgnZmlsbCcsZU1hcFtkb21dKTsKICAgIH0gZWxzZSB7CiAgICAgIGFwcGx5TGF5ZXIoKTsKICAgIH0KICB9KS5jYXRjaChmdW5jdGlvbihlKXsKICAgIGNsZWFyVGltZW91dChkZXRhaWxUaW1lb3V0KTsKICAgIGNvbnNvbGUud2FybignW3NlbGVjdF0nLGUpOwogICAgaWYoU0VMPT09bm0pIHJlbmRlclBhbmVsKG5tLG51bGwpOwogIH0pOwp9CgpmdW5jdGlvbiByZW5kZXJQYW5lbChubSxjdHgpewogIHZhciBkPWcobm0pOwogIGlmKCFkfHwhZC5hdHRlbnRpb24pIGQ9TElWRVtubV18fHt9OyAvLyBmYWxsYmFjayB0byBMSVZFIGlmIFNEIG5vdCBsb2FkZWQKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKCFwYW5lbCkgcmV0dXJuOwogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICB2YXIgaGVhZGVyPQogICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICc8ZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLWVrIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+JysKICAgICAgICAgIChsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKwogICAgICAgICAgKGQuY29uZmlkZW5jZT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/J3JnYmEoNTEsMjA0LDEwMiwwLjEpJzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8ncmdiYSgyMjQsOTAsNDAsMC4xKSc6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknKSsnO2NvbG9yOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8nIzMzY2M2Nic6ZC5jb25maWRlbmNlPT09J01FRElVTSc/JyNlMDVhMjgnOidyZ2JhKDI1NSwyNTUsMjU1LDAuMyknKSsnIj4nK2QuY29uZmlkZW5jZSsnIFNJR05BTDwvc3Bhbj4nOicnKSsKICAgICAgICAgIChkLmlzX3JlZ2lvbmFsX3N0b3J5Pyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpO2NvbG9yOiMzYmI4ZDgiPlJFR0lPTkFMIFNQSUtFPC9zcGFuPic6JycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICc8L2J1dHRvbj4nKwogICAgJzwvZGl2Pic7CgogIHZhciBib2R5PScnOwoKICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgIHZhciBkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPyd1cCc6J2RuJzsKICAgIHZhciBuYXJyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdGw9KGQudGltZWxpbmUmJmQudGltZWxpbmUubGVuZ3RoKT9kLnRpbWVsaW5lOlswLDAsMCwwLDAsMCwwLGQuYXR0ZW50aW9ufHwwXTsKICAgIHZhciB0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgICB2YXIgdHc9MjYwLHRoPTYyLHRwPTU7CiAgICB2YXIgcHRzPXRsLm1hcChmdW5jdGlvbih2LGkpe3JldHVyblt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICAgIHZhciBwRD1wdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuKGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pLmpvaW4oJycpOwogICAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICAgIHZhciBhYz1hQyhkLmF0dGVudGlvbnx8MCk7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ0hvdyBpbnRlbnNlbHkgJysobm0uc3BsaXQoJyAnKVswXSkrJyBpcyBiZWluZyBkaXNjdXNzZWQgbmF0aW9uYWxseS4gU2NvcmUgb2YgJytkLmF0dGVudGlvbisnIG1lYW5zICcrKGQuYXR0ZW50aW9uPjYwPyd2ZXJ5IGhpZ2gg4oCUIHRoaXMgc3RhdGUgZG9taW5hdGVzIG5hdGlvbmFsIGRpc2NvdXJzZSc6ZC5hdHRlbnRpb24+MzU/J2VsZXZhdGVkIOKAlCBjbGVhcmx5IGluIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24nOmQuYXR0ZW50aW9uPjE1Pydtb2RlcmF0ZSDigJQgc29tZSBuYXRpb25hbCBjb3ZlcmFnZSc6ZC5hdHRlbnRpb24+NT8nbG93IOKAlCBsaW1pdGVkIG5hdGlvbmFsIGF0dGVudGlvbic6J21pbmltYWwg4oCUIGZldyBzaWduYWxzIGRldGVjdGVkJykrJy4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0iaW5zaWdodCIgc3R5bGU9IicrKGQuY29uZmlkZW5jZT09PSJMT1ciPydib3JkZXItY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjA2KTtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTppdGFsaWMnOicnKSsnIj4nKyhjdHgmJmN0eC5icmllZj9jdHguYnJpZWY6KGQuY29uZmlkZW5jZT09PSJMT1ciJiYhZC5zdW1tYXJ5KT8nTGltaXRlZCBzaWduYWxzIGRldGVjdGVkIGZvciAnK25tKycuIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy4nOmQuc3VtbWFyeXx8J0NvbGxlY3Rpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJykrJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKG5hcnJbMF0/bmFyclswXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25hcnJbMF0ubmFtZS5zbGljZSgxKTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAobmFyci5sZW5ndGg/CiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrbmFyci5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHZhciBubj1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPjxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25uKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4IiB0aXRsZT0iZ2FpbmluZyB0cmFjdGlvbiI+4oaRPC9zcGFuPic6bi5kaXI9PT0nZG93bic/JyA8c3BhbiBzdHlsZT0iY29sb3I6IzNiYjhkODtmb250LXNpemU6OXB4IiB0aXRsZT0icmV0cmVhdGluZyI+4oaTPC9zcGFuPic6JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS12YWwiPicrdmFsLnRvRml4ZWQoMSkrJyU8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLHZhbCoyLjUpKyclO2JhY2tncm91bmQ6Jysobi5kaXI9PT0ndXAnPycjZTA1YTI4JzpuLmRpcj09PSdkb3duJz8nIzNiYjhkOCc6JyMzMzQ0NTUnKSsnIj48L2Rpdj48L2Rpdj48L2Rpdj4nOwogICAgICAgICAgfSkuam9pbignJykrJzwvZGl2Pic6CiAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvdy1zaWduYWwgcmVnaW9uLiBNb25pdG9yaW5nIHJlZ2lvbmFsIHNvdXJjZXMuPC9kaXY+JykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BdHRlbnRpb24g4oCUIDggZGF5czwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InRsLXdyYXAiPjxzdmcgdmlld0JveD0iMCAwICcrdHcrJyAnK3RoKyciIHN0eWxlPSJ3aWR0aDoxMDAlO2hlaWdodDoxMDAlIj4nKwogICAgICAgICAgJzxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0idGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKyciIHgxPSIwIiB4Mj0iMCIgeTE9IjAiIHkyPSIxIj4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAuMjUiLz4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMCIvPicrCiAgICAgICAgICAnPC9saW5lYXJHcmFkaWVudD48L2RlZnM+JysKICAgICAgICAgICc8cGF0aCBkPSInK2FEKyciIGZpbGw9InVybCgjdGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKycpIiAvPicrCiAgICAgICAgICAnPHBhdGggZD0iJytwRCsnIiBmaWxsPSJub25lIiBzdHJva2U9IicrYWMrJyIgc3Ryb2tlLXdpZHRoPSIxLjIiLz4nKwogICAgICAgICAgcHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAnPGNpcmNsZSBjeD0iJytwWzBdKyciIGN5PSInK3BbMV0rJyIgcj0iJysoaT09PXB0cy5sZW5ndGgtMT8yLjI6MS4yKSsnIiBmaWxsPSInK2FjKyciLz4nO30pLmpvaW4oJycpKwogICAgICAgICc8L3N2Zz48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFscyA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nKyhkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aD9kLmFydGljbGVzLmxlbmd0aDowKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9ImFydC1saXN0Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMubWFwKGZ1bmN0aW9uKGEpe3JldHVybiAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIHNyYz1hLnNyY3x8Jyc7CiAgICAgICAgICAgICAgdmFyIGlzWXQ9c3JjLmluZGV4T2YoJ3lvdXR1YmUnKT49MDsKICAgICAgICAgICAgICB2YXIgaXNSZWQ9c3JjLmluZGV4T2YoJ3JlZGRpdCcpPj0wOwogICAgICAgICAgICAgIHZhciBsYWJlbD1pc1l0PydyZWdpb25hbCBtZWRpYSc6aXNSZWQ/J3B1YmxpYyBkaXNjdXNzaW9uJzpzcmMuc3BsaXQoJy8nKVswXXx8c3JjOwogICAgICAgICAgICAgIHZhciBjb2w9aXNZdHx8aXNSZWQ/J3JnYmEoMjI0LDkwLDQwLDAuNSknOid2YXIoLS1mYWludCknOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0iYXJ0LWl0ZW0iPjxkaXYgY2xhc3M9ImFydC1zcmMiIHN0eWxlPSJjb2xvcjonK2NvbCsnIj4nK2xhYmVsKyc8L2Rpdj48ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj48L2Rpdj4nOwogICAgICAgICAgICB9KShhKTt9KS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjZweCAwIj5ObyBzaWduYWxzIGNvbGxlY3RlZCB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgLy8gVXNlIHNhbWUgZnVuY3Rpb25zIGFzIG1hcCDigJQgZ3VhcmFudGVlZCB0byBtYXRjaAogICAgdmFyIG1hcERvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgIHZhciBicmVha2Rvd249Z2V0RW1vdGlvbkJyZWFrZG93bihubSk7CiAgICB2YXIgZW1vdGlvbnM9YnJlYWtkb3duLmVtb3Rpb25zOwogICAgdmFyIGhhc0Vtb3M9IWJyZWFrZG93bi5lc3RpbWF0ZWQ7CiAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vdGlvbnMpOwogICAgdmFyIGVUb3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICBpZihlVG90PjAmJmVUb3Q8PTEuMDEpe2VMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7fQogICAgdmFyIHRvdD1NYXRoLm1heCgxLGVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApKTsKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgaWYoIWVMLmxlbmd0aCl7cGFuZWwuaW5uZXJIVE1MPWhlYWRlcisnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweCI+Tm8gZW1vdGlvbiBkYXRhIHlldC48L2Rpdj4nO3JldHVybjt9CiAgICAvLyBkb21FbW8gPSBzYW1lIGFzIG1hcCBjb2xvciAoZnJvbSBnZXRFZmZlY3RpdmVFbW90aW9uKQogICAgdmFyIGRvbUVtbz1tYXBEb21FbW87CiAgICAvLyBSZW9yZGVyIGVMIHNvIGRvbWluYW50IHNob3dzIGZpcnN0CiAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7CiAgICAgIGlmKGFbMF09PT1kb21FbW8pIHJldHVybiAtMTsKICAgICAgaWYoYlswXT09PWRvbUVtbykgcmV0dXJuIDE7CiAgICAgIHJldHVybiBiWzFdLWFbMV07CiAgICB9KTsKICAgIHZhciBkb21QY3Q9TWF0aC5yb3VuZCgoZUxbMF0/ZUxbMF1bMV06MjApKjEwMC90b3QpOwogICAgdmFyIG5hcnIyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdG9wTmFyU3RyPW5hcnIyLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignIGFuZCAnKTsKICAgIHZhciB3aGF0SXQ9e2FueGlldHk6J0EgZGlmZnVzZSB1bmVhc2UgaXMgcnVubmluZyB0aHJvdWdoIHNpZ25hbHMgZnJvbSAnK25tKyh0b3BOYXJTdHI/JywgY29uY2VudHJhdGVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBTaWduYWxzIGF0IHRoaXMgc3RhZ2UgdGVuZCB0byBiZSBsb2NhbGx5IGFic29yYmVkIGJlZm9yZSB3aWRlbmluZy4nOicuJyAgKSxhbmdlcjonRnJ1c3RyYXRpb24gc2lnbmFscyBhcmUgZWxldmF0ZWQgaW4gJytubSsodG9wTmFyU3RyPycsIHBhcnRpY3VsYXJseSBhcm91bmQgJyt0b3BOYXJTdHIrJy4gVGhlIHRvbmUgc3VnZ2VzdHMgcHJlc3N1cmUgYnVpbGRpbmcgcmF0aGVyIHRoYW4gYSBzaW5nbGUgZXZlbnQuJzonLiBUaGUgZW1vdGlvbmFsIHJlZ2lzdGVyIGlzIG5vdGljZWFibHkgdGVuc2UuJyksaG9wZTonQW4gdW51c3VhbGx5IG9wdGltaXN0aWMgc2lnbmFsIHJlZ2lzdGVyIGZyb20gJytubSsodG9wTmFyU3RyPycsIG9yaWVudGVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBXb3J0aCB3YXRjaGluZyDigJQgcG9zaXRpdmUgc2lnbmFscyBhdCB0aGlzIGRlbnNpdHkgYXJlIHJlbGF0aXZlbHkgcmFyZS4nOicuIEEgc2lnbmFsIHdvcnRoIG1vbml0b3JpbmcuJykscHJpZGU6J1N0cm9uZyBpZGVudGl0eSBzaWduYWxzIGluICcrbm0rKHRvcE5hclN0cj8nLCBjZW50cmVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBSZWdpb25hbGx5IGNvbmNlbnRyYXRlZCBhbmQgZW1vdGlvbmFsbHkgZGVuc2UuJzonLiBMb2NhbGx5IGNvbmNlbnRyYXRlZCwgZW1vdGlvbmFsbHkgc3Ryb25nLicpLGZlYXI6J0FwcHJlaGVuc2lvbiBzaWduYWxzIGluICcrbm0rKHRvcE5hclN0cj8nLCBhcm91bmQgJyt0b3BOYXJTdHIrJy4gVGhlc2UgdGVuZCB0byBpbnRlbnNpZnkgYmVmb3JlIGFjaGlldmluZyB3aWRlciB2aXNpYmlsaXR5Lic6Jy4gVGhlIHJlZ2lzdGVyIGNhcnJpZXMgYW4gZWRnZSB0aGF0IHRlbmRzIHRvIHByZWNlZGUgbGFyZ2VyIGN5Y2xlcy4nKX07CiAgICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7Y3VtQT1hMjsKICAgICAgdmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICAgIH0pLmpvaW4oJycpOwogICAgdmFyIGVkZXNjPXthbnhpZXR5OidEaWZmdXNlIHVuZWFzZSwgd29ycnkgc2lnbmFscycsYW5nZXI6J0ZydXN0cmF0aW9uLCBwcmVzc3VyZSBzaWduYWxzJyxob3BlOidPcHRpbWlzbSwgZm9yd2FyZCBtb21lbnR1bScscHJpZGU6J0lkZW50aXR5LCByZWdpb25hbCBhc3NlcnRpb24nLGZlYXI6J0FwcHJlaGVuc2lvbiwgdGhyZWF0IHBlcmNlcHRpb24nfTsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnVGhlIGVtb3Rpb25hbCByZWdpc3RlciBvZiBzaWduYWxzIGZyb20gJytubSsnIOKAlCB3aGF0IHRvbmUgcnVucyB0aHJvdWdoIHRoZSBkaXNjb3Vyc2UgYW5kIGhvdyBjb25jZW50cmF0ZWQgaXQgaXMuJysKICAgICc8L2Rpdj4nKwogICAgKCFoYXNFbW9zPyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjZweCAxMXB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7bWFyZ2luLWJvdHRvbToxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpIj5Fc3RpbWF0ZWQgZnJvbSBzaWduYWwgZGlyZWN0aW9uIOKAlCBsaW1pdGVkIGRpcmVjdCBlbW90aW9uIGRhdGEuPC9kaXY+JzonJykrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjE0cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6JytwYWxbZG9tRW1vXSsnMTQ7Ym9yZGVyOjFweCBzb2xpZCAnK3BhbFtkb21FbW9dKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3BhbFtkb21FbW9dKyc7bWFyZ2luLWJvdHRvbTo2cHgiPkRvbWluYW50IGVtb3Rpb248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI2cHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluaykiPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweCI+Jytkb21QY3QrJyUgwrcgJytubSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjhweDtsaW5lLWhlaWdodDoxLjU7Zm9udC1zdHlsZTppdGFsaWMiPicrd2hhdEl0W2RvbUVtb10rJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5FbW90aW9uYWwgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsiPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgNzYgNzYiIHN0eWxlPSJ3aWR0aDo3MnB4O2hlaWdodDo3MnB4O2ZsZXgtc2hyaW5rOjAiPicrYXJjcysnPC9zdmc+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4OyI+JysKICAgICAgICAgICAgZUwubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgICAgICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLHBjdD1NYXRoLnJvdW5kKHYqMTAwL3RvdCk7CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmNlbnRlcjttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo2cHg7Ij48c3BhbiBzdHlsZT0id2lkdGg6N3B4O2hlaWdodDo3cHg7Ym9yZGVyLXJhZGl1czoycHg7YmFja2dyb3VuZDonK3BhbFtrXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTEuNXB4O2NvbG9yOicrKGs9PT1kb21FbW8/J3ZhcigtLWluayknOid2YXIoLS1kaW0pJykrJyI+JytrLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2suc2xpY2UoMSkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0taW5rKSI+JytwY3QrJyU8L3NwYW4+JysKICAgICAgICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6JytwY3QrJyU7YmFja2dyb3VuZDonK3BhbFtrXSsnO29wYWNpdHk6MC43O2JvcmRlci1yYWRpdXM6MXB4Ij48L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICAgKGs9PT1kb21FbW8/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK2VkZXNjW2tdKyc8L2Rpdj4nOicnKSsKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbCBoZWFkbGluZXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHg7Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMuc2xpY2UoMCw1KS5tYXAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIGVDb2xvcj17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6NnB4O3BhZGRpbmc6NnB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzKTsiPicrCiAgICAgICAgICAgICAgICAoYS5lbW90aW9uPyc8c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK2VDb2xvclthLmVtb3Rpb25dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2s7bWFyZ2luLXRvcDo1cHg7ZmxleC1zaHJpbms6MCI+PC9zcGFuPic6JycpKwogICAgICAgICAgICAgICAgJzxkaXY+PGRpdiBzdHlsZT0iZm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQiPicrKGEudHh0fHxhLnRpdGxlfHwnJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweCI+JysoYS5zcmN8fCcnKSsoYS5lbW90aW9uPycgwrcgJythLmVtb3Rpb246JycpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjRweCAwIj5ObyBzaWduYWxzIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSB7CiAgICB2YXIgdmVsPWQudmVsb2NpdHl8fDA7CiAgICB2YXIgdmVsRGlyPXZlbD4wLjE1PydSaXNpbmcgZmFzdCc6dmVsPjAuMDU/J1Jpc2luZyc6dmVsPC0wLjE/J0Nvb2xpbmcgZmFzdCc6dmVsPC0wLjAyPydDb29saW5nJzonU3RhYmxlJzsKICAgIHZhciB2ZWxDb2w9dmVsPjAuMDU/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgdmFyIHZlbERlc2M9eydSaXNpbmcgZmFzdCc6J1NpZ25hbCB2b2x1bWUgYWNjZWxlcmF0aW5nIHNoYXJwbHkg4oCUIHRoaXMgc3RhdGUgaXMgZW50ZXJpbmcgYW4gYWN0aXZlIGRpc2NvdXJzZSBjeWNsZS4nLCdSaXNpbmcnOidBdHRlbnRpb24gaXMgYnVpbGRpbmcg4oCUIHNpZ25hbHMgc3VnZ2VzdCBhIG5hcnJhdGl2ZSBnYWluaW5nIHJlZ2lvbmFsIHRyYWN0aW9uLicsJ1N0YWJsZSc6J1NpZ25hbCBhY3Rpdml0eSBob2xkaW5nIHN0ZWFkeSDigJQgbm8gc2lnbmlmaWNhbnQgYWNjZWxlcmF0aW9uIG9yIHJldHJlYXQgZGV0ZWN0ZWQuJywnQ29vbGluZyc6J0F0dGVudGlvbiBiZWdpbm5pbmcgdG8gZWFzZSDigJQgdGhlIGN1cnJlbnQgbmFycmF0aXZlIGN5Y2xlIG1heSBiZSBydW5uaW5nIGl0cyBjb3Vyc2UuJywnQ29vbGluZyBmYXN0JzonU2lnbmFsIHZvbHVtZSByZXRyZWF0aW5nIHF1aWNrbHkg4oCUIGF0dGVudGlvbiBoYXMgbGlrZWx5IHBlYWtlZCBhbmQgaXMgZGlzcGVyc2luZy4nfTsKICAgIHZhciBuYXJyMz1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHJpc2luZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSd1cCc7fSk7CiAgICB2YXIgZmFsbGluZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSdkb3duJzt9KTsKICAgIHZhciBjdHg9Jyc7CiAgICBpZih2ZWw+MC4wNSYmcmlzaW5nTmFycy5sZW5ndGgpIGN0eD0nQ29uY2VudHJhdGVkIGFyb3VuZCA8ZW0+JytyaXNpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignPC9lbT4gYW5kIDxlbT4nKSsnPC9lbT4g4oCUIHRoZXNlIHNpZ25hbHMgYXJlIGdhaW5pbmcgbW9tZW50dW0gYW5kIG1heSBhdHRyYWN0IGJyb2FkZXIgYXR0ZW50aW9uLic7CiAgICBlbHNlIGlmKHZlbDwtMC4wNSYmZmFsbGluZ05hcnMubGVuZ3RoKSBjdHg9J1NpZ25hbHMgYXJvdW5kIDxlbT4nK2ZhbGxpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignPC9lbT4gYW5kIDxlbT4nKSsnPC9lbT4gYXJlIHJldHJlYXRpbmcg4oCUIHRoZSBkaXNjb3Vyc2UgY3ljbGUgYXBwZWFycyB0byBiZSBjb21wbGV0aW5nLic7CiAgICBlbHNlIGlmKHZlbD4wLjAyKSBjdHg9J1NpZ25hbHMgaW4gJytubSsnIGFyZSBidWlsZGluZyBhY3Jvc3MgbXVsdGlwbGUgbmFycmF0aXZlcyDigJQgbm8gc2luZ2xlIGRvbWluYW50IHRocmVhZCB5ZXQsIGJ1dCBtb21lbnR1bSBpcyBwcmVzZW50Lic7CiAgICBlbHNlIGlmKHZlbDwtMC4wMikgY3R4PSdTaWduYWwgYWN0aXZpdHkgaW4gJytubSsnIGlzIGVhc2luZyDigJQgYXR0ZW50aW9uIGFwcGVhcnMgdG8gYmUgc2hpZnRpbmcgdG93YXJkIG90aGVyIHJlZ2lvbmFsIHN0b3JpZXMuJzsKICAgIGVsc2UgY3R4PSdTaWduYWxzIGZyb20gJytubSsnIGhvbGRpbmcgc3RlYWR5IOKAlCBiZXR3ZWVuIGN5Y2xlcywgbm8gc3Ryb25nIGFjY2VsZXJhdGlvbiBvciByZXRyZWF0IGRldGVjdGVkLic7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ1NpZ25hbCB2ZWxvY2l0eSBmb3IgJytubSsnIOKAlCB3aGV0aGVyIGF0dGVudGlvbiBpcyBidWlsZGluZywgaG9sZGluZywgb3IgYmVnaW5uaW5nIHRvIHJldHJlYXQgZnJvbSB0aGUgY3VycmVudCBjeWNsZS4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrdmVsQ29sKycxNDtib3JkZXI6MXB4IHNvbGlkICcrdmVsQ29sKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3ZlbENvbCsnO21hcmdpbi1ib3R0b206NnB4Ij5TaWduYWwgbW9tZW50dW08L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo4cHg7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MzJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjonK3ZlbENvbCsnO2ZvbnQtd2VpZ2h0OjUwMCI+Jyt2ZWxEaXIrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC1zdHlsZTppdGFsaWM7bGluZS1oZWlnaHQ6MS41Ij4nK3ZlbERlc2NbdmVsRGlyXSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjttYXJnaW4tdG9wOjEwcHg7cGFkZGluZy10b3A6MTBweDtib3JkZXItdG9wOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIj4nK2N0eCsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlZlbG9jaXR5PC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIiBzdHlsZT0iZm9udC1zaXplOjE4cHg7Y29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCDOtDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrKGQuZGVsdGE+PTA/J3VwJzonZG4nKSsnIj4nKyhkLmRlbHRhPj0wPycrJzonJykrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAocmlzaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QWNjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICByaXNpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojZTA1YTI4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJykrCiAgICAgIChmYWxsaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RGVjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICBmYWxsaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoNTksMTg0LDIxNiwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojM2JiOGQ4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJyk7CiAgfQoKICBwYW5lbC5pbm5lckhUTUw9aGVhZGVyK2JvZHk7Cn0KCgpmdW5jdGlvbiB0b2dnbGVGYXYobm0pewogIGlmKEZBVlMuaGFzKG5tKSkgRkFWUy5kZWxldGUobm0pO2Vsc2UgRkFWUy5hZGQobm0pOwogIHJlbmRlclBhbmVsKFNFTCk7cmVuZGVyRmF2cygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdnMoKXsKICB2YXIgcm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXYtcm93Jyk7CiAgaWYoIUZBVlMuc2l6ZSl7cm93LmlubmVySFRNTD0nPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+JztyZXR1cm47fQogIHJvdy5pbm5lckhUTUw9QXJyYXkuZnJvbShGQVZTKS5tYXAoZnVuY3Rpb24obm0pewogICAgdmFyIGQ9ZyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8nI2UwNWEyOCc6JyMzYmI4ZDgnOwogICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZmF2LWNhcmQiIG9uY2xpY2s9InNlbGVjdF8oXCcnK25tKydcJykiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1oZWFkIj48c3BhbiBjbGFzcz0iZmMtbmFtZSI+JytubSsnPC9zcGFuPjxzcGFuIGNsYXNzPSJmYy1zYyI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPk5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj4yNGg8L3NwYW4+PHNwYW4gY2xhc3M9InYiIHN0eWxlPSJjb2xvcjonK2RDKyciPicrZFMrZC5kZWx0YSsnPC9zcGFuPjwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKGMpewogIGMuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7bGF5ZXI9Yy5kYXRhc2V0LmxheWVyO2FwcGx5TGF5ZXIoKTsKICB9KTsKfSk7CgpmdW5jdGlvbiB1cGRhdGVDbG9jaygpewogIHZhciBub3c9bmV3IERhdGUoKSxpc3Q9bmV3IERhdGUobm93LmdldFRpbWUoKStub3cuZ2V0VGltZXpvbmVPZmZzZXQoKSo2MDAwMCsxOTgwMDAwMCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Nsb2NrJykudGV4dENvbnRlbnQ9U3RyaW5nKGlzdC5nZXRIb3VycygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0TWludXRlcygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0U2Vjb25kcygpKS5wYWRTdGFydCgyLCcwJykrJyBJU1QnOwp9CnNldEludGVydmFsKHVwZGF0ZUNsb2NrLDEwMDApO3VwZGF0ZUNsb2NrKCk7CgpmdW5jdGlvbiBidWlsZFdJUlNpZ25hbHMoKXsKICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICB2YXIgc3JjPU9iamVjdC5rZXlzKExJVkUpLmxlbmd0aD9MSVZFOlNEOwogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKHNyYykuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4oa3ZbMV0uYXR0ZW50aW9ufHwwKT4zO30pOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIGVudHJpZXMuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KTsKCiAgdmFyIHVzZWROYXJyYXRpdmVzPVtdLHVzZWRTdGF0ZXM9W107CiAgdmFyIHNpZ25hbHM9W107CiAgZnVuY3Rpb24gdXNlZChuYXIsc3RhdGUpe3JldHVybiB1c2VkTmFycmF0aXZlcy5pbmRleE9mKG5hcik+PTB8fHVzZWRTdGF0ZXMuaW5kZXhPZihzdGF0ZSk+PTA7fQogIGZ1bmN0aW9uIHVzZShuYXIsc3RhdGUpe2lmKG5hcil1c2VkTmFycmF0aXZlcy5wdXNoKG5hcik7aWYoc3RhdGUpdXNlZFN0YXRlcy5wdXNoKHN0YXRlKTt9CgogIC8vIDEuIERvbWluYW50IHNpZ25hbCDigJQgZGlyZWN0LCBncm91bmRlZAogIHZhciB0b3A9ZW50cmllc1swXTsKICBpZih0b3ApewogICAgdmFyIG5hcj10b3BbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwncG9saXRpY2FsIGFjdGl2aXR5JzsKICAgIHZhciBlbW89dG9wWzFdLmRvbWluYW50X2Vtb3Rpb247CiAgICB2YXIgY29sPWVtbz9wYWxbZW1vXTondmFyKC0tYWNjZW50KSc7CiAgICB2YXIgdmVsPXRvcFsxXS52ZWxvY2l0eXx8MDsKICAgIHZhciB0YWlsPXZlbD4wLjA4PycsIGFuZCB0aGUgc2lnbmFsIGlzIHN0aWxsIGJ1aWxkaW5nJzp2ZWw8LTAuMDQ/JywgdGhvdWdoIG1vbWVudHVtIGlzIGJlZ2lubmluZyB0byBlYXNlJzonJzsKICAgIHZhciBlbW9DdHg9e2FuZ2VyOicg4oCUIHdpdGggZnJ1c3RyYXRpb24gYXMgdGhlIHByZXZhaWxpbmcgdG9uZScsYW54aWV0eTonIOKAlCB1bmRlcmN1cnJlbnQgb2YgYW54aWV0eSBydW5uaW5nIHRocm91Z2ggc2lnbmFscycsZmVhcjonIOKAlCBzaWduYWxzIGNhcnJ5aW5nIGFuIGVkZ2Ugb2YgYXBwcmVoZW5zaW9uJyxob3BlOicg4oCUIGEgcmVsYXRpdmVseSBvcHRpbWlzdGljIHJlZ2lzdGVyJyxwcmlkZTonJ307CiAgICBzaWduYWxzLnB1c2goe2NvbDpjb2wsdGFnOidoaWdoZXN0IHNpZ25hbCcsbG9jOnRvcFswXSwKICAgICAgdGV4dDonPHN0cm9uZz4nK3RvcFswXSsnPC9zdHJvbmc+IGlzIGdlbmVyYXRpbmcgdGhlIG1vc3QgYXR0ZW50aW9uIG5hdGlvbmFsbHkgYXJvdW5kIDxlbT4nK25hcisnPC9lbT4nK3RhaWwrKGVtbz9lbW9DdHhbZW1vXXx8Jyc6JycpLGRlbGF5OjB9KTsKICAgIHVzZShuYXIsdG9wWzBdKTsKICB9CgogIC8vIDIuIEVhcmx5IG1vdmVyIOKAlCBzb21ldGhpbmcgYnVpbGRpbmcgYmVmb3JlIGl0IGdvZXMgbmF0aW9uYWwKICB2YXIgZWFybHk9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3YpewogICAgcmV0dXJuKGt2WzFdLnZlbG9jaXR5fHwwKT4wLjA1JiYoa3ZbMV0uYXR0ZW50aW9ufHwwKTwzNSYmIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKTsKICB9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0udmVsb2NpdHl8fDApLShhWzFdLnZlbG9jaXR5fHwwKTt9KVswXTsKICBpZihlYXJseSl7CiAgICB2YXIgZU5hcj1lYXJseVsxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdsb2NhbCBkZXZlbG9wbWVudHMnOwogICAgdmFyIGVFbW89ZWFybHlbMV0uZG9taW5hbnRfZW1vdGlvbjsKICAgIHNpZ25hbHMucHVzaCh7Y29sOmVFbW8/cGFsW2VFbW9dOicjZTA3ODIwJyx0YWc6J2J1aWxkaW5nIHNpZ25hbCcsbG9jOmVhcmx5WzBdLAogICAgICB0ZXh0Oic8ZW0+JytlTmFyLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2VOYXIuc2xpY2UoMSkrJzwvZW0+IHNpZ25hbHMgYXJlIGdhaW5pbmcgdHJhY3Rpb24gaW4gPHN0cm9uZz4nK2Vhcmx5WzBdKyc8L3N0cm9uZz4g4oCUIGVhcmxpZXIgdGhhbiBtb3N0IGN5Y2xlcyBhdCB0aGlzIHN0YWdlJyxkZWxheToxNjB9KTsKICAgIHVzZShlTmFyLGVhcmx5WzBdKTsKICB9CgogIC8vIDMuIEVtb3Rpb25hbCBjb25jZW50cmF0aW9uIOKAlCB0b25lIHJlYWQsIG5vdCBhIGhlYWRsaW5lCiAgdmFyIGVtb0ZvY3VzPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybiBrdlsxXS5kb21pbmFudF9lbW90aW9uJiYhdXNlZChrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUsa3ZbMF0pJiYoa3ZbMV0uYXR0ZW50aW9ufHwwKT40OwogIH0pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYoZW1vRm9jdXMpewogICAgdmFyIGVmTmFyPWVtb0ZvY3VzWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J2RldmVsb3BtZW50cyc7CiAgICB2YXIgZWZFbW89ZW1vRm9jdXNbMV0uZG9taW5hbnRfZW1vdGlvbjsKICAgIHZhciBlZkNvbD1wYWxbZWZFbW9dfHwnIzU1NjY3Nyc7CiAgICB2YXIgZWZSZWFkPXsKICAgICAgYW5nZXI6J1NpZ25hbHMgZnJvbSA8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IGNhcnJ5IGEgbm90aWNlYWJseSBmcnVzdHJhdGVkIHRvbmUg4oCUIHdvcnRoIHdhdGNoaW5nJywKICAgICAgYW54aWV0eTonVGhlcmUgaXMgYSBxdWlldCB1bmVhc2UgaW4gPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiDigJQgc2lnbmFscyBzdWdnZXN0IHRoaXMgaGFzIG5vdCBwZWFrZWQgeWV0JywKICAgICAgZmVhcjonU2lnbmFscyBpbiA8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IGNhcnJ5IGFuIGVkZ2Ug4oCUIHRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgaXMgYXBwcmVoZW5zaXZlJywKICAgICAgaG9wZTonU29tZXdoYXQgdW51c3VhbGx5LCA8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBpcyBzaG93aW5nIGFuIG9wdGltaXN0aWMgc2lnbmFsIHJlZ2lzdGVyIGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4nLAogICAgICBwcmlkZTonPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gc2lnbmFscyBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IGhhdmUgYSBzdHJvbmcgaWRlbnRpdHkgdG9uZSDigJQgbG9jYWxseSBjb25jZW50cmF0ZWQnCiAgICB9OwogICAgc2lnbmFscy5wdXNoKHtjb2w6ZWZDb2wsdGFnOidlbW90aW9uYWwgdG9uZScsbG9jOmVtb0ZvY3VzWzBdLAogICAgICB0ZXh0OmVmUmVhZFtlZkVtb118fCdTaWduYWxzIGZyb20gPHN0cm9uZz4nK2Vtb0ZvY3VzWzBdKyc8L3N0cm9uZz4gYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiBhcmUgd29ydGggd2F0Y2hpbmcnLGRlbGF5OjMyMH0pOwogICAgdXNlKGVmTmFyLGVtb0ZvY3VzWzBdKTsKICB9CgogIC8vIDQuIENvb2xpbmcg4oCUIGN5Y2xlIGNvbXBsZXRpbmcKICB2YXIgY29vbGluZz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7CiAgICByZXR1cm4oa3ZbMV0udmVsb2NpdHl8fDApPC0wLjA0JiYhdXNlZChrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUsa3ZbMF0pJiYoa3ZbMV0uYXR0ZW50aW9ufHwwKT41OwogIH0pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYVsxXS52ZWxvY2l0eXx8MCktKGJbMV0udmVsb2NpdHl8fDApO30pWzBdOwogIGlmKGNvb2xpbmcpewogICAgdmFyIGNOYXI9Y29vbGluZ1sxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdyZWNlbnQgZm9jdXMnOwogICAgc2lnbmFscy5wdXNoKHtjb2w6JyMzYmI4ZDgnLHRhZzonc2lnbmFsIHJldHJlYXRpbmcnLGxvYzpjb29saW5nWzBdLAogICAgICB0ZXh0Oic8ZW0+JytjTmFyLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2NOYXIuc2xpY2UoMSkrJzwvZW0+IGluIDxzdHJvbmc+Jytjb29saW5nWzBdKyc8L3N0cm9uZz4gYXBwZWFycyB0byBiZSBsb3Npbmcgc2lnbmFsIHN0cmVuZ3RoIOKAlCB0aGUgY3ljbGUgbWF5IGJlIHJ1bm5pbmcgaXRzIGNvdXJzZScsZGVsYXk6NDYwfSk7CiAgICB1c2UoY05hcixjb29saW5nWzBdKTsKICB9CgogIC8vIDUuIE5vcnRoZWFzdCDigJQgc2ltcGx5IG9ic2VydmF0aW9uYWwsIG5vIGRyYW1hdGlzYXRpb24KICB2YXIgbmVTdGF0ZXM9WydNYW5pcHVyJywnQXNzYW0nLCdOYWdhbGFuZCcsJ01pem9yYW0nLCdNZWdoYWxheWEnLCdBcnVuYWNoYWwgUHJhZGVzaCcsJ1RyaXB1cmEnXTsKICB2YXIgbmVBY3RpdmU9bmVTdGF0ZXMuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzcmNbc10mJihzcmNbc10uYXR0ZW50aW9ufHwwKT4yJiZ1c2VkU3RhdGVzLmluZGV4T2Yocyk8MDt9KTsKICBpZihuZUFjdGl2ZS5sZW5ndGg+PTIpewogICAgdmFyIG5lTmFyPShzcmNbbmVBY3RpdmVbMF1dJiZzcmNbbmVBY3RpdmVbMF1dLmRvbWluYW50X25hcnJhdGl2ZSl8fCdyZWdpb25hbCBkZXZlbG9wbWVudHMnOwogICAgc2lnbmFscy5wdXNoKHtjb2w6J3JnYmEoMTYwLDE5MCwyMzAsMC40NSknLHRhZzoncmVnaW9uYWwgc2lnbmFsJyxsb2M6J05vcnRoZWFzdCcsCiAgICAgIHRleHQ6bmVBY3RpdmUubGVuZ3RoKycgbm9ydGhlYXN0ZXJuIHN0YXRlcyBhcmUgc2hvd2luZyBjb25jZW50cmF0ZWQgc2lnbmFscyBhcm91bmQgPGVtPicrbmVOYXIrJzwvZW0+IOKAlCBhIHBhdHRlcm4gdGhhdCB0ZW5kcyB0byBwcmVjZWRlIHdpZGVyIG5hdGlvbmFsIGF0dGVudGlvbicsZGVsYXk6NTgwfSk7CiAgfQoKICBpZighc2lnbmFscy5sZW5ndGgpIHJldHVybjsKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3dpci1zaWduYWxzJyk7CiAgaWYoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MPXNpZ25hbHMubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsIiBzdHlsZT0iYW5pbWF0aW9uLWRlbGF5Oicrcy5kZWxheSsnbXMiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLWJhciIgc3R5bGU9ImJhY2tncm91bmQ6JytzLmNvbCsnIj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1jb250ZW50Ij4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLXRleHQiPicrcy50ZXh0Kyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLW1ldGEiPicrCiAgICAgICAgICAnPHNwYW4gY2xhc3M9Indpci1zaWduYWwtdGFnIiBzdHlsZT0iY29sb3I6JytzLmNvbCsnIj4nK3MudGFnKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBjbGFzcz0id2lyLXNpZ25hbC1sb2MiPicrcy5sb2MrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CgoKCnZhciBFTU9fQ09MT1JTPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKdmFyIEVNT19CRz17YW54aWV0eToncmdiYSgxMzYsNjgsMjA0LDAuMSknLGFuZ2VyOidyZ2JhKDIyMSwzNCw2OCwwLjEpJyxob3BlOidyZ2JhKDUxLDIwNCwxMDIsMC4xKScscHJpZGU6J3JnYmEoNTEsMTcwLDIwNCwwLjEpJyxmZWFyOidyZ2JhKDIwNCwxMzYsNTEsMC4xKSd9OwoKCmZ1bmN0aW9uIHJlbmRlck5hckNhcmQobixkaXIpewogIHZhciBjb2w9ZGlyPT09J3Jpc2luZyc/JyNlMDVhMjgnOicjM2JiOGQ4JzsKICB2YXIgYXJyb3c9ZGlyPT09J3Jpc2luZyc/J+KGkSc6J+KGkyc7CiAgdmFyIGxibD1kaXI9PT0ncmlzaW5nJz8nUklTSU5HJzonRkFESU5HJzsKICB2YXIgdz1NYXRoLm1pbigxMDAsKG4uc2lnbmFsX3NoYXJlfHwwKSozKTsKICByZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj4nKwogICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjRweDsiPicrCiAgICAgICc8c3BhbiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjonK2NvbCsnO2xldHRlci1zcGFjaW5nOjAuMDhlbSI+JythcnJvdysnICcrbGJsKyc8L3NwYW4+JysKICAgICc8L2Rpdj4nKwogICAgKG4uc3RhdGVzJiZuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JysoZGlyPT09J3Jpc2luZyc/J0RyaXZlbiBieTogJzonV2FzIGFjdGl2ZSBpbjogJykrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsKICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrdysnJTtiYWNrZ3JvdW5kOicrY29sKyc7b3BhY2l0eTowLjgiPjwvZGl2PjwvZGl2PicrCiAgJzwvZGl2Pic7Cn0KCmZ1bmN0aW9uIHNldEFjdGl2ZVRhYihidG4pewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWIsLnNoaWZ0LXRhYicpLmZvckVhY2goZnVuY3Rpb24oYil7Yi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICBidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7Cn0KCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CiAgLy8gUGVyaW9kIGFmZmVjdHMgaG93IHdlIHdlaWdodCBzaWduYWxzIOKAlCByZWNlbnQgdnMgb2xkZXIKICAvLyAzbSA9IGxhc3QgOTAgZGF5cyB3ZWlnaHQsIDZtID0gYmFsYW5jZSwgMXkgPSBhbGwgc2lnbmFscyBlcXVhbGx5CiAgdmFyIHBlcmlvZExhYmVsPXsnM20nOiczIG1vbnRocycsJzZtJzonNiBtb250aHMnLCcxeSc6JzEgeWVhcid9W3BlcmlvZF18fCczIG1vbnRocyc7CiAgaWYoIWVsKSByZXR1cm47CiAgLy8gQnVpbGQgbmFycmF0aXZlIHNoaWZ0cyBmcm9tIFNEIGRhdGEKICB2YXIgbmM9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBpZighbmNbbi5uYW1lXSkgbmNbbi5uYW1lXT17dXA6MCxkb3duOjAsc3RhdGVzOltdfTsKICAgICAgaWYobi5kaXI9PT0ndXAnKSBuY1tuLm5hbWVdLnVwKz1uLnZhbDsKICAgICAgZWxzZSBpZihuLmRpcj09PSdkb3duJykgbmNbbi5uYW1lXS5kb3duKz1uLnZhbDsKICAgICAgbmNbbi5uYW1lXS5zdGF0ZXMucHVzaCh7c3RhdGU6cy5uYW1lfHwnJyx2YWw6bi52YWwsZGlyOm4uZGlyfSk7CiAgICB9KTsKICB9KTsKICAvLyBTb3J0IGFsbCBuYXJyYXRpdmVzIGJ5IHRvdGFsIHNpZ25hbCDigJQgcmlzaW5nID0gdG9wLCBmYWRpbmcgPSBib3R0b20KICB2YXIgYWxsTmFycz1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnVwLWFbMV0udXA7fSk7CiAgdmFyIHJpc2luZz1hbGxOYXJzLnNsaWNlKDAsMyk7CiAgdmFyIGZhZGluZz1hbGxOYXJzLnNsaWNlKC0zKS5yZXZlcnNlKCk7IC8vIGxlYXN0IGFjdGl2ZSA9IGZhZGluZwogIHZhciBwYWlycz1NYXRoLm1heChyaXNpbmcubGVuZ3RoLGZhZGluZy5sZW5ndGgpOwogIGlmKCFwYWlycyl7ZWwuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+Q29sbGVjdGluZyBzaWduYWwgZGF0YS4uLjwvZGl2Pic7cmV0dXJuO30KICB2YXIgY2FyZHM9W107CiAgZm9yKHZhciBpPTA7aTxwYWlycztpKyspewogICAgdmFyIGY9ZmFkaW5nW2ldLHI9cmlzaW5nW2ldOwogICAgaWYoIWYmJiFyKSBjb250aW51ZTsKICAgIHZhciBmTmFtZT1mP2ZbMF06J+KAlCc7IHZhciByTmFtZT1yP3JbMF06J+KAlCc7CiAgICB2YXIgZlN1Yj1mPyhmWzFdLnN0YXRlcy5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGEudmFsLWIudmFsO30pLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKHMpe3JldHVybiBzLnN0YXRlLnNwbGl0KCcgJylbMF07fSkuam9pbignLCAnKXx8JycpOicnOwogICAgdmFyIHJTdWI9cj8oclsxXS5zdGF0ZXMuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzLmRpcj09PSd1cCc7fSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiLnZhbC1hLnZhbDt9KS5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihzKXtyZXR1cm4gcy5zdGF0ZS5zcGxpdCgnICcpWzBdO30pLmpvaW4oJywgJyl8fCcnKTonJzsKICAgIGNhcmRzLnB1c2goCiAgICAgICc8ZGl2IGNsYXNzPSJzaGlmdC1jYXJkIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzaGlmdC1jYXJkLWZhZGluZyI+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzYy1sYmwiPkZBRElORzwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic2hpZnQtY2FyZC1uYW1lIj4nK2ZOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2ZOYW1lLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICAgKGZTdWI/JzxkaXYgY2xhc3M9InNoaWZ0LWNhcmQtc3ViIj4nK2ZTdWIrJzwvZGl2Pic6JycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzaGlmdC1hcnJvdyI+4oaSPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic2hpZnQtY2FyZC1yaXNpbmciPicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic2MtbGJsIj5SSVNJTkc8L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InNoaWZ0LWNhcmQtbmFtZSI+JytyTmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyTmFtZS5zbGljZSgxKSsnPC9kaXY+JysKICAgICAgICAgIChyU3ViPyc8ZGl2IGNsYXNzPSJzaGlmdC1jYXJkLXN1YiI+JytyU3ViKyc8L2Rpdj4nOicnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicKICAgICk7CiAgfQogIGVsLmlubmVySFRNTD1jYXJkcy5qb2luKCcnKTsKfQoKLy8gSU5JVCDigJQgd2FpdCBmb3IgRE9NCi8vIGkgYnV0dG9uIHRvb2x0aXAg4oCUIHVzZXMgZml4ZWQgcG9zaXRpb25pbmcgc28gaXQncyBuZXZlciBjbGlwcGVkCihmdW5jdGlvbigpewogIHZhciB0aXA9bnVsbDsKICBmdW5jdGlvbiBzaG93VGlwKGUpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICB2YXIgdHh0PXRoaXMuZ2V0QXR0cmlidXRlKCdkYXRhLXRpcCcpOwogICAgaWYoIXR4dHx8IXRpcCkgcmV0dXJuOwogICAgdGlwLnRleHRDb250ZW50PXR4dDsKICAgIHRpcC5jbGFzc0xpc3QuYWRkKCd2aXNpYmxlJyk7CiAgICB2YXIgcmVjdD10aGlzLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgdmFyIHR3PTI0MDsKICAgIHZhciBsZWZ0PU1hdGgubWluKHJlY3QubGVmdCx3aW5kb3cuaW5uZXJXaWR0aC10dy0xMCk7CiAgICB0aXAuc3R5bGUubGVmdD1sZWZ0KydweCc7CiAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC0xMC10aXAub2Zmc2V0SGVpZ2h0fHxyZWN0LnRvcC04MCkrJ3B4JzsKICAgIC8vIFJlcG9zaXRpb24gYWZ0ZXIgcmVuZGVyCiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZnVuY3Rpb24oKXsKICAgICAgdGlwLnN0eWxlLnRvcD0ocmVjdC50b3AtdGlwLm9mZnNldEhlaWdodC04KSsncHgnOwogICAgfSk7CiAgfQogIGZ1bmN0aW9uIGhpZGVUaXAoKXsKICAgIGlmKCF0aXApe3RpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbHRhYi10b29sdGlwJyk7fQogICAgaWYodGlwKSB0aXAuY2xhc3NMaXN0LnJlbW92ZSgndmlzaWJsZScpOwogIH0KICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW92ZXInLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgc2hvd1RpcC5jYWxsKGUudGFyZ2V0LGUpOwogIH0pOwogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3V0JyxmdW5jdGlvbihlKXsKICAgIGlmKGUudGFyZ2V0LmNsYXNzTGlzdC5jb250YWlucygnbHRhYi1pbmZvJykpIGhpZGVUaXAoKTsKICB9KTsKfSkoKTsKCmZ1bmN0aW9uIGRpc21pc3NMb2FkZXIoKXsKICB2YXIgbGRyPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhcHAtbG9hZGVyJyk7CiAgaWYoIWxkcikgcmV0dXJuOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYobGRyKWxkci5zdHlsZS5kaXNwbGF5PSdub25lJzt9LDkwMCk7Cn0KCgpmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7CiAgdmFyIGxkcj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpOwogIGlmKCFsZHIpIHJldHVybjsKICBsZHIuc3R5bGUub3BhY2l0eT0nMCc7CiAgbGRyLnN0eWxlLnZpc2liaWxpdHk9J2hpZGRlbic7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2lmKGxkcikgbGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnO30sOTAwKTsKfQpmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7CiAgdmFyIGxkcj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpOwogIGlmKCFsZHIpIHJldHVybjsKICBsZHIuc3R5bGUub3BhY2l0eT0nMCc7CiAgbGRyLnN0eWxlLnZpc2liaWxpdHk9J2hpZGRlbic7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2lmKGxkcikgbGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnO30sOTAwKTsKfQoKCmZ1bmN0aW9uIGluaXQoKXsKICByZW5kZXJTdHJpcCgnM20nKTsKCiAgLy8gTG9hZCBtYXAgd2l0aCByZXRyeQogIHZhciBtYXBBdHRlbXB0cz0wOwogIGZ1bmN0aW9uIHRyeUxvYWRNYXAoKXsKICAgIGlmKHR5cGVvZiB0b3BvanNvbj09PSd1bmRlZmluZWQnKXsKICAgICAgaWYobWFwQXR0ZW1wdHMrKzwxMCl7c2V0VGltZW91dCh0cnlMb2FkTWFwLDMwMCk7fQogICAgICByZXR1cm47CiAgICB9CiAgICBsb2FkTWFwKCk7CiAgfQogIHRyeUxvYWRNYXAoKTsKCiAgLy8gTG9hZCBmdWxsIGNhY2hlZCBzbmFwc2hvdCBpbW1lZGlhdGVseSBmb3IgaW5zdGFudCBkYXRhCiAgZmV0Y2hGdWxsU25hcHNob3QoKS50aGVuKGZ1bmN0aW9uKG9rKXsKICAgIGlmKG9rKXsKICAgICAgcmVuZGVyTW9tZW50dW0oKTsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe3N0YXJ0UG9sbGluZygpO30sMTAwMCk7CiAgICB9IGVsc2UgewogICAgICBzdGFydFBvbGxpbmcoKTsKICAgIH0KICAgIGRpc21pc3NMb2FkZXIoKTsKICB9KTsKCiAgLy8gRGlzbWlzcyBsb2FkZXIgYWZ0ZXIgbWF4IDRzIHJlZ2FyZGxlc3MKICBzZXRUaW1lb3V0KGRpc21pc3NMb2FkZXIsIDQwMDApOwoKICAvLyBSZXRyeSBtYXAgaWYgc3RpbGwgZW1wdHkKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSwzMDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSw2MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw1MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDgwMDApOwp9CmlmKGRvY3VtZW50LnJlYWR5U3RhdGU9PT0nbG9hZGluZycpewogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ0RPTUNvbnRlbnRMb2FkZWQnLCBpbml0KTsKfSBlbHNlIHsKICAvLyBBbHJlYWR5IGxvYWRlZCDigJQgYnV0IHdhaXQgb25lIHRpY2sgdG8gZW5zdXJlIGFsbCBzY3JpcHRzIHBhcnNlZAogIHNldFRpbWVvdXQoaW5pdCwgMCk7Cn0KCgpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgLy8gQXV0by1zZWxlY3QgaG90dGVzdCBzdGF0ZSBmcm9tIExJVkUgZGF0YQogIHZhciBzcmM9T2JqZWN0LmtleXMoTElWRSkubGVuZ3RoP0xJVkU6U0Q7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhzcmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKHRvcCl7CiAgICB2YXIgZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJyt0b3BbMF0rJyJdJyk7CiAgICBpZihlbCkgc2VsZWN0Xyh0b3BbMF0pOwogIH0KfSwzMDAwKTsKc2V0VGltZW91dChyZW5kZXJGYXZzLDI0MDApOwo8L3NjcmlwdD4KPC9ib2R5Ly8gRmFzdGVzdCBjb29saW5nIOKAlCBsb3dlc3QgdmVsb2NpdHkgc3RhdGUgKGFsd2F5cyBzaG93cyBzb21ldGhpbmcpCiAgdmFyIGNvb2xpbmdTb3J0ZWQ9ZW50cmllcy5zbGljZSgpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYVsxXS52ZWxvY2l0eXx8MCktKGJbMV0udmVsb2NpdHl8fDApO30pOwogIHZhciBjb29saW5nMj1jb29saW5nU29ydGVkWzBdOwogIGlmKGNvb2xpbmcyKXsKICAgIHNldFRleHQoJ3NjLWNvb2wtdmFsJywgY29vbGluZzJbMF0pOwogICAgdmFyIGNWZWw9bm9ybVYoY29vbGluZzJbMV0udmVsb2NpdHl8fDApOwogICAgdmFyIGNOYXI9Y29vbGluZzJbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnJzsKICAgIHZhciBjTGFiZWw9Y1ZlbDwtMC4xPydyZXRyZWF0aW5nJzpjVmVsPDAuMz8nc2xvdyBtb21lbnR1bSc6J2xlYXN0IGFjdGl2ZSc7CiAgICBzZXRUZXh0KCdzYy1jb29sLXN1YicsIGNOYXI/Y05hcisnIMK3ICcrY0xhYmVsOmNMYWJlbCk7CiAgICAoZnVuY3Rpb24oKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2MtY29vbC10aXAnKTtpZighdCkgcmV0dXJuOwogICAgdmFyIG5hcnM9KGNvb2xpbmcyWzFdLm5hcnJhdGl2ZXN8fFtdKS5zbGljZSgwLDMpOwogICAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+TG93ZXN0IG1vbWVudHVtOiAnK2Nvb2xpbmcyWzBdKyc8L2Rpdj4nK25hcnMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ic2MtdGlwLXJvdyI+wrcgJytuLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpKyc8L2Rpdj4nO30pLmpvaW4oJycpO30pKCk7CiAgICB2YXIgY29vbFRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2MtY29vbC10aXAnKTsKICAgIGlmKGNvb2xUaXApewogICAgICB2YXIgY29vbE5hcnM9KGNvb2xpbmcyWzFdLm5hcnJhdGl2ZXN8fFtdKS5zbGljZSgwLDMpOwogICAgICBjb29sVGlwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2MtdGlwLXRpdGxlIj5Mb3dlc3QgbW9tZW50dW06ICcrY29vbGluZzJbMF0rJzwvZGl2PicrCiAgICAgICAgY29vbE5hcnMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ic2MtdGlwLXJvdyI+wrcgJytuLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpKyc8L2Rpdj4nO30pLmpvaW4oJycpOwogICAgfQogIH0+"

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
