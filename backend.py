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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQouc2lnLWhvdmVyLXdyYXB7cG9zaXRpb246cmVsYXRpdmU7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtjdXJzb3I6ZGVmYXVsdH0KLnNpZy1ob3Zlci10aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7dG9wOmNhbGMoMTAwJSArIDEwcHgpO3JpZ2h0OjA7CiAgYmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45Nyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxNHB4O3doaXRlLXNwYWNlOm5vd3JhcDsKICBwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt2aXNpYmlsaXR5OmhpZGRlbjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xOHMsdmlzaWJpbGl0eSAwLjE4czsKICB6LWluZGV4Ojk5OTk7Cn0KLnNpZy1ob3Zlci13cmFwOmhvdmVyIC5zaWctaG92ZXItdGlwe29wYWNpdHk6MTt2aXNpYmlsaXR5OnZpc2libGV9Ci5zaWctaG92ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NXB4O29wYWNpdHk6MC43fQouc2lnLWhvdmVyLXNvdXJjZXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA0ZW19Ci5saXZlLWluZGljYXRvcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo3cHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtOwp9Ci5saXZlLWRvdHt3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiM0YWRlODA7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoNzQsMjIyLDEyOCwwLjcpO2FuaW1hdGlvbjpsZCAyLjVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGxkezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjM1O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KLmNsb2Nre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNGVtfQoKLyogSEVSTyAqLwouaGVyb3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgcGFkZGluZzo3MnB4IDM2cHggMDsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87Cn0KLmhlcm8tZXllYnJvd3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMzJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MjRweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4fQouaGVyby1leWVicm93OjpiZWZvcmV7Y29udGVudDonJzt3aWR0aDoxNnB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1mYWludCk7b3BhY2l0eTowLjV9Ci5oZXJvLWJyYW5kLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOm5vcm1hbDtmb250LXNpemU6Y2xhbXAoMzZweCw0LjJ2dyw2NHB4KTtsaW5lLWhlaWdodDoxO2xldHRlci1zcGFjaW5nOi0wLjAzZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW46MH0KLmhlcm8tYnJhbmQtbmFtZSBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjojZThjNGEwO2FuaW1hdGlvbjpwdWxzZU5hbWVHbG93IDVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlTmFtZUdsb3d7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC43Mn19Ci5oZXJvLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNXB4LDEuNXZ3LDIwcHgpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbjowIDAgMTJweCAwO21heC13aWR0aDo0ODBweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLWRlc2N7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEzcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjY7bWF4LXdpZHRoOjQwMHB4O21hcmdpbjowIDAgNnB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1zdWItbGluZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNik7bWFyZ2luOjAgMCAyMHB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1wdWxzZS1zaWduYWx7cG9zaXRpb246cmVsYXRpdmU7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtmbGV4LXNocmluazowfQouaHBzLWNvcmV7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuOTthbmltYXRpb246aHBzQ29yZSA0cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBocHNDb3JlezAlLDEwMCV7b3BhY2l0eTowLjk7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNDt0cmFuc2Zvcm06c2NhbGUoMC43NSl9fQouaHBzLXJpbmd7cG9zaXRpb246YWJzb2x1dGU7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1hY2NlbnQpO2FuaW1hdGlvbjpocHNSaW5nIDRzIGVhc2Utb3V0IGluZmluaXRlfQouaHBzLXJpbmcucjF7aW5zZXQ6MXB4O2FuaW1hdGlvbi1kZWxheTowc30uaHBzLXJpbmcucjJ7aW5zZXQ6LTNweDthbmltYXRpb24tZGVsYXk6MS40cztib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zNSl9CkBrZXlmcmFtZXMgaHBzUmluZ3swJXtvcGFjaXR5OjAuNjt0cmFuc2Zvcm06c2NhbGUoMC43KX0xMDAle29wYWNpdHk6MDt0cmFuc2Zvcm06c2NhbGUoMS42KX19CgovKiBTSUdOQVRVUkUgSU5TSUdIVCAqLwouc2lnbmF0dXJlLWluc2lnaHR7CiAgbWFyZ2luLXRvcDowOwogIHBhZGRpbmc6MTRweCAyMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNikgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIG1heC13aWR0aDo5MDBweDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47Cn0KLnNpZ25hdHVyZS1pbnNpZ2h0OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgdmFyKC0tYWNjZW50KSwgdHJhbnNwYXJlbnQpOwp9Ci5zaS1sYWJlbHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouc2ktdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE0cHgsMS40dncsMThweCk7CiAgZm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluayk7bGluZS1oZWlnaHQ6MS41O2xldHRlci1zcGFjaW5nOi0wLjAxZW07Cn0KLnNpLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tYWNjZW50KX0KLnNpLXN1YnsKICBtYXJnaW4tdG9wOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTsKICBsZXR0ZXItc3BhY2luZzowLjA0ZW07ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDtmbGV4LXdyYXA6d3JhcDsKfQouc2ktdGFnewogIHBhZGRpbmc6MnB4IDhweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTsKICBmb250LXNpemU6OS41cHg7Cn0KCi8qIE5BUlJBVElWRSBTVFJJUCAqLwoKLnN0cmlwLXRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjRweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQouc3RyaXAtdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xMil9Ci5zdHJpcC10YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbHsKICBmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1iZzEpO3BhZGRpbmc6MDsKfQouc3RyaXAtY29sLWhlYWR7CiAgcGFkZGluZzoxMHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwp9Ci5zdHJpcC1jb2wtaGVhZC5mYWRle2NvbG9yOnZhcigtLWZhbGwpfQouc3RyaXAtY29sLWhlYWQucmlzZTJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5zdHJpcC1jb2wtaGVhZC5zaGlmdHtjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sLWJvZHl7cGFkZGluZzoxMnB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OHB4fQouc3RyaXAtaXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjhweDsKfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5zdHJpcC1ub3Rle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5zdHJpcC1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTtmb250LXNpemU6MTRweDtmbGV4LXNocmluazowfQoKLyogTUFJTiBMQVlPVVQgKi8KLm1haW57CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM2MHB4OwogIGdhcDoxNHB4OwogIG1pbi13aWR0aDowOwp9CgovKiBNQVAgKi8KLm1hcC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBvdmVyZmxvdzpoaWRkZW47cG9zaXRpb246cmVsYXRpdmU7Cn0KLm1hcC1jYXJkOjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA3MCUgNTAlIGF0IDM1JSAwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNSkgMCUsIHRyYW5zcGFyZW50IDYwJSksCiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA1MCUgNDAlIGF0IDgwJSAxMDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMCUsIHRyYW5zcGFyZW50IDYwJSk7Cn0KLm1hcC10b3B7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47CiAgcGFkZGluZzoxMnB4IDE4cHggMDsKfQoubWFwLXRpdGxlLWJsb2NrIC5tdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE3cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtdGl0bGUtYmxvY2sgLm1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5sZWdlbmQtYmFyewogIGhlaWdodDozcHg7d2lkdGg6ODBweDtib3JkZXItcmFkaXVzOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCwjMGUyMDM1LCMxYTU1ODAgMjUlLCM4YTVjMTggNTUlLCNjMDM4MWEgODAlLCNlMDEwMjApOwp9Ci5sYXllci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsKICBwYWRkaW5nOjEwcHggMjBweCA2cHg7Cn0KLmxheWVyLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLmx0YWJze2Rpc3BsYXk6ZmxleDtnYXA6M3B4fQoubHRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6M3B4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5sdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wOCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5sdGFie2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo1cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6dmlzaWJsZX0KLmx0YWItaW5mb3t3aWR0aDoxM3B4O2hlaWdodDoxM3B4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjIpO2ZvbnQtc2l6ZTo4cHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zdHlsZTppdGFsaWM7Zm9udC13ZWlnaHQ6NjAwO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4zNSk7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjdXJzb3I6aGVscDtmbGV4LXNocmluazowO3RyYW5zaXRpb246YWxsIDAuMTVzO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTAwfQoubHRhYi1pbmZvOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2NlbnQpO2NvbG9yOnZhcigtLWFjY2VudCl9CiNsdGFiLXRvb2x0aXB7cG9zaXRpb246Zml4ZWQ7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMTIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjt3aWR0aDoyMzBweDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNik7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzO3otaW5kZXg6OTk5OTk7ZGlzcGxheTpub25lfQojbHRhYi10b29sdGlwLnZpc2libGV7b3BhY2l0eToxO2Rpc3BsYXk6YmxvY2t9Ci5sdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9CgoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmc6MTJweCAxNnB4IDE2cHg7Cn0KLm1hcC1pbm5lcntwb3NpdGlvbjpyZWxhdGl2ZTthc3BlY3QtcmF0aW86MS8xO3dpZHRoOjEwMCV9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrO292ZXJmbG93OnZpc2libGV9CgovKiBtYXAgc3RhdGUgc3R5bGVzICovCiNpbmRpYS1tYXAgLnN0YXRlewogIGN1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246ZmlsdGVyIDAuMjVzIGVhc2UsIHN0cm9rZS13aWR0aCAwLjJzIGVhc2UsIHN0cm9rZSAwLjJzIGVhc2U7Cn0KI2luZGlhLW1hcCAuc3RhdGU6aG92ZXJ7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC43KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxcHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjI1KSBkcm9wLXNoYWRvdygwIDAgMTBweCByZ2JhKDI1NSwyNTUsMjU1LDAuMikpOwp9CiNpbmRpYS1tYXAgLnN0YXRlLnNlbGVjdGVkewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuOSkgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MS40cHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjM1KSBkcm9wLXNoYWRvdygwIDAgMTZweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpOwp9CgovKiBhbmltYXRlZCBwdWxzZSByaW5ncyAqLwoucHVsc2UtcmluZ3tmaWxsOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZX0KLnB1bHNlLXJpbmcucDF7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5wdWxzZS1yaW5nLnAye2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IDAuOXMgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHJ7CiAgMCV7cjo0O29wYWNpdHk6MC43O3N0cm9rZS13aWR0aDoxLjJ9CiAgMTAwJXtyOjI2O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4yfQp9CgovKiBhdG1vc3BoZXJpYyBnbG93IGJlaGluZCBob3Qgc3RhdGVzICovCi5zdGF0ZS1nbG93e3BvaW50ZXItZXZlbnRzOm5vbmU7ZmlsbDpub25lfQpAa2V5ZnJhbWVzIGdsb3dQdWxzZXswJSwxMDAle29wYWNpdHk6MC4xMn01MCV7b3BhY2l0eTowLjIyfX0KCi5tYXAtdG9vbHRpcHsKICBwb3NpdGlvbjphYnNvbHV0ZTtwb2ludGVyLWV2ZW50czpub25lOwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC45NSk7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjlweDsKICBwYWRkaW5nOjEycHggMTRweDtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTJzO3otaW5kZXg6OTk5OTttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmcjtnYXA6MThweDsKfQoubmFyLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE0cHgpO292ZXJmbG93OmhpZGRlbjsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOwp9Ci5uYy1oZWFkewogIHBhZGRpbmc6MTZweCAyMHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsKfQoubmMtYm9keXtwYWRkaW5nOjhweCAyMHB4IDE2cHg7ZmxleDoxO292ZXJmbG93LXk6YXV0bzt9Ci5uYy10aXRsZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKX0KLm5jLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDVlbTttYXJnaW4tdG9wOjJweH0KLm5jLWJvZHl7cGFkZGluZzoxM3B4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MH0KCi5tb20taXR7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4OwogIHBhZGRpbmc6N3B4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubW9tLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tb20tcmt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7d2lkdGg6MTNweDtmbGV4LXNocmluazowfQoubW9tLWluZntmbGV4OjF9Ci5tb20tbm17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5tb20tc3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KLm1vbS1wY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2ZvbnQtd2VpZ2h0OjQwMDtmbGV4LXNocmluazowfQoubW9tLXBjLnJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tb20tcGMuZntjb2xvcjp2YXIoLS1mYWxsKX0KLm1vbS10cntoZWlnaHQ6MS41cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O21hcmdpbjozcHggMCAwO292ZXJmbG93OmhpZGRlbn0KLm1vbS1mbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweH0KCi5yZWctaXR7CiAgZGlzcGxheTpmbGV4O2dhcDo5cHg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDsKICBwYWRkaW5nOjhweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzOwp9Ci5yZWctaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLnJlZy1pdDpob3ZlcntvcGFjaXR5OjAuNzV9Ci5yZWctYmFkZ2V7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjA3ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIHBhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNCk7CiAgY29sb3I6dmFyKC0tYWNjZW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4O3doaXRlLXNwYWNlOm5vd3JhcDsKfQoucmVnLWZse2ZsZXg6MTtmb250LXNpemU6MTEuNXB4O2xpbmUtaGVpZ2h0OjEuNX0KLnJlZy1mcm9te2NvbG9yOnZhcigtLWZhaW50KX0KLnJlZy1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTttYXJnaW46MCA0cHh9Ci5yZWctdG97Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5yZWctdG17Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4fQoKLyogRkFWUyAqLwouZmF2c3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAzNnB4IDQwcHg7Cn0KLmZhdnMtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQouZmF2cy1yb3d7ZGlzcGxheTpmbGV4O2dhcDoxMHB4O292ZXJmbG93LXg6YXV0bztwYWRkaW5nLWJvdHRvbTozcHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXJ7aGVpZ2h0OjJweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MXB4fQouZmF2LWNhcmR7CiAgZmxleDowIDAgMTkwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MTJweDtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4czsKfQouZmF2LWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMjIpO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMil9Ci5mYy1oZWFke2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjdweH0KLmZjLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjp2YXIoLS1pbmspfQouZmMtc2N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouZmMtcm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDozcHh9Ci5mYy1yb3cgLnZ7Y29sb3I6dmFyKC0tZGltKTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHh9Ci5mYXZzLWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LXN0eWxlOml0YWxpYztwYWRkaW5nOjRweCAwfQoKLyogRk9PVCAqLwouZm9vdHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjQ4cHggMzZweCA2MHB4O21heC13aWR0aDo1ODBweDttYXJnaW46MCBhdXRvO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmZvb3QtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbi1ib3R0b206MTRweH0KLmZvb3QtbGluZXtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjEuODttYXJnaW4tYm90dG9tOjEycHh9Ci5mb290LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KX0KCi8qIGFuaW1hdGlvbnMgKi8KQGtleWZyYW1lcyBmYWRlVXB7ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoubWFwLWNhcmQsLnN0YXRlLXBhbmVsLC5uYXItY2FyZCwuc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uOmZhZGVVcCAwLjU1cyBjdWJpYy1iZXppZXIoLjIsLjgsLjIsMSkgYmFja3dhcmRzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjA3c30KLm5hci1jYXJkOm50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC4xNHN9Ci5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb24tZGVsYXk6MC4wNXN9CgpAbWVkaWEobWF4LXdpZHRoOjExMDBweCl7CiAgLm1haW57Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAuc3RhdGUtcGFuZWx7bWF4LWhlaWdodDpub25lfQogIC5uYXItcm93e2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9Cn0KCi8qIOKUgOKUgCBXSEFUIElORElBIElTIFJFQUNUSU5HIFRPIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwoud2lyLXNlY3Rpb257CiAgZmxleDoxO21pbi13aWR0aDowOwogIHBhZGRpbmc6MDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLndpci1oZWFkZXJ7CiAgcGFkZGluZzoxOHB4IDIycHggMTRweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Cn0KLndpci10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuM2VtOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC44NTsKfQoud2lyLWxpdmV7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMWVtOwp9Ci53aXItbGl2ZS1kb3R7CiAgd2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojMzlmZjE0OwogIGJveC1zaGFkb3c6MCAwIDZweCByZ2JhKDU3LDI1NSwyMCwwLjYpOwogIGFuaW1hdGlvbjp3aXJMaXZlUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyB3aXJMaXZlUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC4zfX0KLndpci1zaWduYWxze2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47ZmxleDoxO292ZXJmbG93OmhpZGRlbn0KLndpci1zaWduYWx7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjA7CiAgcGFkZGluZzoxM3B4IDIycHg7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzNSk7CiAgb3BhY2l0eTowOwogIGFuaW1hdGlvbjp3aXJGYWRlSW4gMC42cyBlYXNlIGZvcndhcmRzOwogIHBvc2l0aW9uOnJlbGF0aXZlO2N1cnNvcjpkZWZhdWx0OwogIHRyYW5zaXRpb246YmFja2dyb3VuZCAwLjE1czsKfQoud2lyLXNpZ25hbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci53aXItc2lnbmFsOmxhc3QtY2hpbGR7Ym9yZGVyLWJvdHRvbTpub25lfQpAa2V5ZnJhbWVzIHdpckZhZGVJbntmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWCgtNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoud2lyLXNpZ25hbC1iYXJ7CiAgd2lkdGg6MnB4O2JvcmRlci1yYWRpdXM6MXB4O2ZsZXgtc2hyaW5rOjA7CiAgbWFyZ2luLXJpZ2h0OjE0cHg7bWFyZ2luLXRvcDo0cHg7CiAgYWxpZ24tc2VsZjpzdHJldGNoO21pbi1oZWlnaHQ6MTZweDsKICBvcGFjaXR5OjAuNjsKfQoud2lyLXNpZ25hbC1jb250ZW50e2ZsZXg6MTttaW4td2lkdGg6MH0KLndpci1zaWduYWwtdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE0LjVweDtmb250LXdlaWdodDozMDA7CiAgY29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQoud2lyLXNpZ25hbC10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOmluaGVyaXQ7b3BhY2l0eTowLjh9Ci53aXItc2lnbmFsLW1ldGF7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6NHB4Owp9Ci53aXItc2lnbmFsLXRhZ3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6N3B4O2xldHRlci1zcGFjaW5nOjAuMTRlbTsKICB0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7b3BhY2l0eTowLjQ1Owp9Ci53aXItc2lnbmFsLWxvY3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci53aXItbG9hZGluZ3sKICBkaXNwbGF5OmZsZXg7Z2FwOjZweDtwYWRkaW5nOjIwcHggMjJweDthbGlnbi1pdGVtczpjZW50ZXI7Cn0KLndpci1kb3R7d2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjp3aXJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZX0KLndpci1kb3Q6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjJzfQoud2lyLWRvdDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuNHN9CkBrZXlmcmFtZXMgd2lyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9Imx0YWItdG9vbHRpcCI+PC9kaXY+Cgo8IS0tIExPQURFUiAtLT4KPGRpdiBpZD0iYXBwLWxvYWRlciIgc3R5bGU9InBvc2l0aW9uOmZpeGVkO2luc2V0OjA7ei1pbmRleDo5OTk5ODtiYWNrZ3JvdW5kOiMwNjA5MTA7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOm9wYWNpdHkgMC44cyBlYXNlLHZpc2liaWxpdHkgMC44cyBlYXNlOyI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7d2lkdGg6NjRweDtoZWlnaHQ6NjRweDttYXJnaW4tYm90dG9tOjM2cHgiPgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiNlMDVhMjg7YW5pbWF0aW9uOmxkclB1bHNlIDJzIGVhc2UtaW4tb3V0IGluZmluaXRlIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0OjE2cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjpsZHJSaW5nIDJzIGVhc2Utb3V0IGluZmluaXRlIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTUpO2FuaW1hdGlvbjpsZHJSaW5nIDJzIGVhc2Utb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjVzIj48L2Rpdj4KICAgIDxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2luc2V0Oi0xMHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4wNyk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjFzIj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonUGxheWZhaXIgRGlzcGxheScsR2VvcmdpYSxzZXJpZjtmb250LXNpemU6Y2xhbXAoMjhweCw1dncsNDJweCk7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6I2YwZWNlNDtsaW5lLWhlaWdodDoxO21hcmdpbi1ib3R0b206MTBweCI+CiAgICA8ZW0gc3R5bGU9ImNvbG9yOiNlOGM0YTA7Zm9udC1zdHlsZTppdGFsaWMiPlB1bHNlPC9lbT4gb2YgSW5kaWEKICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonQ291cmllciBOZXcnLG1vbm9zcGFjZTtmb250LXNpemU6MTFweDtsZXR0ZXItc3BhY2luZzowLjI4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC40KTttYXJnaW4tYm90dG9tOjI4cHgiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXM8L2Rpdj4KICA8ZGl2IHN0eWxlPSJmb250LWZhbWlseTonQ291cmllciBOZXcnLG1vbm9zcGFjZTtmb250LXNpemU6MTBweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4yNSk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweCI+CiAgICA8c3Bhbj5Ob3QgbmV3czwvc3Bhbj48c3BhbiBzdHlsZT0ib3BhY2l0eTowLjMiPsK3PC9zcGFuPjxzcGFuPk5vdCBwcmVkaWN0aW9uPC9zcGFuPjxzcGFuIHN0eWxlPSJvcGFjaXR5OjAuMyI+wrc8L3NwYW4+CiAgICA8c3Bhbj5KdXN0IDxzcGFuIHN0eWxlPSJjb2xvcjojMzlmZjE0O3RleHQtc2hhZG93OjAgMCAxMHB4IHJnYmEoNTcsMjU1LDIwLDAuNSk7YW5pbWF0aW9uOmxkckdsb3cgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPm9ic2VydmF0aW9uPC9zcGFuPjwvc3Bhbj4KICA8L2Rpdj4KICA8ZGl2IHN0eWxlPSJtYXJnaW4tdG9wOjQ4cHg7ZGlzcGxheTpmbGV4O2dhcDo2cHgiPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjJzIj48L3NwYW4+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MC40cyI+PC9zcGFuPgogIDwvZGl2Pgo8L2Rpdj4KPHN0eWxlPgpAa2V5ZnJhbWVzIGxkclB1bHNlezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjU7dHJhbnNmb3JtOnNjYWxlKDAuOCl9fQpAa2V5ZnJhbWVzIGxkclJpbmd7MCV7dHJhbnNmb3JtOnNjYWxlKDAuOCk7b3BhY2l0eTowLjZ9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS41KTtvcGFjaXR5OjB9fQpAa2V5ZnJhbWVzIGxkckdsb3d7MCUsMTAwJXt0ZXh0LXNoYWRvdzowIDAgMTBweCByZ2JhKDU3LDI1NSwyMCwwLjUpfTUwJXt0ZXh0LXNoYWRvdzowIDAgMjJweCByZ2JhKDU3LDI1NSwyMCwwLjkpLDAgMCA0MHB4IHJnYmEoNTcsMjU1LDIwLDAuMyl9fQpAa2V5ZnJhbWVzIGxkckRvdHswJSw4MCUsMTAwJXt0cmFuc2Zvcm06c2NhbGUoMC42KTtvcGFjaXR5OjAuM300MCV7dHJhbnNmb3JtOnNjYWxlKDEpO29wYWNpdHk6MX19Cjwvc3R5bGU+Cgo8ZGl2IGNsYXNzPSJ0b3BiYXIiPgogIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLW1hcmsiPjxzcGFuIGNsYXNzPSJicmFuZC1wdWxzZS1kb3QiPjwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kLXRleHQtYmxvY2siPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtbmFtZSI+PGVtIGNsYXNzPSJicmFuZC1wdWxzZS13b3JkIj5QdWxzZTwvZW0+IG9mIEluZGlhPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0iYnJhbmQtdGFnbGluZSI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3NwYW4+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJ0b3BiYXItciI+CiAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9ImxpdmUtaW5kaWNhdG9yIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPiBzaWduYWxzCiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItdGlwIj4KICAgICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItbGFiZWwiPk9ic2VydmVkIGZyb208L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJzaWctaG92ZXItc291cmNlcyI+cmVnaW9uYWwgbWVkaWEgwrcgcHVibGljIGRpc2N1c3Npb24gwrcgaW5kZXBlbmRlbnQgcmVwb3J0aW5nIMK3IHNvY2lhbCBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7YmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpO3BhZGRpbmc6MCAzNnB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOyI+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+U2lnbmFscyB0cmFja2VkPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIHNjLXZhbC1zbSIgaWQ9InNjLXNpZ25hbHMtdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1zaWduYWxzLXN1YiI+bG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIiBvbmNsaWNrPSJzZWxlY3RIb3R0ZXN0KCkiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkhpZ2hlc3QgYXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtaG90dGVzdC10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWFuZ2VyLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtYW5nZXItdGlwIj48L2Rpdj48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIHNjLWhvdmVyYWJsZSI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgcmlzaW5nPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtcmlzaW5nLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtcmlzaW5nLXN1YiI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLXJpc2luZy10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIG5hcnJhdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCBzYy12YWwtc20iIGlkPSJzYy1uYXItdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1uYXItc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtbmFyLXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5MZWFzdCBhY3RpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29sLXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbC1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1jb29sLXRpcCI+PC9kaXY+PC9kaXY+CjwvZGl2CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLWhvdHRlc3QiIHN0eWxlPSJjdXJzb3I6cG9pbnRlciIgb25jbGljaz0ic2VsZWN0SG90dGVzdCgpIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5IaWdoZXN0IGF0dGVudGlvbjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtaG90dGVzdC12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPkNsaWNrIHRvIGV4cGxvcmU8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIj4KICAgIDxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1hbmdlci12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtYW5nZXItc3ViIj5PdXRyYWdlICYgcHJvdGVzdCBzaWduYWxzPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+VG9wIHJpc2luZyBuYXJyYXRpdmU8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLW5hcnJhdGl2ZS12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtbmFycmF0aXZlLXN1YiI+TmF0aW9uYWwgc2lnbmFsIHN1cmdlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCBjb29saW5nPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1jb29saW5nLXZhbCI+4oCUPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29saW5nLXN1YiI+U2lnbmFsIGRlY2F5PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPHN0eWxlPgouc3RhdC1jZWxsewogIGZsZXg6MTtwYWRkaW5nOjEwcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2FwOjJweDsKICB0cmFuc2l0aW9uOmJhY2tncm91bmQgMC4xNXM7Cn0KLnN0YXQtY2VsbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci5zdGF0LWRpdnt3aWR0aDoxcHg7YmFja2dyb3VuZDpyZ2JhKDE2MCwxOTAsMjMwLDAuMDcpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luOjhweCAwfQouc2MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO21hcmdpbi10b3A6MXB4fQouc2Mtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQo8L3N0eWxlPgoKCiAgPCEtLSBTSUdOQVRVUkUgSU5TSUdIVCArIE5BUlJBVElWRSBTVFJJUCBzaWRlIGJ5IHNpZGUgLS0+CiAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoxOHB4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7bWFyZ2luLXRvcDoxNnB4O21hcmdpbi1ib3R0b206MDttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbi1sZWZ0OmF1dG87bWFyZ2luLXJpZ2h0OmF1dG87cGFkZGluZzowIDM2cHg7Ij4KICAgIDxkaXYgY2xhc3M9Indpci1zZWN0aW9uIj4KICAgICAgPGRpdiBjbGFzcz0id2lyLWhlYWRlciI+CiAgICAgICAgPGRpdiBjbGFzcz0id2lyLXRpdGxlIj5XaGF0IEluZGlhIGlzIHJlYWN0aW5nIHRvPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0id2lyLWxpdmUiPjxzcGFuIGNsYXNzPSJ3aXItbGl2ZS1kb3QiPjwvc3Bhbj5saXZlIHNpZ25hbHM8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Indpci1zaWduYWxzIiBpZD0id2lyLXNpZ25hbHMiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci1sb2FkaW5nIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj48c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZmxleDowIDAgMzYwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOyI+CiAgICAgIDwhLS0gaGVhZGVyIC0tPgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsiPgogICAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj5OYXJyYXRpdmUgc2hpZnRzPC9zcGFuPgogICAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MnB4OyI+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPCEtLSBzaGlmdHMgbGlzdCAtLT4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoxMHB4IDE0cHg7Z2FwOjZweDsiIGlkPSJzaGlmdC1saXN0Ij48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+CgoKPCEtLSBNQUlOOiBNQVAgKyBTVEFURSBQQU5FTCAtLT4KPGRpdiBjbGFzcz0ibWFpbiI+CgogIDxkaXYgY2xhc3M9Im1hcC1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im1hcC10b3AiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtdGl0bGUtYmxvY2siPgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij5JbmRpYSAmbWRhc2g7IGNvbGxlY3RpdmUgYXR0ZW50aW9uPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibXMiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IGxpdmUgc2lnbmFsIGNvbXBvc2l0ZTwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGVnZW5kIj48c3Bhbj5xdWlldDwvc3Bhbj48ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj48c3Bhbj5hY3RpdmU8L3NwYW4+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxheWVyLXJvdyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsYXllci1sYWJlbCI+Vmlldzwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibHRhYnMiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iYXR0ZW50aW9uIj5BdHRlbnRpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IldoaWNoIHN0YXRlcyBhcmUgcmVjZWl2aW5nIHRoZSBtb3N0IHB1YmxpYyBmb2N1cy4gSGlnaCBhdHRlbnRpb24gPSBjb25jZW50cmF0ZWQgbmV3cyBjb3ZlcmFnZSBhbmQgcG9saXRpY2FsIGFjdGl2aXR5LiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IlRoZSBkb21pbmFudCBlbW90aW9uYWwgdG9uZSDigJQgYW54aW91cywgYW5ncnksIGhvcGVmdWwsIHByb3VkIG9yIGZlYXJmdWwuIFJldmVhbHMgdGhlIHBzeWNob2xvZ2ljYWwgdW5kZXJjdXJyZW50IG9mIHBvbGl0aWNhbCBhdHRlbnRpb24uIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiIgZGF0YS1sYXllcj0idmVsb2NpdHkiPk1vbWVudHVtIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJJcyBhdHRlbnRpb24gcmlzaW5nIG9yIGZhbGxpbmc/IFJpc2luZyA9IG5hcnJhdGl2ZSBhY2NlbGVyYXRpbmcuIENvb2xpbmcgPSBsb3NpbmcgdHJhY3Rpb24uIFNob3dzIHN0YXRlcyBlbnRlcmluZyBvciBleGl0aW5nIGEgcG9saXRpY2FsIGN5Y2xlLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJtYXAtc3ZnLXdyYXAiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtaW5uZXIiPgogICAgICAgIDxzdmcgaWQ9ImluZGlhLW1hcCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgogICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iYW1iR2xvdyIgY3g9IjUwJSIgY3k9IjUwJSIgcj0iNTAlIj4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSJyZ2JhKDIyNCw5MCw0MCwwLjA0KSIvPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0idHJhbnNwYXJlbnQiLz4KICAgICAgICAgICAgPC9yYWRpYWxHcmFkaWVudD4KICAgICAgICAgICAgPGZpbHRlciBpZD0ic3RhdGVHbG93IiB4PSItMzAlIiB5PSItMzAlIiB3aWR0aD0iMTYwJSIgaGVpZ2h0PSIxNjAlIj4KICAgICAgICAgICAgICA8ZmVHYXVzc2lhbkJsdXIgaW49IlNvdXJjZUdyYXBoaWMiIHN0ZERldmlhdGlvbj0iOCIgcmVzdWx0PSJibHVyIi8+CiAgICAgICAgICAgICAgPGZlQ29tcG9zaXRlIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9ImJsdXIiIG9wZXJhdG9yPSJvdmVyIi8+CiAgICAgICAgICAgIDwvZmlsdGVyPgogICAgICAgICAgPC9kZWZzPgogICAgICAgICAgPHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI4MDAiIGZpbGw9InVybCgjYW1iR2xvdykiLz4KICAgICAgICAgIDxnIGlkPSJtYXAtZ2xvdyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1zdGF0ZXMiPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtcHVsc2VzIj48L2c+CiAgICAgICAgPC9zdmc+CiAgICAgICAgPGRpdiBjbGFzcz0ibWFwLXRvb2x0aXAiIGlkPSJ0b29sdGlwIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBTVEFURSBQQU5FTCAtLT4KICA8ZGl2IGNsYXNzPSJzdGF0ZS1wYW5lbCIgaWQ9InN0YXRlLWRldGFpbCI+CiAgICA8ZGl2IGNsYXNzPSJwYW5lbC1lbXB0eSI+CiAgICAgIDxzdmcgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEiPgogICAgICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0iTTEyIDh2NE0xMiAxNmguMDEiLz4KICAgICAgPC9zdmc+CiAgICAgIDxkaXYgY2xhc3M9InBlLXQiPlNlbGVjdCBhIHN0YXRlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InBlLXMiPkNsaWNrIGFueSByZWdpb24gb24gdGhlIG1hcDxici8+dG8gb3BlbiBpdHMgbmFycmF0aXZlIHBhbmVsLjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cgo8L2Rpdj4KCjwhLS0gTkFSUkFUSVZFIFJPVyAtLT4KPGRpdiBjbGFzcz0ibmFyLXJvdyI+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsiPgogICAgICA8c3BhbiBjbGFzcz0ibmMtZG90IHJpc2UyIj48L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy10aXRsZSI+UmlzaW5nIG5hcnJhdGl2ZXM8L3NwYW4+CiAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tbGVmdDphdXRvIj5nYWluaW5nIHRyYWN0aW9uPC9zcGFuPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0icmlzaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsiPgogICAgICA8c3BhbiBjbGFzcz0ibmMtZG90IGZhbGwiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5EZWNsaW5pbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1sZWZ0OmF1dG8iPmxvc2luZyB0cmFjdGlvbjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9ImRlY2xpbmluZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIEp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDhweCByZ2JhKDU3LDI1NSwyMCwwLjQpIj5vYnNlcnZhdGlvbjwvc3Bhbj4uPC9kaXY+CjwvZGl2PgoKPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vdG9wb2pzb24tY2xpZW50QDMuMS4wL2Rpc3QvdG9wb2pzb24tY2xpZW50Lm1pbi5qcyI+PC9zY3JpcHQ+CjxzY3JpcHQ+CnZhciBBUElfQkFTRT0obG9jYXRpb24uaG9zdG5hbWU9PT0nbG9jYWxob3N0J3x8bG9jYXRpb24uaG9zdG5hbWU9PT0nMTI3LjAuMC4xJyk/J2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMCc6Jyc7CgovLyBBUEkKYXN5bmMgZnVuY3Rpb24gZmV0Y2hBbGxTdGF0ZXMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZXMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgcm93cz1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFyb3dzfHwhcm93cy5sZW5ndGgpIHJldHVybjsKICAgIHJvd3MuZm9yRWFjaChmdW5jdGlvbihyb3cpewogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhyb3cuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbUVtbz1yb3cuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxudWxsOwogICAgICB2YXIgZW50cnk9e2F0dGVudGlvbjpyb3cuYXR0ZW50aW9uLGRlbHRhOnJvdy5kZWx0YV8yNGgsdmVsb2NpdHk6cm93LnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tRW1vLGRvbWluYW50X25hcnJhdGl2ZTpyb3cuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgICBMSVZFW3Jvdy5uYW1lXT1lbnRyeTsKICAgICAgaWYoIVNEW3Jvdy5uYW1lXSkgU0Rbcm93Lm5hbWVdPU9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7CiAgICAgIE9iamVjdC5hc3NpZ24oU0Rbcm93Lm5hbWVdLGVudHJ5KTsKICAgIH0pOwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgcmVuZGVyU3RyaXAoIjNtIik7CiAgICBidWlsZFdJUlNpZ25hbHMoKTsKICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIHNldFRpbWVvdXQocmVuZGVyTW9tZW50dW0sIDUwMCk7CiAgICBpZihTRUwmJkxJVkVbU0VMXSYmZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpKSByZW5kZXJQYW5lbChTRUwpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tBUEldJyxlLm1lc3NhZ2UpO30KfQoKZnVuY3Rpb24gYnVpbGRMb2NhbEluc2lnaHQoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKCiAgLy8gQWdncmVnYXRlIHRvcCBuYXJyYXRpdmVzIGFjcm9zcyBhbGwgc3RhdGVzCiAgdmFyIG5hcj17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIGlmKCFuYXJbbi5uYW1lXSkgbmFyW24ubmFtZV09e3VwOjAsZG93bjowLGZsYXQ6MCx0b3RhbDowfTsKICAgICAgbmFyW24ubmFtZV1bbi5kaXJdPShuYXJbbi5uYW1lXVtuLmRpcl18fDApK24udmFsOwogICAgICBuYXJbbi5uYW1lXS50b3RhbD0obmFyW24ubmFtZV0udG90YWx8fDApK24udmFsOwogICAgfSk7CiAgfSk7CgogIC8vIFRvcCByaXNpbmcgYW5kIGZhbGxpbmcgKGV4Y2x1ZGUgdGllcyB3aGVyZSBzYW1lIG5hbWUgcmlzZXMgYW5kIGZhbGxzKQogIHZhciByaXNpbmc9T2JqZWN0LmVudHJpZXMobmFyKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS51cD5rdlsxXS5kb3duO30pCiAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnVwLWFbMV0udXA7fSkuc2xpY2UoMCwzKTsKICB2YXIgZmFsbGluZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvd24+a3ZbMV0udXA7fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0uZG93bi1hWzFdLmRvd247fSkuc2xpY2UoMCwyKTsKICB2YXIgdG9wMz1PYmplY3QuZW50cmllcyhuYXIpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS50b3RhbC1hWzFdLnRvdGFsO30pLnNsaWNlKDAsMyk7CgogIC8vIEhvdHRlc3Qgc3RhdGUKICB2YXIgaG90dGVzdD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgdmFyIGhvdHRlc3RFbW89aG90dGVzdD8oTElWRVtob3R0ZXN0WzBdXSYmTElWRVtob3R0ZXN0WzBdXS5kb21pbmFudF9lbW90aW9uKXx8Jyc6JycgOwoKICAvLyBCdWlsZCBpbnNpZ2h0IHRleHQg4oCUIG1vcmUgYW5hbHl0aWNhbCwgY29udGV4dC1hd2FyZQogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgaWYoIWVsKSByZXR1cm47CgogIHZhciBsaW5lcz1bXTsKICBpZihyaXNpbmcubGVuZ3RoJiZmYWxsaW5nLmxlbmd0aCYmcmlzaW5nWzBdWzBdIT09ZmFsbGluZ1swXVswXSl7CiAgICBsaW5lcy5wdXNoKCc8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPiBpcyB0aGUgZG9taW5hbnQgc2lnbmFsIGFjcm9zcyBJbmRpYSB0b2RheScpOwogICAgaWYoZmFsbGluZ1swXSkgbGluZXMucHVzaCgnIGFzIDxlbT4nK2ZhbGxpbmdbMF1bMF0rJzwvZW0+IGZhZGVzIGZyb20gbmF0aW9uYWwgZm9jdXMnKTsKICAgIGlmKGhvdHRlc3QpIGxpbmVzLnB1c2goJy4gPHN0cm9uZyBzdHlsZT0iY29sb3I6dmFyKC0taW5rKSI+Jytob3R0ZXN0WzBdKyc8L3N0cm9uZz4gaXMgdGhlIG1vc3QgYWN0aXZlIHN0YXRlJysKICAgICAgKGhvdHRlc3RFbW8/JyB3aXRoICcraG90dGVzdEVtbysnIGFzIHRoZSBwcmltYXJ5IHNpZ25hbCB0b25lJzonJykpOwogICAgaWYocmlzaW5nWzFdKSBsaW5lcy5wdXNoKCcuIFNlY29uZGFyeSBzdXJnZTogPGVtPicrcmlzaW5nWzFdWzBdKyc8L2VtPicpOwogIH0gZWxzZSBpZihyaXNpbmcubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ1NpZ25hbHMgYXJlIGNvbmNlbnRyYXRlZCBhcm91bmQgPGVtPicrcmlzaW5nWzBdWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3Jpc2luZ1swXVswXS5zbGljZSgxKSsnPC9lbT4nKTsKICAgIGlmKGhvdHRlc3QpIGxpbmVzLnB1c2goJy4gPHN0cm9uZyBzdHlsZT0iY29sb3I6dmFyKC0taW5rKSI+Jytob3R0ZXN0WzBdKyc8L3N0cm9uZz4gbGVhZHMgbmF0aW9uYWwgYXR0ZW50aW9uJyk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJyBhbG9uZ3NpZGUgPGVtPicrcmlzaW5nWzFdWzBdKyc8L2VtPicpOwogIH0gZWxzZSBpZih0b3AzLmxlbmd0aCl7CiAgICBsaW5lcy5wdXNoKCdOYXRpb25hbCBzaWduYWxzIGFyZSBkaXNwZXJzZWQuIFRvcCBuYXJyYXRpdmVzOiAnK3RvcDMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGVtPicrblswXSsnPC9lbT4nO30pLmpvaW4oJywgJykpOwogIH0KCiAgaWYobGluZXMubGVuZ3RoKXsKICAgIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytsaW5lcy5qb2luKCcnKSsnLjwvZGl2Pic7CiAgfQoKICAvLyBUYWdzCiAgaWYodEVsKXsKICAgIHZhciB0YWdzPVtdOwogICAgZmFsbGluZy5zbGljZSgwLDEpLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDU5LDE4NCwyMTYsMC4zKTtjb2xvcjojM2JiOGQ4Ij7ihpMgJytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKTsKICAgIH0pOwogICAgcmlzaW5nLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjMpO2NvbG9yOiNlMDVhMjgiPuKGkSAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICBpZih0YWdzLmxlbmd0aCkgdEVsLmlubmVySFRNTD10YWdzLmpvaW4oJycpOwogIH0KCiAgaWYobWV0YUVsKXsKICAgIHZhciBzdGF0ZUNvdW50PU9iamVjdC52YWx1ZXMoTElWRSkuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzLmF0dGVudGlvbj4yO30pLmxlbmd0aDsKICAgIG1ldGFFbC50ZXh0Q29udGVudD0nT2JzZXJ2aW5nICcrc3RhdGVDb3VudCsnIGFjdGl2ZSBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKCkudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicse2hvdXI6JzItZGlnaXQnLG1pbnV0ZTonMi1kaWdpdCd9KTsKICB9Cn0KCmZ1bmN0aW9uIHVwZGF0ZUFsbFN0cmlwcygpewogIGZ1bmN0aW9uIHRpcChpZCx0aXRsZSxuYXJzKXt2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIXQpcmV0dXJuO3QuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzYy10aXAtdGl0bGUiPicrdGl0bGUrJzwvZGl2PicrKG5hcnN8fFtdKS5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9InNjLXRpcC1yb3ciPsK3ICcrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsnPC9kaXY+Jzt9KS5qb2luKCcnKTt9CiAgICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKICB2YXIgaG90dGVzdD1lbnRyaWVzLnJlZHVjZShmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApPihhWzFdLmF0dGVudGlvbnx8MCk/YjphO30sZW50cmllc1swXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC12YWwnLGhvdHRlc3RbMF0pOwogIHRpcCgnc2MtaG90dGVzdC10aXAnLCdXaHkgJytob3R0ZXN0WzBdKyc/Jyxob3R0ZXN0WzFdLm5hcnJhdGl2ZXMpOwogIHNldFRleHQoJ3NjLWhvdHRlc3Qtc3ViJywnQXR0ZW50aW9uICcraG90dGVzdFsxXS5hdHRlbnRpb24udG9GaXhlZCgxKSk7CiAgdmFyIHRvcEFuZ2VyTm09bnVsbCx0b3BBbmdlclBjdD0wOwogIGVudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICB2YXIgZT1rdlsxXS5lbW90aW9uc3x8e307CiAgICB2YXIgYT1lLmFuZ2VyfHwwOwogICAgaWYoYT4wJiZhPD0xKSBhPU1hdGgucm91bmQoYSoxMDApOwogICAgaWYoYT50b3BBbmdlclBjdCl7dG9wQW5nZXJQY3Q9YTt0b3BBbmdlck5tPWt2WzBdO30KICB9KTsKICBpZih0b3BBbmdlck5tJiZ0b3BBbmdlclBjdD4wKXsKICAgIHNldFRleHQoJ3NjLWFuZ2VyLXZhbCcsdG9wQW5nZXJObSk7CiAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdBbmdlciAnK01hdGgucm91bmQodG9wQW5nZXJQY3QpKyclIG9mIHNpZ25hbHMnKTsKICB9IGVsc2UgewogICAgLy8gRmFsbCBiYWNrIHRvIGRvbWluYW50X2Vtb3Rpb249YW5nZXIKICAgIHZhciBhbmdlckRvbT1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvbWluYW50X2Vtb3Rpb249PT0nYW5nZXInO30pOwogICAgaWYoYW5nZXJEb20ubGVuZ3RoKXsKICAgICAgdmFyIHRvcEJ5QXR0PWFuZ2VyRG9tLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogICAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLHRvcEJ5QXR0WzBdKTsKICAgICAgc2V0VGV4dCgnc2MtYW5nZXItc3ViJywnRG9taW5hbnQgZW1vdGlvbjogYW5nZXInKTsKICAgIH0KICB9CiAgdmFyIGNvb2xpbmc9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0udmVsb2NpdHl8fDApPChhWzFdLnZlbG9jaXR5fHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1jb29saW5nLXZhbCcsY29vbGluZ1swXSk7c2V0VGV4dCgnc2MtY29vbGluZy1zdWInLCdWZWxvY2l0eSAnK2Nvb2xpbmdbMV0udmVsb2NpdHkudG9GaXhlZCgzKSk7CiAgdmFyIG5jPXt9O2VudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlKW5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV09KG5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV18fDApKzE7fSk7CiAgdmFyIHRuPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXTsKICBpZih0bil7c2V0VGV4dCgnc2MtbmFycmF0aXZlLXZhbCcsdG5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdG5bMF0uc2xpY2UoMSkpO3NldFRleHQoJ3NjLW5hcnJhdGl2ZS1zdWInLCdEb21pbmFudCBhY3Jvc3MgJyt0blsxXSsnIHN0YXRlcycpO30KfQphc3luYyBmdW5jdGlvbiBmZXRjaERldGFpbChuYW1lKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChuYW1lKSk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhkLmVtb3Rpb25zfHx7fSk7CiAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8ZC5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgU0RbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGVtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlLAogICAgICBuYXJyYXRpdmVzOihkLm5hcnJhdGl2ZXN8fFtdKS5tYXAoZnVuY3Rpb24obil7cmV0dXJue25hbWU6bi5uYW1lLHZhbDpuLnZhbCxkaXI6bi5kaXJ8fCdmbGF0J307fSksCiAgICAgIHJpc2luZzpkLnJpc2luZ3x8W10sZmFsbGluZzpkLmZhbGxpbmd8fFtdLHN1bW1hcnk6ZC5zdW1tYXJ5fHxERUZBVUxULnN1bW1hcnksCiAgICAgIGFydGljbGVzOmQuYXJ0aWNsZXN8fFtdLHRpbWVsaW5lOmQudGltZWxpbmV8fERFRkFVTFQudGltZWxpbmUsCiAgICAgIG5hcnJhdGl2ZUhpc3Rvcnk6ZC5uYXJyYXRpdmVIaXN0b3J5fHxERUZBVUxULm5hcnJhdGl2ZUhpc3Rvcnksc2lnbmFsX2NvdW50OmQuc2lnbmFsX2NvdW50fHwwfTsKICAgIGlmKCFMSVZFW25hbWVdKUxJVkVbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZX07CiAgICBMSVZFW25hbWVdLmVtb3Rpb25zPWVtb3M7TElWRVtuYW1lXS5kb21pbmFudF9lbW90aW9uPWRvbTsKICAgIHJldHVybiBTRFtuYW1lXTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbZmV0Y2hEZXRhaWxdJyxuYW1lLGUubWVzc2FnZSk7cmV0dXJuIFNEW25hbWVdfHxERUZBVUxUO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTbmFwKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc25hcHNob3QvZGFpbHknKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIC8vIHRvcGJhcgogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7CiAgICBpZihlbCYmZC50b3RhbF9zaWduYWxzKSBlbC50ZXh0Q29udGVudD1kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTsKICAgIHZhciBtZXRhPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtbWV0YScpOwogICAgaWYobWV0YSYmZC5hc19vZikgbWV0YS50ZXh0Q29udGVudD0nMzAgc3RhdGVzIMK3IHVwZGF0ZWQgJytuZXcgRGF0ZShkLmFzX29mKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyk7CiAgICAvLyBzdGF0cyBzdHJpcAogICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLCBkLnRvdGFsX3NpZ25hbHM/ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk6Jy0nKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogIH1jYXRjaChlKXt9Cn0KCmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKGVsKWVsLnRleHRDb250ZW50PXZhbDt9CgpmdW5jdGlvbiB1cGRhdGVTdHJpcE5hcnJhdGl2ZSgpe3VwZGF0ZUFsbFN0cmlwcygpO30KZnVuY3Rpb24gdXBkYXRlU3RyaXBBbmdlcigpe30KCmZ1bmN0aW9uIHNlbGVjdEhvdHRlc3QoKXsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKFNEKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApIHNlbGVjdF8odG9wWzBdKTsKfQphc3luYyBmdW5jdGlvbiBmZXRjaEluc2lnaHRzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvaW5zaWdodHMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIHZhciBzaWc9ZC5zaWduYXR1cmU7CiAgICBpZihzaWcpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQudGFncyl0RWwuaW5uZXJIVE1MPWQudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgIH0KICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICBpZihyRWwmJmQucmlzaW5nJiZkLnJpc2luZy5sZW5ndGgpckVsLmlubmVySFRNTD1kLnJpc2luZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrbi5zdGF0ZXMuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMykrJyU7YmFja2dyb3VuZDojZTA1YTI4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICAgIGlmKGZFbCYmZC5mYWxsaW5nJiZkLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytuLnN0YXRlcy5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKSsnJTtiYWNrZ3JvdW5kOiMzYmI4ZDgiPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgICBpZihnRWwmJmQucmVnaW9uYWwmJmQucmVnaW9uYWwubGVuZ3RoKWdFbC5pbm5lckhUTUw9ZC5yZWdpb25hbC5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuIj48c3BhbiBjbGFzcz0ibmktbmFtZSI+JytyLnJlZ2lvbisnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrci5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj48ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrci5ob3R0ZXN0X3N0YXRlKycgwrcgJytyLnRvcF9uYXJyYXRpdmUrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoRnVsbFNuYXBzaG90KCl7CiAgLy8gTG9hZCBBTEwgc3RhdGUgZGF0YSBpbiBvbmUgcmVxdWVzdCBmb3IgaW5zdGFudCBmaXJzdC1sb2FkCiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvZnVsbC1zbmFwc2hvdCcpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC53YXJtaW5nX3VwfHwhZC5zdGF0ZXN8fCFkLnN0YXRlcy5sZW5ndGgpIHJldHVybiBmYWxzZTsKCiAgICAvLyBQb3B1bGF0ZSBTRCBhbmQgTElWRSBmcm9tIGZ1bGwgc25hcHNob3QKICAgIGQuc3RhdGVzLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAgIGlmKCFzLm5hbWUpIHJldHVybjsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8cy5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgICB2YXIgZW50cnk9T2JqZWN0LmFzc2lnbih7fSxzLHtlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRlbHRhOnMuZGVsdGFfMjRofHwwfSk7CiAgICAgIFNEW3MubmFtZV09ZW50cnk7CiAgICAgIExJVkVbcy5uYW1lXT17YXR0ZW50aW9uOnMuYXR0ZW50aW9uLGRlbHRhOnMuZGVsdGFfMjRofHwwLHZlbG9jaXR5OnMudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOnMuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgfSk7CgogICAgLy8gVXBkYXRlIHNpZ25hbHMgY291bnQKICAgIGlmKGQuc25hcHNob3QmJmQuc25hcHNob3QudG90YWxfc2lnbmFscyl7CiAgICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJyxkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKSk7CiAgICB9CgogICAgLy8gVXBkYXRlIGluc2lnaHRzIGZyb20gY2FjaGVkIGRhdGEKICAgIGlmKGQuaW5zaWdodHMmJmQuaW5zaWdodHMuc2lnbmF0dXJlKXsKICAgICAgdmFyIHNpZz1kLmluc2lnaHRzLnNpZ25hdHVyZTsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLmluc2lnaHRzLnRhZ3MpdEVsLmlubmVySFRNTD1kLmluc2lnaHRzLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICAgIGlmKHJFbCYmZC5pbnNpZ2h0cy5yaXNpbmcmJmQuaW5zaWdodHMucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMucmlzaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgICBpZihmRWwmJmQuaW5zaWdodHMuZmFsbGluZyYmZC5pbnNpZ2h0cy5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIH0KCiAgICAvLyBSZW5kZXIgbWFwIGNvbG9ycyBhbmQgc3RyaXBzCiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICByZW5kZXJTdHJpcCgiM20iKTsKICAgIC8vIExvYWQgaW5zaWdodHMgdG9vCiAgICBidWlsZExvY2FsSW5zaWdodCgpOwogICAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7CiAgICAvLyBVc2UgY2FjaGVkIG5hcnJhdGl2ZSBpbnNpZ2h0IGlmIGF2YWlsYWJsZQogICAgaWYoZC5uYXJyYXRpdmVfaW5zaWdodCYmZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0KXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwogICAgICBpZihlbCkgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2QubmFycmF0aXZlX2luc2lnaHQudGV4dCsnPC9kaXY+JzsKICAgICAgaWYodEVsJiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRvcF9uYXJyYXRpdmVzKXsKICAgICAgfQogICAgfQogICAgcmV0dXJuIHRydWU7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbZnVsbC1zbmFwc2hvdF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gZmFsc2U7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKXsKICB0cnl7CiAgICAvLyBUcnkgY2FjaGVkIHZlcnNpb24gZnJvbSBmdWxsLXNuYXBzaG90IGZpcnN0IChhbHJlYWR5IGxvYWRlZCkKICAgIC8vIFRoZW4gY2FsbCBkZWRpY2F0ZWQgZW5kcG9pbnQgZm9yIGZyZXNoIEFJIGFuYWx5c2lzCiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9uYXJyYXRpdmUtaW5zaWdodCcpOwogICAgaWYoIXIub2spIHJldHVybjsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoIWQudGV4dCkgcmV0dXJuOwoKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwoKICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC50ZXh0Kyc8L2Rpdj4nOwoKICAgIC8vIFRhZ3MgZnJvbSB0b3AgbmFycmF0aXZlcwogICAgaWYodEVsJiZkLnRvcF9uYXJyYXRpdmVzJiZkLnRvcF9uYXJyYXRpdmVzLmxlbmd0aCl7CiAgICAgIHRFbC5pbm5lckhUTUw9ZC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICB2YXIgY29sPWk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjYpJzsKICAgICAgICB2YXIgYXJyb3c9aT09PTA/J+KGkSAnOifCtyAnOwogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpO2NvbG9yOicrY29sKyciPicrYXJyb3crbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9CgogICAgaWYobWV0YUVsKXsKICAgICAgdmFyIHQ9bmV3IERhdGUoZC5hc19vZik7CiAgICAgIG1ldGFFbC50ZXh0Q29udGVudD0nU2lnbmFsIGFuYWx5c2lzIMK3ICcrdC50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pKyhkLmZhbGxiYWNrPycgwrcgcGF0dGVybi1iYXNlZCc6JyDCtyBBSSBzeW50aGVzaXplZCcpOwogICAgfQogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tuYXJyYXRpdmVdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlKTt9KTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDEyMDAwMCk7CiAgICAgIHNldEludGVydmFsKGZldGNoSW5zaWdodHMsMzYwMDAwMCk7fQogIH0sMTUwMDApOwp9CgovLyBOQVJSQVRJVkUgREFUQQp2YXIgU0hJRlRTPXsKICAnM20nOlsKICAgIHtmYWRpbmc6J0luZmxhdGlvbicsZmFkaW5nTm90ZTonZWFzaW5nIG5hdGlvbmFsbHknLHJpc2luZzonQm9yZGVyIHNlY3VyaXR5JyxyaXNpbmdOb3RlOidwb3N0LWluY2lkZW50IHN1cmdlJ30sCiAgICB7ZmFkaW5nOidFbGVjdGlvbiByaGV0b3JpYycsZmFkaW5nTm90ZToncG9zdC1jeWNsZSBmYWRlJyxyaXNpbmc6J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3N0ZWFkeSByaXNlJ30sCiAgICB7ZmFkaW5nOidGYXJtZXIgcHJvdGVzdHMnLGZhZGluZ05vdGU6J21vbWVudHVtIGxvc3QnLHJpc2luZzonVW5lbXBsb3ltZW50IGFueGlldHknLHJpc2luZ05vdGU6J3lvdXRoIHNpZ25hbCBzdXJnZSd9LAogIF0sCiAgJzZtJzpbCiAgICB7ZmFkaW5nOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZhZGluZ05vdGU6J3ByZS1lbGVjdGlvbiBwZWFrJyxyaXNpbmc6J0NvcnJ1cHRpb24gYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3Bvc3QtY3ljbGUgcHVzaCd9LAogICAge2ZhZGluZzonUmVsaWdpb3VzIG5hdGlvbmFsaXNtJyxmYWRpbmdOb3RlOidwbGF0ZWF1IHBoYXNlJyxyaXNpbmc6J0Vjb25vbWljIGFueGlldHknLHJpc2luZ05vdGU6J2Nvc3Qtb2YtbGl2aW5nJ30sCiAgICB7ZmFkaW5nOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZmFkaW5nTm90ZToncmliYm9uLWN1dHRpbmcgZG9uZScscmlzaW5nOidMYXcgJiBvcmRlcicscmlzaW5nTm90ZTonY3JpbWUgbmFycmF0aXZlIHJpc2UnfSwKICBdLAogICcxeSc6WwogICAge2ZhZGluZzonUGFuZGVtaWMgcmVjb3ZlcnknLGZhZGluZ05vdGU6J2ZhZGVkIGVhcmx5IHllYXInLHJpc2luZzonSW5mbGF0aW9uJyxyaXNpbmdOb3RlOidkb21pbmF0ZWQgbWlkLXllYXInfSwKICAgIHtmYWRpbmc6J1JlZ2lvbmFsIGlkZW50aXR5JyxmYWRpbmdOb3RlOidsYW5ndWFnZS1sZWQgcGVhaycscmlzaW5nOidTZWN1cml0eSAmIGJvcmRlcnMnLHJpc2luZ05vdGU6J2dlb3BvbGl0aWNhbCBlc2NhbGF0aW9uJ30sCiAgICB7ZmFkaW5nOidHb3Zlcm5hbmNlIG9wdGltaXNtJyxmYWRpbmdOb3RlOidwb2xpY3kgaG9uZXltb29uIGVuZCcscmlzaW5nOidDb3JydXB0aW9uICYgc2NhbXMnLHJpc2luZ05vdGU6J2FjY291bnRhYmlsaXR5IGN5Y2xlJ30sCiAgXSwKfTsKdmFyIFJFR19TSElGVFM9WwogIHtzdGF0ZTonVGFtaWwgTmFkdScsZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLHRvOidGZWRlcmFsIHJlc291cmNlIGRpc3B1dGVzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonQmloYXInLGZyb206J0VsZWN0aW9uIHJoZXRvcmljJyx0bzonVW5lbXBsb3ltZW50ICYgZXhhbSBzY2FtcycsdGltZTonNiB3a3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2tzJ30sCiAge3N0YXRlOidSYWphc3RoYW4nLGZyb206J0Zhcm1lciBwcm90ZXN0cycsdG86J0hlYXQgd2F2ZSDCtyBFbnZpcm9ubWVudCcsdGltZTonMiB3a3MnfSwKICB7c3RhdGU6J0thcm5hdGFrYScsZnJvbTonTWluaW5nIGNvbnRyb3ZlcnN5Jyx0bzonTGFuZ3VhZ2Ugc2lnbmFnZSBwb2xpdGljcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyBpbmZyYXN0cnVjdHVyZScsdG86J0FpciBxdWFsaXR5IGNyaXNpcycsdGltZTonMTAgZGF5cyd9LAogIHtzdGF0ZTonTWFuaXB1cicsZnJvbTonR292ZXJuYW5jZSAmIGNhYmluZXQnLHRvOidFdGhuaWMgdGVuc2lvbnMgwrcgQUZTUEEnLHRpbWU6JzUgd2tzJ30sCiAge3N0YXRlOidQdW5qYWInLGZyb206J1Bvd2VyIGNyaXNpcycsdG86J0JvcmRlciBzZWN1cml0eSDCtyBEcm9uZXMnLHRpbWU6JzMgd2tzJ30sCl07CnZhciBNT0NLX1I9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbiBwcmVzc3VyZScsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2hpZnQtbGlzdCcpOwogIGlmKCFlbCkgcmV0dXJuOwogIHZhciBuYz17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIGlmKCFuY1tuLm5hbWVdKW5jW24ubmFtZV09e3VwOjAsZG93bjowLHN0YXRlczpbXX07CiAgICAgIG5jW24ubmFtZV1bbi5kaXI9PT0ndXAnPyd1cCc6J2Rvd24nXSs9KG4udmFsfHwwKTsKICAgICAgaWYobmNbbi5uYW1lXS5zdGF0ZXMuaW5kZXhPZihzLm5hbWV8fCcnKTwwKW5jW24ubmFtZV0uc3RhdGVzLnB1c2gocy5uYW1lfHwnJyk7CiAgICB9KTsKICB9KTsKICB2YXIgYWxsPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0udXArYlsxXS5kb3duKS0oYVsxXS51cCthWzFdLmRvd24pO30pOwogIHZhciByaXNpbmc9YWxsLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPj1rdlsxXS5kb3duO30pLnNsaWNlKDAsMyk7CiAgdmFyIGZhZGluZz1hbGwuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KTsKICBpZighZmFkaW5nLmxlbmd0aCkgZmFkaW5nPWFsbC5zbGljZSgtMyk7CiAgZmFkaW5nPWZhZGluZy5zbGljZSgwLDMpOwogIGlmKCFhbGwubGVuZ3RoKXsKICAgIGVsLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkNvbGxlY3Rpbmcgc2lnbmFsIGRhdGEuLi48L2Rpdj4nOwogICAgcmV0dXJuOwogIH0KICB2YXIgcm93cz1bXTsKICBmb3IodmFyIGk9MDtpPE1hdGgubWF4KHJpc2luZy5sZW5ndGgsZmFkaW5nLmxlbmd0aCwxKTtpKyspewogICAgdmFyIHI9cmlzaW5nW2ldLGY9ZmFkaW5nW2ldOwogICAgcm93cy5wdXNoKAogICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O292ZXJmbG93OmhpZGRlbjttYXJnaW4tYm90dG9tOjZweCI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjhweCAxMHB4O2JvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjojM2JiOGQ4O21hcmdpbi1ib3R0b206M3B4Ij5GQURJTkc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMCI+JysoZj9mWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2ZbMF0uc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj4nKwogICAgICAgIChmJiZmWzFdLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nK2ZbMV0uc3RhdGVzLnNsaWNlKDAsMikuam9pbignLCAnKSsnPC9kaXY+JzonJykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzowIDhweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zaXplOjE0cHgiPuKGkjwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo4cHggMTBweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjojZTA1YTI4O21hcmdpbi1ib3R0b206M3B4Ij5SSVNJTkc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMCI+Jysocj9yWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3JbMF0uc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj4nKwogICAgICAgIChyJiZyWzFdLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nK3JbMV0uc3RhdGVzLnNsaWNlKDAsMikuam9pbignLCAnKSsnPC9kaXY+JzonJykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JwogICAgKTsKICB9CiAgZWwuaW5uZXJIVE1MPXJvd3Muam9pbignJyk7Cn0KZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odGFiKXsKICB0YWIuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0KXt0LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgdGFiLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlbmRlclN0cmlwKHRhYi5kYXRhc2V0LnBlcmlvZCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gcmVuZGVyTW9tZW50dW0oKXsKICAvLyBSZWFkIGZyb20gU0QgKHBvcHVsYXRlZCBieSBmZXRjaEFsbFN0YXRlcyBmcm9tIGxpdmUgQVBJKQogIHZhciBuYz17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIG5jW24ubmFtZV09KG5jW24ubmFtZV18fDApK24udmFsOwogICAgfSk7CiAgfSk7CiAgdmFyIHNvcnRlZD1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgdmFyIHJpc2luZz1zb3J0ZWQuc2xpY2UoMCw1KTsKICB2YXIgZmFsbGluZz1zb3J0ZWQuc2xpY2UoLTUpLnJldmVyc2UoKTsKICB2YXIgbXg9cmlzaW5nLmxlbmd0aD9yaXNpbmdbMF1bMV06MTAwOwoKICAvLyBXcml0ZSB0byByaXNpbmctbGlzdCAobWF0Y2hlcyBuYXItcm93IEhUTUwpCiAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICBpZihyRWwmJnJpc2luZy5sZW5ndGgpewogICAgckVsLmlubmVySFRNTD1yaXNpbmcubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIGRlY2xpbmluZy1saXN0CiAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICBpZihmRWwmJmZhbGxpbmcubGVuZ3RoKXsKICAgIGZFbC5pbm5lckhUTUw9ZmFsbGluZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIHJlZ2lvbmFsLWxpc3Qg4oCUIHRvcCBzdGF0ZSBwZXIgcmVnaW9uIGZyb20gTElWRQogIHZhciByZWdpb25zPXsKICAgICdOb3J0aCc6WydEZWxoaScsJ1V0dGFyIFByYWRlc2gnLCdQdW5qYWInLCdIYXJ5YW5hJywnSGltYWNoYWwgUHJhZGVzaCcsJ1V0dGFyYWtoYW5kJywnSmFtbXUgYW5kIEthc2htaXInXSwKICAgICdFYXN0JzpbJ1dlc3QgQmVuZ2FsJywnQmloYXInLCdKaGFya2hhbmQnLCdPZGlzaGEnXSwKICAgICdXZXN0JzpbJ01haGFyYXNodHJhJywnR3VqYXJhdCcsJ1JhamFzdGhhbicsJ0dvYSddLAogICAgJ1NvdXRoJzpbJ1RhbWlsIE5hZHUnLCdLYXJuYXRha2EnLCdLZXJhbGEnLCdBbmRocmEgUHJhZGVzaCcsJ1RlbGFuZ2FuYSddLAogICAgJ05FJzpbJ0Fzc2FtJywnTWFuaXB1cicsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ1RyaXB1cmEnLCdBcnVuYWNoYWwgUHJhZGVzaCcsJ1Npa2tpbSddLAogICAgJ0NlbnRyYWwnOlsnTWFkaHlhIFByYWRlc2gnLCdDaGhhdHRpc2dhcmgnXSwKICB9OwogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICBpZihnRWwpewogICAgdmFyIHJlZ0l0ZW1zPU9iamVjdC5lbnRyaWVzKHJlZ2lvbnMpLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciByZWdpb249a3ZbMF0sc3RhdGVzPWt2WzFdOwogICAgICB2YXIgdG9wPXN0YXRlcy5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuIHtuYW1lOnMsYXR0OihMSVZFW3NdJiZMSVZFW3NdLmF0dGVudGlvbil8fDB9O30pCiAgICAgICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSlbMF07CiAgICAgIGlmKCF0b3B8fCF0b3AuYXR0KSByZXR1cm4gbnVsbDsKICAgICAgdmFyIG5hcj0oTElWRVt0b3AubmFtZV0mJkxJVkVbdG9wLm5hbWVdLmRvbWluYW50X25hcnJhdGl2ZSl8fCfigJQnOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6OHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCkiPicrcmVnaW9uKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+Jyt0b3AuYXR0LnRvRml4ZWQoMSkrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrdG9wLm5hbWUrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4OyI+JytuYXIrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuZmlsdGVyKEJvb2xlYW4pLmpvaW4oJycpOwogICAgaWYocmVnSXRlbXMpIGdFbC5pbm5lckhUTUw9cmVnSXRlbXM7CiAgfQp9CgoKLy8gU1RBVEUgREFUQQp2YXIgU0Q9e307Cgp2YXIgTElWRT17fTsKZnVuY3Rpb24gbm9ybWFsaXplRW1vdGlvbnMoZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJue307dmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhlKSx0b3Q9dmFscy5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcyt2O30sMCk7aWYodG90PD0wKXJldHVybnt9O2lmKHRvdDw9MS4wMSl7dmFyIG91dD17fTtPYmplY3Qua2V5cyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGspe291dFtrXT1NYXRoLnJvdW5kKGVba10qMTAwKTt9KTtyZXR1cm4gb3V0O31yZXR1cm4gZTt9CmZ1bmN0aW9uIGRvbWluYW50RW1vdGlvbihlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm4gbnVsbDt2YXIgbXg9MCxkb209bnVsbDtPYmplY3QuZW50cmllcyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXT5teCl7bXg9a3ZbMV07ZG9tPWt2WzBdO319KTtyZXR1cm4gZG9tO30KZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIWVsKXJldHVybjtlbC50ZXh0Q29udGVudD12YWw7aWYodmFsJiZ2YWwhPT0nLScpe2VsLmNsYXNzTGlzdC5yZW1vdmUoJ2xvYWRpbmcnKTt9fQoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjowLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7fSxkb21pbmFudF9lbW90aW9uOm51bGwsZG9taW5hbnRfbmFycmF0aXZlOm51bGwsCiAgbmFycmF0aXZlczpbXSxyaXNpbmc6W10sZmFsbGluZzpbXSwKICBzdW1tYXJ5OicnLGFydGljbGVzOltdLHRpbWVsaW5lOltdLAogIG5hcnJhdGl2ZUhpc3Rvcnk6W10sc2lnbmFsX2NvdW50OjAsCn07CgpmdW5jdGlvbiBnKG4pe3JldHVybiBTRFtuXXx8T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTt9CgpmdW5jdGlvbiBhQyhzKXsKICAvLyBEeW5hbWljIHNjYWxlOiBhbHdheXMgc3ByZWFkIGZ1bGwgY29sb3IgcmFuZ2UgYWNyb3NzIGFjdHVhbCBkYXRhCiAgLy8gR2V0IG1pbi9tYXggZnJvbSBjdXJyZW50IFNEIHRvIG5vcm1hbGl6ZQogIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKGQpe3JldHVybiBkLmF0dGVudGlvbnx8MDt9KTsKICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpOwogIHZhciBteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgLy8gTm9ybWFsaXplIDAtMQogIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwocy1tbikvKG14LW1uKSkpOwogIC8vIE1hcCB0byBjb2xvciBzdG9wczogZGFyayBibHVlIOKGkiB0ZWFsIOKGkiBhbWJlciDihpIgb3JhbmdlIOKGkiByZWQKICBpZihuPDAuMTIpIHJldHVybiAnIzBkMWUzMCc7CiAgaWYobjwwLjI1KSByZXR1cm4gJyMwZTNkNmEnOwogIGlmKG48MC4zOCkgcmV0dXJuICcjMGQ1ZjkwJzsKICBpZihuPDAuNTApIHJldHVybiAnIzBlN2FhYSc7CiAgaWYobjwwLjYyKSByZXR1cm4gJyMxYTkwOTAnOwogIGlmKG48MC43MikgcmV0dXJuICcjYzg3MDEwJzsKICBpZihuPDAuODIpIHJldHVybiAnI2Q4NDAxMCc7CiAgaWYobjwwLjkyKSByZXR1cm4gJyNjYzE4MDgnOwogIHJldHVybiAnI2ZmMDAxMCc7Cn0KZnVuY3Rpb24gZUMoZSl7CiAgdmFyIG14PTAsZG9tPSdwcmlkZSc7CiAgZm9yKHZhciBrIGluIGUpe2lmKGVba10+bXgpe214PWVba107ZG9tPWs7fX0KICByZXR1cm4gKHthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfSlbZG9tXXx8JyMzM2FhY2MnOwp9CmZ1bmN0aW9uIHZDKHYpewogIGlmKHY+MC4yKSByZXR1cm4gJyNkYzA4MTgnOwogIGlmKHY+MC4xKSByZXR1cm4gJyNlMDVhMjgnOwogIGlmKHY+MC4wMikgcmV0dXJuICcjY2M4ODIyJzsKICBpZih2PC0wLjA1KSByZXR1cm4gJyMyMjk5YmInOwogIHJldHVybiAnIzE1MjAzMCc7Cn0KCnZhciBsYXllcj0nYXR0ZW50aW9uJyxTRUw9bnVsbCxGQVZTPW5ldyBTZXQoKTsKCi8vIE1BUApmdW5jdGlvbiBwcm9qXyh3LGgscGFkKXsKICBwYWQ9cGFkfHwyMDsKICB2YXIgbWluTG9uPTY4LjEsbWF4TG9uPTk3LjQsbWluTGF0PTYuNSxtYXhMYXQ9MzcuMTsKICB2YXIgc2NYPSh3LXBhZCoyKS8obWF4TG9uLW1pbkxvbik7CiAgdmFyIHNjWT0oaC1wYWQqMikvKG1heExhdC1taW5MYXQpOwogIHZhciBzYz1NYXRoLm1pbihzY1gsc2NZKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLShtYXhMb24tbWluTG9uKSpzYykvMjsKICB2YXIgb3k9cGFkKyhoLXBhZCoyLShtYXhMYXQtbWluTGF0KSpzYykvMjsKICByZXR1cm4gZnVuY3Rpb24obG9uLGxhdCl7cmV0dXJuIFtveCsobG9uLW1pbkxvbikqc2MsIG95KyhtYXhMYXQtbGF0KSpzY107fTsKfQpmdW5jdGlvbiBnZW8ycGF0aChnZW9tLHBqKXsKICB2YXIgZD0nJzsKICBmdW5jdGlvbiByaW5nKGNzKXt2YXIgcz0nJztjcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cGooY1swXSxjWzFdKTtzKz0oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSk7cmV0dXJuIHMrJ1onO30KICBpZihnZW9tLnR5cGU9PT0nUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pOwogIGVsc2UgaWYoZ2VvbS50eXBlPT09J011bHRpUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihwKXtwLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTt9KTsKICByZXR1cm4gZDsKfQpmdW5jdGlvbiBjdHIoZ2VvbSl7CiAgdmFyIHB0cz1bXTsKICBmdW5jdGlvbiBjb2woYyl7aWYodHlwZW9mIGNbMF09PT0nbnVtYmVyJykgcHRzLnB1c2goYyk7ZWxzZSBjLmZvckVhY2goY29sKTt9CiAgY29sKGdlb20uY29vcmRpbmF0ZXMpOwogIGlmKCFwdHMubGVuZ3RoKSByZXR1cm4gWzAsMF07CiAgcmV0dXJuIFtwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFswXTt9LDApL3B0cy5sZW5ndGgscHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMV07fSwwKS9wdHMubGVuZ3RoXTsKfQpmdW5jdGlvbiBzTmFtZShwcm9wcyl7CiAgdmFyIHJhdz1wcm9wcy5zdF9ubXx8cHJvcHMuTkFNRV8xfHxwcm9wcy5uYW1lfHxwcm9wcy5OQU1FfHwnJzsKICB2YXIgbWFwPXsnTGFkYWtoJzonSmFtbXUgYW5kIEthc2htaXInLCdKYW1tdSAmIEthc2htaXInOidKYW1tdSBhbmQgS2FzaG1pcicsJ1V0dGFyYW5jaGFsJzonVXR0YXJha2hhbmQnLCdBbmRhbWFuIGFuZCBOaWNvYmFyJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnQW5kYW1hbiAmIE5pY29iYXIgSXNsYW5kJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnTkNUIG9mIERlbGhpJzonRGVsaGknLCdQb25kaWNoZXJyeSc6J1B1ZHVjaGVycnknLCdEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpJzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsJ0RhbWFuIGFuZCBEaXUnOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1J307CiAgcmV0dXJuIG1hcFtyYXddfHxyYXc7Cn0KCnZhciBjYWNoZWRHZW89bnVsbDsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoYXR0ZW1wdCl7CiAgYXR0ZW1wdCA9IGF0dGVtcHR8fDE7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goJ2h0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9naC91ZGl0LTAwMS9pbmRpYS1tYXBzLWRhdGFAbWFzdGVyL3RvcG9qc29uL2luZGlhLmpzb24nKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwxMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwzMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciw2MDAwKTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1ttYXBdIGxvYWQgZmFpbGVkIGF0dGVtcHQgJythdHRlbXB0Kyc6JyxlLm1lc3NhZ2UpOwogICAgaWYoYXR0ZW1wdDw1KXsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe2xvYWRNYXAoYXR0ZW1wdCsxKTt9LCBhdHRlbXB0KjIwMDApOwogICAgfSBlbHNlIHsKICAgICAgdmFyIG1pPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKTsKICAgICAgaWYobWkpIG1pLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6IzJhM2E0YTtwYWRkaW5nOjQwcHg7dGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6bW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4Ij5NYXAgdW5hdmFpbGFibGUg4oCUIHJlZnJlc2ggdG8gcmV0cnk8L2Rpdj4nOwogICAgfQogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKLy8gU2luZ2xlIHNvdXJjZSBvZiB0cnV0aCBmb3IgZW1vdGlvbiBjb2xvcgovLyBCb3RoIG1hcCBhbmQgcGFuZWwgY2FsbCB0aGlzIOKAlCBndWFyYW50ZWVzIHRoZXkgYWx3YXlzIG1hdGNoCmZ1bmN0aW9uIGdldEVmZmVjdGl2ZUVtb3Rpb24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgLy8gMS4gVHJ5IExJVkUuZG9taW5hbnRfZW1vdGlvbiAoc2V0IGJ5IC9hcGkvc3RhdGVzKQogIHZhciBkb209bGl2ZS5kb21pbmFudF9lbW90aW9ufHxkLmRvbWluYW50X2Vtb3Rpb247CgogIC8vIDIuIFRyeSBjb21wdXRpbmcgZnJvbSBlbW90aW9ucyBicmVha2Rvd24KICBpZighZG9tKXsKICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgICBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpOwogIH0KCiAgLy8gMy4gRmFsbGJhY2s6IGluZmVyIGZyb20gZG9taW5hbnQgbmFycmF0aXZlIChzYW1lIGxvZ2ljIGV2ZXJ5d2hlcmUpCiAgaWYoIWRvbSl7CiAgICB2YXIgbnA9KGxpdmUuZG9taW5hbnRfbmFycmF0aXZlfHxkLmRvbWluYW50X25hcnJhdGl2ZXx8JycpLnRvTG93ZXJDYXNlKCk7CiAgICBpZihucC5tYXRjaCgvYm9yZGVyfHRlcnJvcnxzZWN1cml0eXxjb25mbGljdHxhdHRhY2t8d2FyfGluZmlsdHJhdC8pKSBkb209J2ZlYXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvc2NhbXxjb3JydXB0fHByb3Rlc3R8YXJyZXN0fHZpb2xlbmNlfG91dHJhZ2V8Y3JpbWUvKSkgZG9tPSdhbmdlcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9kZXZlbG9wfGludmVzdHxncm93dGh8bGF1bmNofGluYXVndXJ8cmVmb3JtfHByb2dyZXNzfGJvb3N0LykpIGRvbT0naG9wZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9jdWx0dXJlfGhlcml0YWdlfHByaWRlfHZpY3Rvcnl8Y2VsZWJyYXR8bWVkYWx8YWNoaWV2ZW1lbnQvKSkgZG9tPSdwcmlkZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9mbG9vZHxkcm91Z2h0fHVuZW1wbG95bWVudHxpbmZsYXRpb258c2hvcnRhZ2V8Y3Jpc2lzfGNvbmNlcm4vKSkgZG9tPSdhbnhpZXR5JzsKICAgIGVsc2UgaWYoKGxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MCk+NSkgZG9tPSdhbnhpZXR5JzsgLy8gYWN0aXZlIHN0YXRlIGRlZmF1bHQKICAgIGVsc2UgZG9tPSdhbnhpZXR5JzsgLy8gZ2xvYmFsIGRlZmF1bHQKICB9CgogIHJldHVybiBkb207Cn0KCi8vIEdldCBlc3RpbWF0ZWQgZW1vdGlvbiBicmVha2Rvd24gKGZvciBwYW5lbCBkb251dCB3aGVuIHJlYWwgZGF0YSBtaXNzaW5nKQpmdW5jdGlvbiBnZXRFbW90aW9uQnJlYWtkb3duKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogIGlmKE9iamVjdC5rZXlzKGVtb3MpLmxlbmd0aCkgcmV0dXJuIHtlbW90aW9uczplbW9zLGVzdGltYXRlZDpmYWxzZX07CiAgLy8gQnVpbGQgc2tld2VkIGRpc3RyaWJ1dGlvbiBmcm9tIGVmZmVjdGl2ZSBlbW90aW9uCiAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICB2YXIgYmFzZT17YW54aWV0eToxMyxhbmdlcjoxMyxob3BlOjEzLHByaWRlOjEzLGZlYXI6MTN9OwogIGJhc2VbZG9tXT00ODsKICByZXR1cm4ge2Vtb3Rpb25zOmJhc2UsZXN0aW1hdGVkOnRydWV9Owp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyksZD1nKG5tKSxmaWxsOwogICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJykgZmlsbD1hQyhkLmF0dGVudGlvbik7CiAgICBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgZGU9Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgIGZpbGw9ZU1hcFtkZV18fCcjMzM0NDU1JzsKICAgIH0KICAgIGVsc2UgZmlsbD12QyhkLnZlbG9jaXR5KTsKICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsJyxmaWxsKTsKICAgIChmdW5jdGlvbigpewogICAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4geC5hdHRlbnRpb258fDA7fSk7CiAgICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogICAgICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKGQuYXR0ZW50aW9uLW1uKS8obXgtbW4pKSk7CiAgICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsLW9wYWNpdHknLGxheWVyPT09J2F0dGVudGlvbic/TWF0aC5tYXgoMC4zLDAuMytuKjAuNyk6MC44NSk7CiAgICB9KSgpOwogIH0pOwp9CgpmdW5jdGlvbiBhdHRhY2hJbnRlcmFjdGlvbnMoKXsKICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXsKICAgICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKTsKICAgICAgdmFyIGQ9ZyhubSk7CiAgICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgICAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogICAgICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGxhdGVzdD0nJzsKICAgICAgaWYoZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXMubGVuZ3RoKSBsYXRlc3Q9ZC5uYXJyYXRpdmVzWzBdLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZC5uYXJyYXRpdmVzWzBdLm5hbWUuc2xpY2UoMSk7CiAgICAgIGVsc2UgaWYobGl2ZS5kb21pbmFudF9uYXJyYXRpdmUpIGxhdGVzdD1saXZlLmRvbWluYW50X25hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStsaXZlLmRvbWluYW50X25hcnJhdGl2ZS5zbGljZSgxKTsKCiAgICAgIHZhciByb3dzPScnOwogICAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgICAgICB2YXIgYXR0PWxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MDsKICAgICAgICB2YXIgZGx0PWxpdmUuZGVsdGF8fGQuZGVsdGF8fDA7CiAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+QXR0ZW50aW9uPC9zcGFuPjxzdHJvbmc+JythdHQudG9GaXhlZCgxKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgIChkbHQhPT0wPyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj4yNGggc2hpZnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JysoZGx0PjA/JyNlMDVhMjgnOicjM2JiOGQ4JykrJyI+JysoZGx0PjA/JysnOicnKStkbHQrJzwvc3Ryb25nPjwvZGl2Pic6JycpKwogICAgICAgICAgKGxhdGVzdD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+VG9wIG5hcnJhdGl2ZTwvc3Bhbj48c3Ryb25nPicrbGF0ZXN0Kyc8L3N0cm9uZz48L2Rpdj4nOicnKTsKICAgICAgfSBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgICB2YXIgZG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICAgIGlmKGRvbUVtbyl7CiAgICAgICAgICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczpkLmVtb3Rpb25zfHx7fTsKICAgICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRvbWluYW50PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrcGFsW2RvbUVtb10rJyI+Jytkb21FbW8uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vLnNsaWNlKDEpKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICAgICAgdmFyIGVMPU9iamVjdC5lbnRyaWVzKGVtb3MpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgICAgICAgdmFyIHRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgICAgICAgIGlmKHRvdD4wJiZ0b3Q8PTEuMDEpe2VMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7dG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApO30KICAgICAgICAgIHJvd3MrPWVMLnNsaWNlKDAsMykubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm4gJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo0cHgiPjxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrcGFsW2t2WzBdXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JytrdlswXSsnPC9zcGFuPjxzdHJvbmc+JytNYXRoLnJvdW5kKGt2WzFdKjEwMC9NYXRoLm1heCgxLHRvdCkpKyclPC9zdHJvbmc+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgdmFyIHZlbD1saXZlLnZlbG9jaXR5fHxkLnZlbG9jaXR5fHwwOwogICAgICAgIHZhciB2ZWxEaXI9dmVsPjAuMT8nUmlzaW5nIGZhc3QnOnZlbD4wLjAyPydSaXNpbmcnOnZlbDwtMC4wNT8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICAgICAgdmFyIHZlbENvbD12ZWw+MC4wMj8nI2UwNWEyOCc6dmVsPC0wLjAyPycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+TW9tZW50dW08L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5EaXJlY3Rpb248L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2ZWxDb2wrJyI+Jyt2ZWxEaXIrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgIH0KCiAgICAgIHRpcC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InR0LW4iPicrbm0rJzwvZGl2Picrcm93cysobGF0ZXN0JiZsYXllciE9PSdhdHRlbnRpb24nPyc8ZGl2IGNsYXNzPSJ0dC1uYXIiPjxzdHJvbmc+TmFycmF0aXZlPC9zdHJvbmc+JytsYXRlc3QrJzwvZGl2Pic6JycpOwogICAgICB2YXIgcmVjdD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJykuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICAgIHRpcC5zdHlsZS5sZWZ0PU1hdGgubWluKGUuY2xpZW50WC1yZWN0LmxlZnQrMTQscmVjdC53aWR0aC0xOTApKydweCc7CiAgICAgIHRpcC5zdHlsZS50b3A9TWF0aC5taW4oZS5jbGllbnRZLXJlY3QudG9wKzE0LHJlY3QuaGVpZ2h0LTE1MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLm9wYWNpdHk9JzEnOwogICAgfSk7CnAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VsZWF2ZScsZnVuY3Rpb24oKXt0aXAuc3R5bGUub3BhY2l0eT0wO30pOwogICAgcC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXtzZWxlY3RfKHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSk7fSk7CiAgfSk7Cn0KCi8vIFNUQVRFIFBBTkVMCmZ1bmN0aW9uIHNlbGVjdF8obm0pewogIFNFTD1ubTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5jbGFzc0xpc3QudG9nZ2xlKCdzZWxlY3RlZCcscC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpPT09bm0pOwogIH0pOwogIC8vIFNob3cgbG9hZGluZyBzdGF0ZSBpbW1lZGlhdGVseSB3aXRoIHdoYXRldmVyIExJVkUgZGF0YSB3ZSBoYXZlCiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZihwYW5lbCl7CiAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICBwYW5lbC5pbm5lckhUTUw9CiAgICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAgICc8ZGl2PjxkaXYgY2xhc3M9InNwLWVrIj4nKyhsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICAgJzwvYnV0dG9uPicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA4ZW0iPicrCiAgICAgICAgJ0xvYWRpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJysKICAgICAgICAobGl2ZS5hdHRlbnRpb24/Jzxicj48YnI+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtjb2xvcjp2YXIoLS1pbmspIj5BdHRlbnRpb24gJytsaXZlLmF0dGVudGlvbi50b0ZpeGVkKDEpKyc8L3NwYW4+JzonJykrCiAgICAgICAgKGxpdmUuZG9taW5hbnRfZW1vdGlvbj8nPGJyPjxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrbGl2ZS5kb21pbmFudF9lbW90aW9uKycgc2lnbmFsIGRvbWluYW50PC9zcGFuPic6JycpKwogICAgICAnPC9kaXY+JzsKICB9CiAgLy8gRmV0Y2ggZnVsbCBkZXRhaWwgdGhlbiByZW5kZXIKICBmZXRjaERldGFpbChubSkudGhlbihmdW5jdGlvbigpewogICAgaWYoU0VMPT09bm0pewogICAgICByZW5kZXJQYW5lbChubSk7CiAgICAgIC8vIFVwZGF0ZSBqdXN0IHRoaXMgc3RhdGUncyBtYXAgY29sb3IgdG8gbWF0Y2ggdGhlIHBhbmVsCiAgICAgIHZhciBwYXRoPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9Iicrbm0rJyJdJyk7CiAgICAgIGlmKHBhdGgmJmxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICAgICAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICAgIHZhciBkb209bGl2ZS5kb21pbmFudF9lbW90aW9ufHxkb21pbmFudEVtb3Rpb24obGl2ZS5lbW90aW9uc3x8e30pOwogICAgICAgIGlmKGRvbSYmZU1hcFtkb21dKSBwYXRoLnNldEF0dHJpYnV0ZSgnZmlsbCcsZU1hcFtkb21dKTsKICAgICAgfSBlbHNlIHsKICAgICAgICBhcHBseUxheWVyKCk7CiAgICAgIH0KICAgIH0KICB9KS5jYXRjaChmdW5jdGlvbihlKXsKICAgIGNvbnNvbGUud2FybignW3NlbGVjdF0nLGUpOwogICAgaWYoU0VMPT09bm0pIHJlbmRlclBhbmVsKG5tKTsKICB9KTsKfQoKZnVuY3Rpb24gcmVuZGVyUGFuZWwobm0pewogIHZhciBkPWcobm0pOwogIHZhciBwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYoIXBhbmVsKSByZXR1cm47CiAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogIHZhciBoZWFkZXI9CiAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgJzxkaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtZWsiIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7Ij4nKwogICAgICAgICAgKGxheWVyPT09J2F0dGVudGlvbic/J05hcnJhdGl2ZSBwYW5lbCc6bGF5ZXI9PT0nZW1vdGlvbic/J0Vtb3Rpb25hbCByZWdpc3Rlcic6J01vbWVudHVtIHBhbmVsJykrCiAgICAgICAgICAoZC5jb25maWRlbmNlPyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8ncmdiYSg1MSwyMDQsMTAyLDAuMSknOmQuY29uZmlkZW5jZT09PSdNRURJVU0nPydyZ2JhKDIyNCw5MCw0MCwwLjEpJzoncmdiYSgyNTUsMjU1LDI1NSwwLjA0KScpKyc7Y29sb3I6JysoZC5jb25maWRlbmNlPT09J0hJR0gnPycjMzNjYzY2JzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8nI2UwNWEyOCc6J3JnYmEoMjU1LDI1NSwyNTUsMC4zKScpKyciPicrZC5jb25maWRlbmNlKycgU0lHTkFMPC9zcGFuPic6JycpKwogICAgICAgICAgKGQuaXNfcmVnaW9uYWxfc3Rvcnk/JzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07cGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMSk7Y29sb3I6IzNiYjhkOCI+UkVHSU9OQUwgU1BJS0U8L3NwYW4+JzonJykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhGQVZTLmhhcyhubSk/J29uJzonJykrJyIgZGF0YS1ubT0iJytubSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYodGhpcy5kYXRhc2V0Lm5tKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoRkFWUy5oYXMobm0pPydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgJzwvYnV0dG9uPicrCiAgICAnPC9kaXY+JzsKCiAgdmFyIGJvZHk9Jyc7CgogIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgdmFyIGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/J3VwJzonZG4nOwogICAgdmFyIG5hcnI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0bD0oZC50aW1lbGluZSYmZC50aW1lbGluZS5sZW5ndGgpP2QudGltZWxpbmU6WzAsMCwwLDAsMCwwLDAsZC5hdHRlbnRpb258fDBdOwogICAgdmFyIHRtbj1NYXRoLm1pbi5hcHBseShudWxsLHRsKSx0bXg9TWF0aC5tYXguYXBwbHkobnVsbCx0bCksdHI9TWF0aC5tYXgoMSx0bXgtdG1uKTsKICAgIHZhciB0dz0yNjAsdGg9NjIsdHA9NTsKICAgIHZhciBwdHM9dGwubWFwKGZ1bmN0aW9uKHYsaSl7cmV0dXJuW3RwKyhpLyh0bC5sZW5ndGgtMSkpKih0dy10cCoyKSx0cCsoMS0odi10bW4pL3RyKSoodGgtdHAqMildO30pOwogICAgdmFyIHBEPXB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSkuam9pbignJyk7CiAgICB2YXIgYUQ9cEQrJyBMJytwdHNbcHRzLmxlbmd0aC0xXVswXSsnLCcrKHRoLXRwKSsnIEwnK3B0c1swXVswXSsnLCcrKHRoLXRwKSsnIFonOwogICAgdmFyIGFjPWFDKGQuYXR0ZW50aW9ufHwwKTsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnSG93IGludGVuc2VseSAnKyhubS5zcGxpdCgnICcpWzBdKSsnIGlzIGJlaW5nIGRpc2N1c3NlZCBuYXRpb25hbGx5LiBTY29yZSBvZiAnK2QuYXR0ZW50aW9uKycgbWVhbnMgJysoZC5hdHRlbnRpb24+NjA/J3ZlcnkgaGlnaCDigJQgdGhpcyBzdGF0ZSBkb21pbmF0ZXMgbmF0aW9uYWwgZGlzY291cnNlJzpkLmF0dGVudGlvbj4zNT8nZWxldmF0ZWQg4oCUIGNsZWFybHkgaW4gdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbic6ZC5hdHRlbnRpb24+MTU/J21vZGVyYXRlIOKAlCBzb21lIG5hdGlvbmFsIGNvdmVyYWdlJzpkLmF0dGVudGlvbj41Pydsb3cg4oCUIGxpbWl0ZWQgbmF0aW9uYWwgYXR0ZW50aW9uJzonbWluaW1hbCDigJQgZmV3IHNpZ25hbHMgZGV0ZWN0ZWQnKSsnLicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJpbnNpZ2h0IiBzdHlsZT0iJysoZC5jb25maWRlbmNlPT09IkxPVyI/J2JvcmRlci1jb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMDYpO2NvbG9yOnZhcigtLWZhaW50KTtmb250LXN0eWxlOml0YWxpYyc6JycpKyciPicrKChkLmNvbmZpZGVuY2U9PT0iTE9XIiYmIWQuc3VtbWFyeSk/J0xpbWl0ZWQgc2lnbmFscyBkZXRlY3RlZCBmb3IgJytubSsnLiBNb25pdG9yaW5nIHJlZ2lvbmFsIHNvdXJjZXMuJzpkLnN1bW1hcnl8fCdDb2xsZWN0aW5nIHNpZ25hbHMgZm9yICcrbm0rJy4uLicpKyc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic2NvcmUtc3RyaXAiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiPicrKGQuYXR0ZW50aW9ufHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCBzaGlmdDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrZEMrJyI+JytkUysoZC5kZWx0YXx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5Ub3AgbmFycmF0aXZlPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhuYXJyWzBdP25hcnJbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuYXJyWzBdLm5hbWUuc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+TmFycmF0aXZlIGJyZWFrZG93bjwvZGl2PicrCiAgICAgICAgKG5hcnIubGVuZ3RoPwogICAgICAgICAgJzxkaXYgY2xhc3M9Im5hci1saXN0Ij4nK25hcnIubWFwKGZ1bmN0aW9uKG4pewogICAgICAgICAgICB2YXIgbm49bi5uYW1lP24ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hbWUuc2xpY2UoMSk6bi5uYW1lOwogICAgICAgICAgICB2YXIgdmFsPXR5cGVvZiBuLnZhbD09PSdudW1iZXInP24udmFsOjA7CiAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0yIj48ZGl2IGNsYXNzPSJuaS1sYWJlbCI+Jytubisobi5kaXI9PT0ndXAnPycgPHNwYW4gc3R5bGU9ImNvbG9yOiNlMDVhMjg7Zm9udC1zaXplOjlweCIgdGl0bGU9ImdhaW5pbmcgdHJhY3Rpb24iPuKGkTwvc3Bhbj4nOm4uZGlyPT09J2Rvd24nPycgPHNwYW4gc3R5bGU9ImNvbG9yOiMzYmI4ZDg7Zm9udC1zaXplOjlweCIgdGl0bGU9InJldHJlYXRpbmciPuKGkzwvc3Bhbj4nOicnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdmFsIj4nK3ZhbC50b0ZpeGVkKDEpKyclPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCx2YWwqMi41KSsnJTtiYWNrZ3JvdW5kOicrKG4uZGlyPT09J3VwJz8nI2UwNWEyOCc6bi5kaXI9PT0nZG93bic/JyMzYmI4ZDgnOicjMzM0NDU1JykrJyI+PC9kaXY+PC9kaXY+PC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpKyc8L2Rpdj4nOgogICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb3ctc2lnbmFsIHJlZ2lvbi4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLjwvZGl2PicpKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QXR0ZW50aW9uIOKAlCA4IGRheXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0bC13cmFwIj48c3ZnIHZpZXdCb3g9IjAgMCAnK3R3KycgJyt0aCsnIiBzdHlsZT0id2lkdGg6MTAwJTtoZWlnaHQ6MTAwJSI+JysKICAgICAgICAgICc8ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9InRsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwLjI1Ii8+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAiLz4nKwogICAgICAgICAgJzwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPicrCiAgICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnKSIgLz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrcEQrJyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSInK2FjKyciIHN0cm9rZS13aWR0aD0iMS4yIi8+JysKICAgICAgICAgIHB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gJzxjaXJjbGUgY3g9IicrcFswXSsnIiBjeT0iJytwWzFdKyciIHI9IicrKGk9PT1wdHMubGVuZ3RoLTE/Mi4yOjEuMikrJyIgZmlsbD0iJythYysnIi8+Jzt9KS5qb2luKCcnKSsKICAgICAgICAnPC9zdmc+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JysoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGg/ZC5hcnRpY2xlcy5sZW5ndGg6MCkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtbGlzdCI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLm1hcChmdW5jdGlvbihhKXtyZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIj48ZGl2IGNsYXNzPSJhcnQtc3JjIj4nKyhhLnNyY3x8JycpKyc8L2Rpdj48ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NnB4IDAiPk5vIHNpZ25hbHMgY29sbGVjdGVkIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAvLyBVc2Ugc2FtZSBmdW5jdGlvbnMgYXMgbWFwIOKAlCBndWFyYW50ZWVkIHRvIG1hdGNoCiAgICB2YXIgbWFwRG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgdmFyIGJyZWFrZG93bj1nZXRFbW90aW9uQnJlYWtkb3duKG5tKTsKICAgIHZhciBlbW90aW9ucz1icmVha2Rvd24uZW1vdGlvbnM7CiAgICB2YXIgaGFzRW1vcz0hYnJlYWtkb3duLmVzdGltYXRlZDsKICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW90aW9ucyk7CiAgICB2YXIgZVRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgIGlmKGVUb3Q+MCYmZVRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt9CiAgICB2YXIgdG90PU1hdGgubWF4KDEsZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogICAgZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICBpZighZUwubGVuZ3RoKXtwYW5lbC5pbm5lckhUTUw9aGVhZGVyKyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4Ij5ObyBlbW90aW9uIGRhdGEgeWV0LjwvZGl2Pic7cmV0dXJuO30KICAgIC8vIGRvbUVtbyA9IHNhbWUgYXMgbWFwIGNvbG9yIChmcm9tIGdldEVmZmVjdGl2ZUVtb3Rpb24pCiAgICB2YXIgZG9tRW1vPW1hcERvbUVtbzsKICAgIC8vIFJlb3JkZXIgZUwgc28gZG9taW5hbnQgc2hvd3MgZmlyc3QKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXsKICAgICAgaWYoYVswXT09PWRvbUVtbykgcmV0dXJuIC0xOwogICAgICBpZihiWzBdPT09ZG9tRW1vKSByZXR1cm4gMTsKICAgICAgcmV0dXJuIGJbMV0tYVsxXTsKICAgIH0pOwogICAgdmFyIGRvbVBjdD1NYXRoLnJvdW5kKChlTFswXT9lTFswXVsxXToyMCkqMTAwL3RvdCk7CiAgICB2YXIgbmFycjI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0b3BOYXJTdHI9bmFycjIuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCcgYW5kICcpOwogICAgdmFyIHdoYXRJdD17YW54aWV0eTonQSBkaWZmdXNlIHVuZWFzZSBpcyBydW5uaW5nIHRocm91Z2ggc2lnbmFscyBmcm9tICcrbm0rKHRvcE5hclN0cj8nLCBjb25jZW50cmF0ZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFNpZ25hbHMgYXQgdGhpcyBzdGFnZSB0ZW5kIHRvIGJlIGxvY2FsbHkgYWJzb3JiZWQgYmVmb3JlIHdpZGVuaW5nLic6Jy4nICApLGFuZ2VyOidGcnVzdHJhdGlvbiBzaWduYWxzIGFyZSBlbGV2YXRlZCBpbiAnK25tKyh0b3BOYXJTdHI/JywgcGFydGljdWxhcmx5IGFyb3VuZCAnK3RvcE5hclN0cisnLiBUaGUgdG9uZSBzdWdnZXN0cyBwcmVzc3VyZSBidWlsZGluZyByYXRoZXIgdGhhbiBhIHNpbmdsZSBldmVudC4nOicuIFRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgaXMgbm90aWNlYWJseSB0ZW5zZS4nKSxob3BlOidBbiB1bnVzdWFsbHkgb3B0aW1pc3RpYyBzaWduYWwgcmVnaXN0ZXIgZnJvbSAnK25tKyh0b3BOYXJTdHI/Jywgb3JpZW50ZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFdvcnRoIHdhdGNoaW5nIOKAlCBwb3NpdGl2ZSBzaWduYWxzIGF0IHRoaXMgZGVuc2l0eSBhcmUgcmVsYXRpdmVseSByYXJlLic6Jy4gQSBzaWduYWwgd29ydGggbW9uaXRvcmluZy4nKSxwcmlkZTonU3Ryb25nIGlkZW50aXR5IHNpZ25hbHMgaW4gJytubSsodG9wTmFyU3RyPycsIGNlbnRyZWQgYXJvdW5kICcrdG9wTmFyU3RyKycuIFJlZ2lvbmFsbHkgY29uY2VudHJhdGVkIGFuZCBlbW90aW9uYWxseSBkZW5zZS4nOicuIExvY2FsbHkgY29uY2VudHJhdGVkLCBlbW90aW9uYWxseSBzdHJvbmcuJyksZmVhcjonQXBwcmVoZW5zaW9uIHNpZ25hbHMgaW4gJytubSsodG9wTmFyU3RyPycsIGFyb3VuZCAnK3RvcE5hclN0cisnLiBUaGVzZSB0ZW5kIHRvIGludGVuc2lmeSBiZWZvcmUgYWNoaWV2aW5nIHdpZGVyIHZpc2liaWxpdHkuJzonLiBUaGUgcmVnaXN0ZXIgY2FycmllcyBhbiBlZGdlIHRoYXQgdGVuZHMgdG8gcHJlY2VkZSBsYXJnZXIgY3ljbGVzLicpfTsKICAgIHZhciBjdW1BPS1NYXRoLlBJLzIsY3g9MzgsY3k9MzgsUj0zMyxyaT0yMDsKICAgIHZhciBhcmNzPWVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0sZnI9di90b3QsYTE9Y3VtQSxhMj1jdW1BK2ZyKk1hdGguUEkqMjtjdW1BPWEyOwogICAgICB2YXIgbGc9KGEyLWExKT5NYXRoLlBJPzE6MDsKICAgICAgdmFyIHgxPWN4K01hdGguY29zKGExKSpSLHkxPWN5K01hdGguc2luKGExKSpSLHgyPWN4K01hdGguY29zKGEyKSpSLHkyPWN5K01hdGguc2luKGEyKSpSOwogICAgICB2YXIgeDM9Y3grTWF0aC5jb3MoYTIpKnJpLHkzPWN5K01hdGguc2luKGEyKSpyaSx4ND1jeCtNYXRoLmNvcyhhMSkqcmkseTQ9Y3krTWF0aC5zaW4oYTEpKnJpOwogICAgICByZXR1cm4gJzxwYXRoIGQ9Ik0nK3gxLnRvRml4ZWQoMSkrJywnK3kxLnRvRml4ZWQoMSkrJyBBJytSKycsJytSKycgMCAnK2xnKycgMSAnK3gyLnRvRml4ZWQoMSkrJywnK3kyLnRvRml4ZWQoMSkrJyBMJyt4My50b0ZpeGVkKDEpKycsJyt5My50b0ZpeGVkKDEpKycgQScrcmkrJywnK3JpKycgMCAnK2xnKycgMCAnK3g0LnRvRml4ZWQoMSkrJywnK3k0LnRvRml4ZWQoMSkrJyBaIiBmaWxsPSInK3BhbFtrXSsnIiBvcGFjaXR5PSIwLjkiLz4nOwogICAgfSkuam9pbignJyk7CiAgICB2YXIgZWRlc2M9e2FueGlldHk6J0RpZmZ1c2UgdW5lYXNlLCB3b3JyeSBzaWduYWxzJyxhbmdlcjonRnJ1c3RyYXRpb24sIHByZXNzdXJlIHNpZ25hbHMnLGhvcGU6J09wdGltaXNtLCBmb3J3YXJkIG1vbWVudHVtJyxwcmlkZTonSWRlbnRpdHksIHJlZ2lvbmFsIGFzc2VydGlvbicsZmVhcjonQXBwcmVoZW5zaW9uLCB0aHJlYXQgcGVyY2VwdGlvbid9OwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdUaGUgZW1vdGlvbmFsIHJlZ2lzdGVyIG9mIHNpZ25hbHMgZnJvbSAnK25tKycg4oCUIHdoYXQgdG9uZSBydW5zIHRocm91Z2ggdGhlIGRpc2NvdXJzZSBhbmQgaG93IGNvbmNlbnRyYXRlZCBpdCBpcy4nKwogICAgJzwvZGl2PicrCiAgICAoIWhhc0Vtb3M/JzxkaXYgc3R5bGU9InBhZGRpbmc6NnB4IDExcHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTttYXJnaW4tYm90dG9tOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCkiPkVzdGltYXRlZCBmcm9tIHNpZ25hbCBkaXJlY3Rpb24g4oCUIGxpbWl0ZWQgZGlyZWN0IGVtb3Rpb24gZGF0YS48L2Rpdj4nOicnKSsKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MTRweDtib3JkZXItcmFkaXVzOjEwcHg7YmFja2dyb3VuZDonK3BhbFtkb21FbW9dKycxNDtib3JkZXI6MXB4IHNvbGlkICcrcGFsW2RvbUVtb10rJzMzO21hcmdpbi1ib3R0b206MTJweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOicrcGFsW2RvbUVtb10rJzttYXJnaW4tYm90dG9tOjZweCI+RG9taW5hbnQgZW1vdGlvbjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjZweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+Jytkb21FbW8uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4Ij4nK2RvbVBjdCsnJSDCtyAnK25tKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6OHB4O2xpbmUtaGVpZ2h0OjEuNTtmb250LXN0eWxlOml0YWxpYyI+Jyt3aGF0SXRbZG9tRW1vXSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkVtb3Rpb25hbCBicmVha2Rvd248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OyI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCA3NiA3NiIgc3R5bGU9IndpZHRoOjcycHg7aGVpZ2h0OjcycHg7ZmxleC1zaHJpbms6MCI+JythcmNzKyc8L3N2Zz4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHg7Ij4nKwogICAgICAgICAgICBlTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICAgICAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0scGN0PU1hdGgucm91bmQodioxMDAvdG90KTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6Y2VudGVyO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweDsiPjxzcGFuIHN0eWxlPSJ3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjJweDtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMS41cHg7Y29sb3I6Jysoaz09PWRvbUVtbz8ndmFyKC0taW5rKSc6J3ZhcigtLWRpbSknKSsnIj4nK2suY2hhckF0KDApLnRvVXBwZXJDYXNlKCkray5zbGljZSgxKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2YXIoLS1pbmspIj4nK3BjdCsnJTwvc3Bhbj4nKwogICAgICAgICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3BjdCsnJTtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7b3BhY2l0eTowLjc7Ym9yZGVyLXJhZGl1czoxcHgiPjwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgICAoaz09PWRvbUVtbz8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrZWRlc2Nba10rJzwvZGl2Pic6JycpKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFsIGhlYWRsaW5lczwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweDsiPicrCiAgICAgICAgICAoKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoKT8KICAgICAgICAgICAgZC5hcnRpY2xlcy5zbGljZSgwLDUpLm1hcChmdW5jdGlvbihhKXsKICAgICAgICAgICAgICB2YXIgZUNvbG9yPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDo2cHg7cGFkZGluZzo2cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpOyI+JysKICAgICAgICAgICAgICAgIChhLmVtb3Rpb24/JzxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrZUNvbG9yW2EuZW1vdGlvbl0rJztkaXNwbGF5OmlubGluZS1ibG9jazttYXJnaW4tdG9wOjVweDtmbGV4LXNocmluazowIj48L3NwYW4+JzonJykrCiAgICAgICAgICAgICAgICAnPGRpdj48ZGl2IHN0eWxlPSJmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNCI+JysoYS50eHR8fGEudGl0bGV8fCcnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nKyhhLnNyY3x8JycpKyhhLmVtb3Rpb24/JyDCtyAnK2EuZW1vdGlvbjonJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NHB4IDAiPk5vIHNpZ25hbHMgeWV0LjwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKCiAgfSBlbHNlIHsKICAgIHZhciB2ZWw9ZC52ZWxvY2l0eXx8MDsKICAgIHZhciB2ZWxEaXI9dmVsPjAuMTU/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wNT8nUmlzaW5nJzp2ZWw8LTAuMT8nQ29vbGluZyBmYXN0Jzp2ZWw8LTAuMDI/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgdmFyIHZlbENvbD12ZWw+MC4wNT8nI2UwNWEyOCc6dmVsPC0wLjAyPycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICB2YXIgdmVsRGVzYz17J1Jpc2luZyBmYXN0JzonU2lnbmFsIHZvbHVtZSBhY2NlbGVyYXRpbmcgc2hhcnBseSDigJQgdGhpcyBzdGF0ZSBpcyBlbnRlcmluZyBhbiBhY3RpdmUgZGlzY291cnNlIGN5Y2xlLicsJ1Jpc2luZyc6J0F0dGVudGlvbiBpcyBidWlsZGluZyDigJQgc2lnbmFscyBzdWdnZXN0IGEgbmFycmF0aXZlIGdhaW5pbmcgcmVnaW9uYWwgdHJhY3Rpb24uJywnU3RhYmxlJzonU2lnbmFsIGFjdGl2aXR5IGhvbGRpbmcgc3RlYWR5IOKAlCBubyBzaWduaWZpY2FudCBhY2NlbGVyYXRpb24gb3IgcmV0cmVhdCBkZXRlY3RlZC4nLCdDb29saW5nJzonQXR0ZW50aW9uIGJlZ2lubmluZyB0byBlYXNlIOKAlCB0aGUgY3VycmVudCBuYXJyYXRpdmUgY3ljbGUgbWF5IGJlIHJ1bm5pbmcgaXRzIGNvdXJzZS4nLCdDb29saW5nIGZhc3QnOidTaWduYWwgdm9sdW1lIHJldHJlYXRpbmcgcXVpY2tseSDigJQgYXR0ZW50aW9uIGhhcyBsaWtlbHkgcGVha2VkIGFuZCBpcyBkaXNwZXJzaW5nLid9OwogICAgdmFyIG5hcnIzPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgcmlzaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J3VwJzt9KTsKICAgIHZhciBmYWxsaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J2Rvd24nO30pOwogICAgdmFyIGN0eD0nJzsKICAgIGlmKHZlbD4wLjA1JiZyaXNpbmdOYXJzLmxlbmd0aCkgY3R4PSdDb25jZW50cmF0ZWQgYXJvdW5kIDxlbT4nK3Jpc2luZ05hcnMuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCc8L2VtPiBhbmQgPGVtPicpKyc8L2VtPiDigJQgdGhlc2Ugc2lnbmFscyBhcmUgZ2FpbmluZyBtb21lbnR1bSBhbmQgbWF5IGF0dHJhY3QgYnJvYWRlciBhdHRlbnRpb24uJzsKICAgIGVsc2UgaWYodmVsPC0wLjA1JiZmYWxsaW5nTmFycy5sZW5ndGgpIGN0eD0nU2lnbmFscyBhcm91bmQgPGVtPicrZmFsbGluZ05hcnMuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCc8L2VtPiBhbmQgPGVtPicpKyc8L2VtPiBhcmUgcmV0cmVhdGluZyDigJQgdGhlIGRpc2NvdXJzZSBjeWNsZSBhcHBlYXJzIHRvIGJlIGNvbXBsZXRpbmcuJzsKICAgIGVsc2UgaWYodmVsPjAuMDIpIGN0eD0nU2lnbmFscyBpbiAnK25tKycgYXJlIGJ1aWxkaW5nIGFjcm9zcyBtdWx0aXBsZSBuYXJyYXRpdmVzIOKAlCBubyBzaW5nbGUgZG9taW5hbnQgdGhyZWFkIHlldCwgYnV0IG1vbWVudHVtIGlzIHByZXNlbnQuJzsKICAgIGVsc2UgaWYodmVsPC0wLjAyKSBjdHg9J1NpZ25hbCBhY3Rpdml0eSBpbiAnK25tKycgaXMgZWFzaW5nIOKAlCBhdHRlbnRpb24gYXBwZWFycyB0byBiZSBzaGlmdGluZyB0b3dhcmQgb3RoZXIgcmVnaW9uYWwgc3Rvcmllcy4nOwogICAgZWxzZSBjdHg9J1NpZ25hbHMgZnJvbSAnK25tKycgaG9sZGluZyBzdGVhZHkg4oCUIGJldHdlZW4gY3ljbGVzLCBubyBzdHJvbmcgYWNjZWxlcmF0aW9uIG9yIHJldHJlYXQgZGV0ZWN0ZWQuJzsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnU2lnbmFsIHZlbG9jaXR5IGZvciAnK25tKycg4oCUIHdoZXRoZXIgYXR0ZW50aW9uIGlzIGJ1aWxkaW5nLCBob2xkaW5nLCBvciBiZWdpbm5pbmcgdG8gcmV0cmVhdCBmcm9tIHRoZSBjdXJyZW50IGN5Y2xlLicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjE0cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6Jyt2ZWxDb2wrJzE0O2JvcmRlcjoxcHggc29saWQgJyt2ZWxDb2wrJzMzO21hcmdpbi1ib3R0b206MTJweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOicrdmVsQ29sKyc7bWFyZ2luLWJvdHRvbTo2cHgiPlNpZ25hbCBtb21lbnR1bTwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtnYXA6MTBweDttYXJnaW4tYm90dG9tOjhweDsiPicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTozMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L2Rpdj4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxNHB4O2NvbG9yOicrdmVsQ29sKyc7Zm9udC13ZWlnaHQ6NTAwIj4nK3ZlbERpcisnPC9kaXY+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXN0eWxlOml0YWxpYztsaW5lLWhlaWdodDoxLjUiPicrdmVsRGVzY1t2ZWxEaXJdKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTEuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42O21hcmdpbi10b3A6MTBweDtwYWRkaW5nLXRvcDoxMHB4O2JvcmRlci10b3A6MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiPicrY3R4Kyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VmVsb2NpdHk8L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiIHN0eWxlPSJmb250LXNpemU6MThweDtjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIM60PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJysoZC5kZWx0YT49MD8ndXAnOidkbicpKyciPicrKGQuZGVsdGE+PTA/JysnOicnKSsoZC5kZWx0YXx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKGQuYXR0ZW50aW9ufHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgIChyaXNpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BY2NlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIHJpc2luZ05hcnMubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6N3B4IDEwcHg7bWFyZ2luLWJvdHRvbTo0cHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiNlMDVhMjgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKSsKICAgICAgKGZhbGxpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5EZWNlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIGZhbGxpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMDUpO2JvcmRlcjoxcHggc29saWQgcmdiYSg1OSwxODQsMjE2LDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiMzYmI4ZDgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKTsKICB9CgogIHBhbmVsLmlubmVySFRNTD1oZWFkZXIrYm9keTsKfQoKCmZ1bmN0aW9uIHRvZ2dsZUZhdihubSl7CiAgaWYoRkFWUy5oYXMobm0pKSBGQVZTLmRlbGV0ZShubSk7ZWxzZSBGQVZTLmFkZChubSk7CiAgcmVuZGVyUGFuZWwoU0VMKTtyZW5kZXJGYXZzKCk7Cn0KZnVuY3Rpb24gcmVuZGVyRmF2cygpewogIHZhciByb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Zhdi1yb3cnKTsKICBpZighRkFWUy5zaXplKXtyb3cuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4nO3JldHVybjt9CiAgcm93LmlubmVySFRNTD1BcnJheS5mcm9tKEZBVlMpLm1hcChmdW5jdGlvbihubSl7CiAgICB2YXIgZD1nKG5tKSxkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPycjZTA1YTI4JzonIzNiYjhkOCc7CiAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJmYXYtY2FyZCIgb25jbGljaz0ic2VsZWN0XyhcJycrbm0rJ1wnKSI+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLWhlYWQiPjxzcGFuIGNsYXNzPSJmYy1uYW1lIj4nK25tKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImZjLXNjIj4nK2QuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+TmFycmF0aXZlPC9zcGFuPjxzcGFuIGNsYXNzPSJ2Ij4nK3RvcCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPjI0aDwvc3Bhbj48c3BhbiBjbGFzcz0idiIgc3R5bGU9ImNvbG9yOicrZEMrJyI+JytkUytkLmRlbHRhKyc8L3NwYW4+PC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9Cgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oYyl7CiAgYy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbih4KXt4LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgYy5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtsYXllcj1jLmRhdGFzZXQubGF5ZXI7YXBwbHlMYXllcigpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHVwZGF0ZUNsb2NrKCl7CiAgdmFyIG5vdz1uZXcgRGF0ZSgpLGlzdD1uZXcgRGF0ZShub3cuZ2V0VGltZSgpK25vdy5nZXRUaW1lem9uZU9mZnNldCgpKjYwMDAwKzE5ODAwMDAwKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2xvY2snKS50ZXh0Q29udGVudD1TdHJpbmcoaXN0LmdldEhvdXJzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRNaW51dGVzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRTZWNvbmRzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnIElTVCc7Cn0Kc2V0SW50ZXJ2YWwodXBkYXRlQ2xvY2ssMTAwMCk7dXBkYXRlQ2xvY2soKTsKCmZ1bmN0aW9uIGJ1aWxkV0lSU2lnbmFscygpewogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogIHZhciBzcmM9T2JqZWN0LmtleXMoTElWRSkubGVuZ3RoP0xJVkU6U0Q7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoc3JjKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5hdHRlbnRpb258fDApPjM7fSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CiAgZW50cmllcy5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pOwoKICB2YXIgdXNlZE5hcnJhdGl2ZXM9W10sdXNlZFN0YXRlcz1bXTsKICB2YXIgc2lnbmFscz1bXTsKICBmdW5jdGlvbiB1c2VkKG5hcixzdGF0ZSl7cmV0dXJuIHVzZWROYXJyYXRpdmVzLmluZGV4T2YobmFyKT49MHx8dXNlZFN0YXRlcy5pbmRleE9mKHN0YXRlKT49MDt9CiAgZnVuY3Rpb24gdXNlKG5hcixzdGF0ZSl7aWYobmFyKXVzZWROYXJyYXRpdmVzLnB1c2gobmFyKTtpZihzdGF0ZSl1c2VkU3RhdGVzLnB1c2goc3RhdGUpO30KCiAgLy8gMS4gRG9taW5hbnQgc2lnbmFsIOKAlCBkaXJlY3QsIGdyb3VuZGVkCiAgdmFyIHRvcD1lbnRyaWVzWzBdOwogIGlmKHRvcCl7CiAgICB2YXIgbmFyPXRvcFsxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdwb2xpdGljYWwgYWN0aXZpdHknOwogICAgdmFyIGVtbz10b3BbMV0uZG9taW5hbnRfZW1vdGlvbjsKICAgIHZhciBjb2w9ZW1vP3BhbFtlbW9dOid2YXIoLS1hY2NlbnQpJzsKICAgIHZhciB2ZWw9dG9wWzFdLnZlbG9jaXR5fHwwOwogICAgdmFyIHRhaWw9dmVsPjAuMDg/JywgYW5kIHRoZSBzaWduYWwgaXMgc3RpbGwgYnVpbGRpbmcnOnZlbDwtMC4wND8nLCB0aG91Z2ggbW9tZW50dW0gaXMgYmVnaW5uaW5nIHRvIGVhc2UnOicnOwogICAgdmFyIGVtb0N0eD17YW5nZXI6JyDigJQgd2l0aCBmcnVzdHJhdGlvbiBhcyB0aGUgcHJldmFpbGluZyB0b25lJyxhbnhpZXR5Oicg4oCUIHVuZGVyY3VycmVudCBvZiBhbnhpZXR5IHJ1bm5pbmcgdGhyb3VnaCBzaWduYWxzJyxmZWFyOicg4oCUIHNpZ25hbHMgY2FycnlpbmcgYW4gZWRnZSBvZiBhcHByZWhlbnNpb24nLGhvcGU6JyDigJQgYSByZWxhdGl2ZWx5IG9wdGltaXN0aWMgcmVnaXN0ZXInLHByaWRlOicnfTsKICAgIHNpZ25hbHMucHVzaCh7Y29sOmNvbCx0YWc6J2hpZ2hlc3Qgc2lnbmFsJyxsb2M6dG9wWzBdLAogICAgICB0ZXh0Oic8c3Ryb25nPicrdG9wWzBdKyc8L3N0cm9uZz4gaXMgZ2VuZXJhdGluZyB0aGUgbW9zdCBhdHRlbnRpb24gbmF0aW9uYWxseSBhcm91bmQgPGVtPicrbmFyKyc8L2VtPicrdGFpbCsoZW1vP2Vtb0N0eFtlbW9dfHwnJzonJyksZGVsYXk6MH0pOwogICAgdXNlKG5hcix0b3BbMF0pOwogIH0KCiAgLy8gMi4gRWFybHkgbW92ZXIg4oCUIHNvbWV0aGluZyBidWlsZGluZyBiZWZvcmUgaXQgZ29lcyBuYXRpb25hbAogIHZhciBlYXJseT1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7CiAgICByZXR1cm4oa3ZbMV0udmVsb2NpdHl8fDApPjAuMDUmJihrdlsxXS5hdHRlbnRpb258fDApPDM1JiYhdXNlZChrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUsa3ZbMF0pOwogIH0pLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4oYlsxXS52ZWxvY2l0eXx8MCktKGFbMV0udmVsb2NpdHl8fDApO30pWzBdOwogIGlmKGVhcmx5KXsKICAgIHZhciBlTmFyPWVhcmx5WzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J2xvY2FsIGRldmVsb3BtZW50cyc7CiAgICB2YXIgZUVtbz1lYXJseVsxXS5kb21pbmFudF9lbW90aW9uOwogICAgc2lnbmFscy5wdXNoKHtjb2w6ZUVtbz9wYWxbZUVtb106JyNlMDc4MjAnLHRhZzonYnVpbGRpbmcgc2lnbmFsJyxsb2M6ZWFybHlbMF0sCiAgICAgIHRleHQ6JzxlbT4nK2VOYXIuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZU5hci5zbGljZSgxKSsnPC9lbT4gc2lnbmFscyBhcmUgZ2FpbmluZyB0cmFjdGlvbiBpbiA8c3Ryb25nPicrZWFybHlbMF0rJzwvc3Ryb25nPiDigJQgZWFybGllciB0aGFuIG1vc3QgY3ljbGVzIGF0IHRoaXMgc3RhZ2UnLGRlbGF5OjE2MH0pOwogICAgdXNlKGVOYXIsZWFybHlbMF0pOwogIH0KCiAgLy8gMy4gRW1vdGlvbmFsIGNvbmNlbnRyYXRpb24g4oCUIHRvbmUgcmVhZCwgbm90IGEgaGVhZGxpbmUKICB2YXIgZW1vRm9jdXM9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3YpewogICAgcmV0dXJuIGt2WzFdLmRvbWluYW50X2Vtb3Rpb24mJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSkmJihrdlsxXS5hdHRlbnRpb258fDApPjQ7CiAgfSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZihlbW9Gb2N1cyl7CiAgICB2YXIgZWZOYXI9ZW1vRm9jdXNbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnZGV2ZWxvcG1lbnRzJzsKICAgIHZhciBlZkVtbz1lbW9Gb2N1c1sxXS5kb21pbmFudF9lbW90aW9uOwogICAgdmFyIGVmQ29sPXBhbFtlZkVtb118fCcjNTU2Njc3JzsKICAgIHZhciBlZlJlYWQ9ewogICAgICBhbmdlcjonU2lnbmFscyBmcm9tIDxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4gY2FycnkgYSBub3RpY2VhYmx5IGZydXN0cmF0ZWQgdG9uZSDigJQgd29ydGggd2F0Y2hpbmcnLAogICAgICBhbnhpZXR5OidUaGVyZSBpcyBhIHF1aWV0IHVuZWFzZSBpbiA8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IOKAlCBzaWduYWxzIHN1Z2dlc3QgdGhpcyBoYXMgbm90IHBlYWtlZCB5ZXQnLAogICAgICBmZWFyOidTaWduYWxzIGluIDxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4gY2FycnkgYW4gZWRnZSDigJQgdGhlIGVtb3Rpb25hbCByZWdpc3RlciBpcyBhcHByZWhlbnNpdmUnLAogICAgICBob3BlOidTb21ld2hhdCB1bnVzdWFsbHksIDxzdHJvbmc+JytlbW9Gb2N1c1swXSsnPC9zdHJvbmc+IGlzIHNob3dpbmcgYW4gb3B0aW1pc3RpYyBzaWduYWwgcmVnaXN0ZXIgYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPicsCiAgICAgIHByaWRlOic8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBzaWduYWxzIGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4gaGF2ZSBhIHN0cm9uZyBpZGVudGl0eSB0b25lIOKAlCBsb2NhbGx5IGNvbmNlbnRyYXRlZCcKICAgIH07CiAgICBzaWduYWxzLnB1c2goe2NvbDplZkNvbCx0YWc6J2Vtb3Rpb25hbCB0b25lJyxsb2M6ZW1vRm9jdXNbMF0sCiAgICAgIHRleHQ6ZWZSZWFkW2VmRW1vXXx8J1NpZ25hbHMgZnJvbSA8c3Ryb25nPicrZW1vRm9jdXNbMF0rJzwvc3Ryb25nPiBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IGFyZSB3b3J0aCB3YXRjaGluZycsZGVsYXk6MzIwfSk7CiAgICB1c2UoZWZOYXIsZW1vRm9jdXNbMF0pOwogIH0KCiAgLy8gNC4gQ29vbGluZyDigJQgY3ljbGUgY29tcGxldGluZwogIHZhciBjb29saW5nPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybihrdlsxXS52ZWxvY2l0eXx8MCk8LTAuMDQmJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSkmJihrdlsxXS5hdHRlbnRpb258fDApPjU7CiAgfSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihhWzFdLnZlbG9jaXR5fHwwKS0oYlsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYoY29vbGluZyl7CiAgICB2YXIgY05hcj1jb29saW5nWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J3JlY2VudCBmb2N1cyc7CiAgICBzaWduYWxzLnB1c2goe2NvbDonIzNiYjhkOCcsdGFnOidzaWduYWwgcmV0cmVhdGluZycsbG9jOmNvb2xpbmdbMF0sCiAgICAgIHRleHQ6JzxlbT4nK2NOYXIuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrY05hci5zbGljZSgxKSsnPC9lbT4gaW4gPHN0cm9uZz4nK2Nvb2xpbmdbMF0rJzwvc3Ryb25nPiBhcHBlYXJzIHRvIGJlIGxvc2luZyBzaWduYWwgc3RyZW5ndGgg4oCUIHRoZSBjeWNsZSBtYXkgYmUgcnVubmluZyBpdHMgY291cnNlJyxkZWxheTo0NjB9KTsKICAgIHVzZShjTmFyLGNvb2xpbmdbMF0pOwogIH0KCiAgLy8gNS4gTm9ydGhlYXN0IOKAlCBzaW1wbHkgb2JzZXJ2YXRpb25hbCwgbm8gZHJhbWF0aXNhdGlvbgogIHZhciBuZVN0YXRlcz1bJ01hbmlwdXInLCdBc3NhbScsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ0FydW5hY2hhbCBQcmFkZXNoJywnVHJpcHVyYSddOwogIHZhciBuZUFjdGl2ZT1uZVN0YXRlcy5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHNyY1tzXSYmKHNyY1tzXS5hdHRlbnRpb258fDApPjImJnVzZWRTdGF0ZXMuaW5kZXhPZihzKTwwO30pOwogIGlmKG5lQWN0aXZlLmxlbmd0aD49Mil7CiAgICB2YXIgbmVOYXI9KHNyY1tuZUFjdGl2ZVswXV0mJnNyY1tuZUFjdGl2ZVswXV0uZG9taW5hbnRfbmFycmF0aXZlKXx8J3JlZ2lvbmFsIGRldmVsb3BtZW50cyc7CiAgICBzaWduYWxzLnB1c2goe2NvbDoncmdiYSgxNjAsMTkwLDIzMCwwLjQ1KScsdGFnOidyZWdpb25hbCBzaWduYWwnLGxvYzonTm9ydGhlYXN0JywKICAgICAgdGV4dDpuZUFjdGl2ZS5sZW5ndGgrJyBub3J0aGVhc3Rlcm4gc3RhdGVzIGFyZSBzaG93aW5nIGNvbmNlbnRyYXRlZCBzaWduYWxzIGFyb3VuZCA8ZW0+JytuZU5hcisnPC9lbT4g4oCUIGEgcGF0dGVybiB0aGF0IHRlbmRzIHRvIHByZWNlZGUgd2lkZXIgbmF0aW9uYWwgYXR0ZW50aW9uJyxkZWxheTo1ODB9KTsKICB9CgogIGlmKCFzaWduYWxzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnd2lyLXNpZ25hbHMnKTsKICBpZighZWwpIHJldHVybjsKICBlbC5pbm5lckhUTUw9c2lnbmFscy5tYXAoZnVuY3Rpb24ocyl7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9Indpci1zaWduYWwiIHN0eWxlPSJhbmltYXRpb24tZGVsYXk6JytzLmRlbGF5KydtcyI+JysKICAgICAgJzxkaXYgY2xhc3M9Indpci1zaWduYWwtYmFyIiBzdHlsZT0iYmFja2dyb3VuZDonK3MuY29sKyciPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLWNvbnRlbnQiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9Indpci1zaWduYWwtdGV4dCI+JytzLnRleHQrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9Indpci1zaWduYWwtbWV0YSI+JysKICAgICAgICAgICc8c3BhbiBjbGFzcz0id2lyLXNpZ25hbC10YWciIHN0eWxlPSJjb2xvcjonK3MuY29sKyciPicrcy50YWcrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIGNsYXNzPSJ3aXItc2lnbmFsLWxvYyI+JytzLmxvYysnPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KCgovLyBJTklUIOKAlCB3YWl0IGZvciBET00KLy8gaSBidXR0b24gdG9vbHRpcCDigJQgdXNlcyBmaXhlZCBwb3NpdGlvbmluZyBzbyBpdCdzIG5ldmVyIGNsaXBwZWQKKGZ1bmN0aW9uKCl7CiAgdmFyIHRpcD1udWxsOwogIGZ1bmN0aW9uIHNob3dUaXAoZSl7CiAgICBpZighdGlwKXt0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2x0YWItdG9vbHRpcCcpO30KICAgIHZhciB0eHQ9dGhpcy5nZXRBdHRyaWJ1dGUoJ2RhdGEtdGlwJyk7CiAgICBpZighdHh0fHwhdGlwKSByZXR1cm47CiAgICB0aXAudGV4dENvbnRlbnQ9dHh0OwogICAgdGlwLmNsYXNzTGlzdC5hZGQoJ3Zpc2libGUnKTsKICAgIHZhciByZWN0PXRoaXMuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICB2YXIgdHc9MjQwOwogICAgdmFyIGxlZnQ9TWF0aC5taW4ocmVjdC5sZWZ0LHdpbmRvdy5pbm5lcldpZHRoLXR3LTEwKTsKICAgIHRpcC5zdHlsZS5sZWZ0PWxlZnQrJ3B4JzsKICAgIHRpcC5zdHlsZS50b3A9KHJlY3QudG9wLTEwLXRpcC5vZmZzZXRIZWlnaHR8fHJlY3QudG9wLTgwKSsncHgnOwogICAgLy8gUmVwb3NpdGlvbiBhZnRlciByZW5kZXIKICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmdW5jdGlvbigpewogICAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC10aXAub2Zmc2V0SGVpZ2h0LTgpKydweCc7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gaGlkZVRpcCgpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICBpZih0aXApIHRpcC5jbGFzc0xpc3QucmVtb3ZlKCd2aXNpYmxlJyk7CiAgfQogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3ZlcicsZnVuY3Rpb24oZSl7CiAgICBpZihlLnRhcmdldC5jbGFzc0xpc3QuY29udGFpbnMoJ2x0YWItaW5mbycpKSBzaG93VGlwLmNhbGwoZS50YXJnZXQsZSk7CiAgfSk7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VvdXQnLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgaGlkZVRpcCgpOwogIH0pOwp9KSgpOwoKZnVuY3Rpb24gZGlzbWlzc0xvYWRlcigpewogIHZhciBsZHI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2FwcC1sb2FkZXInKTsKICBpZighbGRyKSByZXR1cm47CiAgbGRyLnN0eWxlLm9wYWNpdHk9JzAnOwogIGxkci5zdHlsZS52aXNpYmlsaXR5PSdoaWRkZW4nOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZihsZHIpbGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnO30sOTAwKTsKfQoKCmZ1bmN0aW9uIGRpc21pc3NMb2FkZXIoKXsKICB2YXIgbGRyPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhcHAtbG9hZGVyJyk7CiAgaWYoIWxkcikgcmV0dXJuOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYobGRyKSBsZHIuc3R5bGUuZGlzcGxheT0nbm9uZSc7fSw5MDApOwp9CmZ1bmN0aW9uIGRpc21pc3NMb2FkZXIoKXsKICB2YXIgbGRyPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhcHAtbG9hZGVyJyk7CiAgaWYoIWxkcikgcmV0dXJuOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYobGRyKSBsZHIuc3R5bGUuZGlzcGxheT0nbm9uZSc7fSw5MDApOwp9CgoKZnVuY3Rpb24gaW5pdCgpewogIHJlbmRlclN0cmlwKCczbScpOwoKICAvLyBMb2FkIG1hcCB3aXRoIHJldHJ5CiAgdmFyIG1hcEF0dGVtcHRzPTA7CiAgZnVuY3Rpb24gdHJ5TG9hZE1hcCgpewogICAgaWYodHlwZW9mIHRvcG9qc29uPT09J3VuZGVmaW5lZCcpewogICAgICBpZihtYXBBdHRlbXB0cysrPDEwKXtzZXRUaW1lb3V0KHRyeUxvYWRNYXAsMzAwKTt9CiAgICAgIHJldHVybjsKICAgIH0KICAgIGxvYWRNYXAoKTsKICB9CiAgdHJ5TG9hZE1hcCgpOwoKICAvLyBMb2FkIGZ1bGwgY2FjaGVkIHNuYXBzaG90IGltbWVkaWF0ZWx5IGZvciBpbnN0YW50IGRhdGEKICBmZXRjaEZ1bGxTbmFwc2hvdCgpLnRoZW4oZnVuY3Rpb24ob2spewogICAgaWYob2spewogICAgICByZW5kZXJNb21lbnR1bSgpOwogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7c3RhcnRQb2xsaW5nKCk7fSwxMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHN0YXJ0UG9sbGluZygpOwogICAgfQogICAgZGlzbWlzc0xvYWRlcigpOwogIH0pOwoKICAvLyBEaXNtaXNzIGxvYWRlciBhZnRlciBtYXggNHMgcmVnYXJkbGVzcwogIHNldFRpbWVvdXQoZGlzbWlzc0xvYWRlciwgNDAwMCk7CgogIC8vIFJldHJ5IG1hcCBpZiBzdGlsbCBlbXB0eQogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDMwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDYwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDUwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKS5jYXRjaChmdW5jdGlvbigpe30pO30sODAwMCk7Cn0KaWYoZG9jdW1lbnQucmVhZHlTdGF0ZT09PSdsb2FkaW5nJyl7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsIGluaXQpOwp9IGVsc2UgewogIC8vIEFscmVhZHkgbG9hZGVkIOKAlCBidXQgd2FpdCBvbmUgdGljayB0byBlbnN1cmUgYWxsIHNjcmlwdHMgcGFyc2VkCiAgc2V0VGltZW91dChpbml0LCAwKTsKfQoKCnNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAvLyBBdXRvLXNlbGVjdCBob3R0ZXN0IHN0YXRlIGZyb20gTElWRSBkYXRhCiAgdmFyIHNyYz1PYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg/TElWRTpTRDsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKHNyYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKXsKICAgIHZhciBlbD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK3RvcFswXSsnIl0nKTsKICAgIGlmKGVsKSBzZWxlY3RfKHRvcFswXSk7CiAgfQp9LDMwMDApOwpzZXRUaW1lb3V0KHJlbmRlckZhdnMsMjQwMCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"

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
