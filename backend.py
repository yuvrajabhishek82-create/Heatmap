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
    # Independent / alternative journalism — HIGH WEIGHT
    {"id": "UCJqobJNbMB0DcMSvdQkfMzg", "name": "Newslaundry",       "type": "independent"},
    {"id": "UC3M7l8ved_rYQ45AVzS0RGA", "name": "Dhruv Rathee",       "type": "independent"},
    {"id": "UCGkJFMGBqUbNmXwqh0iAeow", "name": "Ravish Kumar",       "type": "independent"},  # Ravish Kumar Official
    {"id": "UCfIJut6tiwYV0sbHN3mFiIg", "name": "The Red Mic",        "type": "independent"},
    {"id": "UCHqGVpJB_NKkPdJZlSBHBXQ", "name": "News Pinch",         "type": "independent"},
    {"id": "UC3yY9qBsUTUiP-gBLhDVUoA", "name": "The Wire",           "type": "national_alt"},
    {"id": "UCBcRF18a7Qf58cCRy5xuWwQ", "name": "Newslaundry Hindi",  "type": "independent"},
    {"id": "UCQpxtMxixOmBXCuOEPQwgYA", "name": "Drishti IAS",        "type": "national_alt"},
    {"id": "UCvEA9LoNlMnQEd2HMLl1U8Q", "name": "NDTV",               "type": "national"},
    {"id": "UCt4t-jeY85JegMlZ-E5UWuQ", "name": "Aaj Tak",            "type": "national"},
    {"id": "UCoHRYZ5cOvWTwRnPpWYB4Ng", "name": "India Today",        "type": "national"},
    {"id": "UCYPvAwZP8pZhSMW8qs7cVCw", "name": "ABP News",           "type": "national"},
    {"id": "UCghr3UCP3mRhTIKDnO9w7dA", "name": "Zee News",           "type": "national"},
    {"id": "UCU_1TJIW__YIUQKSDlLBxeA", "name": "Republic TV",        "type": "national"},
    # Alt commentary
    {"id": "UCXGfUHVBzRuoQgmPMqUNZLQ", "name": "Desh Bhakt",         "type": "entertainment"},
    {"id": "UCJpGpSNRDsKI_p7h2yFPpgg", "name": "Gist",               "type": "independent"},
]

# ── STATE-SPECIFIC CHANNELS ────────────────────────────────────
STATE_YT_CHANNELS: dict[str, list[dict]] = {
    "Tamil Nadu": [
        {"id": "UCqMFRFBaOPzBEjzJPBi8IQQ", "name": "Thanthi TV",        "type": "regional"},
        {"id": "UCdDqEGvwdRDVpQJM1S6rQYg", "name": "Polimer News",       "type": "regional"},
        {"id": "UCg0NaSNfAkWLWzMeNWHqFHA", "name": "ABP Nadu",           "type": "regional"},
        {"id": "UCkPdAr7bDaVLKIVUw-9YQXA", "name": "Sun News",           "type": "regional"},
        {"id": "UCXGbQV2LMdBqLJRnFMfkWUg", "name": "Puthiya Thalaimurai","type": "regional"},
        {"id": "UCsHoPPt2iqjMd-hWjGJwY1Q", "name": "Sathiyam TV",        "type": "regional"},
    ],
    "Kerala": [
        {"id": "UCBQMoIGFNkh8GxiXIUCZoew", "name": "Manorama News",     "type": "regional"},
        {"id": "UCmgBdxkFVwEP2PxEGqZwJ-A", "name": "Asianet News",       "type": "regional"},
        {"id": "UCXGb5U1a8IHNX9l1VvfCGsQ", "name": "MediaOne",           "type": "regional"},
        {"id": "UCBMpOyN-XgKlHJxMfSZBMGg", "name": "Mathrubhumi News",   "type": "regional"},
        {"id": "UCkfOCCDpPiuBBWkV2H5mgxw", "name": "Reporterlive",       "type": "grassroots"},
        {"id": "UCnWbkjfKDiJa11b5TsEJMcA", "name": "Kerala Kaumudi",     "type": "regional"},
    ],
    "Karnataka": [
        {"id": "UCsMZFMTVz0rcXVJVMBgMdvg", "name": "TV9 Kannada",        "type": "regional"},
        {"id": "UC1PZFe3MEr9_7OsalJYGxhg", "name": "Suvarna News",       "type": "regional"},
        {"id": "UCBqiRL_2GqpqfAmO-8YQBQA", "name": "Kasturi News",       "type": "regional"},
        {"id": "UC7xRoNbFi9KYx2bTm4RJLhQ", "name": "Samaya News",        "type": "regional"},
        {"id": "UCkQPzBLMlHHSaqPfTMiUYXg", "name": "Janashri News",      "type": "regional"},
    ],
    "Andhra Pradesh": [
        {"id": "UCbGa9MVQVPy7Y55Tr8hIMzg", "name": "TV9 Telugu",         "type": "regional"},
        {"id": "UC4M7Q_EWQ4XJhSb2wU5BwOA", "name": "ABN Andhra Jyothy",  "type": "regional"},
        {"id": "UCCdLJhUDmiMHwbAjEjnFp7Q", "name": "Sakshi TV",          "type": "regional"},
        {"id": "UCBsYDCl5w_-wgST4nIZ-tXg", "name": "NTV Telugu",         "type": "regional"},
        {"id": "UCwSDrR5bJ-HGE8FIFsGdAnQ", "name": "10TV News",          "type": "regional"},
    ],
    "Telangana": [
        {"id": "UCbGa9MVQVPy7Y55Tr8hIMzg", "name": "TV9 Telugu",         "type": "regional"},
        {"id": "UCe11HL_VBXZF0qQmK3Grfhg", "name": "V6 News",            "type": "regional"},
        {"id": "UCvQGKqd1K0kRwUKSFaegSQg", "name": "T News",             "type": "regional"},
        {"id": "UC4M7Q_EWQ4XJhSb2wU5BwOA", "name": "ABN Andhra Jyothy",  "type": "regional"},
    ],
    "Maharashtra": [
        {"id": "UCsBjURrPoezykLs9EqgamOA", "name": "ABP Majha",          "type": "regional"},
        {"id": "UCLfGnuTVqkGf_aLdBz7Qd7w", "name": "TV9 Marathi",        "type": "regional"},
        {"id": "UC7RXQFWU5R90fLvIpWCpg_A", "name": "Zee 24 Taas",        "type": "regional"},
        {"id": "UCItpmPXfElCFREvqBjaTNFg", "name": "News18 Lokmat",      "type": "regional"},
        {"id": "UCMixZ8rHDqmhIVCVMzY-XtA", "name": "Loksatta Live",      "type": "regional"},
    ],
    "Gujarat": [
        {"id": "UCMQu1CKtsbhQFnCpYKnr7DA", "name": "Sandesh News",       "type": "regional"},
        {"id": "UC6mcOKNQzB2HrMDSIFVnBrg", "name": "VTV Gujarati",       "type": "regional"},
        {"id": "UCcjQkrJCXzFHlv5O_2RI4Sg", "name": "News18 Gujarat",     "type": "regional"},
        {"id": "UCJ1mRNH9HjpzNPHMkMVa_9w", "name": "ABP Asmita",         "type": "regional"},
    ],
    "Rajasthan": [
        {"id": "UCt6E2ooCm-iCERQUX3oKVEg", "name": "First India News",   "type": "regional"},
        {"id": "UCUMEQq7HwmkbNijZBJC5DqQ", "name": "ETV Rajasthan",      "type": "regional"},
        {"id": "UCFnbsUCJNAe5LT6jv-kijTQ", "name": "News18 Rajasthan",   "type": "regional"},
    ],
    "Uttar Pradesh": [
        {"id": "UC9rFWQDE_pEFk_xh-n7YCBQ", "name": "ABP Ganga",          "type": "regional"},
        {"id": "UC6T5bxriFsqHEjBFhypWMfQ", "name": "News18 UP Uttarakhand","type": "regional"},
        {"id": "UCuLYM2K70M0gVBxIHyPFiXQ", "name": "Aaj Tak UP",         "type": "regional"},
        {"id": "UCULKIoqpWHcDFnC0G8xkXFg", "name": "Samachar Plus",      "type": "grassroots"},
    ],
    "Bihar": [
        {"id": "UCBwECRXNhRjt2V7WxWXYeYg", "name": "ETV Bihar Jharkhand","type": "regional"},
        {"id": "UCTRdAzFCnTVE5RtKiE9YZRA", "name": "News18 Bihar",       "type": "regional"},
        {"id": "UC9RkgwWFPOXZb7V0W7BVPwA", "name": "Mahua News",         "type": "regional"},
    ],
    "West Bengal": [
        {"id": "UC_1v_3fqxGNKW72vHdQRlJg", "name": "ABP Ananda",         "type": "regional"},
        {"id": "UCsdB3XM7S9YIJG9Z0bHUzfg", "name": "Zee 24 Ghanta",      "type": "regional"},
        {"id": "UCjfOPnFGhHM28aQZ41Kv3Mg", "name": "News18 Bangla",      "type": "regional"},
        {"id": "UCtbxlj9SJSQjJpjzPHg_PFQ", "name": "TV9 Bangla",         "type": "regional"},
    ],
    "Punjab": [
        {"id": "UCnHoMNrPKMI4hMxUbLfHGEg", "name": "PTC Punjab",         "type": "regional"},
        {"id": "UCyV_qc6s0OoMr3ZQdPwW_UA", "name": "News18 Punjab",      "type": "regional"},
        {"id": "UCz7Dqm4zUB0Bkl-Xdk3TfFg", "name": "ABP Sanjha",         "type": "regional"},
    ],
    "Haryana": [
        {"id": "UC0Gz_Vl5BFqBJFaqRGNGwOw", "name": "Haryana TV",         "type": "regional"},
        {"id": "UCFsHv_Kz7mHDhY-D6YENQYQ", "name": "News18 Haryana",     "type": "regional"},
    ],
    "Madhya Pradesh": [
        {"id": "UCHpEqHxelHiCxoMVRVtQkFg", "name": "Bansal News",        "type": "regional"},
        {"id": "UC6QVlUb0ZQMBE8mujqT6x9g", "name": "News18 MP Chhattisgarh","type": "regional"},
        {"id": "UCuNSPkm2KqkJuH7Iy6FRRgw", "name": "IBC24",              "type": "regional"},
    ],
    "Odisha": [
        {"id": "UCeUdWRPzBDq1DPu6ViIRNKg", "name": "OTV",                "type": "regional"},
        {"id": "UC6tQvUiTK8zSb9UPWIh7bVA", "name": "Kanak News",         "type": "regional"},
        {"id": "UC8gC_7yMDiVMDLPFhkzG5Xg", "name": "Sambad",             "type": "regional"},
    ],
    "Assam": [
        {"id": "UC3kxaVJZOYdQX-xU9a2bIhg", "name": "News18 Assam NE",    "type": "regional"},
        {"id": "UCNAzh1xJ0R3_NVH5BpOkNNg", "name": "DY365",              "type": "regional"},
        {"id": "UCh7mF2ql3hWJEPgB8c7YA-g", "name": "Pratidin Time",      "type": "regional"},
    ],
    "Jammu and Kashmir": [
        {"id": "UCxHFGBSjBvHiGRyL5p82kLA", "name": "DD Kashir",          "type": "regional"},
        {"id": "UCvPFO8XWdEf4g8hPSXsEmGQ", "name": "Kashmir Crown",      "type": "grassroots"},
        {"id": "UCNn-5E7cDnHbvPkBcDOLFzg", "name": "Greater Kashmir",    "type": "regional"},
    ],
    "Manipur": [
        {"id": "UCpuUMdVfL3NZRKGdoIhg9Aw", "name": "IFP Manipur",        "type": "grassroots"},
        {"id": "UCgTdgOJCjpSXL0c2HxFzMBw", "name": "E-Pao Manipur",      "type": "grassroots"},
    ],
    "Delhi": [
        {"id": "UCt4t-jeY85JegMlZ-E5UWuQ", "name": "Aaj Tak",            "type": "national"},
        {"id": "UCJqobJNbMB0DcMSvdQkfMzg", "name": "Newslaundry",        "type": "independent"},
        {"id": "UC3M7l8ved_rYQ45AVzS0RGA", "name": "Dhruv Rathee",        "type": "independent"},
        {"id": "UCfIJut6tiwYV0sbHN3mFiIg", "name": "The Red Mic",         "type": "independent"},
    ],
    "Himachal Pradesh": [
        {"id": "UCHFfmLPvNe5LjJhV-h8j2Nw", "name": "Himachal Abhi Abhi", "type": "regional"},
    ],
    "Uttarakhand": [
        {"id": "UCGylVDFBkCIBPHhMBe4HrIQ", "name": "Uttarakhand Tak",    "type": "regional"},
    ],
    "Goa": [
        {"id": "UClmBEAp7sPnlCWvFCG0hFEA", "name": "Goa365",             "type": "regional"},
    ],
    "Nagaland": [
        {"id": "UCrVX3BmL9rhtSQidQkuEEyg", "name": "Morung Express",     "type": "grassroots"},
    ],
    "Meghalaya": [
        {"id": "UCkG2R6WVq-XAtohvtq4hC7A", "name": "Shillong Times",     "type": "grassroots"},
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

async def fetch_youtube_channel_videos(channel_id: str, client: httpx.AsyncClient, max_results: int = 8) -> list[dict]:
    """Fetch recent videos from a YouTube channel."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        url = (
            f"https://www.googleapis.com/youtube/v3/search"
            f"?key={YOUTUBE_API_KEY}&channelId={channel_id}"
            f"&part=snippet&order=date&type=video&maxResults={max_results}"
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
            # Only last 7 days
            if (datetime.now(timezone.utc) - pub_dt).days > 7:
                continue
            videos.append({
                "video_id":    video_id,
                "title":       snippet.get("title", ""),
                "description": snippet.get("description", "")[:200],
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
    """Ingest YouTube signals for a state from state-specific channels."""
    if not YOUTUBE_API_KEY:
        return 0
    channels = STATE_YT_CHANNELS.get(state, [])
    if not channels:
        return 0

    added = 0
    all_videos = []
    video_to_channel: dict[str, dict] = {}

    for ch in channels[:6]:  # max 6 channels per state per cycle
        videos = await fetch_youtube_channel_videos(ch["id"], client, max_results=5)
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
    """Ingest national YouTube channels and geo-tag to relevant states."""
    if not YOUTUBE_API_KEY:
        return 0
    added = 0
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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjIwcHggMjJweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNykgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAyKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIHBvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjI4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWFjY2VudCk7bWFyZ2luLWJvdHRvbToxMnB4O29wYWNpdHk6MC44Owp9CiNzaWctaW5zaWdodHttYXJnaW4tYm90dG9tOjE0cHh9CiNzaWctdGFnc3tkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjZweDttaW4taGVpZ2h0OjB9CiNzaWctdGFnczplbXB0eXtkaXNwbGF5Om5vbmV9CiNzaWctbWV0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnJnYmEoNjIsNzcsOTYsMC40NSk7bWFyZ2luLXRvcDoxMHB4O2xldHRlci1zcGFjaW5nOjAuMDVlbX0KI3NpZy1tZXRhOmVtcHR5e2Rpc3BsYXk6bm9uZX0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNnB4LDEuNnZ3LDIycHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQouc2ktdGV4dCBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1hY2NlbnQpO2ZvbnQtd2VpZ2h0OjQwMH0KLnNpLXRleHQgc3Ryb25ne2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjpyZ2JhKDI0MCwyMzUsMjI1LDAuOSl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKICBhbGlnbi1pdGVtczpzdGFydDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnN0aWNreTt0b3A6NjBweDsKfQoubWFwLWNhcmQ6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7CiAgYmFja2dyb3VuZDoKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDcwJSA1MCUgYXQgMzUlIDAlLCByZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSwgdHJhbnNwYXJlbnQgNjAlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDUwJSA0MCUgYXQgODAlIDEwMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAwJSwgdHJhbnNwYXJlbnQgNjAlKTsKfQoubWFwLXRvcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjEycHggMThweCAwOwp9Ci5tYXAtdGl0bGUtYmxvY2sgLm10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTdweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLm1hcC10aXRsZS1ibG9jayAubXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO21hcmdpbi10b3A6MnB4fQoubGVnZW5ke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmxlZ2VuZC1iYXJ7CiAgaGVpZ2h0OjNweDt3aWR0aDo4MHB4O2JvcmRlci1yYWRpdXM6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIHJpZ2h0LCMwZTIwMzUsIzFhNTU4MCAyNSUsIzhhNWMxOCA1NSUsI2MwMzgxYSA4MCUsI2UwMTAyMCk7Cn0KLmxheWVyLXJvd3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OwogIHBhZGRpbmc6MTBweCAyMHB4IDZweDsKfQoubGF5ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQoubHRhYnN7ZGlzcGxheTpmbGV4O2dhcDozcHh9Ci5sdGFiewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzozcHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLmx0YWIuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA4KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKX0KLmx0YWJ7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjVweDtwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzp2aXNpYmxlfQoubHRhYi1pbmZve3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMik7Zm9udC1zaXplOjhweDtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXN0eWxlOml0YWxpYztmb250LXdlaWdodDo2MDA7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjM1KTtkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2N1cnNvcjpoZWxwO2ZsZXgtc2hyaW5rOjA7dHJhbnNpdGlvbjphbGwgMC4xNXM7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxMDB9Ci5sdGFiLWluZm86aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWFjY2VudCk7Y29sb3I6dmFyKC0tYWNjZW50KX0KI2x0YWItdG9vbHRpcHtwb3NpdGlvbjpmaXhlZDtiYWNrZ3JvdW5kOnJnYmEoOCwxMiwyMCwwLjk4KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4xMik7Ym9yZGVyLXJhZGl1czo4cHg7cGFkZGluZzoxMHB4IDEzcHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42O3dpZHRoOjIzMHB4O3doaXRlLXNwYWNlOm5vcm1hbDt0ZXh0LWFsaWduOmxlZnQ7Ym94LXNoYWRvdzowIDhweCAzMnB4IHJnYmEoMCwwLDAsMC42KTtwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7ei1pbmRleDo5OTk5OTtkaXNwbGF5Om5vbmV9CiNsdGFiLXRvb2x0aXAudmlzaWJsZXtvcGFjaXR5OjE7ZGlzcGxheTpibG9ja30KLmx0YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KCi5tYXAtc3ZnLXdyYXB7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZzoxMnB4IDE2cHggMTZweDsKfQoubWFwLWlubmVye3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxLzE7d2lkdGg6MTAwJX0KI2luZGlhLW1hcHt3aWR0aDoxMDAlO2hlaWdodDoxMDAlO2Rpc3BsYXk6YmxvY2s7b3ZlcmZsb3c6dmlzaWJsZX0KCi8qIG1hcCBzdGF0ZSBzdHlsZXMgKi8KI2luZGlhLW1hcCAuc3RhdGV7CiAgY3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpmaWx0ZXIgMC4yNXMgZWFzZSwgc3Ryb2tlLXdpZHRoIDAuMnMgZWFzZSwgc3Ryb2tlIDAuMnMgZWFzZTsKfQojaW5kaWEtbWFwIC5zdGF0ZTpob3ZlcnsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjcpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjFweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMjUpIGRyb3Atc2hhZG93KDAgMCAxMHB4IHJnYmEoMjU1LDI1NSwyNTUsMC4yKSk7Cn0KI2luZGlhLW1hcCAuc3RhdGUuc2VsZWN0ZWR7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC45KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjRweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMzUpIGRyb3Atc2hhZG93KDAgMCAxNnB4IHJnYmEoMjU1LDI1NSwyNTUsMC4zKSk7Cn0KCi8qIGFuaW1hdGVkIHB1bHNlIHJpbmdzICovCi5wdWxzZS1yaW5ne2ZpbGw6bm9uZTtwb2ludGVyLWV2ZW50czpub25lfQoucHVsc2UtcmluZy5wMXthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCBpbmZpbml0ZX0KLnB1bHNlLXJpbmcucDJ7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgMC45cyBpbmZpbml0ZX0KQGtleWZyYW1lcyBwcnsKICAwJXtyOjQ7b3BhY2l0eTowLjc7c3Ryb2tlLXdpZHRoOjEuMn0KICAxMDAle3I6MjY7b3BhY2l0eTowO3N0cm9rZS13aWR0aDowLjJ9Cn0KCi8qIGF0bW9zcGhlcmljIGdsb3cgYmVoaW5kIGhvdCBzdGF0ZXMgKi8KLnN0YXRlLWdsb3d7cG9pbnRlci1ldmVudHM6bm9uZTtmaWxsOm5vbmV9CkBrZXlmcmFtZXMgZ2xvd1B1bHNlezAlLDEwMCV7b3BhY2l0eTowLjEyfTUwJXtvcGFjaXR5OjAuMjJ9fQoKLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDUsNywxMiwwLjk1KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6OXB4OwogIHBhZGRpbmc6MTJweCAxNHB4O29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xMnM7ei1pbmRleDo5OTk5O21pbi13aWR0aDoxNzBweDsKfQoudHQtbntmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO21hcmdpbi1ib3R0b206OHB4O2NvbG9yOnZhcigtLWluayl9Ci50dC1ye2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweH0KLnR0LXIgc3Ryb25ne2NvbG9yOnZhcigtLWluayl9Ci50dC1uYXJ7CiAgbWFyZ2luLXRvcDo4cHg7cGFkZGluZy10b3A6OHB4O2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7Cn0KLnR0LW5hciBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtkaXNwbGF5OmJsb2NrO21hcmdpbi1ib3R0b206MnB4fQoKLyogU1RBVEUgUEFORUwgKi8KLnN0YXRlLXBhbmVsewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBwYWRkaW5nOjIwcHg7b3ZlcmZsb3cteTphdXRvOwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwogIHBvc2l0aW9uOnN0aWNreTt0b3A6NjBweDsKICBtYXgtaGVpZ2h0OmNhbGMoMTAwdmggLSA4MHB4KTsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjNweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7CiAgaGVpZ2h0OjEwMCU7bWluLWhlaWdodDozMjBweDt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjMycHggMjBweDsKfQoucGFuZWwtZW1wdHkgc3Zne29wYWNpdHk6MC4xNTttYXJnaW4tYm90dG9tOjE4cHh9Ci5wYW5lbC1lbXB0eSAucGUtdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjhweH0KLnBhbmVsLWVtcHR5IC5wZS1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW07bGluZS1oZWlnaHQ6MS43fQoKLyogc3RhdGUgcGFuZWwgaW50ZXJuYWxzICovCi5zcC1oZWFkewogIGRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIG1hcmdpbi1ib3R0b206MTZweDtwYWRkaW5nLWJvdHRvbToxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnNwLWVre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTtjb2xvcjp2YXIoLS1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NXB4fQouc3AtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI4cHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07bGluZS1oZWlnaHQ6MTtjb2xvcjp2YXIoLS1pbmspfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Y29sb3I6dmFyKC0tZmFpbnQpOwogIHdpZHRoOjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo2cHg7Y3Vyc29yOnBvaW50ZXI7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMThzO3BhZGRpbmc6MDtmbGV4LXNocmluazowOwp9Ci5mYXYtYnRuOmhvdmVye2NvbG9yOnZhcigtLWRpbSk7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5mYXYtYnRuLm9ue2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHh9CgovKiBuYXJyYXRpdmUgdGltZWxpbmUg4oCUIHRoZSBzaWduYXR1cmUgZmVhdHVyZSAqLwoubmFyLXRpbWVsaW5lewogIG1hcmdpbi1ib3R0b206MTZweDsKfQoubnQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLm50LWZsb3d7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nLWxlZnQ6MTZweDsKfQoubnQtZmxvdzo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDo1cHg7dG9wOjZweDtib3R0b206NnB4O3dpZHRoOjFweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx2YXIoLS1ib3JkZXIpKTtvcGFjaXR5OjAuNDsKfQoubnQtc3RlcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweDsKICBwYWRkaW5nOjVweCAwO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5udC1kb3R7CiAgd2lkdGg6MTBweDtoZWlnaHQ6MTBweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4LXNocmluazowOwogIHBvc2l0aW9uOmFic29sdXRlO2xlZnQ6LTE2cHg7dG9wOjdweDsKICBib3JkZXI6MS41cHggc29saWQgY3VycmVudENvbG9yO2JhY2tncm91bmQ6dmFyKC0tYmcpOwp9Ci5udC1zdGVwLnBhc3QgLm50LWRvdHtjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCl9Ci5udC1zdGVwLmN1cnJlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCAxMHB4IHJnYmEoMjI0LDkwLDQwLDAuNSl9Ci5udC1jb250ZW50e2ZsZXg6MX0KLm50LXRvcGlje2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4zfQoubnQtc3RlcC5wYXN0IC5udC10b3BpY3tjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZGltKX0KLm50LXdoZW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9CgovKiBpbnNpZ2h0IGJsb2NrICovCi5pbnNpZ2h0ewogIG1hcmdpbi1ib3R0b206MTRweDsKICBwYWRkaW5nOjEycHggMTRweCAxMnB4IDE2cHg7CiAgYm9yZGVyLWxlZnQ6MS41cHggc29saWQgdmFyKC0tYWNjZW50KTsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDMpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxMy41cHg7Zm9udC1zdHlsZTppdGFsaWM7CiAgY29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU1O2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLyogY29tcGFjdCBzY29yZSBzdHJpcCAqLwouc2NvcmUtc3RyaXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsKICBwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6N3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgbWFyZ2luLWJvdHRvbToxNHB4Owp9Ci5zcy1pdGVte2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjJweH0KLnNzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5zcy12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2NvbG9yOnZhcigtLWluayl9Ci5zcy1kZWx0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjJweCA3cHg7Ym9yZGVyLXJhZGl1czozcHh9Ci5zcy1kZWx0YS51cHtjb2xvcjojZTA2MDMwO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKX0KLnNzLWRlbHRhLmRue2NvbG9yOiMzYmI4ZDg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKX0KLnNzLWRpdmlkZXJ7d2lkdGg6MXB4O2hlaWdodDozMnB4O2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowfQouc3MtbmFye2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDB9Cgouc3Atc2VjdGlvbnttYXJnaW4tYm90dG9tOjE0cHh9Ci5zcC1zZWMtdGl0bGV7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206OXB4Owp9CgovKiBuYXJyYXRpdmVzICovCi5uYXItbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9Ci5uYXItaXRlbTJ7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgYXV0bztnYXA6NnB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLm5pLWxhYmVse2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKX0KLm5pLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5uaS10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi10b3A6LTNweH0KLm5pLWZpbGx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHg7dHJhbnNpdGlvbjp3aWR0aCAwLjdzfQoKLyogbW92ZW1lbnQgKi8KLm12LWdyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDo3cHh9Ci5tdi1ibG9ja3tiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6N3B4O3BhZGRpbmc6OXB4fQoubXYtaHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206N3B4fQoubXYtYmxvY2sudXAgLm12LWh7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tdi1ibG9jay5kbiAubXYtaHtjb2xvcjp2YXIoLS1mYWxsKX0KLm12LWl0e2ZvbnQtc2l6ZToxMC41cHg7cGFkZGluZzo0cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2NvbG9yOnZhcigtLWZhaW50KX0KLm12LWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tdi1pdCBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTFweH0KLm12LWl0IHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweH0KCi8qIGVtb3Rpb24gKi8KLmVtLXJvd3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQouZW0tZG9udXR7d2lkdGg6NzZweDtoZWlnaHQ6NzZweDtmbGV4LXNocmluazowfQouZW0tbGVne2ZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHh9Ci5lbS1pdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVtLXN3e3dpZHRoOjZweDtoZWlnaHQ6NnB4O2JvcmRlci1yYWRpdXM6MnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1ue2ZsZXg6MTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5lbS1we2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmspfQoKLyogdGltZWxpbmUgY2hhcnQgKi8KLnRsLXdyYXB7aGVpZ2h0OjcycHh9CgovKiBhcnRpY2xlcyAqLwouYXJ0LWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4fQouYXJ0LWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzo3cHggOXB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydC1pdGVtOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5hcnQtc3Jje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO3dpZHRoOjQ0cHg7cGFkZGluZy10b3A6MXB4fQouYXJ0LXR4dHtmb250LXNpemU6MTFweDtsaW5lLWhlaWdodDoxLjQ7Y29sb3I6dmFyKC0tZGltKX0KCi8qIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XICovCi5uYXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoxOHB4Owp9Ci5uYXItY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTRweCk7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNnB4IDIwcHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowOwp9Ci5uYy1ib2R5e3BhZGRpbmc6OHB4IDIwcHggMTZweDtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO30KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAzNnB4IDYwcHg7bWF4LXdpZHRoOjU4MHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZm9vdC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luLWJvdHRvbToxNHB4fQouZm9vdC1saW5le2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS44O21hcmdpbi1ib3R0b206MTJweH0KLmZvb3Qtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpfQoKLyogYW5pbWF0aW9ucyAqLwpAa2V5ZnJhbWVzIGZhZGVVcHtmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWSg2cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci5tYXAtY2FyZCwuc3RhdGUtcGFuZWwsLm5hci1jYXJkLC5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb246ZmFkZVVwIDAuNTVzIGN1YmljLWJlemllciguMiwuOCwuMiwxKSBiYWNrd2FyZHN9Ci5uYXItY2FyZDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMDdzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDMpe2FuaW1hdGlvbi1kZWxheTowLjE0c30KLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbi1kZWxheTowLjA1c30KCkBtZWRpYShtYXgtd2lkdGg6MTEwMHB4KXsKICAubWFpbntncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQogIC5zdGF0ZS1wYW5lbHttYXgtaGVpZ2h0Om5vbmV9CiAgLm5hci1yb3d7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KfQoKCi8qIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAogICBNT0JJTEUgU1RZTEVTIOKAlCBAbWVkaWEgbWF4LXdpZHRoOjc2OHB4CiAgIFRoZXNlIHJ1bGVzIE9OTFkgYXBwbHkgb24gbW9iaWxlLiBEZXNrdG9wIGlzIGNvbXBsZXRlbHkgdW50b3VjaGVkLgogICDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAgKi8KCkBtZWRpYSAobWF4LXdpZHRoOiA3NjhweCkgewoKICAvKiDilIDilIAgVE9QQkFSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogIC50b3BiYXJ7CiAgICBwYWRkaW5nOjAgMTZweDsKICAgIGhlaWdodDo1MnB4OwogIH0KICAuYnJhbmQtdGFnbGluZXtkaXNwbGF5Om5vbmV9CiAgLmJyYW5kLW5hbWV7Zm9udC1zaXplOjEzcHh9CiAgLnRvcGJhci1yIC5saXZlLWRvdC13cmFwIHNwYW46bGFzdC1jaGlsZHtkaXNwbGF5Om5vbmV9IC8qIGhpZGUgInNpZ25hbHMiIHRleHQgKi8KICAjbGl2ZS1jb3VudHtmb250LXNpemU6MTBweH0KCiAgLyog4pSA4pSAIEhFUk8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLmhlcm97CiAgICBwYWRkaW5nOjcycHggMjBweCAyMHB4ICFpbXBvcnRhbnQ7CiAgICB0ZXh0LWFsaWduOmNlbnRlcjsKICB9CiAgLmhlcm8tZXllYnJvd3tqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2ZvbnQtc2l6ZTo4cHg7bWFyZ2luLWJvdHRvbToxNnB4fQogIC5oZXJvLWV5ZWJyb3c6OmJlZm9yZXtkaXNwbGF5Om5vbmV9CiAgLmhlcm8tYnJhbmQtYmxvY2t7anVzdGlmeS1jb250ZW50OmNlbnRlcjtnYXA6MTJweDttYXJnaW4tYm90dG9tOjEycHh9CiAgLmhlcm8tYnJhbmQtbmFtZXtmb250LXNpemU6Y2xhbXAoMzJweCw5dncsNDhweCkgIWltcG9ydGFudH0KICAuaGVyby10YWdsaW5lewogICAgZm9udC1zaXplOjE1cHggIWltcG9ydGFudDsKICAgIG1heC13aWR0aDoxMDAlOwogICAgdGV4dC1hbGlnbjpjZW50ZXI7CiAgICBtYXJnaW46MCBhdXRvIDEwcHggYXV0bzsKICB9CiAgLmhlcm8tZGVzY3sKICAgIGZvbnQtc2l6ZToxMnB4OwogICAgbWF4LXdpZHRoOjEwMCU7CiAgICB0ZXh0LWFsaWduOmNlbnRlcjsKICAgIG1hcmdpbjowIGF1dG8gNnB4IGF1dG87CiAgfQogIC5oZXJvLXN1Yi1saW5le3RleHQtYWxpZ246Y2VudGVyfQoKICAvKiBTdGF0cyBzdHJpcCDigJQgMiBjb2x1bW5zIG9uIG1vYmlsZSAqLwogIC5zdGF0cy1zdHJpcHsKICAgIGRpc3BsYXk6Z3JpZCAhaW1wb3J0YW50OwogICAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7CiAgICBib3JkZXItcmFkaXVzOjEwcHg7CiAgICBtYXJnaW4tYm90dG9tOjE2cHg7CiAgfQogIC5zYy1kaXZpZGVye2Rpc3BsYXk6bm9uZX0KICAuc2MtaXRlbXsKICAgIHBhZGRpbmc6MTJweCAxNHB4OwogICAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICAgIGJvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICB9CiAgLnNjLWl0ZW06bnRoLWNoaWxkKDJuKXtib3JkZXItcmlnaHQ6bm9uZX0KICAuc2MtaXRlbTpudGgtbGFzdC1jaGlsZCgtbisyKXtib3JkZXItYm90dG9tOm5vbmV9CiAgLnNjLXZhbHtmb250LXNpemU6MThweCAhaW1wb3J0YW50fQoKICAvKiBTaWduYXR1cmUgaW5zaWdodCArIG5hcnJhdGl2ZSBzdHJpcCDigJQgc3RhY2sgdmVydGljYWxseSAqLwogIC5oZXJvID4gZGl2W3N0eWxlKj0iZGlzcGxheTpmbGV4Il1bc3R5bGUqPSJnYXA6MThweCJdewogICAgZmxleC1kaXJlY3Rpb246Y29sdW1uICFpbXBvcnRhbnQ7CiAgICBnYXA6MTRweCAhaW1wb3J0YW50OwogICAgcGFkZGluZzowICFpbXBvcnRhbnQ7CiAgICBtYXJnaW4tdG9wOjEycHggIWltcG9ydGFudDsKICB9CiAgLnNpZ25hdHVyZS1pbnNpZ2h0e21hcmdpbi10b3A6MCAhaW1wb3J0YW50fQogIC5zaS10ZXh0e2ZvbnQtc2l6ZToxNHB4ICFpbXBvcnRhbnR9CgogIC8qIE5hcnJhdGl2ZSBzaGlmdHMgcGFuZWwg4oCUIGhpZGUgb24gbW9iaWxlIChzaG93biBiZWxvdyBtYXAgaW5zdGVhZCkgKi8KICAuc2hpZnQtcGFuZWx7ZGlzcGxheTpub25lfQoKICAvKiDilIDilIAgTUFQIFNFQ1RJT04g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLm1haW57CiAgICBkaXNwbGF5OmZsZXggIWltcG9ydGFudDsKICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbiAhaW1wb3J0YW50OwogICAgcGFkZGluZzowIDEycHggMjBweCAhaW1wb3J0YW50OwogICAgZ2FwOjAgIWltcG9ydGFudDsKICB9CgogIC8qIE1hcCBjYXJkIOKAlCBmdWxsIHdpZHRoLCBubyBib3JkZXIgcmFkaXVzIG9uIHNpZGVzICovCiAgLm1hcC1jYXJkewogICAgcG9zaXRpb246cmVsYXRpdmUgIWltcG9ydGFudDsKICAgIHRvcDphdXRvICFpbXBvcnRhbnQ7CiAgICBib3JkZXItcmFkaXVzOjE0cHg7CiAgICBtYXJnaW4tYm90dG9tOjA7CiAgfQoKICAvKiBMYXllciB0YWJzIOKAlCBjb21wYWN0ICovCiAgLmx0YWJze2dhcDo0cHg7cGFkZGluZzoxMHB4IDEycHh9CiAgLmx0YWJ7CiAgICBmb250LXNpemU6OHB4OwogICAgcGFkZGluZzo0cHggOHB4OwogICAgbGV0dGVyLXNwYWNpbmc6MC4wNmVtOwogIH0KICAubHRhYi1pbmZve2Rpc3BsYXk6bm9uZX0gLyogaGlkZSBpIGJ1dHRvbnMgb24gbW9iaWxlIOKAlCB0b29sdGlwIHVudXNhYmxlICovCiAgLm1hcC1sZWdlbmR7cGFkZGluZzo4cHggMTJweH0KICAjbWFwLW1ldGF7Zm9udC1zaXplOjhweH0KCiAgLyog4pSA4pSAIFNUQVRFIFBBTkVMIOKAlCBiZWNvbWVzIGJvdHRvbSBkcmF3ZXIgb24gbW9iaWxlIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogIC5zdGF0ZS1wYW5lbHsKICAgIHBvc2l0aW9uOmZpeGVkICFpbXBvcnRhbnQ7CiAgICBib3R0b206MCAhaW1wb3J0YW50OwogICAgbGVmdDowICFpbXBvcnRhbnQ7CiAgICByaWdodDowICFpbXBvcnRhbnQ7CiAgICB0b3A6YXV0byAhaW1wb3J0YW50OwogICAgbWF4LWhlaWdodDo3MnZoICFpbXBvcnRhbnQ7CiAgICBib3JkZXItcmFkaXVzOjIwcHggMjBweCAwIDAgIWltcG9ydGFudDsKICAgIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKSAhaW1wb3J0YW50OwogICAgYm9yZGVyLWJvdHRvbTpub25lICFpbXBvcnRhbnQ7CiAgICB6LWluZGV4OjgwMDAgIWltcG9ydGFudDsKICAgIHRyYW5zZm9ybTp0cmFuc2xhdGVZKDEwMCUpICFpbXBvcnRhbnQ7CiAgICB0cmFuc2l0aW9uOnRyYW5zZm9ybSAwLjM1cyBjdWJpYy1iZXppZXIoMC4zMiwwLjcyLDAsMSkgIWltcG9ydGFudDsKICAgIHBhZGRpbmc6MCAhaW1wb3J0YW50OwogICAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjBweCkgIWltcG9ydGFudDsKICAgIGJhY2tncm91bmQ6cmdiYSg5LDEzLDIxLDAuOTcpICFpbXBvcnRhbnQ7CiAgICBvdmVyZmxvdy15OmF1dG87CiAgfQogIC8qIERyYWcgaGFuZGxlICovCiAgLnN0YXRlLXBhbmVsOjpiZWZvcmV7CiAgICBjb250ZW50OicnOwogICAgZGlzcGxheTpibG9jazsKICAgIHdpZHRoOjM2cHg7aGVpZ2h0OjRweDsKICAgIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjE1KTsKICAgIGJvcmRlci1yYWRpdXM6MnB4OwogICAgbWFyZ2luOjEycHggYXV0byA4cHggYXV0bzsKICAgIGZsZXgtc2hyaW5rOjA7CiAgfQogIC8qIFNob3cgcGFuZWwgd2hlbiBzdGF0ZSBzZWxlY3RlZCAqLwogIC5zdGF0ZS1wYW5lbC5wYW5lbC1vcGVuewogICAgdHJhbnNmb3JtOnRyYW5zbGF0ZVkoMCkgIWltcG9ydGFudDsKICB9CiAgI3N0YXRlLWRldGFpbHtwYWRkaW5nOjAgMThweCAzMnB4fQogIC5zcC1oZWFke3BhZGRpbmc6NHB4IDAgMTJweH0KICAuc3AtbmFtZXtmb250LXNpemU6MjJweH0KCiAgLyogRGltIG92ZXJsYXkgd2hlbiBwYW5lbCBvcGVuICovCiAgLm1hcC1vdmVybGF5LWRpbXsKICAgIGRpc3BsYXk6bm9uZTsKICAgIHBvc2l0aW9uOmZpeGVkO2luc2V0OjA7CiAgICBiYWNrZ3JvdW5kOnJnYmEoMCwwLDAsMC40KTsKICAgIHotaW5kZXg6Nzk5OTsKICAgIGFuaW1hdGlvbjpmYWRlSW4gMC4ycyBlYXNlOwogIH0KICAubWFwLW92ZXJsYXktZGltLmFjdGl2ZXtkaXNwbGF5OmJsb2NrfQogIEBrZXlmcmFtZXMgZmFkZUlue2Zyb217b3BhY2l0eTowfXRve29wYWNpdHk6MX19CgogIC8qIOKUgOKUgCBOQVJSQVRJVkUgQ0FSRFMgKGJlbG93IG1hcCkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLm5hci1yb3d7CiAgICBkaXNwbGF5OmZsZXggIWltcG9ydGFudDsKICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbiAhaW1wb3J0YW50OwogICAgcGFkZGluZzoxNnB4IDEycHggIWltcG9ydGFudDsKICAgIGdhcDoxMnB4ICFpbXBvcnRhbnQ7CiAgfQogIC5uYXItY2FyZHtib3JkZXItcmFkaXVzOjEycHh9CiAgLm5jLWhlYWR7cGFkZGluZzoxMnB4IDE2cHh9CiAgLm5jLWJvZHl7cGFkZGluZzo0cHggMTZweCAxNHB4fQogIC5uYy10aXRsZXtmb250LXNpemU6MTRweH0KCiAgLyog4pSA4pSAIFJFUExBWSBJTkRJQSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KICAucmVwbGF5LXNlY3Rpb257cGFkZGluZzowIDEycHggMjRweH0KICAucmVwbGF5LWhlYWRlcntmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweH0KICAucmVwbGF5LWNvbnRyb2xze3dpZHRoOjEwMCU7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW59CiAgLnJwLWJ0bntmbGV4OjE7dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzo2cHggNHB4O2ZvbnQtc2l6ZTo4cHh9CiAgLnJlcGxheS1zbmFwc2hvdHtkaXNwbGF5Om5vbmV9IC8qIGhpZGUgc3RhdGUgY2FyZHMgb24gbW9iaWxlICovCgogIC8qIOKUgOKUgCBGT09URVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLmZvb3R7cGFkZGluZzozMnB4IDIwcHggNDhweH0KICAuZm9vdC1uYW1le2ZvbnQtc2l6ZToxM3B4fQogIC5mb290LWxpbmV7Zm9udC1zaXplOjExcHh9CgogIC8qIOKUgOKUgCBMT0FERVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgI2FwcC1sb2FkZXIgPiBkaXY6Zmlyc3Qtb2YtdHlwZXttYXJnaW4tYm90dG9tOjI0cHh9Cgp9Ci8qIEVORCBNT0JJTEUgKi8KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9Imx0YWItdG9vbHRpcCI+PC9kaXY+Cgo8IS0tIExPQURFUiAtLT4KPGRpdiBpZD0iYXBwLWxvYWRlciIgc3R5bGU9IgogIHBvc2l0aW9uOmZpeGVkO2luc2V0OjA7ei1pbmRleDo5OTk5ODsKICBiYWNrZ3JvdW5kOiMwNjA5MTA7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBnYXA6MDsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC44cyBlYXNlLCB2aXNpYmlsaXR5IDAuOHMgZWFzZTsKIj4KICA8IS0tIFNpZ25hbCByaW5ncyAtLT4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt3aWR0aDo2NHB4O2hlaWdodDo2NHB4O21hcmdpbi1ib3R0b206MzZweCI+CiAgICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDoyNHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6I2UwNWEyODthbmltYXRpb246bGRyUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNSk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuNXMiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6LTEwcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjA3KTthbmltYXRpb246bGRyUmluZyAycyBlYXNlLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MXMiPjwvZGl2PgogIDwvZGl2PgoKICA8IS0tIEJyYW5kIC0tPgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidQbGF5ZmFpciBEaXNwbGF5JyxHZW9yZ2lhLHNlcmlmO2ZvbnQtc2l6ZTpjbGFtcCgyOHB4LDV2dyw0MnB4KTtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtjb2xvcjojZjBlY2U0O2xpbmUtaGVpZ2h0OjE7bWFyZ2luLWJvdHRvbToxMHB4Ij4KICAgIDxlbSBzdHlsZT0iY29sb3I6I2U4YzRhMDtmb250LXN0eWxlOml0YWxpYyI+UHVsc2U8L2VtPiBvZiBJbmRpYQogIDwvZGl2PgoKICA8IS0tIFRhZ2xpbmUgLS0+CiAgPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6J0NvdXJpZXIgTmV3Jyxtb25vc3BhY2U7Zm9udC1zaXplOjExcHg7bGV0dGVyLXNwYWNpbmc6MC4yOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuNCk7bWFyZ2luLWJvdHRvbToyOHB4Ij4KICAgIFRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMKICA8L2Rpdj4KCiAgPCEtLSBOb3QgbmV3cyBsaW5lIC0tPgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjI1KTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4Ij4KICAgIDxzcGFuPk5vdCBuZXdzPC9zcGFuPgogICAgPHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPk5vdCBwcmVkaWN0aW9uPC9zcGFuPgogICAgPHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPkp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KTthbmltYXRpb246bGRyR2xvdyAycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+b2JzZXJ2YXRpb248L3NwYW4+PC9zcGFuPgogIDwvZGl2PgoKICA8IS0tIExvYWRpbmcgZG90cyAtLT4KICA8ZGl2IHN0eWxlPSJtYXJnaW4tdG9wOjQ4cHg7ZGlzcGxheTpmbGV4O2dhcDo2cHgiPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjJzIj48L3NwYW4+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MC40cyI+PC9zcGFuPgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KQGtleWZyYW1lcyBsZHJQdWxzZXswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC41O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KQGtleWZyYW1lcyBsZHJSaW5nezAle3RyYW5zZm9ybTpzY2FsZSgwLjgpO29wYWNpdHk6MC42fTEwMCV7dHJhbnNmb3JtOnNjYWxlKDEuNSk7b3BhY2l0eTowfX0KQGtleWZyYW1lcyBsZHJHbG93ezAlLDEwMCV7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KX01MCV7dGV4dC1zaGFkb3c6MCAwIDIwcHggcmdiYSg1NywyNTUsMjAsMC45KSwwIDAgNDBweCByZ2JhKDU3LDI1NSwyMCwwLjMpfX0KQGtleWZyYW1lcyBsZHJEb3R7MCUsODAlLDEwMCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjN9NDAle3RyYW5zZm9ybTpzY2FsZSgxKTtvcGFjaXR5OjF9fQo8L3N0eWxlPgoKPGRpdiBjbGFzcz0idG9wYmFyIj4KICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC1tYXJrIj48c3BhbiBjbGFzcz0iYnJhbmQtcHVsc2UtZG90Ij48L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC10ZXh0LWJsb2NrIj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLW5hbWUiPjxlbSBjbGFzcz0iYnJhbmQtcHVsc2Utd29yZCI+UHVsc2U8L2VtPiBvZiBJbmRpYTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXIiPgogICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgPHNwYW4gaWQ9ImxpdmUtY291bnQiPuKApjwvc3Bhbj4gc2lnbmFscwogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSIKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7CiAgYmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpOwogIHBhZGRpbmc6MCAzNnB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOwoiPgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLXNpZ25hbHMiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlNpZ25hbHMgdHJhY2tlZDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIj5MaXZlIGluZ2VzdGlvbjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1ob3R0ZXN0IiBzdHlsZT0iY3Vyc29yOnBvaW50ZXIiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWhvdHRlc3QtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj5DbGljayB0byBleHBsb3JlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWFuZ2VyLXN1YiI+T3V0cmFnZSAmIHByb3Rlc3Qgc2lnbmFsczwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlRvcCByaXNpbmcgbmFycmF0aXZlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1uYXJyYXRpdmUtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hcnJhdGl2ZS1zdWIiPk5hdGlvbmFsIHNpZ25hbCBzdXJnZTwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbGluZy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbGluZy1zdWIiPlNpZ25hbCBkZWNheTwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KLnN0YXQtY2VsbHsKICBmbGV4OjE7cGFkZGluZzoxMHB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5zdGF0LWNlbGw6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpfQouc3RhdC1kaXZ7d2lkdGg6MXB4O2JhY2tncm91bmQ6cmdiYSgxNjAsMTkwLDIzMCwwLjA3KTtmbGV4LXNocmluazowO21hcmdpbjo4cHggMH0KLnNjLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW4tdG9wOjFweH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KPC9zdHlsZT4KCgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4O21pbi1oZWlnaHQ6MjgwcHg7Ij4KICAgIDxkaXYgY2xhc3M9InNpZ25hdHVyZS1pbnNpZ2h0IiBzdHlsZT0ibWFyZ2luLXRvcDowO2ZsZXg6MTttaW4td2lkdGg6MCI+CiAgICAgIDxkaXYgY2xhc3M9InNpLWxhYmVsIj5JbmRpYSByaWdodCBub3c8L2Rpdj4KICAgICAgPGRpdiBpZD0ic2lnLWluc2lnaHQiPgogICAgICAgIDxkaXYgY2xhc3M9InNpLXRleHQiIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOjE0cHgiPkFuYWx5c2luZyBzaWduYWxzIGFjcm9zcyAzMCBzdGF0ZXMuLi48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgaWQ9InNpZy10YWdzIj48L2Rpdj4KICAgICAgPGRpdiBpZD0ic2lnLW1ldGEiPjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IHN0eWxlPSJmbGV4OjAgMCAzNjBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7b3ZlcmZsb3c6aGlkZGVuO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Ij4KICAgICAgPCEtLSBoZWFkZXIgLS0+CiAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzoxMHB4IDE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowOyI+CiAgICAgICAgPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCkiPk5hcnJhdGl2ZSBzaGlmdHM8L3NwYW4+CiAgICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoycHg7Ij4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiBhY3RpdmUiIGRhdGEtcGVyaW9kPSIzbSI+M008L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjZtIj42TTwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIiBkYXRhLXBlcmlvZD0iMXkiPjFZPC9idXR0b24+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8IS0tIHNoaWZ0cyBsaXN0IC0tPgogICAgICA8ZGl2IHN0eWxlPSJmbGV4OjE7b3ZlcmZsb3c6aGlkZGVuO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47anVzdGlmeS1jb250ZW50OmNlbnRlcjtwYWRkaW5nOjEwcHggMTRweDtnYXA6NnB4OyIgaWQ9InNoaWZ0LWxpc3QiPjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCgo8IS0tIE1BSU46IE1BUCArIFNUQVRFIFBBTkVMIC0tPgo8ZGl2IGNsYXNzPSJtYWluIj4KCiAgPGRpdiBjbGFzcz0ibWFwLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibWFwLXRvcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC10aXRsZS1ibG9jayI+CiAgICAgICAgPGRpdiBjbGFzcz0ibXQiPkluZGlhICZtZGFzaDsgY29sbGVjdGl2ZSBhdHRlbnRpb248L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtcyIgaWQ9Im1hcC1tZXRhIj4zMCBzdGF0ZXMgJm1pZGRvdDsgbGl2ZSBzaWduYWwgY29tcG9zaXRlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJsZWdlbmQiPjxzcGFuPnF1aWV0PC9zcGFuPjxkaXYgY2xhc3M9ImxlZ2VuZC1iYXIiPjwvZGl2PjxzcGFuPmFjdGl2ZTwvc3Bhbj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibGF5ZXItcm93Ij4KICAgICAgPHNwYW4gY2xhc3M9ImxheWVyLWxhYmVsIj5WaWV3PC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJsdGFicyI+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIgYWN0aXZlIiBkYXRhLWxheWVyPSJhdHRlbnRpb24iPkF0dGVudGlvbiA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iV2hpY2ggc3RhdGVzIGFyZSByZWNlaXZpbmcgdGhlIG1vc3QgcHVibGljIGZvY3VzLiBIaWdoIGF0dGVudGlvbiA9IGNvbmNlbnRyYXRlZCBuZXdzIGNvdmVyYWdlIGFuZCBwb2xpdGljYWwgYWN0aXZpdHkuIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiIgZGF0YS1sYXllcj0iZW1vdGlvbiI+RW1vdGlvbiA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iVGhlIGRvbWluYW50IGVtb3Rpb25hbCB0b25lIOKAlCBhbnhpb3VzLCBhbmdyeSwgaG9wZWZ1bCwgcHJvdWQgb3IgZmVhcmZ1bC4gUmV2ZWFscyB0aGUgcHN5Y2hvbG9naWNhbCB1bmRlcmN1cnJlbnQgb2YgcG9saXRpY2FsIGF0dGVudGlvbi4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJ2ZWxvY2l0eSI+TW9tZW50dW0gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IklzIGF0dGVudGlvbiByaXNpbmcgb3IgZmFsbGluZz8gUmlzaW5nID0gbmFycmF0aXZlIGFjY2VsZXJhdGluZy4gQ29vbGluZyA9IGxvc2luZyB0cmFjdGlvbi4gU2hvd3Mgc3RhdGVzIGVudGVyaW5nIG9yIGV4aXRpbmcgYSBwb2xpdGljYWwgY3ljbGUuIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im1hcC1zdmctd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC1pbm5lciI+CiAgICAgICAgPHN2ZyBpZD0iaW5kaWEtbWFwIiB2aWV3Qm94PSIwIDAgODAwIDgwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgbWVldCI+CiAgICAgICAgICA8ZGVmcz4KICAgICAgICAgICAgPHJhZGlhbEdyYWRpZW50IGlkPSJhbWJHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9InJnYmEoMjI0LDkwLDQwLDAuMDQpIi8+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICAgICAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgICAgICAgICA8ZmlsdGVyIGlkPSJzdGF0ZUdsb3ciIHg9Ii0zMCUiIHk9Ii0zMCUiIHdpZHRoPSIxNjAlIiBoZWlnaHQ9IjE2MCUiPgogICAgICAgICAgICAgIDxmZUdhdXNzaWFuQmx1ciBpbj0iU291cmNlR3JhcGhpYyIgc3RkRGV2aWF0aW9uPSI4IiByZXN1bHQ9ImJsdXIiLz4KICAgICAgICAgICAgICA8ZmVDb21wb3NpdGUgaW49IlNvdXJjZUdyYXBoaWMiIGluMj0iYmx1ciIgb3BlcmF0b3I9Im92ZXIiLz4KICAgICAgICAgICAgPC9maWx0ZXI+CiAgICAgICAgICA8L2RlZnM+CiAgICAgICAgICA8cmVjdCB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgZmlsbD0idXJsKCNhbWJHbG93KSIvPgogICAgICAgICAgPGcgaWQ9Im1hcC1nbG93Ij48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXN0YXRlcyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1wdWxzZXMiPjwvZz4KICAgICAgICA8L3N2Zz4KICAgICAgICA8ZGl2IGNsYXNzPSJtYXAtdG9vbHRpcCIgaWQ9InRvb2x0aXAiPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFNUQVRFIFBBTkVMIC0tPgogIDxkaXYgY2xhc3M9InN0YXRlLXBhbmVsIiBpZD0ic3RhdGUtZGV0YWlsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLWVtcHR5Ij4KICAgICAgPHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMSI+CiAgICAgICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgOHY0TTEyIDE2aC4wMSIvPgogICAgICA8L3N2Zz4KICAgICAgPGRpdiBjbGFzcz0icGUtdCI+U2VsZWN0IGEgc3RhdGU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0icGUtcyI+Q2xpY2sgYW55IHJlZ2lvbiBvbiB0aGUgbWFwPGJyLz50byBvcGVuIGl0cyBuYXJyYXRpdmUgcGFuZWwuPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCjwvZGl2PgoKPCEtLSBOQVJSQVRJVkUgUk9XIC0tPgo8ZGl2IGNsYXNzPSJuYXItcm93Ij4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj48c3BhbiBjbGFzcz0ibmMtZG90IHJpc2UyIj48L3NwYW4+PHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SaXNpbmcgbmFycmF0aXZlczwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyaXNpbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj48c3BhbiBjbGFzcz0ibmMtZG90IGZhbGwiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPkRlY2xpbmluZyBuYXJyYXRpdmVzPC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9ImRlY2xpbmluZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJlZ2lvbmFsIHNoaWZ0czwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyZWdpb25hbC1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gUkVQTEFZIElORElBIC0tPgo8c2VjdGlvbiBjbGFzcz0icmVwbGF5LXNlY3Rpb24iPgogIDxkaXYgY2xhc3M9InJlcGxheS1oZWFkZXIiPgogICAgPGRpdj48ZGl2IGNsYXNzPSJyZXBsYXktbGFiZWwiPlJlcGxheSBJbmRpYTwvZGl2PjxkaXYgY2xhc3M9InJlcGxheS1zdWIiPldhdGNoIGhvdyBjb2xsZWN0aXZlIGF0dGVudGlvbiBzaGlmdGVkIG92ZXIgdGltZTwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0icmVwbGF5LWNvbnRyb2xzIj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIGFjdGl2ZSIgZGF0YS1wZXJpb2Q9IjdkIj43IGRheXM8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIiBkYXRhLXBlcmlvZD0iMzBkIj4zMCBkYXlzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biIgZGF0YS1wZXJpb2Q9IjZtIj42IG1vbnRoczwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4iIGRhdGEtcGVyaW9kPSJlbGVjdGlvbiI+RWxlY3Rpb25zPC9idXR0b24+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktc2NydWJiZXIiPgogICAgPGRpdiBjbGFzcz0icnAtdHJhY2siIGlkPSJycC10cmFjayI+PGRpdiBjbGFzcz0icnAtZmlsbCIgaWQ9InJwLWZpbGwiPjwvZGl2PjxkaXYgY2xhc3M9InJwLXRodW1iIiBpZD0icnAtdGh1bWIiPjwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0icnAtZGF0ZXMiIGlkPSJycC1kYXRlcyI+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0icmVwbGF5LXBsYXliYWNrIj4KICAgIDxidXR0b24gY2xhc3M9InJwLXBsYXkiIGlkPSJycC1wbGF5LWJ0biIgb25jbGljaz0idG9nZ2xlUmVwbGF5KCkiPgogICAgICA8c3ZnIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwb2x5Z29uIHBvaW50cz0iMiwxIDksNSAyLDkiIGlkPSJycC1wbGF5LWljb24iLz48L3N2Zz4KICAgIDwvYnV0dG9uPgogICAgPGRpdiBjbGFzcz0icnAtY3VycmVudC1kYXRlIiBpZD0icnAtY3VycmVudC1kYXRlIj5TZWxlY3QgYSBwZXJpb2QgYW5kIHByZXNzIHBsYXk8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJwLXNwZWVkIj48c3BhbiBjbGFzcz0icnAtc3BlZWQtbGFiZWwiPlNwZWVkPC9zcGFuPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQgYWN0aXZlIiBkYXRhLXNwZD0iMSI+MXg8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iMiI+Mng8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iNCI+NHg8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InJlcGxheS1zbmFwc2hvdCI+CiAgICA8ZGl2IGNsYXNzPSJycC1zbmFwLWxhYmVsIj5OYXRpb25hbCBuYXJyYXRpdmUgYXQgdGhpcyBtb21lbnQ8L2Rpdj4KICAgIDxkaXYgaWQ9InJwLW5hdGlvbmFsLW5hcnJhdGl2ZSIgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTRweCwxLjN2dywxOHB4KTtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjY7cGFkZGluZzoxMnB4IDAgNHB4IDA7bWluLWhlaWdodDo2MHB4Ij4KICAgICAgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweCI+UHJlc3MgcGxheSB0byBvYnNlcnZlIEluZGlhIGluIG1vdGlvbi48L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgaWQ9InJwLW5hcnJhdGl2ZS10YWdzIiBzdHlsZT0iZGlzcGxheTpmbGV4O2ZsZXgtd3JhcDp3cmFwO2dhcDo2cHg7bWFyZ2luLXRvcDo4cHgiPjwvZGl2PgogICAgPGRpdiBpZD0icnAtbmFycmF0aXZlLW1ldGEiIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnJnYmEoNjIsNzcsOTYsMC40KTttYXJnaW4tdG9wOjEwcHg7bGV0dGVyLXNwYWNpbmc6MC4wNWVtIj48L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgo8c3R5bGU+Ci5yZXBsYXktc2VjdGlvbntwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7bWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAzNnB4IDM2cHh9Ci5yZXBsYXktaGVhZGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LWVuZDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjIwcHg7Z2FwOjIwcHg7ZmxleC13cmFwOndyYXB9Ci5yZXBsYXktbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMHB4O2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1pbmspO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5yZXBsYXktc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDo0cHh9Ci5yZXBsYXktY29udHJvbHN7ZGlzcGxheTpmbGV4O2dhcDo0cHg7ZmxleC13cmFwOndyYXB9Ci5ycC1idG57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtwYWRkaW5nOjVweCAxMnB4O2JvcmRlci1yYWRpdXM6NHB4O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2NvbG9yOnZhcigtLWZhaW50KTtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE1c30KLnJwLWJ0bi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpfQoucmVwbGF5LXNjcnViYmVye2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTJweDtwYWRkaW5nOjE4cHggMjBweCAxNHB4O21hcmdpbi1ib3R0b206MTJweH0KLnJwLXRyYWNre3Bvc2l0aW9uOnJlbGF0aXZlO2hlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MnB4O2N1cnNvcjpwb2ludGVyO21hcmdpbi1ib3R0b206MTBweH0KLnJwLWZpbGx7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjAlO2JhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIHJpZ2h0LHJnYmEoMjI0LDkwLDQwLDAuNCksdmFyKC0tYWNjZW50KSk7Ym9yZGVyLXJhZGl1czoycHh9Ci5ycC10aHVtYntwb3NpdGlvbjphYnNvbHV0ZTt0b3A6NTAlO3RyYW5zZm9ybTp0cmFuc2xhdGUoLTUwJSwtNTAlKTt3aWR0aDoxMnB4O2hlaWdodDoxMnB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3JkZXI6MnB4IHNvbGlkIHJnYmEoOSwxMywyMSwwLjgpO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDIyNCw5MCw0MCwwLjQpO2xlZnQ6MCU7Y3Vyc29yOmdyYWJ9Ci5ycC1kYXRlc3tkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5yZXBsYXktcGxheWJhY2t7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDttYXJnaW4tYm90dG9tOjE2cHh9Ci5ycC1wbGF5e3dpZHRoOjI4cHg7aGVpZ2h0OjI4cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4yNSk7Y29sb3I6dmFyKC0tYWNjZW50KTtjdXJzb3I6cG9pbnRlcjtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xNXN9Ci5ycC1jdXJyZW50LWRhdGV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTtmbGV4OjF9Ci5ycC1zcGVlZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo0cHh9Ci5ycC1zcGVlZC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tcmlnaHQ6MnB4fQoucnAtc3Bke2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtwYWRkaW5nOjNweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Y29sb3I6dmFyKC0tZmFpbnQpO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQoucnAtc3BkLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyKX0KLnJlcGxheS1zbmFwc2hvdHtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjEycHg7cGFkZGluZzoxNnB4IDIwcHh9Ci5ycC1zbmFwLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMnB4fQoucnAtc25hcC1zdGF0ZXN7ZGlzcGxheTpmbGV4O2ZsZXgtd3JhcDp3cmFwO2dhcDo4cHh9Ci5ycC1sb2ctZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9Ci5ycC1zdGF0ZS1jYXJke3BhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo2cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTttaW4td2lkdGg6MTQwcHh9Ci5ycC1zdGF0ZS1uYW1le2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwO21hcmdpbi1ib3R0b206M3B4fQoucnAtc3RhdGUtbmFye2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5ycC1zdGF0ZS1hdHR7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1hY2NlbnQpfQo8L3N0eWxlPgo8IS0tIEZBVlMgLS0+CjxzZWN0aW9uIGNsYXNzPSJmYXZzIj4KICA8ZGl2IGNsYXNzPSJmYXZzLWxhYmVsIj5UcmFja2VkIHN0YXRlczwvZGl2PgogIDxkaXYgY2xhc3M9ImZhdnMtcm93IiBpZD0iZmF2LXJvdyI+CiAgICA8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKPGRpdiBjbGFzcz0iZm9vdCI+CiAgPGRpdiBjbGFzcz0iZm9vdC1uYW1lIj5QdWxzZSBvZiBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3QtbGluZSI+T2JzZXJ2ZXMgaG93IHB1YmxpYyBhdHRlbnRpb24gc2hpZnRzIGFjcm9zcyB0aGUgY291bnRyeSDigJQgdXNpbmcgc2lnbmFscyBmcm9tIG5ld3MsIGRpc2NvdXJzZSwgYW5kIHJlZ2lvbmFsIGRldmVsb3BtZW50cy48L2Rpdj4KICA8ZGl2IGNsYXNzPSJmb290LXN1YiI+Tm90IG5ld3MuIE5vdCBwcmVkaWN0aW9uLiBKdXN0IDxzcGFuIHN0eWxlPSJjb2xvcjojMzlmZjE0O3RleHQtc2hhZG93OjAgMCA4cHggcmdiYSg1NywyNTUsMjAsMC40KSI+b2JzZXJ2YXRpb248L3NwYW4+LjwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICBpZighcm93c3x8IXJvd3MubGVuZ3RoKSByZXR1cm47CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocm93LmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb21FbW89cm93LmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihlbW9zKXx8bnVsbDsKICAgICAgdmFyIGVudHJ5PXthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbUVtbyxkb21pbmFudF9uYXJyYXRpdmU6cm93LmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgICAgTElWRVtyb3cubmFtZV09ZW50cnk7CiAgICAgIGlmKCFTRFtyb3cubmFtZV0pIFNEW3Jvdy5uYW1lXT1PYmplY3QuYXNzaWduKHt9LERFRkFVTFQpOwogICAgICBPYmplY3QuYXNzaWduKFNEW3Jvdy5uYW1lXSxlbnRyeSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIHNldFRpbWVvdXQocmVuZGVyTW9tZW50dW0sIDUwMCk7CiAgICBpZihTRUwmJkxJVkVbU0VMXSYmZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpKSByZW5kZXJQYW5lbChTRUwpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tBUEldJyxlLm1lc3NhZ2UpO30KfQoKZnVuY3Rpb24gYnVpbGRMb2NhbEluc2lnaHQoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKCiAgLy8gQWdncmVnYXRlIHRvcCBuYXJyYXRpdmVzIGFjcm9zcyBhbGwgc3RhdGVzCiAgdmFyIG5hcj17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIGlmKCFuYXJbbi5uYW1lXSkgbmFyW24ubmFtZV09e3VwOjAsZG93bjowLGZsYXQ6MCx0b3RhbDowfTsKICAgICAgbmFyW24ubmFtZV1bbi5kaXJdPShuYXJbbi5uYW1lXVtuLmRpcl18fDApK24udmFsOwogICAgICBuYXJbbi5uYW1lXS50b3RhbD0obmFyW24ubmFtZV0udG90YWx8fDApK24udmFsOwogICAgfSk7CiAgfSk7CgogIC8vIFRvcCByaXNpbmcgYW5kIGZhbGxpbmcgKGV4Y2x1ZGUgdGllcyB3aGVyZSBzYW1lIG5hbWUgcmlzZXMgYW5kIGZhbGxzKQogIHZhciByaXNpbmc9T2JqZWN0LmVudHJpZXMobmFyKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS51cD5rdlsxXS5kb3duO30pCiAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnVwLWFbMV0udXA7fSkuc2xpY2UoMCwzKTsKICB2YXIgZmFsbGluZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvd24+a3ZbMV0udXA7fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0uZG93bi1hWzFdLmRvd247fSkuc2xpY2UoMCwyKTsKICB2YXIgdG9wMz1PYmplY3QuZW50cmllcyhuYXIpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS50b3RhbC1hWzFdLnRvdGFsO30pLnNsaWNlKDAsMyk7CgogIC8vIEhvdHRlc3Qgc3RhdGUKICB2YXIgaG90dGVzdD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgdmFyIGhvdHRlc3RFbW89aG90dGVzdD8oTElWRVtob3R0ZXN0WzBdXSYmTElWRVtob3R0ZXN0WzBdXS5kb21pbmFudF9lbW90aW9uKXx8Jyc6JycgOwoKICAvLyBCdWlsZCBpbnNpZ2h0IHRleHQg4oCUIG1vcmUgYW5hbHl0aWNhbCwgY29udGV4dC1hd2FyZQogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgaWYoIWVsKSByZXR1cm47CgogIC8vIENvdW50IGhvdyBtYW55IHN0YXRlcyBhcmUgYWN0aXZlCiAgdmFyIGFjdGl2ZVN0YXRlcz1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihlKXtyZXR1cm4gZVsxXS5hdHRlbnRpb24+NTt9KS5sZW5ndGg7CiAgdmFyIGRvbWluYW50UmVnaW9uPScnOwogIC8vIEZpbmQgd2hpY2ggcmVnaW9uIGhhcyBtb3N0IGFjdGl2ZSBzdGF0ZXMKICB2YXIgcmVnaW9uTWFwPXsnTm9ydGgnOlsnRGVsaGknLCdVdHRhciBQcmFkZXNoJywnUHVuamFiJywnSGFyeWFuYScsJ0phbW11IGFuZCBLYXNobWlyJ10sCiAgICAnU291dGgnOlsnVGFtaWwgTmFkdScsJ0thcm5hdGFrYScsJ0tlcmFsYScsJ0FuZGhyYSBQcmFkZXNoJywnVGVsYW5nYW5hJ10sCiAgICAnV2VzdCc6WydNYWhhcmFzaHRyYScsJ0d1amFyYXQnLCdSYWphc3RoYW4nLCdHb2EnXSwKICAgICdFYXN0JzpbJ1dlc3QgQmVuZ2FsJywnQmloYXInLCdPZGlzaGEnLCdKaGFya2hhbmQnXSwKICAgICdORSc6WydBc3NhbScsJ01hbmlwdXInLCdOYWdhbGFuZCcsJ01pem9yYW0nLCdNZWdoYWxheWEnLCdUcmlwdXJhJywnQXJ1bmFjaGFsIFByYWRlc2gnXX07CiAgdmFyIHRvcFJlZ2lvblNjb3JlPTA7CiAgT2JqZWN0LmVudHJpZXMocmVnaW9uTWFwKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBzY29yZT1rdlsxXS5yZWR1Y2UoZnVuY3Rpb24ocyxzdCl7cmV0dXJuIHMrKChMSVZFW3N0XSYmTElWRVtzdF0uYXR0ZW50aW9uKXx8MCk7fSwwKTsKICAgIGlmKHNjb3JlPnRvcFJlZ2lvblNjb3JlKXt0b3BSZWdpb25TY29yZT1zY29yZTtkb21pbmFudFJlZ2lvbj1rdlswXTt9CiAgfSk7CgogIHZhciBsaW5lcz1bXTsKICBpZihyaXNpbmcubGVuZ3RoJiZmYWxsaW5nLmxlbmd0aCYmcmlzaW5nWzBdWzBdIT09ZmFsbGluZ1swXVswXSl7CiAgICAvLyBTdHJvbmcgbmFycmF0aXZlIHNoaWZ0CiAgICBsaW5lcy5wdXNoKCc8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgbGluZXMucHVzaCgnIGlzIGRlZmluaW5nIG5hdGlvbmFsIGRpc2NvdXJzZScpOwogICAgaWYoZmFsbGluZ1swXSkgbGluZXMucHVzaCgnLCBlY2xpcHNpbmcgPGVtPicrZmFsbGluZ1swXVswXSsnPC9lbT4nKTsKICAgIGlmKGhvdHRlc3QpIGxpbmVzLnB1c2goJy4gPHN0cm9uZz4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBpcyBhdCB0aGUgY2VudHJlJyk7CiAgICBpZihob3R0ZXN0RW1vJiZob3R0ZXN0RW1vIT09J251bGwnKSBsaW5lcy5wdXNoKCcg4oCUIHB1YmxpYyB0b25lIGlzIDxlbT4nK2hvdHRlc3RFbW8rJzwvZW0+Jyk7CiAgICBpZihyaXNpbmdbMV0mJnJpc2luZ1sxXVswXSE9PXJpc2luZ1swXVswXSkgbGluZXMucHVzaCgnLiA8ZW0+JytyaXNpbmdbMV1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzFdWzBdLnNsaWNlKDEpKyc8L2VtPiBpcyBidWlsZGluZyBpbiB0aGUgYmFja2dyb3VuZCcpOwogICAgaWYoZG9taW5hbnRSZWdpb24pIGxpbmVzLnB1c2goJy4gVGhlICcrZG9taW5hbnRSZWdpb24rJyBpcyBtb3N0IGFjdGl2ZSB0b2RheScpOwogIH0gZWxzZSBpZihyaXNpbmcubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ0F0dGVudGlvbiBpcyBjb25jZW50cmF0aW5nIGFyb3VuZCA8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnIHdpdGggPHN0cm9uZz4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBnZW5lcmF0aW5nIHRoZSBtb3N0IHNpZ25hbCcpOwogICAgaWYocmlzaW5nWzFdKSBsaW5lcy5wdXNoKCcuIDxlbT4nK3Jpc2luZ1sxXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMV1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIGFsc28gcmlzaW5nJyk7CiAgICBpZihhY3RpdmVTdGF0ZXM+MCkgbGluZXMucHVzaCgnLiAnK2FjdGl2ZVN0YXRlcysnIHN0YXRlcyBhcmUgaW4gYWN0aXZlIGRpc2NvdXJzZScpOwogIH0gZWxzZSBpZih0b3AzLmxlbmd0aCl7CiAgICBsaW5lcy5wdXNoKCdTaWduYWxzIGFyZSBkaXNwZXJzZWQgYWNyb3NzIDxlbT4nK3RvcDNbMF1bMF0rJzwvZW0+LCA8ZW0+Jyt0b3AzWzFdWzBdKyc8L2VtPicpOwogICAgaWYodG9wM1syXSkgbGluZXMucHVzaCgnIGFuZCA8ZW0+Jyt0b3AzWzJdWzBdKyc8L2VtPicpOwogICAgbGluZXMucHVzaCgnIOKAlCBubyBzaW5nbGUgZG9taW5hbnQgbmFycmF0aXZlIGhhcyBlbWVyZ2VkJyk7CiAgfQoKICBpZihsaW5lcy5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2xpbmVzLmpvaW4oJycpKyc8L2Rpdj4nOwogIH0KCiAgLy8gVGFncwogIGlmKHRFbCl7CiAgICB2YXIgdGFncz1bXTsKICAgIGZhbGxpbmcuc2xpY2UoMCwxKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICB0YWdzLnB1c2goJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSg1OSwxODQsMjE2LDAuMyk7Y29sb3I6IzNiYjhkOCI+4oaTICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIHJpc2luZy5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICB0YWdzLnB1c2goJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtjb2xvcjojZTA1YTI4Ij7ihpEgJytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKTsKICAgIH0pOwogICAgaWYodGFncy5sZW5ndGgpIHRFbC5pbm5lckhUTUw9dGFncy5qb2luKCcnKTsKICB9CgogIGlmKG1ldGFFbCl7CiAgICB2YXIgc3RhdGVDb3VudD1PYmplY3QudmFsdWVzKExJVkUpLmZpbHRlcihmdW5jdGlvbihzKXtyZXR1cm4gcy5hdHRlbnRpb24+Mjt9KS5sZW5ndGg7CiAgICBtZXRhRWwudGV4dENvbnRlbnQ9J09ic2VydmluZyAnK3N0YXRlQ291bnQrJyBhY3RpdmUgc3RhdGVzIMK3IHVwZGF0ZWQgJytuZXcgRGF0ZSgpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nLHtob3VyOicyLWRpZ2l0JyxtaW51dGU6JzItZGlnaXQnfSk7CiAgfQp9CgpmdW5jdGlvbiB1cGRhdGVBbGxTdHJpcHMoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKICB2YXIgaG90dGVzdD1lbnRyaWVzLnJlZHVjZShmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApPihhWzFdLmF0dGVudGlvbnx8MCk/YjphO30sZW50cmllc1swXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC12YWwnLGhvdHRlc3RbMF0pOwogIHNldFRleHQoJ3NjLWhvdHRlc3Qtc3ViJywnQXR0ZW50aW9uICcraG90dGVzdFsxXS5hdHRlbnRpb24udG9GaXhlZCgxKSk7CiAgdmFyIHRvcEFuZ2VyTm09bnVsbCx0b3BBbmdlclBjdD0wOwogIGVudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICB2YXIgZT1rdlsxXS5lbW90aW9uc3x8e307CiAgICB2YXIgYT1lLmFuZ2VyfHwwOwogICAgaWYoYT4wJiZhPD0xKSBhPU1hdGgucm91bmQoYSoxMDApOwogICAgaWYoYT50b3BBbmdlclBjdCl7dG9wQW5nZXJQY3Q9YTt0b3BBbmdlck5tPWt2WzBdO30KICB9KTsKICBpZih0b3BBbmdlck5tJiZ0b3BBbmdlclBjdD4wKXsKICAgIHNldFRleHQoJ3NjLWFuZ2VyLXZhbCcsdG9wQW5nZXJObSk7CiAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdBbmdlciAnK01hdGgucm91bmQodG9wQW5nZXJQY3QpKyclIG9mIHNpZ25hbHMnKTsKICB9IGVsc2UgewogICAgLy8gRmFsbCBiYWNrIHRvIGRvbWluYW50X2Vtb3Rpb249YW5nZXIKICAgIHZhciBhbmdlckRvbT1lbnRyaWVzLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvbWluYW50X2Vtb3Rpb249PT0nYW5nZXInO30pOwogICAgaWYoYW5nZXJEb20ubGVuZ3RoKXsKICAgICAgdmFyIHRvcEJ5QXR0PWFuZ2VyRG9tLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogICAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLHRvcEJ5QXR0WzBdKTsKICAgICAgc2V0VGV4dCgnc2MtYW5nZXItc3ViJywnRG9taW5hbnQgZW1vdGlvbjogYW5nZXInKTsKICAgIH0KICB9CiAgdmFyIGNvb2xpbmc9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0udmVsb2NpdHl8fDApPChhWzFdLnZlbG9jaXR5fHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1jb29saW5nLXZhbCcsY29vbGluZ1swXSk7c2V0VGV4dCgnc2MtY29vbGluZy1zdWInLCdWZWxvY2l0eSAnK2Nvb2xpbmdbMV0udmVsb2NpdHkudG9GaXhlZCgzKSk7CiAgdmFyIG5jPXt9O2VudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlKW5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV09KG5jW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZV18fDApKzE7fSk7CiAgdmFyIHRuPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KVswXTsKICBpZih0bil7c2V0VGV4dCgnc2MtbmFycmF0aXZlLXZhbCcsdG5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdG5bMF0uc2xpY2UoMSkpO3NldFRleHQoJ3NjLW5hcnJhdGl2ZS1zdWInLCdEb21pbmFudCBhY3Jvc3MgJyt0blsxXSsnIHN0YXRlcycpO30KfQphc3luYyBmdW5jdGlvbiBmZXRjaERldGFpbChuYW1lKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZS8nK2VuY29kZVVSSUNvbXBvbmVudChuYW1lKSk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhkLmVtb3Rpb25zfHx7fSk7CiAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8ZC5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgU0RbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGVtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlLAogICAgICBuYXJyYXRpdmVzOihkLm5hcnJhdGl2ZXN8fFtdKS5tYXAoZnVuY3Rpb24obil7cmV0dXJue25hbWU6bi5uYW1lLHZhbDpuLnZhbCxkaXI6bi5kaXJ8fCdmbGF0J307fSksCiAgICAgIHJpc2luZzpkLnJpc2luZ3x8W10sZmFsbGluZzpkLmZhbGxpbmd8fFtdLHN1bW1hcnk6ZC5zdW1tYXJ5fHxERUZBVUxULnN1bW1hcnksCiAgICAgIGFydGljbGVzOmQuYXJ0aWNsZXN8fFtdLHRpbWVsaW5lOmQudGltZWxpbmV8fERFRkFVTFQudGltZWxpbmUsCiAgICAgIG5hcnJhdGl2ZUhpc3Rvcnk6ZC5uYXJyYXRpdmVIaXN0b3J5fHxERUZBVUxULm5hcnJhdGl2ZUhpc3Rvcnksc2lnbmFsX2NvdW50OmQuc2lnbmFsX2NvdW50fHwwfTsKICAgIGlmKCFMSVZFW25hbWVdKUxJVkVbbmFtZV09e2F0dGVudGlvbjpkLmF0dGVudGlvbixkZWx0YTpkLmRlbHRhXzI0aCx2ZWxvY2l0eTpkLnZlbG9jaXR5LGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZX07CiAgICBMSVZFW25hbWVdLmVtb3Rpb25zPWVtb3M7TElWRVtuYW1lXS5kb21pbmFudF9lbW90aW9uPWRvbTsKICAgIHJldHVybiBTRFtuYW1lXTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbZmV0Y2hEZXRhaWxdJyxuYW1lLGUubWVzc2FnZSk7cmV0dXJuIFNEW25hbWVdfHxERUZBVUxUO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTbmFwKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc25hcHNob3QvZGFpbHknKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIC8vIHRvcGJhcgogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7CiAgICBpZihlbCYmZC50b3RhbF9zaWduYWxzKSBlbC50ZXh0Q29udGVudD1kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTsKICAgIHZhciBtZXRhPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtbWV0YScpOwogICAgaWYobWV0YSYmZC5hc19vZikgbWV0YS50ZXh0Q29udGVudD0nMzAgc3RhdGVzIMK3IHVwZGF0ZWQgJytuZXcgRGF0ZShkLmFzX29mKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyk7CiAgICAvLyBzdGF0cyBzdHJpcAogICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLCBkLnRvdGFsX3NpZ25hbHM/ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk6Jy0nKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogIH1jYXRjaChlKXt9Cn0KCmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKGVsKWVsLnRleHRDb250ZW50PXZhbDt9CgpmdW5jdGlvbiB1cGRhdGVTdHJpcE5hcnJhdGl2ZSgpe3VwZGF0ZUFsbFN0cmlwcygpO30KZnVuY3Rpb24gdXBkYXRlU3RyaXBBbmdlcigpe30KCmZ1bmN0aW9uIHNlbGVjdEhvdHRlc3QoKXsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKFNEKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApIHNlbGVjdF8odG9wWzBdKTsKfQphc3luYyBmdW5jdGlvbiBmZXRjaEluc2lnaHRzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvaW5zaWdodHMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQuZXJyb3IpIHJldHVybjsKICAgIHZhciBzaWc9ZC5zaWduYXR1cmU7CiAgICBpZihzaWcpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQudGFncyl0RWwuaW5uZXJIVE1MPWQudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgIH0KICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICBpZihyRWwmJmQucmlzaW5nJiZkLnJpc2luZy5sZW5ndGgpckVsLmlubmVySFRNTD1kLnJpc2luZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrbi5zdGF0ZXMuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMykrJyU7YmFja2dyb3VuZDojZTA1YTI4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICAgIGlmKGZFbCYmZC5mYWxsaW5nJiZkLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytuLnN0YXRlcy5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKSsnJTtiYWNrZ3JvdW5kOiMzYmI4ZDgiPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgICBpZihnRWwmJmQucmVnaW9uYWwmJmQucmVnaW9uYWwubGVuZ3RoKWdFbC5pbm5lckhUTUw9ZC5yZWdpb25hbC5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuIj48c3BhbiBjbGFzcz0ibmktbmFtZSI+JytyLnJlZ2lvbisnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrci5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj48ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrci5ob3R0ZXN0X3N0YXRlKycgwrcgJytyLnRvcF9uYXJyYXRpdmUrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZS5tZXNzYWdlKTt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoRnVsbFNuYXBzaG90KCl7CiAgLy8gTG9hZCBBTEwgc3RhdGUgZGF0YSBpbiBvbmUgcmVxdWVzdCBmb3IgaW5zdGFudCBmaXJzdC1sb2FkCiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvZnVsbC1zbmFwc2hvdCcpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC53YXJtaW5nX3VwfHwhZC5zdGF0ZXN8fCFkLnN0YXRlcy5sZW5ndGgpIHJldHVybiBmYWxzZTsKCiAgICAvLyBQb3B1bGF0ZSBTRCBhbmQgTElWRSBmcm9tIGZ1bGwgc25hcHNob3QKICAgIGQuc3RhdGVzLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAgIGlmKCFzLm5hbWUpIHJldHVybjsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKXx8cy5kb21pbmFudF9lbW90aW9ufHxudWxsOwogICAgICB2YXIgZW50cnk9T2JqZWN0LmFzc2lnbih7fSxzLHtlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRlbHRhOnMuZGVsdGFfMjRofHwwfSk7CiAgICAgIFNEW3MubmFtZV09ZW50cnk7CiAgICAgIExJVkVbcy5uYW1lXT17YXR0ZW50aW9uOnMuYXR0ZW50aW9uLGRlbHRhOnMuZGVsdGFfMjRofHwwLHZlbG9jaXR5OnMudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb20sZG9taW5hbnRfbmFycmF0aXZlOnMuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgfSk7CgogICAgLy8gVXBkYXRlIHNpZ25hbHMgY291bnQKICAgIGlmKGQuc25hcHNob3QmJmQuc25hcHNob3QudG90YWxfc2lnbmFscyl7CiAgICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJyxkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKSk7CiAgICB9CgogICAgLy8gVXBkYXRlIGluc2lnaHRzIGZyb20gY2FjaGVkIGRhdGEKICAgIGlmKGQuaW5zaWdodHMmJmQuaW5zaWdodHMuc2lnbmF0dXJlKXsKICAgICAgdmFyIHNpZz1kLmluc2lnaHRzLnNpZ25hdHVyZTsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLmluc2lnaHRzLnRhZ3MpdEVsLmlubmVySFRNTD1kLmluc2lnaHRzLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICAgIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgICAgIGlmKHJFbCYmZC5pbnNpZ2h0cy5yaXNpbmcmJmQuaW5zaWdodHMucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMucmlzaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgICBpZihmRWwmJmQuaW5zaWdodHMuZmFsbGluZyYmZC5pbnNpZ2h0cy5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuaW5zaWdodHMuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIH0KCiAgICAvLyBSZW5kZXIgbWFwIGNvbG9ycyBhbmQgc3RyaXBzCiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICAvLyBMb2FkIGluc2lnaHRzIHRvbwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgZGlzbWlzc0xvYWRlcigpOwogICAgLy8gVXNlIGNhY2hlZCBuYXJyYXRpdmUgaW5zaWdodCBpZiBhdmFpbGFibGUKICAgIGlmKGQubmFycmF0aXZlX2luc2lnaHQmJmQubmFycmF0aXZlX2luc2lnaHQudGV4dCl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICAgICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQrJzwvZGl2Pic7CiAgICAgIGlmKHRFbCYmZC5uYXJyYXRpdmVfaW5zaWdodC50b3BfbmFycmF0aXZlcyl7CiAgICAgICAgdEVsLmlubmVySFRNTD1kLm5hcnJhdGl2ZV9pbnNpZ2h0LnRvcF9uYXJyYXRpdmVzLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgICAgdmFyIGNvbD1pPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC42KSc7CiAgICAgICAgICB2YXIgYXJyPWk9PT0wPydcdTIxOTEgJzonXHUwMGI3ICc7CiAgICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPVwic2ktdGFnXCIgc3R5bGU9XCJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKTtjb2xvcjonK2NvbCsnXCI+JythcnIrbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgICB9KS5qb2luKCcnKTsKICAgICAgfQogICAgfQogICAgcmV0dXJuIHRydWU7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbZnVsbC1zbmFwc2hvdF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gZmFsc2U7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKXsKICB0cnl7CiAgICAvLyBUcnkgY2FjaGVkIHZlcnNpb24gZnJvbSBmdWxsLXNuYXBzaG90IGZpcnN0IChhbHJlYWR5IGxvYWRlZCkKICAgIC8vIFRoZW4gY2FsbCBkZWRpY2F0ZWQgZW5kcG9pbnQgZm9yIGZyZXNoIEFJIGFuYWx5c2lzCiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9uYXJyYXRpdmUtaW5zaWdodCcpOwogICAgaWYoIXIub2spIHJldHVybjsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoIWQudGV4dCkgcmV0dXJuOwoKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwoKICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC50ZXh0Kyc8L2Rpdj4nOwoKICAgIC8vIFRhZ3MgZnJvbSB0b3AgbmFycmF0aXZlcwogICAgaWYodEVsJiZkLnRvcF9uYXJyYXRpdmVzJiZkLnRvcF9uYXJyYXRpdmVzLmxlbmd0aCl7CiAgICAgIHRFbC5pbm5lckhUTUw9ZC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICB2YXIgY29sPWk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjYpJzsKICAgICAgICB2YXIgYXJyb3c9aT09PTA/J+KGkSAnOifCtyAnOwogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpO2NvbG9yOicrY29sKyciPicrYXJyb3crbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9CgogICAgaWYobWV0YUVsKXsKICAgICAgdmFyIHQ9bmV3IERhdGUoZC5hc19vZik7CiAgICAgIG1ldGFFbC50ZXh0Q29udGVudD0nU2lnbmFsIGFuYWx5c2lzIMK3ICcrdC50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pKyhkLmZhbGxiYWNrPycgwrcgcGF0dGVybi1iYXNlZCc6JyDCtyBBSSBzeW50aGVzaXplZCcpOwogICAgfQogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tuYXJyYXRpdmVdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hTdGF0ZUNvbnRleHQobm0pewogIC8vIEZldGNoIGNvbnRleHR1YWwgYnJpZWYg4oCUIGNvbWJpbmVzIEdvb2dsZSBOZXdzICsgc3RvcmVkIHNpZ25hbHMgKyBBSQogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlLWNvbnRleHQvJytlbmNvZGVVUklDb21wb25lbnQobm0pKTsKICAgIGlmKCFyLm9rKSByZXR1cm4gbnVsbDsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgcmV0dXJuIGQ7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbY29udGV4dF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gbnVsbDsKICB9Cn0KCmFzeW5jIGZ1bmN0aW9uIHN0YXJ0UG9sbGluZygpewogIGF3YWl0IFByb21pc2UuYWxsKFtmZXRjaEFsbFN0YXRlcygpLGZldGNoU25hcCgpXSk7CiAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKGUpe2NvbnNvbGUud2FybignW2luc2lnaHRzXScsZSk7fSk7CiAgdmFyIG49MDsKICB2YXIgdD1zZXRJbnRlcnZhbChhc3luYyBmdW5jdGlvbigpewogICAgbisrO2F3YWl0IGZldGNoQWxsU3RhdGVzKCk7YXdhaXQgZmV0Y2hTbmFwKCk7CiAgICBpZihTRUwpIHJlbmRlclBhbmVsKFNFTCk7CiAgICBpZihuPj0xMil7Y2xlYXJJbnRlcnZhbCh0KTtzZXRJbnRlcnZhbChhc3luYyBmdW5jdGlvbigpe2F3YWl0IGZldGNoQWxsU3RhdGVzKCk7YXdhaXQgZmV0Y2hTbmFwKCk7aWYoU0VMKXJlbmRlclBhbmVsKFNFTCk7fSwxMjAwMDApOwogICAgICBzZXRJbnRlcnZhbChmZXRjaEluc2lnaHRzLDM2MDAwMDApO30KICB9LDE1MDAwKTsKfQoKLy8gTkFSUkFUSVZFIERBVEEKdmFyIFNISUZUUz17CiAgJzNtJzpbCiAgICB7ZmFkaW5nOidJbmZsYXRpb24nLGZhZGluZ05vdGU6J2Vhc2luZyBuYXRpb25hbGx5JyxyaXNpbmc6J0JvcmRlciBzZWN1cml0eScscmlzaW5nTm90ZToncG9zdC1pbmNpZGVudCBzdXJnZSd9LAogICAge2ZhZGluZzonRWxlY3Rpb24gcmhldG9yaWMnLGZhZGluZ05vdGU6J3Bvc3QtY3ljbGUgZmFkZScscmlzaW5nOidHb3Zlcm5hbmNlIGFjY291bnRhYmlsaXR5JyxyaXNpbmdOb3RlOidzdGVhZHkgcmlzZSd9LAogICAge2ZhZGluZzonRmFybWVyIHByb3Rlc3RzJyxmYWRpbmdOb3RlOidtb21lbnR1bSBsb3N0JyxyaXNpbmc6J1VuZW1wbG95bWVudCBhbnhpZXR5JyxyaXNpbmdOb3RlOid5b3V0aCBzaWduYWwgc3VyZ2UnfSwKICBdLAogICc2bSc6WwogICAge2ZhZGluZzonQ2FzdGUgbW9iaWxpc2F0aW9uJyxmYWRpbmdOb3RlOidwcmUtZWxlY3Rpb24gcGVhaycscmlzaW5nOidDb3JydXB0aW9uIGFjY291bnRhYmlsaXR5JyxyaXNpbmdOb3RlOidwb3N0LWN5Y2xlIHB1c2gnfSwKICAgIHtmYWRpbmc6J1JlbGlnaW91cyBuYXRpb25hbGlzbScsZmFkaW5nTm90ZToncGxhdGVhdSBwaGFzZScscmlzaW5nOidFY29ub21pYyBhbnhpZXR5JyxyaXNpbmdOb3RlOidjb3N0LW9mLWxpdmluZyd9LAogICAge2ZhZGluZzonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLGZhZGluZ05vdGU6J3JpYmJvbi1jdXR0aW5nIGRvbmUnLHJpc2luZzonTGF3ICYgb3JkZXInLHJpc2luZ05vdGU6J2NyaW1lIG5hcnJhdGl2ZSByaXNlJ30sCiAgXSwKICAnMXknOlsKICAgIHtmYWRpbmc6J1BhbmRlbWljIHJlY292ZXJ5JyxmYWRpbmdOb3RlOidmYWRlZCBlYXJseSB5ZWFyJyxyaXNpbmc6J0luZmxhdGlvbicscmlzaW5nTm90ZTonZG9taW5hdGVkIG1pZC15ZWFyJ30sCiAgICB7ZmFkaW5nOidSZWdpb25hbCBpZGVudGl0eScsZmFkaW5nTm90ZTonbGFuZ3VhZ2UtbGVkIHBlYWsnLHJpc2luZzonU2VjdXJpdHkgJiBib3JkZXJzJyxyaXNpbmdOb3RlOidnZW9wb2xpdGljYWwgZXNjYWxhdGlvbid9LAogICAge2ZhZGluZzonR292ZXJuYW5jZSBvcHRpbWlzbScsZmFkaW5nTm90ZToncG9saWN5IGhvbmV5bW9vbiBlbmQnLHJpc2luZzonQ29ycnVwdGlvbiAmIHNjYW1zJyxyaXNpbmdOb3RlOidhY2NvdW50YWJpbGl0eSBjeWNsZSd9LAogIF0sCn07CnZhciBSRUdfU0hJRlRTPVsKICB7c3RhdGU6J1RhbWlsIE5hZHUnLGZyb206J1JlZ2lvbmFsIGlkZW50aXR5Jyx0bzonRmVkZXJhbCByZXNvdXJjZSBkaXNwdXRlcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0JpaGFyJyxmcm9tOidFbGVjdGlvbiByaGV0b3JpYycsdG86J1VuZW1wbG95bWVudCAmIGV4YW0gc2NhbXMnLHRpbWU6JzYgd2tzJ30sCiAge3N0YXRlOidXZXN0IEJlbmdhbCcsZnJvbTonQnlwb2xsIHBvbGl0aWNzJyx0bzonTGF3ICYgb3JkZXIgwrcgQm9yZGVyJyx0aW1lOic0IHdrcyd9LAogIHtzdGF0ZTonUmFqYXN0aGFuJyxmcm9tOidGYXJtZXIgcHJvdGVzdHMnLHRvOidIZWF0IHdhdmUgwrcgRW52aXJvbm1lbnQnLHRpbWU6JzIgd2tzJ30sCiAge3N0YXRlOidLYXJuYXRha2EnLGZyb206J01pbmluZyBjb250cm92ZXJzeScsdG86J0xhbmd1YWdlIHNpZ25hZ2UgcG9saXRpY3MnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidEZWxoaScsZnJvbTonTWV0cm8gaW5mcmFzdHJ1Y3R1cmUnLHRvOidBaXIgcXVhbGl0eSBjcmlzaXMnLHRpbWU6JzEwIGRheXMnfSwKICB7c3RhdGU6J01hbmlwdXInLGZyb206J0dvdmVybmFuY2UgJiBjYWJpbmV0Jyx0bzonRXRobmljIHRlbnNpb25zIMK3IEFGU1BBJyx0aW1lOic1IHdrcyd9LAogIHtzdGF0ZTonUHVuamFiJyxmcm9tOidQb3dlciBjcmlzaXMnLHRvOidCb3JkZXIgc2VjdXJpdHkgwrcgRHJvbmVzJyx0aW1lOiczIHdrcyd9LApdOwp2YXIgTU9DS19SPVsKICB7bmFtZTonQm9yZGVyIHNlY3VyaXR5JyxzdGF0ZXM6J0omSyDCtyBQdW5qYWIgwrcgUmFqYXN0aGFuJyxwY3Q6Jys0MSUnfSwKICB7bmFtZTonVW5lbXBsb3ltZW50JyxzdGF0ZXM6J0JpaGFyIMK3IFVQIMK3IEpoYXJraGFuZCcscGN0OicrMjglJ30sCiAge25hbWU6J0xhbmd1YWdlIHBvbGl0aWNzJyxzdGF0ZXM6J1ROIMK3IEthcm5hdGFrYSDCtyBNSCcscGN0OicrMjIlJ30sCiAge25hbWU6J0Vudmlyb25tZW50YWwgY3Jpc2lzJyxzdGF0ZXM6J0RlbGhpIMK3IFJhamFzdGhhbiDCtyBBUCcscGN0OicrMTklJ30sCiAge25hbWU6J0V0aG5pYyB0ZW5zaW9ucycsc3RhdGVzOidNYW5pcHVyIMK3IEFzc2FtIMK3IFdCJyxwY3Q6JysxNyUnfSwKXTsKdmFyIE1PQ0tfRj1bCiAge25hbWU6J0VsZWN0aW9uIHJoZXRvcmljJyxzdGF0ZXM6J05hdGlvbmFsIHBvc3QtY3ljbGUnLHBjdDonLTM4JSd9LAogIHtuYW1lOidJbmZsYXRpb24gcHJlc3N1cmUnLHN0YXRlczonRWFzaW5nIG5hdGlvbmFsbHknLHBjdDonLTI0JSd9LAogIHtuYW1lOidGYXJtZXIgcHJvdGVzdHMnLHN0YXRlczonTW9tZW50dW0gbG9zdCcscGN0OictMTklJ30sCiAge25hbWU6J0luZnJhc3RydWN0dXJlIHByaWRlJyxzdGF0ZXM6J1JpYmJvbi1jdXR0aW5nIGRvbmUnLHBjdDonLTE0JSd9LAogIHtuYW1lOidSZWxpZ2lvdXMgZmVzdGl2YWxzJyxzdGF0ZXM6J1Bvc3Qtc2Vhc29uIGZhZGUnLHBjdDonLTExJSd9LApdOwoKZnVuY3Rpb24gcmVuZGVyU3RyaXAocGVyaW9kKXsKICB2YXIgZGF0YT1TSElGVFNbcGVyaW9kXXx8U0hJRlRTWyczbSddOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2hpZnQtbGlzdCcpOwogIGlmKCFlbCkgcmV0dXJuOwogIGVsLmlubmVySFRNTD1kYXRhLm1hcChmdW5jdGlvbihzKXsKICAgIHJldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O292ZXJmbG93OmhpZGRlbjsiPicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo2cHggMTBweDtib3JkZXItcmlnaHQ6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWxsKTttYXJnaW4tYm90dG9tOjNweDsiPmZhZGluZzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuMjsiPicrcy5mYWRpbmcrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK3MuZmFkaW5nTm90ZSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJ3aWR0aDoyOHB4O2ZsZXgtc2hyaW5rOjA7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjQ1O2ZvbnQtc2l6ZToxM3B4OyI+4oaSPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjhweCAxMHB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tcmlzZSk7bWFyZ2luLWJvdHRvbTozcHg7Ij5yaXNpbmc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjI7Ij4nK3MucmlzaW5nKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytzLnJpc2luZ05vdGUrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odGFiKXsKICB0YWIuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0KXt0LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgdGFiLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlbmRlclN0cmlwKHRhYi5kYXRhc2V0LnBlcmlvZCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gcmVuZGVyTW9tZW50dW0oKXsKICAvLyBSZWFkIGZyb20gU0QgKHBvcHVsYXRlZCBieSBmZXRjaEFsbFN0YXRlcyBmcm9tIGxpdmUgQVBJKQogIHZhciBuYz17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIG5jW24ubmFtZV09KG5jW24ubmFtZV18fDApK24udmFsOwogICAgfSk7CiAgfSk7CiAgdmFyIHNvcnRlZD1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgdmFyIHJpc2luZz1zb3J0ZWQuc2xpY2UoMCw1KTsKICB2YXIgZmFsbGluZz1zb3J0ZWQuc2xpY2UoLTUpLnJldmVyc2UoKTsKICB2YXIgbXg9cmlzaW5nLmxlbmd0aD9yaXNpbmdbMF1bMV06MTAwOwoKICAvLyBXcml0ZSB0byByaXNpbmctbGlzdCAobWF0Y2hlcyBuYXItcm93IEhUTUwpCiAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICBpZihyRWwmJnJpc2luZy5sZW5ndGgpewogICAgckVsLmlubmVySFRNTD1yaXNpbmcubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIGRlY2xpbmluZy1saXN0CiAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICBpZihmRWwmJmZhbGxpbmcubGVuZ3RoKXsKICAgIGZFbC5pbm5lckhUTUw9ZmFsbGluZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIHJlZ2lvbmFsLWxpc3Qg4oCUIHRvcCBzdGF0ZSBwZXIgcmVnaW9uIGZyb20gTElWRQogIHZhciByZWdpb25zPXsKICAgICdOb3J0aCc6WydEZWxoaScsJ1V0dGFyIFByYWRlc2gnLCdQdW5qYWInLCdIYXJ5YW5hJywnSGltYWNoYWwgUHJhZGVzaCcsJ1V0dGFyYWtoYW5kJywnSmFtbXUgYW5kIEthc2htaXInXSwKICAgICdFYXN0JzpbJ1dlc3QgQmVuZ2FsJywnQmloYXInLCdKaGFya2hhbmQnLCdPZGlzaGEnXSwKICAgICdXZXN0JzpbJ01haGFyYXNodHJhJywnR3VqYXJhdCcsJ1JhamFzdGhhbicsJ0dvYSddLAogICAgJ1NvdXRoJzpbJ1RhbWlsIE5hZHUnLCdLYXJuYXRha2EnLCdLZXJhbGEnLCdBbmRocmEgUHJhZGVzaCcsJ1RlbGFuZ2FuYSddLAogICAgJ05FJzpbJ0Fzc2FtJywnTWFuaXB1cicsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ1RyaXB1cmEnLCdBcnVuYWNoYWwgUHJhZGVzaCcsJ1Npa2tpbSddLAogICAgJ0NlbnRyYWwnOlsnTWFkaHlhIFByYWRlc2gnLCdDaGhhdHRpc2dhcmgnXSwKICB9OwogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICBpZihnRWwpewogICAgdmFyIHJlZ0l0ZW1zPU9iamVjdC5lbnRyaWVzKHJlZ2lvbnMpLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciByZWdpb249a3ZbMF0sc3RhdGVzPWt2WzFdOwogICAgICB2YXIgdG9wPXN0YXRlcy5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuIHtuYW1lOnMsYXR0OihMSVZFW3NdJiZMSVZFW3NdLmF0dGVudGlvbil8fDB9O30pCiAgICAgICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSlbMF07CiAgICAgIGlmKCF0b3B8fCF0b3AuYXR0KSByZXR1cm4gbnVsbDsKICAgICAgdmFyIG5hcj0oTElWRVt0b3AubmFtZV0mJkxJVkVbdG9wLm5hbWVdLmRvbWluYW50X25hcnJhdGl2ZSl8fCfigJQnOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6OHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCkiPicrcmVnaW9uKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+Jyt0b3AuYXR0LnRvRml4ZWQoMSkrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrdG9wLm5hbWUrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4OyI+JytuYXIrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuZmlsdGVyKEJvb2xlYW4pLmpvaW4oJycpOwogICAgaWYocmVnSXRlbXMpIGdFbC5pbm5lckhUTUw9cmVnSXRlbXM7CiAgfQp9CgoKLy8gU1RBVEUgREFUQQp2YXIgU0Q9e307Cgp2YXIgTElWRT17fTsKZnVuY3Rpb24gbm9ybWFsaXplRW1vdGlvbnMoZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJue307dmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhlKSx0b3Q9dmFscy5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcyt2O30sMCk7aWYodG90PD0wKXJldHVybnt9O2lmKHRvdDw9MS4wMSl7dmFyIG91dD17fTtPYmplY3Qua2V5cyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGspe291dFtrXT1NYXRoLnJvdW5kKGVba10qMTAwKTt9KTtyZXR1cm4gb3V0O31yZXR1cm4gZTt9CmZ1bmN0aW9uIGRvbWluYW50RW1vdGlvbihlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm4gbnVsbDt2YXIgbXg9MCxkb209bnVsbDtPYmplY3QuZW50cmllcyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXT5teCl7bXg9a3ZbMV07ZG9tPWt2WzBdO319KTtyZXR1cm4gZG9tO30KZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIWVsKXJldHVybjtlbC50ZXh0Q29udGVudD12YWw7aWYodmFsJiZ2YWwhPT0nLScpe2VsLmNsYXNzTGlzdC5yZW1vdmUoJ2xvYWRpbmcnKTt9fQoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjowLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7fSxkb21pbmFudF9lbW90aW9uOm51bGwsZG9taW5hbnRfbmFycmF0aXZlOm51bGwsCiAgbmFycmF0aXZlczpbXSxyaXNpbmc6W10sZmFsbGluZzpbXSwKICBzdW1tYXJ5OicnLGFydGljbGVzOltdLHRpbWVsaW5lOltdLAogIG5hcnJhdGl2ZUhpc3Rvcnk6W10sc2lnbmFsX2NvdW50OjAsCn07CgpmdW5jdGlvbiBnKG4pe3JldHVybiBTRFtuXXx8T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTt9CgpmdW5jdGlvbiBhQyhzKXsKICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSkuZmlsdGVyKGZ1bmN0aW9uKHYpe3JldHVybiB2PjA7fSk7CiAgaWYoIXNjb3Jlcy5sZW5ndGgpIHJldHVybiAnIzBkMWUzMCc7CiAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKTsKICB2YXIgbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwocy1tbikvKG14LW1uKSkpOwoKICAvLyAxMCB2aXN1YWxseSBkaXN0aW5jdCBzdG9wcyDigJQgY29sZCB0byBob3QKICAvLyBEZWVwIG5hdnkg4oaSIHN0ZWVsIGJsdWUg4oaSIHRlYWwg4oaSIHNhZ2Ug4oaSIGdvbGQg4oaSIGFtYmVyIOKGkiBidXJudCDihpIgY3JpbXNvbiDihpIgcmVkIOKGkiBicmlnaHQgcmVkCiAgdmFyIHN0b3BzPVsKICAgIFswLjAwLCcjMGExNjI4J10sICAvLyBkZWVwIG5hdnkgKHNpbGVudCkKICAgIFswLjEwLCcjMGUyZDUyJ10sICAvLyBuYXZ5CiAgICBbMC4yMCwnIzBhNGE3YSddLCAgLy8gc3RlZWwgYmx1ZQogICAgWzAuMzAsJyMwZDcwOTAnXSwgIC8vIHRlYWwKICAgIFswLjQyLCcjMGU5MDgwJ10sICAvLyBzZWEgZ3JlZW4KICAgIFswLjU0LCcjMmE4YTRhJ10sICAvLyBzYWdlIGdyZWVuCiAgICBbMC42NCwnI2M4OTYwYSddLCAgLy8gZ29sZAogICAgWzAuNzQsJyNkODYwMjAnXSwgIC8vIGFtYmVyCiAgICBbMC44NCwnI2NjMjgwOCddLCAgLy8gY3JpbXNvbgogICAgWzAuOTMsJyNlODAwMTAnXSwgIC8vIHJlZAogICAgWzEuMDAsJyNmZjEwMjAnXSwgIC8vIGJyaWdodCByZWQgKG1heGltdW0pCiAgXTsKCiAgLy8gRmluZCBzdXJyb3VuZGluZyBzdG9wcyBhbmQgbGVycAogIGZvcih2YXIgaT0wO2k8c3RvcHMubGVuZ3RoLTE7aSsrKXsKICAgIHZhciBzMD1zdG9wc1tpXSxzMT1zdG9wc1tpKzFdOwogICAgaWYobj49czBbMF0mJm48PXMxWzBdKXsKICAgICAgdmFyIHQ9KG4tczBbMF0pLyhzMVswXS1zMFswXSk7CiAgICAgIC8vIFBhcnNlIGhleCBhbmQgbGVycAogICAgICB2YXIgYzA9aGV4VG9SZ2IoczBbMV0pLGMxPWhleFRvUmdiKHMxWzFdKTsKICAgICAgdmFyIHI9TWF0aC5yb3VuZChjMFswXSsoYzFbMF0tYzBbMF0pKnQpOwogICAgICB2YXIgZz1NYXRoLnJvdW5kKGMwWzFdKyhjMVsxXS1jMFsxXSkqdCk7CiAgICAgIHZhciBiPU1hdGgucm91bmQoYzBbMl0rKGMxWzJdLWMwWzJdKSp0KTsKICAgICAgcmV0dXJuICdyZ2IoJytyKycsJytnKycsJytiKycpJzsKICAgIH0KICB9CiAgcmV0dXJuIHN0b3BzW3N0b3BzLmxlbmd0aC0xXVsxXTsKfQoKZnVuY3Rpb24gaGV4VG9SZ2IoaGV4KXsKICB2YXIgcj1wYXJzZUludChoZXguc2xpY2UoMSwzKSwxNik7CiAgdmFyIGc9cGFyc2VJbnQoaGV4LnNsaWNlKDMsNSksMTYpOwogIHZhciBiPXBhcnNlSW50KGhleC5zbGljZSg1LDcpLDE2KTsKICByZXR1cm4gW3IsZyxiXTsKfQpmdW5jdGlvbiBlQyhlKXsKICB2YXIgbXg9MCxkb209J3ByaWRlJzsKICBmb3IodmFyIGsgaW4gZSl7aWYoZVtrXT5teCl7bXg9ZVtrXTtkb209azt9fQogIHJldHVybiAoe2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9KVtkb21dfHwnIzMzYWFjYyc7Cn0KZnVuY3Rpb24gdkModil7CiAgLy8gTW9tZW50dW06IGNvb2xpbmcgKGJsdWUpIOKGkCBzdGFibGUgKHNsYXRlKSDihpIgcmlzaW5nICh3YXJtKSDihpIgc3VyZ2luZyAocmVkKQogIC8vIFVzZSBzbW9vdGggaW50ZXJwb2xhdGlvbiBmb3IgYmV0dGVyIHZpc3VhbCBkaXN0aW5jdGlvbgogIGlmKHY+MC4zKSAgcmV0dXJuICcjZTgxMDEwJzsgIC8vIHN1cmdpbmcgZmFzdCDigJQgYnJpZ2h0IHJlZAogIGlmKHY+MC4xNSkgcmV0dXJuICcjZDg0MDEwJzsgIC8vIHJpc2luZyBmYXN0ICDigJQgb3JhbmdlIHJlZAogIGlmKHY+MC4wNykgcmV0dXJuICcjZTA3ODIwJzsgIC8vIHJpc2luZyAgICAgICDigJQgYW1iZXIgb3JhbmdlCiAgaWYodj4wLjAyKSByZXR1cm4gJyNjOGEwMjAnOyAgLy8gc2xpZ2h0IHJpc2UgIOKAlCBnb2xkCiAgaWYodj4tMC4wMikgcmV0dXJuICcjMzM0NDU1JzsgLy8gc3RhYmxlICAgICAgIOKAlCBzbGF0ZQogIGlmKHY+LTAuMDcpIHJldHVybiAnIzFhNzA5MCc7IC8vIHNsaWdodCBjb29sICDigJQgdGVhbAogIGlmKHY+LTAuMTUpIHJldHVybiAnIzEwNTBhMCc7IC8vIGNvb2xpbmcgICAgICDigJQgYmx1ZQogIHJldHVybiAnIzBhMjg2OCc7ICAgICAgICAgICAgIC8vIGNvb2xpbmcgZmFzdCDigJQgZGVlcCBibHVlCn0KCnZhciBsYXllcj0nYXR0ZW50aW9uJyxTRUw9bnVsbCxGQVZTPW5ldyBTZXQoKTsKCi8vIE1BUApmdW5jdGlvbiBwcm9qXyh3LGgscGFkKXsKICBwYWQ9cGFkfHwyMDsKICB2YXIgbWluTG9uPTY4LjEsbWF4TG9uPTk3LjQsbWluTGF0PTYuNSxtYXhMYXQ9MzcuMTsKICB2YXIgc2NYPSh3LXBhZCoyKS8obWF4TG9uLW1pbkxvbik7CiAgdmFyIHNjWT0oaC1wYWQqMikvKG1heExhdC1taW5MYXQpOwogIHZhciBzYz1NYXRoLm1pbihzY1gsc2NZKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLShtYXhMb24tbWluTG9uKSpzYykvMjsKICB2YXIgb3k9cGFkKyhoLXBhZCoyLShtYXhMYXQtbWluTGF0KSpzYykvMjsKICByZXR1cm4gZnVuY3Rpb24obG9uLGxhdCl7cmV0dXJuIFtveCsobG9uLW1pbkxvbikqc2MsIG95KyhtYXhMYXQtbGF0KSpzY107fTsKfQpmdW5jdGlvbiBnZW8ycGF0aChnZW9tLHBqKXsKICB2YXIgZD0nJzsKICBmdW5jdGlvbiByaW5nKGNzKXt2YXIgcz0nJztjcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cGooY1swXSxjWzFdKTtzKz0oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSk7cmV0dXJuIHMrJ1onO30KICBpZihnZW9tLnR5cGU9PT0nUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pOwogIGVsc2UgaWYoZ2VvbS50eXBlPT09J011bHRpUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihwKXtwLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTt9KTsKICByZXR1cm4gZDsKfQpmdW5jdGlvbiBjdHIoZ2VvbSl7CiAgdmFyIHB0cz1bXTsKICBmdW5jdGlvbiBjb2woYyl7aWYodHlwZW9mIGNbMF09PT0nbnVtYmVyJykgcHRzLnB1c2goYyk7ZWxzZSBjLmZvckVhY2goY29sKTt9CiAgY29sKGdlb20uY29vcmRpbmF0ZXMpOwogIGlmKCFwdHMubGVuZ3RoKSByZXR1cm4gWzAsMF07CiAgcmV0dXJuIFtwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFswXTt9LDApL3B0cy5sZW5ndGgscHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMV07fSwwKS9wdHMubGVuZ3RoXTsKfQpmdW5jdGlvbiBzTmFtZShwcm9wcyl7CiAgdmFyIHJhdz1wcm9wcy5zdF9ubXx8cHJvcHMuTkFNRV8xfHxwcm9wcy5uYW1lfHxwcm9wcy5OQU1FfHwnJzsKICB2YXIgbWFwPXsnTGFkYWtoJzonSmFtbXUgYW5kIEthc2htaXInLCdKYW1tdSAmIEthc2htaXInOidKYW1tdSBhbmQgS2FzaG1pcicsJ1V0dGFyYW5jaGFsJzonVXR0YXJha2hhbmQnLCdBbmRhbWFuIGFuZCBOaWNvYmFyJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnQW5kYW1hbiAmIE5pY29iYXIgSXNsYW5kJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnTkNUIG9mIERlbGhpJzonRGVsaGknLCdQb25kaWNoZXJyeSc6J1B1ZHVjaGVycnknLCdEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpJzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsJ0RhbWFuIGFuZCBEaXUnOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1J307CiAgcmV0dXJuIG1hcFtyYXddfHxyYXc7Cn0KCnZhciBjYWNoZWRHZW89bnVsbDsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoYXR0ZW1wdCl7CiAgYXR0ZW1wdCA9IGF0dGVtcHR8fDE7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goJ2h0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9naC91ZGl0LTAwMS9pbmRpYS1tYXBzLWRhdGFAbWFzdGVyL3RvcG9qc29uL2luZGlhLmpzb24nKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwxMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwzMDAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciw2MDAwKTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1ttYXBdIGxvYWQgZmFpbGVkIGF0dGVtcHQgJythdHRlbXB0Kyc6JyxlLm1lc3NhZ2UpOwogICAgaWYoYXR0ZW1wdDw1KXsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe2xvYWRNYXAoYXR0ZW1wdCsxKTt9LCBhdHRlbXB0KjIwMDApOwogICAgfSBlbHNlIHsKICAgICAgdmFyIG1pPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKTsKICAgICAgaWYobWkpIG1pLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6IzJhM2E0YTtwYWRkaW5nOjQwcHg7dGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6bW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4Ij5NYXAgdW5hdmFpbGFibGUg4oCUIHJlZnJlc2ggdG8gcmV0cnk8L2Rpdj4nOwogICAgfQogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKLy8gU2luZ2xlIHNvdXJjZSBvZiB0cnV0aCBmb3IgZW1vdGlvbiBjb2xvcgovLyBCb3RoIG1hcCBhbmQgcGFuZWwgY2FsbCB0aGlzIOKAlCBndWFyYW50ZWVzIHRoZXkgYWx3YXlzIG1hdGNoCmZ1bmN0aW9uIGdldEVmZmVjdGl2ZUVtb3Rpb24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgLy8gMS4gVHJ5IExJVkUuZG9taW5hbnRfZW1vdGlvbiAoc2V0IGJ5IC9hcGkvc3RhdGVzKQogIHZhciBkb209bGl2ZS5kb21pbmFudF9lbW90aW9ufHxkLmRvbWluYW50X2Vtb3Rpb247CgogIC8vIDIuIFRyeSBjb21wdXRpbmcgZnJvbSBlbW90aW9ucyBicmVha2Rvd24KICBpZighZG9tKXsKICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgICBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpOwogIH0KCiAgLy8gMy4gRmFsbGJhY2s6IGluZmVyIGZyb20gZG9taW5hbnQgbmFycmF0aXZlIChzYW1lIGxvZ2ljIGV2ZXJ5d2hlcmUpCiAgaWYoIWRvbSl7CiAgICB2YXIgbnA9KGxpdmUuZG9taW5hbnRfbmFycmF0aXZlfHxkLmRvbWluYW50X25hcnJhdGl2ZXx8JycpLnRvTG93ZXJDYXNlKCk7CiAgICBpZihucC5tYXRjaCgvYm9yZGVyfHRlcnJvcnxzZWN1cml0eXxjb25mbGljdHxhdHRhY2t8d2FyfGluZmlsdHJhdC8pKSBkb209J2ZlYXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvc2NhbXxjb3JydXB0fHByb3Rlc3R8YXJyZXN0fHZpb2xlbmNlfG91dHJhZ2V8Y3JpbWUvKSkgZG9tPSdhbmdlcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9kZXZlbG9wfGludmVzdHxncm93dGh8bGF1bmNofGluYXVndXJ8cmVmb3JtfHByb2dyZXNzfGJvb3N0LykpIGRvbT0naG9wZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9jdWx0dXJlfGhlcml0YWdlfHByaWRlfHZpY3Rvcnl8Y2VsZWJyYXR8bWVkYWx8YWNoaWV2ZW1lbnQvKSkgZG9tPSdwcmlkZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9mbG9vZHxkcm91Z2h0fHVuZW1wbG95bWVudHxpbmZsYXRpb258c2hvcnRhZ2V8Y3Jpc2lzfGNvbmNlcm4vKSkgZG9tPSdhbnhpZXR5JzsKICAgIGVsc2UgaWYoKGxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MCk+NSkgZG9tPSdhbnhpZXR5JzsgLy8gYWN0aXZlIHN0YXRlIGRlZmF1bHQKICAgIGVsc2UgZG9tPSdhbnhpZXR5JzsgLy8gZ2xvYmFsIGRlZmF1bHQKICB9CgogIHJldHVybiBkb207Cn0KCi8vIEdldCBlc3RpbWF0ZWQgZW1vdGlvbiBicmVha2Rvd24gKGZvciBwYW5lbCBkb251dCB3aGVuIHJlYWwgZGF0YSBtaXNzaW5nKQpmdW5jdGlvbiBnZXRFbW90aW9uQnJlYWtkb3duKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogIGlmKE9iamVjdC5rZXlzKGVtb3MpLmxlbmd0aCkgcmV0dXJuIHtlbW90aW9uczplbW9zLGVzdGltYXRlZDpmYWxzZX07CiAgLy8gQnVpbGQgc2tld2VkIGRpc3RyaWJ1dGlvbiBmcm9tIGVmZmVjdGl2ZSBlbW90aW9uCiAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICB2YXIgYmFzZT17YW54aWV0eToxMyxhbmdlcjoxMyxob3BlOjEzLHByaWRlOjEzLGZlYXI6MTN9OwogIGJhc2VbZG9tXT00ODsKICByZXR1cm4ge2Vtb3Rpb25zOmJhc2UsZXN0aW1hdGVkOnRydWV9Owp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyksZD1nKG5tKSxmaWxsOwogICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJykgZmlsbD1hQyhkLmF0dGVudGlvbik7CiAgICBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgZGU9Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgIGZpbGw9ZU1hcFtkZV18fCcjMzM0NDU1JzsKICAgIH0KICAgIGVsc2UgZmlsbD12QyhkLnZlbG9jaXR5KTsKICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsJyxmaWxsKTsKICAgIChmdW5jdGlvbigpewogICAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4geC5hdHRlbnRpb258fDA7fSk7CiAgICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogICAgICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKGQuYXR0ZW50aW9uLW1uKS8obXgtbW4pKSk7CiAgICAgIHAuc2V0QXR0cmlidXRlKCdmaWxsLW9wYWNpdHknLGxheWVyPT09J2F0dGVudGlvbic/TWF0aC5tYXgoMC4zLDAuMytuKjAuNyk6MC44NSk7CiAgICB9KSgpOwogIH0pOwp9CgpmdW5jdGlvbiBhdHRhY2hJbnRlcmFjdGlvbnMoKXsKICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXsKICAgICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKTsKICAgICAgdmFyIGQ9ZyhubSk7CiAgICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgICAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogICAgICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGxhdGVzdD0nJzsKICAgICAgaWYoZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXMubGVuZ3RoKSBsYXRlc3Q9ZC5uYXJyYXRpdmVzWzBdLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZC5uYXJyYXRpdmVzWzBdLm5hbWUuc2xpY2UoMSk7CiAgICAgIGVsc2UgaWYobGl2ZS5kb21pbmFudF9uYXJyYXRpdmUpIGxhdGVzdD1saXZlLmRvbWluYW50X25hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStsaXZlLmRvbWluYW50X25hcnJhdGl2ZS5zbGljZSgxKTsKCiAgICAgIHZhciByb3dzPScnOwogICAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgICAgICB2YXIgYXR0PWxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MDsKICAgICAgICB2YXIgZGx0PWxpdmUuZGVsdGF8fGQuZGVsdGF8fDA7CiAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+QXR0ZW50aW9uPC9zcGFuPjxzdHJvbmc+JythdHQudG9GaXhlZCgxKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgIChkbHQhPT0wPyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj4yNGggc2hpZnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JysoZGx0PjA/JyNlMDVhMjgnOicjM2JiOGQ4JykrJyI+JysoZGx0PjA/JysnOicnKStkbHQrJzwvc3Ryb25nPjwvZGl2Pic6JycpKwogICAgICAgICAgKGxhdGVzdD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+VG9wIG5hcnJhdGl2ZTwvc3Bhbj48c3Ryb25nPicrbGF0ZXN0Kyc8L3N0cm9uZz48L2Rpdj4nOicnKTsKICAgICAgfSBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgICB2YXIgZG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICAgIGlmKGRvbUVtbyl7CiAgICAgICAgICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczpkLmVtb3Rpb25zfHx7fTsKICAgICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRvbWluYW50PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrcGFsW2RvbUVtb10rJyI+Jytkb21FbW8uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vLnNsaWNlKDEpKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICAgICAgdmFyIGVMPU9iamVjdC5lbnRyaWVzKGVtb3MpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgICAgICAgdmFyIHRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgICAgICAgIGlmKHRvdD4wJiZ0b3Q8PTEuMDEpe2VMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7dG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApO30KICAgICAgICAgIHJvd3MrPWVMLnNsaWNlKDAsMykubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm4gJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo0cHgiPjxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrcGFsW2t2WzBdXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JytrdlswXSsnPC9zcGFuPjxzdHJvbmc+JytNYXRoLnJvdW5kKGt2WzFdKjEwMC9NYXRoLm1heCgxLHRvdCkpKyclPC9zdHJvbmc+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgdmFyIHZlbD1saXZlLnZlbG9jaXR5fHxkLnZlbG9jaXR5fHwwOwogICAgICAgIHZhciB2ZWxEaXI9dmVsPjAuMT8nUmlzaW5nIGZhc3QnOnZlbD4wLjAyPydSaXNpbmcnOnZlbDwtMC4wNT8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICAgICAgdmFyIHZlbENvbD12ZWw+MC4wMj8nI2UwNWEyOCc6dmVsPC0wLjAyPycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+TW9tZW50dW08L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9zdHJvbmc+PC9kaXY+JysKICAgICAgICAgICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5EaXJlY3Rpb248L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6Jyt2ZWxDb2wrJyI+Jyt2ZWxEaXIrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgIH0KCiAgICAgIHRpcC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InR0LW4iPicrbm0rJzwvZGl2Picrcm93cysobGF0ZXN0JiZsYXllciE9PSdhdHRlbnRpb24nPyc8ZGl2IGNsYXNzPSJ0dC1uYXIiPjxzdHJvbmc+TmFycmF0aXZlPC9zdHJvbmc+JytsYXRlc3QrJzwvZGl2Pic6JycpOwogICAgICB2YXIgcmVjdD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJykuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICAgIHRpcC5zdHlsZS5sZWZ0PU1hdGgubWluKGUuY2xpZW50WC1yZWN0LmxlZnQrMTQscmVjdC53aWR0aC0xOTApKydweCc7CiAgICAgIHRpcC5zdHlsZS50b3A9TWF0aC5taW4oZS5jbGllbnRZLXJlY3QudG9wKzE0LHJlY3QuaGVpZ2h0LTE1MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLm9wYWNpdHk9JzEnOwogICAgfSk7CnAuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VsZWF2ZScsZnVuY3Rpb24oKXt0aXAuc3R5bGUub3BhY2l0eT0wO30pOwogICAgcC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXtzZWxlY3RfKHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSk7fSk7CiAgfSk7Cn0KCi8vIFNUQVRFIFBBTkVMCmZ1bmN0aW9uIHNlbGVjdF8obm0pewogIFNFTD1ubTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5jbGFzc0xpc3QudG9nZ2xlKCdzZWxlY3RlZCcscC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpPT09bm0pOwogIH0pOwogIC8vIFNob3cgbG9hZGluZyBzdGF0ZSBpbW1lZGlhdGVseSB3aXRoIHdoYXRldmVyIExJVkUgZGF0YSB3ZSBoYXZlCiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZihwYW5lbCl7CiAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICBwYW5lbC5pbm5lckhUTUw9CiAgICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAgICc8ZGl2PjxkaXYgY2xhc3M9InNwLWVrIj4nKyhsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICAgJzwvYnV0dG9uPicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA4ZW0iPicrCiAgICAgICAgJ0xvYWRpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJysKICAgICAgICAobGl2ZS5hdHRlbnRpb24/Jzxicj48YnI+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtjb2xvcjp2YXIoLS1pbmspIj5BdHRlbnRpb24gJytsaXZlLmF0dGVudGlvbi50b0ZpeGVkKDEpKyc8L3NwYW4+JzonJykrCiAgICAgICAgKGxpdmUuZG9taW5hbnRfZW1vdGlvbj8nPGJyPjxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrbGl2ZS5kb21pbmFudF9lbW90aW9uKycgc2lnbmFsIGRvbWluYW50PC9zcGFuPic6JycpKwogICAgICAnPC9kaXY+JzsKICB9CiAgLy8gRmV0Y2ggZnVsbCBkZXRhaWwgdGhlbiByZW5kZXIKICAvLyBGZXRjaCBkZXRhaWwgYW5kIGNvbnRleHQgaW4gcGFyYWxsZWwKICBQcm9taXNlLmFsbChbCiAgICBmZXRjaERldGFpbChubSksCiAgICBsYXllcj09PSdhdHRlbnRpb24nP2ZldGNoU3RhdGVDb250ZXh0KG5tKTpQcm9taXNlLnJlc29sdmUobnVsbCkKICBdKS50aGVuKGZ1bmN0aW9uKHJlc3VsdHMpewogICAgaWYoU0VMIT09bm0pIHJldHVybjsKICAgIHZhciBjdHg9cmVzdWx0c1sxXTsKICAgIHJlbmRlclBhbmVsKG5tLCBjdHgpOwogICAgLy8gVXBkYXRlIG1hcCBjb2xvcgogICAgdmFyIHBhdGg9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJytubSsnIl0nKTsKICAgIGlmKHBhdGgmJmxheWVyPT09J2Vtb3Rpb24nKXsKICAgICAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgZG9tPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICBpZihlTWFwW2RvbV0pIHBhdGguc2V0QXR0cmlidXRlKCdmaWxsJyxlTWFwW2RvbV0pOwogICAgfSBlbHNlIHsKICAgICAgYXBwbHlMYXllcigpOwogICAgfQogIH0pLmNhdGNoKGZ1bmN0aW9uKGUpewogICAgY29uc29sZS53YXJuKCdbc2VsZWN0XScsZSk7CiAgICBpZihTRUw9PT1ubSkgcmVuZGVyUGFuZWwobm0sIG51bGwpOwogIH0pOwp9CgpmdW5jdGlvbiByZW5kZXJQYW5lbChubSwgY3R4KXsKICB2YXIgZD1nKG5tKTsKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKCFwYW5lbCkgcmV0dXJuOwogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICB2YXIgaGVhZGVyPQogICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICc8ZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLWVrIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+JysKICAgICAgICAgIChsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKwogICAgICAgICAgKGQuY29uZmlkZW5jZT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/J3JnYmEoNTEsMjA0LDEwMiwwLjEpJzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8ncmdiYSgyMjQsOTAsNDAsMC4xKSc6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknKSsnO2NvbG9yOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8nIzMzY2M2Nic6ZC5jb25maWRlbmNlPT09J01FRElVTSc/JyNlMDVhMjgnOidyZ2JhKDI1NSwyNTUsMjU1LDAuMyknKSsnIj4nK2QuY29uZmlkZW5jZSsnIFNJR05BTDwvc3Bhbj4nOicnKSsKICAgICAgICAgIChkLmlzX3JlZ2lvbmFsX3N0b3J5Pyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpO2NvbG9yOiMzYmI4ZDgiPlJFR0lPTkFMIFNQSUtFPC9zcGFuPic6JycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICc8L2J1dHRvbj4nKwogICAgJzwvZGl2Pic7CgogIHZhciBib2R5PScnOwoKICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgIHZhciBkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPyd1cCc6J2RuJzsKICAgIHZhciBuYXJyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdGw9KGQudGltZWxpbmUmJmQudGltZWxpbmUubGVuZ3RoKT9kLnRpbWVsaW5lOlswLDAsMCwwLDAsMCwwLGQuYXR0ZW50aW9ufHwwXTsKICAgIHZhciB0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgICB2YXIgdHc9MjYwLHRoPTYyLHRwPTU7CiAgICB2YXIgcHRzPXRsLm1hcChmdW5jdGlvbih2LGkpe3JldHVyblt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICAgIHZhciBwRD1wdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuKGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pLmpvaW4oJycpOwogICAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICAgIHZhciBhYz1hQyhkLmF0dGVudGlvbnx8MCk7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ0hvdyBpbnRlbnNlbHkgJysobm0uc3BsaXQoIiAiKVswXSkrJyBpcyBiZWluZyBkaXNjdXNzZWQgbmF0aW9uYWxseS4gU2NvcmUgb2YgJytkLmF0dGVudGlvbisnIG1lYW5zICcrKGQuYXR0ZW50aW9uPjYwPyd2ZXJ5IGhpZ2gg4oCUIGRvbWluYXRlcyBuYXRpb25hbCBkaXNjb3Vyc2UnOmQuYXR0ZW50aW9uPjM1PydlbGV2YXRlZCDigJQgY2xlYXJseSBpbiB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uJzpkLmF0dGVudGlvbj4xNT8nbW9kZXJhdGUg4oCUIHNvbWUgbmF0aW9uYWwgY292ZXJhZ2UnOmQuYXR0ZW50aW9uPjU/J2xvdyDigJQgbGltaXRlZCBzaWduYWxzJzonbWluaW1hbCDigJQgZmV3IHNpZ25hbHMgZGV0ZWN0ZWQnKSsnLicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJpbnNpZ2h0IiBzdHlsZT0iJysoY3R4PycnOidib3JkZXItY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjA2KScpKyciPicrCiAgICAgIChjdHgmJmN0eC5icmllZgogICAgICAgID8gY3R4LmJyaWVmKyhjdHguc291cmNlPT09ImFpIj8nJzonJykKICAgICAgICA6IChkLmNvbmZpZGVuY2U9PT0iTE9XIiYmIWQuc3VtbWFyeQogICAgICAgICAgICA/ICdMaW1pdGVkIHNpZ25hbHMgZnJvbSAnK25tKycuIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy4nCiAgICAgICAgICAgIDogZC5zdW1tYXJ5fHwnQ29sbGVjdGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKSkrCiAgICAnPC9kaXY+JysKICAgICcnKwogICAgICAnPGRpdiBjbGFzcz0ic2NvcmUtc3RyaXAiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiPicrKGQuYXR0ZW50aW9ufHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCBzaGlmdDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrZEMrJyI+JytkUysoZC5kZWx0YXx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5Ub3AgbmFycmF0aXZlPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhuYXJyWzBdP25hcnJbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuYXJyWzBdLm5hbWUuc2xpY2UoMSk6J+KAlCcpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+TmFycmF0aXZlIGJyZWFrZG93bjwvZGl2PicrCiAgICAgICAgKG5hcnIubGVuZ3RoPwogICAgICAgICAgJzxkaXYgY2xhc3M9Im5hci1saXN0Ij4nK25hcnIubWFwKGZ1bmN0aW9uKG4pewogICAgICAgICAgICB2YXIgbm49bi5uYW1lP24ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hbWUuc2xpY2UoMSk6bi5uYW1lOwogICAgICAgICAgICB2YXIgdmFsPXR5cGVvZiBuLnZhbD09PSdudW1iZXInP24udmFsOjA7CiAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0yIj48ZGl2IGNsYXNzPSJuaS1sYWJlbCI+Jytubisobi5kaXI9PT0ndXAnPycgPHNwYW4gc3R5bGU9ImNvbG9yOiNlMDVhMjg7Zm9udC1zaXplOjlweCI+4oaRPC9zcGFuPic6bi5kaXI9PT0nZG93bic/JyA8c3BhbiBzdHlsZT0iY29sb3I6IzNiYjhkODtmb250LXNpemU6OXB4Ij7ihpM8L3NwYW4+JzonJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXZhbCI+Jyt2YWwudG9GaXhlZCgxKSsnJTwvZGl2PicrCiAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsdmFsKjIuNSkrJyU7YmFja2dyb3VuZDonKyhuLmRpcj09PSd1cCc/JyNlMDVhMjgnOm4uZGlyPT09J2Rvd24nPycjM2JiOGQ4JzonIzMzNDQ1NScpKyciPjwvZGl2PjwvZGl2PjwvZGl2Pic7CiAgICAgICAgICB9KS5qb2luKCcnKSsnPC9kaXY+JzoKICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG93LXNpZ25hbCByZWdpb24uIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy48L2Rpdj4nKSsKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkF0dGVudGlvbiDigJQgOCBkYXlzPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0idGwtd3JhcCI+PHN2ZyB2aWV3Qm94PSIwIDAgJyt0dysnICcrdGgrJyIgc3R5bGU9IndpZHRoOjEwMCU7aGVpZ2h0OjEwMCUiPicrCiAgICAgICAgICAnPGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJ0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJyIgeDE9IjAiIHgyPSIwIiB5MT0iMCIgeTI9IjEiPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMC4yNSIvPicrCiAgICAgICAgICAgICc8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwIi8+JysKICAgICAgICAgICc8L2xpbmVhckdyYWRpZW50PjwvZGVmcz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrYUQrJyIgZmlsbD0idXJsKCN0bGcnK25tLnJlcGxhY2UoL1teYS16XS9naSwnJykrJykiIC8+JysKICAgICAgICAgICc8cGF0aCBkPSInK3BEKyciIGZpbGw9Im5vbmUiIHN0cm9rZT0iJythYysnIiBzdHJva2Utd2lkdGg9IjEuMiIvPicrCiAgICAgICAgICBwdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuICc8Y2lyY2xlIGN4PSInK3BbMF0rJyIgY3k9IicrcFsxXSsnIiByPSInKyhpPT09cHRzLmxlbmd0aC0xPzIuMjoxLjIpKyciIGZpbGw9IicrYWMrJyIvPic7fSkuam9pbignJykrCiAgICAgICAgJzwvc3ZnPjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWxzIDxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCkiPicrKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoP2QuYXJ0aWNsZXMubGVuZ3RoOjApKyc8L3NwYW4+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0iYXJ0LWxpc3QiPicrCiAgICAgICAgICAoKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoKT8KICAgICAgICAgICAgZC5hcnRpY2xlcy5tYXAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIGlzUmVkZGl0PWEuc3JjJiZhLnNyYy5pbmRleE9mKCdyZWRkaXQnKT4tMTsKICAgICAgICAgICAgICB2YXIgaXNZdD1hLnNyYyYmYS5zcmMuaW5kZXhPZigneW91dHViZScpPi0xOwogICAgICAgICAgICAgIHZhciBpc1Blb3BsZXM9aXNSZWRkaXR8fGlzWXQ7CiAgICAgICAgICAgICAgdmFyIHNyY0xhYmVsPWlzUGVvcGxlcz8KICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iY29sb3I6cmdiYSgyMjQsOTAsNDAsMC43KTtmb250LXNpemU6OHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2xldHRlci1zcGFjaW5nOjAuMDZlbSI+dm9pY2VzPC9zcGFuPic6CiAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCkiPicrKCBhLnNyY3x8JycpKyc8L3NwYW4+JzsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIiBzdHlsZT0iJysoaXNSZWRkaXQ/J2JvcmRlci1sZWZ0OjJweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjIpO3BhZGRpbmctbGVmdDo4cHg7JzonJykrJyI+JysgCiAgICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0iYXJ0LXNyYyI+JytzcmNMYWJlbCsnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjZweCAwIj5ObyBzaWduYWxzIGNvbGxlY3RlZCB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgLy8gVXNlIHNhbWUgZnVuY3Rpb25zIGFzIG1hcCDigJQgZ3VhcmFudGVlZCB0byBtYXRjaAogICAgdmFyIG1hcERvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgIHZhciBicmVha2Rvd249Z2V0RW1vdGlvbkJyZWFrZG93bihubSk7CiAgICB2YXIgZW1vdGlvbnM9YnJlYWtkb3duLmVtb3Rpb25zOwogICAgdmFyIGhhc0Vtb3M9IWJyZWFrZG93bi5lc3RpbWF0ZWQ7CiAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vdGlvbnMpOwogICAgdmFyIGVUb3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICBpZihlVG90PjAmJmVUb3Q8PTEuMDEpe2VMPWVMLm1hcChmdW5jdGlvbihrdil7cmV0dXJuW2t2WzBdLE1hdGgucm91bmQoa3ZbMV0qMTAwKV07fSk7fQogICAgdmFyIHRvdD1NYXRoLm1heCgxLGVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApKTsKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogICAgaWYoIWVMLmxlbmd0aCl7cGFuZWwuaW5uZXJIVE1MPWhlYWRlcisnPGRpdiBzdHlsZT0icGFkZGluZzoyMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweCI+Tm8gZW1vdGlvbiBkYXRhIHlldC48L2Rpdj4nO3JldHVybjt9CiAgICAvLyBkb21FbW8gPSBzYW1lIGFzIG1hcCBjb2xvciAoZnJvbSBnZXRFZmZlY3RpdmVFbW90aW9uKQogICAgdmFyIGRvbUVtbz1tYXBEb21FbW87CiAgICAvLyBSZW9yZGVyIGVMIHNvIGRvbWluYW50IHNob3dzIGZpcnN0CiAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7CiAgICAgIGlmKGFbMF09PT1kb21FbW8pIHJldHVybiAtMTsKICAgICAgaWYoYlswXT09PWRvbUVtbykgcmV0dXJuIDE7CiAgICAgIHJldHVybiBiWzFdLWFbMV07CiAgICB9KTsKICAgIHZhciBkb21QY3Q9TWF0aC5yb3VuZCgoZUxbMF0/ZUxbMF1bMV06MjApKjEwMC90b3QpOwogICAgdmFyIG5hcnIyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdG9wTmFyU3RyPW5hcnIyLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignIGFuZCAnKTsKICAgIHZhciB3aGF0SXQ9e2FueGlldHk6J1VuY2VydGFpbnR5IGFuZCB1bmVhc2UgaW4gJytubSsodG9wTmFyU3RyPycuIFNpZ25hbHM6ICcrdG9wTmFyU3RyKycuJzonJyksYW5nZXI6J091dHJhZ2UgYW5kIHByZXNzdXJlIGluICcrbm0rKHRvcE5hclN0cj8nLiBEcml2ZW4gYnk6ICcrdG9wTmFyU3RyKycuJzonJyksaG9wZTonT3B0aW1pc20gYW5kIHByb2dyZXNzIGluICcrbm0rKHRvcE5hclN0cj8nLiBBcm91bmQ6ICcrdG9wTmFyU3RyKycuJzonJykscHJpZGU6J0lkZW50aXR5IGFuZCBhY2hpZXZlbWVudCBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gQXJvdW5kOiAnK3RvcE5hclN0cisnLic6JycpLGZlYXI6J1RocmVhdCBwZXJjZXB0aW9uIGluICcrbm0rKHRvcE5hclN0cj8nLiBBcm91bmQ6ICcrdG9wTmFyU3RyKycuJzonJyl9OwogICAgdmFyIGN1bUE9LU1hdGguUEkvMixjeD0zOCxjeT0zOCxSPTMzLHJpPTIwOwogICAgdmFyIGFyY3M9ZUwubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgdmFyIGs9a3ZbMF0sdj1rdlsxXSxmcj12L3RvdCxhMT1jdW1BLGEyPWN1bUErZnIqTWF0aC5QSSoyO2N1bUE9YTI7CiAgICAgIHZhciBsZz0oYTItYTEpPk1hdGguUEk/MTowOwogICAgICB2YXIgeDE9Y3grTWF0aC5jb3MoYTEpKlIseTE9Y3krTWF0aC5zaW4oYTEpKlIseDI9Y3grTWF0aC5jb3MoYTIpKlIseTI9Y3krTWF0aC5zaW4oYTIpKlI7CiAgICAgIHZhciB4Mz1jeCtNYXRoLmNvcyhhMikqcmkseTM9Y3krTWF0aC5zaW4oYTIpKnJpLHg0PWN4K01hdGguY29zKGExKSpyaSx5ND1jeStNYXRoLnNpbihhMSkqcmk7CiAgICAgIHJldHVybiAnPHBhdGggZD0iTScreDEudG9GaXhlZCgxKSsnLCcreTEudG9GaXhlZCgxKSsnIEEnK1IrJywnK1IrJyAwICcrbGcrJyAxICcreDIudG9GaXhlZCgxKSsnLCcreTIudG9GaXhlZCgxKSsnIEwnK3gzLnRvRml4ZWQoMSkrJywnK3kzLnRvRml4ZWQoMSkrJyBBJytyaSsnLCcrcmkrJyAwICcrbGcrJyAwICcreDQudG9GaXhlZCgxKSsnLCcreTQudG9GaXhlZCgxKSsnIFoiIGZpbGw9IicrcGFsW2tdKyciIG9wYWNpdHk9IjAuOSIvPic7CiAgICB9KS5qb2luKCcnKTsKICAgIHZhciBlZGVzYz17YW54aWV0eTonVW5jZXJ0YWludHksIHdvcnJ5JyxhbmdlcjonT3V0cmFnZSwgcHJvdGVzdCcsaG9wZTonT3B0aW1pc20sIHByb2dyZXNzJyxwcmlkZTonQWNoaWV2ZW1lbnQsIGlkZW50aXR5JyxmZWFyOidUaHJlYXQsIGluc2VjdXJpdHknfTsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnVGhlIGVtb3Rpb25hbCB1bmRlcmN1cnJlbnQgb2Ygc2lnbmFscyBmcm9tICcrbm0rJy4gV2hhdCB0b25lIGRvbWluYXRlcyB0aGUgcG9saXRpY2FsIGRpc2NvdXJzZSDigJQgb3V0cmFnZSwgaG9wZSwgZmVhciwgb3IgYW54aWV0eT8nKwogICAgJzwvZGl2PicrCiAgICAoIWhhc0Vtb3M/JzxkaXYgc3R5bGU9InBhZGRpbmc6NnB4IDExcHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTttYXJnaW4tYm90dG9tOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCkiPkVzdGltYXRlZCBmcm9tIHNpZ25hbCBkaXJlY3Rpb24g4oCUIGxpbWl0ZWQgZGlyZWN0IGVtb3Rpb24gZGF0YS48L2Rpdj4nOicnKSsKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MTRweDtib3JkZXItcmFkaXVzOjEwcHg7YmFja2dyb3VuZDonK3BhbFtkb21FbW9dKycxNDtib3JkZXI6MXB4IHNvbGlkICcrcGFsW2RvbUVtb10rJzMzO21hcmdpbi1ib3R0b206MTJweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOicrcGFsW2RvbUVtb10rJzttYXJnaW4tYm90dG9tOjZweCI+RG9taW5hbnQgZW1vdGlvbjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjZweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+Jytkb21FbW8uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZG9tRW1vLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4Ij4nK2RvbVBjdCsnJSDCtyAnK25tKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6OHB4O2xpbmUtaGVpZ2h0OjEuNTtmb250LXN0eWxlOml0YWxpYyI+Jyt3aGF0SXRbZG9tRW1vXSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkVtb3Rpb25hbCBicmVha2Rvd248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OyI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCA3NiA3NiIgc3R5bGU9IndpZHRoOjcycHg7aGVpZ2h0OjcycHg7ZmxleC1zaHJpbms6MCI+JythcmNzKyc8L3N2Zz4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHg7Ij4nKwogICAgICAgICAgICBlTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICAgICAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0scGN0PU1hdGgucm91bmQodioxMDAvdG90KTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6Y2VudGVyO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweDsiPjxzcGFuIHN0eWxlPSJ3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjJweDtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMS41cHg7Y29sb3I6Jysoaz09PWRvbUVtbz8ndmFyKC0taW5rKSc6J3ZhcigtLWRpbSknKSsnIj4nK2suY2hhckF0KDApLnRvVXBwZXJDYXNlKCkray5zbGljZSgxKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2YXIoLS1pbmspIj4nK3BjdCsnJTwvc3Bhbj4nKwogICAgICAgICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3BjdCsnJTtiYWNrZ3JvdW5kOicrcGFsW2tdKyc7b3BhY2l0eTowLjc7Ym9yZGVyLXJhZGl1czoxcHgiPjwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgICAoaz09PWRvbUVtbz8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrZWRlc2Nba10rJzwvZGl2Pic6JycpKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFsIGhlYWRsaW5lczwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweDsiPicrCiAgICAgICAgICAoKGQuYXJ0aWNsZXMmJmQuYXJ0aWNsZXMubGVuZ3RoKT8KICAgICAgICAgICAgZC5hcnRpY2xlcy5zbGljZSgwLDUpLm1hcChmdW5jdGlvbihhKXsKICAgICAgICAgICAgICB2YXIgZUNvbG9yPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDo2cHg7cGFkZGluZzo2cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpOyI+JysKICAgICAgICAgICAgICAgIChhLmVtb3Rpb24/JzxzcGFuIHN0eWxlPSJ3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrZUNvbG9yW2EuZW1vdGlvbl0rJztkaXNwbGF5OmlubGluZS1ibG9jazttYXJnaW4tdG9wOjVweDtmbGV4LXNocmluazowIj48L3NwYW4+JzonJykrCiAgICAgICAgICAgICAgICAnPGRpdj48ZGl2IHN0eWxlPSJmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNCI+JysoYS50eHR8fGEudGl0bGV8fCcnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4Ij4nKyhhLnNyY3x8JycpKyhhLmVtb3Rpb24/JyDCtyAnK2EuZW1vdGlvbjonJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NHB4IDAiPk5vIHNpZ25hbHMgeWV0LjwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKCiAgfSBlbHNlIHsKICAgIHZhciB2ZWw9ZC52ZWxvY2l0eXx8MDsKICAgIHZhciB2ZWxEaXI9dmVsPjAuMTU/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wNT8nUmlzaW5nJzp2ZWw8LTAuMT8nQ29vbGluZyBmYXN0Jzp2ZWw8LTAuMDI/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgdmFyIHZlbENvbD12ZWw+MC4wNT8nI2UwNWEyOCc6dmVsPC0wLjAyPycjM2JiOGQ4JzonIzU1NjY3Nyc7CiAgICB2YXIgdmVsRGVzYz17J1Jpc2luZyBmYXN0JzonU2lnbmFsIHZvbHVtZSBzdXJnaW5nLicsJ1Jpc2luZyc6J0F0dGVudGlvbiBidWlsZGluZy4nLCdTdGFibGUnOidCYWxhbmNlZCBtb21lbnR1bS4nLCdDb29saW5nJzonQXR0ZW50aW9uIGZhZGluZy4nLCdDb29saW5nIGZhc3QnOidTaGFycCBzaWduYWwgZGVjYXkuJ307CiAgICB2YXIgbmFycjM9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciByaXNpbmdOYXJzPW5hcnIzLmZpbHRlcihmdW5jdGlvbihuKXtyZXR1cm4gbi5kaXI9PT0ndXAnO30pOwogICAgdmFyIGZhbGxpbmdOYXJzPW5hcnIzLmZpbHRlcihmdW5jdGlvbihuKXtyZXR1cm4gbi5kaXI9PT0nZG93bic7fSk7CiAgICB2YXIgY3R4PScnOwogICAgaWYodmVsPjAuMDUmJnJpc2luZ05hcnMubGVuZ3RoKSBjdHg9J0RyaXZlbiBieSByaXNpbmcgc2lnbmFscyBhcm91bmQgPHN0cm9uZz4nK3Jpc2luZ05hcnMuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCc8L3N0cm9uZz4gYW5kIDxzdHJvbmc+JykrJzwvc3Ryb25nPi4nOwogICAgZWxzZSBpZih2ZWw8LTAuMDUmJmZhbGxpbmdOYXJzLmxlbmd0aCkgY3R4PSdTaWduYWxzIGFyb3VuZCA8c3Ryb25nPicrZmFsbGluZ05hcnMuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCc8L3N0cm9uZz4gYW5kIDxzdHJvbmc+JykrJzwvc3Ryb25nPiBsb3NpbmcgdHJhY3Rpb24uJzsKICAgIGVsc2UgY3R4PSdTaWduYWwgdm9sdW1lICcrKHZlbD4wLjAyPydidWlsZGluZyc6J3N0YWJsZScpKycgaW4gJytubSsnLic7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ0lzIGF0dGVudGlvbiBmb3IgJytubSsnIGdyb3dpbmcgb3IgZmFkaW5nPyBSaXNpbmcgbW9tZW50dW0gbWVhbnMgYSBuYXJyYXRpdmUgaXMgYWNjZWxlcmF0aW5nLiBDb29saW5nIG1lYW5zIHRoZSBzdG9yeSBpcyBsb3NpbmcgdHJhY3Rpb24uJysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MTRweDtib3JkZXItcmFkaXVzOjEwcHg7YmFja2dyb3VuZDonK3ZlbENvbCsnMTQ7Ym9yZGVyOjFweCBzb2xpZCAnK3ZlbENvbCsnMzM7bWFyZ2luLWJvdHRvbToxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6Jyt2ZWxDb2wrJzttYXJnaW4tYm90dG9tOjZweCI+U2lnbmFsIG1vbWVudHVtPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDoxMHB4O21hcmdpbi1ib3R0b206OHB4OyI+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjMycHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluaykiPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvZGl2PicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjE0cHg7Y29sb3I6Jyt2ZWxDb2wrJztmb250LXdlaWdodDo1MDAiPicrdmVsRGlyKyc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtc3R5bGU6aXRhbGljO2xpbmUtaGVpZ2h0OjEuNSI+Jyt2ZWxEZXNjW3ZlbERpcl0rJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjY7bWFyZ2luLXRvcDoxMHB4O3BhZGRpbmctdG9wOjEwcHg7Ym9yZGVyLXRvcDoxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA1KSI+JytjdHgrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic2NvcmUtc3RyaXAiPicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5WZWxvY2l0eTwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCIgc3R5bGU9ImZvbnQtc2l6ZToxOHB4O2NvbG9yOicrdmVsQ29sKyciPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj4yNGggzrQ8L2Rpdj48ZGl2IGNsYXNzPSJzcy1kZWx0YSAnKyhkLmRlbHRhPj0wPyd1cCc6J2RuJykrJyI+JysoZC5kZWx0YT49MD8nKyc6JycpKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLW5hciI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgKHJpc2luZ05hcnMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkFjY2VsZXJhdGluZzwvZGl2PicrCiAgICAgICAgcmlzaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDUpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xMikiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspIj4nK3IubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyLm5hbWUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6I2UwNWEyOCI+JytyLnZhbC50b0ZpeGVkKDEpKyclPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJykrJzwvZGl2Pic6JycpKwogICAgICAoZmFsbGluZ05hcnMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPkRlY2VsZXJhdGluZzwvZGl2PicrCiAgICAgICAgZmFsbGluZ05hcnMubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6N3B4IDEwcHg7bWFyZ2luLWJvdHRvbTo0cHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDU5LDE4NCwyMTYsMC4xMikiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspIj4nK3IubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyLm5hbWUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6IzNiYjhkOCI+JytyLnZhbC50b0ZpeGVkKDEpKyclPC9zcGFuPjwvZGl2Pic7fSkuam9pbignJykrJzwvZGl2Pic6JycpOwogIH0KCiAgcGFuZWwuaW5uZXJIVE1MPWhlYWRlcitib2R5Owp9CgoKZnVuY3Rpb24gdG9nZ2xlRmF2KG5tKXsKICBpZihGQVZTLmhhcyhubSkpIEZBVlMuZGVsZXRlKG5tKTtlbHNlIEZBVlMuYWRkKG5tKTsKICByZW5kZXJQYW5lbChTRUwpO3JlbmRlckZhdnMoKTsKfQpmdW5jdGlvbiByZW5kZXJGYXZzKCl7CiAgdmFyIHJvdz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmF2LXJvdycpOwogIGlmKCFGQVZTLnNpemUpe3Jvdy5pbm5lckhUTUw9JzxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2Pic7cmV0dXJuO30KICByb3cuaW5uZXJIVE1MPUFycmF5LmZyb20oRkFWUykubWFwKGZ1bmN0aW9uKG5tKXsKICAgIHZhciBkPWcobm0pLGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/JyNlMDVhMjgnOicjM2JiOGQ4JzsKICAgIHZhciB0b3A9ZC5uYXJyYXRpdmVzJiZkLm5hcnJhdGl2ZXNbMF0/ZC5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCc7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImZhdi1jYXJkIiBvbmNsaWNrPSJzZWxlY3RfKFwnJytubSsnXCcpIj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtaGVhZCI+PHNwYW4gY2xhc3M9ImZjLW5hbWUiPicrbm0rJzwvc3Bhbj48c3BhbiBjbGFzcz0iZmMtc2MiPicrZC5hdHRlbnRpb24rJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj5OYXJyYXRpdmU8L3NwYW4+PHNwYW4gY2xhc3M9InYiPicrdG9wKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+MjRoPC9zcGFuPjxzcGFuIGNsYXNzPSJ2IiBzdHlsZT0iY29sb3I6JytkQysnIj4nK2RTK2QuZGVsdGErJzwvc3Bhbj48L2Rpdj4nKwogICAgJzwvZGl2Pic7CiAgfSkuam9pbignJyk7Cn0KCmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbihjKXsKICBjLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHgpe3guY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICBjLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO2xheWVyPWMuZGF0YXNldC5sYXllcjthcHBseUxheWVyKCk7CiAgfSk7Cn0pOwoKZnVuY3Rpb24gdXBkYXRlQ2xvY2soKXsKICB2YXIgbm93PW5ldyBEYXRlKCksaXN0PW5ldyBEYXRlKG5vdy5nZXRUaW1lKCkrbm93LmdldFRpbWV6b25lT2Zmc2V0KCkqNjAwMDArMTk4MDAwMDApOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbG9jaycpLnRleHRDb250ZW50PVN0cmluZyhpc3QuZ2V0SG91cnMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldE1pbnV0ZXMoKSkucGFkU3RhcnQoMiwnMCcpKyc6JytTdHJpbmcoaXN0LmdldFNlY29uZHMoKSkucGFkU3RhcnQoMiwnMCcpKycgSVNUJzsKfQpzZXRJbnRlcnZhbCh1cGRhdGVDbG9jaywxMDAwKTt1cGRhdGVDbG9jaygpOwoKLy8gSU5JVCDigJQgd2FpdCBmb3IgRE9NCi8vIGkgYnV0dG9uIHRvb2x0aXAg4oCUIHVzZXMgZml4ZWQgcG9zaXRpb25pbmcgc28gaXQncyBuZXZlciBjbGlwcGVkCihmdW5jdGlvbigpewogIHZhciB0aXA9bnVsbDsKICBmdW5jdGlvbiBzaG93VGlwKGUpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICB2YXIgdHh0PXRoaXMuZ2V0QXR0cmlidXRlKCdkYXRhLXRpcCcpOwogICAgaWYoIXR4dHx8IXRpcCkgcmV0dXJuOwogICAgdGlwLnRleHRDb250ZW50PXR4dDsKICAgIHRpcC5jbGFzc0xpc3QuYWRkKCd2aXNpYmxlJyk7CiAgICB2YXIgcmVjdD10aGlzLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgdmFyIHR3PTI0MDsKICAgIHZhciBsZWZ0PU1hdGgubWluKHJlY3QubGVmdCx3aW5kb3cuaW5uZXJXaWR0aC10dy0xMCk7CiAgICB0aXAuc3R5bGUubGVmdD1sZWZ0KydweCc7CiAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC0xMC10aXAub2Zmc2V0SGVpZ2h0fHxyZWN0LnRvcC04MCkrJ3B4JzsKICAgIC8vIFJlcG9zaXRpb24gYWZ0ZXIgcmVuZGVyCiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZnVuY3Rpb24oKXsKICAgICAgdGlwLnN0eWxlLnRvcD0ocmVjdC50b3AtdGlwLm9mZnNldEhlaWdodC04KSsncHgnOwogICAgfSk7CiAgfQogIGZ1bmN0aW9uIGhpZGVUaXAoKXsKICAgIGlmKCF0aXApe3RpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbHRhYi10b29sdGlwJyk7fQogICAgaWYodGlwKSB0aXAuY2xhc3NMaXN0LnJlbW92ZSgndmlzaWJsZScpOwogIH0KICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW92ZXInLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgc2hvd1RpcC5jYWxsKGUudGFyZ2V0LGUpOwogIH0pOwogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3V0JyxmdW5jdGlvbihlKXsKICAgIGlmKGUudGFyZ2V0LmNsYXNzTGlzdC5jb250YWlucygnbHRhYi1pbmZvJykpIGhpZGVUaXAoKTsKICB9KTsKfSkoKTsKCi8vIOKUgOKUgCBNT0JJTEUgQk9UVE9NIFNIRUVUIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAovLyBPbmx5IGFjdGl2YXRlcyBvbiBtb2JpbGUg4oCUIG5vIGVmZmVjdCBvbiBkZXNrdG9wCihmdW5jdGlvbigpewogIHZhciBpc01vYmlsZT1mdW5jdGlvbigpe3JldHVybiB3aW5kb3cuaW5uZXJXaWR0aDw9NzY4O307CiAgdmFyIG92ZXJsYXk9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7CiAgb3ZlcmxheS5jbGFzc05hbWU9J21hcC1vdmVybGF5LWRpbSc7CiAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChvdmVybGF5KTsKCiAgZnVuY3Rpb24gb3BlblBhbmVsKCl7CiAgICBpZighaXNNb2JpbGUoKSkgcmV0dXJuOwogICAgdmFyIHBhbmVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5zdGF0ZS1wYW5lbCcpOwogICAgaWYocGFuZWwpe3BhbmVsLmNsYXNzTGlzdC5hZGQoJ3BhbmVsLW9wZW4nKTt9CiAgICBvdmVybGF5LmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogICAgZG9jdW1lbnQuYm9keS5zdHlsZS5vdmVyZmxvdz0naGlkZGVuJzsKICB9CiAgZnVuY3Rpb24gY2xvc2VQYW5lbCgpewogICAgdmFyIHBhbmVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5zdGF0ZS1wYW5lbCcpOwogICAgaWYocGFuZWwpe3BhbmVsLmNsYXNzTGlzdC5yZW1vdmUoJ3BhbmVsLW9wZW4nKTt9CiAgICBvdmVybGF5LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpOwogICAgZG9jdW1lbnQuYm9keS5zdHlsZS5vdmVyZmxvdz0nJzsKICB9CgogIC8vIE9wZW4gcGFuZWwgd2hlbiBzdGF0ZSBzZWxlY3RlZAogIHZhciBvcmlnU2VsZWN0PXdpbmRvdy5zZWxlY3RfOwogIHdpbmRvdy5zZWxlY3RfPWZ1bmN0aW9uKG5tKXsKICAgIG9yaWdTZWxlY3Qobm0pOwogICAgaWYoaXNNb2JpbGUoKSkgc2V0VGltZW91dChvcGVuUGFuZWwsNTApOwogIH07CgogIC8vIENsb3NlIG9uIG92ZXJsYXkgdGFwCiAgb3ZlcmxheS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsY2xvc2VQYW5lbCk7CgogIC8vIENsb3NlIG9uIHN3aXBlIGRvd24KICB2YXIgc3RhcnRZPTA7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcigndG91Y2hzdGFydCcsZnVuY3Rpb24oZSl7CiAgICBzdGFydFk9ZS50b3VjaGVzWzBdLmNsaWVudFk7CiAgfSx7cGFzc2l2ZTp0cnVlfSk7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcigndG91Y2hlbmQnLGZ1bmN0aW9uKGUpewogICAgdmFyIHBhbmVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5zdGF0ZS1wYW5lbCcpOwogICAgaWYoIXBhbmVsfHwhcGFuZWwuY2xhc3NMaXN0LmNvbnRhaW5zKCdwYW5lbC1vcGVuJykpIHJldHVybjsKICAgIHZhciBkeT1lLmNoYW5nZWRUb3VjaGVzWzBdLmNsaWVudFktc3RhcnRZOwogICAgaWYoZHk+NjApIGNsb3NlUGFuZWwoKTsgLy8gc3dpcGUgZG93biA2MHB4IHRvIGNsb3NlCiAgfSx7cGFzc2l2ZTp0cnVlfSk7Cn0pKCk7CgpmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7CiAgdmFyIGxkcj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpOwogIGlmKCFsZHIpIHJldHVybjsKICBsZHIuc3R5bGUub3BhY2l0eT0nMCc7CiAgbGRyLnN0eWxlLnZpc2liaWxpdHk9J2hpZGRlbic7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2xkci5zdHlsZS5kaXNwbGF5PSdub25lJzt9LDgwMCk7Cn0KCmZ1bmN0aW9uIGluaXQoKXsKICByZW5kZXJTdHJpcCgnM20nKTsKCiAgLy8gTG9hZCBtYXAgd2l0aCByZXRyeQogIHZhciBtYXBBdHRlbXB0cz0wOwogIGZ1bmN0aW9uIHRyeUxvYWRNYXAoKXsKICAgIGlmKHR5cGVvZiB0b3BvanNvbj09PSd1bmRlZmluZWQnKXsKICAgICAgaWYobWFwQXR0ZW1wdHMrKzwxMCl7c2V0VGltZW91dCh0cnlMb2FkTWFwLDMwMCk7fQogICAgICByZXR1cm47CiAgICB9CiAgICBsb2FkTWFwKCk7CiAgfQogIHRyeUxvYWRNYXAoKTsKCiAgLy8gTG9hZCBmdWxsIGNhY2hlZCBzbmFwc2hvdCBpbW1lZGlhdGVseSBmb3IgaW5zdGFudCBkYXRhCiAgZmV0Y2hGdWxsU25hcHNob3QoKS50aGVuKGZ1bmN0aW9uKG9rKXsKICAgIGlmKG9rKXsKICAgICAgcmVuZGVyTW9tZW50dW0oKTsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe3N0YXJ0UG9sbGluZygpO30sMTAwMCk7CiAgICB9IGVsc2UgewogICAgICBzdGFydFBvbGxpbmcoKTsKICAgIH0KICAgIC8vIERpc21pc3MgbG9hZGVyIGFmdGVyIG1heCAzcyByZWdhcmRsZXNzCiAgICBzZXRUaW1lb3V0KGRpc21pc3NMb2FkZXIsIDMwMDApOwogIH0pOwoKICAvLyBSZXRyeSBtYXAgaWYgc3RpbGwgZW1wdHkKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSwzMDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSw2MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw1MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDgwMDApOwp9CmlmKGRvY3VtZW50LnJlYWR5U3RhdGU9PT0nbG9hZGluZycpewogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ0RPTUNvbnRlbnRMb2FkZWQnLCBpbml0KTsKfSBlbHNlIHsKICAvLyBBbHJlYWR5IGxvYWRlZCDigJQgYnV0IHdhaXQgb25lIHRpY2sgdG8gZW5zdXJlIGFsbCBzY3JpcHRzIHBhcnNlZAogIHNldFRpbWVvdXQoaW5pdCwgMCk7Cn0KCi8vIFJFUExBWSBJTkRJQQp2YXIgUkVQTEFZX1BFUklPRFM9eyc3ZCc6e2RheXM6NyxsYWJlbDonUGFzdCA3IGRheXMnfSwnMzBkJzp7ZGF5czozMCxsYWJlbDonUGFzdCAzMCBkYXlzJ30sJzZtJzp7ZGF5czoxODAsbGFiZWw6J1Bhc3QgNiBtb250aHMnfSwnZWxlY3Rpb24nOntkYXlzOjM2NSxsYWJlbDonRWxlY3Rpb25zIOKAlCBwYXN0IHllYXInfX07CnZhciByZXBsYXlQZXJpb2Q9JzdkJyxyZXBsYXlQb3M9MCxyZXBsYXlQbGF5aW5nPWZhbHNlLHJlcGxheVRpbWVyPW51bGwscmVwbGF5U3BlZWQ9MSxsYXN0U25hcFBvcz0tMTsKdmFyIEhJU1RPUllfREFUQT17fTsgLy8gS2V5ZWQgYnkgc3RhdGUgLT4gW3tkYXRlLGF0dGVudGlvbixkb21pbmFudF9lbW90aW9uLC4uLn1dCnZhciBOQVRJT05BTF9ISVNUT1JZPVtdOyAvLyBbe2RhdGUsbmFycmF0aXZlcyxlbW90aW9uLGhvdHRlc3Rfc3RhdGV9XQp2YXIgSElTVE9SWV9MT0FERUQ9ZmFsc2U7Cgphc3luYyBmdW5jdGlvbiBsb2FkSGlzdG9yeURhdGEoZGF5cyl7CiAgdHJ5ewogICAgdmFyIFtzdGF0ZVJlcywgbmF0UmVzXSA9IGF3YWl0IFByb21pc2UuYWxsKFsKICAgICAgZmV0Y2goQVBJX0JBU0UrJy9hcGkvaGlzdG9yeT9kYXlzPScrKGRheXN8fDcpKSwKICAgICAgZmV0Y2goQVBJX0JBU0UrJy9hcGkvbmF0aW9uYWwtaGlzdG9yeT9kYXlzPScrKGRheXN8fDcpKQogICAgXSk7CiAgICAvLyBTdGF0ZSBoaXN0b3J5CiAgICBpZihzdGF0ZVJlcy5vayl7CiAgICAgIHZhciBzZD1hd2FpdCBzdGF0ZVJlcy5qc29uKCk7CiAgICAgIGlmKCFzZC5lcnJvciYmc2QuaGFzX2RhdGEpe0hJU1RPUllfREFUQT1zZC5zdGF0ZXN8fHt9O30KICAgIH0KICAgIC8vIE5hdGlvbmFsIG5hcnJhdGl2ZSBoaXN0b3J5CiAgICBpZihuYXRSZXMub2spewogICAgICB2YXIgbmQ9YXdhaXQgbmF0UmVzLmpzb24oKTsKICAgICAgaWYoIW5kLmVycm9yJiZuZC5kYXRhJiZuZC5kYXRhLmxlbmd0aCl7CiAgICAgICAgTkFUSU9OQUxfSElTVE9SWT1uZC5kYXRhOwogICAgICAgIGNvbnNvbGUubG9nKCdbcmVwbGF5XSBOYXRpb25hbCBoaXN0b3J5OicsbmQuZGF0YS5sZW5ndGgsJ2RheXMnKTsKICAgICAgfQogICAgfQogICAgSElTVE9SWV9MT0FERUQ9T2JqZWN0LmtleXMoSElTVE9SWV9EQVRBKS5sZW5ndGg+MHx8TkFUSU9OQUxfSElTVE9SWS5sZW5ndGg+MDsKICAgIHJldHVybiBISVNUT1JZX0xPQURFRDsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1tyZXBsYXldIEhpc3RvcnkgbG9hZCBmYWlsZWQ6JyxlLm1lc3NhZ2UpOwogICAgcmV0dXJuIGZhbHNlOwogIH0KfQoKZnVuY3Rpb24gZ2V0TmF0aW9uYWxOYXJyYXRpdmVBdCh0YXJnZXREYXRlKXsKICAvLyBGaW5kIG5hdGlvbmFsIG5hcnJhdGl2ZSBjbG9zZXN0IHRvIHRhcmdldERhdGUKICBpZighTkFUSU9OQUxfSElTVE9SWS5sZW5ndGgpIHJldHVybiBudWxsOwogIHZhciB0YXJnZXRTdHI9dGFyZ2V0RGF0ZS50b0lTT1N0cmluZygpLnNsaWNlKDAsMTApOwogIHZhciBiZXN0PW51bGwsYmVzdERpZmY9SW5maW5pdHk7CiAgTkFUSU9OQUxfSElTVE9SWS5mb3JFYWNoKGZ1bmN0aW9uKGgpewogICAgdmFyIGRpZmY9TWF0aC5hYnMobmV3IERhdGUoaC5kYXRlKS10YXJnZXREYXRlKTsKICAgIGlmKGRpZmY8YmVzdERpZmYpe2Jlc3REaWZmPWRpZmY7YmVzdD1oO30KICB9KTsKICByZXR1cm4gYmVzdDsKfQpmdW5jdGlvbiBmbXREYXRlKGQpe3JldHVybiBkLnRvTG9jYWxlRGF0ZVN0cmluZygnZW4tSU4nLHtkYXk6J251bWVyaWMnLG1vbnRoOidzaG9ydCd9KTt9CmZ1bmN0aW9uIGluaXRSZXBsYXkoKXsKICB2YXIgcD1SRVBMQVlfUEVSSU9EU1tyZXBsYXlQZXJpb2RdLG5vdz1uZXcgRGF0ZSgpLHN0YXJ0PW5ldyBEYXRlKG5vdy1wLmRheXMqODY0MDAwMDApOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtZGF0ZXMnKTsKICBpZihlbCllbC5pbm5lckhUTUw9JzxzcGFuPicrZm10RGF0ZShzdGFydCkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjMzKSkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjY2KSkrJzwvc3Bhbj48c3Bhbj5Ub2RheTwvc3Bhbj4nOwogIHNldFJlcGxheVBvcygwKTsKfQpmdW5jdGlvbiBnZXRIaXN0b3JpY2FsQXR0ZW50aW9uKHN0YXRlLCB0YXJnZXREYXRlKXsKICAvLyBHZXQgYXR0ZW50aW9uIHNjb3JlIGZvciBhIHN0YXRlIGF0IGEgc3BlY2lmaWMgZGF0ZSBmcm9tIERCIGhpc3RvcnkKICB2YXIgaGlzdD1ISVNUT1JZX0RBVEFbc3RhdGVdOwogIGlmKCFoaXN0fHwhaGlzdC5sZW5ndGgpIHJldHVybiBudWxsOwogIHZhciB0YXJnZXRTdHI9dGFyZ2V0RGF0ZS50b0lTT1N0cmluZygpLnNsaWNlKDAsMTApOwogIC8vIEZpbmQgY2xvc2VzdCBkYXRlCiAgdmFyIGJlc3Q9bnVsbCxiZXN0RGlmZj1JbmZpbml0eTsKICBoaXN0LmZvckVhY2goZnVuY3Rpb24oaCl7CiAgICB2YXIgZGlmZj1NYXRoLmFicyhuZXcgRGF0ZShoLmRhdGUpLXRhcmdldERhdGUpOwogICAgaWYoZGlmZjxiZXN0RGlmZil7YmVzdERpZmY9ZGlmZjtiZXN0PWg7fQogIH0pOwogIHJldHVybiBiZXN0Owp9CgpmdW5jdGlvbiBzZXRSZXBsYXlQb3MocG9zKXsKICByZXBsYXlQb3M9TWF0aC5tYXgoMCxNYXRoLm1pbigxLHBvcykpOwogIHZhciBmaWxsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1maWxsJyksdGh1bWI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRodW1iJyksZGF0ZUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1jdXJyZW50LWRhdGUnKTsKICBpZihmaWxsKWZpbGwuc3R5bGUud2lkdGg9KHJlcGxheVBvcyoxMDApKyclJzsKICBpZih0aHVtYil0aHVtYi5zdHlsZS5sZWZ0PShyZXBsYXlQb3MqMTAwKSsnJSc7CiAgdmFyIHA9UkVQTEFZX1BFUklPRFNbcmVwbGF5UGVyaW9kXSxub3c9bmV3IERhdGUoKSxzdGFydD1uZXcgRGF0ZShub3ctcC5kYXlzKjg2NDAwMDAwKTsKICB2YXIgY3VyPW5ldyBEYXRlKHN0YXJ0LmdldFRpbWUoKStyZXBsYXlQb3MqcC5kYXlzKjg2NDAwMDAwKTsKICBpZihkYXRlRWwpZGF0ZUVsLnRleHRDb250ZW50PWZtdERhdGUoY3VyKSsnIOKAlCAnK3AubGFiZWwrKEhJU1RPUllfTE9BREVEPycgwrcgcmVhbCBkYXRhJzonwrcgc2ltdWxhdGVkJyk7CgogIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgLy8gQ29sbGVjdCBhbGwgYXR0ZW50aW9uIHNjb3JlcyBhdCB0aGlzIHBvaW50IGZvciBub3JtYWxpemF0aW9uCiAgdmFyIGFsbFNjb3Jlcz1bXTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHBhdGgpewogICAgdmFyIG5tPXBhdGguZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKTsKICAgIHZhciBoaXN0X2VudHJ5PUhJU1RPUllfTE9BREVEP2dldEhpc3RvcmljYWxBdHRlbnRpb24obm0sY3VyKTpudWxsOwogICAgdmFyIGF0dD1oaXN0X2VudHJ5P2hpc3RfZW50cnkuYXR0ZW50aW9uOigoZyhubSkuYXR0ZW50aW9ufHwwKSooMC4zNStyZXBsYXlQb3MqMC42NSkpOwogICAgYWxsU2NvcmVzLnB1c2goYXR0KTsKICB9KTsKICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxhbGxTY29yZXMpLG14PU1hdGgubWF4LmFwcGx5KG51bGwsYWxsU2NvcmVzKXx8MTsKCiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwYXRoKXsKICAgIHZhciBubT1wYXRoLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk7CiAgICB2YXIgaGlzdF9lbnRyeT1ISVNUT1JZX0xPQURFRD9nZXRIaXN0b3JpY2FsQXR0ZW50aW9uKG5tLGN1cik6bnVsbDsKICAgIGlmKGhpc3RfZW50cnkpewogICAgICAvLyBSZWFsIGhpc3RvcmljYWwgZGF0YQogICAgICB2YXIgYXR0PWhpc3RfZW50cnkuYXR0ZW50aW9uOwogICAgICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKGF0dC1tbikvKG14LW1uKSkpOwogICAgICBwYXRoLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoYXR0KSk7CiAgICAgIHBhdGguc2V0QXR0cmlidXRlKCdmaWxsLW9wYWNpdHknLE1hdGgubWF4KDAuMiwwLjIrbiowLjgpKTsKICAgIH0gZWxzZSB7CiAgICAgIC8vIFNpbXVsYXRlZCBmYWxsYmFjawogICAgICB2YXIgc2NhbGU9MC4zNStyZXBsYXlQb3MqMC42NTsKICAgICAgdmFyIHNhPShnKG5tKS5hdHRlbnRpb258fDApKnNjYWxlOwogICAgICB2YXIgbjI9TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzYS1tbikvKG14LW1uKSkpOwogICAgICBwYXRoLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoc2EpKTsKICAgICAgcGF0aC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsTWF0aC5tYXgoMC4yLDAuMituMiowLjgpKTsKICAgIH0KICB9KTsKCiAgaWYoTWF0aC5hYnMocmVwbGF5UG9zLWxhc3RTbmFwUG9zKT4wLjEyKXtsYXN0U25hcFBvcz1yZXBsYXlQb3M7dXBkYXRlUmVwbGF5U25hcHNob3QoY3VyKTt9Cn0KZnVuY3Rpb24gdXBkYXRlUmVwbGF5U25hcHNob3QodGFyZ2V0RGF0ZSl7CiAgdmFyIHBvcz1yZXBsYXlQb3M7CiAgdmFyIG5hckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1uYXRpb25hbC1uYXJyYXRpdmUnKTsKICB2YXIgdGFnc0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1uYXJyYXRpdmUtdGFncycpOwogIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLW5hcnJhdGl2ZS1tZXRhJyk7CiAgaWYoIW5hckVsKSByZXR1cm47CgogIGlmKEhJU1RPUllfTE9BREVEJiZ0eXBlb2YgdGFyZ2V0RGF0ZT09PSdvYmplY3QnKXsKICAgIC8vIFRyeSByZWFsIG5hdGlvbmFsIG5hcnJhdGl2ZSBoaXN0b3J5IGZpcnN0CiAgICB2YXIgbmF0U25hcD1nZXROYXRpb25hbE5hcnJhdGl2ZUF0KHRhcmdldERhdGUpOwogICAgaWYobmF0U25hcCYmbmF0U25hcC5uYXJyYXRpdmVzJiZuYXRTbmFwLm5hcnJhdGl2ZXMubGVuZ3RoKXsKICAgICAgdmFyIGVwYWwwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIHAwPW5hdFNuYXAubmFycmF0aXZlc1swXS5uYW1lLHMyMD1uYXRTbmFwLm5hcnJhdGl2ZXNbMV0/bmF0U25hcC5uYXJyYXRpdmVzWzFdLm5hbWU6bnVsbDsKICAgICAgdmFyIGUwPW5hdFNuYXAuZW1vdGlvbixzdDA9bmF0U25hcC5ob3R0ZXN0X3N0YXRlOwogICAgICB2YXIgZVN0cjA9ZTA/JyDigJQgdG9uZTogPGVtIHN0eWxlPSJjb2xvcjonKyhlcGFsMFtlMF18fCcjYWFhJykrJyI+JytlMCsnPC9lbT4nOicnOwogICAgICB2YXIgc3RTdHIwPXN0MD8nIDxzdHJvbmcgc3R5bGU9ImNvbG9yOnZhcigtLWluaykiPicrc3QwKyc8L3N0cm9uZz4gbGVkIGF0dGVudGlvbi4nOicnOwogICAgICBuYXJFbC5pbm5lckhUTUw9JzxlbT4nK3AwLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3AwLnNsaWNlKDEpKyc8L2VtPiB3YXMgdGhlIGRvbWluYW50IHNpZ25hbCcrKHMyMD8nLCBmb2xsb3dlZCBieSA8ZW0+JytzMjArJzwvZW0+JzonJykrZVN0cjArJy4nK3N0U3RyMDsKICAgICAgaWYodGFnc0VsKSB0YWdzRWwuaW5uZXJIVE1MPW5hdFNuYXAubmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsJysoaT09PTA/JzAuMyc6JzAuMTInKSsnKTtjb2xvcjonKyhpPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC41KScpKyciPicrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsnPC9zcGFuPic7CiAgICAgIH0pLmpvaW4oJycpOwogICAgICBpZihtZXRhRWwpIG1ldGFFbC50ZXh0Q29udGVudD1mbXREYXRlKG5ldyBEYXRlKG5hdFNuYXAuZGF0ZSkpKycgwrcgcmVhbCBkYXRhJzsKICAgICAgcmV0dXJuOwogICAgfQogICAgLy8gRmFsbCBiYWNrIHRvIGRlcml2aW5nIGZyb20gc3RhdGUgc25hcHNob3RzCiAgICB2YXIgbmFyQ291bnRzPXt9OwogICAgdmFyIGVtb0NvdW50cz17fTsKICAgIHZhciB0b3BTdGF0ZT1udWxsLCB0b3BBdHQ9MDsKICAgIE9iamVjdC5rZXlzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKG5tKXsKICAgICAgdmFyIGg9Z2V0SGlzdG9yaWNhbEF0dGVudGlvbihubSx0YXJnZXREYXRlKTsKICAgICAgaWYoIWgpIHJldHVybjsKICAgICAgaWYoaC5hdHRlbnRpb24+dG9wQXR0KXt0b3BBdHQ9aC5hdHRlbnRpb247dG9wU3RhdGU9bm07fQogICAgICBpZihoLmRvbWluYW50X25hcnJhdGl2ZSkgbmFyQ291bnRzW2guZG9taW5hbnRfbmFycmF0aXZlXT0obmFyQ291bnRzW2guZG9taW5hbnRfbmFycmF0aXZlXXx8MCkraC5hdHRlbnRpb247CiAgICAgIGlmKGguZG9taW5hbnRfZW1vdGlvbikgZW1vQ291bnRzW2guZG9taW5hbnRfZW1vdGlvbl09KGVtb0NvdW50c1toLmRvbWluYW50X2Vtb3Rpb25dfHwwKSsxOwogICAgfSk7CiAgICB2YXIgdG9wTmFycz1PYmplY3QuZW50cmllcyhuYXJDb3VudHMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pLnNsaWNlKDAsMyk7CiAgICB2YXIgdG9wRW1vPU9iamVjdC5lbnRyaWVzKGVtb0NvdW50cykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgICB2YXIgZXBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogICAgaWYodG9wTmFycy5sZW5ndGgpewogICAgICB2YXIgcHJpbWFyeT10b3BOYXJzWzBdWzBdOwogICAgICB2YXIgc2Vjb25kYXJ5PXRvcE5hcnNbMV0/dG9wTmFyc1sxXVswXTpudWxsOwogICAgICB2YXIgZW1vU3RyPXRvcEVtbz8oJyDigJQgcHVibGljIHRvbmU6IDxlbSBzdHlsZT0iY29sb3I6JytlcGFsW3RvcEVtb1swXV0rJyI+Jyt0b3BFbW9bMF0rJzwvZW0+Jyk6Jyc7CiAgICAgIHZhciBzdGF0ZVN0cj10b3BTdGF0ZT8oJyA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK3RvcFN0YXRlKyc8L3N0cm9uZz4gbGVkIGF0dGVudGlvbi4nKTonJzsKICAgICAgbmFyRWwuaW5uZXJIVE1MPSc8ZW0+JytwcmltYXJ5LmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3ByaW1hcnkuc2xpY2UoMSkrJzwvZW0+IHdhcyB0aGUgZG9taW5hbnQgc2lnbmFsJysoc2Vjb25kYXJ5PycsIGZvbGxvd2VkIGJ5IDxlbT4nK3NlY29uZGFyeSsnPC9lbT4nOicnKStlbW9TdHIrJy4nK3N0YXRlU3RyOwogICAgICBpZih0YWdzRWwpIHRhZ3NFbC5pbm5lckhUTUw9dG9wTmFycy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsJysoaT09PTA/JzAuMyc6JzAuMTInKSsnKTtjb2xvcjonKyhpPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC41KScpKyciPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9IGVsc2UgewogICAgICBuYXJFbC5pbm5lckhUTUw9JzxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHgiPk5vIGhpc3RvcmljYWwgZGF0YSBmb3IgdGhpcyBkYXRlLjwvc3Bhbj4nOwogICAgfQoKICB9IGVsc2UgewogICAgLy8gU2ltdWxhdGVkIOKAlCBidWlsZCBmcm9tIGN1cnJlbnQgU0Qgc2NhbGVkIGJ5IHBvc2l0aW9uCiAgICB2YXIgbmM9e30sZWM9e307CiAgICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgICB2YXIgc2NhbGU9MC4zNStwb3MqMC42NTsKICAgICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7bmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWwqc2NhbGU7fSk7CiAgICAgIGlmKHMuZG9taW5hbnRfZW1vdGlvbikgZWNbcy5kb21pbmFudF9lbW90aW9uXT0oZWNbcy5kb21pbmFudF9lbW90aW9uXXx8MCkrMTsKICAgIH0pOwogICAgdmFyIHRvcE49T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pLnNsaWNlKDAsMyk7CiAgICB2YXIgdG9wRT1PYmplY3QuZW50cmllcyhlYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgICB2YXIgZXBhbDI9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgaWYodG9wTi5sZW5ndGgpewogICAgICB2YXIgcD10b3BOWzBdWzBdLHMyPXRvcE5bMV0/dG9wTlsxXVswXTpudWxsOwogICAgICB2YXIgZVN0cj10b3BFPycg4oCUIHRvbmU6IDxlbSBzdHlsZT0iY29sb3I6JytlcGFsMlt0b3BFWzBdXSsnIj4nK3RvcEVbMF0rJzwvZW0+JzonJzsKICAgICAgbmFyRWwuaW5uZXJIVE1MPSc8ZW0+JytwLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3Auc2xpY2UoMSkrJzwvZW0+IHNpZ25hbHMgZG9taW5hbnQnKyhzMj8nLCB3aXRoIDxlbT4nK3MyKyc8L2VtPiByaXNpbmcnOicnKStlU3RyKycuIDxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zaXplOjExcHgiPihzaW11bGF0ZWQpPC9zcGFuPic7CiAgICAgIGlmKHRhZ3NFbCkgdGFnc0VsLmlubmVySFRNTD10b3BOLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwnKyhpPT09MD8nMC4zJzonMC4xMicpKycpO2NvbG9yOicrKGk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjUpJykrJyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nOwogICAgICB9KS5qb2luKCcnKTsKICAgIH0KICB9CgogIGlmKG1ldGFFbCkgbWV0YUVsLnRleHRDb250ZW50PXR5cGVvZiB0YXJnZXREYXRlPT09J29iamVjdCc/Zm10RGF0ZSh0YXJnZXREYXRlKSsnIHNuYXBzaG90IMK3ICcrKEhJU1RPUllfTE9BREVEPydyZWFsIGRhdGEnOidzaW11bGF0ZWQnKTonJzsKfQpmdW5jdGlvbiB0b2dnbGVSZXBsYXkoKXsKICByZXBsYXlQbGF5aW5nPSFyZXBsYXlQbGF5aW5nOwogIHZhciBpY29uPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1wbGF5LWljb24nKTsKICBpZihyZXBsYXlQbGF5aW5nKXtpZihyZXBsYXlQb3M+PTAuOTkpc2V0UmVwbGF5UG9zKDApO2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzMsMiA3LDIgNyw4IDMsOCBNOCwyIDEyLDIgMTIsOCA4LDgnKTtydW5SZXBsYXkoKTt9CiAgZWxzZXtpZihpY29uKWljb24uc2V0QXR0cmlidXRlKCdwb2ludHMnLCcyLDEgOSw1IDIsOScpO2NsZWFySW50ZXJ2YWwocmVwbGF5VGltZXIpO2FwcGx5TGF5ZXIoKTt9Cn0KZnVuY3Rpb24gcnVuUmVwbGF5KCl7CiAgY2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7CiAgcmVwbGF5VGltZXI9c2V0SW50ZXJ2YWwoZnVuY3Rpb24oKXsKICAgIHJlcGxheVBvcys9MC4wMDMqcmVwbGF5U3BlZWQ7CiAgICBpZihyZXBsYXlQb3M+PTEpe3JlcGxheVBvcz0xO3NldFJlcGxheVBvcygxKTtyZXBsYXlQbGF5aW5nPWZhbHNlO3ZhciBpY29uPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1wbGF5LWljb24nKTtpZihpY29uKWljb24uc2V0QXR0cmlidXRlKCdwb2ludHMnLCcyLDEgOSw1IDIsOScpO2NsZWFySW50ZXJ2YWwocmVwbGF5VGltZXIpO2FwcGx5TGF5ZXIoKTtyZXR1cm47fQogICAgc2V0UmVwbGF5UG9zKHJlcGxheVBvcyk7CiAgfSw2MCk7Cn0KKGZ1bmN0aW9uKCl7dmFyIHRyYWNrPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC10cmFjaycpO2lmKCF0cmFjaylyZXR1cm47dmFyIGRyYWc9ZmFsc2U7CnRyYWNrLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlZG93bicsZnVuY3Rpb24oZSl7ZHJhZz10cnVlO3ZhciByZWN0PXRyYWNrLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpO3NldFJlcGxheVBvcygoZS5jbGllbnRYLXJlY3QubGVmdCkvcmVjdC53aWR0aCk7fSk7CmRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsZnVuY3Rpb24oZSl7aWYoIWRyYWcpcmV0dXJuO3ZhciByZWN0PXRyYWNrLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpO3NldFJlcGxheVBvcygoZS5jbGllbnRYLXJlY3QubGVmdCkvcmVjdC53aWR0aCk7fSk7CmRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNldXAnLGZ1bmN0aW9uKCl7aWYoZHJhZyl7ZHJhZz1mYWxzZTtpZighcmVwbGF5UGxheWluZylhcHBseUxheWVyKCk7fX0pO30pKCk7CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1idG4nKS5mb3JFYWNoKGZ1bmN0aW9uKGJ0bil7YnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe2RvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1idG4nKS5mb3JFYWNoKGZ1bmN0aW9uKGIpe2IuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7YnRuLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlcGxheVBlcmlvZD1idG4uZGF0YXNldC5wZXJpb2Q7cmVwbGF5UG9zPTA7bGFzdFNuYXBQb3M9LTE7dmFyIGRheXM9UkVQTEFZX1BFUklPRFNbcmVwbGF5UGVyaW9kXS5kYXlzO2xvYWRIaXN0b3J5RGF0YShNYXRoLm1pbihkYXlzLDMwKSkudGhlbihmdW5jdGlvbigpe2luaXRSZXBsYXkoKTt9KTt9KTt9KTsKLy8gTG9hZCBpbml0aWFsIGhpc3RvcnkKbG9hZEhpc3RvcnlEYXRhKDcpOwpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucnAtc3BkJykuZm9yRWFjaChmdW5jdGlvbihidG4pe2J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXtkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucnAtc3BkJykuZm9yRWFjaChmdW5jdGlvbihiKXtiLmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pO2J0bi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtyZXBsYXlTcGVlZD1wYXJzZUludChidG4uZGF0YXNldC5zcGQpO30pO30pOwppbml0UmVwbGF5KCk7CnNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAvLyBBdXRvLXNlbGVjdCBob3R0ZXN0IHN0YXRlIGZyb20gTElWRSBkYXRhCiAgdmFyIHNyYz1PYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg/TElWRTpTRDsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKHNyYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKXsKICAgIHZhciBlbD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK3RvcFswXSsnIl0nKTsKICAgIGlmKGVsKSBzZWxlY3RfKHRvcFswXSk7CiAgfQp9LDMwMDApOwpzZXRUaW1lb3V0KHJlbmRlckZhdnMsMjQwMCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"

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
