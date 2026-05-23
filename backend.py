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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjIycHggMjRweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjA7CiAgZmxleDoxOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjI4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWFjY2VudCk7bWFyZ2luLWJvdHRvbToxMnB4O29wYWNpdHk6MC44Owp9CiNzaWctaW5zaWdodHttYXJnaW4tYm90dG9tOjE0cHh9CiNzaWctdGFnc3tkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjZweDttaW4taGVpZ2h0OjB9CiNzaWctdGFnczplbXB0eXtkaXNwbGF5Om5vbmV9CiNzaWctbWV0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnJnYmEoNjIsNzcsOTYsMC40NSk7bWFyZ2luLXRvcDoxMHB4O2xldHRlci1zcGFjaW5nOjAuMDVlbX0KI3NpZy1tZXRhOmVtcHR5e2Rpc3BsYXk6bm9uZX0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNnB4LDEuNnZ3LDIycHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQouc2ktdGV4dCBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1hY2NlbnQpO2ZvbnQtd2VpZ2h0OjQwMH0KLnNpLXRleHQgc3Ryb25ne2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjpyZ2JhKDI0MCwyMzUsMjI1LDAuOSl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKICBhbGlnbi1pdGVtczpzdGFydDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnN0aWNreTt0b3A6NjBweDsKfQoubWFwLWNhcmQ6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7CiAgYmFja2dyb3VuZDoKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDcwJSA1MCUgYXQgMzUlIDAlLCByZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSwgdHJhbnNwYXJlbnQgNjAlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDUwJSA0MCUgYXQgODAlIDEwMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAwJSwgdHJhbnNwYXJlbnQgNjAlKTsKfQoubWFwLXRvcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjEycHggMThweCAwOwp9Ci5tYXAtdGl0bGUtYmxvY2sgLm10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTdweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLm1hcC10aXRsZS1ibG9jayAubXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO21hcmdpbi10b3A6MnB4fQoubGVnZW5ke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmxlZ2VuZC1iYXJ7CiAgaGVpZ2h0OjNweDt3aWR0aDo4MHB4O2JvcmRlci1yYWRpdXM6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIHJpZ2h0LCMwZTIwMzUsIzFhNTU4MCAyNSUsIzhhNWMxOCA1NSUsI2MwMzgxYSA4MCUsI2UwMTAyMCk7Cn0KLmxheWVyLXJvd3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OwogIHBhZGRpbmc6MTBweCAyMHB4IDZweDsKfQoubGF5ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQoubHRhYnN7ZGlzcGxheTpmbGV4O2dhcDozcHh9Ci5sdGFiewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzozcHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLmx0YWIuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA4KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKX0KLmx0YWJ7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjVweDtwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzp2aXNpYmxlfQoubHRhYi1pbmZve3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMik7Zm9udC1zaXplOjhweDtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXN0eWxlOml0YWxpYztmb250LXdlaWdodDo2MDA7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjM1KTtkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2N1cnNvcjpoZWxwO2ZsZXgtc2hyaW5rOjA7dHJhbnNpdGlvbjphbGwgMC4xNXM7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxMDB9Ci5sdGFiLWluZm86aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWFjY2VudCk7Y29sb3I6dmFyKC0tYWNjZW50KX0KI2x0YWItdG9vbHRpcHtwb3NpdGlvbjpmaXhlZDtiYWNrZ3JvdW5kOnJnYmEoOCwxMiwyMCwwLjk4KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4xMik7Ym9yZGVyLXJhZGl1czo4cHg7cGFkZGluZzoxMHB4IDEzcHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42O3dpZHRoOjIzMHB4O3doaXRlLXNwYWNlOm5vcm1hbDt0ZXh0LWFsaWduOmxlZnQ7Ym94LXNoYWRvdzowIDhweCAzMnB4IHJnYmEoMCwwLDAsMC42KTtwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7ei1pbmRleDo5OTk5OTtkaXNwbGF5Om5vbmV9CiNsdGFiLXRvb2x0aXAudmlzaWJsZXtvcGFjaXR5OjE7ZGlzcGxheTpibG9ja30KLmx0YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KCi5tYXAtc3ZnLXdyYXB7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZzoxMnB4IDE2cHggMTZweDsKfQoubWFwLWlubmVye3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxLzE7d2lkdGg6MTAwJX0KI2luZGlhLW1hcHt3aWR0aDoxMDAlO2hlaWdodDoxMDAlO2Rpc3BsYXk6YmxvY2s7b3ZlcmZsb3c6dmlzaWJsZX0KCi8qIG1hcCBzdGF0ZSBzdHlsZXMgKi8KI2luZGlhLW1hcCAuc3RhdGV7CiAgY3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpmaWx0ZXIgMC4yNXMgZWFzZSwgc3Ryb2tlLXdpZHRoIDAuMnMgZWFzZSwgc3Ryb2tlIDAuMnMgZWFzZTsKfQojaW5kaWEtbWFwIC5zdGF0ZTpob3ZlcnsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjcpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjFweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMjUpIGRyb3Atc2hhZG93KDAgMCAxMHB4IHJnYmEoMjU1LDI1NSwyNTUsMC4yKSk7Cn0KI2luZGlhLW1hcCAuc3RhdGUuc2VsZWN0ZWR7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC45KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjRweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMzUpIGRyb3Atc2hhZG93KDAgMCAxNnB4IHJnYmEoMjU1LDI1NSwyNTUsMC4zKSk7Cn0KCi8qIGFuaW1hdGVkIHB1bHNlIHJpbmdzICovCi5wdWxzZS1yaW5ne2ZpbGw6bm9uZTtwb2ludGVyLWV2ZW50czpub25lfQoucHVsc2UtcmluZy5wMXthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCBpbmZpbml0ZX0KLnB1bHNlLXJpbmcucDJ7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgMC45cyBpbmZpbml0ZX0KQGtleWZyYW1lcyBwcnsKICAwJXtyOjQ7b3BhY2l0eTowLjc7c3Ryb2tlLXdpZHRoOjEuMn0KICAxMDAle3I6MjY7b3BhY2l0eTowO3N0cm9rZS13aWR0aDowLjJ9Cn0KCi8qIGF0bW9zcGhlcmljIGdsb3cgYmVoaW5kIGhvdCBzdGF0ZXMgKi8KLnN0YXRlLWdsb3d7cG9pbnRlci1ldmVudHM6bm9uZTtmaWxsOm5vbmV9CkBrZXlmcmFtZXMgZ2xvd1B1bHNlezAlLDEwMCV7b3BhY2l0eTowLjEyfTUwJXtvcGFjaXR5OjAuMjJ9fQoKLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDUsNywxMiwwLjk1KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6OXB4OwogIHBhZGRpbmc6MTJweCAxNHB4O29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xMnM7ei1pbmRleDo5OTk5O21pbi13aWR0aDoxNzBweDsKfQoudHQtbntmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO21hcmdpbi1ib3R0b206OHB4O2NvbG9yOnZhcigtLWluayl9Ci50dC1ye2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweH0KLnR0LXIgc3Ryb25ne2NvbG9yOnZhcigtLWluayl9Ci50dC1uYXJ7CiAgbWFyZ2luLXRvcDo4cHg7cGFkZGluZy10b3A6OHB4O2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7Cn0KLnR0LW5hciBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtkaXNwbGF5OmJsb2NrO21hcmdpbi1ib3R0b206MnB4fQoKLyogU1RBVEUgUEFORUwgKi8KLnN0YXRlLXBhbmVsewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBwYWRkaW5nOjIwcHg7b3ZlcmZsb3cteTphdXRvOwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwogIHBvc2l0aW9uOnN0aWNreTt0b3A6NjBweDsKICBtYXgtaGVpZ2h0OmNhbGMoMTAwdmggLSA4MHB4KTsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjNweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7CiAgaGVpZ2h0OjEwMCU7bWluLWhlaWdodDozMjBweDt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjMycHggMjBweDsKfQoucGFuZWwtZW1wdHkgc3Zne29wYWNpdHk6MC4xNTttYXJnaW4tYm90dG9tOjE4cHh9Ci5wYW5lbC1lbXB0eSAucGUtdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjhweH0KLnBhbmVsLWVtcHR5IC5wZS1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW07bGluZS1oZWlnaHQ6MS43fQoKLyogc3RhdGUgcGFuZWwgaW50ZXJuYWxzICovCi5zcC1oZWFkewogIGRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIG1hcmdpbi1ib3R0b206MTZweDtwYWRkaW5nLWJvdHRvbToxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnNwLWVre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTtjb2xvcjp2YXIoLS1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NXB4fQouc3AtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI4cHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07bGluZS1oZWlnaHQ6MTtjb2xvcjp2YXIoLS1pbmspfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Y29sb3I6dmFyKC0tZmFpbnQpOwogIHdpZHRoOjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo2cHg7Y3Vyc29yOnBvaW50ZXI7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMThzO3BhZGRpbmc6MDtmbGV4LXNocmluazowOwp9Ci5mYXYtYnRuOmhvdmVye2NvbG9yOnZhcigtLWRpbSk7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5mYXYtYnRuLm9ue2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHh9CgovKiBuYXJyYXRpdmUgdGltZWxpbmUg4oCUIHRoZSBzaWduYXR1cmUgZmVhdHVyZSAqLwoubmFyLXRpbWVsaW5lewogIG1hcmdpbi1ib3R0b206MTZweDsKfQoubnQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLm50LWZsb3d7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nLWxlZnQ6MTZweDsKfQoubnQtZmxvdzo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDo1cHg7dG9wOjZweDtib3R0b206NnB4O3dpZHRoOjFweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx2YXIoLS1ib3JkZXIpKTtvcGFjaXR5OjAuNDsKfQoubnQtc3RlcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweDsKICBwYWRkaW5nOjVweCAwO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5udC1kb3R7CiAgd2lkdGg6MTBweDtoZWlnaHQ6MTBweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4LXNocmluazowOwogIHBvc2l0aW9uOmFic29sdXRlO2xlZnQ6LTE2cHg7dG9wOjdweDsKICBib3JkZXI6MS41cHggc29saWQgY3VycmVudENvbG9yO2JhY2tncm91bmQ6dmFyKC0tYmcpOwp9Ci5udC1zdGVwLnBhc3QgLm50LWRvdHtjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCl9Ci5udC1zdGVwLmN1cnJlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCAxMHB4IHJnYmEoMjI0LDkwLDQwLDAuNSl9Ci5udC1jb250ZW50e2ZsZXg6MX0KLm50LXRvcGlje2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4zfQoubnQtc3RlcC5wYXN0IC5udC10b3BpY3tjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZGltKX0KLm50LXdoZW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9CgovKiBpbnNpZ2h0IGJsb2NrICovCi5pbnNpZ2h0ewogIG1hcmdpbi1ib3R0b206MTRweDsKICBwYWRkaW5nOjEycHggMTRweCAxMnB4IDE2cHg7CiAgYm9yZGVyLWxlZnQ6MS41cHggc29saWQgdmFyKC0tYWNjZW50KTsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDMpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxMy41cHg7Zm9udC1zdHlsZTppdGFsaWM7CiAgY29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU1O2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLyogY29tcGFjdCBzY29yZSBzdHJpcCAqLwouc2NvcmUtc3RyaXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsKICBwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6N3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgbWFyZ2luLWJvdHRvbToxNHB4Owp9Ci5zcy1pdGVte2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjJweH0KLnNzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5zcy12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2NvbG9yOnZhcigtLWluayl9Ci5zcy1kZWx0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjJweCA3cHg7Ym9yZGVyLXJhZGl1czozcHh9Ci5zcy1kZWx0YS51cHtjb2xvcjojZTA2MDMwO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKX0KLnNzLWRlbHRhLmRue2NvbG9yOiMzYmI4ZDg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKX0KLnNzLWRpdmlkZXJ7d2lkdGg6MXB4O2hlaWdodDozMnB4O2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowfQouc3MtbmFye2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDB9Cgouc3Atc2VjdGlvbnttYXJnaW4tYm90dG9tOjE0cHh9Ci5zcC1zZWMtdGl0bGV7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206OXB4Owp9CgovKiBuYXJyYXRpdmVzICovCi5uYXItbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9Ci5uYXItaXRlbTJ7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgYXV0bztnYXA6NnB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLm5pLWxhYmVse2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKX0KLm5pLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5uaS10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi10b3A6LTNweH0KLm5pLWZpbGx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHg7dHJhbnNpdGlvbjp3aWR0aCAwLjdzfQoKLyogbW92ZW1lbnQgKi8KLm12LWdyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDo3cHh9Ci5tdi1ibG9ja3tiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6N3B4O3BhZGRpbmc6OXB4fQoubXYtaHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206N3B4fQoubXYtYmxvY2sudXAgLm12LWh7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tdi1ibG9jay5kbiAubXYtaHtjb2xvcjp2YXIoLS1mYWxsKX0KLm12LWl0e2ZvbnQtc2l6ZToxMC41cHg7cGFkZGluZzo0cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2NvbG9yOnZhcigtLWZhaW50KX0KLm12LWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tdi1pdCBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTFweH0KLm12LWl0IHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweH0KCi8qIGVtb3Rpb24gKi8KLmVtLXJvd3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQouZW0tZG9udXR7d2lkdGg6NzZweDtoZWlnaHQ6NzZweDtmbGV4LXNocmluazowfQouZW0tbGVne2ZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHh9Ci5lbS1pdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVtLXN3e3dpZHRoOjZweDtoZWlnaHQ6NnB4O2JvcmRlci1yYWRpdXM6MnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1ue2ZsZXg6MTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5lbS1we2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmspfQoKLyogdGltZWxpbmUgY2hhcnQgKi8KLnRsLXdyYXB7aGVpZ2h0OjcycHh9CgovKiBhcnRpY2xlcyAqLwouYXJ0LWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4fQouYXJ0LWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzo3cHggOXB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydC1pdGVtOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5hcnQtc3Jje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO3dpZHRoOjQ0cHg7cGFkZGluZy10b3A6MXB4fQouYXJ0LXR4dHtmb250LXNpemU6MTFweDtsaW5lLWhlaWdodDoxLjQ7Y29sb3I6dmFyKC0tZGltKX0KCi8qIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XICovCi5uYXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjE4cHg7Cn0KLm5hci1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNHB4KTtvdmVyZmxvdzpoaWRkZW47CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKfQoubmMtaGVhZHsKICBwYWRkaW5nOjE2cHggMjBweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4Owp9Ci5uYy1ib2R5e3BhZGRpbmc6OHB4IDIwcHggMTZweDtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO30KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAzNnB4IDYwcHg7bWF4LXdpZHRoOjU4MHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZm9vdC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luLWJvdHRvbToxNHB4fQouZm9vdC1saW5le2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS44O21hcmdpbi1ib3R0b206MTJweH0KLmZvb3Qtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpfQoKLyogYW5pbWF0aW9ucyAqLwpAa2V5ZnJhbWVzIGZhZGVVcHtmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWSg2cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci5tYXAtY2FyZCwuc3RhdGUtcGFuZWwsLm5hci1jYXJkLC5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb246ZmFkZVVwIDAuNTVzIGN1YmljLWJlemllciguMiwuOCwuMiwxKSBiYWNrd2FyZHN9Ci5uYXItY2FyZDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMDdzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDMpe2FuaW1hdGlvbi1kZWxheTowLjE0c30KLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbi1kZWxheTowLjA1c30KCkBtZWRpYShtYXgtd2lkdGg6MTEwMHB4KXsKICAubWFpbntncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQogIC5zdGF0ZS1wYW5lbHttYXgtaGVpZ2h0Om5vbmV9CiAgLm5hci1yb3d7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KfQoKCi8qIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAogICBNT0JJTEUgU1RZTEVTIOKAlCBAbWVkaWEgbWF4LXdpZHRoOjc2OHB4CiAgIFRoZXNlIHJ1bGVzIE9OTFkgYXBwbHkgb24gbW9iaWxlLiBEZXNrdG9wIGlzIGNvbXBsZXRlbHkgdW50b3VjaGVkLgogICDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAgKi8KCkBtZWRpYSAobWF4LXdpZHRoOiA3NjhweCkgewoKICAvKiDilIDilIAgVE9QQkFSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogIC50b3BiYXJ7CiAgICBwYWRkaW5nOjAgMTZweDsKICAgIGhlaWdodDo1MnB4OwogIH0KICAuYnJhbmQtdGFnbGluZXtkaXNwbGF5Om5vbmV9CiAgLmJyYW5kLW5hbWV7Zm9udC1zaXplOjEzcHh9CiAgLnRvcGJhci1yIC5saXZlLWRvdC13cmFwIHNwYW46bGFzdC1jaGlsZHtkaXNwbGF5Om5vbmV9IC8qIGhpZGUgInNpZ25hbHMiIHRleHQgKi8KICAjbGl2ZS1jb3VudHtmb250LXNpemU6MTBweH0KCiAgLyog4pSA4pSAIEhFUk8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLmhlcm97CiAgICBwYWRkaW5nOjcycHggMjBweCAyMHB4ICFpbXBvcnRhbnQ7CiAgICB0ZXh0LWFsaWduOmNlbnRlcjsKICB9CiAgLmhlcm8tZXllYnJvd3tqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2ZvbnQtc2l6ZTo4cHg7bWFyZ2luLWJvdHRvbToxNnB4fQogIC5oZXJvLWV5ZWJyb3c6OmJlZm9yZXtkaXNwbGF5Om5vbmV9CiAgLmhlcm8tYnJhbmQtYmxvY2t7anVzdGlmeS1jb250ZW50OmNlbnRlcjtnYXA6MTJweDttYXJnaW4tYm90dG9tOjEycHh9CiAgLmhlcm8tYnJhbmQtbmFtZXtmb250LXNpemU6Y2xhbXAoMzJweCw5dncsNDhweCkgIWltcG9ydGFudH0KICAuaGVyby10YWdsaW5lewogICAgZm9udC1zaXplOjE1cHggIWltcG9ydGFudDsKICAgIG1heC13aWR0aDoxMDAlOwogICAgdGV4dC1hbGlnbjpjZW50ZXI7CiAgICBtYXJnaW46MCBhdXRvIDEwcHggYXV0bzsKICB9CiAgLmhlcm8tZGVzY3sKICAgIGZvbnQtc2l6ZToxMnB4OwogICAgbWF4LXdpZHRoOjEwMCU7CiAgICB0ZXh0LWFsaWduOmNlbnRlcjsKICAgIG1hcmdpbjowIGF1dG8gNnB4IGF1dG87CiAgfQogIC5oZXJvLXN1Yi1saW5le3RleHQtYWxpZ246Y2VudGVyfQoKICAvKiBTdGF0cyBzdHJpcCDigJQgMiBjb2x1bW5zIG9uIG1vYmlsZSAqLwogIC5zdGF0cy1zdHJpcHsKICAgIGRpc3BsYXk6Z3JpZCAhaW1wb3J0YW50OwogICAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7CiAgICBib3JkZXItcmFkaXVzOjEwcHg7CiAgICBtYXJnaW4tYm90dG9tOjE2cHg7CiAgfQogIC5zYy1kaXZpZGVye2Rpc3BsYXk6bm9uZX0KICAuc2MtaXRlbXsKICAgIHBhZGRpbmc6MTJweCAxNHB4OwogICAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICAgIGJvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICB9CiAgLnNjLWl0ZW06bnRoLWNoaWxkKDJuKXtib3JkZXItcmlnaHQ6bm9uZX0KICAuc2MtaXRlbTpudGgtbGFzdC1jaGlsZCgtbisyKXtib3JkZXItYm90dG9tOm5vbmV9CiAgLnNjLXZhbHtmb250LXNpemU6MThweCAhaW1wb3J0YW50fQoKICAvKiBTaWduYXR1cmUgaW5zaWdodCArIG5hcnJhdGl2ZSBzdHJpcCDigJQgc3RhY2sgdmVydGljYWxseSAqLwogIC5oZXJvID4gZGl2W3N0eWxlKj0iZGlzcGxheTpmbGV4Il1bc3R5bGUqPSJnYXA6MThweCJdewogICAgZmxleC1kaXJlY3Rpb246Y29sdW1uICFpbXBvcnRhbnQ7CiAgICBnYXA6MTRweCAhaW1wb3J0YW50OwogICAgcGFkZGluZzowICFpbXBvcnRhbnQ7CiAgICBtYXJnaW4tdG9wOjEycHggIWltcG9ydGFudDsKICB9CiAgLnNpZ25hdHVyZS1pbnNpZ2h0e21hcmdpbi10b3A6MCAhaW1wb3J0YW50fQogIC5zaS10ZXh0e2ZvbnQtc2l6ZToxNHB4ICFpbXBvcnRhbnR9CgogIC8qIE5hcnJhdGl2ZSBzaGlmdHMgcGFuZWwg4oCUIGhpZGUgb24gbW9iaWxlIChzaG93biBiZWxvdyBtYXAgaW5zdGVhZCkgKi8KICAuc2hpZnQtcGFuZWx7ZGlzcGxheTpub25lfQoKICAvKiDilIDilIAgTUFQIFNFQ1RJT04g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLm1haW57CiAgICBkaXNwbGF5OmZsZXggIWltcG9ydGFudDsKICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbiAhaW1wb3J0YW50OwogICAgcGFkZGluZzowIDEycHggMjBweCAhaW1wb3J0YW50OwogICAgZ2FwOjAgIWltcG9ydGFudDsKICB9CgogIC8qIE1hcCBjYXJkIOKAlCBmdWxsIHdpZHRoLCBubyBib3JkZXIgcmFkaXVzIG9uIHNpZGVzICovCiAgLm1hcC1jYXJkewogICAgcG9zaXRpb246cmVsYXRpdmUgIWltcG9ydGFudDsKICAgIHRvcDphdXRvICFpbXBvcnRhbnQ7CiAgICBib3JkZXItcmFkaXVzOjE0cHg7CiAgICBtYXJnaW4tYm90dG9tOjA7CiAgfQoKICAvKiBMYXllciB0YWJzIOKAlCBjb21wYWN0ICovCiAgLmx0YWJze2dhcDo0cHg7cGFkZGluZzoxMHB4IDEycHh9CiAgLmx0YWJ7CiAgICBmb250LXNpemU6OHB4OwogICAgcGFkZGluZzo0cHggOHB4OwogICAgbGV0dGVyLXNwYWNpbmc6MC4wNmVtOwogIH0KICAubHRhYi1pbmZve2Rpc3BsYXk6bm9uZX0gLyogaGlkZSBpIGJ1dHRvbnMgb24gbW9iaWxlIOKAlCB0b29sdGlwIHVudXNhYmxlICovCiAgLm1hcC1sZWdlbmR7cGFkZGluZzo4cHggMTJweH0KICAjbWFwLW1ldGF7Zm9udC1zaXplOjhweH0KCiAgLyog4pSA4pSAIFNUQVRFIFBBTkVMIOKAlCBiZWNvbWVzIGJvdHRvbSBkcmF3ZXIgb24gbW9iaWxlIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogIC5zdGF0ZS1wYW5lbHsKICAgIHBvc2l0aW9uOmZpeGVkICFpbXBvcnRhbnQ7CiAgICBib3R0b206MCAhaW1wb3J0YW50OwogICAgbGVmdDowICFpbXBvcnRhbnQ7CiAgICByaWdodDowICFpbXBvcnRhbnQ7CiAgICB0b3A6YXV0byAhaW1wb3J0YW50OwogICAgbWF4LWhlaWdodDo3MnZoICFpbXBvcnRhbnQ7CiAgICBib3JkZXItcmFkaXVzOjIwcHggMjBweCAwIDAgIWltcG9ydGFudDsKICAgIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKSAhaW1wb3J0YW50OwogICAgYm9yZGVyLWJvdHRvbTpub25lICFpbXBvcnRhbnQ7CiAgICB6LWluZGV4OjgwMDAgIWltcG9ydGFudDsKICAgIHRyYW5zZm9ybTp0cmFuc2xhdGVZKDEwMCUpICFpbXBvcnRhbnQ7CiAgICB0cmFuc2l0aW9uOnRyYW5zZm9ybSAwLjM1cyBjdWJpYy1iZXppZXIoMC4zMiwwLjcyLDAsMSkgIWltcG9ydGFudDsKICAgIHBhZGRpbmc6MCAhaW1wb3J0YW50OwogICAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjBweCkgIWltcG9ydGFudDsKICAgIGJhY2tncm91bmQ6cmdiYSg5LDEzLDIxLDAuOTcpICFpbXBvcnRhbnQ7CiAgICBvdmVyZmxvdy15OmF1dG87CiAgfQogIC8qIERyYWcgaGFuZGxlICovCiAgLnN0YXRlLXBhbmVsOjpiZWZvcmV7CiAgICBjb250ZW50OicnOwogICAgZGlzcGxheTpibG9jazsKICAgIHdpZHRoOjM2cHg7aGVpZ2h0OjRweDsKICAgIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjE1KTsKICAgIGJvcmRlci1yYWRpdXM6MnB4OwogICAgbWFyZ2luOjEycHggYXV0byA4cHggYXV0bzsKICAgIGZsZXgtc2hyaW5rOjA7CiAgfQogIC8qIFNob3cgcGFuZWwgd2hlbiBzdGF0ZSBzZWxlY3RlZCAqLwogIC5zdGF0ZS1wYW5lbC5wYW5lbC1vcGVuewogICAgdHJhbnNmb3JtOnRyYW5zbGF0ZVkoMCkgIWltcG9ydGFudDsKICB9CiAgI3N0YXRlLWRldGFpbHtwYWRkaW5nOjAgMThweCAzMnB4fQogIC5zcC1oZWFke3BhZGRpbmc6NHB4IDAgMTJweH0KICAuc3AtbmFtZXtmb250LXNpemU6MjJweH0KCiAgLyogRGltIG92ZXJsYXkgd2hlbiBwYW5lbCBvcGVuICovCiAgLm1hcC1vdmVybGF5LWRpbXsKICAgIGRpc3BsYXk6bm9uZTsKICAgIHBvc2l0aW9uOmZpeGVkO2luc2V0OjA7CiAgICBiYWNrZ3JvdW5kOnJnYmEoMCwwLDAsMC40KTsKICAgIHotaW5kZXg6Nzk5OTsKICAgIGFuaW1hdGlvbjpmYWRlSW4gMC4ycyBlYXNlOwogIH0KICAubWFwLW92ZXJsYXktZGltLmFjdGl2ZXtkaXNwbGF5OmJsb2NrfQogIEBrZXlmcmFtZXMgZmFkZUlue2Zyb217b3BhY2l0eTowfXRve29wYWNpdHk6MX19CgogIC8qIOKUgOKUgCBOQVJSQVRJVkUgQ0FSRFMgKGJlbG93IG1hcCkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLm5hci1yb3d7CiAgICBkaXNwbGF5OmZsZXggIWltcG9ydGFudDsKICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbiAhaW1wb3J0YW50OwogICAgcGFkZGluZzoxNnB4IDEycHggIWltcG9ydGFudDsKICAgIGdhcDoxMnB4ICFpbXBvcnRhbnQ7CiAgfQogIC5uYXItY2FyZHtib3JkZXItcmFkaXVzOjEycHh9CiAgLm5jLWhlYWR7cGFkZGluZzoxMnB4IDE2cHh9CiAgLm5jLWJvZHl7cGFkZGluZzo0cHggMTZweCAxNHB4fQogIC5uYy10aXRsZXtmb250LXNpemU6MTRweH0KCiAgLyog4pSA4pSAIFJFUExBWSBJTkRJQSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KICAucmVwbGF5LXNlY3Rpb257cGFkZGluZzowIDEycHggMjRweH0KICAucmVwbGF5LWhlYWRlcntmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweH0KICAucmVwbGF5LWNvbnRyb2xze3dpZHRoOjEwMCU7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW59CiAgLnJwLWJ0bntmbGV4OjE7dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzo2cHggNHB4O2ZvbnQtc2l6ZTo4cHh9CiAgLnJlcGxheS1zbmFwc2hvdHtkaXNwbGF5Om5vbmV9IC8qIGhpZGUgc3RhdGUgY2FyZHMgb24gbW9iaWxlICovCgogIC8qIOKUgOKUgCBGT09URVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgLmZvb3R7cGFkZGluZzozMnB4IDIwcHggNDhweH0KICAuZm9vdC1uYW1le2ZvbnQtc2l6ZToxM3B4fQogIC5mb290LWxpbmV7Zm9udC1zaXplOjExcHh9CgogIC8qIOKUgOKUgCBMT0FERVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgI2FwcC1sb2FkZXIgPiBkaXY6Zmlyc3Qtb2YtdHlwZXttYXJnaW4tYm90dG9tOjI0cHh9Cgp9Ci8qIEVORCBNT0JJTEUgKi8KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9Imx0YWItdG9vbHRpcCI+PC9kaXY+Cgo8IS0tIExPQURFUiAtLT4KPGRpdiBpZD0iYXBwLWxvYWRlciIgc3R5bGU9IgogIHBvc2l0aW9uOmZpeGVkO2luc2V0OjA7ei1pbmRleDo5OTk5ODsKICBiYWNrZ3JvdW5kOiMwNjA5MTA7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBnYXA6MDsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC44cyBlYXNlLCB2aXNpYmlsaXR5IDAuOHMgZWFzZTsKIj4KICA8IS0tIFNpZ25hbCByaW5ncyAtLT4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt3aWR0aDo2NHB4O2hlaWdodDo2NHB4O21hcmdpbi1ib3R0b206MzZweCI+CiAgICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDoyNHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6I2UwNWEyODthbmltYXRpb246bGRyUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNSk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuNXMiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6LTEwcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjA3KTthbmltYXRpb246bGRyUmluZyAycyBlYXNlLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MXMiPjwvZGl2PgogIDwvZGl2PgoKICA8IS0tIEJyYW5kIC0tPgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidQbGF5ZmFpciBEaXNwbGF5JyxHZW9yZ2lhLHNlcmlmO2ZvbnQtc2l6ZTpjbGFtcCgyOHB4LDV2dyw0MnB4KTtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtjb2xvcjojZjBlY2U0O2xpbmUtaGVpZ2h0OjE7bWFyZ2luLWJvdHRvbToxMHB4Ij4KICAgIDxlbSBzdHlsZT0iY29sb3I6I2U4YzRhMDtmb250LXN0eWxlOml0YWxpYyI+UHVsc2U8L2VtPiBvZiBJbmRpYQogIDwvZGl2PgoKICA8IS0tIFRhZ2xpbmUgLS0+CiAgPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6J0NvdXJpZXIgTmV3Jyxtb25vc3BhY2U7Zm9udC1zaXplOjExcHg7bGV0dGVyLXNwYWNpbmc6MC4yOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuNCk7bWFyZ2luLWJvdHRvbToyOHB4Ij4KICAgIFRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMKICA8L2Rpdj4KCiAgPCEtLSBOb3QgbmV3cyBsaW5lIC0tPgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjI1KTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4Ij4KICAgIDxzcGFuPk5vdCBuZXdzPC9zcGFuPgogICAgPHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPk5vdCBwcmVkaWN0aW9uPC9zcGFuPgogICAgPHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPkp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KTthbmltYXRpb246bGRyR2xvdyAycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+b2JzZXJ2YXRpb248L3NwYW4+PC9zcGFuPgogIDwvZGl2PgoKICA8IS0tIExvYWRpbmcgZG90cyAtLT4KICA8ZGl2IHN0eWxlPSJtYXJnaW4tdG9wOjQ4cHg7ZGlzcGxheTpmbGV4O2dhcDo2cHgiPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjJzIj48L3NwYW4+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MC40cyI+PC9zcGFuPgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KQGtleWZyYW1lcyBsZHJQdWxzZXswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC41O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KQGtleWZyYW1lcyBsZHJSaW5nezAle3RyYW5zZm9ybTpzY2FsZSgwLjgpO29wYWNpdHk6MC42fTEwMCV7dHJhbnNmb3JtOnNjYWxlKDEuNSk7b3BhY2l0eTowfX0KQGtleWZyYW1lcyBsZHJHbG93ezAlLDEwMCV7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KX01MCV7dGV4dC1zaGFkb3c6MCAwIDIwcHggcmdiYSg1NywyNTUsMjAsMC45KSwwIDAgNDBweCByZ2JhKDU3LDI1NSwyMCwwLjMpfX0KQGtleWZyYW1lcyBsZHJEb3R7MCUsODAlLDEwMCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjN9NDAle3RyYW5zZm9ybTpzY2FsZSgxKTtvcGFjaXR5OjF9fQo8L3N0eWxlPgoKPGRpdiBjbGFzcz0idG9wYmFyIj4KICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC1tYXJrIj48c3BhbiBjbGFzcz0iYnJhbmQtcHVsc2UtZG90Ij48L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC10ZXh0LWJsb2NrIj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLW5hbWUiPjxlbSBjbGFzcz0iYnJhbmQtcHVsc2Utd29yZCI+UHVsc2U8L2VtPiBvZiBJbmRpYTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXIiPgogICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgPHNwYW4gaWQ9ImxpdmUtY291bnQiPuKApjwvc3Bhbj4gc2lnbmFscwogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSIKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7CiAgYmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpOwogIHBhZGRpbmc6MCAzNnB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOwoiPgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLXNpZ25hbHMiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlNpZ25hbHMgdHJhY2tlZDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIj5MaXZlIGluZ2VzdGlvbjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1ob3R0ZXN0IiBzdHlsZT0iY3Vyc29yOnBvaW50ZXIiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWhvdHRlc3QtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj5DbGljayB0byBleHBsb3JlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWFuZ2VyLXN1YiI+T3V0cmFnZSAmIHByb3Rlc3Qgc2lnbmFsczwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlRvcCByaXNpbmcgbmFycmF0aXZlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1uYXJyYXRpdmUtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hcnJhdGl2ZS1zdWIiPk5hdGlvbmFsIHNpZ25hbCBzdXJnZTwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbGluZy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbGluZy1zdWIiPlNpZ25hbCBkZWNheTwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KLnN0YXQtY2VsbHsKICBmbGV4OjE7cGFkZGluZzoxMHB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5zdGF0LWNlbGw6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpfQouc3RhdC1kaXZ7d2lkdGg6MXB4O2JhY2tncm91bmQ6cmdiYSgxNjAsMTkwLDIzMCwwLjA3KTtmbGV4LXNocmluazowO21hcmdpbjo4cHggMH0KLnNjLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW4tdG9wOjFweH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KPC9zdHlsZT4KCgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MjBweCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJzaWduYXR1cmUtaW5zaWdodCIgc3R5bGU9Im1hcmdpbi10b3A6MDtmbGV4OjE7bWluLXdpZHRoOjAiPgogICAgICA8ZGl2IGNsYXNzPSJzaS1sYWJlbCI+SW5kaWEgcmlnaHQgbm93PC9kaXY+CiAgICAgIDxkaXYgaWQ9InNpZy1pbnNpZ2h0Ij4KICAgICAgICA8ZGl2IGNsYXNzPSJzaS10ZXh0IiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtc2l6ZToxNHB4Ij5BbmFseXNpbmcgc2lnbmFscyBhY3Jvc3MgMzAgc3RhdGVzLi4uPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGlkPSJzaWctdGFncyI+PC9kaXY+CiAgICAgIDxkaXYgaWQ9InNpZy1tZXRhIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZmxleDowIDAgMzYwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOyI+CiAgICAgIDwhLS0gaGVhZGVyIC0tPgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsiPgogICAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj5OYXJyYXRpdmUgc2hpZnRzPC9zcGFuPgogICAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MnB4OyI+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIgYWN0aXZlIiBkYXRhLXBlcmlvZD0iM20iPjNNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSI2bSI+Nk08L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9InN0cmlwLXRhYiIgZGF0YS1wZXJpb2Q9IjF5Ij4xWTwvYnV0dG9uPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPCEtLSBzaGlmdHMgbGlzdCAtLT4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxO292ZXJmbG93OmhpZGRlbjtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoxMHB4IDE0cHg7Z2FwOjZweDsiIGlkPSJzaGlmdC1saXN0Ij48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+CgoKPCEtLSBNQUlOOiBNQVAgKyBTVEFURSBQQU5FTCAtLT4KPGRpdiBjbGFzcz0ibWFpbiI+CgogIDxkaXYgY2xhc3M9Im1hcC1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im1hcC10b3AiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtdGl0bGUtYmxvY2siPgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij5JbmRpYSAmbWRhc2g7IGNvbGxlY3RpdmUgYXR0ZW50aW9uPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibXMiIGlkPSJtYXAtbWV0YSI+MzAgc3RhdGVzICZtaWRkb3Q7IGxpdmUgc2lnbmFsIGNvbXBvc2l0ZTwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGVnZW5kIj48c3Bhbj5xdWlldDwvc3Bhbj48ZGl2IGNsYXNzPSJsZWdlbmQtYmFyIj48L2Rpdj48c3Bhbj5hY3RpdmU8L3NwYW4+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxheWVyLXJvdyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsYXllci1sYWJlbCI+Vmlldzwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0ibHRhYnMiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iYXR0ZW50aW9uIj5BdHRlbnRpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IldoaWNoIHN0YXRlcyBhcmUgcmVjZWl2aW5nIHRoZSBtb3N0IHB1YmxpYyBmb2N1cy4gSGlnaCBhdHRlbnRpb24gPSBjb25jZW50cmF0ZWQgbmV3cyBjb3ZlcmFnZSBhbmQgcG9saXRpY2FsIGFjdGl2aXR5LiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImVtb3Rpb24iPkVtb3Rpb24gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IlRoZSBkb21pbmFudCBlbW90aW9uYWwgdG9uZSDigJQgYW54aW91cywgYW5ncnksIGhvcGVmdWwsIHByb3VkIG9yIGZlYXJmdWwuIFJldmVhbHMgdGhlIHBzeWNob2xvZ2ljYWwgdW5kZXJjdXJyZW50IG9mIHBvbGl0aWNhbCBhdHRlbnRpb24uIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiIgZGF0YS1sYXllcj0idmVsb2NpdHkiPk1vbWVudHVtIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJJcyBhdHRlbnRpb24gcmlzaW5nIG9yIGZhbGxpbmc/IFJpc2luZyA9IG5hcnJhdGl2ZSBhY2NlbGVyYXRpbmcuIENvb2xpbmcgPSBsb3NpbmcgdHJhY3Rpb24uIFNob3dzIHN0YXRlcyBlbnRlcmluZyBvciBleGl0aW5nIGEgcG9saXRpY2FsIGN5Y2xlLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJtYXAtc3ZnLXdyYXAiPgogICAgICA8ZGl2IGNsYXNzPSJtYXAtaW5uZXIiPgogICAgICAgIDxzdmcgaWQ9ImluZGlhLW1hcCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgogICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iYW1iR2xvdyIgY3g9IjUwJSIgY3k9IjUwJSIgcj0iNTAlIj4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSJyZ2JhKDIyNCw5MCw0MCwwLjA0KSIvPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0idHJhbnNwYXJlbnQiLz4KICAgICAgICAgICAgPC9yYWRpYWxHcmFkaWVudD4KICAgICAgICAgICAgPGZpbHRlciBpZD0ic3RhdGVHbG93IiB4PSItMzAlIiB5PSItMzAlIiB3aWR0aD0iMTYwJSIgaGVpZ2h0PSIxNjAlIj4KICAgICAgICAgICAgICA8ZmVHYXVzc2lhbkJsdXIgaW49IlNvdXJjZUdyYXBoaWMiIHN0ZERldmlhdGlvbj0iOCIgcmVzdWx0PSJibHVyIi8+CiAgICAgICAgICAgICAgPGZlQ29tcG9zaXRlIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9ImJsdXIiIG9wZXJhdG9yPSJvdmVyIi8+CiAgICAgICAgICAgIDwvZmlsdGVyPgogICAgICAgICAgPC9kZWZzPgogICAgICAgICAgPHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI4MDAiIGZpbGw9InVybCgjYW1iR2xvdykiLz4KICAgICAgICAgIDxnIGlkPSJtYXAtZ2xvdyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1zdGF0ZXMiPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtcHVsc2VzIj48L2c+CiAgICAgICAgPC9zdmc+CiAgICAgICAgPGRpdiBjbGFzcz0ibWFwLXRvb2x0aXAiIGlkPSJ0b29sdGlwIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBTVEFURSBQQU5FTCAtLT4KICA8ZGl2IGNsYXNzPSJzdGF0ZS1wYW5lbCIgaWQ9InN0YXRlLWRldGFpbCI+CiAgICA8ZGl2IGNsYXNzPSJwYW5lbC1lbXB0eSI+CiAgICAgIDxzdmcgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEiPgogICAgICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0iTTEyIDh2NE0xMiAxNmguMDEiLz4KICAgICAgPC9zdmc+CiAgICAgIDxkaXYgY2xhc3M9InBlLXQiPlNlbGVjdCBhIHN0YXRlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InBlLXMiPkNsaWNrIGFueSByZWdpb24gb24gdGhlIG1hcDxici8+dG8gb3BlbiBpdHMgbmFycmF0aXZlIHBhbmVsLjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cgo8L2Rpdj4KCjwhLS0gTkFSUkFUSVZFIFJPVyAtLT4KPGRpdiBjbGFzcz0ibmFyLXJvdyI+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SaXNpbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1sZWZ0OmF1dG8iPmdhaW5pbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyaXNpbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLWRvdCBmYWxsIj48L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy10aXRsZSI+RGVjbGluaW5nIG5hcnJhdGl2ZXM8L3NwYW4+CiAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tbGVmdDphdXRvIj5sb3NpbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJkZWNsaW5pbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIElORElBOiBMQVNUIDI0IEhPVVJTIC0tPgo8c2VjdGlvbiBjbGFzcz0icDI0LXNlY3Rpb24iPgogIDxkaXYgY2xhc3M9InAyNC1oZWFkZXIiPgogICAgPGRpdj4KICAgICAgPGRpdiBjbGFzcz0icDI0LXRpdGxlIj5JbmRpYSBpbiB0aGUgbGFzdCAyNCBob3VyczwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJwMjQtc3ViIj5XaGF0IHRoZSBuYXRpb24gd2FzIGZvY3VzZWQgb24sIGV2ZXJ5IGZvdXIgaG91cnM8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InAyNC1jYXJkcyIgaWQ9InAyNC1jYXJkcyI+CiAgICA8ZGl2IGNsYXNzPSJwMjQtZW1wdHkiPkxvYWRpbmcgc25hcHNob3RzLi4uPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjxzdHlsZT4KLnAyNC1zZWN0aW9ue3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDhweH0KLnAyNC1oZWFkZXJ7bWFyZ2luLWJvdHRvbToyMnB4fQoucDI0LXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjBweDtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0taW5rKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoucDI0LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6NHB4fQoucDI0LWNhcmRze2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KDMsMWZyKTtnYXA6MTRweH0KLnAyNC1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7Z3JpZC1jb2x1bW46MS8tMTtwYWRkaW5nOjIwcHggMH0KLnAyNC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDsKICBwYWRkaW5nOjE4cHggMjBweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxMHB4OwogIHBvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjsKICB0cmFuc2l0aW9uOmJvcmRlci1jb2xvciAwLjJzOwp9Ci5wMjQtY2FyZDpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5wMjQtY2FyZDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKfQoucDI0LWNhcmQtdGltZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQoucDI0LWNhcmQtbmFye2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjN9Ci5wMjQtY2FyZC1pbnNpZ2h0e2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMS41cHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42fQoucDI0LWNhcmQtc3RhdGV7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6MnB4fQoucDI0LWNhcmQtc3RhdGUtbmFtZXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnAyNC1jYXJkLXN0YXRlLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQoucDI0LWNhcmQtZm9vdGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Ym9yZGVyLXRvcDoxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtwYWRkaW5nLXRvcDo4cHg7bWFyZ2luLXRvcDoycHh9Ci5wMjQtY2FyZC1lbW97Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnAyNC1jYXJkLXNpZ3N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5wMjQtY2FyZC1uYXJze2Rpc3BsYXk6ZmxleDtmbGV4LXdyYXA6d3JhcDtnYXA6NHB4fQoucDI0LWNhcmQtbmFyLXRhZ3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpO2NvbG9yOnZhcigtLWZhaW50KX0KPC9zdHlsZT4KCjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIEp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDhweCByZ2JhKDU3LDI1NSwyMCwwLjQpIj5vYnNlcnZhdGlvbjwvc3Bhbj4uPC9kaXY+CjwvZGl2PgoKPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vdG9wb2pzb24tY2xpZW50QDMuMS4wL2Rpc3QvdG9wb2pzb24tY2xpZW50Lm1pbi5qcyI+PC9zY3JpcHQ+CjxzY3JpcHQ+CnZhciBBUElfQkFTRT0obG9jYXRpb24uaG9zdG5hbWU9PT0nbG9jYWxob3N0J3x8bG9jYXRpb24uaG9zdG5hbWU9PT0nMTI3LjAuMC4xJyk/J2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMCc6Jyc7CgovLyBBUEkKYXN5bmMgZnVuY3Rpb24gZmV0Y2hBbGxTdGF0ZXMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zdGF0ZXMnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgcm93cz1hd2FpdCByLmpzb24oKTsKICAgIGlmKCFyb3dzfHwhcm93cy5sZW5ndGgpIHJldHVybjsKICAgIHJvd3MuZm9yRWFjaChmdW5jdGlvbihyb3cpewogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhyb3cuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbUVtbz1yb3cuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxudWxsOwogICAgICB2YXIgZW50cnk9e2F0dGVudGlvbjpyb3cuYXR0ZW50aW9uLGRlbHRhOnJvdy5kZWx0YV8yNGgsdmVsb2NpdHk6cm93LnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tRW1vLGRvbWluYW50X25hcnJhdGl2ZTpyb3cuZG9taW5hbnRfbmFycmF0aXZlLGVtb3Rpb25zOmVtb3N9OwogICAgICBMSVZFW3Jvdy5uYW1lXT1lbnRyeTsKICAgICAgaWYoIVNEW3Jvdy5uYW1lXSkgU0Rbcm93Lm5hbWVdPU9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7CiAgICAgIE9iamVjdC5hc3NpZ24oU0Rbcm93Lm5hbWVdLGVudHJ5KTsKICAgIH0pOwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgc2V0VGltZW91dChyZW5kZXJNb21lbnR1bSwgNTAwKTsKICAgIGlmKFNFTCYmTElWRVtTRUxdJiZkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJykpIHJlbmRlclBhbmVsKFNFTCk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW0FQSV0nLGUubWVzc2FnZSk7fQp9CgpmdW5jdGlvbiBidWlsZExvY2FsSW5zaWdodCgpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwoKICAvLyBBZ2dyZWdhdGUgdG9wIG5hcnJhdGl2ZXMgYWNyb3NzIGFsbCBzdGF0ZXMKICB2YXIgbmFyPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5hcltuLm5hbWVdKSBuYXJbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuYXJbbi5uYW1lXVtuLmRpcl09KG5hcltuLm5hbWVdW24uZGlyXXx8MCkrbi52YWw7CiAgICAgIG5hcltuLm5hbWVdLnRvdGFsPShuYXJbbi5uYW1lXS50b3RhbHx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKCiAgLy8gVG9wIHJpc2luZyBhbmQgZmFsbGluZyAoZXhjbHVkZSB0aWVzIHdoZXJlIHNhbWUgbmFtZSByaXNlcyBhbmQgZmFsbHMpCiAgdmFyIHJpc2luZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPmt2WzFdLmRvd247fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udXAtYVsxXS51cDt9KS5zbGljZSgwLDMpOwogIHZhciBmYWxsaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS5kb3duLWFbMV0uZG93bjt9KS5zbGljZSgwLDIpOwogIHZhciB0b3AzPU9iamVjdC5lbnRyaWVzKG5hcikuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnRvdGFsLWFbMV0udG90YWw7fSkuc2xpY2UoMCwzKTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PWVudHJpZXMuc2xpY2UoKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgaG90dGVzdEVtbz1ob3R0ZXN0PyhMSVZFW2hvdHRlc3RbMF1dJiZMSVZFW2hvdHRlc3RbMF1dLmRvbWluYW50X2Vtb3Rpb24pfHwnJzonJyA7CgogIC8vIEJ1aWxkIGluc2lnaHQgdGV4dCDigJQgbW9yZSBhbmFseXRpY2FsLCBjb250ZXh0LWF3YXJlCiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICBpZighZWwpIHJldHVybjsKCiAgLy8gQ291bnQgaG93IG1hbnkgc3RhdGVzIGFyZSBhY3RpdmUKICB2YXIgYWN0aXZlU3RhdGVzPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGUpe3JldHVybiBlWzFdLmF0dGVudGlvbj41O30pLmxlbmd0aDsKICB2YXIgZG9taW5hbnRSZWdpb249Jyc7CiAgLy8gRmluZCB3aGljaCByZWdpb24gaGFzIG1vc3QgYWN0aXZlIHN0YXRlcwogIHZhciByZWdpb25NYXA9eydOb3J0aCc6WydEZWxoaScsJ1V0dGFyIFByYWRlc2gnLCdQdW5qYWInLCdIYXJ5YW5hJywnSmFtbXUgYW5kIEthc2htaXInXSwKICAgICdTb3V0aCc6WydUYW1pbCBOYWR1JywnS2FybmF0YWthJywnS2VyYWxhJywnQW5kaHJhIFByYWRlc2gnLCdUZWxhbmdhbmEnXSwKICAgICdXZXN0JzpbJ01haGFyYXNodHJhJywnR3VqYXJhdCcsJ1JhamFzdGhhbicsJ0dvYSddLAogICAgJ0Vhc3QnOlsnV2VzdCBCZW5nYWwnLCdCaWhhcicsJ09kaXNoYScsJ0poYXJraGFuZCddLAogICAgJ05FJzpbJ0Fzc2FtJywnTWFuaXB1cicsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ1RyaXB1cmEnLCdBcnVuYWNoYWwgUHJhZGVzaCddfTsKICB2YXIgdG9wUmVnaW9uU2NvcmU9MDsKICBPYmplY3QuZW50cmllcyhyZWdpb25NYXApLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgdmFyIHNjb3JlPWt2WzFdLnJlZHVjZShmdW5jdGlvbihzLHN0KXtyZXR1cm4gcysoKExJVkVbc3RdJiZMSVZFW3N0XS5hdHRlbnRpb24pfHwwKTt9LDApOwogICAgaWYoc2NvcmU+dG9wUmVnaW9uU2NvcmUpe3RvcFJlZ2lvblNjb3JlPXNjb3JlO2RvbWluYW50UmVnaW9uPWt2WzBdO30KICB9KTsKCiAgdmFyIGxpbmVzPVtdOwogIGlmKHJpc2luZy5sZW5ndGgmJmZhbGxpbmcubGVuZ3RoJiZyaXNpbmdbMF1bMF0hPT1mYWxsaW5nWzBdWzBdKXsKICAgIC8vIFN0cm9uZyBuYXJyYXRpdmUgc2hpZnQKICAgIGxpbmVzLnB1c2goJzxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+Jyk7CiAgICBsaW5lcy5wdXNoKCcgaXMgZGVmaW5pbmcgbmF0aW9uYWwgZGlzY291cnNlJyk7CiAgICBpZihmYWxsaW5nWzBdKSBsaW5lcy5wdXNoKCcsIGVjbGlwc2luZyA8ZW0+JytmYWxsaW5nWzBdWzBdKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nPicraG90dGVzdFswXSsnPC9zdHJvbmc+IGlzIGF0IHRoZSBjZW50cmUnKTsKICAgIGlmKGhvdHRlc3RFbW8mJmhvdHRlc3RFbW8hPT0nbnVsbCcpIGxpbmVzLnB1c2goJyDigJQgcHVibGljIHRvbmUgaXMgPGVtPicraG90dGVzdEVtbysnPC9lbT4nKTsKICAgIGlmKHJpc2luZ1sxXSYmcmlzaW5nWzFdWzBdIT09cmlzaW5nWzBdWzBdKSBsaW5lcy5wdXNoKCcuIDxlbT4nK3Jpc2luZ1sxXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMV1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIGJ1aWxkaW5nIGluIHRoZSBiYWNrZ3JvdW5kJyk7CiAgICBpZihkb21pbmFudFJlZ2lvbikgbGluZXMucHVzaCgnLiBUaGUgJytkb21pbmFudFJlZ2lvbisnIGlzIG1vc3QgYWN0aXZlIHRvZGF5Jyk7CiAgfSBlbHNlIGlmKHJpc2luZy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnQXR0ZW50aW9uIGlzIGNvbmNlbnRyYXRpbmcgYXJvdW5kIDxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+Jyk7CiAgICBpZihob3R0ZXN0KSBsaW5lcy5wdXNoKCcgd2l0aCA8c3Ryb25nPicraG90dGVzdFswXSsnPC9zdHJvbmc+IGdlbmVyYXRpbmcgdGhlIG1vc3Qgc2lnbmFsJyk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJy4gPGVtPicrcmlzaW5nWzFdWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3Jpc2luZ1sxXVswXS5zbGljZSgxKSsnPC9lbT4gaXMgYWxzbyByaXNpbmcnKTsKICAgIGlmKGFjdGl2ZVN0YXRlcz4wKSBsaW5lcy5wdXNoKCcuICcrYWN0aXZlU3RhdGVzKycgc3RhdGVzIGFyZSBpbiBhY3RpdmUgZGlzY291cnNlJyk7CiAgfSBlbHNlIGlmKHRvcDMubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ1NpZ25hbHMgYXJlIGRpc3BlcnNlZCBhY3Jvc3MgPGVtPicrdG9wM1swXVswXSsnPC9lbT4sIDxlbT4nK3RvcDNbMV1bMF0rJzwvZW0+Jyk7CiAgICBpZih0b3AzWzJdKSBsaW5lcy5wdXNoKCcgYW5kIDxlbT4nK3RvcDNbMl1bMF0rJzwvZW0+Jyk7CiAgICBsaW5lcy5wdXNoKCcg4oCUIG5vIHNpbmdsZSBkb21pbmFudCBuYXJyYXRpdmUgaGFzIGVtZXJnZWQnKTsKICB9CgogIGlmKGxpbmVzLmxlbmd0aCl7CiAgICBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrbGluZXMuam9pbignJykrJzwvZGl2Pic7CiAgfQoKICAvLyBUYWdzCiAgaWYodEVsKXsKICAgIHZhciB0YWdzPVtdOwogICAgZmFsbGluZy5zbGljZSgwLDEpLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDU5LDE4NCwyMTYsMC4zKTtjb2xvcjojM2JiOGQ4Ij7ihpMgJytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKTsKICAgIH0pOwogICAgcmlzaW5nLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjMpO2NvbG9yOiNlMDVhMjgiPuKGkSAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICBpZih0YWdzLmxlbmd0aCkgdEVsLmlubmVySFRNTD10YWdzLmpvaW4oJycpOwogIH0KCiAgaWYobWV0YUVsKXsKICAgIHZhciBzdGF0ZUNvdW50PU9iamVjdC52YWx1ZXMoTElWRSkuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzLmF0dGVudGlvbj4yO30pLmxlbmd0aDsKICAgIG1ldGFFbC50ZXh0Q29udGVudD0nT2JzZXJ2aW5nICcrc3RhdGVDb3VudCsnIGFjdGl2ZSBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKCkudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicse2hvdXI6JzItZGlnaXQnLG1pbnV0ZTonMi1kaWdpdCd9KTsKICB9Cn0KCmZ1bmN0aW9uIHVwZGF0ZUFsbFN0cmlwcygpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBob3R0ZXN0PWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCk+KGFbMV0uYXR0ZW50aW9ufHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsaG90dGVzdFswXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCdBdHRlbnRpb24gJytob3R0ZXN0WzFdLmF0dGVudGlvbi50b0ZpeGVkKDEpKTsKICB2YXIgdG9wQW5nZXJObT1udWxsLHRvcEFuZ2VyUGN0PTA7CiAgZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBlPWt2WzFdLmVtb3Rpb25zfHx7fTsKICAgIHZhciBhPWUuYW5nZXJ8fDA7CiAgICBpZihhPjAmJmE8PTEpIGE9TWF0aC5yb3VuZChhKjEwMCk7CiAgICBpZihhPnRvcEFuZ2VyUGN0KXt0b3BBbmdlclBjdD1hO3RvcEFuZ2VyTm09a3ZbMF07fQogIH0pOwogIGlmKHRvcEFuZ2VyTm0mJnRvcEFuZ2VyUGN0PjApewogICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlck5tKTsKICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsJ0FuZ2VyICcrTWF0aC5yb3VuZCh0b3BBbmdlclBjdCkrJyUgb2Ygc2lnbmFscycpOwogIH0gZWxzZSB7CiAgICAvLyBGYWxsIGJhY2sgdG8gZG9taW5hbnRfZW1vdGlvbj1hbmdlcgogICAgdmFyIGFuZ2VyRG9tPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcic7fSk7CiAgICBpZihhbmdlckRvbS5sZW5ndGgpewogICAgICB2YXIgdG9wQnlBdHQ9YW5nZXJEb20uc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgICAgIHNldFRleHQoJ3NjLWFuZ2VyLXZhbCcsdG9wQnlBdHRbMF0pOwogICAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdEb21pbmFudCBlbW90aW9uOiBhbmdlcicpOwogICAgfQogIH0KICB2YXIgY29vbGluZz1lbnRyaWVzLnJlZHVjZShmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS52ZWxvY2l0eXx8MCk8KGFbMV0udmVsb2NpdHl8fDApP2I6YTt9LGVudHJpZXNbMF0pOwogIHNldFRleHQoJ3NjLWNvb2xpbmctdmFsJyxjb29saW5nWzBdKTtzZXRUZXh0KCdzYy1jb29saW5nLXN1YicsJ1ZlbG9jaXR5ICcrY29vbGluZ1sxXS52ZWxvY2l0eS50b0ZpeGVkKDMpKTsKICB2YXIgbmM9e307ZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUpbmNba3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlXT0obmNba3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlXXx8MCkrMTt9KTsKICB2YXIgdG49T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pWzBdOwogIGlmKHRuKXtzZXRUZXh0KCdzYy1uYXJyYXRpdmUtdmFsJyx0blswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0blswXS5zbGljZSgxKSk7c2V0VGV4dCgnc2MtbmFycmF0aXZlLXN1YicsJ0RvbWluYW50IGFjcm9zcyAnK3RuWzFdKycgc3RhdGVzJyk7fQp9CmFzeW5jIGZ1bmN0aW9uIGZldGNoRGV0YWlsKG5hbWUpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlLycrZW5jb2RlVVJJQ29tcG9uZW50KG5hbWUpKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKGQuZW1vdGlvbnN8fHt9KTsKICAgIHZhciBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxkLmRvbWluYW50X2Vtb3Rpb258fG51bGw7CiAgICBTRFtuYW1lXT17YXR0ZW50aW9uOmQuYXR0ZW50aW9uLGRlbHRhOmQuZGVsdGFfMjRoLHZlbG9jaXR5OmQudmVsb2NpdHksZW1vdGlvbnM6ZW1vcyxkb21pbmFudF9lbW90aW9uOmRvbSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmUsCiAgICAgIG5hcnJhdGl2ZXM6KGQubmFycmF0aXZlc3x8W10pLm1hcChmdW5jdGlvbihuKXtyZXR1cm57bmFtZTpuLm5hbWUsdmFsOm4udmFsLGRpcjpuLmRpcnx8J2ZsYXQnfTt9KSwKICAgICAgcmlzaW5nOmQucmlzaW5nfHxbXSxmYWxsaW5nOmQuZmFsbGluZ3x8W10sc3VtbWFyeTpkLnN1bW1hcnl8fERFRkFVTFQuc3VtbWFyeSwKICAgICAgYXJ0aWNsZXM6ZC5hcnRpY2xlc3x8W10sdGltZWxpbmU6ZC50aW1lbGluZXx8REVGQVVMVC50aW1lbGluZSwKICAgICAgbmFycmF0aXZlSGlzdG9yeTpkLm5hcnJhdGl2ZUhpc3Rvcnl8fERFRkFVTFQubmFycmF0aXZlSGlzdG9yeSxzaWduYWxfY291bnQ6ZC5zaWduYWxfY291bnR8fDB9OwogICAgaWYoIUxJVkVbbmFtZV0pTElWRVtuYW1lXT17YXR0ZW50aW9uOmQuYXR0ZW50aW9uLGRlbHRhOmQuZGVsdGFfMjRoLHZlbG9jaXR5OmQudmVsb2NpdHksZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlfTsKICAgIExJVkVbbmFtZV0uZW1vdGlvbnM9ZW1vcztMSVZFW25hbWVdLmRvbWluYW50X2Vtb3Rpb249ZG9tOwogICAgcmV0dXJuIFNEW25hbWVdOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tmZXRjaERldGFpbF0nLG5hbWUsZS5tZXNzYWdlKTtyZXR1cm4gU0RbbmFtZV18fERFRkFVTFQ7fQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaFNuYXAoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zbmFwc2hvdC9kYWlseScpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC5lcnJvcikgcmV0dXJuOwogICAgLy8gdG9wYmFyCiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xpdmUtY291bnQnKTsKICAgIGlmKGVsJiZkLnRvdGFsX3NpZ25hbHMpIGVsLnRleHRDb250ZW50PWQudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOwogICAgdmFyIG1ldGE9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1tZXRhJyk7CiAgICBpZihtZXRhJiZkLmFzX29mKSBtZXRhLnRleHRDb250ZW50PSczMCBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKGQuYXNfb2YpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nKTsKICAgIC8vIHN0YXRzIHN0cmlwCiAgICBzZXRUZXh0KCdzYy1zaWduYWxzLXZhbCcsIGQudG90YWxfc2lnbmFscz9kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTonLScpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgfWNhdGNoKGUpe30KfQoKZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoZWwpZWwudGV4dENvbnRlbnQ9dmFsO30KCmZ1bmN0aW9uIHVwZGF0ZVN0cmlwTmFycmF0aXZlKCl7dXBkYXRlQWxsU3RyaXBzKCk7fQpmdW5jdGlvbiB1cGRhdGVTdHJpcEFuZ2VyKCl7fQoKZnVuY3Rpb24gc2VsZWN0SG90dGVzdCgpewogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoU0QpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKHRvcCkgc2VsZWN0Xyh0b3BbMF0pOwp9CmFzeW5jIGZ1bmN0aW9uIGZldGNoSW5zaWdodHMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9pbnNpZ2h0cycpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC5lcnJvcikgcmV0dXJuOwogICAgdmFyIHNpZz1kLnNpZ25hdHVyZTsKICAgIGlmKHNpZyl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgaWYoZWwpZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij48ZW0+JytzaWcuZmFkaW5nLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3NpZy5mYWRpbmcuc2xpY2UoMSkrJzwvZW0+IGZhZGluZyBhcyA8ZW0+JytzaWcucmlzaW5nX3ByaW1hcnkrIjwvZW0+Iisoc2lnLnJpc2luZ19zZWNvbmRhcnk/IiBhbG9uZ3NpZGUgPGVtPiIrc2lnLnJpc2luZ19zZWNvbmRhcnkrIjwvZW0+IjoiIikrIiBhY3Jvc3MgdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gPHN0cm9uZyBzdHlsZT1cImNvbG9yOnZhcigtLWluaylcIj4iK3NpZy5ob3R0ZXN0X3N0YXRlKyI8L3N0cm9uZz4gZG9taW5hdGVzLjwvZGl2PiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC50YWdzKXRFbC5pbm5lckhUTUw9ZC50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgfQogICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgIGlmKHJFbCYmZC5yaXNpbmcmJmQucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQucmlzaW5nLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTsKICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTJweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjRweDsiPicrCiAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjE0cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOiNlMDVhMjg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4mIzg1OTM7IFJJU0lORzwvc3Bhbj4nKwogICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAobi5zdGF0ZXMmJm4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo2cHg7Ij5Ecml2ZW4gYnk6ICcrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsKICAgICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjJweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCxyZ2JhKDIyNCw5MCw0MCwwLjQpLCNlMDVhMjgpO2JvcmRlci1yYWRpdXM6MnB4Ij48L2Rpdj48L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nOwogICAgICB9KS5qb2luKCcnKTs7CiAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgaWYoZkVsJiZkLmZhbGxpbmcmJmQuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pewogICAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTsKICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTJweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjRweDsiPicrCiAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjE0cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOiMzYmI4ZDg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4mIzg1OTU7IEZBRElORzwvc3Bhbj4nKwogICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgICAobi5zdGF0ZXMmJm4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo2cHg7Ij5XYXMgYWN0aXZlIGluOiAnK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoycHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQscmdiYSg1OSwxODQsMjE2LDAuNCksIzNiYjhkOCk7Ym9yZGVyLXJhZGl1czoycHgiPjwvZGl2PjwvZGl2PicrCiAgICAgICAgJzwvZGl2Pic7CiAgICAgIH0pLmpvaW4oJycpOzsKICAgIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgIGlmKGdFbCYmZC5yZWdpb25hbCYmZC5yZWdpb25hbC5sZW5ndGgpZ0VsLmlubmVySFRNTD1yZW5kZXJSZWdpb25hbENhcmRzKGQucmVnaW9uYWwpOzsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hGdWxsU25hcHNob3QoKXsKICAvLyBMb2FkIEFMTCBzdGF0ZSBkYXRhIGluIG9uZSByZXF1ZXN0IGZvciBpbnN0YW50IGZpcnN0LWxvYWQKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9mdWxsLXNuYXBzaG90Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLndhcm1pbmdfdXB8fCFkLnN0YXRlc3x8IWQuc3RhdGVzLmxlbmd0aCkgcmV0dXJuIGZhbHNlOwoKICAgIC8vIFBvcHVsYXRlIFNEIGFuZCBMSVZFIGZyb20gZnVsbCBzbmFwc2hvdAogICAgZC5zdGF0ZXMuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgICAgaWYoIXMubmFtZSkgcmV0dXJuOwogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhzLmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxzLmRvbWluYW50X2Vtb3Rpb258fG51bGw7CiAgICAgIHZhciBlbnRyeT1PYmplY3QuYXNzaWduKHt9LHMse2Vtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZGVsdGE6cy5kZWx0YV8yNGh8fDB9KTsKICAgICAgU0Rbcy5uYW1lXT1lbnRyeTsKICAgICAgTElWRVtzLm5hbWVdPXthdHRlbnRpb246cy5hdHRlbnRpb24sZGVsdGE6cy5kZWx0YV8yNGh8fDAsdmVsb2NpdHk6cy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbSxkb21pbmFudF9uYXJyYXRpdmU6cy5kb21pbmFudF9uYXJyYXRpdmUsZW1vdGlvbnM6ZW1vc307CiAgICB9KTsKCiAgICAvLyBVcGRhdGUgc2lnbmFscyBjb3VudAogICAgaWYoZC5zbmFwc2hvdCYmZC5zbmFwc2hvdC50b3RhbF9zaWduYWxzKXsKICAgICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLGQuc25hcHNob3QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpKTsKICAgIH0KCiAgICAvLyBVcGRhdGUgaW5zaWdodHMgZnJvbSBjYWNoZWQgZGF0YQogICAgaWYoZC5pbnNpZ2h0cyYmZC5pbnNpZ2h0cy5zaWduYXR1cmUpewogICAgICB2YXIgc2lnPWQuaW5zaWdodHMuc2lnbmF0dXJlOwogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQuaW5zaWdodHMudGFncyl0RWwuaW5uZXJIVE1MPWQuaW5zaWdodHMudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgICAgaWYockVsJiZkLmluc2lnaHRzLnJpc2luZyYmZC5pbnNpZ2h0cy5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTtyZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6I2UwNWEyOCI+4oaRIHJpc2luZzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NHB4OyI+JytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiNlMDVhMjg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICAgIGlmKGZFbCYmZC5pbnNpZ2h0cy5mYWxsaW5nJiZkLmluc2lnaHRzLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgfQoKICAgIC8vIFJlbmRlciBldmVyeXRoaW5nIGluc3RhbnRseSBmcm9tIGNhY2hlZCBkYXRhCiAgICBhcHBseUxheWVyKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgZGlzbWlzc0xvYWRlcigpOwogICAgLy8gUG9wdWxhdGUgbmFycmF0aXZlIGNhcmRzIGZyb20gY2FjaGVkIGluc2lnaHRzIGluIGZ1bGwtc25hcHNob3QKICAgIGlmKGQuaW5zaWdodHMmJmQuaW5zaWdodHMucmlzaW5nJiZkLmluc2lnaHRzLnJpc2luZy5sZW5ndGgpewogICAgICB2YXIgckVsMj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgICAgaWYockVsMikgckVsMi5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pewogICAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTsKICAgICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTJweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo0cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjE0cHg7Y29sb3I6dmFyKC0taW5rKSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOiNlMDVhMjg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4mIzg1OTM7IFJJU0lORzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcyYmbi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjZweDsiPkRyaXZlbiBieTogJytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjJweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCxyZ2JhKDIyNCw5MCw0MCwwLjQpLCNlMDVhMjgpO2JvcmRlci1yYWRpdXM6MnB4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nOwogICAgICB9KS5qb2luKCcnKTsKICAgICAgdmFyIGZFbDI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICAgIGlmKGZFbDImJmQuaW5zaWdodHMuZmFsbGluZykgZkVsMi5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXsKICAgICAgICB2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7CiAgICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEycHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NHB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxNHB4O2NvbG9yOnZhcigtLWluaykiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjojM2JiOGQ4O2xldHRlci1zcGFjaW5nOjAuMDhlbSI+JiM4NTk1OyBGQURJTkc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMmJm4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo2cHg7Ij5XYXMgYWN0aXZlIGluOiAnK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MnB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIHJpZ2h0LHJnYmEoNTksMTg0LDIxNiwwLjQpLCMzYmI4ZDgpO2JvcmRlci1yYWRpdXM6MnB4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nOwogICAgICB9KS5qb2luKCcnKTsKICAgICAgdmFyIGdFbDI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgICAgaWYoZ0VsMiYmZC5pbnNpZ2h0cy5yZWdpb25hbCkgZ0VsMi5pbm5lckhUTUw9cmVuZGVyUmVnaW9uYWxDYXJkcyhkLmluc2lnaHRzLnJlZ2lvbmFsKTsKICAgIH0KICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgLy8gVXNlIGNhY2hlZCBuYXJyYXRpdmUgaW5zaWdodCBpZiBhdmFpbGFibGUKICAgIGlmKGQubmFycmF0aXZlX2luc2lnaHQmJmQubmFycmF0aXZlX2luc2lnaHQudGV4dCl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICAgICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQrJzwvZGl2Pic7CiAgICAgIGlmKHRFbCYmZC5uYXJyYXRpdmVfaW5zaWdodC50b3BfbmFycmF0aXZlcyl7CiAgICAgICAgdEVsLmlubmVySFRNTD1kLm5hcnJhdGl2ZV9pbnNpZ2h0LnRvcF9uYXJyYXRpdmVzLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgICAgdmFyIGNvbD1pPT09MD8nI2UwNWEyOCc6J3JnYmEoMTYwLDE5MCwyMzAsMC42KSc7CiAgICAgICAgICB2YXIgYXJyPWk9PT0wPydcdTIxOTEgJzonXHUwMGI3ICc7CiAgICAgICAgICByZXR1cm4gJzxzcGFuIGNsYXNzPVwic2ktdGFnXCIgc3R5bGU9XCJib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKTtjb2xvcjonK2NvbCsnXCI+JythcnIrbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgICB9KS5qb2luKCcnKTsKICAgICAgfQogICAgfQogICAgcmV0dXJuIHRydWU7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbZnVsbC1zbmFwc2hvdF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gZmFsc2U7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKXsKICB0cnl7CiAgICAvLyBUcnkgY2FjaGVkIHZlcnNpb24gZnJvbSBmdWxsLXNuYXBzaG90IGZpcnN0IChhbHJlYWR5IGxvYWRlZCkKICAgIC8vIFRoZW4gY2FsbCBkZWRpY2F0ZWQgZW5kcG9pbnQgZm9yIGZyZXNoIEFJIGFuYWx5c2lzCiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9uYXJyYXRpdmUtaW5zaWdodCcpOwogICAgaWYoIXIub2spIHJldHVybjsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoIWQudGV4dCkgcmV0dXJuOwoKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwoKICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC50ZXh0Kyc8L2Rpdj4nOwoKICAgIC8vIFRhZ3MgZnJvbSB0b3AgbmFycmF0aXZlcwogICAgaWYodEVsJiZkLnRvcF9uYXJyYXRpdmVzJiZkLnRvcF9uYXJyYXRpdmVzLmxlbmd0aCl7CiAgICAgIHRFbC5pbm5lckhUTUw9ZC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICB2YXIgY29sPWk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjYpJzsKICAgICAgICB2YXIgYXJyb3c9aT09PTA/J+KGkSAnOifCtyAnOwogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpO2NvbG9yOicrY29sKyciPicrYXJyb3crbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9CgogICAgaWYobWV0YUVsKXsKICAgICAgdmFyIHQ9bmV3IERhdGUoZC5hc19vZik7CiAgICAgIG1ldGFFbC50ZXh0Q29udGVudD0nU2lnbmFsIGFuYWx5c2lzIMK3ICcrdC50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pKyhkLmZhbGxiYWNrPycgwrcgcGF0dGVybi1iYXNlZCc6JyDCtyBBSSBzeW50aGVzaXplZCcpOwogICAgfQogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tuYXJyYXRpdmVdJyxlLm1lc3NhZ2UpO30KfQoKZnVuY3Rpb24gcmVuZGVyUmVnaW9uYWxDYXJkcyhyZWdpb25zKXsKICB2YXIgZXBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgcmV0dXJuIHJlZ2lvbnMubWFwKGZ1bmN0aW9uKHIpewogICAgdmFyIGVtb0NvbD1yLmRvbWluYW50X2Vtb3Rpb24/ZXBhbFtyLmRvbWluYW50X2Vtb3Rpb25dfHwnIzU1NjY3Nyc6JyM1NTY2NzcnOwogICAgdmFyIG5hcnM9ci5uYXJyYXRpdmVzfHxbci50b3BfbmFycmF0aXZlfHwn4oCUJ107CiAgICB2YXIgaXNVbmlxdWU9ci51bmlxdWVfZm9jdXM7CiAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTJweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAvLyBSZWdpb24gaGVhZGVyCiAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj4nK3IucmVnaW9uKyc8L3NwYW4+JysKICAgICAgICAgIChpc1VuaXF1ZT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7cGFkZGluZzoxcHggNXB4O2JvcmRlci1yYWRpdXM6MnB4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMSk7Y29sb3I6IzNiYjhkODtsZXR0ZXItc3BhY2luZzowLjA2ZW0iPnVuaXF1ZSBmb2N1czwvc3Bhbj4nOicnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+JysKICAgICAgICAgIChyLmRvbWluYW50X2Vtb3Rpb24/JzxzcGFuIHN0eWxlPSJ3aWR0aDo2cHg7aGVpZ2h0OjZweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOicrZW1vQ29sKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siIHRpdGxlPSInK3IuZG9taW5hbnRfZW1vdGlvbisnIj48L3NwYW4+JzonJykrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+JytyLmF0dGVudGlvbisnPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAvLyBIb3R0ZXN0IHN0YXRlCiAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjNweDsiPicrci5ob3R0ZXN0X3N0YXRlKyc8L2Rpdj4nKwogICAgICAvLyBOYXJyYXRpdmVzCiAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjRweDttYXJnaW4tdG9wOjRweDsiPicrCiAgICAgICAgbmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXsKICAgICAgICAgIHJldHVybiAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7cGFkZGluZzoycHggN3B4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAzKTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNik7Y29sb3I6dmFyKC0tZmFpbnQpIj4nK24rJzwvc3Bhbj4nOwogICAgICAgIH0pLmpvaW4oJycpKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaFN0YXRlQ29udGV4dChubSl7CiAgLy8gRmV0Y2ggY29udGV4dHVhbCBicmllZiDigJQgY29tYmluZXMgR29vZ2xlIE5ld3MgKyBzdG9yZWQgc2lnbmFscyArIEFJCiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUtY29udGV4dC8nK2VuY29kZVVSSUNvbXBvbmVudChubSkpOwogICAgaWYoIXIub2spIHJldHVybiBudWxsOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICByZXR1cm4gZDsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1tjb250ZXh0XScsZS5tZXNzYWdlKTsKICAgIHJldHVybiBudWxsOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlKTt9KTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDEyMDAwMCk7CiAgICAgIHNldEludGVydmFsKGZldGNoSW5zaWdodHMsMzYwMDAwMCk7fQogIH0sMTUwMDApOwp9CgovLyBOQVJSQVRJVkUgREFUQQp2YXIgU0hJRlRTPXsKICAnM20nOlsKICAgIHtmYWRpbmc6J0luZmxhdGlvbicsZmFkaW5nTm90ZTonZWFzaW5nIG5hdGlvbmFsbHknLHJpc2luZzonQm9yZGVyIHNlY3VyaXR5JyxyaXNpbmdOb3RlOidwb3N0LWluY2lkZW50IHN1cmdlJ30sCiAgICB7ZmFkaW5nOidFbGVjdGlvbiByaGV0b3JpYycsZmFkaW5nTm90ZToncG9zdC1jeWNsZSBmYWRlJyxyaXNpbmc6J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3N0ZWFkeSByaXNlJ30sCiAgICB7ZmFkaW5nOidGYXJtZXIgcHJvdGVzdHMnLGZhZGluZ05vdGU6J21vbWVudHVtIGxvc3QnLHJpc2luZzonVW5lbXBsb3ltZW50IGFueGlldHknLHJpc2luZ05vdGU6J3lvdXRoIHNpZ25hbCBzdXJnZSd9LAogIF0sCiAgJzZtJzpbCiAgICB7ZmFkaW5nOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZhZGluZ05vdGU6J3ByZS1lbGVjdGlvbiBwZWFrJyxyaXNpbmc6J0NvcnJ1cHRpb24gYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3Bvc3QtY3ljbGUgcHVzaCd9LAogICAge2ZhZGluZzonUmVsaWdpb3VzIG5hdGlvbmFsaXNtJyxmYWRpbmdOb3RlOidwbGF0ZWF1IHBoYXNlJyxyaXNpbmc6J0Vjb25vbWljIGFueGlldHknLHJpc2luZ05vdGU6J2Nvc3Qtb2YtbGl2aW5nJ30sCiAgICB7ZmFkaW5nOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZmFkaW5nTm90ZToncmliYm9uLWN1dHRpbmcgZG9uZScscmlzaW5nOidMYXcgJiBvcmRlcicscmlzaW5nTm90ZTonY3JpbWUgbmFycmF0aXZlIHJpc2UnfSwKICBdLAogICcxeSc6WwogICAge2ZhZGluZzonUGFuZGVtaWMgcmVjb3ZlcnknLGZhZGluZ05vdGU6J2ZhZGVkIGVhcmx5IHllYXInLHJpc2luZzonSW5mbGF0aW9uJyxyaXNpbmdOb3RlOidkb21pbmF0ZWQgbWlkLXllYXInfSwKICAgIHtmYWRpbmc6J1JlZ2lvbmFsIGlkZW50aXR5JyxmYWRpbmdOb3RlOidsYW5ndWFnZS1sZWQgcGVhaycscmlzaW5nOidTZWN1cml0eSAmIGJvcmRlcnMnLHJpc2luZ05vdGU6J2dlb3BvbGl0aWNhbCBlc2NhbGF0aW9uJ30sCiAgICB7ZmFkaW5nOidHb3Zlcm5hbmNlIG9wdGltaXNtJyxmYWRpbmdOb3RlOidwb2xpY3kgaG9uZXltb29uIGVuZCcscmlzaW5nOidDb3JydXB0aW9uICYgc2NhbXMnLHJpc2luZ05vdGU6J2FjY291bnRhYmlsaXR5IGN5Y2xlJ30sCiAgXSwKfTsKdmFyIFJFR19TSElGVFM9WwogIHtzdGF0ZTonVGFtaWwgTmFkdScsZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLHRvOidGZWRlcmFsIHJlc291cmNlIGRpc3B1dGVzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonQmloYXInLGZyb206J0VsZWN0aW9uIHJoZXRvcmljJyx0bzonVW5lbXBsb3ltZW50ICYgZXhhbSBzY2FtcycsdGltZTonNiB3a3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2tzJ30sCiAge3N0YXRlOidSYWphc3RoYW4nLGZyb206J0Zhcm1lciBwcm90ZXN0cycsdG86J0hlYXQgd2F2ZSDCtyBFbnZpcm9ubWVudCcsdGltZTonMiB3a3MnfSwKICB7c3RhdGU6J0thcm5hdGFrYScsZnJvbTonTWluaW5nIGNvbnRyb3ZlcnN5Jyx0bzonTGFuZ3VhZ2Ugc2lnbmFnZSBwb2xpdGljcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyBpbmZyYXN0cnVjdHVyZScsdG86J0FpciBxdWFsaXR5IGNyaXNpcycsdGltZTonMTAgZGF5cyd9LAogIHtzdGF0ZTonTWFuaXB1cicsZnJvbTonR292ZXJuYW5jZSAmIGNhYmluZXQnLHRvOidFdGhuaWMgdGVuc2lvbnMgwrcgQUZTUEEnLHRpbWU6JzUgd2tzJ30sCiAge3N0YXRlOidQdW5qYWInLGZyb206J1Bvd2VyIGNyaXNpcycsdG86J0JvcmRlciBzZWN1cml0eSDCtyBEcm9uZXMnLHRpbWU6JzMgd2tzJ30sCl07CnZhciBNT0NLX1I9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbiBwcmVzc3VyZScsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBkYXRhPVNISUZUU1twZXJpb2RdfHxTSElGVFNbJzNtJ107CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MPWRhdGEubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDowO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo4cHg7b3ZlcmZsb3c6aGlkZGVuOyI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjZweCAxMHB4O2JvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhbGwpO21hcmdpbi1ib3R0b206M3B4OyI+ZmFkaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLmZhZGluZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5mYWRpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9IndpZHRoOjI4cHg7ZmxleC1zaHJpbms6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNDU7Zm9udC1zaXplOjEzcHg7Ij7ihpI8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6OHB4IDEwcHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1yaXNlKTttYXJnaW4tYm90dG9tOjNweDsiPnJpc2luZzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuMjsiPicrcy5yaXNpbmcrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK3MucmlzaW5nTm90ZSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0YWIpewogIHRhYi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHQpe3QuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICB0YWIuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVuZGVyU3RyaXAodGFiLmRhdGFzZXQucGVyaW9kKTsKICB9KTsKfSk7CgpmdW5jdGlvbiByZW5kZXJNb21lbnR1bSgpewogIC8vIFJlYWQgZnJvbSBTRCAocG9wdWxhdGVkIGJ5IGZldGNoQWxsU3RhdGVzIGZyb20gbGl2ZSBBUEkpCiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgbmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpOwogIHZhciBmYWxsaW5nPXNvcnRlZC5zbGljZSgtNSkucmV2ZXJzZSgpOwogIHZhciBteD1yaXNpbmcubGVuZ3RoP3Jpc2luZ1swXVsxXToxMDA7CgogIC8vIFdyaXRlIHRvIHJpc2luZy1saXN0IChtYXRjaGVzIG5hci1yb3cgSFRNTCkKICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogIGlmKHJFbCYmcmlzaW5nLmxlbmd0aCl7CiAgICByRWwuaW5uZXJIVE1MPXJpc2luZy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gZGVjbGluaW5nLWxpc3QKICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogIGlmKGZFbCYmZmFsbGluZy5sZW5ndGgpewogICAgZkVsLmlubmVySFRNTD1mYWxsaW5nLm1hcChmdW5jdGlvbihuKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gcmVnaW9uYWwtbGlzdCDigJQgdG9wIHN0YXRlIHBlciByZWdpb24gZnJvbSBMSVZFCiAgdmFyIHJlZ2lvbnM9ewogICAgJ05vcnRoJzpbJ0RlbGhpJywnVXR0YXIgUHJhZGVzaCcsJ1B1bmphYicsJ0hhcnlhbmEnLCdIaW1hY2hhbCBQcmFkZXNoJywnVXR0YXJha2hhbmQnLCdKYW1tdSBhbmQgS2FzaG1pciddLAogICAgJ0Vhc3QnOlsnV2VzdCBCZW5nYWwnLCdCaWhhcicsJ0poYXJraGFuZCcsJ09kaXNoYSddLAogICAgJ1dlc3QnOlsnTWFoYXJhc2h0cmEnLCdHdWphcmF0JywnUmFqYXN0aGFuJywnR29hJ10sCiAgICAnU291dGgnOlsnVGFtaWwgTmFkdScsJ0thcm5hdGFrYScsJ0tlcmFsYScsJ0FuZGhyYSBQcmFkZXNoJywnVGVsYW5nYW5hJ10sCiAgICAnTkUnOlsnQXNzYW0nLCdNYW5pcHVyJywnTmFnYWxhbmQnLCdNaXpvcmFtJywnTWVnaGFsYXlhJywnVHJpcHVyYScsJ0FydW5hY2hhbCBQcmFkZXNoJywnU2lra2ltJ10sCiAgICAnQ2VudHJhbCc6WydNYWRoeWEgUHJhZGVzaCcsJ0NoaGF0dGlzZ2FyaCddLAogIH07CiAgLy8gUmVnaW9uYWwgYnVpbHQgZnJvbSBMSVZFIOKAlCB1c2UgcmVuZGVyUmVnaW9uYWxDYXJkcyBpZiB3ZSBoYXZlIGluc2lnaHRzCiAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogIGlmKGdFbCl7CiAgICB2YXIgcmVnRGF0YT1PYmplY3QuZW50cmllcyhyZWdpb25zKS5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgcmVnaW9uPWt2WzBdLHN0YXRlcz1rdlsxXTsKICAgICAgdmFyIHJlZ2lvblN0YXRlcz1zdGF0ZXMubWFwKGZ1bmN0aW9uKHMpewogICAgICAgIHJldHVybntuYW1lOnMsYXR0OihMSVZFW3NdJiZMSVZFW3NdLmF0dGVudGlvbil8fDAsCiAgICAgICAgICAgICAgIGVtbzooTElWRVtzXSYmTElWRVtzXS5kb21pbmFudF9lbW90aW9uKXx8bnVsbCwKICAgICAgICAgICAgICAgbmFyOihMSVZFW3NdJiZMSVZFW3NdLmRvbWluYW50X25hcnJhdGl2ZSl8fG51bGx9OwogICAgICB9KS5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuYXR0PjA7fSk7CiAgICAgIGlmKCFyZWdpb25TdGF0ZXMubGVuZ3RoKSByZXR1cm4gbnVsbDsKICAgICAgcmVnaW9uU3RhdGVzLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSk7CiAgICAgIHZhciB0b3A9cmVnaW9uU3RhdGVzWzBdOwogICAgICB2YXIgbmFycz1bLi4ubmV3IFNldChyZWdpb25TdGF0ZXMubWFwKGZ1bmN0aW9uKHMpe3JldHVybiBzLm5hcjt9KS5maWx0ZXIoQm9vbGVhbikpXS5zbGljZSgwLDIpOwogICAgICB2YXIgYXZnQXR0PXJlZ2lvblN0YXRlcy5yZWR1Y2UoZnVuY3Rpb24ocyxyKXtyZXR1cm4gcytyLmF0dDt9LDApL3JlZ2lvblN0YXRlcy5sZW5ndGg7CiAgICAgIHJldHVybntyZWdpb246cmVnaW9uLGhvdHRlc3Rfc3RhdGU6dG9wLm5hbWUsYXR0ZW50aW9uOnRvcC5hdHQudG9GaXhlZCgxKSwKICAgICAgICAgICAgIGF2Z19hdHRlbnRpb246YXZnQXR0LnRvRml4ZWQoMSksZG9taW5hbnRfZW1vdGlvbjp0b3AuZW1vLAogICAgICAgICAgICAgbmFycmF0aXZlczpuYXJzLHRvcF9uYXJyYXRpdmU6bmFyc1swXXx8J+KAlCcsCiAgICAgICAgICAgICB1bmlxdWVfZm9jdXM6ZmFsc2Usc3RhdGVfY291bnQ6cmVnaW9uU3RhdGVzLmxlbmd0aH07CiAgICB9KS5maWx0ZXIoQm9vbGVhbik7CiAgICByZWdEYXRhLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gcGFyc2VGbG9hdChiLmF2Z19hdHRlbnRpb24pLXBhcnNlRmxvYXQoYS5hdmdfYXR0ZW50aW9uKTt9KTsKICAgIGlmKHJlZ0RhdGEubGVuZ3RoKSBnRWwuaW5uZXJIVE1MPXJlbmRlclJlZ2lvbmFsQ2FyZHMocmVnRGF0YSk7CiAgfQp9CgoKLy8gU1RBVEUgREFUQQp2YXIgU0Q9e307Cgp2YXIgTElWRT17fTsKZnVuY3Rpb24gbm9ybWFsaXplRW1vdGlvbnMoZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJue307dmFyIHZhbHM9T2JqZWN0LnZhbHVlcyhlKSx0b3Q9dmFscy5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcyt2O30sMCk7aWYodG90PD0wKXJldHVybnt9O2lmKHRvdDw9MS4wMSl7dmFyIG91dD17fTtPYmplY3Qua2V5cyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGspe291dFtrXT1NYXRoLnJvdW5kKGVba10qMTAwKTt9KTtyZXR1cm4gb3V0O31yZXR1cm4gZTt9CmZ1bmN0aW9uIGRvbWluYW50RW1vdGlvbihlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm4gbnVsbDt2YXIgbXg9MCxkb209bnVsbDtPYmplY3QuZW50cmllcyhlKS5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXT5teCl7bXg9a3ZbMV07ZG9tPWt2WzBdO319KTtyZXR1cm4gZG9tO30KZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIWVsKXJldHVybjtlbC50ZXh0Q29udGVudD12YWw7aWYodmFsJiZ2YWwhPT0nLScpe2VsLmNsYXNzTGlzdC5yZW1vdmUoJ2xvYWRpbmcnKTt9fQoKdmFyIERFRkFVTFQ9ewogIGF0dGVudGlvbjowLGRlbHRhOjAsdmVsb2NpdHk6MCwKICBlbW90aW9uczp7fSxkb21pbmFudF9lbW90aW9uOm51bGwsZG9taW5hbnRfbmFycmF0aXZlOm51bGwsCiAgbmFycmF0aXZlczpbXSxyaXNpbmc6W10sZmFsbGluZzpbXSwKICBzdW1tYXJ5OicnLGFydGljbGVzOltdLHRpbWVsaW5lOltdLAogIG5hcnJhdGl2ZUhpc3Rvcnk6W10sc2lnbmFsX2NvdW50OjAsCn07CgpmdW5jdGlvbiBnKG4pe3JldHVybiBTRFtuXXx8T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTt9CgpmdW5jdGlvbiBhQyhzKXsKICBpZighYUMuX3RzfHxEYXRlLm5vdygpLWFDLl90cz41MDAwKXsKICAgIHZhciBzYz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oZCl7cmV0dXJuIGQuYXR0ZW50aW9ufHwwO30pLmZpbHRlcihmdW5jdGlvbih2KXtyZXR1cm4gdj4wO30pOwogICAgYUMuX21uPXNjLmxlbmd0aD9NYXRoLm1pbi5hcHBseShudWxsLHNjKTowOwogICAgYUMuX214PXNjLmxlbmd0aD8oTWF0aC5tYXguYXBwbHkobnVsbCxzYyl8fDEpOjE7CiAgICBhQy5fdHM9RGF0ZS5ub3coKTsKICB9CiAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzLWFDLl9tbikvKGFDLl9teC1hQy5fbW4pKSk7CiAgdmFyIHN0b3BzPVsKICAgIFswLjAwLCcjMGExNjI4J10sICAvLyBkZWVwIG5hdnkgKHNpbGVudCkKICAgIFswLjEwLCcjMGUyZDUyJ10sICAvLyBuYXZ5CiAgICBbMC4yMCwnIzBhNGE3YSddLCAgLy8gc3RlZWwgYmx1ZQogICAgWzAuMzAsJyMwZDcwOTAnXSwgIC8vIHRlYWwKICAgIFswLjQyLCcjMGU5MDgwJ10sICAvLyBzZWEgZ3JlZW4KICAgIFswLjU0LCcjMmE4YTRhJ10sICAvLyBzYWdlIGdyZWVuCiAgICBbMC42NCwnI2M4OTYwYSddLCAgLy8gZ29sZAogICAgWzAuNzQsJyNkODYwMjAnXSwgIC8vIGFtYmVyCiAgICBbMC44NCwnI2NjMjgwOCddLCAgLy8gY3JpbXNvbgogICAgWzAuOTMsJyNlODAwMTAnXSwgIC8vIHJlZAogICAgWzEuMDAsJyNmZjEwMjAnXSwgIC8vIGJyaWdodCByZWQgKG1heGltdW0pCiAgXTsKCiAgLy8gRmluZCBzdXJyb3VuZGluZyBzdG9wcyBhbmQgbGVycAogIGZvcih2YXIgaT0wO2k8c3RvcHMubGVuZ3RoLTE7aSsrKXsKICAgIHZhciBzMD1zdG9wc1tpXSxzMT1zdG9wc1tpKzFdOwogICAgaWYobj49czBbMF0mJm48PXMxWzBdKXsKICAgICAgdmFyIHQ9KG4tczBbMF0pLyhzMVswXS1zMFswXSk7CiAgICAgIC8vIFBhcnNlIGhleCBhbmQgbGVycAogICAgICB2YXIgYzA9aGV4VG9SZ2IoczBbMV0pLGMxPWhleFRvUmdiKHMxWzFdKTsKICAgICAgdmFyIHI9TWF0aC5yb3VuZChjMFswXSsoYzFbMF0tYzBbMF0pKnQpOwogICAgICB2YXIgZz1NYXRoLnJvdW5kKGMwWzFdKyhjMVsxXS1jMFsxXSkqdCk7CiAgICAgIHZhciBiPU1hdGgucm91bmQoYzBbMl0rKGMxWzJdLWMwWzJdKSp0KTsKICAgICAgcmV0dXJuICdyZ2IoJytyKycsJytnKycsJytiKycpJzsKICAgIH0KICB9CiAgcmV0dXJuIHN0b3BzW3N0b3BzLmxlbmd0aC0xXVsxXTsKfQoKZnVuY3Rpb24gaGV4VG9SZ2IoaGV4KXsKICB2YXIgcj1wYXJzZUludChoZXguc2xpY2UoMSwzKSwxNik7CiAgdmFyIGc9cGFyc2VJbnQoaGV4LnNsaWNlKDMsNSksMTYpOwogIHZhciBiPXBhcnNlSW50KGhleC5zbGljZSg1LDcpLDE2KTsKICByZXR1cm4gW3IsZyxiXTsKfQpmdW5jdGlvbiBlQyhlKXsKICB2YXIgbXg9MCxkb209J3ByaWRlJzsKICBmb3IodmFyIGsgaW4gZSl7aWYoZVtrXT5teCl7bXg9ZVtrXTtkb209azt9fQogIHJldHVybiAoe2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9KVtkb21dfHwnIzMzYWFjYyc7Cn0KZnVuY3Rpb24gdkModil7CiAgLy8gTW9tZW50dW06IGNvb2xpbmcgKGJsdWUpIOKGkCBzdGFibGUgKHNsYXRlKSDihpIgcmlzaW5nICh3YXJtKSDihpIgc3VyZ2luZyAocmVkKQogIC8vIFVzZSBzbW9vdGggaW50ZXJwb2xhdGlvbiBmb3IgYmV0dGVyIHZpc3VhbCBkaXN0aW5jdGlvbgogIGlmKHY+MC4zKSAgcmV0dXJuICcjZTgxMDEwJzsgIC8vIHN1cmdpbmcgZmFzdCDigJQgYnJpZ2h0IHJlZAogIGlmKHY+MC4xNSkgcmV0dXJuICcjZDg0MDEwJzsgIC8vIHJpc2luZyBmYXN0ICDigJQgb3JhbmdlIHJlZAogIGlmKHY+MC4wNykgcmV0dXJuICcjZTA3ODIwJzsgIC8vIHJpc2luZyAgICAgICDigJQgYW1iZXIgb3JhbmdlCiAgaWYodj4wLjAyKSByZXR1cm4gJyNjOGEwMjAnOyAgLy8gc2xpZ2h0IHJpc2UgIOKAlCBnb2xkCiAgaWYodj4tMC4wMikgcmV0dXJuICcjMzM0NDU1JzsgLy8gc3RhYmxlICAgICAgIOKAlCBzbGF0ZQogIGlmKHY+LTAuMDcpIHJldHVybiAnIzFhNzA5MCc7IC8vIHNsaWdodCBjb29sICDigJQgdGVhbAogIGlmKHY+LTAuMTUpIHJldHVybiAnIzEwNTBhMCc7IC8vIGNvb2xpbmcgICAgICDigJQgYmx1ZQogIHJldHVybiAnIzBhMjg2OCc7ICAgICAgICAgICAgIC8vIGNvb2xpbmcgZmFzdCDigJQgZGVlcCBibHVlCn0KCnZhciBsYXllcj0nYXR0ZW50aW9uJyxTRUw9bnVsbCxGQVZTPW5ldyBTZXQoKTsKCi8vIE1BUApmdW5jdGlvbiBwcm9qXyh3LGgscGFkKXsKICBwYWQ9cGFkfHwyMDsKICB2YXIgbWluTG9uPTY4LjEsbWF4TG9uPTk3LjQsbWluTGF0PTYuNSxtYXhMYXQ9MzcuMTsKICB2YXIgc2NYPSh3LXBhZCoyKS8obWF4TG9uLW1pbkxvbik7CiAgdmFyIHNjWT0oaC1wYWQqMikvKG1heExhdC1taW5MYXQpOwogIHZhciBzYz1NYXRoLm1pbihzY1gsc2NZKTsKICB2YXIgb3g9cGFkKyh3LXBhZCoyLShtYXhMb24tbWluTG9uKSpzYykvMjsKICB2YXIgb3k9cGFkKyhoLXBhZCoyLShtYXhMYXQtbWluTGF0KSpzYykvMjsKICByZXR1cm4gZnVuY3Rpb24obG9uLGxhdCl7cmV0dXJuIFtveCsobG9uLW1pbkxvbikqc2MsIG95KyhtYXhMYXQtbGF0KSpzY107fTsKfQpmdW5jdGlvbiBnZW8ycGF0aChnZW9tLHBqKXsKICB2YXIgZD0nJzsKICBmdW5jdGlvbiByaW5nKGNzKXt2YXIgcz0nJztjcy5mb3JFYWNoKGZ1bmN0aW9uKGMsaSl7dmFyIHA9cGooY1swXSxjWzFdKTtzKz0oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSk7cmV0dXJuIHMrJ1onO30KICBpZihnZW9tLnR5cGU9PT0nUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pOwogIGVsc2UgaWYoZ2VvbS50eXBlPT09J011bHRpUG9seWdvbicpIGdlb20uY29vcmRpbmF0ZXMuZm9yRWFjaChmdW5jdGlvbihwKXtwLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTt9KTsKICByZXR1cm4gZDsKfQpmdW5jdGlvbiBjdHIoZ2VvbSl7CiAgdmFyIHB0cz1bXTsKICBmdW5jdGlvbiBjb2woYyl7aWYodHlwZW9mIGNbMF09PT0nbnVtYmVyJykgcHRzLnB1c2goYyk7ZWxzZSBjLmZvckVhY2goY29sKTt9CiAgY29sKGdlb20uY29vcmRpbmF0ZXMpOwogIGlmKCFwdHMubGVuZ3RoKSByZXR1cm4gWzAsMF07CiAgcmV0dXJuIFtwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFswXTt9LDApL3B0cy5sZW5ndGgscHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMV07fSwwKS9wdHMubGVuZ3RoXTsKfQpmdW5jdGlvbiBzTmFtZShwcm9wcyl7CiAgdmFyIHJhdz1wcm9wcy5zdF9ubXx8cHJvcHMuTkFNRV8xfHxwcm9wcy5uYW1lfHxwcm9wcy5OQU1FfHwnJzsKICB2YXIgbWFwPXsnTGFkYWtoJzonSmFtbXUgYW5kIEthc2htaXInLCdKYW1tdSAmIEthc2htaXInOidKYW1tdSBhbmQgS2FzaG1pcicsJ1V0dGFyYW5jaGFsJzonVXR0YXJha2hhbmQnLCdBbmRhbWFuIGFuZCBOaWNvYmFyJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnQW5kYW1hbiAmIE5pY29iYXIgSXNsYW5kJzonQW5kYW1hbiBhbmQgTmljb2JhciBJc2xhbmRzJywnTkNUIG9mIERlbGhpJzonRGVsaGknLCdQb25kaWNoZXJyeSc6J1B1ZHVjaGVycnknLCdEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpJzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdScsJ0RhbWFuIGFuZCBEaXUnOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1J307CiAgcmV0dXJuIG1hcFtyYXddfHxyYXc7Cn0KCnZhciBjYWNoZWRHZW89bnVsbDsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRNYXAoYXR0ZW1wdCl7CiAgYXR0ZW1wdCA9IGF0dGVtcHR8fDE7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goJ2h0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9naC91ZGl0LTAwMS9pbmRpYS1tYXBzLWRhdGFAbWFzdGVyL3RvcG9qc29uL2luZGlhLmpzb24nKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgdG9wbz1hd2FpdCByLmpzb24oKTsKICAgIGNhY2hlZEdlbz10b3BvanNvbi5mZWF0dXJlKHRvcG8sdG9wby5vYmplY3RzLnN0YXRlcyk7CiAgICByZW5kZXJNYXAoY2FjaGVkR2VvKTsKICAgIC8vIEFwcGx5IGNvbG9ycyBpbW1lZGlhdGVseSBpZiBkYXRhIGFscmVhZHkgbG9hZGVkCiAgICBpZihPYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg+MCl7CiAgICAgIGFwcGx5TGF5ZXIoKTsKICAgICAgcmVuZGVyTW9tZW50dW0oKTsKICAgICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICB9CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsNTAwKTsKICAgIHNldFRpbWVvdXQoYXBwbHlMYXllciwyMDAwKTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1ttYXBdIGxvYWQgZmFpbGVkIGF0dGVtcHQgJythdHRlbXB0Kyc6JyxlLm1lc3NhZ2UpOwogICAgaWYoYXR0ZW1wdDw1KXsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe2xvYWRNYXAoYXR0ZW1wdCsxKTt9LCBhdHRlbXB0KjIwMDApOwogICAgfSBlbHNlIHsKICAgICAgdmFyIG1pPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKTsKICAgICAgaWYobWkpIG1pLmlubmVySFRNTD0nPGRpdiBzdHlsZT0iY29sb3I6IzJhM2E0YTtwYWRkaW5nOjQwcHg7dGV4dC1hbGlnbjpjZW50ZXI7Zm9udC1mYW1pbHk6bW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4Ij5NYXAgdW5hdmFpbGFibGUg4oCUIHJlZnJlc2ggdG8gcmV0cnk8L2Rpdj4nOwogICAgfQogIH0KfQoKZnVuY3Rpb24gcmVuZGVyTWFwKHN0YXRlcyl7CiAgdmFyIHc9ODAwLGg9ODAwLHBqPXByb2pfKHcsaCwyOCk7CiAgdmFyIHNnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtc3RhdGVzJyk7CiAgdmFyIHBnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtcHVsc2VzJyk7CiAgdmFyIGdnPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtYXAtZ2xvdycpOwogIHNnLmlubmVySFRNTD0nJztwZy5pbm5lckhUTUw9Jyc7Z2cuaW5uZXJIVE1MPScnOwoKICBzdGF0ZXMuZmVhdHVyZXMuZm9yRWFjaChmdW5jdGlvbihmKXsKICAgIGlmKCFmLmdlb21ldHJ5KSByZXR1cm47CiAgICB2YXIgbm09c05hbWUoZi5wcm9wZXJ0aWVzKSxkPWcobm0pOwogICAgdmFyIHBhdGhFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywncGF0aCcpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZCcsZ2VvMnBhdGgoZi5nZW9tZXRyeSxwaikpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdzdGF0ZScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyxubSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2UnLCdyZ2JhKDI1NSwyNTUsMjU1LDAuMDcpJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcwLjUnKTsKICAgIHNnLmFwcGVuZENoaWxkKHBhdGhFbCk7CgogICAgdmFyIGN0PWN0cihmLmdlb21ldHJ5KSxjcD1waihjdFswXSxjdFsxXSk7CgogICAgLy8gQXRtb3NwaGVyaWMgZ2xvdyBmb3IgaGlnaC1hdHRlbnRpb24gc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NjUpewogICAgICB2YXIgZ2xvd0VsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdlbGxpcHNlJyk7CiAgICAgIHZhciBnbG93Uj1NYXRoLm1pbig2MCwyMCtkLmF0dGVudGlvbiowLjUpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J4JyxnbG93Uik7Z2xvd0VsLnNldEF0dHJpYnV0ZSgncnknLGdsb3dSKjAuNyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ29wYWNpdHknLCcwLjA4Jyk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2ZpbHRlcicsJ3VybCgjc3RhdGVHbG93KScpOwogICAgICBnbG93RWwuc3R5bGUuYW5pbWF0aW9uPSdnbG93UHVsc2UgJysoMi41K01hdGgucmFuZG9tKCkpKydzIGVhc2UtaW4tb3V0ICcrKE1hdGgucmFuZG9tKCkqMikrJ3MgaW5maW5pdGUnOwogICAgICBnZy5hcHBlbmRDaGlsZChnbG93RWwpOwogICAgfQoKICAgIC8vIER1YWwgcHVsc2UgcmluZ3MgZm9yIHZlcnkgaG90IHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTcyKXsKICAgICAgWzAsMV0uZm9yRWFjaChmdW5jdGlvbihpKXsKICAgICAgICB2YXIgcmluZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnY2lyY2xlJyk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7cmluZy5zZXRBdHRyaWJ1dGUoJ2N5JyxjcFsxXSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywncHVsc2UtcmluZyBwJysoaSsxKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMScpOwogICAgICAgIHJpbmcuc3R5bGUuYW5pbWF0aW9uRGVsYXk9KE1hdGgucmFuZG9tKCkqMi41KSsncyc7CiAgICAgICAgcGcuYXBwZW5kQ2hpbGQocmluZyk7CiAgICAgIH0pOwogICAgfQogIH0pOwogIGFwcGx5TGF5ZXIoKTsKICBhdHRhY2hJbnRlcmFjdGlvbnMoKTsKfQoKLy8gU2luZ2xlIHNvdXJjZSBvZiB0cnV0aCBmb3IgZW1vdGlvbiBjb2xvcgovLyBCb3RoIG1hcCBhbmQgcGFuZWwgY2FsbCB0aGlzIOKAlCBndWFyYW50ZWVzIHRoZXkgYWx3YXlzIG1hdGNoCmZ1bmN0aW9uIGdldEVmZmVjdGl2ZUVtb3Rpb24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgLy8gMS4gVHJ5IExJVkUuZG9taW5hbnRfZW1vdGlvbiAoc2V0IGJ5IC9hcGkvc3RhdGVzKQogIHZhciBkb209bGl2ZS5kb21pbmFudF9lbW90aW9ufHxkLmRvbWluYW50X2Vtb3Rpb247CgogIC8vIDIuIFRyeSBjb21wdXRpbmcgZnJvbSBlbW90aW9ucyBicmVha2Rvd24KICBpZighZG9tKXsKICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgICBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpOwogIH0KCiAgLy8gMy4gRmFsbGJhY2s6IGluZmVyIGZyb20gZG9taW5hbnQgbmFycmF0aXZlIChzYW1lIGxvZ2ljIGV2ZXJ5d2hlcmUpCiAgaWYoIWRvbSl7CiAgICB2YXIgbnA9KGxpdmUuZG9taW5hbnRfbmFycmF0aXZlfHxkLmRvbWluYW50X25hcnJhdGl2ZXx8JycpLnRvTG93ZXJDYXNlKCk7CiAgICBpZihucC5tYXRjaCgvYm9yZGVyfHRlcnJvcnxzZWN1cml0eXxjb25mbGljdHxhdHRhY2t8d2FyfGluZmlsdHJhdC8pKSBkb209J2ZlYXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvc2NhbXxjb3JydXB0fHByb3Rlc3R8YXJyZXN0fHZpb2xlbmNlfG91dHJhZ2V8Y3JpbWUvKSkgZG9tPSdhbmdlcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9kZXZlbG9wfGludmVzdHxncm93dGh8bGF1bmNofGluYXVndXJ8cmVmb3JtfHByb2dyZXNzfGJvb3N0LykpIGRvbT0naG9wZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9jdWx0dXJlfGhlcml0YWdlfHByaWRlfHZpY3Rvcnl8Y2VsZWJyYXR8bWVkYWx8YWNoaWV2ZW1lbnQvKSkgZG9tPSdwcmlkZSc7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9mbG9vZHxkcm91Z2h0fHVuZW1wbG95bWVudHxpbmZsYXRpb258c2hvcnRhZ2V8Y3Jpc2lzfGNvbmNlcm4vKSkgZG9tPSdhbnhpZXR5JzsKICAgIGVsc2UgaWYoKGxpdmUuYXR0ZW50aW9ufHxkLmF0dGVudGlvbnx8MCk+NSkgZG9tPSdhbnhpZXR5JzsgLy8gYWN0aXZlIHN0YXRlIGRlZmF1bHQKICAgIGVsc2UgZG9tPSdhbnhpZXR5JzsgLy8gZ2xvYmFsIGRlZmF1bHQKICB9CgogIHJldHVybiBkb207Cn0KCi8vIEdldCBlc3RpbWF0ZWQgZW1vdGlvbiBicmVha2Rvd24gKGZvciBwYW5lbCBkb251dCB3aGVuIHJlYWwgZGF0YSBtaXNzaW5nKQpmdW5jdGlvbiBnZXRFbW90aW9uQnJlYWtkb3duKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogIGlmKE9iamVjdC5rZXlzKGVtb3MpLmxlbmd0aCkgcmV0dXJuIHtlbW90aW9uczplbW9zLGVzdGltYXRlZDpmYWxzZX07CiAgLy8gQnVpbGQgc2tld2VkIGRpc3RyaWJ1dGlvbiBmcm9tIGVmZmVjdGl2ZSBlbW90aW9uCiAgdmFyIGRvbT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICB2YXIgYmFzZT17YW54aWV0eToxMyxhbmdlcjoxMyxob3BlOjEzLHByaWRlOjEzLGZlYXI6MTN9OwogIGJhc2VbZG9tXT00ODsKICByZXR1cm4ge2Vtb3Rpb25zOmJhc2UsZXN0aW1hdGVkOnRydWV9Owp9CgpmdW5jdGlvbiBhcHBseUxheWVyKCl7CiAgdmFyIF9wYXRocz1kb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKTsKICBpZighX3BhdGhzLmxlbmd0aCkgcmV0dXJuOwogIGFDLl90cz0wOwogIF9wYXRocy5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLGZpbGw7CiAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKSBmaWxsPWFDKGQuYXR0ZW50aW9uKTsKICAgIGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBkZT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgZmlsbD1lTWFwW2RlXXx8JyMzMzQ0NTUnOwogICAgfQogICAgZWxzZSBmaWxsPXZDKGQudmVsb2NpdHkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGZpbGwpOwogICAgKGZ1bmN0aW9uKCl7CiAgICAgIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiB4LmF0dGVudGlvbnx8MDt9KTsKICAgICAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKSxteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgICAgIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwoZC5hdHRlbnRpb24tbW4pLyhteC1tbikpKTsKICAgICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsbGF5ZXI9PT0nYXR0ZW50aW9uJz9NYXRoLm1heCgwLjMsMC4zK24qMC43KTowLjg1KTsKICAgIH0pKCk7CiAgfSk7Cn0KCmZ1bmN0aW9uIGF0dGFjaEludGVyYWN0aW9ucygpewogIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpewogICAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpOwogICAgICB2YXIgZD1nKG5tKTsKICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgICAgIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgbGF0ZXN0PScnOwogICAgICBpZihkLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlcy5sZW5ndGgpIGxhdGVzdD1kLm5hcnJhdGl2ZXNbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkLm5hcnJhdGl2ZXNbMF0ubmFtZS5zbGljZSgxKTsKICAgICAgZWxzZSBpZihsaXZlLmRvbWluYW50X25hcnJhdGl2ZSkgbGF0ZXN0PWxpdmUuZG9taW5hbnRfbmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2xpdmUuZG9taW5hbnRfbmFycmF0aXZlLnNsaWNlKDEpOwoKICAgICAgdmFyIHJvd3M9Jyc7CiAgICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgICAgIHZhciBhdHQ9bGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwOwogICAgICAgIHZhciBkbHQ9bGl2ZS5kZWx0YXx8ZC5kZWx0YXx8MDsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5BdHRlbnRpb248L3NwYW4+PHN0cm9uZz4nK2F0dC50b0ZpeGVkKDEpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgKGRsdCE9PTA/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaGlmdDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonKyhkbHQ+MD8nI2UwNWEyOCc6JyMzYmI4ZDgnKSsnIj4nKyhkbHQ+MD8nKyc6JycpK2RsdCsnPC9zdHJvbmc+PC9kaXY+JzonJykrCiAgICAgICAgICAobGF0ZXN0Pyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Ub3AgbmFycmF0aXZlPC9zcGFuPjxzdHJvbmc+JytsYXRlc3QrJzwvc3Ryb25nPjwvZGl2Pic6JycpOwogICAgICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBkb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgICAgaWYoZG9tRW1vKXsKICAgICAgICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOmQuZW1vdGlvbnN8fHt9OwogICAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RG9taW5hbnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JytwYWxbZG9tRW1vXSsnIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgICAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vcykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICAgICAgICB2YXIgdG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgICAgICAgaWYodG90PjAmJnRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7fQogICAgICAgICAgcm93cys9ZUwuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4gc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweCI+PHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytwYWxba3ZbMF1dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nK2t2WzBdKyc8L3NwYW4+PHN0cm9uZz4nK01hdGgucm91bmQoa3ZbMV0qMTAwL01hdGgubWF4KDEsdG90KSkrJyU8L3N0cm9uZz48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICAgIH0KICAgICAgfSBlbHNlIHsKICAgICAgICB2YXIgdmVsPWxpdmUudmVsb2NpdHl8fGQudmVsb2NpdHl8fDA7CiAgICAgICAgdmFyIHZlbERpcj12ZWw+MC4xPydSaXNpbmcgZmFzdCc6dmVsPjAuMDI/J1Jpc2luZyc6dmVsPC0wLjA1PydDb29saW5nJzonU3RhYmxlJzsKICAgICAgICB2YXIgdmVsQ29sPXZlbD4wLjAyPycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Nb21lbnR1bTwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRpcmVjdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nK3ZlbERpcisnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfQoKICAgICAgdGlwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0idHQtbiI+JytubSsnPC9kaXY+Jytyb3dzKyhsYXRlc3QmJmxheWVyIT09J2F0dGVudGlvbic/JzxkaXYgY2xhc3M9InR0LW5hciI+PHN0cm9uZz5OYXJyYXRpdmU8L3N0cm9uZz4nK2xhdGVzdCsnPC9kaXY+JzonJyk7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE5MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTUwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0nMSc7CiAgICB9KTsKcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJyxmdW5jdGlvbigpe3RpcC5zdHlsZS5vcGFjaXR5PTA7fSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe3NlbGVjdF8ocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8gU1RBVEUgUEFORUwKZnVuY3Rpb24gc2VsZWN0XyhubSl7CiAgU0VMPW5tOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmNsYXNzTGlzdC50b2dnbGUoJ3NlbGVjdGVkJyxwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk9PT1ubSk7CiAgfSk7CiAgLy8gU2hvdyBsb2FkaW5nIHN0YXRlIGltbWVkaWF0ZWx5IHdpdGggd2hhdGV2ZXIgTElWRSBkYXRhIHdlIGhhdmUKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKHBhbmVsKXsKICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgIHBhbmVsLmlubmVySFRNTD0KICAgICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICAgJzxkaXY+PGRpdiBjbGFzcz0ic3AtZWsiPicrKGxheWVyPT09J2F0dGVudGlvbic/J05hcnJhdGl2ZSBwYW5lbCc6bGF5ZXI9PT0nZW1vdGlvbic/J0Vtb3Rpb25hbCByZWdpc3Rlcic6J01vbWVudHVtIHBhbmVsJykrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoRkFWUy5oYXMobm0pPydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgICAnPC9idXR0b24+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDhlbSI+JysKICAgICAgICAnTG9hZGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKwogICAgICAgIChsaXZlLmF0dGVudGlvbj8nPGJyPjxicj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxOHB4O2NvbG9yOnZhcigtLWluaykiPkF0dGVudGlvbiAnK2xpdmUuYXR0ZW50aW9uLnRvRml4ZWQoMSkrJzwvc3Bhbj4nOicnKSsKICAgICAgICAobGl2ZS5kb21pbmFudF9lbW90aW9uPyc8YnI+PHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JytsaXZlLmRvbWluYW50X2Vtb3Rpb24rJyBzaWduYWwgZG9taW5hbnQ8L3NwYW4+JzonJykrCiAgICAgICc8L2Rpdj4nOwogIH0KICAvLyBGZXRjaCBmdWxsIGRldGFpbCB0aGVuIHJlbmRlcgogIC8vIEZldGNoIGRldGFpbCBhbmQgY29udGV4dCBpbiBwYXJhbGxlbAogIFByb21pc2UuYWxsKFsKICAgIGZldGNoRGV0YWlsKG5tKSwKICAgIGxheWVyPT09J2F0dGVudGlvbic/ZmV0Y2hTdGF0ZUNvbnRleHQobm0pOlByb21pc2UucmVzb2x2ZShudWxsKQogIF0pLnRoZW4oZnVuY3Rpb24ocmVzdWx0cyl7CiAgICBpZihTRUwhPT1ubSkgcmV0dXJuOwogICAgdmFyIGN0eD1yZXN1bHRzWzFdOwogICAgcmVuZGVyUGFuZWwobm0sIGN0eCk7CiAgICAvLyBVcGRhdGUgbWFwIGNvbG9yCiAgICB2YXIgcGF0aD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK25tKyciXScpOwogICAgaWYocGF0aCYmbGF5ZXI9PT0nZW1vdGlvbicpewogICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBkb209Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgIGlmKGVNYXBbZG9tXSkgcGF0aC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGVNYXBbZG9tXSk7CiAgICB9IGVsc2UgewogICAgICBhcHBseUxheWVyKCk7CiAgICB9CiAgfSkuY2F0Y2goZnVuY3Rpb24oZSl7CiAgICBjb25zb2xlLndhcm4oJ1tzZWxlY3RdJyxlKTsKICAgIGlmKFNFTD09PW5tKSByZW5kZXJQYW5lbChubSwgbnVsbCk7CiAgfSk7Cn0KCmZ1bmN0aW9uIHJlbmRlclBhbmVsKG5tLCBjdHgpewogIHZhciBkPWcobm0pOwogIHZhciBwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYoIXBhbmVsKSByZXR1cm47CiAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogIHZhciBoZWFkZXI9CiAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgJzxkaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtZWsiIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7Ij4nKwogICAgICAgICAgKGxheWVyPT09J2F0dGVudGlvbic/J05hcnJhdGl2ZSBwYW5lbCc6bGF5ZXI9PT0nZW1vdGlvbic/J0Vtb3Rpb25hbCByZWdpc3Rlcic6J01vbWVudHVtIHBhbmVsJykrCiAgICAgICAgICAoZC5jb25maWRlbmNlPyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8ncmdiYSg1MSwyMDQsMTAyLDAuMSknOmQuY29uZmlkZW5jZT09PSdNRURJVU0nPydyZ2JhKDIyNCw5MCw0MCwwLjEpJzoncmdiYSgyNTUsMjU1LDI1NSwwLjA0KScpKyc7Y29sb3I6JysoZC5jb25maWRlbmNlPT09J0hJR0gnPycjMzNjYzY2JzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8nI2UwNWEyOCc6J3JnYmEoMjU1LDI1NSwyNTUsMC4zKScpKyciPicrZC5jb25maWRlbmNlKycgU0lHTkFMPC9zcGFuPic6JycpKwogICAgICAgICAgKGQuaXNfcmVnaW9uYWxfc3Rvcnk/JzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07cGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMSk7Y29sb3I6IzNiYjhkOCI+UkVHSU9OQUwgU1BJS0U8L3NwYW4+JzonJykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhGQVZTLmhhcyhubSk/J29uJzonJykrJyIgZGF0YS1ubT0iJytubSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYodGhpcy5kYXRhc2V0Lm5tKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoRkFWUy5oYXMobm0pPydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgJzwvYnV0dG9uPicrCiAgICAnPC9kaXY+JzsKCiAgdmFyIGJvZHk9Jyc7CgogIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgdmFyIGRTPWQuZGVsdGE+PTA/JysnOicnLGRDPWQuZGVsdGE+PTA/J3VwJzonZG4nOwogICAgdmFyIG5hcnI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0bD0oZC50aW1lbGluZSYmZC50aW1lbGluZS5sZW5ndGgpP2QudGltZWxpbmU6WzAsMCwwLDAsMCwwLDAsZC5hdHRlbnRpb258fDBdOwogICAgdmFyIHRtbj1NYXRoLm1pbi5hcHBseShudWxsLHRsKSx0bXg9TWF0aC5tYXguYXBwbHkobnVsbCx0bCksdHI9TWF0aC5tYXgoMSx0bXgtdG1uKTsKICAgIHZhciB0dz0yNjAsdGg9NjIsdHA9NTsKICAgIHZhciBwdHM9dGwubWFwKGZ1bmN0aW9uKHYsaSl7cmV0dXJuW3RwKyhpLyh0bC5sZW5ndGgtMSkpKih0dy10cCoyKSx0cCsoMS0odi10bW4pL3RyKSoodGgtdHAqMildO30pOwogICAgdmFyIHBEPXB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4oaT09PTA/J00nOidMJykrcFswXS50b0ZpeGVkKDEpKycsJytwWzFdLnRvRml4ZWQoMSk7fSkuam9pbignJyk7CiAgICB2YXIgYUQ9cEQrJyBMJytwdHNbcHRzLmxlbmd0aC0xXVswXSsnLCcrKHRoLXRwKSsnIEwnK3B0c1swXVswXSsnLCcrKHRoLXRwKSsnIFonOwogICAgdmFyIGFjPWFDKGQuYXR0ZW50aW9ufHwwKTsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnSG93IGludGVuc2VseSAnKyhubS5zcGxpdCgiICIpWzBdKSsnIGlzIGJlaW5nIGRpc2N1c3NlZCBuYXRpb25hbGx5LiBTY29yZSBvZiAnK2QuYXR0ZW50aW9uKycgbWVhbnMgJysoZC5hdHRlbnRpb24+NjA/J3ZlcnkgaGlnaCDigJQgZG9taW5hdGVzIG5hdGlvbmFsIGRpc2NvdXJzZSc6ZC5hdHRlbnRpb24+MzU/J2VsZXZhdGVkIOKAlCBjbGVhcmx5IGluIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24nOmQuYXR0ZW50aW9uPjE1Pydtb2RlcmF0ZSDigJQgc29tZSBuYXRpb25hbCBjb3ZlcmFnZSc6ZC5hdHRlbnRpb24+NT8nbG93IOKAlCBsaW1pdGVkIHNpZ25hbHMnOidtaW5pbWFsIOKAlCBmZXcgc2lnbmFscyBkZXRlY3RlZCcpKycuJysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9Imluc2lnaHQiIHN0eWxlPSInKyhjdHg/Jyc6J2JvcmRlci1jb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMDYpJykrJyI+JysKICAgICAgKGN0eCYmY3R4LmJyaWVmCiAgICAgICAgPyBjdHguYnJpZWYrKGN0eC5zb3VyY2U9PT0iYWkiPycnOicnKQogICAgICAgIDogKGQuY29uZmlkZW5jZT09PSJMT1ciJiYhZC5zdW1tYXJ5CiAgICAgICAgICAgID8gJ0xpbWl0ZWQgc2lnbmFscyBmcm9tICcrbm0rJy4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLicKICAgICAgICAgICAgOiBkLnN1bW1hcnl8fCdDb2xsZWN0aW5nIHNpZ25hbHMgZm9yICcrbm0rJy4uLicpKSsKICAgICc8L2Rpdj4nKwogICAgJycrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKG5hcnJbMF0/bmFyclswXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25hcnJbMF0ubmFtZS5zbGljZSgxKTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAobmFyci5sZW5ndGg/CiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrbmFyci5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHZhciBubj1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPjxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25uKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4Ij7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojM2JiOGQ4O2ZvbnQtc2l6ZTo5cHgiPuKGkzwvc3Bhbj4nOicnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdmFsIj4nK3ZhbC50b0ZpeGVkKDEpKyclPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCx2YWwqMi41KSsnJTtiYWNrZ3JvdW5kOicrKG4uZGlyPT09J3VwJz8nI2UwNWEyOCc6bi5kaXI9PT0nZG93bic/JyMzYmI4ZDgnOicjMzM0NDU1JykrJyI+PC9kaXY+PC9kaXY+PC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpKyc8L2Rpdj4nOgogICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb3ctc2lnbmFsIHJlZ2lvbi4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLjwvZGl2PicpKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QXR0ZW50aW9uIOKAlCA4IGRheXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0bC13cmFwIj48c3ZnIHZpZXdCb3g9IjAgMCAnK3R3KycgJyt0aCsnIiBzdHlsZT0id2lkdGg6MTAwJTtoZWlnaHQ6MTAwJSI+JysKICAgICAgICAgICc8ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9InRsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwLjI1Ii8+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAiLz4nKwogICAgICAgICAgJzwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPicrCiAgICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnKSIgLz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrcEQrJyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSInK2FjKyciIHN0cm9rZS13aWR0aD0iMS4yIi8+JysKICAgICAgICAgIHB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gJzxjaXJjbGUgY3g9IicrcFswXSsnIiBjeT0iJytwWzFdKyciIHI9IicrKGk9PT1wdHMubGVuZ3RoLTE/Mi4yOjEuMikrJyIgZmlsbD0iJythYysnIi8+Jzt9KS5qb2luKCcnKSsKICAgICAgICAnPC9zdmc+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JysoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGg/ZC5hcnRpY2xlcy5sZW5ndGg6MCkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtbGlzdCI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLm1hcChmdW5jdGlvbihhKXsKICAgICAgICAgICAgICB2YXIgaXNSZWRkaXQ9YS5zcmMmJmEuc3JjLmluZGV4T2YoJ3JlZGRpdCcpPi0xOwogICAgICAgICAgICAgIHZhciBpc1l0PWEuc3JjJiZhLnNyYy5pbmRleE9mKCd5b3V0dWJlJyk+LTE7CiAgICAgICAgICAgICAgdmFyIGlzUGVvcGxlcz1pc1JlZGRpdHx8aXNZdDsKICAgICAgICAgICAgICB2YXIgc3JjTGFiZWw9aXNQZW9wbGVzPwogICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJjb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjcpO2ZvbnQtc2l6ZTo4cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7bGV0dGVyLXNwYWNpbmc6MC4wNmVtIj52b2ljZXM8L3NwYW4+JzoKICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KSI+JysoIGEuc3JjfHwnJykrJzwvc3Bhbj4nOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBjbGFzcz0iYXJ0LWl0ZW0iIHN0eWxlPSInKyhpc1JlZGRpdD8nYm9yZGVyLWxlZnQ6MnB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMik7cGFkZGluZy1sZWZ0OjhweDsnOicnKSsnIj4nKyAKICAgICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtc3JjIj4nK3NyY0xhYmVsKyc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgY2xhc3M9ImFydC10eHQiPicrKGEudHh0fHxhLnRpdGxlfHwnJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NnB4IDAiPk5vIHNpZ25hbHMgY29sbGVjdGVkIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAvLyBVc2Ugc2FtZSBmdW5jdGlvbnMgYXMgbWFwIOKAlCBndWFyYW50ZWVkIHRvIG1hdGNoCiAgICB2YXIgbWFwRG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgdmFyIGJyZWFrZG93bj1nZXRFbW90aW9uQnJlYWtkb3duKG5tKTsKICAgIHZhciBlbW90aW9ucz1icmVha2Rvd24uZW1vdGlvbnM7CiAgICB2YXIgaGFzRW1vcz0hYnJlYWtkb3duLmVzdGltYXRlZDsKICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW90aW9ucyk7CiAgICB2YXIgZVRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgIGlmKGVUb3Q+MCYmZVRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt9CiAgICB2YXIgdG90PU1hdGgubWF4KDEsZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogICAgZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICBpZighZUwubGVuZ3RoKXtwYW5lbC5pbm5lckhUTUw9aGVhZGVyKyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4Ij5ObyBlbW90aW9uIGRhdGEgeWV0LjwvZGl2Pic7cmV0dXJuO30KICAgIC8vIGRvbUVtbyA9IHNhbWUgYXMgbWFwIGNvbG9yIChmcm9tIGdldEVmZmVjdGl2ZUVtb3Rpb24pCiAgICB2YXIgZG9tRW1vPW1hcERvbUVtbzsKICAgIC8vIFJlb3JkZXIgZUwgc28gZG9taW5hbnQgc2hvd3MgZmlyc3QKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXsKICAgICAgaWYoYVswXT09PWRvbUVtbykgcmV0dXJuIC0xOwogICAgICBpZihiWzBdPT09ZG9tRW1vKSByZXR1cm4gMTsKICAgICAgcmV0dXJuIGJbMV0tYVsxXTsKICAgIH0pOwogICAgdmFyIGRvbVBjdD1NYXRoLnJvdW5kKChlTFswXT9lTFswXVsxXToyMCkqMTAwL3RvdCk7CiAgICB2YXIgbmFycjI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0b3BOYXJTdHI9bmFycjIuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCcgYW5kICcpOwogICAgdmFyIHdoYXRJdD17YW54aWV0eTonVW5jZXJ0YWludHkgYW5kIHVuZWFzZSBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gU2lnbmFsczogJyt0b3BOYXJTdHIrJy4nOicnKSxhbmdlcjonT3V0cmFnZSBhbmQgcHJlc3N1cmUgaW4gJytubSsodG9wTmFyU3RyPycuIERyaXZlbiBieTogJyt0b3BOYXJTdHIrJy4nOicnKSxob3BlOidPcHRpbWlzbSBhbmQgcHJvZ3Jlc3MgaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKSxwcmlkZTonSWRlbnRpdHkgYW5kIGFjaGlldmVtZW50IGluICcrbm0rKHRvcE5hclN0cj8nLiBBcm91bmQ6ICcrdG9wTmFyU3RyKycuJzonJyksZmVhcjonVGhyZWF0IHBlcmNlcHRpb24gaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKX07CiAgICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7Y3VtQT1hMjsKICAgICAgdmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICAgIH0pLmpvaW4oJycpOwogICAgdmFyIGVkZXNjPXthbnhpZXR5OidVbmNlcnRhaW50eSwgd29ycnknLGFuZ2VyOidPdXRyYWdlLCBwcm90ZXN0Jyxob3BlOidPcHRpbWlzbSwgcHJvZ3Jlc3MnLHByaWRlOidBY2hpZXZlbWVudCwgaWRlbnRpdHknLGZlYXI6J1RocmVhdCwgaW5zZWN1cml0eSd9OwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdUaGUgZW1vdGlvbmFsIHVuZGVyY3VycmVudCBvZiBzaWduYWxzIGZyb20gJytubSsnLiBXaGF0IHRvbmUgZG9taW5hdGVzIHRoZSBwb2xpdGljYWwgZGlzY291cnNlIOKAlCBvdXRyYWdlLCBob3BlLCBmZWFyLCBvciBhbnhpZXR5PycrCiAgICAnPC9kaXY+JysKICAgICghaGFzRW1vcz8nPGRpdiBzdHlsZT0icGFkZGluZzo2cHggMTFweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO21hcmdpbi1ib3R0b206MTBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KSI+RXN0aW1hdGVkIGZyb20gc2lnbmFsIGRpcmVjdGlvbiDigJQgbGltaXRlZCBkaXJlY3QgZW1vdGlvbiBkYXRhLjwvZGl2Pic6JycpKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrcGFsW2RvbUVtb10rJzE0O2JvcmRlcjoxcHggc29saWQgJytwYWxbZG9tRW1vXSsnMzM7bWFyZ2luLWJvdHRvbToxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6JytwYWxbZG9tRW1vXSsnO21hcmdpbi1ib3R0b206NnB4Ij5Eb21pbmFudCBlbW90aW9uPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyNnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHgiPicrZG9tUGN0KyclIMK3ICcrbm0rJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo4cHg7bGluZS1oZWlnaHQ6MS41O2ZvbnQtc3R5bGU6aXRhbGljIj4nK3doYXRJdFtkb21FbW9dKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RW1vdGlvbmFsIGJyZWFrZG93bjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE2cHg7Ij4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDc2IDc2IiBzdHlsZT0id2lkdGg6NzJweDtoZWlnaHQ6NzJweDtmbGV4LXNocmluazowIj4nK2FyY3MrJzwvc3ZnPicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjVweDsiPicrCiAgICAgICAgICAgIGVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgICAgICAgICAgdmFyIGs9a3ZbMF0sdj1rdlsxXSxwY3Q9TWF0aC5yb3VuZCh2KjEwMC90b3QpOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpjZW50ZXI7bWFyZ2luLWJvdHRvbToycHg7Ij4nKwogICAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OyI+PHNwYW4gc3R5bGU9IndpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6MnB4O2JhY2tncm91bmQ6JytwYWxba10rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjonKyhrPT09ZG9tRW1vPyd2YXIoLS1pbmspJzondmFyKC0tZGltKScpKyciPicray5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStrLnNsaWNlKDEpKyc8L3NwYW4+PC9kaXY+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWluaykiPicrcGN0KyclPC9zcGFuPicrCiAgICAgICAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrcGN0KyclO2JhY2tncm91bmQ6JytwYWxba10rJztvcGFjaXR5OjAuNztib3JkZXItcmFkaXVzOjFweCI+PC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAgIChrPT09ZG9tRW1vPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytlZGVzY1trXSsnPC9kaXY+JzonJykrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWwgaGVhZGxpbmVzPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NHB4OyI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLnNsaWNlKDAsNSkubWFwKGZ1bmN0aW9uKGEpewogICAgICAgICAgICAgIHZhciBlQ29sb3I9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjZweDtwYWRkaW5nOjZweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ij4nKwogICAgICAgICAgICAgICAgKGEuZW1vdGlvbj8nPHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytlQ29sb3JbYS5lbW90aW9uXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrO21hcmdpbi10b3A6NXB4O2ZsZXgtc2hyaW5rOjAiPjwvc3Bhbj4nOicnKSsKICAgICAgICAgICAgICAgICc8ZGl2PjxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS40Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHgiPicrKGEuc3JjfHwnJykrKGEuZW1vdGlvbj8nIMK3ICcrYS5lbW90aW9uOicnKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJyk6CiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo0cHggMCI+Tm8gc2lnbmFscyB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgewogICAgdmFyIHZlbD1kLnZlbG9jaXR5fHwwOwogICAgdmFyIHZlbERpcj12ZWw+MC4xNT8nUmlzaW5nIGZhc3QnOnZlbD4wLjA1PydSaXNpbmcnOnZlbDwtMC4xPydDb29saW5nIGZhc3QnOnZlbDwtMC4wMj8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICB2YXIgdmVsQ29sPXZlbD4wLjA1PycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgIHZhciB2ZWxEZXNjPXsnUmlzaW5nIGZhc3QnOidTaWduYWwgdm9sdW1lIHN1cmdpbmcuJywnUmlzaW5nJzonQXR0ZW50aW9uIGJ1aWxkaW5nLicsJ1N0YWJsZSc6J0JhbGFuY2VkIG1vbWVudHVtLicsJ0Nvb2xpbmcnOidBdHRlbnRpb24gZmFkaW5nLicsJ0Nvb2xpbmcgZmFzdCc6J1NoYXJwIHNpZ25hbCBkZWNheS4nfTsKICAgIHZhciBuYXJyMz1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHJpc2luZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSd1cCc7fSk7CiAgICB2YXIgZmFsbGluZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSdkb3duJzt9KTsKICAgIHZhciBjdHg9Jyc7CiAgICBpZih2ZWw+MC4wNSYmcmlzaW5nTmFycy5sZW5ndGgpIGN0eD0nRHJpdmVuIGJ5IHJpc2luZyBzaWduYWxzIGFyb3VuZCA8c3Ryb25nPicrcmlzaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+Lic7CiAgICBlbHNlIGlmKHZlbDwtMC4wNSYmZmFsbGluZ05hcnMubGVuZ3RoKSBjdHg9J1NpZ25hbHMgYXJvdW5kIDxzdHJvbmc+JytmYWxsaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+IGxvc2luZyB0cmFjdGlvbi4nOwogICAgZWxzZSBjdHg9J1NpZ25hbCB2b2x1bWUgJysodmVsPjAuMDI/J2J1aWxkaW5nJzonc3RhYmxlJykrJyBpbiAnK25tKycuJzsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnSXMgYXR0ZW50aW9uIGZvciAnK25tKycgZ3Jvd2luZyBvciBmYWRpbmc/IFJpc2luZyBtb21lbnR1bSBtZWFucyBhIG5hcnJhdGl2ZSBpcyBhY2NlbGVyYXRpbmcuIENvb2xpbmcgbWVhbnMgdGhlIHN0b3J5IGlzIGxvc2luZyB0cmFjdGlvbi4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrdmVsQ29sKycxNDtib3JkZXI6MXB4IHNvbGlkICcrdmVsQ29sKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3ZlbENvbCsnO21hcmdpbi1ib3R0b206NnB4Ij5TaWduYWwgbW9tZW50dW08L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo4cHg7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MzJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjonK3ZlbENvbCsnO2ZvbnQtd2VpZ2h0OjUwMCI+Jyt2ZWxEaXIrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC1zdHlsZTppdGFsaWM7bGluZS1oZWlnaHQ6MS41Ij4nK3ZlbERlc2NbdmVsRGlyXSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjttYXJnaW4tdG9wOjEwcHg7cGFkZGluZy10b3A6MTBweDtib3JkZXItdG9wOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIj4nK2N0eCsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlZlbG9jaXR5PC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIiBzdHlsZT0iZm9udC1zaXplOjE4cHg7Y29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCDOtDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrKGQuZGVsdGE+PTA/J3VwJzonZG4nKSsnIj4nKyhkLmRlbHRhPj0wPycrJzonJykrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAocmlzaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QWNjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICByaXNpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojZTA1YTI4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJykrCiAgICAgIChmYWxsaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RGVjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICBmYWxsaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoNTksMTg0LDIxNiwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojM2JiOGQ4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJyk7CiAgfQoKICBwYW5lbC5pbm5lckhUTUw9aGVhZGVyK2JvZHk7Cn0KCgpmdW5jdGlvbiB0b2dnbGVGYXYobm0pewogIGlmKEZBVlMuaGFzKG5tKSkgRkFWUy5kZWxldGUobm0pO2Vsc2UgRkFWUy5hZGQobm0pOwogIHJlbmRlclBhbmVsKFNFTCk7cmVuZGVyRmF2cygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdnMoKXsKICB2YXIgcm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXYtcm93Jyk7CiAgaWYoIUZBVlMuc2l6ZSl7cm93LmlubmVySFRNTD0nPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+JztyZXR1cm47fQogIHJvdy5pbm5lckhUTUw9QXJyYXkuZnJvbShGQVZTKS5tYXAoZnVuY3Rpb24obm0pewogICAgdmFyIGQ9ZyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8nI2UwNWEyOCc6JyMzYmI4ZDgnOwogICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZmF2LWNhcmQiIG9uY2xpY2s9InNlbGVjdF8oXCcnK25tKydcJykiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1oZWFkIj48c3BhbiBjbGFzcz0iZmMtbmFtZSI+JytubSsnPC9zcGFuPjxzcGFuIGNsYXNzPSJmYy1zYyI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPk5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj4yNGg8L3NwYW4+PHNwYW4gY2xhc3M9InYiIHN0eWxlPSJjb2xvcjonK2RDKyciPicrZFMrZC5kZWx0YSsnPC9zcGFuPjwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKGMpewogIGMuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7bGF5ZXI9Yy5kYXRhc2V0LmxheWVyO2FwcGx5TGF5ZXIoKTsKICB9KTsKfSk7CgpmdW5jdGlvbiB1cGRhdGVDbG9jaygpewogIHZhciBub3c9bmV3IERhdGUoKSxpc3Q9bmV3IERhdGUobm93LmdldFRpbWUoKStub3cuZ2V0VGltZXpvbmVPZmZzZXQoKSo2MDAwMCsxOTgwMDAwMCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Nsb2NrJykudGV4dENvbnRlbnQ9U3RyaW5nKGlzdC5nZXRIb3VycygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0TWludXRlcygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0U2Vjb25kcygpKS5wYWRTdGFydCgyLCcwJykrJyBJU1QnOwp9CnNldEludGVydmFsKHVwZGF0ZUNsb2NrLDEwMDApO3VwZGF0ZUNsb2NrKCk7CgovLyBJTklUIOKAlCB3YWl0IGZvciBET00KLy8gaSBidXR0b24gdG9vbHRpcCDigJQgdXNlcyBmaXhlZCBwb3NpdGlvbmluZyBzbyBpdCdzIG5ldmVyIGNsaXBwZWQKKGZ1bmN0aW9uKCl7CiAgdmFyIHRpcD1udWxsOwogIGZ1bmN0aW9uIHNob3dUaXAoZSl7CiAgICBpZighdGlwKXt0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2x0YWItdG9vbHRpcCcpO30KICAgIHZhciB0eHQ9dGhpcy5nZXRBdHRyaWJ1dGUoJ2RhdGEtdGlwJyk7CiAgICBpZighdHh0fHwhdGlwKSByZXR1cm47CiAgICB0aXAudGV4dENvbnRlbnQ9dHh0OwogICAgdGlwLmNsYXNzTGlzdC5hZGQoJ3Zpc2libGUnKTsKICAgIHZhciByZWN0PXRoaXMuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICB2YXIgdHc9MjQwOwogICAgdmFyIGxlZnQ9TWF0aC5taW4ocmVjdC5sZWZ0LHdpbmRvdy5pbm5lcldpZHRoLXR3LTEwKTsKICAgIHRpcC5zdHlsZS5sZWZ0PWxlZnQrJ3B4JzsKICAgIHRpcC5zdHlsZS50b3A9KHJlY3QudG9wLTEwLXRpcC5vZmZzZXRIZWlnaHR8fHJlY3QudG9wLTgwKSsncHgnOwogICAgLy8gUmVwb3NpdGlvbiBhZnRlciByZW5kZXIKICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmdW5jdGlvbigpewogICAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC10aXAub2Zmc2V0SGVpZ2h0LTgpKydweCc7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gaGlkZVRpcCgpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICBpZih0aXApIHRpcC5jbGFzc0xpc3QucmVtb3ZlKCd2aXNpYmxlJyk7CiAgfQogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3ZlcicsZnVuY3Rpb24oZSl7CiAgICBpZihlLnRhcmdldC5jbGFzc0xpc3QuY29udGFpbnMoJ2x0YWItaW5mbycpKSBzaG93VGlwLmNhbGwoZS50YXJnZXQsZSk7CiAgfSk7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VvdXQnLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgaGlkZVRpcCgpOwogIH0pOwp9KSgpOwoKLy8g4pSA4pSAIE1PQklMRSBCT1RUT00gU0hFRVQg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACi8vIE9ubHkgYWN0aXZhdGVzIG9uIG1vYmlsZSDigJQgbm8gZWZmZWN0IG9uIGRlc2t0b3AKKGZ1bmN0aW9uKCl7CiAgdmFyIGlzTW9iaWxlPWZ1bmN0aW9uKCl7cmV0dXJuIHdpbmRvdy5pbm5lcldpZHRoPD03Njg7fTsKICB2YXIgb3ZlcmxheT1kb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTsKICBvdmVybGF5LmNsYXNzTmFtZT0nbWFwLW92ZXJsYXktZGltJzsKICBkb2N1bWVudC5ib2R5LmFwcGVuZENoaWxkKG92ZXJsYXkpOwoKICBmdW5jdGlvbiBvcGVuUGFuZWwoKXsKICAgIGlmKCFpc01vYmlsZSgpKSByZXR1cm47CiAgICB2YXIgcGFuZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLnN0YXRlLXBhbmVsJyk7CiAgICBpZihwYW5lbCl7cGFuZWwuY2xhc3NMaXN0LmFkZCgncGFuZWwtb3BlbicpO30KICAgIG92ZXJsYXkuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7CiAgICBkb2N1bWVudC5ib2R5LnN0eWxlLm92ZXJmbG93PSdoaWRkZW4nOwogIH0KICBmdW5jdGlvbiBjbG9zZVBhbmVsKCl7CiAgICB2YXIgcGFuZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLnN0YXRlLXBhbmVsJyk7CiAgICBpZihwYW5lbCl7cGFuZWwuY2xhc3NMaXN0LnJlbW92ZSgncGFuZWwtb3BlbicpO30KICAgIG92ZXJsYXkuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7CiAgICBkb2N1bWVudC5ib2R5LnN0eWxlLm92ZXJmbG93PScnOwogIH0KCiAgLy8gT3BlbiBwYW5lbCB3aGVuIHN0YXRlIHNlbGVjdGVkCiAgdmFyIG9yaWdTZWxlY3Q9d2luZG93LnNlbGVjdF87CiAgd2luZG93LnNlbGVjdF89ZnVuY3Rpb24obm0pewogICAgb3JpZ1NlbGVjdChubSk7CiAgICBpZihpc01vYmlsZSgpKSBzZXRUaW1lb3V0KG9wZW5QYW5lbCw1MCk7CiAgfTsKCiAgLy8gQ2xvc2Ugb24gb3ZlcmxheSB0YXAKICBvdmVybGF5LmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxjbG9zZVBhbmVsKTsKCiAgLy8gQ2xvc2Ugb24gc3dpcGUgZG93bgogIHZhciBzdGFydFk9MDsKICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCd0b3VjaHN0YXJ0JyxmdW5jdGlvbihlKXsKICAgIHN0YXJ0WT1lLnRvdWNoZXNbMF0uY2xpZW50WTsKICB9LHtwYXNzaXZlOnRydWV9KTsKICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCd0b3VjaGVuZCcsZnVuY3Rpb24oZSl7CiAgICB2YXIgcGFuZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLnN0YXRlLXBhbmVsJyk7CiAgICBpZighcGFuZWx8fCFwYW5lbC5jbGFzc0xpc3QuY29udGFpbnMoJ3BhbmVsLW9wZW4nKSkgcmV0dXJuOwogICAgdmFyIGR5PWUuY2hhbmdlZFRvdWNoZXNbMF0uY2xpZW50WS1zdGFydFk7CiAgICBpZihkeT42MCkgY2xvc2VQYW5lbCgpOyAvLyBzd2lwZSBkb3duIDYwcHggdG8gY2xvc2UKICB9LHtwYXNzaXZlOnRydWV9KTsKfSkoKTsKCmZ1bmN0aW9uIGRpc21pc3NMb2FkZXIoKXsKICB2YXIgbGRyPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhcHAtbG9hZGVyJyk7CiAgaWYoIWxkcikgcmV0dXJuOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7bGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnO30sODAwKTsKfQoKZnVuY3Rpb24gaW5pdCgpewogIHJlbmRlclN0cmlwKCczbScpOwoKICAvLyBTdGVwIDE6IExvYWQgZGF0YSBhbmQgbWFwIElOIFBBUkFMTEVMIGZvciBzcGVlZAogIHZhciBkYXRhUHJvbWlzZSA9IGZldGNoRnVsbFNuYXBzaG90KCk7CiAgdmFyIG1hcFByb21pc2UgPSBuZXcgUHJvbWlzZShmdW5jdGlvbihyZXNvbHZlKXsKICAgIHZhciBhdHRlbXB0cz0wOwogICAgZnVuY3Rpb24gdHJ5TG9hZCgpewogICAgICBpZih0eXBlb2YgdG9wb2pzb249PT0ndW5kZWZpbmVkJyl7CiAgICAgICAgaWYoYXR0ZW1wdHMrKzwxNSl7c2V0VGltZW91dCh0cnlMb2FkLDIwMCk7fSBlbHNlIHJlc29sdmUoZmFsc2UpOwogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICBsb2FkTWFwKCkudGhlbihyZXNvbHZlKS5jYXRjaChmdW5jdGlvbigpe3Jlc29sdmUoZmFsc2UpO30pOwogICAgfQogICAgdHJ5TG9hZCgpOwogIH0pOwoKICAvLyBTdGVwIDI6IFdoZW4gQk9USCBkYXRhIGFuZCBtYXAgcmVhZHkg4oCUIGFwcGx5IGNvbG9ycyBpbW1lZGlhdGVseQogIFByb21pc2UuYWxsKFtkYXRhUHJvbWlzZSwgbWFwUHJvbWlzZV0pLnRoZW4oZnVuY3Rpb24ocmVzdWx0cyl7CiAgICB2YXIgZGF0YU9rPXJlc3VsdHNbMF07CiAgICBpZihkYXRhT2spewogICAgICBhcHBseUxheWVyKCk7CiAgICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgICBidWlsZExvY2FsSW5zaWdodCgpOwogICAgfQogICAgZGlzbWlzc0xvYWRlcigpOwogICAgLy8gU3RhcnQgcG9sbGluZyBhZnRlciBpbml0aWFsIHJlbmRlcgogICAgc2V0VGltZW91dChmdW5jdGlvbigpe3N0YXJ0UG9sbGluZygpO30sNTAwKTsKICB9KTsKCiAgLy8gRmFsbGJhY2s6IGRpc21pc3MgbG9hZGVyIGFmdGVyIDRzIG5vIG1hdHRlciB3aGF0CiAgc2V0VGltZW91dChkaXNtaXNzTG9hZGVyLCA0MDAwKTsKCiAgLy8gTG9hZCBpbnNpZ2h0cyBhZnRlciBkYXRhIHNldHRsZXMKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIGZldGNoTmFycmF0aXZlSW5zaWdodCgpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7CiAgICBsb2FkSGlzdG9yeURhdGEoNyk7CiAgfSwzMDAwKTsKfQppZihkb2N1bWVudC5yZWFkeVN0YXRlPT09J2xvYWRpbmcnKXsKICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgaW5pdCk7Cn0gZWxzZSB7CiAgLy8gQWxyZWFkeSBsb2FkZWQg4oCUIGJ1dCB3YWl0IG9uZSB0aWNrIHRvIGVuc3VyZSBhbGwgc2NyaXB0cyBwYXJzZWQKICBzZXRUaW1lb3V0KGluaXQsIDApOwp9CgoKLy8g4pSA4pSAIElORElBOiBMQVNUIDI0IEhPVVJTIOKAlCA0LWhvdXIgc25hcHNob3RzIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAp2YXIgRU1PX0NPTE9SUz17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CnZhciBFTU9fQkc9e2FueGlldHk6J3JnYmEoMTM2LDY4LDIwNCwwLjEpJyxhbmdlcjoncmdiYSgyMjEsMzQsNjgsMC4xKScsaG9wZToncmdiYSg1MSwyMDQsMTAyLDAuMSknLHByaWRlOidyZ2JhKDUxLDE3MCwyMDQsMC4xKScsZmVhcjoncmdiYSgyMDQsMTM2LDUxLDAuMSknfTsKCmFzeW5jIGZ1bmN0aW9uIGxvYWRQdWxzZTI0KCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvcHVsc2Utc25hcHNob3RzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICB2YXIgc25hcHM9ZC5zbmFwc2hvdHN8fFtdOwogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwMjQtY2FyZHMnKTsKICAgIGlmKCFlbCkgcmV0dXJuOwogICAgaWYoIXNuYXBzLmxlbmd0aCl7CiAgICAgIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0icDI0LWVtcHR5Ij5TaWduYWxzIGFyZSBzdGlsbCBiZWluZyBjb2xsZWN0ZWQuIENoZWNrIGJhY2sgc2hvcnRseS48L2Rpdj4nOwogICAgICByZXR1cm47CiAgICB9CiAgICBlbC5pbm5lckhUTUw9c25hcHMubWFwKGZ1bmN0aW9uKHMpewogICAgICB2YXIgZW1vPXMuZG9taW5hbnRfZW1vdGlvbjsKICAgICAgdmFyIGVDb2w9ZW1vP0VNT19DT0xPUlNbZW1vXToncmdiYSgxNjAsMTkwLDIzMCwwLjQpJzsKICAgICAgdmFyIGVCZz1lbW8/RU1PX0JHW2Vtb106J3JnYmEoMjU1LDI1NSwyNTUsMC4wMiknOwogICAgICB2YXIgbmFyPXMucHJpbWFyeV9uYXJyYXRpdmV8fCfigJQnOwogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9InAyNC1jYXJkIiBzdHlsZT0iYm9yZGVyLWxlZnQtY29sb3I6JytlQ29sKyciPicrCiAgICAgICAgJzxkaXYgc3R5bGU9InBvc2l0aW9uOmFic29sdXRlO2xlZnQ6MDt0b3A6MDtib3R0b206MDt3aWR0aDoycHg7YmFja2dyb3VuZDonK2VDb2wrJyI+PC9kaXY+JysKICAgICAgICAvLyBUaW1lIGxhYmVsCiAgICAgICAgJzxkaXYgY2xhc3M9InAyNC1jYXJkLXRpbWUiPicrcy53aW5kb3dfc3RhcnQrJyDigJMgJytzLndpbmRvd19lbmQrJyZuYnNwOyZuYnNwOycrcy5sYWJlbCsnPC9kaXY+JysKICAgICAgICAvLyBQcmltYXJ5IG5hcnJhdGl2ZSDigJQgdGhlIGJpZyBpbnNpZ2h0CiAgICAgICAgJzxkaXYgY2xhc3M9InAyNC1jYXJkLW5hciI+JytuYXIuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbmFyLnNsaWNlKDEpKyc8L2Rpdj4nKwogICAgICAgIC8vIEluc2lnaHQgc2VudGVuY2UKICAgICAgICAocy5pbnNpZ2h0Pyc8ZGl2IGNsYXNzPSJwMjQtY2FyZC1pbnNpZ2h0Ij4nK3MuaW5zaWdodCsnPC9kaXY+JzonJykrCiAgICAgICAgLy8gSG90dGVzdCBzdGF0ZQogICAgICAgIChzLmhvdHRlc3Rfc3RhdGU/CiAgICAgICAgICAnPGRpdiBjbGFzcz0icDI0LWNhcmQtc3RhdGUiPicrCiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJ3aWR0aDo2cHg7aGVpZ2h0OjZweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7ZmxleC1zaHJpbms6MCI+PC9kaXY+JysKICAgICAgICAgICAgJzxzcGFuIGNsYXNzPSJwMjQtY2FyZC1zdGF0ZS1sYWJlbCI+Q2VudHJlIG9mIGF0dGVudGlvbjwvc3Bhbj4nKwogICAgICAgICAgICAnPHNwYW4gY2xhc3M9InAyNC1jYXJkLXN0YXRlLW5hbWUiPicrcy5ob3R0ZXN0X3N0YXRlKyc8L3NwYW4+JysKICAgICAgICAgICAgKHMuc2Vjb25kX3N0YXRlPyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCkiPisgJytzLnNlY29uZF9zdGF0ZSsnPC9zcGFuPic6JycpKwogICAgICAgICAgJzwvZGl2Pic6JycpKwogICAgICAgIC8vIE5hcnJhdGl2ZSB0YWdzCiAgICAgICAgJzxkaXYgY2xhc3M9InAyNC1jYXJkLW5hcnMiPicrCiAgICAgICAgICBzLm5hcnJhdGl2ZXMuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InAyNC1jYXJkLW5hci10YWciPicrbisnPC9zcGFuPic7CiAgICAgICAgICB9KS5qb2luKCcnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAvLyBGb290ZXIKICAgICAgICAnPGRpdiBjbGFzcz0icDI0LWNhcmQtZm9vdGVyIj4nKwogICAgICAgICAgKGVtbz8nPHNwYW4gY2xhc3M9InAyNC1jYXJkLWVtbyIgc3R5bGU9ImJhY2tncm91bmQ6JytlQmcrJztjb2xvcjonK2VDb2wrJyI+JytlbW8rJzwvc3Bhbj4nOic8c3Bhbj48L3NwYW4+JykrCiAgICAgICAgICAnPHNwYW4gY2xhc3M9InAyNC1jYXJkLXNpZ3MiPicrcy5zaWduYWxfY291bnQrJyBzaWduYWxzPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbcHVsc2UyNF0nLGUubWVzc2FnZSk7CiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3AyNC1jYXJkcycpOwogICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0icDI0LWVtcHR5Ij5VbmFibGUgdG8gbG9hZCBzbmFwc2hvdHMuPC9kaXY+JzsKICB9Cn0KCnNldFRpbWVvdXQoZnVuY3Rpb24oKXtsb2FkUHVsc2UyNCgpO30sMjUwMCk7CgpzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgLy8gQXV0by1zZWxlY3QgaG90dGVzdCBzdGF0ZSBmcm9tIExJVkUgZGF0YQogIHZhciBzcmM9T2JqZWN0LmtleXMoTElWRSkubGVuZ3RoP0xJVkU6U0Q7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhzcmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKHRvcCl7CiAgICB2YXIgZWw9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJyt0b3BbMF0rJyJdJyk7CiAgICBpZihlbCkgc2VsZWN0Xyh0b3BbMF0pOwogIH0KfSwzMDAwKTsKc2V0VGltZW91dChyZW5kZXJGYXZzLDI0MDApOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="

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
