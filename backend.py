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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4fQoudG9wYmFyLXNvdXJjZXtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoycHg7dGV4dC1hbGlnbjpyaWdodH0KLnRzLW9ic2VydmVke2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSl9Ci50cy1zb3VyY2Vze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtjb2xvcjpyZ2JhKDEyMCwxNTAsMTgwLDAuMzUpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KLnRvcGJhci1zaWduYWxze2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjVweDtmbGV4LXNocmluazowfQoudHMtc2lnLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLmxpdmUtaW5kaWNhdG9yewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdweDsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA1ZW07Cn0KLmxpdmUtZG90e3dpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6IzRhZGU4MDtib3gtc2hhZG93OjAgMCA4cHggcmdiYSg3NCwyMjIsMTI4LDAuNyk7YW5pbWF0aW9uOmxkIDIuNXMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgbGR7MCUsMTAwJXtvcGFjaXR5OjE7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuMzU7dHJhbnNmb3JtOnNjYWxlKDAuOCl9fQouY2xvY2t7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW19CgovKiBIRVJPICovCi5oZXJvewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBwYWRkaW5nOjcycHggMzZweCAwOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKfQouaGVyby1leWVicm93e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4zMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToyNHB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEwcHh9Ci5oZXJvLWV5ZWJyb3c6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjE2cHg7aGVpZ2h0OjFweDtiYWNrZ3JvdW5kOnZhcigtLWZhaW50KTtvcGFjaXR5OjAuNX0KLmhlcm8tYnJhbmQtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtc2l6ZTpjbGFtcCgzNnB4LDQuMnZ3LDY0cHgpO2xpbmUtaGVpZ2h0OjE7bGV0dGVyLXNwYWNpbmc6LTAuMDNlbTtjb2xvcjp2YXIoLS1pbmspO21hcmdpbjowfQouaGVyby1icmFuZC1uYW1lIGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOiNlOGM0YTA7YW5pbWF0aW9uOnB1bHNlTmFtZUdsb3cgNXMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHVsc2VOYW1lR2xvd3swJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmhlcm8tdGFnbGluZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE1cHgsMS41dncsMjBweCk7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS40O2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luOjAgMCAxMnB4IDA7bWF4LXdpZHRoOjQ4MHB4O3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tZGVzY3tmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTNweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjEuNjttYXgtd2lkdGg6NDAwcHg7bWFyZ2luOjAgMCA2cHggMDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLXN1Yi1saW5le2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoNjIsNzcsOTYsMC42KTttYXJnaW46MCAwIDIwcHggMDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLXB1bHNlLXNpZ25hbHtwb3NpdGlvbjpyZWxhdGl2ZTt3aWR0aDoxNnB4O2hlaWdodDoxNnB4O2ZsZXgtc2hyaW5rOjB9Ci5ocHMtY29yZXtwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC45O2FuaW1hdGlvbjpocHNDb3JlIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGhwc0NvcmV7MCUsMTAwJXtvcGFjaXR5OjAuOTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC40O3RyYW5zZm9ybTpzY2FsZSgwLjc1KX19Ci5ocHMtcmluZ3twb3NpdGlvbjphYnNvbHV0ZTtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWFjY2VudCk7YW5pbWF0aW9uOmhwc1JpbmcgNHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5ocHMtcmluZy5yMXtpbnNldDoxcHg7YW5pbWF0aW9uLWRlbGF5OjBzfS5ocHMtcmluZy5yMntpbnNldDotM3B4O2FuaW1hdGlvbi1kZWxheToxLjRzO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjM1KX0KQGtleWZyYW1lcyBocHNSaW5nezAle29wYWNpdHk6MC42O3RyYW5zZm9ybTpzY2FsZSgwLjcpfTEwMCV7b3BhY2l0eTowO3RyYW5zZm9ybTpzY2FsZSgxLjYpfX0KCi8qIFNJR05BVFVSRSBJTlNJR0hUICovCi5zaWduYXR1cmUtaW5zaWdodHsKICBtYXJnaW4tdG9wOjA7CiAgcGFkZGluZzoxNHB4IDIwcHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjE0cHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTM1ZGVnLCByZ2JhKDIyNCw5MCw0MCwwLjA2KSAwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDMpIDEwMCUpOwogIGJhY2tkcm9wLWZpbHRlcjpibHVyKDhweCk7CiAgbWF4LXdpZHRoOjkwMHB4OwogIHBvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjsKfQouc2lnbmF0dXJlLWluc2lnaHQ6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2xlZnQ6MDt0b3A6MDtib3R0b206MDt3aWR0aDoycHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gYm90dG9tLCB2YXIoLS1hY2NlbnQpLCB0cmFuc3BhcmVudCk7Cn0KLnNpLWxhYmVsewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjI1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWFjY2VudCk7bWFyZ2luLWJvdHRvbToxMHB4Owp9Ci5zaS10ZXh0ewogIGZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTRweCwxLjR2dywxOHB4KTsKICBmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQouc2ktdGV4dCBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1hY2NlbnQpfQouc2ktc3ViewogIG1hcmdpbi10b3A6MTBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pOwogIGxldHRlci1zcGFjaW5nOjAuMDRlbTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNHB4O2ZsZXgtd3JhcDp3cmFwOwp9Ci5zaS10YWd7CiAgcGFkZGluZzoycHggOHB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpOwogIGZvbnQtc2l6ZTo5LjVweDsKfQoKLyogTkFSUkFUSVZFIFNUUklQICovCgouc3RyaXAtdGFiewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6NHB4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjpub25lO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5zdHJpcC10YWIuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEyKX0KLnN0cmlwLXRhYjpob3Zlcntjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sewogIGZsZXg6MTtiYWNrZ3JvdW5kOnZhcigtLWJnMSk7cGFkZGluZzowOwp9Ci5zdHJpcC1jb2wtaGVhZHsKICBwYWRkaW5nOjEwcHggMTZweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Cn0KLnN0cmlwLWNvbC1oZWFkLmZhZGV7Y29sb3I6dmFyKC0tZmFsbCl9Ci5zdHJpcC1jb2wtaGVhZC5yaXNlMntjb2xvcjp2YXIoLS1yaXNlKX0KLnN0cmlwLWNvbC1oZWFkLnNoaWZ0e2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2wtYm9keXtwYWRkaW5nOjEycHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo4cHh9Ci5zdHJpcC1pdGVtewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtnYXA6OHB4Owp9Ci5zdHJpcC10b3BpY3tmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnN0cmlwLW5vdGV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnN0cmlwLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O2ZvbnQtc2l6ZToxNHB4O2ZsZXgtc2hyaW5rOjB9CgovKiBNQUlOIExBWU9VVCAqLwoubWFpbnsKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwogIHBhZGRpbmc6MCAzNnB4IDI4cHg7CiAgZGlzcGxheTpncmlkOwogIGdyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMzYwcHg7CiAgZ2FwOjE0cHg7CiAgbWluLXdpZHRoOjA7Cn0KCi8qIE1BUCAqLwoubWFwLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNnB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE2cHgpOwogIG92ZXJmbG93OmhpZGRlbjtwb3NpdGlvbjpyZWxhdGl2ZTsKfQoubWFwLWNhcmQ6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7CiAgYmFja2dyb3VuZDoKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDcwJSA1MCUgYXQgMzUlIDAlLCByZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSwgdHJhbnNwYXJlbnQgNjAlKSwKICAgIHJhZGlhbC1ncmFkaWVudChlbGxpcHNlIDUwJSA0MCUgYXQgODAlIDEwMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAwJSwgdHJhbnNwYXJlbnQgNjAlKTsKfQoubWFwLXRvcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOjEycHggMThweCAwOwp9Ci5tYXAtdGl0bGUtYmxvY2sgLm10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTdweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLm1hcC10aXRsZS1ibG9jayAubXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO21hcmdpbi10b3A6MnB4fQoubGVnZW5ke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmxlZ2VuZC1iYXJ7CiAgaGVpZ2h0OjNweDt3aWR0aDo4MHB4O2JvcmRlci1yYWRpdXM6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIHJpZ2h0LCMwZTIwMzUsIzFhNTU4MCAyNSUsIzhhNWMxOCA1NSUsI2MwMzgxYSA4MCUsI2UwMTAyMCk7Cn0KLmxheWVyLXJvd3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OwogIHBhZGRpbmc6MTBweCAyMHB4IDZweDsKfQoubGF5ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQoubHRhYnN7ZGlzcGxheTpmbGV4O2dhcDozcHh9Ci5sdGFiewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzozcHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLmx0YWIuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA4KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yKX0KLmx0YWJ7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjVweDtwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzp2aXNpYmxlfQoubHRhYi1pbmZve3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMik7Zm9udC1zaXplOjhweDtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXN0eWxlOml0YWxpYztmb250LXdlaWdodDo2MDA7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjM1KTtkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2N1cnNvcjpoZWxwO2ZsZXgtc2hyaW5rOjA7dHJhbnNpdGlvbjphbGwgMC4xNXM7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxMDB9Ci5sdGFiLWluZm86aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWFjY2VudCk7Y29sb3I6dmFyKC0tYWNjZW50KX0KI2x0YWItdG9vbHRpcHtwb3NpdGlvbjpmaXhlZDtiYWNrZ3JvdW5kOnJnYmEoOCwxMiwyMCwwLjk4KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4xMik7Ym9yZGVyLXJhZGl1czo4cHg7cGFkZGluZzoxMHB4IDEzcHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42O3dpZHRoOjIzMHB4O3doaXRlLXNwYWNlOm5vcm1hbDt0ZXh0LWFsaWduOmxlZnQ7Ym94LXNoYWRvdzowIDhweCAzMnB4IHJnYmEoMCwwLDAsMC42KTtwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7ei1pbmRleDo5OTk5OTtkaXNwbGF5Om5vbmV9CiNsdGFiLXRvb2x0aXAudmlzaWJsZXtvcGFjaXR5OjE7ZGlzcGxheTpibG9ja30KLmx0YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KCi5tYXAtc3ZnLXdyYXB7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZzoxMnB4IDE2cHggMTZweDsKfQoubWFwLWlubmVye3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxLzE7d2lkdGg6MTAwJX0KI2luZGlhLW1hcHt3aWR0aDoxMDAlO2hlaWdodDoxMDAlO2Rpc3BsYXk6YmxvY2s7b3ZlcmZsb3c6dmlzaWJsZX0KCi8qIG1hcCBzdGF0ZSBzdHlsZXMgKi8KI2luZGlhLW1hcCAuc3RhdGV7CiAgY3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpmaWx0ZXIgMC4yNXMgZWFzZSwgc3Ryb2tlLXdpZHRoIDAuMnMgZWFzZSwgc3Ryb2tlIDAuMnMgZWFzZTsKfQojaW5kaWEtbWFwIC5zdGF0ZTpob3ZlcnsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjcpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjFweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMjUpIGRyb3Atc2hhZG93KDAgMCAxMHB4IHJnYmEoMjU1LDI1NSwyNTUsMC4yKSk7Cn0KI2luZGlhLW1hcCAuc3RhdGUuc2VsZWN0ZWR7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC45KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxLjRweCAhaW1wb3J0YW50OwogIGZpbHRlcjpicmlnaHRuZXNzKDEuMzUpIGRyb3Atc2hhZG93KDAgMCAxNnB4IHJnYmEoMjU1LDI1NSwyNTUsMC4zKSk7Cn0KCi8qIGFuaW1hdGVkIHB1bHNlIHJpbmdzICovCi5wdWxzZS1yaW5ne2ZpbGw6bm9uZTtwb2ludGVyLWV2ZW50czpub25lfQoucHVsc2UtcmluZy5wMXthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCBpbmZpbml0ZX0KLnB1bHNlLXJpbmcucDJ7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgMC45cyBpbmZpbml0ZX0KQGtleWZyYW1lcyBwcnsKICAwJXtyOjQ7b3BhY2l0eTowLjc7c3Ryb2tlLXdpZHRoOjEuMn0KICAxMDAle3I6MjY7b3BhY2l0eTowO3N0cm9rZS13aWR0aDowLjJ9Cn0KCi8qIGF0bW9zcGhlcmljIGdsb3cgYmVoaW5kIGhvdCBzdGF0ZXMgKi8KLnN0YXRlLWdsb3d7cG9pbnRlci1ldmVudHM6bm9uZTtmaWxsOm5vbmV9CkBrZXlmcmFtZXMgZ2xvd1B1bHNlezAlLDEwMCV7b3BhY2l0eTowLjEyfTUwJXtvcGFjaXR5OjAuMjJ9fQoKLm1hcC10b29sdGlwewogIHBvc2l0aW9uOmFic29sdXRlO3BvaW50ZXItZXZlbnRzOm5vbmU7CiAgYmFja2dyb3VuZDpyZ2JhKDUsNywxMiwwLjk1KTtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6OXB4OwogIHBhZGRpbmc6MTJweCAxNHB4O29wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC4xMnM7ei1pbmRleDo5OTk5O21pbi13aWR0aDoxNzBweDsKfQoudHQtbntmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO21hcmdpbi1ib3R0b206OHB4O2NvbG9yOnZhcigtLWluayl9Ci50dC1ye2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweH0KLnR0LXIgc3Ryb25ne2NvbG9yOnZhcigtLWluayl9Ci50dC1uYXJ7CiAgbWFyZ2luLXRvcDo4cHg7cGFkZGluZy10b3A6OHB4O2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7Cn0KLnR0LW5hciBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtkaXNwbGF5OmJsb2NrO21hcmdpbi1ib3R0b206MnB4fQoKLyogU1RBVEUgUEFORUwgKi8KLnN0YXRlLXBhbmVsewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBwYWRkaW5nOjIwcHg7b3ZlcmZsb3cteTphdXRvO21heC1oZWlnaHQ6NzgwcHg7CiAgbWluLXdpZHRoOjA7b3ZlcmZsb3cteDpoaWRkZW47Cn0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhcnt3aWR0aDozcHh9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXItdGh1bWJ7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjJweH0KCi5wYW5lbC1lbXB0eXsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyOwogIGhlaWdodDoxMDAlO21pbi1oZWlnaHQ6MzIwcHg7dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzozMnB4IDIwcHg7Cn0KLnBhbmVsLWVtcHR5IHN2Z3tvcGFjaXR5OjAuMTU7bWFyZ2luLWJvdHRvbToxOHB4fQoucGFuZWwtZW1wdHkgLnBlLXR7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxOHB4O2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLWJvdHRvbTo4cHh9Ci5wYW5lbC1lbXB0eSAucGUtc3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNGVtO2xpbmUtaGVpZ2h0OjEuN30KCi8qIHN0YXRlIHBhbmVsIGludGVybmFscyAqLwouc3AtaGVhZHsKICBkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6ZmxleC1zdGFydDsKICBtYXJnaW4tYm90dG9tOjE2cHg7cGFkZGluZy1ib3R0b206MTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5zcC1la3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07Y29sb3I6dmFyKC0tZmFpbnQpO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTttYXJnaW4tYm90dG9tOjVweH0KLnNwLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyOHB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2xpbmUtaGVpZ2h0OjE7Y29sb3I6dmFyKC0taW5rKX0KLmZhdi1idG57CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2NvbG9yOnZhcigtLWZhaW50KTsKICB3aWR0aDozMHB4O2hlaWdodDozMHB4O2JvcmRlci1yYWRpdXM6NnB4O2N1cnNvcjpwb2ludGVyOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4cztwYWRkaW5nOjA7ZmxleC1zaHJpbms6MDsKfQouZmF2LWJ0bjpob3Zlcntjb2xvcjp2YXIoLS1kaW0pO2JvcmRlci1jb2xvcjp2YXIoLS1kaW0pfQouZmF2LWJ0bi5vbntjb2xvcjp2YXIoLS1hY2NlbnQpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjMpO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyl9Ci5mYXYtYnRuIHN2Z3t3aWR0aDoxM3B4O2hlaWdodDoxM3B4fQoKLyogbmFycmF0aXZlIHRpbWVsaW5lIOKAlCB0aGUgc2lnbmF0dXJlIGZlYXR1cmUgKi8KLm5hci10aW1lbGluZXsKICBtYXJnaW4tYm90dG9tOjE2cHg7Cn0KLm50LWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5udC1mbG93ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjA7CiAgcG9zaXRpb246cmVsYXRpdmU7cGFkZGluZy1sZWZ0OjE2cHg7Cn0KLm50LWZsb3c6OmJlZm9yZXsKICBjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO2xlZnQ6NXB4O3RvcDo2cHg7Ym90dG9tOjZweDt3aWR0aDoxcHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gYm90dG9tLHZhcigtLWFjY2VudCksdmFyKC0tYm9yZGVyKSk7b3BhY2l0eTowLjQ7Cn0KLm50LXN0ZXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjEwcHg7CiAgcGFkZGluZzo1cHggMDtwb3NpdGlvbjpyZWxhdGl2ZTsKfQoubnQtZG90ewogIHdpZHRoOjEwcHg7aGVpZ2h0OjEwcHg7Ym9yZGVyLXJhZGl1czo1MCU7ZmxleC1zaHJpbms6MDsKICBwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0Oi0xNnB4O3RvcDo3cHg7CiAgYm9yZGVyOjEuNXB4IHNvbGlkIGN1cnJlbnRDb2xvcjtiYWNrZ3JvdW5kOnZhcigtLWJnKTsKfQoubnQtc3RlcC5wYXN0IC5udC1kb3R7Y29sb3I6dmFyKC0tZmFpbnQpfQoubnQtc3RlcC5yZWNlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDIyNCw5MCw0MCwwLjQpfQoubnQtc3RlcC5jdXJyZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgMTBweCByZ2JhKDIyNCw5MCw0MCwwLjUpfQoubnQtY29udGVudHtmbGV4OjF9Ci5udC10b3BpY3tmb250LXNpemU6MTIuNXB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuM30KLm50LXN0ZXAucGFzdCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZmFpbnQpfQoubnQtc3RlcC5yZWNlbnQgLm50LXRvcGlje2NvbG9yOnZhcigtLWRpbSl9Ci5udC13aGVue2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoKLyogaW5zaWdodCBibG9jayAqLwouaW5zaWdodHsKICBtYXJnaW4tYm90dG9tOjE0cHg7CiAgcGFkZGluZzoxMnB4IDE0cHggMTJweCAxNnB4OwogIGJvcmRlci1sZWZ0OjEuNXB4IHNvbGlkIHZhcigtLWFjY2VudCk7CiAgYmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAzKTtib3JkZXItcmFkaXVzOjAgOHB4IDhweCAwOwogIGZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTMuNXB4O2ZvbnQtc3R5bGU6aXRhbGljOwogIGNvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS41NTtmb250LXdlaWdodDozMDA7Cn0KCi8qIGNvbXBhY3Qgc2NvcmUgc3RyaXAgKi8KLnNjb3JlLXN0cmlwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE2cHg7CiAgcGFkZGluZzo4cHggMTJweDtib3JkZXItcmFkaXVzOjdweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIG1hcmdpbi1ib3R0b206MTRweDsKfQouc3MtaXRlbXtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoycHh9Ci5zcy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTVlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjJweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtjb2xvcjp2YXIoLS1pbmspfQouc3MtZGVsdGF7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzoycHggN3B4O2JvcmRlci1yYWRpdXM6M3B4fQouc3MtZGVsdGEudXB7Y29sb3I6I2UwNjAzMDtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMSl9Ci5zcy1kZWx0YS5kbntjb2xvcjojM2JiOGQ4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMSl9Ci5zcy1kaXZpZGVye3dpZHRoOjFweDtoZWlnaHQ6MzJweDtiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MH0KLnNzLW5hcntmb250LXNpemU6MTEuNXB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC13ZWlnaHQ6NTAwfQoKLnNwLXNlY3Rpb257bWFyZ2luLWJvdHRvbToxNHB4fQouc3Atc2VjLXRpdGxlewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjlweDsKfQoKLyogbmFycmF0aXZlcyAqLwoubmFyLWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NnB4fQoubmFyLWl0ZW0ye2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIGF1dG87Z2FwOjZweDthbGlnbi1pdGVtczpjZW50ZXJ9Ci5uaS1sYWJlbHtmb250LXNpemU6MTEuNXB4O2NvbG9yOnZhcigtLWluayl9Ci5uaS12YWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpfQoubmktdHJhY2t7Z3JpZC1jb2x1bW46MS8tMTtoZWlnaHQ6MS41cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O292ZXJmbG93OmhpZGRlbjttYXJnaW4tdG9wOi0zcHh9Ci5uaS1maWxse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4O3RyYW5zaXRpb246d2lkdGggMC43c30KCi8qIG1vdmVtZW50ICovCi5tdi1ncmlke2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmcjtnYXA6N3B4fQoubXYtYmxvY2t7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjdweDtwYWRkaW5nOjlweH0KLm12LWh7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjdweH0KLm12LWJsb2NrLnVwIC5tdi1oe2NvbG9yOnZhcigtLXJpc2UpfQoubXYtYmxvY2suZG4gLm12LWh7Y29sb3I6dmFyKC0tZmFsbCl9Ci5tdi1pdHtmb250LXNpemU6MTAuNXB4O3BhZGRpbmc6NHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjb2xvcjp2YXIoLS1mYWludCl9Ci5tdi1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoubXYtaXQgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7Zm9udC13ZWlnaHQ6NTAwO2Rpc3BsYXk6YmxvY2s7Zm9udC1zaXplOjExcHh9Ci5tdi1pdCBzcGFue2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHh9CgovKiBlbW90aW9uICovCi5lbS1yb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTJweH0KLmVtLWRvbnV0e3dpZHRoOjc2cHg7aGVpZ2h0Ojc2cHg7ZmxleC1zaHJpbms6MH0KLmVtLWxlZ3tmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NHB4fQouZW0taXRlbXtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo2cHh9Ci5lbS1zd3t3aWR0aDo2cHg7aGVpZ2h0OjZweDtib3JkZXItcmFkaXVzOjJweDtmbGV4LXNocmluazowfQouZW0tbntmbGV4OjE7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2YXIoLS1kaW0pfQouZW0tcHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0taW5rKX0KCi8qIHRpbWVsaW5lIGNoYXJ0ICovCi50bC13cmFwe2hlaWdodDo3MnB4fQoKLyogYXJ0aWNsZXMgKi8KLmFydC1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjVweH0KLmFydC1pdGVtewogIGRpc3BsYXk6ZmxleDtnYXA6OHB4O3BhZGRpbmc6N3B4IDlweDtib3JkZXItcmFkaXVzOjZweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDEpOwogIHRyYW5zaXRpb246YWxsIDAuMTJzOwp9Ci5hcnQtaXRlbTpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyLWNvbG9yOnZhcigtLWJvcmRlcjIpfQouYXJ0LXNyY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7ZmxleC1zaHJpbms6MDt3aWR0aDo0NHB4O3BhZGRpbmctdG9wOjFweH0KLmFydC10eHR7Zm9udC1zaXplOjExcHg7bGluZS1oZWlnaHQ6MS40O2NvbG9yOnZhcigtLWRpbSl9CgovKiBOQVJSQVRJVkUgSU5URUxMSUdFTkNFIFJPVyAqLwoubmFyLXJvd3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwogIHBhZGRpbmc6MCAzNnB4IDI4cHg7CiAgZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDoxOHB4Owp9Ci5uYXItY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTRweCk7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNnB4IDIwcHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowOwp9Ci5uYy1ib2R5e3BhZGRpbmc6OHB4IDIwcHggMTZweDtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO30KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAzNnB4IDYwcHg7bWF4LXdpZHRoOjU4MHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZm9vdC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luLWJvdHRvbToxNHB4fQouZm9vdC1saW5le2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS44O21hcmdpbi1ib3R0b206MTJweH0KLmZvb3Qtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpfQoKLyogYW5pbWF0aW9ucyAqLwpAa2V5ZnJhbWVzIGZhZGVVcHtmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWSg2cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci5tYXAtY2FyZCwuc3RhdGUtcGFuZWwsLm5hci1jYXJkLC5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb246ZmFkZVVwIDAuNTVzIGN1YmljLWJlemllciguMiwuOCwuMiwxKSBiYWNrd2FyZHN9Ci5uYXItY2FyZDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMDdzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDMpe2FuaW1hdGlvbi1kZWxheTowLjE0c30KLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbi1kZWxheTowLjA1c30KCkBtZWRpYShtYXgtd2lkdGg6MTEwMHB4KXsKICAubWFpbntncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQogIC5zdGF0ZS1wYW5lbHttYXgtaGVpZ2h0Om5vbmV9CiAgLm5hci1yb3d7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KfQoKLyog4pSA4pSAIFdIQVQgSU5ESUEgSVMgUkVBQ1RJTkcgVE8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi53aXItc2VjdGlvbnsKICBmbGV4OjE7bWluLXdpZHRoOjA7CiAgcGFkZGluZzowOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjE0cHg7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKfQoud2lyLWhlYWRlcnsKICBwYWRkaW5nOjE4cHggMjJweCAxNHB4OwogIGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjsKfQoud2lyLXRpdGxlewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4zZW07CiAgdGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjg1Owp9Ci53aXItbGl2ZXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo2cHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4xZW07Cn0KLndpci1saXZlLWRvdHsKICB3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiMzOWZmMTQ7CiAgYm94LXNoYWRvdzowIDAgNnB4IHJnYmEoNTcsMjU1LDIwLDAuNik7CiAgYW5pbWF0aW9uOndpckxpdmVQdWxzZSAycyBlYXNlLWluLW91dCBpbmZpbml0ZTsKfQpAa2V5ZnJhbWVzIHdpckxpdmVQdWxzZXswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjN9fQoud2lyLXNpZ25hbHN7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtmbGV4OjE7b3ZlcmZsb3c6aGlkZGVufQoud2lyLXNpZ25hbHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MDsKICBwYWRkaW5nOjEzcHggMjJweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDM1KTsKICBvcGFjaXR5OjA7CiAgYW5pbWF0aW9uOndpckZhZGVJbiAwLjZzIGVhc2UgZm9yd2FyZHM7CiAgcG9zaXRpb246cmVsYXRpdmU7Y3Vyc29yOmRlZmF1bHQ7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci53aXItc2lnbmFsOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKX0KLndpci1zaWduYWw6bGFzdC1jaGlsZHtib3JkZXItYm90dG9tOm5vbmV9CkBrZXlmcmFtZXMgd2lyRmFkZUlue2Zyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVYKC02cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci53aXItc2lnbmFsLWJhcnsKICB3aWR0aDoycHg7Ym9yZGVyLXJhZGl1czoxcHg7ZmxleC1zaHJpbms6MDsKICBtYXJnaW4tcmlnaHQ6MTRweDttYXJnaW4tdG9wOjRweDsKICBhbGlnbi1zZWxmOnN0cmV0Y2g7bWluLWhlaWdodDoxNnB4OwogIG9wYWNpdHk6MC42Owp9Ci53aXItc2lnbmFsLWNvbnRlbnR7ZmxleDoxO21pbi13aWR0aDowfQoud2lyLXNpZ25hbC10ZXh0ewogIGZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTQuNXB4O2ZvbnQtd2VpZ2h0OjMwMDsKICBjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTtsZXR0ZXItc3BhY2luZzotMC4wMWVtOwp9Ci53aXItc2lnbmFsLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6aW5oZXJpdDtvcGFjaXR5OjAuOH0KLndpci1zaWduYWwtbWV0YXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7bWFyZ2luLXRvcDo0cHg7Cn0KLndpci1zaWduYWwtdGFnewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtvcGFjaXR5OjAuNDU7Cn0KLndpci1zaWduYWwtbG9jewogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7Cn0KLndpci1sb2FkaW5newogIGRpc3BsYXk6ZmxleDtnYXA6NnB4O3BhZGRpbmc6MjBweCAyMnB4O2FsaWduLWl0ZW1zOmNlbnRlcjsKfQoud2lyLWRvdHt3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOndpckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQoud2lyLWRvdDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMnN9Ci53aXItZG90Om50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC40c30KQGtleWZyYW1lcyB3aXJEb3R7MCUsODAlLDEwMCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjN9NDAle3RyYW5zZm9ybTpzY2FsZSgxKTtvcGFjaXR5OjF9fQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PgoKPGRpdiBpZD0ibHRhYi10b29sdGlwIj48L2Rpdj4KPGRpdiBjbGFzcz0idG9wYmFyIj4KICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC1tYXJrIj48c3BhbiBjbGFzcz0iYnJhbmQtcHVsc2UtZG90Ij48L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC10ZXh0LWJsb2NrIj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLW5hbWUiPjxlbSBjbGFzcz0iYnJhbmQtcHVsc2Utd29yZCI+UHVsc2U8L2VtPiBvZiBJbmRpYTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXIiPgogICAgPGRpdiBjbGFzcz0idG9wYmFyLXNvdXJjZSI+CiAgICAgIDxkaXYgY2xhc3M9InRzLW9ic2VydmVkIj5PYnNlcnZlZCBmcm9tPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InRzLXNvdXJjZXMiPnJlZ2lvbmFsIG1lZGlhIMK3IHB1YmxpYyBkaXNjdXNzaW9uIMK3IGluZGVwZW5kZW50IHJlcG9ydGluZyDCtyBzb2NpYWwgc2lnbmFsczwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJ0b3BiYXItc2lnbmFscyI+CiAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICA8c3BhbiBpZD0ibGl2ZS1jb3VudCI+4oCmPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0idHMtc2lnLWxhYmVsIj5zaWduYWxzPC9zcGFuPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5PYnNlcnZpbmcgSW5kaWEgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5EZXRlY3RpbmcgcmVnaW9uYWwgYXR0ZW50aW9uIHNoaWZ0cywgZW1vdGlvbmFsIG1vdmVtZW50LCBhbmQgZW1lcmdpbmcgbmFycmF0aXZlcyBiZWZvcmUgdGhleSBlbnRlciBtYWluc3RyZWFtIG5hdGlvbmFsIGRpc2NvdXJzZS48L3A+CiAgPHAgY2xhc3M9Imhlcm8tc3ViLWxpbmUiPkVhcmx5IHNpZ25hbCBkZXRlY3Rpb24gwrcgbm90IG5ld3Mgwrcgbm90IHByZWRpY3Rpb248L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSIKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7CiAgYmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpOwogIHBhZGRpbmc6MCAzNnB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOwoiPgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLXNpZ25hbHMiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlNpZ25hbHMgdHJhY2tlZDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIj5MaXZlIGluZ2VzdGlvbjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1ob3R0ZXN0IiBzdHlsZT0iY3Vyc29yOnBvaW50ZXIiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWhvdHRlc3QtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj5DbGljayB0byBleHBsb3JlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWFuZ2VyLXN1YiI+T3V0cmFnZSAmIHByb3Rlc3Qgc2lnbmFsczwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlRvcCByaXNpbmcgbmFycmF0aXZlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1uYXJyYXRpdmUtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hcnJhdGl2ZS1zdWIiPk5hdGlvbmFsIHNpZ25hbCBzdXJnZTwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbGluZy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbGluZy1zdWIiPlNpZ25hbCBkZWNheTwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KLnN0YXQtY2VsbHsKICBmbGV4OjE7cGFkZGluZzoxMHB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5zdGF0LWNlbGw6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpfQouc3RhdC1kaXZ7d2lkdGg6MXB4O2JhY2tncm91bmQ6cmdiYSgxNjAsMTkwLDIzMCwwLjA3KTtmbGV4LXNocmluazowO21hcmdpbjo4cHggMH0KLnNjLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW4tdG9wOjFweH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KPC9zdHlsZT4KCgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJ3aXItc2VjdGlvbiI+CiAgICAgIDxkaXYgY2xhc3M9Indpci1oZWFkZXIiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci10aXRsZSI+V2hhdCBJbmRpYSBpcyByZWFjdGluZyB0bzwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Indpci1saXZlIj48c3BhbiBjbGFzcz0id2lyLWxpdmUtZG90Ij48L3NwYW4+bGl2ZSBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ3aXItc2lnbmFscyIgaWQ9Indpci1zaWduYWxzIj4KICAgICAgICA8ZGl2IGNsYXNzPSJ3aXItbG9hZGluZyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+TmFycmF0aXZlIHNoaWZ0czwvc3Bhbj4KICAgICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjJweDsiPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIGFjdGl2ZSIgZGF0YS1wZXJpb2Q9IjNtIj4zTTwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIiBkYXRhLXBlcmlvZD0iNm0iPjZNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSIxeSI+MVk8L2J1dHRvbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDwhLS0gc2hpZnRzIGxpc3QgLS0+CiAgICAgIDxkaXYgc3R5bGU9ImZsZXg6MTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3BhZGRpbmc6MTBweCAxNHB4O2dhcDo2cHg7IiBpZD0ic2hpZnQtbGlzdCI+PC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKCjwhLS0gTUFJTjogTUFQICsgU1RBVEUgUEFORUwgLS0+CjxkaXYgY2xhc3M9Im1haW4iPgoKICA8ZGl2IGNsYXNzPSJtYXAtY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJtYXAtdG9wIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLXRpdGxlLWJsb2NrIj4KICAgICAgICA8ZGl2IGNsYXNzPSJtdCI+SW5kaWEgJm1kYXNoOyBjb2xsZWN0aXZlIGF0dGVudGlvbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1zIiBpZD0ibWFwLW1ldGEiPjMwIHN0YXRlcyAmbWlkZG90OyBsaXZlIHNpZ25hbCBjb21wb3NpdGU8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImxlZ2VuZCI+PHNwYW4+cXVpZXQ8L3NwYW4+PGRpdiBjbGFzcz0ibGVnZW5kLWJhciI+PC9kaXY+PHNwYW4+YWN0aXZlPC9zcGFuPjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJsYXllci1yb3ciPgogICAgICA8c3BhbiBjbGFzcz0ibGF5ZXItbGFiZWwiPlZpZXc8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9Imx0YWJzIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiBhY3RpdmUiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJlbW90aW9uIj5FbW90aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJUaGUgZG9taW5hbnQgZW1vdGlvbmFsIHRvbmUg4oCUIGFueGlvdXMsIGFuZ3J5LCBob3BlZnVsLCBwcm91ZCBvciBmZWFyZnVsLiBSZXZlYWxzIHRoZSBwc3ljaG9sb2dpY2FsIHVuZGVyY3VycmVudCBvZiBwb2xpdGljYWwgYXR0ZW50aW9uLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9InZlbG9jaXR5Ij5Nb21lbnR1bSA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iSXMgYXR0ZW50aW9uIHJpc2luZyBvciBmYWxsaW5nPyBSaXNpbmcgPSBuYXJyYXRpdmUgYWNjZWxlcmF0aW5nLiBDb29saW5nID0gbG9zaW5nIHRyYWN0aW9uLiBTaG93cyBzdGF0ZXMgZW50ZXJpbmcgb3IgZXhpdGluZyBhIHBvbGl0aWNhbCBjeWNsZS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLXN2Zy13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLWlubmVyIj4KICAgICAgICA8c3ZnIGlkPSJpbmRpYS1tYXAiIHZpZXdCb3g9IjAgMCA4MDAgODAwIiBwcmVzZXJ2ZUFzcGVjdFJhdGlvPSJ4TWlkWU1pZCBtZWV0Ij4KICAgICAgICAgIDxkZWZzPgogICAgICAgICAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImFtYkdsb3ciIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjUwJSI+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0icmdiYSgyMjQsOTAsNDAsMC4wNCkiLz4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgICAgICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICAgICAgICAgIDxmaWx0ZXIgaWQ9InN0YXRlR2xvdyIgeD0iLTMwJSIgeT0iLTMwJSIgd2lkdGg9IjE2MCUiIGhlaWdodD0iMTYwJSI+CiAgICAgICAgICAgICAgPGZlR2F1c3NpYW5CbHVyIGluPSJTb3VyY2VHcmFwaGljIiBzdGREZXZpYXRpb249IjgiIHJlc3VsdD0iYmx1ciIvPgogICAgICAgICAgICAgIDxmZUNvbXBvc2l0ZSBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJibHVyIiBvcGVyYXRvcj0ib3ZlciIvPgogICAgICAgICAgICA8L2ZpbHRlcj4KICAgICAgICAgIDwvZGVmcz4KICAgICAgICAgIDxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJ1cmwoI2FtYkdsb3cpIi8+CiAgICAgICAgICA8ZyBpZD0ibWFwLWdsb3ciPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtc3RhdGVzIj48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXB1bHNlcyI+PC9nPgogICAgICAgIDwvc3ZnPgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC10b29sdGlwIiBpZD0idG9vbHRpcCI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gU1RBVEUgUEFORUwgLS0+CiAgPGRpdiBjbGFzcz0ic3RhdGUtcGFuZWwiIGlkPSJzdGF0ZS1kZXRhaWwiPgogICAgPGRpdiBjbGFzcz0icGFuZWwtZW1wdHkiPgogICAgICA8c3ZnIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxIj4KICAgICAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiA4djRNMTIgMTZoLjAxIi8+CiAgICAgIDwvc3ZnPgogICAgICA8ZGl2IGNsYXNzPSJwZS10Ij5TZWxlY3QgYSBzdGF0ZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJwZS1zIj5DbGljayBhbnkgcmVnaW9uIG9uIHRoZSBtYXA8YnIvPnRvIG9wZW4gaXRzIG5hcnJhdGl2ZSBwYW5lbC48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKPC9kaXY+Cgo8IS0tIE5BUlJBVElWRSBST1cgLS0+CjxkaXYgY2xhc3M9Im5hci1yb3ciPgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7Ij4KICAgICAgPHNwYW4gY2xhc3M9Im5jLWRvdCByaXNlMiI+PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJpc2luZyBuYXJyYXRpdmVzPC9zcGFuPgogICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWxlZnQ6YXV0byI+Z2FpbmluZyB0cmFjdGlvbjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9InJpc2luZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7Ij4KICAgICAgPHNwYW4gY2xhc3M9Im5jLWRvdCBmYWxsIj48L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy10aXRsZSI+RGVjbGluaW5nIG5hcnJhdGl2ZXM8L3NwYW4+CiAgICAgIDxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tbGVmdDphdXRvIj5sb3NpbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJkZWNsaW5pbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEZBVlMgLS0+CjxzZWN0aW9uIGNsYXNzPSJmYXZzIj4KICA8ZGl2IGNsYXNzPSJmYXZzLWxhYmVsIj5UcmFja2VkIHN0YXRlczwvZGl2PgogIDxkaXYgY2xhc3M9ImZhdnMtcm93IiBpZD0iZmF2LXJvdyI+CiAgICA8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKPGRpdiBjbGFzcz0iZm9vdCI+CiAgPGRpdiBjbGFzcz0iZm9vdC1uYW1lIj5QdWxzZSBvZiBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3QtbGluZSI+RGV0ZWN0cyByZWdpb25hbCBhdHRlbnRpb24gc2hpZnRzLCBlbW90aW9uYWwgbW92ZW1lbnQsIGFuZCBlbWVyZ2luZyBuYXJyYXRpdmVzIOKAlCBvYnNlcnZpbmcgY29sbGVjdGl2ZSBzaWduYWxzIGJlZm9yZSB0aGV5IGFjaGlldmUgZnVsbCBuYXRpb25hbCBhbXBsaWZpY2F0aW9uLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIE9ic2VydmF0aW9uLjwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICBpZighcm93c3x8IXJvd3MubGVuZ3RoKSByZXR1cm47CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocm93LmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb21FbW89cm93LmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihlbW9zKXx8bnVsbDsKICAgICAgdmFyIGVudHJ5PXthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbUVtbyxkb21pbmFudF9uYXJyYXRpdmU6cm93LmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgICAgTElWRVtyb3cubmFtZV09ZW50cnk7CiAgICAgIGlmKCFTRFtyb3cubmFtZV0pIFNEW3Jvdy5uYW1lXT1PYmplY3QuYXNzaWduKHt9LERFRkFVTFQpOwogICAgICBPYmplY3QuYXNzaWduKFNEW3Jvdy5uYW1lXSxlbnRyeSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGJ1aWxkV0lSU2lnbmFscygpOwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgc2V0VGltZW91dChyZW5kZXJNb21lbnR1bSwgNTAwKTsKICAgIGlmKFNFTCYmTElWRVtTRUxdJiZkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJykpIHJlbmRlclBhbmVsKFNFTCk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW0FQSV0nLGUubWVzc2FnZSk7fQp9CgpmdW5jdGlvbiBidWlsZExvY2FsSW5zaWdodCgpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwoKICAvLyBBZ2dyZWdhdGUgdG9wIG5hcnJhdGl2ZXMgYWNyb3NzIGFsbCBzdGF0ZXMKICB2YXIgbmFyPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5hcltuLm5hbWVdKSBuYXJbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuYXJbbi5uYW1lXVtuLmRpcl09KG5hcltuLm5hbWVdW24uZGlyXXx8MCkrbi52YWw7CiAgICAgIG5hcltuLm5hbWVdLnRvdGFsPShuYXJbbi5uYW1lXS50b3RhbHx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKCiAgLy8gVG9wIHJpc2luZyBhbmQgZmFsbGluZyAoZXhjbHVkZSB0aWVzIHdoZXJlIHNhbWUgbmFtZSByaXNlcyBhbmQgZmFsbHMpCiAgdmFyIHJpc2luZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPmt2WzFdLmRvd247fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udXAtYVsxXS51cDt9KS5zbGljZSgwLDMpOwogIHZhciBmYWxsaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS5kb3duLWFbMV0uZG93bjt9KS5zbGljZSgwLDIpOwogIHZhciB0b3AzPU9iamVjdC5lbnRyaWVzKG5hcikuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnRvdGFsLWFbMV0udG90YWw7fSkuc2xpY2UoMCwzKTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PWVudHJpZXMuc2xpY2UoKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgaG90dGVzdEVtbz1ob3R0ZXN0PyhMSVZFW2hvdHRlc3RbMF1dJiZMSVZFW2hvdHRlc3RbMF1dLmRvbWluYW50X2Vtb3Rpb24pfHwnJzonJyA7CgogIC8vIEJ1aWxkIGluc2lnaHQgdGV4dCDigJQgbW9yZSBhbmFseXRpY2FsLCBjb250ZXh0LWF3YXJlCiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICBpZighZWwpIHJldHVybjsKCiAgdmFyIGxpbmVzPVtdOwogIGlmKHJpc2luZy5sZW5ndGgmJmZhbGxpbmcubGVuZ3RoJiZyaXNpbmdbMF1bMF0hPT1mYWxsaW5nWzBdWzBdKXsKICAgIGxpbmVzLnB1c2goJzxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIHRoZSBkb21pbmFudCBzaWduYWwgYWNyb3NzIEluZGlhIHRvZGF5Jyk7CiAgICBpZihmYWxsaW5nWzBdKSBsaW5lcy5wdXNoKCcgYXMgPGVtPicrZmFsbGluZ1swXVswXSsnPC9lbT4gZmFkZXMgZnJvbSBuYXRpb25hbCBmb2N1cycpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBpcyB0aGUgbW9zdCBhY3RpdmUgc3RhdGUnKwogICAgICAoaG90dGVzdEVtbz8nIHdpdGggJytob3R0ZXN0RW1vKycgYXMgdGhlIHByaW1hcnkgc2lnbmFsIHRvbmUnOicnKSk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJy4gU2Vjb25kYXJ5IHN1cmdlOiA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHJpc2luZy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnU2lnbmFscyBhcmUgY29uY2VudHJhdGVkIGFyb3VuZCA8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBsZWFkcyBuYXRpb25hbCBhdHRlbnRpb24nKTsKICAgIGlmKHJpc2luZ1sxXSkgbGluZXMucHVzaCgnIGFsb25nc2lkZSA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHRvcDMubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ05hdGlvbmFsIHNpZ25hbHMgYXJlIGRpc3BlcnNlZC4gVG9wIG5hcnJhdGl2ZXM6ICcrdG9wMy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZW0+JytuWzBdKyc8L2VtPic7fSkuam9pbignLCAnKSk7CiAgfQoKICBpZihsaW5lcy5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2xpbmVzLmpvaW4oJycpKycuPC9kaXY+JzsKICB9CgogIC8vIFRhZ3MKICBpZih0RWwpewogICAgdmFyIHRhZ3M9W107CiAgICBmYWxsaW5nLnNsaWNlKDAsMSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoNTksMTg0LDIxNiwwLjMpO2NvbG9yOiMzYmI4ZDgiPuKGkyAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICByaXNpbmcuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7Y29sb3I6I2UwNWEyOCI+4oaRICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIGlmKHRhZ3MubGVuZ3RoKSB0RWwuaW5uZXJIVE1MPXRhZ3Muam9pbignJyk7CiAgfQoKICBpZihtZXRhRWwpewogICAgdmFyIHN0YXRlQ291bnQ9T2JqZWN0LnZhbHVlcyhMSVZFKS5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuYXR0ZW50aW9uPjI7fSkubGVuZ3RoOwogICAgbWV0YUVsLnRleHRDb250ZW50PSdPYnNlcnZpbmcgJytzdGF0ZUNvdW50KycgYWN0aXZlIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pOwogIH0KfQoKZnVuY3Rpb24gdXBkYXRlQWxsU3RyaXBzKCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CiAgdmFyIGhvdHRlc3Q9ZW50cmllcy5yZWR1Y2UoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKT4oYVsxXS5hdHRlbnRpb258fDApP2I6YTt9LGVudHJpZXNbMF0pOwogIHNldFRleHQoJ3NjLWhvdHRlc3QtdmFsJyxob3R0ZXN0WzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXN1YicsJ0F0dGVudGlvbiAnK2hvdHRlc3RbMV0uYXR0ZW50aW9uLnRvRml4ZWQoMSkpOwogIHZhciB0b3BBbmdlck5tPW51bGwsdG9wQW5nZXJQY3Q9MDsKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgdmFyIGU9a3ZbMV0uZW1vdGlvbnN8fHt9OwogICAgdmFyIGE9ZS5hbmdlcnx8MDsKICAgIGlmKGE+MCYmYTw9MSkgYT1NYXRoLnJvdW5kKGEqMTAwKTsKICAgIGlmKGE+dG9wQW5nZXJQY3Qpe3RvcEFuZ2VyUGN0PWE7dG9wQW5nZXJObT1rdlswXTt9CiAgfSk7CiAgaWYodG9wQW5nZXJObSYmdG9wQW5nZXJQY3Q+MCl7CiAgICBzZXRUZXh0KCdzYy1hbmdlci12YWwnLHRvcEFuZ2VyTm0pOwogICAgc2V0VGV4dCgnc2MtYW5nZXItc3ViJywnQW5nZXIgJytNYXRoLnJvdW5kKHRvcEFuZ2VyUGN0KSsnJSBvZiBzaWduYWxzJyk7CiAgfSBlbHNlIHsKICAgIC8vIEZhbGwgYmFjayB0byBkb21pbmFudF9lbW90aW9uPWFuZ2VyCiAgICB2YXIgYW5nZXJEb209ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS5kb21pbmFudF9lbW90aW9uPT09J2FuZ2VyJzt9KTsKICAgIGlmKGFuZ2VyRG9tLmxlbmd0aCl7CiAgICAgIHZhciB0b3BCeUF0dD1hbmdlckRvbS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICAgICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BCeUF0dFswXSk7CiAgICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsJ0RvbWluYW50IGVtb3Rpb246IGFuZ2VyJyk7CiAgICB9CiAgfQogIHZhciBjb29saW5nPWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLnZlbG9jaXR5fHwwKTwoYVsxXS52ZWxvY2l0eXx8MCk/YjphO30sZW50cmllc1swXSk7CiAgc2V0VGV4dCgnc2MtY29vbGluZy12YWwnLGNvb2xpbmdbMF0pO3NldFRleHQoJ3NjLWNvb2xpbmctc3ViJywnVmVsb2NpdHkgJytjb29saW5nWzFdLnZlbG9jaXR5LnRvRml4ZWQoMykpOwogIHZhciBuYz17fTtlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSluY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdPShuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdfHwwKSsxO30pOwogIHZhciB0bj1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgaWYodG4pe3NldFRleHQoJ3NjLW5hcnJhdGl2ZS12YWwnLHRuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3RuWzBdLnNsaWNlKDEpKTtzZXRUZXh0KCdzYy1uYXJyYXRpdmUtc3ViJywnRG9taW5hbnQgYWNyb3NzICcrdG5bMV0rJyBzdGF0ZXMnKTt9Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUvJytlbmNvZGVVUklDb21wb25lbnQobmFtZSkpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMoZC5lbW90aW9uc3x8e30pOwogICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgIFNEW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZSwKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSxzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OmQubmFycmF0aXZlSGlzdG9yeXx8REVGQVVMVC5uYXJyYXRpdmVIaXN0b3J5LHNpZ25hbF9jb3VudDpkLnNpZ25hbF9jb3VudHx8MH07CiAgICBpZighTElWRVtuYW1lXSlMSVZFW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmV9OwogICAgTElWRVtuYW1lXS5lbW90aW9ucz1lbW9zO0xJVkVbbmFtZV0uZG9taW5hbnRfZW1vdGlvbj1kb207CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbmFtZSxlLm1lc3NhZ2UpO3JldHVybiBTRFtuYW1lXXx8REVGQVVMVDt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICAvLyB0b3BiYXIKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGl2ZS1jb3VudCcpOwogICAgaWYoZWwmJmQudG90YWxfc2lnbmFscykgZWwudGV4dENvbnRlbnQ9ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk7CiAgICB2YXIgbWV0YT1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLW1ldGEnKTsKICAgIGlmKG1ldGEmJmQuYXNfb2YpIG1ldGEudGV4dENvbnRlbnQ9JzMwIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoZC5hc19vZikudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicpOwogICAgLy8gc3RhdHMgc3RyaXAKICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJywgZC50b3RhbF9zaWduYWxzP2QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOictJyk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICB9Y2F0Y2goZSl7fQp9CgpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZihlbCllbC50ZXh0Q29udGVudD12YWw7fQoKZnVuY3Rpb24gdXBkYXRlU3RyaXBOYXJyYXRpdmUoKXt1cGRhdGVBbGxTdHJpcHMoKTt9CmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXt9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hJbnNpZ2h0cygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2luc2lnaHRzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICB2YXIgc2lnPWQuc2lnbmF0dXJlOwogICAgaWYoc2lnKXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLnRhZ3MpdEVsLmlubmVySFRNTD1kLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICB9CiAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgaWYockVsJiZkLnJpc2luZyYmZC5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6I2UwNWEyOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICBpZihmRWwmJmQuZmFsbGluZyYmZC5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IGNsYXNzPSJuaS1zdGF0ZXMiPicrbi5zdGF0ZXMuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgY2xhc3M9Im5pLXRyYWNrIj48ZGl2IGNsYXNzPSJuaS1maWxsIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMykrJyU7YmFja2dyb3VuZDojM2JiOGQ4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogICAgaWYoZ0VsJiZkLnJlZ2lvbmFsJiZkLnJlZ2lvbmFsLmxlbmd0aClnRWwuaW5uZXJIVE1MPWQucmVnaW9uYWwubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbiI+PHNwYW4gY2xhc3M9Im5pLW5hbWUiPicrci5yZWdpb24rJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3IuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+PGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK3IuaG90dGVzdF9zdGF0ZSsnIMK3ICcrci50b3BfbmFycmF0aXZlKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUubWVzc2FnZSk7fQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaEZ1bGxTbmFwc2hvdCgpewogIC8vIExvYWQgQUxMIHN0YXRlIGRhdGEgaW4gb25lIHJlcXVlc3QgZm9yIGluc3RhbnQgZmlyc3QtbG9hZAogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2Z1bGwtc25hcHNob3QnKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIGlmKGQud2FybWluZ191cHx8IWQuc3RhdGVzfHwhZC5zdGF0ZXMubGVuZ3RoKSByZXR1cm4gZmFsc2U7CgogICAgLy8gUG9wdWxhdGUgU0QgYW5kIExJVkUgZnJvbSBmdWxsIHNuYXBzaG90CiAgICBkLnN0YXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgICBpZighcy5uYW1lKSByZXR1cm47CiAgICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKHMuZW1vdGlvbnN8fHt9KTsKICAgICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fHMuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgICAgdmFyIGVudHJ5PU9iamVjdC5hc3NpZ24oe30scyx7ZW1vdGlvbnM6ZW1vcyxkb21pbmFudF9lbW90aW9uOmRvbSxkZWx0YTpzLmRlbHRhXzI0aHx8MH0pOwogICAgICBTRFtzLm5hbWVdPWVudHJ5OwogICAgICBMSVZFW3MubmFtZV09e2F0dGVudGlvbjpzLmF0dGVudGlvbixkZWx0YTpzLmRlbHRhXzI0aHx8MCx2ZWxvY2l0eTpzLnZlbG9jaXR5LGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpzLmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgIH0pOwoKICAgIC8vIFVwZGF0ZSBzaWduYWxzIGNvdW50CiAgICBpZihkLnNuYXBzaG90JiZkLnNuYXBzaG90LnRvdGFsX3NpZ25hbHMpewogICAgICBzZXRUZXh0KCdzYy1zaWduYWxzLXZhbCcsZC5zbmFwc2hvdC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCkpOwogICAgfQoKICAgIC8vIFVwZGF0ZSBpbnNpZ2h0cyBmcm9tIGNhY2hlZCBkYXRhCiAgICBpZihkLmluc2lnaHRzJiZkLmluc2lnaHRzLnNpZ25hdHVyZSl7CiAgICAgIHZhciBzaWc9ZC5pbnNpZ2h0cy5zaWduYXR1cmU7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgaWYoZWwpZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij48ZW0+JytzaWcuZmFkaW5nLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3NpZy5mYWRpbmcuc2xpY2UoMSkrJzwvZW0+IGZhZGluZyBhcyA8ZW0+JytzaWcucmlzaW5nX3ByaW1hcnkrIjwvZW0+Iisoc2lnLnJpc2luZ19zZWNvbmRhcnk/IiBhbG9uZ3NpZGUgPGVtPiIrc2lnLnJpc2luZ19zZWNvbmRhcnkrIjwvZW0+IjoiIikrIiBhY3Jvc3MgdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gPHN0cm9uZyBzdHlsZT1cImNvbG9yOnZhcigtLWluaylcIj4iK3NpZy5ob3R0ZXN0X3N0YXRlKyI8L3N0cm9uZz4gZG9taW5hdGVzLjwvZGl2PiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC5pbnNpZ2h0cy50YWdzKXRFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgICBpZihyRWwmJmQuaW5zaWdodHMucmlzaW5nJiZkLmluc2lnaHRzLnJpc2luZy5sZW5ndGgpckVsLmlubmVySFRNTD1kLmluc2lnaHRzLnJpc2luZy5tYXAoZnVuY3Rpb24obil7dmFyIHc9TWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpO3JldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvc3Bhbj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPjwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo0cHg7Ij4nK24uc3RhdGVzLnNsaWNlKDAsMykuam9pbignLCAnKSsnPC9kaXY+JzonJykrJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgICAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICAgICAgaWYoZkVsJiZkLmluc2lnaHRzLmZhbGxpbmcmJmQuaW5zaWdodHMuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmluc2lnaHRzLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTtyZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6IzNiYjhkOCI+4oaTIGZhZGluZzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NHB4OyI+JytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiMzYmI4ZDg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB9CgogICAgLy8gUmVuZGVyIG1hcCBjb2xvcnMgYW5kIHN0cmlwcwogICAgYXBwbHlMYXllcigpOwogICAgcmVuZGVyTW9tZW50dW0oKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgLy8gTG9hZCBpbnNpZ2h0cyB0b28KICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIC8vIFVzZSBjYWNoZWQgbmFycmF0aXZlIGluc2lnaHQgaWYgYXZhaWxhYmxlCiAgICBpZihkLm5hcnJhdGl2ZV9pbnNpZ2h0JiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0Kyc8L2Rpdj4nOwogICAgICBpZih0RWwmJmQubmFycmF0aXZlX2luc2lnaHQudG9wX25hcnJhdGl2ZXMpewogICAgICB9CiAgICB9CiAgICByZXR1cm4gdHJ1ZTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1tmdWxsLXNuYXBzaG90XScsZS5tZXNzYWdlKTsKICAgIHJldHVybiBmYWxzZTsKICB9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoTmFycmF0aXZlSW5zaWdodCgpewogIHRyeXsKICAgIC8vIFRyeSBjYWNoZWQgdmVyc2lvbiBmcm9tIGZ1bGwtc25hcHNob3QgZmlyc3QgKGFscmVhZHkgbG9hZGVkKQogICAgLy8gVGhlbiBjYWxsIGRlZGljYXRlZCBlbmRwb2ludCBmb3IgZnJlc2ggQUkgYW5hbHlzaXMKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL25hcnJhdGl2ZS1pbnNpZ2h0Jyk7CiAgICBpZighci5vaykgcmV0dXJuOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZighZC50ZXh0KSByZXR1cm47CgogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CgogICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytkLnRleHQrJzwvZGl2Pic7CgogICAgLy8gVGFncyBmcm9tIHRvcCBuYXJyYXRpdmVzCiAgICBpZih0RWwmJmQudG9wX25hcnJhdGl2ZXMmJmQudG9wX25hcnJhdGl2ZXMubGVuZ3RoKXsKICAgICAgdEVsLmlubmVySFRNTD1kLnRvcF9uYXJyYXRpdmVzLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgIHZhciBjb2w9aT09PTA/JyNlMDVhMjgnOidyZ2JhKDE2MCwxOTAsMjMwLDAuNiknOwogICAgICAgIHZhciBhcnJvdz1pPT09MD8n4oaRICc6J8K3ICc7CiAgICAgICAgcmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMik7Y29sb3I6Jytjb2wrJyI+JythcnJvdytuLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24uc2xpY2UoMSkrJzwvc3Bhbj4nOwogICAgICB9KS5qb2luKCcnKTsKICAgIH0KCiAgICBpZihtZXRhRWwpewogICAgICB2YXIgdD1uZXcgRGF0ZShkLmFzX29mKTsKICAgICAgbWV0YUVsLnRleHRDb250ZW50PSdTaWduYWwgYW5hbHlzaXMgwrcgJyt0LnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nLHtob3VyOicyLWRpZ2l0JyxtaW51dGU6JzItZGlnaXQnfSkrKGQuZmFsbGJhY2s/JyDCtyBwYXR0ZXJuLWJhc2VkJzonIMK3IEFJIHN5bnRoZXNpemVkJyk7CiAgICB9CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW25hcnJhdGl2ZV0nLGUubWVzc2FnZSk7fQp9Cgphc3luYyBmdW5jdGlvbiBzdGFydFBvbGxpbmcoKXsKICBhd2FpdCBQcm9taXNlLmFsbChbZmV0Y2hBbGxTdGF0ZXMoKSxmZXRjaFNuYXAoKV0pOwogIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbihlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUpO30pOwogIHZhciBuPTA7CiAgdmFyIHQ9c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXsKICAgIG4rKzthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpOwogICAgaWYoU0VMKSByZW5kZXJQYW5lbChTRUwpOwogICAgaWYobj49MTIpe2NsZWFySW50ZXJ2YWwodCk7c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpO2lmKFNFTClyZW5kZXJQYW5lbChTRUwpO30sMTIwMDAwKTsKICAgICAgc2V0SW50ZXJ2YWwoZmV0Y2hJbnNpZ2h0cywzNjAwMDAwKTt9CiAgfSwxNTAwMCk7Cn0KCi8vIE5BUlJBVElWRSBEQVRBCnZhciBTSElGVFM9ewogICczbSc6WwogICAge2ZhZGluZzonSW5mbGF0aW9uJyxmYWRpbmdOb3RlOidlYXNpbmcgbmF0aW9uYWxseScscmlzaW5nOidCb3JkZXIgc2VjdXJpdHknLHJpc2luZ05vdGU6J3Bvc3QtaW5jaWRlbnQgc3VyZ2UnfSwKICAgIHtmYWRpbmc6J0VsZWN0aW9uIHJoZXRvcmljJyxmYWRpbmdOb3RlOidwb3N0LWN5Y2xlIGZhZGUnLHJpc2luZzonR292ZXJuYW5jZSBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZTonc3RlYWR5IHJpc2UnfSwKICAgIHtmYWRpbmc6J0Zhcm1lciBwcm90ZXN0cycsZmFkaW5nTm90ZTonbW9tZW50dW0gbG9zdCcscmlzaW5nOidVbmVtcGxveW1lbnQgYW54aWV0eScscmlzaW5nTm90ZToneW91dGggc2lnbmFsIHN1cmdlJ30sCiAgXSwKICAnNm0nOlsKICAgIHtmYWRpbmc6J0Nhc3RlIG1vYmlsaXNhdGlvbicsZmFkaW5nTm90ZToncHJlLWVsZWN0aW9uIHBlYWsnLHJpc2luZzonQ29ycnVwdGlvbiBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZToncG9zdC1jeWNsZSBwdXNoJ30sCiAgICB7ZmFkaW5nOidSZWxpZ2lvdXMgbmF0aW9uYWxpc20nLGZhZGluZ05vdGU6J3BsYXRlYXUgcGhhc2UnLHJpc2luZzonRWNvbm9taWMgYW54aWV0eScscmlzaW5nTm90ZTonY29zdC1vZi1saXZpbmcnfSwKICAgIHtmYWRpbmc6J0luZnJhc3RydWN0dXJlIHByaWRlJyxmYWRpbmdOb3RlOidyaWJib24tY3V0dGluZyBkb25lJyxyaXNpbmc6J0xhdyAmIG9yZGVyJyxyaXNpbmdOb3RlOidjcmltZSBuYXJyYXRpdmUgcmlzZSd9LAogIF0sCiAgJzF5JzpbCiAgICB7ZmFkaW5nOidQYW5kZW1pYyByZWNvdmVyeScsZmFkaW5nTm90ZTonZmFkZWQgZWFybHkgeWVhcicscmlzaW5nOidJbmZsYXRpb24nLHJpc2luZ05vdGU6J2RvbWluYXRlZCBtaWQteWVhcid9LAogICAge2ZhZGluZzonUmVnaW9uYWwgaWRlbnRpdHknLGZhZGluZ05vdGU6J2xhbmd1YWdlLWxlZCBwZWFrJyxyaXNpbmc6J1NlY3VyaXR5ICYgYm9yZGVycycscmlzaW5nTm90ZTonZ2VvcG9saXRpY2FsIGVzY2FsYXRpb24nfSwKICAgIHtmYWRpbmc6J0dvdmVybmFuY2Ugb3B0aW1pc20nLGZhZGluZ05vdGU6J3BvbGljeSBob25leW1vb24gZW5kJyxyaXNpbmc6J0NvcnJ1cHRpb24gJiBzY2FtcycscmlzaW5nTm90ZTonYWNjb3VudGFiaWxpdHkgY3ljbGUnfSwKICBdLAp9Owp2YXIgUkVHX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidCaWhhcicsZnJvbTonRWxlY3Rpb24gcmhldG9yaWMnLHRvOidVbmVtcGxveW1lbnQgJiBleGFtIHNjYW1zJyx0aW1lOic2IHdrcyd9LAogIHtzdGF0ZTonV2VzdCBCZW5nYWwnLGZyb206J0J5cG9sbCBwb2xpdGljcycsdG86J0xhdyAmIG9yZGVyIMK3IEJvcmRlcicsdGltZTonNCB3a3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdrcyd9LAogIHtzdGF0ZTonS2FybmF0YWthJyxmcm9tOidNaW5pbmcgY29udHJvdmVyc3knLHRvOidMYW5ndWFnZSBzaWduYWdlIHBvbGl0aWNzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonRGVsaGknLGZyb206J01ldHJvIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3a3MnfSwKICB7c3RhdGU6J1B1bmphYicsZnJvbTonUG93ZXIgY3Jpc2lzJyx0bzonQm9yZGVyIHNlY3VyaXR5IMK3IERyb25lcycsdGltZTonMyB3a3MnfSwKXTsKdmFyIE1PQ0tfUj1bCiAge25hbWU6J0JvcmRlciBzZWN1cml0eScsc3RhdGVzOidKJksgwrcgUHVuamFiIMK3IFJhamFzdGhhbicscGN0OicrNDElJ30sCiAge25hbWU6J1VuZW1wbG95bWVudCcsc3RhdGVzOidCaWhhciDCtyBVUCDCtyBKaGFya2hhbmQnLHBjdDonKzI4JSd9LAogIHtuYW1lOidMYW5ndWFnZSBwb2xpdGljcycsc3RhdGVzOidUTiDCtyBLYXJuYXRha2EgwrcgTUgnLHBjdDonKzIyJSd9LAogIHtuYW1lOidFbnZpcm9ubWVudGFsIGNyaXNpcycsc3RhdGVzOidEZWxoaSDCtyBSYWphc3RoYW4gwrcgQVAnLHBjdDonKzE5JSd9LAogIHtuYW1lOidFdGhuaWMgdGVuc2lvbnMnLHN0YXRlczonTWFuaXB1ciDCtyBBc3NhbSDCtyBXQicscGN0OicrMTclJ30sCl07CnZhciBNT0NLX0Y9WwogIHtuYW1lOidFbGVjdGlvbiByaGV0b3JpYycsc3RhdGVzOidOYXRpb25hbCBwb3N0LWN5Y2xlJyxwY3Q6Jy0zOCUnfSwKICB7bmFtZTonSW5mbGF0aW9uIHByZXNzdXJlJyxzdGF0ZXM6J0Vhc2luZyBuYXRpb25hbGx5JyxwY3Q6Jy0yNCUnfSwKICB7bmFtZTonRmFybWVyIHByb3Rlc3RzJyxzdGF0ZXM6J01vbWVudHVtIGxvc3QnLHBjdDonLTE5JSd9LAogIHtuYW1lOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsc3RhdGVzOidSaWJib24tY3V0dGluZyBkb25lJyxwY3Q6Jy0xNCUnfSwKICB7bmFtZTonUmVsaWdpb3VzIGZlc3RpdmFscycsc3RhdGVzOidQb3N0LXNlYXNvbiBmYWRlJyxwY3Q6Jy0xMSUnfSwKXTsKCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGRhdGE9U0hJRlRTW3BlcmlvZF18fFNISUZUU1snM20nXTsKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NoaWZ0LWxpc3QnKTsKICBpZighZWwpIHJldHVybjsKICBlbC5pbm5lckhUTUw9ZGF0YS5tYXAoZnVuY3Rpb24ocyl7CiAgICByZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjA7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjhweDtvdmVyZmxvdzpoaWRkZW47Ij4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6NnB4IDEwcHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFsbCk7bWFyZ2luLWJvdHRvbTozcHg7Ij5mYWRpbmc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjI7Ij4nK3MuZmFkaW5nKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytzLmZhZGluZ05vdGUrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0id2lkdGg6MjhweDtmbGV4LXNocmluazowO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC40NTtmb250LXNpemU6MTNweDsiPuKGkjwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7cGFkZGluZzo4cHggMTBweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLXJpc2UpO21hcmdpbi1ib3R0b206M3B4OyI+cmlzaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLnJpc2luZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5yaXNpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHRhYil7CiAgdGFiLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpewogICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnN0cmlwLXRhYicpLmZvckVhY2goZnVuY3Rpb24odCl7dC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIHRhYi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtyZW5kZXJTdHJpcCh0YWIuZGF0YXNldC5wZXJpb2QpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHJlbmRlck1vbWVudHVtKCl7CiAgLy8gUmVhZCBmcm9tIFNEIChwb3B1bGF0ZWQgYnkgZmV0Y2hBbGxTdGF0ZXMgZnJvbSBsaXZlIEFQSSkKICB2YXIgbmM9e307CiAgT2JqZWN0LnZhbHVlcyhTRCkuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgIChzLm5hcnJhdGl2ZXN8fFtdKS5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBuY1tuLm5hbWVdPShuY1tuLm5hbWVdfHwwKStuLnZhbDsKICAgIH0pOwogIH0pOwogIHZhciBzb3J0ZWQ9T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pOwogIHZhciByaXNpbmc9c29ydGVkLnNsaWNlKDAsNSk7CiAgdmFyIGZhbGxpbmc9c29ydGVkLnNsaWNlKC01KS5yZXZlcnNlKCk7CiAgdmFyIG14PXJpc2luZy5sZW5ndGg/cmlzaW5nWzBdWzFdOjEwMDsKCiAgLy8gV3JpdGUgdG8gcmlzaW5nLWxpc3QgKG1hdGNoZXMgbmFyLXJvdyBIVE1MKQogIHZhciByRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jpc2luZy1saXN0Jyk7CiAgaWYockVsJiZyaXNpbmcubGVuZ3RoKXsKICAgIHJFbC5pbm5lckhUTUw9cmlzaW5nLm1hcChmdW5jdGlvbihuLGkpewogICAgICB2YXIgdz1NYXRoLm1pbigxMDAsblsxXS9teCoxMDApOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6I2UwNWEyOCI+4oaRIHJpc2luZzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiNlMDVhMjg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byBkZWNsaW5pbmctbGlzdAogIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgaWYoZkVsJiZmYWxsaW5nLmxlbmd0aCl7CiAgICBmRWwuaW5uZXJIVE1MPWZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pewogICAgICB2YXIgdz1NYXRoLm1pbigxMDAsblsxXS9teCoxMDApOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6IzNiYjhkOCI+4oaTIGZhZGluZzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiMzYmI4ZDg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuam9pbignJyk7CiAgfQoKICAvLyBXcml0ZSB0byByZWdpb25hbC1saXN0IOKAlCB0b3Agc3RhdGUgcGVyIHJlZ2lvbiBmcm9tIExJVkUKICB2YXIgcmVnaW9ucz17CiAgICAnTm9ydGgnOlsnRGVsaGknLCdVdHRhciBQcmFkZXNoJywnUHVuamFiJywnSGFyeWFuYScsJ0hpbWFjaGFsIFByYWRlc2gnLCdVdHRhcmFraGFuZCcsJ0phbW11IGFuZCBLYXNobWlyJ10sCiAgICAnRWFzdCc6WydXZXN0IEJlbmdhbCcsJ0JpaGFyJywnSmhhcmtoYW5kJywnT2Rpc2hhJ10sCiAgICAnV2VzdCc6WydNYWhhcmFzaHRyYScsJ0d1amFyYXQnLCdSYWphc3RoYW4nLCdHb2EnXSwKICAgICdTb3V0aCc6WydUYW1pbCBOYWR1JywnS2FybmF0YWthJywnS2VyYWxhJywnQW5kaHJhIFByYWRlc2gnLCdUZWxhbmdhbmEnXSwKICAgICdORSc6WydBc3NhbScsJ01hbmlwdXInLCdOYWdhbGFuZCcsJ01pem9yYW0nLCdNZWdoYWxheWEnLCdUcmlwdXJhJywnQXJ1bmFjaGFsIFByYWRlc2gnLCdTaWtraW0nXSwKICAgICdDZW50cmFsJzpbJ01hZGh5YSBQcmFkZXNoJywnQ2hoYXR0aXNnYXJoJ10sCiAgfTsKICB2YXIgZ0VsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZWdpb25hbC1saXN0Jyk7CiAgaWYoZ0VsKXsKICAgIHZhciByZWdJdGVtcz1PYmplY3QuZW50cmllcyhyZWdpb25zKS5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgcmVnaW9uPWt2WzBdLHN0YXRlcz1rdlsxXTsKICAgICAgdmFyIHRvcD1zdGF0ZXMubWFwKGZ1bmN0aW9uKHMpe3JldHVybiB7bmFtZTpzLGF0dDooTElWRVtzXSYmTElWRVtzXS5hdHRlbnRpb24pfHwwfTt9KQogICAgICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pWzBdOwogICAgICBpZighdG9wfHwhdG9wLmF0dCkgcmV0dXJuIG51bGw7CiAgICAgIHZhciBuYXI9KExJVkVbdG9wLm5hbWVdJiZMSVZFW3RvcC5uYW1lXS5kb21pbmFudF9uYXJyYXRpdmUpfHwn4oCUJzsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjhweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6YmFzZWxpbmU7bWFyZ2luLWJvdHRvbToycHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpIj4nK3JlZ2lvbisnPC9zcGFuPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWFjY2VudCkiPicrdG9wLmF0dC50b0ZpeGVkKDEpKyc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK3RvcC5uYW1lKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweDsiPicrbmFyKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmZpbHRlcihCb29sZWFuKS5qb2luKCcnKTsKICAgIGlmKHJlZ0l0ZW1zKSBnRWwuaW5uZXJIVE1MPXJlZ0l0ZW1zOwogIH0KfQoKCi8vIFNUQVRFIERBVEEKdmFyIFNEPXt9OwoKdmFyIExJVkU9e307CmZ1bmN0aW9uIG5vcm1hbGl6ZUVtb3Rpb25zKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybnt9O3ZhciB2YWxzPU9iamVjdC52YWx1ZXMoZSksdG90PXZhbHMucmVkdWNlKGZ1bmN0aW9uKHMsdil7cmV0dXJuIHMrdjt9LDApO2lmKHRvdDw9MClyZXR1cm57fTtpZih0b3Q8PTEuMDEpe3ZhciBvdXQ9e307T2JqZWN0LmtleXMoZSkuZm9yRWFjaChmdW5jdGlvbihrKXtvdXRba109TWF0aC5yb3VuZChlW2tdKjEwMCk7fSk7cmV0dXJuIG91dDt9cmV0dXJuIGU7fQpmdW5jdGlvbiBkb21pbmFudEVtb3Rpb24oZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJuIG51bGw7dmFyIG14PTAsZG9tPW51bGw7T2JqZWN0LmVudHJpZXMoZSkuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0+bXgpe214PWt2WzFdO2RvbT1rdlswXTt9fSk7cmV0dXJuIGRvbTt9CmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKCFlbClyZXR1cm47ZWwudGV4dENvbnRlbnQ9dmFsO2lmKHZhbCYmdmFsIT09Jy0nKXtlbC5jbGFzc0xpc3QucmVtb3ZlKCdsb2FkaW5nJyk7fX0KCnZhciBERUZBVUxUPXsKICBhdHRlbnRpb246MCxkZWx0YTowLHZlbG9jaXR5OjAsCiAgZW1vdGlvbnM6e30sZG9taW5hbnRfZW1vdGlvbjpudWxsLGRvbWluYW50X25hcnJhdGl2ZTpudWxsLAogIG5hcnJhdGl2ZXM6W10scmlzaW5nOltdLGZhbGxpbmc6W10sCiAgc3VtbWFyeTonJyxhcnRpY2xlczpbXSx0aW1lbGluZTpbXSwKICBuYXJyYXRpdmVIaXN0b3J5OltdLHNpZ25hbF9jb3VudDowLAp9OwoKZnVuY3Rpb24gZyhuKXtyZXR1cm4gU0Rbbl18fE9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7fQoKZnVuY3Rpb24gYUMocyl7CiAgLy8gRHluYW1pYyBzY2FsZTogYWx3YXlzIHNwcmVhZCBmdWxsIGNvbG9yIHJhbmdlIGFjcm9zcyBhY3R1YWwgZGF0YQogIC8vIEdldCBtaW4vbWF4IGZyb20gY3VycmVudCBTRCB0byBub3JtYWxpemUKICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSk7CiAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKTsKICB2YXIgbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogIC8vIE5vcm1hbGl6ZSAwLTEKICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHMtbW4pLyhteC1tbikpKTsKICAvLyBNYXAgdG8gY29sb3Igc3RvcHM6IGRhcmsgYmx1ZSDihpIgdGVhbCDihpIgYW1iZXIg4oaSIG9yYW5nZSDihpIgcmVkCiAgaWYobjwwLjEyKSByZXR1cm4gJyMwZDFlMzAnOwogIGlmKG48MC4yNSkgcmV0dXJuICcjMGUzZDZhJzsKICBpZihuPDAuMzgpIHJldHVybiAnIzBkNWY5MCc7CiAgaWYobjwwLjUwKSByZXR1cm4gJyMwZTdhYWEnOwogIGlmKG48MC42MikgcmV0dXJuICcjMWE5MDkwJzsKICBpZihuPDAuNzIpIHJldHVybiAnI2M4NzAxMCc7CiAgaWYobjwwLjgyKSByZXR1cm4gJyNkODQwMTAnOwogIGlmKG48MC45MikgcmV0dXJuICcjY2MxODA4JzsKICByZXR1cm4gJyNmZjAwMTAnOwp9CmZ1bmN0aW9uIGVDKGUpewogIHZhciBteD0wLGRvbT0ncHJpZGUnOwogIGZvcih2YXIgayBpbiBlKXtpZihlW2tdPm14KXtteD1lW2tdO2RvbT1rO319CiAgcmV0dXJuICh7YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ30pW2RvbV18fCcjMzNhYWNjJzsKfQpmdW5jdGlvbiB2Qyh2KXsKICBpZih2PjAuMikgcmV0dXJuICcjZGMwODE4JzsKICBpZih2PjAuMSkgcmV0dXJuICcjZTA1YTI4JzsKICBpZih2PjAuMDIpIHJldHVybiAnI2NjODgyMic7CiAgaWYodjwtMC4wNSkgcmV0dXJuICcjMjI5OWJiJzsKICByZXR1cm4gJyMxNTIwMzAnOwp9Cgp2YXIgbGF5ZXI9J2F0dGVudGlvbicsU0VMPW51bGwsRkFWUz1uZXcgU2V0KCk7CgovLyBNQVAKZnVuY3Rpb24gcHJval8odyxoLHBhZCl7CiAgcGFkPXBhZHx8MjA7CiAgdmFyIG1pbkxvbj02OC4xLG1heExvbj05Ny40LG1pbkxhdD02LjUsbWF4TGF0PTM3LjE7CiAgdmFyIHNjWD0ody1wYWQqMikvKG1heExvbi1taW5Mb24pOwogIHZhciBzY1k9KGgtcGFkKjIpLyhtYXhMYXQtbWluTGF0KTsKICB2YXIgc2M9TWF0aC5taW4oc2NYLHNjWSk7CiAgdmFyIG94PXBhZCsody1wYWQqMi0obWF4TG9uLW1pbkxvbikqc2MpLzI7CiAgdmFyIG95PXBhZCsoaC1wYWQqMi0obWF4TGF0LW1pbkxhdCkqc2MpLzI7CiAgcmV0dXJuIGZ1bmN0aW9uKGxvbixsYXQpe3JldHVybiBbb3grKGxvbi1taW5Mb24pKnNjLCBveSsobWF4TGF0LWxhdCkqc2NdO307Cn0KZnVuY3Rpb24gZ2VvMnBhdGgoZ2VvbSxwail7CiAgdmFyIGQ9Jyc7CiAgZnVuY3Rpb24gcmluZyhjcyl7dmFyIHM9Jyc7Y3MuZm9yRWFjaChmdW5jdGlvbihjLGkpe3ZhciBwPXBqKGNbMF0sY1sxXSk7cys9KGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pO3JldHVybiBzKydaJzt9CiAgaWYoZ2VvbS50eXBlPT09J1BvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocil7ZCs9cmluZyhyKTt9KTsKICBlbHNlIGlmKGdlb20udHlwZT09PSdNdWx0aVBvbHlnb24nKSBnZW9tLmNvb3JkaW5hdGVzLmZvckVhY2goZnVuY3Rpb24ocCl7cC5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7fSk7CiAgcmV0dXJuIGQ7Cn0KZnVuY3Rpb24gY3RyKGdlb20pewogIHZhciBwdHM9W107CiAgZnVuY3Rpb24gY29sKGMpe2lmKHR5cGVvZiBjWzBdPT09J251bWJlcicpIHB0cy5wdXNoKGMpO2Vsc2UgYy5mb3JFYWNoKGNvbCk7fQogIGNvbChnZW9tLmNvb3JkaW5hdGVzKTsKICBpZighcHRzLmxlbmd0aCkgcmV0dXJuIFswLDBdOwogIHJldHVybiBbcHRzLnJlZHVjZShmdW5jdGlvbihzLHApe3JldHVybiBzK3BbMF07fSwwKS9wdHMubGVuZ3RoLHB0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzFdO30sMCkvcHRzLmxlbmd0aF07Cn0KZnVuY3Rpb24gc05hbWUocHJvcHMpewogIHZhciByYXc9cHJvcHMuc3Rfbm18fHByb3BzLk5BTUVfMXx8cHJvcHMubmFtZXx8cHJvcHMuTkFNRXx8Jyc7CiAgdmFyIG1hcD17J0xhZGFraCc6J0phbW11IGFuZCBLYXNobWlyJywnSmFtbXUgJiBLYXNobWlyJzonSmFtbXUgYW5kIEthc2htaXInLCdVdHRhcmFuY2hhbCc6J1V0dGFyYWtoYW5kJywnQW5kYW1hbiBhbmQgTmljb2Jhcic6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ0FuZGFtYW4gJiBOaWNvYmFyIElzbGFuZCc6J0FuZGFtYW4gYW5kIE5pY29iYXIgSXNsYW5kcycsJ05DVCBvZiBEZWxoaSc6J0RlbGhpJywnUG9uZGljaGVycnknOidQdWR1Y2hlcnJ5JywnRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnLCdEYW1hbiBhbmQgRGl1JzonRGFkcmEgYW5kIE5hZ2FyIEhhdmVsaSBhbmQgRGFtYW4gYW5kIERpdSd9OwogIHJldHVybiBtYXBbcmF3XXx8cmF3Owp9Cgp2YXIgY2FjaGVkR2VvPW51bGw7Cgphc3luYyBmdW5jdGlvbiBsb2FkTWFwKGF0dGVtcHQpewogIGF0dGVtcHQgPSBhdHRlbXB0fHwxOwogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKCdodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvZ2gvdWRpdC0wMDEvaW5kaWEtbWFwcy1kYXRhQG1hc3Rlci90b3BvanNvbi9pbmRpYS5qc29uJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHRvcG89YXdhaXQgci5qc29uKCk7CiAgICBjYWNoZWRHZW89dG9wb2pzb24uZmVhdHVyZSh0b3BvLHRvcG8ub2JqZWN0cy5zdGF0ZXMpOwogICAgcmVuZGVyTWFwKGNhY2hlZEdlbyk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMTAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsMzAwMCk7CiAgICBzZXRUaW1lb3V0KGFwcGx5TGF5ZXIsNjAwMCk7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbbWFwXSBsb2FkIGZhaWxlZCBhdHRlbXB0ICcrYXR0ZW1wdCsnOicsZS5tZXNzYWdlKTsKICAgIGlmKGF0dGVtcHQ8NSl7CiAgICAgIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtsb2FkTWFwKGF0dGVtcHQrMSk7fSwgYXR0ZW1wdCoyMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHZhciBtaT1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubWFwLWlubmVyJyk7CiAgICAgIGlmKG1pKSBtaS5pbm5lckhUTUw9JzxkaXYgc3R5bGU9ImNvbG9yOiMyYTNhNGE7cGFkZGluZzo0MHB4O3RleHQtYWxpZ246Y2VudGVyO2ZvbnQtZmFtaWx5Om1vbm9zcGFjZTtmb250LXNpemU6MTFweCI+TWFwIHVuYXZhaWxhYmxlIOKAlCByZWZyZXNoIHRvIHJldHJ5PC9kaXY+JzsKICAgIH0KICB9Cn0KCmZ1bmN0aW9uIHJlbmRlck1hcChzdGF0ZXMpewogIHZhciB3PTgwMCxoPTgwMCxwaj1wcm9qXyh3LGgsMjgpOwogIHZhciBzZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXN0YXRlcycpOwogIHZhciBwZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLXB1bHNlcycpOwogIHZhciBnZz1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLWdsb3cnKTsKICBzZy5pbm5lckhUTUw9Jyc7cGcuaW5uZXJIVE1MPScnO2dnLmlubmVySFRNTD0nJzsKCiAgc3RhdGVzLmZlYXR1cmVzLmZvckVhY2goZnVuY3Rpb24oZil7CiAgICBpZighZi5nZW9tZXRyeSkgcmV0dXJuOwogICAgdmFyIG5tPXNOYW1lKGYucHJvcGVydGllcyksZD1nKG5tKTsKICAgIHZhciBwYXRoRWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ3BhdGgnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2QnLGdlbzJwYXRoKGYuZ2VvbWV0cnkscGopKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywnc3RhdGUnKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScsbm0pOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJywncmdiYSgyNTUsMjU1LDI1NSwwLjA3KScpOwogICAgcGF0aEVsLnNldEF0dHJpYnV0ZSgnc3Ryb2tlLXdpZHRoJywnMC41Jyk7CiAgICBzZy5hcHBlbmRDaGlsZChwYXRoRWwpOwoKICAgIHZhciBjdD1jdHIoZi5nZW9tZXRyeSksY3A9cGooY3RbMF0sY3RbMV0pOwoKICAgIC8vIEF0bW9zcGhlcmljIGdsb3cgZm9yIGhpZ2gtYXR0ZW50aW9uIHN0YXRlcwogICAgaWYoZC5hdHRlbnRpb24+PTY1KXsKICAgICAgdmFyIGdsb3dFbD1kb2N1bWVudC5jcmVhdGVFbGVtZW50TlMoJ2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJywnZWxsaXBzZScpOwogICAgICB2YXIgZ2xvd1I9TWF0aC5taW4oNjAsMjArZC5hdHRlbnRpb24qMC41KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtnbG93RWwuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdyeCcsZ2xvd1IpO2dsb3dFbC5zZXRBdHRyaWJ1dGUoJ3J5JyxnbG93UiowLjcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWxsJyxhQyhkLmF0dGVudGlvbikpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdvcGFjaXR5JywnMC4wOCcpOwogICAgICBnbG93RWwuc2V0QXR0cmlidXRlKCdmaWx0ZXInLCd1cmwoI3N0YXRlR2xvdyknKTsKICAgICAgZ2xvd0VsLnN0eWxlLmFuaW1hdGlvbj0nZ2xvd1B1bHNlICcrKDIuNStNYXRoLnJhbmRvbSgpKSsncyBlYXNlLWluLW91dCAnKyhNYXRoLnJhbmRvbSgpKjIpKydzIGluZmluaXRlJzsKICAgICAgZ2cuYXBwZW5kQ2hpbGQoZ2xvd0VsKTsKICAgIH0KCiAgICAvLyBEdWFsIHB1bHNlIHJpbmdzIGZvciB2ZXJ5IGhvdCBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj03Mil7CiAgICAgIFswLDFdLmZvckVhY2goZnVuY3Rpb24oaSl7CiAgICAgICAgdmFyIHJpbmc9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2NpcmNsZScpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjeCcsY3BbMF0pO3Jpbmcuc2V0QXR0cmlidXRlKCdjeScsY3BbMV0pOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3B1bHNlLXJpbmcgcCcrKGkrMSkpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2UnLGFDKGQuYXR0ZW50aW9uKSk7CiAgICAgICAgcmluZy5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzEnKTsKICAgICAgICByaW5nLnN0eWxlLmFuaW1hdGlvbkRlbGF5PShNYXRoLnJhbmRvbSgpKjIuNSkrJ3MnOwogICAgICAgIHBnLmFwcGVuZENoaWxkKHJpbmcpOwogICAgICB9KTsKICAgIH0KICB9KTsKICBhcHBseUxheWVyKCk7CiAgYXR0YWNoSW50ZXJhY3Rpb25zKCk7Cn0KCi8vIFNpbmdsZSBzb3VyY2Ugb2YgdHJ1dGggZm9yIGVtb3Rpb24gY29sb3IKLy8gQm90aCBtYXAgYW5kIHBhbmVsIGNhbGwgdGhpcyDigJQgZ3VhcmFudGVlcyB0aGV5IGFsd2F5cyBtYXRjaApmdW5jdGlvbiBnZXRFZmZlY3RpdmVFbW90aW9uKG5tKXsKICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgdmFyIGQ9U0Rbbm1dfHx7fTsKICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CgogIC8vIDEuIFRyeSBMSVZFLmRvbWluYW50X2Vtb3Rpb24gKHNldCBieSAvYXBpL3N0YXRlcykKICB2YXIgZG9tPWxpdmUuZG9taW5hbnRfZW1vdGlvbnx8ZC5kb21pbmFudF9lbW90aW9uOwoKICAvLyAyLiBUcnkgY29tcHV0aW5nIGZyb20gZW1vdGlvbnMgYnJlYWtkb3duCiAgaWYoIWRvbSl7CiAgICB2YXIgZW1vcz1saXZlLmVtb3Rpb25zJiZPYmplY3Qua2V5cyhsaXZlLmVtb3Rpb25zKS5sZW5ndGg/bGl2ZS5lbW90aW9uczooZC5lbW90aW9uc3x8e30pOwogICAgZG9tPWRvbWluYW50RW1vdGlvbihlbW9zKTsKICB9CgogIC8vIDMuIEZhbGxiYWNrOiBpbmZlciBmcm9tIGRvbWluYW50IG5hcnJhdGl2ZSAoc2FtZSBsb2dpYyBldmVyeXdoZXJlKQogIGlmKCFkb20pewogICAgdmFyIG5wPShsaXZlLmRvbWluYW50X25hcnJhdGl2ZXx8ZC5kb21pbmFudF9uYXJyYXRpdmV8fCcnKS50b0xvd2VyQ2FzZSgpOwogICAgaWYobnAubWF0Y2goL2JvcmRlcnx0ZXJyb3J8c2VjdXJpdHl8Y29uZmxpY3R8YXR0YWNrfHdhcnxpbmZpbHRyYXQvKSkgZG9tPSdmZWFyJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL3NjYW18Y29ycnVwdHxwcm90ZXN0fGFycmVzdHx2aW9sZW5jZXxvdXRyYWdlfGNyaW1lLykpIGRvbT0nYW5nZXInOwogICAgZWxzZSBpZihucC5tYXRjaCgvZGV2ZWxvcHxpbnZlc3R8Z3Jvd3RofGxhdW5jaHxpbmF1Z3VyfHJlZm9ybXxwcm9ncmVzc3xib29zdC8pKSBkb209J2hvcGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvY3VsdHVyZXxoZXJpdGFnZXxwcmlkZXx2aWN0b3J5fGNlbGVicmF0fG1lZGFsfGFjaGlldmVtZW50LykpIGRvbT0ncHJpZGUnOwogICAgZWxzZSBpZihucC5tYXRjaCgvZmxvb2R8ZHJvdWdodHx1bmVtcGxveW1lbnR8aW5mbGF0aW9ufHNob3J0YWdlfGNyaXNpc3xjb25jZXJuLykpIGRvbT0nYW54aWV0eSc7CiAgICBlbHNlIGlmKChsaXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDApPjUpIGRvbT0nYW54aWV0eSc7IC8vIGFjdGl2ZSBzdGF0ZSBkZWZhdWx0CiAgICBlbHNlIGRvbT0nYW54aWV0eSc7IC8vIGdsb2JhbCBkZWZhdWx0CiAgfQoKICByZXR1cm4gZG9tOwp9CgovLyBHZXQgZXN0aW1hdGVkIGVtb3Rpb24gYnJlYWtkb3duIChmb3IgcGFuZWwgZG9udXQgd2hlbiByZWFsIGRhdGEgbWlzc2luZykKZnVuY3Rpb24gZ2V0RW1vdGlvbkJyZWFrZG93bihubSl7CiAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogIHZhciBkPVNEW25tXXx8e307CiAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICBpZihPYmplY3Qua2V5cyhlbW9zKS5sZW5ndGgpIHJldHVybiB7ZW1vdGlvbnM6ZW1vcyxlc3RpbWF0ZWQ6ZmFsc2V9OwogIC8vIEJ1aWxkIHNrZXdlZCBkaXN0cmlidXRpb24gZnJvbSBlZmZlY3RpdmUgZW1vdGlvbgogIHZhciBkb209Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgdmFyIGJhc2U9e2FueGlldHk6MTMsYW5nZXI6MTMsaG9wZToxMyxwcmlkZToxMyxmZWFyOjEzfTsKICBiYXNlW2RvbV09NDg7CiAgcmV0dXJuIHtlbW90aW9uczpiYXNlLGVzdGltYXRlZDp0cnVlfTsKfQoKZnVuY3Rpb24gYXBwbHlMYXllcigpewogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpLGQ9ZyhubSksZmlsbDsKICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpIGZpbGw9YUMoZC5hdHRlbnRpb24pOwogICAgZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgdmFyIGRlPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgICBmaWxsPWVNYXBbZGVdfHwnIzMzNDQ1NSc7CiAgICB9CiAgICBlbHNlIGZpbGw9dkMoZC52ZWxvY2l0eSk7CiAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbCcsZmlsbCk7CiAgICAoZnVuY3Rpb24oKXsKICAgICAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oeCl7cmV0dXJuIHguYXR0ZW50aW9ufHwwO30pOwogICAgICB2YXIgbW49TWF0aC5taW4uYXBwbHkobnVsbCxzY29yZXMpLG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAgICAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChkLmF0dGVudGlvbi1tbikvKG14LW1uKSkpOwogICAgICBwLnNldEF0dHJpYnV0ZSgnZmlsbC1vcGFjaXR5JyxsYXllcj09PSdhdHRlbnRpb24nP01hdGgubWF4KDAuMywwLjMrbiowLjcpOjAuODUpOwogICAgfSkoKTsKICB9KTsKfQoKZnVuY3Rpb24gYXR0YWNoSW50ZXJhY3Rpb25zKCl7CiAgdmFyIHRpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9vbHRpcCcpOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsZnVuY3Rpb24oZSl7CiAgICAgIHZhciBubT1wLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk7CiAgICAgIHZhciBkPWcobm0pOwogICAgICB2YXIgbGl2ZT1MSVZFW25tXXx8e307CiAgICAgIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICAgICAgdmFyIHBhbD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBsYXRlc3Q9Jyc7CiAgICAgIGlmKGQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzLmxlbmd0aCkgbGF0ZXN0PWQubmFycmF0aXZlc1swXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2QubmFycmF0aXZlc1swXS5uYW1lLnNsaWNlKDEpOwogICAgICBlbHNlIGlmKGxpdmUuZG9taW5hbnRfbmFycmF0aXZlKSBsYXRlc3Q9bGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbGl2ZS5kb21pbmFudF9uYXJyYXRpdmUuc2xpY2UoMSk7CgogICAgICB2YXIgcm93cz0nJzsKICAgICAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICAgICAgdmFyIGF0dD1saXZlLmF0dGVudGlvbnx8ZC5hdHRlbnRpb258fDA7CiAgICAgICAgdmFyIGRsdD1saXZlLmRlbHRhfHxkLmRlbHRhfHwwOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkF0dGVudGlvbjwvc3Bhbj48c3Ryb25nPicrYXR0LnRvRml4ZWQoMSkrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAoZGx0IT09MD8nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+MjRoIHNoaWZ0PC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrKGRsdD4wPycjZTA1YTI4JzonIzNiYjhkOCcpKyciPicrKGRsdD4wPycrJzonJykrZGx0Kyc8L3N0cm9uZz48L2Rpdj4nOicnKSsKICAgICAgICAgIChsYXRlc3Q/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPlRvcCBuYXJyYXRpdmU8L3NwYW4+PHN0cm9uZz4nK2xhdGVzdCsnPC9zdHJvbmc+PC9kaXY+JzonJyk7CiAgICAgIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIGRvbUVtbz1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgICBpZihkb21FbW8pewogICAgICAgICAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6ZC5lbW90aW9uc3x8e307CiAgICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Eb21pbmFudDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3BhbFtkb21FbW9dKyciPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW9zKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgICAgICAgIHZhciB0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7CiAgICAgICAgICBpZih0b3Q+MCYmdG90PD0xLjAxKXtlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybltrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO3RvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTt9CiAgICAgICAgICByb3dzKz1lTC5zbGljZSgwLDMpLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJ0dC1yIj48c3BhbiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NHB4Ij48c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK3BhbFtrdlswXV0rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicra3ZbMF0rJzwvc3Bhbj48c3Ryb25nPicrTWF0aC5yb3VuZChrdlsxXSoxMDAvTWF0aC5tYXgoMSx0b3QpKSsnJTwvc3Ryb25nPjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgICAgfQogICAgICB9IGVsc2UgewogICAgICAgIHZhciB2ZWw9bGl2ZS52ZWxvY2l0eXx8ZC52ZWxvY2l0eXx8MDsKICAgICAgICB2YXIgdmVsRGlyPXZlbD4wLjE/J1Jpc2luZyBmYXN0Jzp2ZWw+MC4wMj8nUmlzaW5nJzp2ZWw8LTAuMDU/J0Nvb2xpbmcnOidTdGFibGUnOwogICAgICAgIHZhciB2ZWxDb2w9dmVsPjAuMDI/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgICAgIHJvd3M9JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPk1vbWVudHVtPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvc3Ryb25nPjwvZGl2PicrCiAgICAgICAgICAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RGlyZWN0aW9uPC9zcGFuPjxzdHJvbmcgc3R5bGU9ImNvbG9yOicrdmVsQ29sKyciPicrdmVsRGlyKyc8L3N0cm9uZz48L2Rpdj4nOwogICAgICB9CgogICAgICB0aXAuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJ0dC1uIj4nK25tKyc8L2Rpdj4nK3Jvd3MrKGxhdGVzdCYmbGF5ZXIhPT0nYXR0ZW50aW9uJz8nPGRpdiBjbGFzcz0idHQtbmFyIj48c3Ryb25nPk5hcnJhdGl2ZTwvc3Ryb25nPicrbGF0ZXN0Kyc8L2Rpdj4nOicnKTsKICAgICAgdmFyIHJlY3Q9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgICB0aXAuc3R5bGUubGVmdD1NYXRoLm1pbihlLmNsaWVudFgtcmVjdC5sZWZ0KzE0LHJlY3Qud2lkdGgtMTkwKSsncHgnOwogICAgICB0aXAuc3R5bGUudG9wPU1hdGgubWluKGUuY2xpZW50WS1yZWN0LnRvcCsxNCxyZWN0LmhlaWdodC0xNTApKydweCc7CiAgICAgIHRpcC5zdHlsZS5vcGFjaXR5PScxJzsKICAgIH0pOwpwLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbGVhdmUnLGZ1bmN0aW9uKCl7dGlwLnN0eWxlLm9wYWNpdHk9MDt9KTsKICAgIHAuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7c2VsZWN0XyhwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJykpO30pOwogIH0pOwp9CgovLyBTVEFURSBQQU5FTApmdW5jdGlvbiBzZWxlY3RfKG5tKXsKICBTRUw9bm07CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykuZm9yRWFjaChmdW5jdGlvbihwKXsKICAgIHAuY2xhc3NMaXN0LnRvZ2dsZSgnc2VsZWN0ZWQnLHAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKT09PW5tKTsKICB9KTsKICAvLyBTaG93IGxvYWRpbmcgc3RhdGUgaW1tZWRpYXRlbHkgd2l0aCB3aGF0ZXZlciBMSVZFIGRhdGEgd2UgaGF2ZQogIHZhciBwYW5lbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJyk7CiAgaWYocGFuZWwpewogICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgcGFuZWwuaW5uZXJIVE1MPQogICAgICAnPGRpdiBjbGFzcz0ic3AtaGVhZCI+JysKICAgICAgICAnPGRpdj48ZGl2IGNsYXNzPSJzcC1layI+JysobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGJ1dHRvbiBjbGFzcz0iZmF2LWJ0biAnKyhGQVZTLmhhcyhubSk/J29uJzonJykrJyIgZGF0YS1ubT0iJytubSsnIiBvbmNsaWNrPSJ0b2dnbGVGYXYodGhpcy5kYXRhc2V0Lm5tKSIgdGl0bGU9IlRyYWNrIj4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAgICc8L2J1dHRvbj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MjBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wOGVtIj4nKwogICAgICAgICdMb2FkaW5nIHNpZ25hbHMgZm9yICcrbm0rJy4uLicrCiAgICAgICAgKGxpdmUuYXR0ZW50aW9uPyc8YnI+PGJyPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Y29sb3I6dmFyKC0taW5rKSI+QXR0ZW50aW9uICcrbGl2ZS5hdHRlbnRpb24udG9GaXhlZCgxKSsnPC9zcGFuPic6JycpKwogICAgICAgIChsaXZlLmRvbWluYW50X2Vtb3Rpb24/Jzxicj48c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nK2xpdmUuZG9taW5hbnRfZW1vdGlvbisnIHNpZ25hbCBkb21pbmFudDwvc3Bhbj4nOicnKSsKICAgICAgJzwvZGl2Pic7CiAgfQogIC8vIEZldGNoIGZ1bGwgZGV0YWlsIHRoZW4gcmVuZGVyCiAgZmV0Y2hEZXRhaWwobm0pLnRoZW4oZnVuY3Rpb24oKXsKICAgIGlmKFNFTD09PW5tKXsKICAgICAgcmVuZGVyUGFuZWwobm0pOwogICAgICAvLyBVcGRhdGUganVzdCB0aGlzIHN0YXRlJ3MgbWFwIGNvbG9yIHRvIG1hdGNoIHRoZSBwYW5lbAogICAgICB2YXIgcGF0aD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK25tKyciXScpOwogICAgICBpZihwYXRoJiZsYXllcj09PSdlbW90aW9uJyl7CiAgICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICAgIHZhciBlTWFwPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKICAgICAgICB2YXIgZG9tPWxpdmUuZG9taW5hbnRfZW1vdGlvbnx8ZG9taW5hbnRFbW90aW9uKGxpdmUuZW1vdGlvbnN8fHt9KTsKICAgICAgICBpZihkb20mJmVNYXBbZG9tXSkgcGF0aC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGVNYXBbZG9tXSk7CiAgICAgIH0gZWxzZSB7CiAgICAgICAgYXBwbHlMYXllcigpOwogICAgICB9CiAgICB9CiAgfSkuY2F0Y2goZnVuY3Rpb24oZSl7CiAgICBjb25zb2xlLndhcm4oJ1tzZWxlY3RdJyxlKTsKICAgIGlmKFNFTD09PW5tKSByZW5kZXJQYW5lbChubSk7CiAgfSk7Cn0KCmZ1bmN0aW9uIHJlbmRlclBhbmVsKG5tKXsKICB2YXIgZD1nKG5tKTsKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKCFwYW5lbCkgcmV0dXJuOwogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICB2YXIgaGVhZGVyPQogICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICc8ZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLWVrIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+JysKICAgICAgICAgIChsYXllcj09PSdhdHRlbnRpb24nPydOYXJyYXRpdmUgcGFuZWwnOmxheWVyPT09J2Vtb3Rpb24nPydFbW90aW9uYWwgcmVnaXN0ZXInOidNb21lbnR1bSBwYW5lbCcpKwogICAgICAgICAgKGQuY29uZmlkZW5jZT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/J3JnYmEoNTEsMjA0LDEwMiwwLjEpJzpkLmNvbmZpZGVuY2U9PT0nTUVESVVNJz8ncmdiYSgyMjQsOTAsNDAsMC4xKSc6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknKSsnO2NvbG9yOicrKGQuY29uZmlkZW5jZT09PSdISUdIJz8nIzMzY2M2Nic6ZC5jb25maWRlbmNlPT09J01FRElVTSc/JyNlMDVhMjgnOidyZ2JhKDI1NSwyNTUsMjU1LDAuMyknKSsnIj4nK2QuY29uZmlkZW5jZSsnIFNJR05BTDwvc3Bhbj4nOicnKSsKICAgICAgICAgIChkLmlzX3JlZ2lvbmFsX3N0b3J5Pyc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3BhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpO2NvbG9yOiMzYmI4ZDgiPlJFR0lPTkFMIFNQSUtFPC9zcGFuPic6JycpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1uYW1lIj4nK25tKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IicrKEZBVlMuaGFzKG5tKT8nY3VycmVudENvbG9yJzonbm9uZScpKyciIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+PHBhdGggZD0iTTE5IDIxbC03LTUtNyA1VjVhMiAyIDAgMCAxIDItMmgxMGEyIDIgMCAwIDEgMiAyeiIvPjwvc3ZnPicrCiAgICAgICc8L2J1dHRvbj4nKwogICAgJzwvZGl2Pic7CgogIHZhciBib2R5PScnOwoKICBpZihsYXllcj09PSdhdHRlbnRpb24nKXsKICAgIHZhciBkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPyd1cCc6J2RuJzsKICAgIHZhciBuYXJyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdGw9KGQudGltZWxpbmUmJmQudGltZWxpbmUubGVuZ3RoKT9kLnRpbWVsaW5lOlswLDAsMCwwLDAsMCwwLGQuYXR0ZW50aW9ufHwwXTsKICAgIHZhciB0bW49TWF0aC5taW4uYXBwbHkobnVsbCx0bCksdG14PU1hdGgubWF4LmFwcGx5KG51bGwsdGwpLHRyPU1hdGgubWF4KDEsdG14LXRtbik7CiAgICB2YXIgdHc9MjYwLHRoPTYyLHRwPTU7CiAgICB2YXIgcHRzPXRsLm1hcChmdW5jdGlvbih2LGkpe3JldHVyblt0cCsoaS8odGwubGVuZ3RoLTEpKSoodHctdHAqMiksdHArKDEtKHYtdG1uKS90cikqKHRoLXRwKjIpXTt9KTsKICAgIHZhciBwRD1wdHMubWFwKGZ1bmN0aW9uKHAsaSl7cmV0dXJuKGk9PT0wPydNJzonTCcpK3BbMF0udG9GaXhlZCgxKSsnLCcrcFsxXS50b0ZpeGVkKDEpO30pLmpvaW4oJycpOwogICAgdmFyIGFEPXBEKycgTCcrcHRzW3B0cy5sZW5ndGgtMV1bMF0rJywnKyh0aC10cCkrJyBMJytwdHNbMF1bMF0rJywnKyh0aC10cCkrJyBaJzsKICAgIHZhciBhYz1hQyhkLmF0dGVudGlvbnx8MCk7CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ0hvdyBpbnRlbnNlbHkgJysobm0uc3BsaXQoJyAnKVswXSkrJyBpcyBiZWluZyBkaXNjdXNzZWQgbmF0aW9uYWxseS4gU2NvcmUgb2YgJytkLmF0dGVudGlvbisnIG1lYW5zICcrKGQuYXR0ZW50aW9uPjYwPyd2ZXJ5IGhpZ2gg4oCUIHRoaXMgc3RhdGUgZG9taW5hdGVzIG5hdGlvbmFsIGRpc2NvdXJzZSc6ZC5hdHRlbnRpb24+MzU/J2VsZXZhdGVkIOKAlCBjbGVhcmx5IGluIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24nOmQuYXR0ZW50aW9uPjE1Pydtb2RlcmF0ZSDigJQgc29tZSBuYXRpb25hbCBjb3ZlcmFnZSc6ZC5hdHRlbnRpb24+NT8nbG93IOKAlCBsaW1pdGVkIG5hdGlvbmFsIGF0dGVudGlvbic6J21pbmltYWwg4oCUIGZldyBzaWduYWxzIGRldGVjdGVkJykrJy4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBjbGFzcz0iaW5zaWdodCIgc3R5bGU9IicrKGQuY29uZmlkZW5jZT09PSJMT1ciPydib3JkZXItY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjA2KTtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1zdHlsZTppdGFsaWMnOicnKSsnIj4nKygoZC5jb25maWRlbmNlPT09IkxPVyImJiFkLnN1bW1hcnkpPydMaW1pdGVkIHNpZ25hbHMgZGV0ZWN0ZWQgZm9yICcrbm0rJy4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLic6ZC5zdW1tYXJ5fHwnQ29sbGVjdGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKSsnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj4yNGggc2hpZnQ8L2Rpdj48ZGl2IGNsYXNzPSJzcy1kZWx0YSAnK2RDKyciPicrZFMrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VG9wIG5hcnJhdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNzLW5hciI+JysobmFyclswXT9uYXJyWzBdLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbmFyclswXS5uYW1lLnNsaWNlKDEpOifigJQnKSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPk5hcnJhdGl2ZSBicmVha2Rvd248L2Rpdj4nKwogICAgICAgIChuYXJyLmxlbmd0aD8KICAgICAgICAgICc8ZGl2IGNsYXNzPSJuYXItbGlzdCI+JytuYXJyLm1hcChmdW5jdGlvbihuKXsKICAgICAgICAgICAgdmFyIG5uPW4ubmFtZT9uLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYW1lLnNsaWNlKDEpOm4ubmFtZTsKICAgICAgICAgICAgdmFyIHZhbD10eXBlb2Ygbi52YWw9PT0nbnVtYmVyJz9uLnZhbDowOwogICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtMiI+PGRpdiBjbGFzcz0ibmktbGFiZWwiPicrbm4rKG4uZGlyPT09J3VwJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojZTA1YTI4O2ZvbnQtc2l6ZTo5cHgiPuKGkTwvc3Bhbj4nOm4uZGlyPT09J2Rvd24nPycgPHNwYW4gc3R5bGU9ImNvbG9yOiMzYmI4ZDg7Zm9udC1zaXplOjlweCI+4oaTPC9zcGFuPic6JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS12YWwiPicrdmFsLnRvRml4ZWQoMSkrJyU8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLHZhbCoyLjUpKyclO2JhY2tncm91bmQ6Jysobi5kaXI9PT0ndXAnPycjZTA1YTI4JzpuLmRpcj09PSdkb3duJz8nIzNiYjhkOCc6JyMzMzQ0NTUnKSsnIj48L2Rpdj48L2Rpdj48L2Rpdj4nOwogICAgICAgICAgfSkuam9pbignJykrJzwvZGl2Pic6CiAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvdy1zaWduYWwgcmVnaW9uLiBNb25pdG9yaW5nIHJlZ2lvbmFsIHNvdXJjZXMuPC9kaXY+JykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BdHRlbnRpb24g4oCUIDggZGF5czwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InRsLXdyYXAiPjxzdmcgdmlld0JveD0iMCAwICcrdHcrJyAnK3RoKyciIHN0eWxlPSJ3aWR0aDoxMDAlO2hlaWdodDoxMDAlIj4nKwogICAgICAgICAgJzxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0idGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKyciIHgxPSIwIiB4Mj0iMCIgeTE9IjAiIHkyPSIxIj4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAuMjUiLz4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMCIvPicrCiAgICAgICAgICAnPC9saW5lYXJHcmFkaWVudD48L2RlZnM+JysKICAgICAgICAgICc8cGF0aCBkPSInK2FEKyciIGZpbGw9InVybCgjdGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKycpIiAvPicrCiAgICAgICAgICAnPHBhdGggZD0iJytwRCsnIiBmaWxsPSJub25lIiBzdHJva2U9IicrYWMrJyIgc3Ryb2tlLXdpZHRoPSIxLjIiLz4nKwogICAgICAgICAgcHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAnPGNpcmNsZSBjeD0iJytwWzBdKyciIGN5PSInK3BbMV0rJyIgcj0iJysoaT09PXB0cy5sZW5ndGgtMT8yLjI6MS4yKSsnIiBmaWxsPSInK2FjKyciLz4nO30pLmpvaW4oJycpKwogICAgICAgICc8L3N2Zz48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFscyA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nKyhkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aD9kLmFydGljbGVzLmxlbmd0aDowKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9ImFydC1saXN0Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMubWFwKGZ1bmN0aW9uKGEpe3JldHVybiAnPGRpdiBjbGFzcz0iYXJ0LWl0ZW0iPjxkaXYgY2xhc3M9ImFydC1zcmMiPicrKGEuc3JjfHwnJykrJzwvZGl2PjxkaXYgY2xhc3M9ImFydC10eHQiPicrKGEudHh0fHxhLnRpdGxlfHwnJykrJzwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk6CiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo2cHggMCI+Tm8gc2lnbmFscyBjb2xsZWN0ZWQgeWV0LjwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKCiAgfSBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgIC8vIFVzZSBzYW1lIGZ1bmN0aW9ucyBhcyBtYXAg4oCUIGd1YXJhbnRlZWQgdG8gbWF0Y2gKICAgIHZhciBtYXBEb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICB2YXIgYnJlYWtkb3duPWdldEVtb3Rpb25CcmVha2Rvd24obm0pOwogICAgdmFyIGVtb3Rpb25zPWJyZWFrZG93bi5lbW90aW9uczsKICAgIHZhciBoYXNFbW9zPSFicmVha2Rvd24uZXN0aW1hdGVkOwogICAgdmFyIGVMPU9iamVjdC5lbnRyaWVzKGVtb3Rpb25zKTsKICAgIHZhciBlVG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgaWYoZVRvdD4wJiZlVG90PD0xLjAxKXtlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybltrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO30KICAgIHZhciB0b3Q9TWF0aC5tYXgoMSxlTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKSk7CiAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgIGlmKCFlTC5sZW5ndGgpe3BhbmVsLmlubmVySFRNTD1oZWFkZXIrJzxkaXYgc3R5bGU9InBhZGRpbmc6MjBweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHgiPk5vIGVtb3Rpb24gZGF0YSB5ZXQuPC9kaXY+JztyZXR1cm47fQogICAgLy8gZG9tRW1vID0gc2FtZSBhcyBtYXAgY29sb3IgKGZyb20gZ2V0RWZmZWN0aXZlRW1vdGlvbikKICAgIHZhciBkb21FbW89bWFwRG9tRW1vOwogICAgLy8gUmVvcmRlciBlTCBzbyBkb21pbmFudCBzaG93cyBmaXJzdAogICAgZUwuc29ydChmdW5jdGlvbihhLGIpewogICAgICBpZihhWzBdPT09ZG9tRW1vKSByZXR1cm4gLTE7CiAgICAgIGlmKGJbMF09PT1kb21FbW8pIHJldHVybiAxOwogICAgICByZXR1cm4gYlsxXS1hWzFdOwogICAgfSk7CiAgICB2YXIgZG9tUGN0PU1hdGgucm91bmQoKGVMWzBdP2VMWzBdWzFdOjIwKSoxMDAvdG90KTsKICAgIHZhciBuYXJyMj1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHRvcE5hclN0cj1uYXJyMi5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJyBhbmQgJyk7CiAgICB2YXIgd2hhdEl0PXthbnhpZXR5OidVbmNlcnRhaW50eSBhbmQgdW5lYXNlIGluICcrbm0rKHRvcE5hclN0cj8nLiBTaWduYWxzOiAnK3RvcE5hclN0cisnLic6JycpLGFuZ2VyOidPdXRyYWdlIGFuZCBwcmVzc3VyZSBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gRHJpdmVuIGJ5OiAnK3RvcE5hclN0cisnLic6JycpLGhvcGU6J09wdGltaXNtIGFuZCBwcm9ncmVzcyBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gQXJvdW5kOiAnK3RvcE5hclN0cisnLic6JycpLHByaWRlOidJZGVudGl0eSBhbmQgYWNoaWV2ZW1lbnQgaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKSxmZWFyOidUaHJlYXQgcGVyY2VwdGlvbiBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gQXJvdW5kOiAnK3RvcE5hclN0cisnLic6JycpfTsKICAgIHZhciBjdW1BPS1NYXRoLlBJLzIsY3g9MzgsY3k9MzgsUj0zMyxyaT0yMDsKICAgIHZhciBhcmNzPWVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciBrPWt2WzBdLHY9a3ZbMV0sZnI9di90b3QsYTE9Y3VtQSxhMj1jdW1BK2ZyKk1hdGguUEkqMjtjdW1BPWEyOwogICAgICB2YXIgbGc9KGEyLWExKT5NYXRoLlBJPzE6MDsKICAgICAgdmFyIHgxPWN4K01hdGguY29zKGExKSpSLHkxPWN5K01hdGguc2luKGExKSpSLHgyPWN4K01hdGguY29zKGEyKSpSLHkyPWN5K01hdGguc2luKGEyKSpSOwogICAgICB2YXIgeDM9Y3grTWF0aC5jb3MoYTIpKnJpLHkzPWN5K01hdGguc2luKGEyKSpyaSx4ND1jeCtNYXRoLmNvcyhhMSkqcmkseTQ9Y3krTWF0aC5zaW4oYTEpKnJpOwogICAgICByZXR1cm4gJzxwYXRoIGQ9Ik0nK3gxLnRvRml4ZWQoMSkrJywnK3kxLnRvRml4ZWQoMSkrJyBBJytSKycsJytSKycgMCAnK2xnKycgMSAnK3gyLnRvRml4ZWQoMSkrJywnK3kyLnRvRml4ZWQoMSkrJyBMJyt4My50b0ZpeGVkKDEpKycsJyt5My50b0ZpeGVkKDEpKycgQScrcmkrJywnK3JpKycgMCAnK2xnKycgMCAnK3g0LnRvRml4ZWQoMSkrJywnK3k0LnRvRml4ZWQoMSkrJyBaIiBmaWxsPSInK3BhbFtrXSsnIiBvcGFjaXR5PSIwLjkiLz4nOwogICAgfSkuam9pbignJyk7CiAgICB2YXIgZWRlc2M9e2FueGlldHk6J1VuY2VydGFpbnR5LCB3b3JyeScsYW5nZXI6J091dHJhZ2UsIHByb3Rlc3QnLGhvcGU6J09wdGltaXNtLCBwcm9ncmVzcycscHJpZGU6J0FjaGlldmVtZW50LCBpZGVudGl0eScsZmVhcjonVGhyZWF0LCBpbnNlY3VyaXR5J307CiAgICBib2R5Kz0KICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7bGV0dGVyLXNwYWNpbmc6MC4wOGVtO2NvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjhweCAwIDRweCAwO2xpbmUtaGVpZ2h0OjEuNiI+JysKICAgICAgJ1RoZSBlbW90aW9uYWwgdW5kZXJjdXJyZW50IG9mIHNpZ25hbHMgZnJvbSAnK25tKycuIFdoYXQgdG9uZSBkb21pbmF0ZXMgdGhlIHBvbGl0aWNhbCBkaXNjb3Vyc2Ug4oCUIG91dHJhZ2UsIGhvcGUsIGZlYXIsIG9yIGFueGlldHk/JysKICAgICc8L2Rpdj4nKwogICAgKCFoYXNFbW9zPyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjZweCAxMXB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7bWFyZ2luLWJvdHRvbToxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpIj5Fc3RpbWF0ZWQgZnJvbSBzaWduYWwgZGlyZWN0aW9uIOKAlCBsaW1pdGVkIGRpcmVjdCBlbW90aW9uIGRhdGEuPC9kaXY+JzonJykrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjE0cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6JytwYWxbZG9tRW1vXSsnMTQ7Ym9yZGVyOjFweCBzb2xpZCAnK3BhbFtkb21FbW9dKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3BhbFtkb21FbW9dKyc7bWFyZ2luLWJvdHRvbTo2cHgiPkRvbWluYW50IGVtb3Rpb248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI2cHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluaykiPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweCI+Jytkb21QY3QrJyUgwrcgJytubSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjhweDtsaW5lLWhlaWdodDoxLjU7Zm9udC1zdHlsZTppdGFsaWMiPicrd2hhdEl0W2RvbUVtb10rJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5FbW90aW9uYWwgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsiPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgNzYgNzYiIHN0eWxlPSJ3aWR0aDo3MnB4O2hlaWdodDo3MnB4O2ZsZXgtc2hyaW5rOjAiPicrYXJjcysnPC9zdmc+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4OyI+JysKICAgICAgICAgICAgZUwubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgICAgICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLHBjdD1NYXRoLnJvdW5kKHYqMTAwL3RvdCk7CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmNlbnRlcjttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo2cHg7Ij48c3BhbiBzdHlsZT0id2lkdGg6N3B4O2hlaWdodDo3cHg7Ym9yZGVyLXJhZGl1czoycHg7YmFja2dyb3VuZDonK3BhbFtrXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTEuNXB4O2NvbG9yOicrKGs9PT1kb21FbW8/J3ZhcigtLWluayknOid2YXIoLS1kaW0pJykrJyI+JytrLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2suc2xpY2UoMSkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0taW5rKSI+JytwY3QrJyU8L3NwYW4+JysKICAgICAgICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6JytwY3QrJyU7YmFja2dyb3VuZDonK3BhbFtrXSsnO29wYWNpdHk6MC43O2JvcmRlci1yYWRpdXM6MXB4Ij48L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICAgKGs9PT1kb21FbW8/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK2VkZXNjW2tdKyc8L2Rpdj4nOicnKSsKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbCBoZWFkbGluZXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHg7Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMuc2xpY2UoMCw1KS5tYXAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIGVDb2xvcj17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6NnB4O3BhZGRpbmc6NnB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzKTsiPicrCiAgICAgICAgICAgICAgICAoYS5lbW90aW9uPyc8c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK2VDb2xvclthLmVtb3Rpb25dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2s7bWFyZ2luLXRvcDo1cHg7ZmxleC1zaHJpbms6MCI+PC9zcGFuPic6JycpKwogICAgICAgICAgICAgICAgJzxkaXY+PGRpdiBzdHlsZT0iZm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQiPicrKGEudHh0fHxhLnRpdGxlfHwnJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweCI+JysoYS5zcmN8fCcnKSsoYS5lbW90aW9uPycgwrcgJythLmVtb3Rpb246JycpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjRweCAwIj5ObyBzaWduYWxzIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSB7CiAgICB2YXIgdmVsPWQudmVsb2NpdHl8fDA7CiAgICB2YXIgdmVsRGlyPXZlbD4wLjE1PydSaXNpbmcgZmFzdCc6dmVsPjAuMDU/J1Jpc2luZyc6dmVsPC0wLjE/J0Nvb2xpbmcgZmFzdCc6dmVsPC0wLjAyPydDb29saW5nJzonU3RhYmxlJzsKICAgIHZhciB2ZWxDb2w9dmVsPjAuMDU/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgdmFyIHZlbERlc2M9eydSaXNpbmcgZmFzdCc6J1NpZ25hbCB2b2x1bWUgc3VyZ2luZy4nLCdSaXNpbmcnOidBdHRlbnRpb24gYnVpbGRpbmcuJywnU3RhYmxlJzonQmFsYW5jZWQgbW9tZW50dW0uJywnQ29vbGluZyc6J0F0dGVudGlvbiBmYWRpbmcuJywnQ29vbGluZyBmYXN0JzonU2hhcnAgc2lnbmFsIGRlY2F5Lid9OwogICAgdmFyIG5hcnIzPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgcmlzaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J3VwJzt9KTsKICAgIHZhciBmYWxsaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J2Rvd24nO30pOwogICAgdmFyIGN0eD0nJzsKICAgIGlmKHZlbD4wLjA1JiZyaXNpbmdOYXJzLmxlbmd0aCkgY3R4PSdEcml2ZW4gYnkgcmlzaW5nIHNpZ25hbHMgYXJvdW5kIDxzdHJvbmc+JytyaXNpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignPC9zdHJvbmc+IGFuZCA8c3Ryb25nPicpKyc8L3N0cm9uZz4uJzsKICAgIGVsc2UgaWYodmVsPC0wLjA1JiZmYWxsaW5nTmFycy5sZW5ndGgpIGN0eD0nU2lnbmFscyBhcm91bmQgPHN0cm9uZz4nK2ZhbGxpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignPC9zdHJvbmc+IGFuZCA8c3Ryb25nPicpKyc8L3N0cm9uZz4gbG9zaW5nIHRyYWN0aW9uLic7CiAgICBlbHNlIGN0eD0nU2lnbmFsIHZvbHVtZSAnKyh2ZWw+MC4wMj8nYnVpbGRpbmcnOidzdGFibGUnKSsnIGluICcrbm0rJy4nOwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdJcyBhdHRlbnRpb24gZm9yICcrbm0rJyBncm93aW5nIG9yIGZhZGluZz8gUmlzaW5nIG1vbWVudHVtIG1lYW5zIGEgbmFycmF0aXZlIGlzIGFjY2VsZXJhdGluZy4gQ29vbGluZyBtZWFucyB0aGUgc3RvcnkgaXMgbG9zaW5nIHRyYWN0aW9uLicrCiAgICAnPC9kaXY+JysKICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjE0cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6Jyt2ZWxDb2wrJzE0O2JvcmRlcjoxcHggc29saWQgJyt2ZWxDb2wrJzMzO21hcmdpbi1ib3R0b206MTJweDsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOicrdmVsQ29sKyc7bWFyZ2luLWJvdHRvbTo2cHgiPlNpZ25hbCBtb21lbnR1bTwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTtnYXA6MTBweDttYXJnaW4tYm90dG9tOjhweDsiPicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTozMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L2Rpdj4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxNHB4O2NvbG9yOicrdmVsQ29sKyc7Zm9udC13ZWlnaHQ6NTAwIj4nK3ZlbERpcisnPC9kaXY+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXN0eWxlOml0YWxpYztsaW5lLWhlaWdodDoxLjUiPicrdmVsRGVzY1t2ZWxEaXJdKyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTEuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS42O21hcmdpbi10b3A6MTBweDtwYWRkaW5nLXRvcDoxMHB4O2JvcmRlci10b3A6MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiPicrY3R4Kyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VmVsb2NpdHk8L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiIHN0eWxlPSJmb250LXNpemU6MThweDtjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIM60PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJysoZC5kZWx0YT49MD8ndXAnOidkbicpKyciPicrKGQuZGVsdGE+PTA/JysnOicnKSsoZC5kZWx0YXx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKGQuYXR0ZW50aW9ufHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgIChyaXNpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BY2NlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIHJpc2luZ05hcnMubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6N3B4IDEwcHg7bWFyZ2luLWJvdHRvbTo0cHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiNlMDVhMjgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKSsKICAgICAgKGZhbGxpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5EZWNlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIGZhbGxpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMDUpO2JvcmRlcjoxcHggc29saWQgcmdiYSg1OSwxODQsMjE2LDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiMzYmI4ZDgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKTsKICB9CgogIHBhbmVsLmlubmVySFRNTD1oZWFkZXIrYm9keTsKfQoKCmZ1bmN0aW9uIHRvZ2dsZUZhdihubSl7CiAgaWYoRkFWUy5oYXMobm0pKSBGQVZTLmRlbGV0ZShubSk7ZWxzZSBGQVZTLmFkZChubSk7CiAgcmVuZGVyUGFuZWwoU0VMKTtyZW5kZXJGYXZzKCk7Cn0KZnVuY3Rpb24gcmVuZGVyRmF2cygpewogIHZhciByb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Zhdi1yb3cnKTsKICBpZighRkFWUy5zaXplKXtyb3cuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4nO3JldHVybjt9CiAgcm93LmlubmVySFRNTD1BcnJheS5mcm9tKEZBVlMpLm1hcChmdW5jdGlvbihubSl7CiAgICB2YXIgZD1nKG5tKSxkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPycjZTA1YTI4JzonIzNiYjhkOCc7CiAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJmYXYtY2FyZCIgb25jbGljaz0ic2VsZWN0XyhcJycrbm0rJ1wnKSI+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLWhlYWQiPjxzcGFuIGNsYXNzPSJmYy1uYW1lIj4nK25tKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImZjLXNjIj4nK2QuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+TmFycmF0aXZlPC9zcGFuPjxzcGFuIGNsYXNzPSJ2Ij4nK3RvcCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPjI0aDwvc3Bhbj48c3BhbiBjbGFzcz0idiIgc3R5bGU9ImNvbG9yOicrZEMrJyI+JytkUytkLmRlbHRhKyc8L3NwYW4+PC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9Cgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oYyl7CiAgYy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbih4KXt4LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgYy5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtsYXllcj1jLmRhdGFzZXQubGF5ZXI7YXBwbHlMYXllcigpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHVwZGF0ZUNsb2NrKCl7CiAgdmFyIG5vdz1uZXcgRGF0ZSgpLGlzdD1uZXcgRGF0ZShub3cuZ2V0VGltZSgpK25vdy5nZXRUaW1lem9uZU9mZnNldCgpKjYwMDAwKzE5ODAwMDAwKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2xvY2snKS50ZXh0Q29udGVudD1TdHJpbmcoaXN0LmdldEhvdXJzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRNaW51dGVzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRTZWNvbmRzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnIElTVCc7Cn0Kc2V0SW50ZXJ2YWwodXBkYXRlQ2xvY2ssMTAwMCk7dXBkYXRlQ2xvY2soKTsKCmZ1bmN0aW9uIGJ1aWxkV0lSU2lnbmFscygpewogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogIHZhciBzcmM9T2JqZWN0LmtleXMoTElWRSkubGVuZ3RoP0xJVkU6U0Q7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoc3JjKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5hdHRlbnRpb258fDApPjM7fSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CiAgZW50cmllcy5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pOwoKICB2YXIgdXNlZE5hcnJhdGl2ZXM9W10sIHVzZWRTdGF0ZXM9W107CiAgdmFyIHNpZ25hbHM9W107CgogIGZ1bmN0aW9uIHVzZWQobmFyLHN0YXRlKXtyZXR1cm4gdXNlZE5hcnJhdGl2ZXMuaW5kZXhPZihuYXIpPj0wfHx1c2VkU3RhdGVzLmluZGV4T2Yoc3RhdGUpPj0wO30KICBmdW5jdGlvbiB1c2UobmFyLHN0YXRlKXtpZihuYXIpdXNlZE5hcnJhdGl2ZXMucHVzaChuYXIpO2lmKHN0YXRlKXVzZWRTdGF0ZXMucHVzaChzdGF0ZSk7fQoKICAvLyAxLiBEb21pbmFudCBlbWVyZ2luZyBzaWduYWwKICB2YXIgdG9wPWVudHJpZXNbMF07CiAgaWYodG9wKXsKICAgIHZhciBuYXI9dG9wWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J3JlZ2lvbmFsIGFjdGl2aXR5JzsKICAgIHZhciBlbW89dG9wWzFdLmRvbWluYW50X2Vtb3Rpb247CiAgICB2YXIgY29sPWVtbz9wYWxbZW1vXTondmFyKC0tYWNjZW50KSc7CiAgICB2YXIgdmVsPXRvcFsxXS52ZWxvY2l0eXx8MDsKICAgIHZhciB2ZWxQaHJhc2U9dmVsPjAuMDg/JyDigJQgbW9tZW50dW0gYWNjZWxlcmF0aW5nIGJleW9uZCByZWdpb25hbCBib3VuZGFyaWVzJzp2ZWw+MC4wMz8nIOKAlCBzaWduYWwgZ2FpbmluZyByZWdpb25hbCB0cmFjdGlvbic6dmVsPC0wLjAzPycg4oCUIGF0dGVudGlvbiBiZWdpbm5pbmcgdG8gY29vbCc6JyDigJQgaG9sZGluZyBzdGVhZHkgaW4gcmVnaW9uYWwgZGlzY291cnNlJzsKICAgIHZhciBlbW9QaHJhc2U9ewogICAgICBhbmdlcjonIHdpdGggb3V0cmFnZSBhcyB0aGUgZG9taW5hbnQgcmVnaXN0ZXInLAogICAgICBhbnhpZXR5Oicgd2l0aCBhbnhpZXR5IGFzIHRoZSBwcmltYXJ5IHNpZ25hbCB0b25lJywKICAgICAgZmVhcjonIHdpdGggZmVhciBzaWduYWxzIGNvbmNlbnRyYXRpbmcgaW4gdGhlIHJlZ2lvbicsCiAgICAgIGhvcGU6JyB3aXRoIGFuIHVudXN1YWwgdW5kZXJjdXJyZW50IG9mIG9wdGltaXNtJywKICAgICAgcHJpZGU6JyB3aXRoIHN0cm9uZyByZWdpb25hbCBpZGVudGl0eSBzaWduYWxzJwogICAgfTsKICAgIHZhciB0ZXh0PSdMb2NhbGl6ZWQgYXR0ZW50aW9uIGFyb3VuZCA8ZW0+JytuYXIrJzwvZW0+IGNvbmNlbnRyYXRpbmcgaW4gPHN0cm9uZz4nK3RvcFswXSsnPC9zdHJvbmc+Jyt2ZWxQaHJhc2UrKGVtbz9lbW9QaHJhc2VbZW1vXXx8Jyc6JycpOwogICAgc2lnbmFscy5wdXNoKHtjb2w6Y29sLHRhZzoncHJpbWFyeSBzaWduYWwnLGxvYzp0b3BbMF0sdGV4dDp0ZXh0LGRlbGF5OjB9KTsKICAgIHVzZShuYXIsdG9wWzBdKTsKICB9CgogIC8vIDIuIFVuZGVyLW9ic2VydmVkIGVtZXJnaW5nIG1vdmVtZW50CiAgdmFyIGVtZXJnaW5nPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybiAoa3ZbMV0udmVsb2NpdHl8fDApPjAuMDQKICAgICAgJiYgKGt2WzFdLmF0dGVudGlvbnx8MCk8NDAKICAgICAgJiYgIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKTsKICB9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0udmVsb2NpdHl8fDApLShhWzFdLnZlbG9jaXR5fHwwKTt9KVswXTsKICBpZihlbWVyZ2luZyl7CiAgICB2YXIgZU5hcj1lbWVyZ2luZ1sxXS5kb21pbmFudF9uYXJyYXRpdmV8fCdyZWdpb25hbCBkaXNjb3Vyc2UnOwogICAgdmFyIGVFbW89ZW1lcmdpbmdbMV0uZG9taW5hbnRfZW1vdGlvbjsKICAgIHZhciBlQ29sPWVFbW8/cGFsW2VFbW9dOicjZTA3ODIwJzsKICAgIHZhciB0ZXh0Mj0nUmVnaW9uYWwgZGlzY291cnNlIGFyb3VuZCA8ZW0+JytlTmFyKyc8L2VtPiBzaG93aW5nIGVhcmx5IGFjY2VsZXJhdGlvbiBpbiA8c3Ryb25nPicrZW1lcmdpbmdbMF0rJzwvc3Ryb25nPiDigJQgbm90IHlldCB3aWRlbHkgYW1wbGlmaWVkIGluIG5hdGlvbmFsIGNvdmVyYWdlJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOmVDb2wsdGFnOidlYXJseSBtb3ZlbWVudCcsbG9jOmVtZXJnaW5nWzBdLHRleHQ6dGV4dDIsZGVsYXk6MTYwfSk7CiAgICB1c2UoZU5hcixlbWVyZ2luZ1swXSk7CiAgfQoKICAvLyAzLiBFbW90aW9uYWwgY29uY2VudHJhdGlvbiDigJQgbm90IGFuZ2VyL291dHJhZ2UsIGJ1dCBkZWVwZXIgc2lnbmFsCiAgdmFyIGVtb3Rpb25hbEZvY3VzPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBlPWt2WzFdLmRvbWluYW50X2Vtb3Rpb247CiAgICByZXR1cm4gZSYmZSE9PSdwcmlkZScmJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSkmJihrdlsxXS5hdHRlbnRpb258fDApPjQ7CiAgfSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZihlbW90aW9uYWxGb2N1cyl7CiAgICB2YXIgZWZOYXI9ZW1vdGlvbmFsRm9jdXNbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnbG9jYWwgZGV2ZWxvcG1lbnRzJzsKICAgIHZhciBlZkVtbz1lbW90aW9uYWxGb2N1c1sxXS5kb21pbmFudF9lbW90aW9uOwogICAgdmFyIGVmQ29sPXBhbFtlZkVtb118fCcjNTU2Njc3JzsKICAgIHZhciBlZlBocmFzZXM9ewogICAgICBhbmdlcjonR3JvdW5kLWxldmVsIHNpZ25hbHMgaW4gPHN0cm9uZz4nK2Vtb3Rpb25hbEZvY3VzWzBdKyc8L3N0cm9uZz4gaW5kaWNhdGUgPGVtPicrZWZOYXIrJzwvZW0+IGdlbmVyYXRpbmcgbG9jYWxpemVkIHByZXNzdXJlIOKAlCBxdWlldGx5IGludGVuc2lmeWluZyBiZWxvdyBuYXRpb25hbCBhbXBsaWZpY2F0aW9uJywKICAgICAgYW54aWV0eTonPGVtPicrZWZOYXIuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrZWZOYXIuc2xpY2UoMSkrJzwvZW0+IHNpZ25hbHMgY3JlYXRpbmcgZGlmZnVzZSBhbnhpZXR5IGFjcm9zcyA8c3Ryb25nPicrZW1vdGlvbmFsRm9jdXNbMF0rJzwvc3Ryb25nPiDigJQgcmVnaW9uYWxseSBjb25jZW50cmF0ZWQsIG5vdCB5ZXQgbmF0aW9uYWxseSB2aXNpYmxlJywKICAgICAgZmVhcjonVW5kZXItb2JzZXJ2ZWQgZmVhciBzaWduYWxzIGFyb3VuZCA8ZW0+JytlZk5hcisnPC9lbT4gZW1lcmdpbmcgaW4gPHN0cm9uZz4nK2Vtb3Rpb25hbEZvY3VzWzBdKyc8L3N0cm9uZz4g4oCUIGxvY2FsaXplZCBidXQgc2hvd2luZyBlYXJseSBtb21lbnR1bScsCiAgICAgIGhvcGU6J0FuIGVtZXJnaW5nIG9wdGltaXNtIHNpZ25hbCBhcm91bmQgPGVtPicrZWZOYXIrJzwvZW0+IGZvcm1pbmcgaW4gPHN0cm9uZz4nK2Vtb3Rpb25hbEZvY3VzWzBdKyc8L3N0cm9uZz4g4oCUIHJlZ2lvbmFsIGluIG9yaWdpbiwgbm90IHlldCBpbiBuYXRpb25hbCBkaXNjb3Vyc2UnLAogICAgfTsKICAgIHZhciB0ZXh0Mz1lZlBocmFzZXNbZWZFbW9dfHwnU2lnbmFsIGNvbmNlbnRyYXRpb24gYXJvdW5kIDxlbT4nK2VmTmFyKyc8L2VtPiBpbiA8c3Ryb25nPicrZW1vdGlvbmFsRm9jdXNbMF0rJzwvc3Ryb25nPiDigJQgbW9uaXRvcmluZyBmb3IgbmF0aW9uYWwgYW1wbGlmaWNhdGlvbic7CiAgICBzaWduYWxzLnB1c2goe2NvbDplZkNvbCx0YWc6J2Vtb3Rpb25hbCBzaWduYWwnLGxvYzplbW90aW9uYWxGb2N1c1swXSx0ZXh0OnRleHQzLGRlbGF5OjMyMH0pOwogICAgdXNlKGVmTmFyLGVtb3Rpb25hbEZvY3VzWzBdKTsKICB9CgogIC8vIDQuIENvb2xpbmcg4oCUIHdoYXQgd2FzIGFtcGxpZmllZCBpcyBub3cgcmV0cmVhdGluZwogIHZhciBjb29saW5nPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybihrdlsxXS52ZWxvY2l0eXx8MCk8LTAuMDQmJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSkmJihrdlsxXS5hdHRlbnRpb258fDApPjU7CiAgfSkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihhWzFdLnZlbG9jaXR5fHwwKS0oYlsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYoY29vbGluZyl7CiAgICB2YXIgY05hcj1jb29saW5nWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J3ByZXZpb3VzIGN5Y2xlJzsKICAgIHZhciB0ZXh0ND0nQXR0ZW50aW9uIGFyb3VuZCA8ZW0+JytjTmFyKyc8L2VtPiBpbiA8c3Ryb25nPicrY29vbGluZ1swXSsnPC9zdHJvbmc+IGRlY2VsZXJhdGluZyDigJQgc2lnbmFscyBzdWdnZXN0IHRoZSByZWdpb25hbCBjeWNsZSBpcyBjb21wbGV0aW5nLCBhdHRlbnRpb24gbGlrZWx5IHRvIHNoaWZ0JzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOicjM2JiOGQ4Jyx0YWc6J3NpZ25hbCBjb29saW5nJyxsb2M6Y29vbGluZ1swXSx0ZXh0OnRleHQ0LGRlbGF5OjQ2MH0pOwogICAgdXNlKGNOYXIsY29vbGluZ1swXSk7CiAgfQoKICAvLyA1LiBOb3J0aGVhc3QgZGl2ZXJnZW5jZSDigJQgdW5kZXItb2JzZXJ2ZWQgcmVnaW9uYWwgcmVhbGl0eQogIHZhciBuZVN0YXRlcz1bJ01hbmlwdXInLCdBc3NhbScsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ0FydW5hY2hhbCBQcmFkZXNoJywnVHJpcHVyYSddOwogIHZhciBuZUFjdGl2ZT1uZVN0YXRlcy5maWx0ZXIoZnVuY3Rpb24ocyl7CiAgICByZXR1cm4gc3JjW3NdJiYoc3JjW3NdLmF0dGVudGlvbnx8MCk+MiYmdXNlZFN0YXRlcy5pbmRleE9mKHMpPDA7CiAgfSk7CiAgaWYobmVBY3RpdmUubGVuZ3RoPj0yKXsKICAgIHZhciBuZU5hcj0oc3JjW25lQWN0aXZlWzBdXSYmc3JjW25lQWN0aXZlWzBdXS5kb21pbmFudF9uYXJyYXRpdmUpfHwncmVnaW9uYWwgZGV2ZWxvcG1lbnRzJzsKICAgIHZhciB0ZXh0NT0nUmVnaW9uYWwgZGlzY291cnNlIGFyb3VuZCA8ZW0+JytuZU5hcisnPC9lbT4gY29uY2VudHJhdGluZyBhY3Jvc3MgJytuZUFjdGl2ZS5sZW5ndGgrJyBub3J0aGVhc3Rlcm4gc3RhdGVzIOKAlCB1bmRlci1vYnNlcnZlZCBpbiBuYXRpb25hbCBtZWRpYSBjeWNsZXMsIHNob3dpbmcgbG9jYWxpemVkIHNpZ25hbCBkZW5zaXR5JzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOidyZ2JhKDE2MCwxOTAsMjMwLDAuNDUpJyx0YWc6J3VuZGVyLW9ic2VydmVkIHJlZ2lvbicsbG9jOidOb3J0aGVhc3QnLHRleHQ6dGV4dDUsZGVsYXk6NTgwfSk7CiAgfQoKICBpZighc2lnbmFscy5sZW5ndGgpIHJldHVybjsKICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3dpci1zaWduYWxzJyk7CiAgaWYoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MPXNpZ25hbHMubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsIiBzdHlsZT0iYW5pbWF0aW9uLWRlbGF5Oicrcy5kZWxheSsnbXMiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLWJhciIgc3R5bGU9ImJhY2tncm91bmQ6JytzLmNvbCsnIj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1jb250ZW50Ij4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLXRleHQiPicrcy50ZXh0Kyc8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ3aXItc2lnbmFsLW1ldGEiPicrCiAgICAgICAgICAnPHNwYW4gY2xhc3M9Indpci1zaWduYWwtdGFnIiBzdHlsZT0iY29sb3I6JytzLmNvbCsnIj4nK3MudGFnKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBjbGFzcz0id2lyLXNpZ25hbC1sb2MiPicrcy5sb2MrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9CgoKLy8gSU5JVCDigJQgd2FpdCBmb3IgRE9NCi8vIGkgYnV0dG9uIHRvb2x0aXAg4oCUIHVzZXMgZml4ZWQgcG9zaXRpb25pbmcgc28gaXQncyBuZXZlciBjbGlwcGVkCihmdW5jdGlvbigpewogIHZhciB0aXA9bnVsbDsKICBmdW5jdGlvbiBzaG93VGlwKGUpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICB2YXIgdHh0PXRoaXMuZ2V0QXR0cmlidXRlKCdkYXRhLXRpcCcpOwogICAgaWYoIXR4dHx8IXRpcCkgcmV0dXJuOwogICAgdGlwLnRleHRDb250ZW50PXR4dDsKICAgIHRpcC5jbGFzc0xpc3QuYWRkKCd2aXNpYmxlJyk7CiAgICB2YXIgcmVjdD10aGlzLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpOwogICAgdmFyIHR3PTI0MDsKICAgIHZhciBsZWZ0PU1hdGgubWluKHJlY3QubGVmdCx3aW5kb3cuaW5uZXJXaWR0aC10dy0xMCk7CiAgICB0aXAuc3R5bGUubGVmdD1sZWZ0KydweCc7CiAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC0xMC10aXAub2Zmc2V0SGVpZ2h0fHxyZWN0LnRvcC04MCkrJ3B4JzsKICAgIC8vIFJlcG9zaXRpb24gYWZ0ZXIgcmVuZGVyCiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZnVuY3Rpb24oKXsKICAgICAgdGlwLnN0eWxlLnRvcD0ocmVjdC50b3AtdGlwLm9mZnNldEhlaWdodC04KSsncHgnOwogICAgfSk7CiAgfQogIGZ1bmN0aW9uIGhpZGVUaXAoKXsKICAgIGlmKCF0aXApe3RpcD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbHRhYi10b29sdGlwJyk7fQogICAgaWYodGlwKSB0aXAuY2xhc3NMaXN0LnJlbW92ZSgndmlzaWJsZScpOwogIH0KICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW92ZXInLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgc2hvd1RpcC5jYWxsKGUudGFyZ2V0LGUpOwogIH0pOwogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3V0JyxmdW5jdGlvbihlKXsKICAgIGlmKGUudGFyZ2V0LmNsYXNzTGlzdC5jb250YWlucygnbHRhYi1pbmZvJykpIGhpZGVUaXAoKTsKICB9KTsKfSkoKTsKCmZ1bmN0aW9uIGluaXQoKXsKICByZW5kZXJTdHJpcCgnM20nKTsKCiAgLy8gTG9hZCBtYXAgd2l0aCByZXRyeQogIHZhciBtYXBBdHRlbXB0cz0wOwogIGZ1bmN0aW9uIHRyeUxvYWRNYXAoKXsKICAgIGlmKHR5cGVvZiB0b3BvanNvbj09PSd1bmRlZmluZWQnKXsKICAgICAgaWYobWFwQXR0ZW1wdHMrKzwxMCl7c2V0VGltZW91dCh0cnlMb2FkTWFwLDMwMCk7fQogICAgICByZXR1cm47CiAgICB9CiAgICBsb2FkTWFwKCk7CiAgfQogIHRyeUxvYWRNYXAoKTsKCiAgLy8gTG9hZCBmdWxsIGNhY2hlZCBzbmFwc2hvdCBpbW1lZGlhdGVseSBmb3IgaW5zdGFudCBkYXRhCiAgZmV0Y2hGdWxsU25hcHNob3QoKS50aGVuKGZ1bmN0aW9uKG9rKXsKICAgIGlmKG9rKXsKICAgICAgcmVuZGVyTW9tZW50dW0oKTsKICAgICAgc2V0VGltZW91dChmdW5jdGlvbigpe3N0YXJ0UG9sbGluZygpO30sMTAwMCk7CiAgICB9IGVsc2UgewogICAgICBzdGFydFBvbGxpbmcoKTsKICAgIH0KICB9KTsKCiAgLy8gUmV0cnkgbWFwIGlmIHN0aWxsIGVtcHR5CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2lmKCFkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5sZW5ndGgpbG9hZE1hcCgpO30sMzAwMCk7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2lmKCFkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5sZW5ndGgpbG9hZE1hcCgpO30sNjAwMCk7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2ZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pO30sNTAwMCk7CiAgc2V0VGltZW91dChmdW5jdGlvbigpe2ZldGNoTmFycmF0aXZlSW5zaWdodCgpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw4MDAwKTsKfQppZihkb2N1bWVudC5yZWFkeVN0YXRlPT09J2xvYWRpbmcnKXsKICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgaW5pdCk7Cn0gZWxzZSB7CiAgLy8gQWxyZWFkeSBsb2FkZWQg4oCUIGJ1dCB3YWl0IG9uZSB0aWNrIHRvIGVuc3VyZSBhbGwgc2NyaXB0cyBwYXJzZWQKICBzZXRUaW1lb3V0KGluaXQsIDApOwp9CgoKc2V0VGltZW91dChmdW5jdGlvbigpewogIC8vIEF1dG8tc2VsZWN0IGhvdHRlc3Qgc3RhdGUgZnJvbSBMSVZFIGRhdGEKICB2YXIgc3JjPU9iamVjdC5rZXlzKExJVkUpLmxlbmd0aD9MSVZFOlNEOwogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoc3JjKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApewogICAgdmFyIGVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9IicrdG9wWzBdKyciXScpOwogICAgaWYoZWwpIHNlbGVjdF8odG9wWzBdKTsKICB9Cn0sMzAwMCk7CnNldFRpbWVvdXQocmVuZGVyRmF2cywyNDAwKTsKPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPgo="

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
