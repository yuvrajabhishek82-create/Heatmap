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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQoubGl2ZS1pbmRpY2F0b3J7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoubGl2ZS1kb3R7d2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojNGFkZTgwO2JveC1zaGFkb3c6MCAwIDhweCByZ2JhKDc0LDIyMiwxMjgsMC43KTthbmltYXRpb246bGQgMi41cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBsZHswJSwxMDAle29wYWNpdHk6MTt0cmFuc2Zvcm06c2NhbGUoMSl9NTAle29wYWNpdHk6MC4zNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19Ci5jbG9ja3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbX0KCi8qIEhFUk8gKi8KLmhlcm97CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIHBhZGRpbmc6NzJweCAzNnB4IDA7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvOwp9Ci5oZXJvLWV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjMyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjI0cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLmhlcm8tZXllYnJvdzo6YmVmb3Jle2NvbnRlbnQ6Jyc7d2lkdGg6MTZweDtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tZmFpbnQpO29wYWNpdHk6MC41fQouaGVyby1icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC1zaXplOmNsYW1wKDM2cHgsNC4ydncsNjRweCk7bGluZS1oZWlnaHQ6MTtsZXR0ZXItc3BhY2luZzotMC4wM2VtO2NvbG9yOnZhcigtLWluayk7bWFyZ2luOjB9Ci5oZXJvLWJyYW5kLW5hbWUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VOYW1lR2xvdyA1cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBwdWxzZU5hbWVHbG93ezAlLDEwMCV7b3BhY2l0eToxfTUwJXtvcGFjaXR5OjAuNzJ9fQouaGVyby10YWdsaW5le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjV2dywyMHB4KTtmb250LXdlaWdodDozMDA7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQ7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTttYXJnaW46MCAwIDEycHggMDttYXgtd2lkdGg6NDgwcHg7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1kZXNje2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS42O21heC13aWR0aDo0MDBweDttYXJnaW46MCAwIDZweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tc3ViLWxpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjYpO21hcmdpbjowIDAgMjBweCAwO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmhlcm8tcHVsc2Utc2lnbmFse3Bvc2l0aW9uOnJlbGF0aXZlO3dpZHRoOjE2cHg7aGVpZ2h0OjE2cHg7ZmxleC1zaHJpbms6MH0KLmhwcy1jb3Jle3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjk7YW5pbWF0aW9uOmhwc0NvcmUgNHMgZWFzZS1pbi1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgaHBzQ29yZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ7dHJhbnNmb3JtOnNjYWxlKDAuNzUpfX0KLmhwcy1yaW5ne3Bvc2l0aW9uOmFic29sdXRlO2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYWNjZW50KTthbmltYXRpb246aHBzUmluZyA0cyBlYXNlLW91dCBpbmZpbml0ZX0KLmhwcy1yaW5nLnIxe2luc2V0OjFweDthbmltYXRpb24tZGVsYXk6MHN9Lmhwcy1yaW5nLnIye2luc2V0Oi0zcHg7YW5pbWF0aW9uLWRlbGF5OjEuNHM7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMzUpfQpAa2V5ZnJhbWVzIGhwc1Jpbmd7MCV7b3BhY2l0eTowLjY7dHJhbnNmb3JtOnNjYWxlKDAuNyl9MTAwJXtvcGFjaXR5OjA7dHJhbnNmb3JtOnNjYWxlKDEuNil9fQoKLyogU0lHTkFUVVJFIElOU0lHSFQgKi8KLnNpZ25hdHVyZS1pbnNpZ2h0ewogIG1hcmdpbi10b3A6MDsKICBwYWRkaW5nOjE0cHggMjBweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MTRweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsIHJnYmEoMjI0LDkwLDQwLDAuMDYpIDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMTAwJSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoOHB4KTsKICBtYXgtd2lkdGg6OTAwcHg7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwp9Ci5zaWduYXR1cmUtaW5zaWdodDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sIHZhcigtLWFjY2VudCksIHRyYW5zcGFyZW50KTsKfQouc2ktbGFiZWx7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMjVlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjEwcHg7Cn0KLnNpLXRleHR7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNHB4LDEuNHZ3LDE4cHgpOwogIGZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuNTtsZXR0ZXItc3BhY2luZzotMC4wMWVtOwp9Ci5zaS10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWFjY2VudCl9Ci5zaS1zdWJ7CiAgbWFyZ2luLXRvcDoxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4wNGVtO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE0cHg7ZmxleC13cmFwOndyYXA7Cn0KLnNpLXRhZ3sKICBwYWRkaW5nOjJweCA4cHg7Ym9yZGVyLXJhZGl1czozcHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7CiAgZm9udC1zaXplOjkuNXB4Owp9CgovKiBOQVJSQVRJVkUgU1RSSVAgKi8KCi5zdHJpcC10YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo0cHggOXB4O2JvcmRlci1yYWRpdXM6M3B4O2N1cnNvcjpwb2ludGVyOwogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOm5vbmU7dHJhbnNpdGlvbjphbGwgMC4xNXM7Cn0KLnN0cmlwLXRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMTIpfQouc3RyaXAtdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9Ci5zdHJpcC1jb2x7CiAgZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tYmcxKTtwYWRkaW5nOjA7Cn0KLnN0cmlwLWNvbC1oZWFkewogIHBhZGRpbmc6MTBweCAxNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKfQouc3RyaXAtY29sLWhlYWQuZmFkZXtjb2xvcjp2YXIoLS1mYWxsKX0KLnN0cmlwLWNvbC1oZWFkLnJpc2Uye2NvbG9yOnZhcigtLXJpc2UpfQouc3RyaXAtY29sLWhlYWQuc2hpZnR7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbC1ib2R5e3BhZGRpbmc6MTJweCAxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweH0KLnN0cmlwLWl0ZW17CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDo4cHg7Cn0KLnN0cmlwLXRvcGlje2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwfQouc3RyaXAtbm90ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouc3RyaXAtYXJye2NvbG9yOnZhcigtLWFjY2VudCk7b3BhY2l0eTowLjU7Zm9udC1zaXplOjE0cHg7ZmxleC1zaHJpbms6MH0KCi8qIE1BSU4gTEFZT1VUICovCi5tYWluewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAzNjBweDsKICBnYXA6MTRweDsKICBtaW4td2lkdGg6MDsKfQoKLyogTUFQICovCi5tYXAtY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5tYXAtY2FyZDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNzAlIDUwJSBhdCAzNSUgMCUsIHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCA4MCUgMTAwJSwgcmdiYSg1OSwxODQsMjE2LDAuMDMpIDAlLCB0cmFuc3BhcmVudCA2MCUpOwp9Ci5tYXAtdG9wewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTJweCAxOHB4IDA7Cn0KLm1hcC10aXRsZS1ibG9jayAubXR7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxN3B4O2ZvbnQtd2VpZ2h0OjQwMDtsZXR0ZXItc3BhY2luZzotMC4wMWVtfQoubWFwLXRpdGxlLWJsb2NrIC5tc3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA2ZW07bWFyZ2luLXRvcDoycHh9Ci5sZWdlbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1kaW0pfQoubGVnZW5kLWJhcnsKICBoZWlnaHQ6M3B4O3dpZHRoOjgwcHg7Ym9yZGVyLXJhZGl1czoycHg7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQsIzBlMjAzNSwjMWE1NTgwIDI1JSwjOGE1YzE4IDU1JSwjYzAzODFhIDgwJSwjZTAxMDIwKTsKfQoubGF5ZXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7CiAgcGFkZGluZzoxMHB4IDIwcHggNnB4Owp9Ci5sYXllci1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5sdGFic3tkaXNwbGF5OmZsZXg7Z2FwOjNweH0KLmx0YWJ7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjNweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQoubHRhYi5hY3RpdmV7Y29sb3I6dmFyKC0taW5rKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDgpO2JvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpfQoubHRhYntkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NXB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OnZpc2libGV9Ci5sdGFiLWluZm97d2lkdGg6MTNweDtoZWlnaHQ6MTNweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4yKTtmb250LXNpemU6OHB4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc3R5bGU6aXRhbGljO2ZvbnQtd2VpZ2h0OjYwMDtjb2xvcjpyZ2JhKDE2MCwxOTAsMjMwLDAuMzUpO2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y3Vyc29yOmhlbHA7ZmxleC1zaHJpbms6MDt0cmFuc2l0aW9uOmFsbCAwLjE1cztwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEwMH0KLmx0YWItaW5mbzpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjZW50KTtjb2xvcjp2YXIoLS1hY2NlbnQpfQojbHRhYi10b29sdGlwe3Bvc2l0aW9uOmZpeGVkO2JhY2tncm91bmQ6cmdiYSg4LDEyLDIwLDAuOTgpO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjEyKTtib3JkZXItcmFkaXVzOjhweDtwYWRkaW5nOjEwcHggMTNweDtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTFweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjY7d2lkdGg6MjMwcHg7d2hpdGUtc3BhY2U6bm9ybWFsO3RleHQtYWxpZ246bGVmdDtib3gtc2hhZG93OjAgOHB4IDMycHggcmdiYSgwLDAsMCwwLjYpO3BvaW50ZXItZXZlbnRzOm5vbmU7b3BhY2l0eTowO3RyYW5zaXRpb246b3BhY2l0eSAwLjE1czt6LWluZGV4Ojk5OTk5O2Rpc3BsYXk6bm9uZX0KI2x0YWItdG9vbHRpcC52aXNpYmxle29wYWNpdHk6MTtkaXNwbGF5OmJsb2NrfQoubHRhYjpob3Zlcntjb2xvcjp2YXIoLS1kaW0pfQoKLm1hcC1zdmctd3JhcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nOjEycHggMTZweCAxNnB4Owp9Ci5tYXAtaW5uZXJ7cG9zaXRpb246cmVsYXRpdmU7YXNwZWN0LXJhdGlvOjEvMTt3aWR0aDoxMDAlfQojaW5kaWEtbWFwe3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7ZGlzcGxheTpibG9jaztvdmVyZmxvdzp2aXNpYmxlfQoKLyogbWFwIHN0YXRlIHN0eWxlcyAqLwojaW5kaWEtbWFwIC5zdGF0ZXsKICBjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOmZpbHRlciAwLjI1cyBlYXNlLCBzdHJva2Utd2lkdGggMC4ycyBlYXNlLCBzdHJva2UgMC4ycyBlYXNlOwp9CiNpbmRpYS1tYXAgLnN0YXRlOmhvdmVyewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuNykgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MXB4ICFpbXBvcnRhbnQ7CiAgZmlsdGVyOmJyaWdodG5lc3MoMS4yNSkgZHJvcC1zaGFkb3coMCAwIDEwcHggcmdiYSgyNTUsMjU1LDI1NSwwLjIpKTsKfQojaW5kaWEtbWFwIC5zdGF0ZS5zZWxlY3RlZHsKICBzdHJva2U6cmdiYSgyNTUsMjU1LDI1NSwwLjkpICFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjEuNHB4ICFpbXBvcnRhbnQ7CiAgZmlsdGVyOmJyaWdodG5lc3MoMS4zNSkgZHJvcC1zaGFkb3coMCAwIDE2cHggcmdiYSgyNTUsMjU1LDI1NSwwLjMpKTsKfQoKLyogYW5pbWF0ZWQgcHVsc2UgcmluZ3MgKi8KLnB1bHNlLXJpbmd7ZmlsbDpub25lO3BvaW50ZXItZXZlbnRzOm5vbmV9Ci5wdWxzZS1yaW5nLnAxe2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IGluZmluaXRlfQoucHVsc2UtcmluZy5wMnthbmltYXRpb246cHIgMi44cyBlYXNlLW91dCAwLjlzIGluZmluaXRlfQpAa2V5ZnJhbWVzIHByewogIDAle3I6NDtvcGFjaXR5OjAuNztzdHJva2Utd2lkdGg6MS4yfQogIDEwMCV7cjoyNjtvcGFjaXR5OjA7c3Ryb2tlLXdpZHRoOjAuMn0KfQoKLyogYXRtb3NwaGVyaWMgZ2xvdyBiZWhpbmQgaG90IHN0YXRlcyAqLwouc3RhdGUtZ2xvd3twb2ludGVyLWV2ZW50czpub25lO2ZpbGw6bm9uZX0KQGtleWZyYW1lcyBnbG93UHVsc2V7MCUsMTAwJXtvcGFjaXR5OjAuMTJ9NTAle29wYWNpdHk6MC4yMn19CgoubWFwLXRvb2x0aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7cG9pbnRlci1ldmVudHM6bm9uZTsKICBiYWNrZ3JvdW5kOnJnYmEoNSw3LDEyLDAuOTUpO2JhY2tkcm9wLWZpbHRlcjpibHVyKDEycHgpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czo5cHg7CiAgcGFkZGluZzoxMnB4IDE0cHg7b3BhY2l0eTowO3RyYW5zaXRpb246b3BhY2l0eSAwLjEyczt6LWluZGV4Ojk5OTk7bWluLXdpZHRoOjE3MHB4Owp9Ci50dC1ue2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bWFyZ2luLWJvdHRvbTo4cHg7Y29sb3I6dmFyKC0taW5rKX0KLnR0LXJ7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NHB4fQoudHQtciBzdHJvbmd7Y29sb3I6dmFyKC0taW5rKX0KLnR0LW5hcnsKICBtYXJnaW4tdG9wOjhweDtwYWRkaW5nLXRvcDo4cHg7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTsKfQoudHQtbmFyIHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2Rpc3BsYXk6YmxvY2s7bWFyZ2luLWJvdHRvbToycHh9CgovKiBTVEFURSBQQU5FTCAqLwouc3RhdGUtcGFuZWx7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNnB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE2cHgpOwogIHBhZGRpbmc6MjBweDtvdmVyZmxvdy15OmF1dG87bWF4LWhlaWdodDo3ODBweDsKICBtaW4td2lkdGg6MDtvdmVyZmxvdy14OmhpZGRlbjsKfQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjNweH0KLnN0YXRlLXBhbmVsOjotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MnB4fQoKLnBhbmVsLWVtcHR5ewogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7CiAgaGVpZ2h0OjEwMCU7bWluLWhlaWdodDozMjBweDt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjMycHggMjBweDsKfQoucGFuZWwtZW1wdHkgc3Zne29wYWNpdHk6MC4xNTttYXJnaW4tYm90dG9tOjE4cHh9Ci5wYW5lbC1lbXB0eSAucGUtdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjhweH0KLnBhbmVsLWVtcHR5IC5wZS1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtsZXR0ZXItc3BhY2luZzowLjA0ZW07bGluZS1oZWlnaHQ6MS43fQoKLyogc3RhdGUgcGFuZWwgaW50ZXJuYWxzICovCi5zcC1oZWFkewogIGRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIG1hcmdpbi1ib3R0b206MTZweDtwYWRkaW5nLWJvdHRvbToxNHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Cn0KLnNwLWVre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTtjb2xvcjp2YXIoLS1mYWludCk7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1ib3R0b206NXB4fQouc3AtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI4cHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07bGluZS1oZWlnaHQ6MTtjb2xvcjp2YXIoLS1pbmspfQouZmF2LWJ0bnsKICBiYWNrZ3JvdW5kOnRyYW5zcGFyZW50O2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Y29sb3I6dmFyKC0tZmFpbnQpOwogIHdpZHRoOjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo2cHg7Y3Vyc29yOnBvaW50ZXI7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246YWxsIDAuMThzO3BhZGRpbmc6MDtmbGV4LXNocmluazowOwp9Ci5mYXYtYnRuOmhvdmVye2NvbG9yOnZhcigtLWRpbSk7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5mYXYtYnRuLm9ue2NvbG9yOnZhcigtLWFjY2VudCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA3KX0KLmZhdi1idG4gc3Zne3dpZHRoOjEzcHg7aGVpZ2h0OjEzcHh9CgovKiBuYXJyYXRpdmUgdGltZWxpbmUg4oCUIHRoZSBzaWduYXR1cmUgZmVhdHVyZSAqLwoubmFyLXRpbWVsaW5lewogIG1hcmdpbi1ib3R0b206MTZweDsKfQoubnQtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MTBweH0KLm50LWZsb3d7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtwYWRkaW5nLWxlZnQ6MTZweDsKfQoubnQtZmxvdzo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDo1cHg7dG9wOjZweDtib3R0b206NnB4O3dpZHRoOjFweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx2YXIoLS1ib3JkZXIpKTtvcGFjaXR5OjAuNDsKfQoubnQtc3RlcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTBweDsKICBwYWRkaW5nOjVweCAwO3Bvc2l0aW9uOnJlbGF0aXZlOwp9Ci5udC1kb3R7CiAgd2lkdGg6MTBweDtoZWlnaHQ6MTBweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4LXNocmluazowOwogIHBvc2l0aW9uOmFic29sdXRlO2xlZnQ6LTE2cHg7dG9wOjdweDsKICBib3JkZXI6MS41cHggc29saWQgY3VycmVudENvbG9yO2JhY2tncm91bmQ6dmFyKC0tYmcpOwp9Ci5udC1zdGVwLnBhc3QgLm50LWRvdHtjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCl9Ci5udC1zdGVwLmN1cnJlbnQgLm50LWRvdHtjb2xvcjp2YXIoLS1hY2NlbnQpO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCAxMHB4IHJnYmEoMjI0LDkwLDQwLDAuNSl9Ci5udC1jb250ZW50e2ZsZXg6MX0KLm50LXRvcGlje2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4zfQoubnQtc3RlcC5wYXN0IC5udC10b3BpY3tjb2xvcjp2YXIoLS1mYWludCl9Ci5udC1zdGVwLnJlY2VudCAubnQtdG9waWN7Y29sb3I6dmFyKC0tZGltKX0KLm50LXdoZW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHh9CgovKiBpbnNpZ2h0IGJsb2NrICovCi5pbnNpZ2h0ewogIG1hcmdpbi1ib3R0b206MTRweDsKICBwYWRkaW5nOjEycHggMTRweCAxMnB4IDE2cHg7CiAgYm9yZGVyLWxlZnQ6MS41cHggc29saWQgdmFyKC0tYWNjZW50KTsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDMpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxMy41cHg7Zm9udC1zdHlsZTppdGFsaWM7CiAgY29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU1O2ZvbnQtd2VpZ2h0OjMwMDsKfQoKLyogY29tcGFjdCBzY29yZSBzdHJpcCAqLwouc2NvcmUtc3RyaXB7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsKICBwYWRkaW5nOjhweCAxMnB4O2JvcmRlci1yYWRpdXM6N3B4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgbWFyZ2luLWJvdHRvbToxNHB4Owp9Ci5zcy1pdGVte2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjJweH0KLnNzLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCl9Ci5zcy12YWx7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyMnB4O2ZvbnQtd2VpZ2h0OjMwMDtsZXR0ZXItc3BhY2luZzotMC4wMmVtO2NvbG9yOnZhcigtLWluayl9Ci5zcy1kZWx0YXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjJweCA3cHg7Ym9yZGVyLXJhZGl1czozcHh9Ci5zcy1kZWx0YS51cHtjb2xvcjojZTA2MDMwO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xKX0KLnNzLWRlbHRhLmRue2NvbG9yOiMzYmI4ZDg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKX0KLnNzLWRpdmlkZXJ7d2lkdGg6MXB4O2hlaWdodDozMnB4O2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowfQouc3MtbmFye2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDB9Cgouc3Atc2VjdGlvbnttYXJnaW4tYm90dG9tOjE0cHh9Ci5zcC1zZWMtdGl0bGV7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206OXB4Owp9CgovKiBuYXJyYXRpdmVzICovCi5uYXItbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9Ci5uYXItaXRlbTJ7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgYXV0bztnYXA6NnB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLm5pLWxhYmVse2ZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKX0KLm5pLXZhbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5uaS10cmFja3tncmlkLWNvbHVtbjoxLy0xO2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi10b3A6LTNweH0KLm5pLWZpbGx7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoxcHg7dHJhbnNpdGlvbjp3aWR0aCAwLjdzfQoKLyogbW92ZW1lbnQgKi8KLm12LWdyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgMWZyO2dhcDo3cHh9Ci5tdi1ibG9ja3tiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6N3B4O3BhZGRpbmc6OXB4fQoubXYtaHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMTRlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206N3B4fQoubXYtYmxvY2sudXAgLm12LWh7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tdi1ibG9jay5kbiAubXYtaHtjb2xvcjp2YXIoLS1mYWxsKX0KLm12LWl0e2ZvbnQtc2l6ZToxMC41cHg7cGFkZGluZzo0cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2NvbG9yOnZhcigtLWZhaW50KX0KLm12LWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tdi1pdCBzdHJvbmd7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7ZGlzcGxheTpibG9jaztmb250LXNpemU6MTFweH0KLm12LWl0IHNwYW57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweH0KCi8qIGVtb3Rpb24gKi8KLmVtLXJvd3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQouZW0tZG9udXR7d2lkdGg6NzZweDtoZWlnaHQ6NzZweDtmbGV4LXNocmluazowfQouZW0tbGVne2ZsZXg6MTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHh9Ci5lbS1pdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjZweH0KLmVtLXN3e3dpZHRoOjZweDtoZWlnaHQ6NnB4O2JvcmRlci1yYWRpdXM6MnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1ue2ZsZXg6MTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5lbS1we2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1pbmspfQoKLyogdGltZWxpbmUgY2hhcnQgKi8KLnRsLXdyYXB7aGVpZ2h0OjcycHh9CgovKiBhcnRpY2xlcyAqLwouYXJ0LWxpc3R7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4fQouYXJ0LWl0ZW17CiAgZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzo3cHggOXB4O2JvcmRlci1yYWRpdXM6NnB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMSk7CiAgdHJhbnNpdGlvbjphbGwgMC4xMnM7Cn0KLmFydC1pdGVtOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXItY29sb3I6dmFyKC0tYm9yZGVyMil9Ci5hcnQtc3Jje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO3dpZHRoOjQ0cHg7cGFkZGluZy10b3A6MXB4fQouYXJ0LXR4dHtmb250LXNpemU6MTFweDtsaW5lLWhlaWdodDoxLjQ7Y29sb3I6dmFyKC0tZGltKX0KCi8qIE5BUlJBVElWRSBJTlRFTExJR0VOQ0UgUk9XICovCi5uYXItcm93ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87CiAgcGFkZGluZzowIDM2cHggMjhweDsKICBkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnIgMWZyO2dhcDoxOHB4Owp9Ci5uYXItY2FyZHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE0cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTRweCk7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLm5jLWhlYWR7CiAgcGFkZGluZzoxNnB4IDIwcHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTtmbGV4LXNocmluazowOwp9Ci5uYy1ib2R5e3BhZGRpbmc6OHB4IDIwcHggMTZweDtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO30KLm5jLXRpdGxle2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTZweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspfQoubmMtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtO21hcmdpbi10b3A6MnB4fQoubmMtYm9keXtwYWRkaW5nOjEzcHggMTZweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoKLm1vbS1pdHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7CiAgcGFkZGluZzo3cHggMDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwp9Ci5tb20taXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm1vbS1ya3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTt3aWR0aDoxM3B4O2ZsZXgtc2hyaW5rOjB9Ci5tb20taW5me2ZsZXg6MX0KLm1vbS1ubXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLm1vbS1zdHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4fQoubW9tLXBje2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NDAwO2ZsZXgtc2hyaW5rOjB9Ci5tb20tcGMucntjb2xvcjp2YXIoLS1yaXNlKX0KLm1vbS1wYy5me2NvbG9yOnZhcigtLWZhbGwpfQoubW9tLXRye2hlaWdodDoxLjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoxcHg7bWFyZ2luOjNweCAwIDA7b3ZlcmZsb3c6aGlkZGVufQoubW9tLWZse2hlaWdodDoxMDAlO2JvcmRlci1yYWRpdXM6MXB4fQoKLnJlZy1pdHsKICBkaXNwbGF5OmZsZXg7Z2FwOjlweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0OwogIHBhZGRpbmc6OHB4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtjdXJzb3I6cG9pbnRlcjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xNXM7Cn0KLnJlZy1pdDpmaXJzdC1vZi10eXBle2JvcmRlci10b3A6bm9uZTtwYWRkaW5nLXRvcDowfQoucmVnLWl0OmhvdmVye29wYWNpdHk6MC43NX0KLnJlZy1iYWRnZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMDdlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgcGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4OwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjE0KTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHg7d2hpdGUtc3BhY2U6bm93cmFwOwp9Ci5yZWctZmx7ZmxleDoxO2ZvbnQtc2l6ZToxMS41cHg7bGluZS1oZWlnaHQ6MS41fQoucmVnLWZyb217Y29sb3I6dmFyKC0tZmFpbnQpfQoucmVnLWFycntjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC41O21hcmdpbjowIDRweH0KLnJlZy10b3tjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMH0KLnJlZy10bXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7bWFyZ2luLXRvcDoycHh9CgovKiBGQVZTICovCi5mYXZzewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggNDBweDsKfQouZmF2cy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEwcHh9Ci5mYXZzLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjEwcHg7b3ZlcmZsb3cteDphdXRvO3BhZGRpbmctYm90dG9tOjNweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhcntoZWlnaHQ6MnB4fQouZmF2cy1yb3c6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxcHh9Ci5mYXYtY2FyZHsKICBmbGV4OjAgMCAxOTBweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4O2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMThzOwp9Ci5mYXYtY2FyZDpob3Zlcntib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4yMik7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjAyKX0KLmZjLWhlYWR7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206N3B4fQouZmMtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC13ZWlnaHQ6NDAwO2NvbG9yOnZhcigtLWluayl9Ci5mYy1zY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5mYy1yb3d7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjNweH0KLmZjLXJvdyAudntjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweH0KLmZhdnMtZW1wdHl7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljO3BhZGRpbmc6NHB4IDB9CgovKiBGT09UICovCi5mb290e3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6NDhweCAzNnB4IDYwcHg7bWF4LXdpZHRoOjU4MHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZm9vdC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xldHRlci1zcGFjaW5nOi0wLjAxZW07bWFyZ2luLWJvdHRvbToxNHB4fQouZm9vdC1saW5le2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1mYWludCk7bGluZS1oZWlnaHQ6MS44O21hcmdpbi1ib3R0b206MTJweH0KLmZvb3Qtc3Vie2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSg2Miw3Nyw5NiwwLjUpfQoKLyogYW5pbWF0aW9ucyAqLwpAa2V5ZnJhbWVzIGZhZGVVcHtmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWSg2cHgpfXRve29wYWNpdHk6MTt0cmFuc2Zvcm06bm9uZX19Ci5tYXAtY2FyZCwuc3RhdGUtcGFuZWwsLm5hci1jYXJkLC5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb246ZmFkZVVwIDAuNTVzIGN1YmljLWJlemllciguMiwuOCwuMiwxKSBiYWNrd2FyZHN9Ci5uYXItY2FyZDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMDdzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDMpe2FuaW1hdGlvbi1kZWxheTowLjE0c30KLnNpZ25hdHVyZS1pbnNpZ2h0e2FuaW1hdGlvbi1kZWxheTowLjA1c30KCkBtZWRpYShtYXgtd2lkdGg6MTEwMHB4KXsKICAubWFpbntncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyfQogIC5zdGF0ZS1wYW5lbHttYXgtaGVpZ2h0Om5vbmV9CiAgLm5hci1yb3d7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KfQoKLyog4pSA4pSAIFdIQVQgSU5ESUEgSVMgUkVBQ1RJTkcgVE8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi53aXItc2VjdGlvbntmbGV4OjE7bWluLXdpZHRoOjA7cGFkZGluZzoyMnB4IDI2cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JhY2tkcm9wLWZpbHRlcjpibHVyKDE2cHgpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjt9Ci53aXItc2VjdGlvbjo6YmVmb3Jle2NvbnRlbnQ6Jyc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3RvcDowO2JvdHRvbTowO3dpZHRoOjJweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byBib3R0b20sdmFyKC0tYWNjZW50KSx0cmFuc3BhcmVudCk7fQoud2lyLWhlYWRlcnttYXJnaW4tYm90dG9tOjE2cHh9Ci53aXItdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMjhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tYWNjZW50KTttYXJnaW4tYm90dG9tOjZweDtvcGFjaXR5OjAuOX0KLndpci1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjV9Ci53aXItc2lnbmFsc3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowfQoud2lyLXNpZ25hbHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6MTJweDtwYWRkaW5nOjExcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO29wYWNpdHk6MDthbmltYXRpb246d2lyRmFkZUluIDAuNXMgZWFzZSBmb3J3YXJkc30KLndpci1zaWduYWw6bGFzdC1jaGlsZHtib3JkZXItYm90dG9tOm5vbmV9CkBrZXlmcmFtZXMgd2lyRmFkZUlue2Zyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVZKDRweCl9dG97b3BhY2l0eToxO3RyYW5zZm9ybTpub25lfX0KLndpci1zaWduYWwtbGVmdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NHB4O2ZsZXgtc2hyaW5rOjA7cGFkZGluZy10b3A6NHB4O3dpZHRoOjUycHh9Ci53aXItcHVsc2V7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czo1MCU7ZmxleC1zaHJpbms6MDtwb3NpdGlvbjpyZWxhdGl2ZX0KLndpci1wdWxzZTo6YWZ0ZXJ7Y29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDotM3B4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgY3VycmVudENvbG9yO29wYWNpdHk6MDthbmltYXRpb246d2lyUmlwcGxlIDIuNXMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgd2lyUmlwcGxlezAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC41fTEwMCV7dHJhbnNmb3JtOnNjYWxlKDEuOCk7b3BhY2l0eTowfX0KLndpci10YWd7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjYuNXB4O2xldHRlci1zcGFjaW5nOjAuMWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtwYWRkaW5nOjJweCA1cHg7Ym9yZGVyLXJhZGl1czoycHg7Ym9yZGVyOjFweCBzb2xpZDtvcGFjaXR5OjAuNDU7d2hpdGUtc3BhY2U6bm93cmFwO3RleHQtYWxpZ246Y2VudGVyfQoud2lyLXNpZ25hbC10ZXh0e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTRweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU1O2xldHRlci1zcGFjaW5nOi0wLjAwNWVtfQoud2lyLXNpZ25hbC10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWFjY2VudCl9Ci53aXItc2lnbmFsLXRleHQgc3Ryb25ne2ZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjpyZ2JhKDI0MCwyMzUsMjI1LDAuODUpfQoud2lyLWxvYWRpbmd7ZGlzcGxheTpmbGV4O2dhcDo2cHg7cGFkZGluZzoxMnB4IDA7YWxpZ24taXRlbXM6Y2VudGVyfQoud2lyLWRvdHt3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOndpckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQoud2lyLWRvdDpudGgtY2hpbGQoMil7YW5pbWF0aW9uLWRlbGF5OjAuMnN9Ci53aXItZG90Om50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC40c30KQGtleWZyYW1lcyB3aXJEb3R7MCUsODAlLDEwMCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjN9NDAle3RyYW5zZm9ybTpzY2FsZSgxKTtvcGFjaXR5OjF9fQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PgoKPGRpdiBpZD0ibHRhYi10b29sdGlwIj48L2Rpdj4KPGRpdiBjbGFzcz0idG9wYmFyIj4KICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC1tYXJrIj48c3BhbiBjbGFzcz0iYnJhbmQtcHVsc2UtZG90Ij48L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJicmFuZC10ZXh0LWJsb2NrIj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLW5hbWUiPjxlbSBjbGFzcz0iYnJhbmQtcHVsc2Utd29yZCI+UHVsc2U8L2VtPiBvZiBJbmRpYTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImJyYW5kLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9zcGFuPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0idG9wYmFyLXIiPgogICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICA8c3BhbiBjbGFzcz0ibGl2ZS1kb3QiPjwvc3Bhbj4KICAgICAgPHNwYW4gaWQ9ImxpdmUtY291bnQiPuKApjwvc3Bhbj4gc2lnbmFscwogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjbG9jayIgaWQ9ImNsb2NrIj4tLTotLTotLSBJU1Q8L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIEhFUk8gLS0+CjxzZWN0aW9uIGNsYXNzPSJoZXJvIiBzdHlsZT0icGFkZGluZy10b3A6ODBweDtwYWRkaW5nLWJvdHRvbToyNHB4O3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbiI+CiAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7d2lkdGg6NjAwcHg7aGVpZ2h0OjM1MHB4O3RvcDotNjBweDtsZWZ0Oi04MHB4O2JhY2tncm91bmQ6cmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgYXQgNDAlIDUwJSxyZ2JhKDIyNCw5MCw0MCwwLjA1KSAwJSx0cmFuc3BhcmVudCA2NSUpO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowO2FuaW1hdGlvbjphbWJpZW50U2hpZnQgMTJzIGVhc2UtaW4tb3V0IGluZmluaXRlIGFsdGVybmF0ZSI+PC9kaXY+CiAgPHN0eWxlPkBrZXlmcmFtZXMgYW1iaWVudFNoaWZ0ezAle3RyYW5zZm9ybTp0cmFuc2xhdGVYKDApfTEwMCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMjRweCkgdHJhbnNsYXRlWSgtMTJweCl9fTwvc3R5bGU+CiAgPGRpdiBjbGFzcz0iaGVyby1leWVicm93IiBzdHlsZT0icG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxIj5Db2xsZWN0aXZlIGF0dGVudGlvbiAmbWlkZG90OyBJbmRpYTwvZGl2PgogIDxkaXYgY2xhc3M9Imhlcm8tYnJhbmQtYmxvY2siIHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxOHB4O21hcmdpbi1ib3R0b206MTZweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPgogICAgPGRpdiBjbGFzcz0iaGVyby1wdWxzZS1zaWduYWwiPgogICAgICA8c3BhbiBjbGFzcz0iaHBzLWNvcmUiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjEiPjwvc3Bhbj48c3BhbiBjbGFzcz0iaHBzLXJpbmcgcjIiPjwvc3Bhbj4KICAgIDwvZGl2PgogICAgPGgxIGNsYXNzPSJoZXJvLWJyYW5kLW5hbWUiPjxlbT5QdWxzZTwvZW0+IG9mIEluZGlhPC9oMT4KICA8L2Rpdj4KICA8cCBjbGFzcz0iaGVyby10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvcD4KICA8cCBjbGFzcz0iaGVyby1kZXNjIj5PYnNlcnZlIGhvdyBJbmRpYSdzIG5hcnJhdGl2ZXMgYW5kIHB1YmxpYyBhdHRlbnRpb24gc2hpZnQgaW4gcmVhbCB0aW1lLjwvcD4KICA8cCBjbGFzcz0iaGVyby1zdWItbGluZSI+T2JzZXJ2aW5nIEluZGlhIGluIG1vdGlvbi48L3A+CgogIDwhLS0gTElWRSBTVEFUUyBTVFJJUCAtLT4KPGRpdiBpZD0ic3RhdHMtc3RyaXAiIHN0eWxlPSIKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjI7CiAgYmFja2dyb3VuZDpyZ2JhKDksMTMsMjEsMC45KTsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDgpOwogIHBhZGRpbmc6MCAzNnB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpzdHJldGNoOwoiPgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCIgaWQ9InNjLXNpZ25hbHMiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlNpZ25hbHMgdHJhY2tlZDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIj5MaXZlIGluZ2VzdGlvbjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiIGlkPSJzYy1ob3R0ZXN0IiBzdHlsZT0iY3Vyc29yOnBvaW50ZXIiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWhvdHRlc3QtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWhvdHRlc3Qtc3ViIj5DbGljayB0byBleHBsb3JlPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCI+CiAgICA8ZGl2IGNsYXNzPSJzYy1sYWJlbCI+UGVhayBhbmdlciBzdGF0ZTwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLWFuZ2VyLXN1YiI+T3V0cmFnZSAmIHByb3Rlc3Qgc2lnbmFsczwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPlRvcCByaXNpbmcgbmFycmF0aXZlPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1uYXJyYXRpdmUtdmFsIj7igJQ8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hcnJhdGl2ZS1zdWIiPk5hdGlvbmFsIHNpZ25hbCBzdXJnZTwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPgogICAgPGRpdiBjbGFzcz0ic2MtbGFiZWwiPkZhc3Rlc3QgY29vbGluZzwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtY29vbGluZy12YWwiPuKAlDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtY29vbGluZy1zdWIiPlNpZ25hbCBkZWNheTwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzdHlsZT4KLnN0YXQtY2VsbHsKICBmbGV4OjE7cGFkZGluZzoxMHB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7CiAgdHJhbnNpdGlvbjpiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5zdGF0LWNlbGw6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpfQouc3RhdC1kaXZ7d2lkdGg6MXB4O2JhY2tncm91bmQ6cmdiYSgxNjAsMTkwLDIzMCwwLjA3KTtmbGV4LXNocmluazowO21hcmdpbjo4cHggMH0KLnNjLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNjLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE4cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW4tdG9wOjFweH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KPC9zdHlsZT4KCgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJ3aXItc2VjdGlvbiI+CiAgICAgIDxkaXYgY2xhc3M9Indpci1oZWFkZXIiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci10aXRsZSI+V2hhdCBJbmRpYSBpcyByZWFjdGluZyB0bzwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Indpci1zdWIiPkxpdmUgb2JzZXJ2YXRpb25zIGZyb20gcHVibGljIGF0dGVudGlvbiwgcmVnaW9uYWwgZGlzY291cnNlIGFuZCBlbWVyZ2luZyBuYXRpb25hbCBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ3aXItc2lnbmFscyIgaWQ9Indpci1zaWduYWxzIj4KICAgICAgICA8ZGl2IGNsYXNzPSJ3aXItbG9hZGluZyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+TmFycmF0aXZlIHNoaWZ0czwvc3Bhbj4KICAgICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjJweDsiPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIGFjdGl2ZSIgZGF0YS1wZXJpb2Q9IjNtIj4zTTwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic3RyaXAtdGFiIiBkYXRhLXBlcmlvZD0iNm0iPjZNPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzdHJpcC10YWIiIGRhdGEtcGVyaW9kPSIxeSI+MVk8L2J1dHRvbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDwhLS0gc2hpZnRzIGxpc3QgLS0+CiAgICAgIDxkaXYgc3R5bGU9ImZsZXg6MTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3BhZGRpbmc6MTBweCAxNHB4O2dhcDo2cHg7IiBpZD0ic2hpZnQtbGlzdCI+PC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKCjwhLS0gTUFJTjogTUFQICsgU1RBVEUgUEFORUwgLS0+CjxkaXYgY2xhc3M9Im1haW4iPgoKICA8ZGl2IGNsYXNzPSJtYXAtY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJtYXAtdG9wIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLXRpdGxlLWJsb2NrIj4KICAgICAgICA8ZGl2IGNsYXNzPSJtdCI+SW5kaWEgJm1kYXNoOyBjb2xsZWN0aXZlIGF0dGVudGlvbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1zIiBpZD0ibWFwLW1ldGEiPjMwIHN0YXRlcyAmbWlkZG90OyBsaXZlIHNpZ25hbCBjb21wb3NpdGU8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImxlZ2VuZCI+PHNwYW4+cXVpZXQ8L3NwYW4+PGRpdiBjbGFzcz0ibGVnZW5kLWJhciI+PC9kaXY+PHNwYW4+YWN0aXZlPC9zcGFuPjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJsYXllci1yb3ciPgogICAgICA8c3BhbiBjbGFzcz0ibGF5ZXItbGFiZWwiPlZpZXc8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9Imx0YWJzIj4KICAgICAgICA8c3BhbiBjbGFzcz0ibHRhYiBhY3RpdmUiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJlbW90aW9uIj5FbW90aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJUaGUgZG9taW5hbnQgZW1vdGlvbmFsIHRvbmUg4oCUIGFueGlvdXMsIGFuZ3J5LCBob3BlZnVsLCBwcm91ZCBvciBmZWFyZnVsLiBSZXZlYWxzIHRoZSBwc3ljaG9sb2dpY2FsIHVuZGVyY3VycmVudCBvZiBwb2xpdGljYWwgYXR0ZW50aW9uLiI+aTwvc3Bhbj48L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9InZlbG9jaXR5Ij5Nb21lbnR1bSA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iSXMgYXR0ZW50aW9uIHJpc2luZyBvciBmYWxsaW5nPyBSaXNpbmcgPSBuYXJyYXRpdmUgYWNjZWxlcmF0aW5nLiBDb29saW5nID0gbG9zaW5nIHRyYWN0aW9uLiBTaG93cyBzdGF0ZXMgZW50ZXJpbmcgb3IgZXhpdGluZyBhIHBvbGl0aWNhbCBjeWNsZS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWFwLXN2Zy13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibWFwLWlubmVyIj4KICAgICAgICA8c3ZnIGlkPSJpbmRpYS1tYXAiIHZpZXdCb3g9IjAgMCA4MDAgODAwIiBwcmVzZXJ2ZUFzcGVjdFJhdGlvPSJ4TWlkWU1pZCBtZWV0Ij4KICAgICAgICAgIDxkZWZzPgogICAgICAgICAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImFtYkdsb3ciIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjUwJSI+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0icmdiYSgyMjQsOTAsNDAsMC4wNCkiLz4KICAgICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgICAgICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICAgICAgICAgIDxmaWx0ZXIgaWQ9InN0YXRlR2xvdyIgeD0iLTMwJSIgeT0iLTMwJSIgd2lkdGg9IjE2MCUiIGhlaWdodD0iMTYwJSI+CiAgICAgICAgICAgICAgPGZlR2F1c3NpYW5CbHVyIGluPSJTb3VyY2VHcmFwaGljIiBzdGREZXZpYXRpb249IjgiIHJlc3VsdD0iYmx1ciIvPgogICAgICAgICAgICAgIDxmZUNvbXBvc2l0ZSBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJibHVyIiBvcGVyYXRvcj0ib3ZlciIvPgogICAgICAgICAgICA8L2ZpbHRlcj4KICAgICAgICAgIDwvZGVmcz4KICAgICAgICAgIDxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJ1cmwoI2FtYkdsb3cpIi8+CiAgICAgICAgICA8ZyBpZD0ibWFwLWdsb3ciPjwvZz4KICAgICAgICAgIDxnIGlkPSJtYXAtc3RhdGVzIj48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXB1bHNlcyI+PC9nPgogICAgICAgIDwvc3ZnPgogICAgICAgIDxkaXYgY2xhc3M9Im1hcC10b29sdGlwIiBpZD0idG9vbHRpcCI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gU1RBVEUgUEFORUwgLS0+CiAgPGRpdiBjbGFzcz0ic3RhdGUtcGFuZWwiIGlkPSJzdGF0ZS1kZXRhaWwiPgogICAgPGRpdiBjbGFzcz0icGFuZWwtZW1wdHkiPgogICAgICA8c3ZnIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxIj4KICAgICAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiA4djRNMTIgMTZoLjAxIi8+CiAgICAgIDwvc3ZnPgogICAgICA8ZGl2IGNsYXNzPSJwZS10Ij5TZWxlY3QgYSBzdGF0ZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJwZS1zIj5DbGljayBhbnkgcmVnaW9uIG9uIHRoZSBtYXA8YnIvPnRvIG9wZW4gaXRzIG5hcnJhdGl2ZSBwYW5lbC48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKPC9kaXY+Cgo8IS0tIE5BUlJBVElWRSBST1cgLS0+CjxkaXYgY2xhc3M9Im5hci1yb3ciPgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj48c3BhbiBjbGFzcz0ibmMtdGl0bGUiPlJpc2luZyBuYXJyYXRpdmVzPC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9InJpc2luZy1saXN0Ij48ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo4cHggMCI+TG9hZGluZy4uLjwvZGl2PjwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9Im5hci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9Im5jLWhlYWQiPjxzcGFuIGNsYXNzPSJuYy1kb3QgZmFsbCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJuYy10aXRsZSI+RGVjbGluaW5nIG5hcnJhdGl2ZXM8L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0iZGVjbGluaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0ibmFyLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibmMtaGVhZCI+PHNwYW4gY2xhc3M9Im5jLWRvdCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJuYy10aXRsZSI+UmVnaW9uYWwgc2hpZnRzPC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ibmMtYm9keSIgaWQ9InJlZ2lvbmFsLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSBSRVBMQVkgSU5ESUEgLS0+CjxzZWN0aW9uIGNsYXNzPSJyZXBsYXktc2VjdGlvbiI+CiAgPGRpdiBjbGFzcz0icmVwbGF5LWhlYWRlciI+CiAgICA8ZGl2PjxkaXYgY2xhc3M9InJlcGxheS1sYWJlbCI+UmVwbGF5IEluZGlhPC9kaXY+PGRpdiBjbGFzcz0icmVwbGF5LXN1YiI+V2F0Y2ggaG93IGNvbGxlY3RpdmUgYXR0ZW50aW9uIHNoaWZ0ZWQgb3ZlciB0aW1lPC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyZXBsYXktY29udHJvbHMiPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4gYWN0aXZlIiBkYXRhLXBlcmlvZD0iN2QiPjcgZGF5czwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1idG4iIGRhdGEtcGVyaW9kPSIzMGQiPjMwIGRheXM8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtYnRuIiBkYXRhLXBlcmlvZD0iNm0iPjYgbW9udGhzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9InJwLWJ0biIgZGF0YS1wZXJpb2Q9ImVsZWN0aW9uIj5FbGVjdGlvbiAyMDI0PC9idXR0b24+CiAgICA8L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJyZXBsYXktc2NydWJiZXIiPgogICAgPGRpdiBjbGFzcz0icnAtdHJhY2siIGlkPSJycC10cmFjayI+PGRpdiBjbGFzcz0icnAtZmlsbCIgaWQ9InJwLWZpbGwiPjwvZGl2PjxkaXYgY2xhc3M9InJwLXRodW1iIiBpZD0icnAtdGh1bWIiPjwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0icnAtZGF0ZXMiIGlkPSJycC1kYXRlcyI+PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0icmVwbGF5LXBsYXliYWNrIj4KICAgIDxidXR0b24gY2xhc3M9InJwLXBsYXkiIGlkPSJycC1wbGF5LWJ0biIgb25jbGljaz0idG9nZ2xlUmVwbGF5KCkiPgogICAgICA8c3ZnIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwb2x5Z29uIHBvaW50cz0iMiwxIDksNSAyLDkiIGlkPSJycC1wbGF5LWljb24iLz48L3N2Zz4KICAgIDwvYnV0dG9uPgogICAgPGRpdiBjbGFzcz0icnAtY3VycmVudC1kYXRlIiBpZD0icnAtY3VycmVudC1kYXRlIj5TZWxlY3QgYSBwZXJpb2QgYW5kIHByZXNzIHBsYXk8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJwLXNwZWVkIj48c3BhbiBjbGFzcz0icnAtc3BlZWQtbGFiZWwiPlNwZWVkPC9zcGFuPgogICAgICA8YnV0dG9uIGNsYXNzPSJycC1zcGQgYWN0aXZlIiBkYXRhLXNwZD0iMSI+MXg8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iMiI+Mng8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0icnAtc3BkIiBkYXRhLXNwZD0iNCI+NHg8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InJlcGxheS1zbmFwc2hvdCI+PGRpdiBjbGFzcz0icnAtc25hcC1sYWJlbCI+TmFycmF0aXZlIHNuYXBzaG90IGF0IHRoaXMgbW9tZW50PC9kaXY+PGRpdiBjbGFzcz0icnAtc25hcC1zdGF0ZXMiIGlkPSJycC1zbmFwLXN0YXRlcyI+PGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5QcmVzcyBwbGF5IHRvIG9ic2VydmUgSW5kaWEgaW4gbW90aW9uLjwvZGl2PjwvZGl2PjwvZGl2Pgo8L3NlY3Rpb24+CjxzdHlsZT4KLnJlcGxheS1zZWN0aW9ue3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTttYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87cGFkZGluZzowIDM2cHggMzZweH0KLnJlcGxheS1oZWFkZXJ7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtZW5kO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206MjBweDtnYXA6MjBweDtmbGV4LXdyYXA6d3JhcH0KLnJlcGxheS1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIwcHg7Zm9udC13ZWlnaHQ6MzAwO2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOnZhcigtLWluayk7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbX0KLnJlcGxheS1zdWJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjRweH0KLnJlcGxheS1jb250cm9sc3tkaXNwbGF5OmZsZXg7Z2FwOjRweDtmbGV4LXdyYXA6d3JhcH0KLnJwLWJ0bntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO3BhZGRpbmc6NXB4IDEycHg7Ym9yZGVyLXJhZGl1czo0cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Y29sb3I6dmFyKC0tZmFpbnQpO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246YWxsIDAuMTVzfQoucnAtYnRuLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNyk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5yZXBsYXktc2NydWJiZXJ7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MThweCAyMHB4IDE0cHg7bWFyZ2luLWJvdHRvbToxMnB4fQoucnAtdHJhY2t7cG9zaXRpb246cmVsYXRpdmU7aGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyLXJhZGl1czoycHg7Y3Vyc29yOnBvaW50ZXI7bWFyZ2luLWJvdHRvbToxMHB4fQoucnAtZmlsbHtwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MCU7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQodG8gcmlnaHQscmdiYSgyMjQsOTAsNDAsMC40KSx2YXIoLS1hY2NlbnQpKTtib3JkZXItcmFkaXVzOjJweH0KLnJwLXRodW1ie3Bvc2l0aW9uOmFic29sdXRlO3RvcDo1MCU7dHJhbnNmb3JtOnRyYW5zbGF0ZSgtNTAlLC01MCUpO3dpZHRoOjEycHg7aGVpZ2h0OjEycHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JvcmRlcjoycHggc29saWQgcmdiYSg5LDEzLDIxLDAuOCk7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoMjI0LDkwLDQwLDAuNCk7bGVmdDowJTtjdXJzb3I6Z3JhYn0KLnJwLWRhdGVze2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJlcGxheS1wbGF5YmFja3tkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNHB4O21hcmdpbi1ib3R0b206MTZweH0KLnJwLXBsYXl7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjI1KTtjb2xvcjp2YXIoLS1hY2NlbnQpO2N1cnNvcjpwb2ludGVyO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE1c30KLnJwLWN1cnJlbnQtZGF0ZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO2ZsZXg6MX0KLnJwLXNwZWVke2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweH0KLnJwLXNwZWVkLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1yaWdodDoycHh9Ci5ycC1zcGR7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O3BhZGRpbmc6M3B4IDhweDtib3JkZXItcmFkaXVzOjNweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDp0cmFuc3BhcmVudDtjb2xvcjp2YXIoLS1mYWludCk7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xNXN9Ci5ycC1zcGQuYWN0aXZle2NvbG9yOnZhcigtLWluayk7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIpfQoucmVwbGF5LXNuYXBzaG90e2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTJweDtwYWRkaW5nOjE2cHggMjBweH0KLnJwLXNuYXAtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjEycHh9Ci5ycC1zbmFwLXN0YXRlc3tkaXNwbGF5OmZsZXg7ZmxleC13cmFwOndyYXA7Z2FwOjhweH0KLnJwLWxvZy1lbXB0eXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNSk7Zm9udC1zdHlsZTppdGFsaWM7cGFkZGluZzo0cHggMH0KLnJwLXN0YXRlLWNhcmR7cGFkZGluZzo4cHggMTJweDtib3JkZXItcmFkaXVzOjZweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO21pbi13aWR0aDoxNDBweH0KLnJwLXN0YXRlLW5hbWV7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDA7bWFyZ2luLWJvdHRvbTozcHh9Ci5ycC1zdGF0ZS1uYXJ7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KX0KLnJwLXN0YXRlLWF0dHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWFjY2VudCl9Cjwvc3R5bGU+CjwhLS0gRkFWUyAtLT4KPHNlY3Rpb24gY2xhc3M9ImZhdnMiPgogIDxkaXYgY2xhc3M9ImZhdnMtbGFiZWwiPlRyYWNrZWQgc3RhdGVzPC9kaXY+CiAgPGRpdiBjbGFzcz0iZmF2cy1yb3ciIGlkPSJmYXYtcm93Ij4KICAgIDxkaXYgY2xhc3M9ImZhdnMtZW1wdHkiPk5vIHN0YXRlcyB0cmFja2VkLiBCb29rbWFyayBhbnkgc3RhdGUgcGFuZWwgdG8gZm9sbG93IGl0cyBuYXJyYXRpdmUgZXZvbHV0aW9uLjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8ZGl2IGNsYXNzPSJmb290Ij4KICA8ZGl2IGNsYXNzPSJmb290LW5hbWUiPlB1bHNlIG9mIEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1saW5lIj5PYnNlcnZlcyBob3cgcHVibGljIGF0dGVudGlvbiBzaGlmdHMgYWNyb3NzIHRoZSBjb3VudHJ5IOKAlCB1c2luZyBzaWduYWxzIGZyb20gbmV3cywgZGlzY291cnNlLCBhbmQgcmVnaW9uYWwgZGV2ZWxvcG1lbnRzLjwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Qtc3ViIj5Ob3QgbmV3cy4gTm90IHByZWRpY3Rpb24uIE9ic2VydmF0aW9uLjwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RvcG9qc29uLWNsaWVudEAzLjEuMC9kaXN0L3RvcG9qc29uLWNsaWVudC5taW4uanMiPjwvc2NyaXB0Pgo8c2NyaXB0Pgp2YXIgQVBJX0JBU0U9KGxvY2F0aW9uLmhvc3RuYW1lPT09J2xvY2FsaG9zdCd8fGxvY2F0aW9uLmhvc3RuYW1lPT09JzEyNy4wLjAuMScpPydodHRwOi8vbG9jYWxob3N0OjgwMDAnOicnOwoKLy8gQVBJCmFzeW5jIGZ1bmN0aW9uIGZldGNoQWxsU3RhdGVzKCl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGVzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIHJvd3M9YXdhaXQgci5qc29uKCk7CiAgICBpZighcm93c3x8IXJvd3MubGVuZ3RoKSByZXR1cm47CiAgICByb3dzLmZvckVhY2goZnVuY3Rpb24ocm93KXsKICAgICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMocm93LmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb21FbW89cm93LmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihlbW9zKXx8bnVsbDsKICAgICAgdmFyIGVudHJ5PXthdHRlbnRpb246cm93LmF0dGVudGlvbixkZWx0YTpyb3cuZGVsdGFfMjRoLHZlbG9jaXR5OnJvdy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbUVtbyxkb21pbmFudF9uYXJyYXRpdmU6cm93LmRvbWluYW50X25hcnJhdGl2ZSxlbW90aW9uczplbW9zfTsKICAgICAgTElWRVtyb3cubmFtZV09ZW50cnk7CiAgICAgIGlmKCFTRFtyb3cubmFtZV0pIFNEW3Jvdy5uYW1lXT1PYmplY3QuYXNzaWduKHt9LERFRkFVTFQpOwogICAgICBPYmplY3QuYXNzaWduKFNEW3Jvdy5uYW1lXSxlbnRyeSk7CiAgICB9KTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIHNldFRpbWVvdXQocmVuZGVyTW9tZW50dW0sIDUwMCk7CiAgICBpZihTRUwmJkxJVkVbU0VMXSYmZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpKSByZW5kZXJQYW5lbChTRUwpOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tBUEldJyxlLm1lc3NhZ2UpO30KfQoKZnVuY3Rpb24gYnVpbGRMb2NhbEluc2lnaHQoKXsKICB2YXIgZW50cmllcz1PYmplY3QuZW50cmllcyhMSVZFKTsKICBpZighZW50cmllcy5sZW5ndGgpIHJldHVybjsKCiAgLy8gQWdncmVnYXRlIHRvcCBuYXJyYXRpdmVzIGFjcm9zcyBhbGwgc3RhdGVzCiAgdmFyIG5hcj17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIGlmKCFuYXJbbi5uYW1lXSkgbmFyW24ubmFtZV09e3VwOjAsZG93bjowLGZsYXQ6MCx0b3RhbDowfTsKICAgICAgbmFyW24ubmFtZV1bbi5kaXJdPShuYXJbbi5uYW1lXVtuLmRpcl18fDApK24udmFsOwogICAgICBuYXJbbi5uYW1lXS50b3RhbD0obmFyW24ubmFtZV0udG90YWx8fDApK24udmFsOwogICAgfSk7CiAgfSk7CgogIC8vIFRvcCByaXNpbmcgYW5kIGZhbGxpbmcgKGV4Y2x1ZGUgdGllcyB3aGVyZSBzYW1lIG5hbWUgcmlzZXMgYW5kIGZhbGxzKQogIHZhciByaXNpbmc9T2JqZWN0LmVudHJpZXMobmFyKS5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS51cD5rdlsxXS5kb3duO30pCiAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnVwLWFbMV0udXA7fSkuc2xpY2UoMCwzKTsKICB2YXIgZmFsbGluZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLmRvd24+a3ZbMV0udXA7fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0uZG93bi1hWzFdLmRvd247fSkuc2xpY2UoMCwyKTsKICB2YXIgdG9wMz1PYmplY3QuZW50cmllcyhuYXIpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS50b3RhbC1hWzFdLnRvdGFsO30pLnNsaWNlKDAsMyk7CgogIC8vIEhvdHRlc3Qgc3RhdGUKICB2YXIgaG90dGVzdD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgdmFyIGhvdHRlc3RFbW89aG90dGVzdD8oTElWRVtob3R0ZXN0WzBdXSYmTElWRVtob3R0ZXN0WzBdXS5kb21pbmFudF9lbW90aW9uKXx8Jyc6JycgOwoKICAvLyBCdWlsZCBpbnNpZ2h0IHRleHQg4oCUIG1vcmUgYW5hbHl0aWNhbCwgY29udGV4dC1hd2FyZQogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgaWYoIWVsKSByZXR1cm47CgogIHZhciBsaW5lcz1bXTsKICBpZihyaXNpbmcubGVuZ3RoJiZmYWxsaW5nLmxlbmd0aCYmcmlzaW5nWzBdWzBdIT09ZmFsbGluZ1swXVswXSl7CiAgICBsaW5lcy5wdXNoKCc8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPiBpcyB0aGUgZG9taW5hbnQgc2lnbmFsIGFjcm9zcyBJbmRpYSB0b2RheScpOwogICAgaWYoZmFsbGluZ1swXSkgbGluZXMucHVzaCgnIGFzIDxlbT4nK2ZhbGxpbmdbMF1bMF0rJzwvZW0+IGZhZGVzIGZyb20gbmF0aW9uYWwgZm9jdXMnKTsKICAgIGlmKGhvdHRlc3QpIGxpbmVzLnB1c2goJy4gPHN0cm9uZyBzdHlsZT0iY29sb3I6dmFyKC0taW5rKSI+Jytob3R0ZXN0WzBdKyc8L3N0cm9uZz4gaXMgdGhlIG1vc3QgYWN0aXZlIHN0YXRlJysKICAgICAgKGhvdHRlc3RFbW8/JyB3aXRoICcraG90dGVzdEVtbysnIGFzIHRoZSBwcmltYXJ5IHNpZ25hbCB0b25lJzonJykpOwogICAgaWYocmlzaW5nWzFdKSBsaW5lcy5wdXNoKCcuIFNlY29uZGFyeSBzdXJnZTogPGVtPicrcmlzaW5nWzFdWzBdKyc8L2VtPicpOwogIH0gZWxzZSBpZihyaXNpbmcubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ1NpZ25hbHMgYXJlIGNvbmNlbnRyYXRlZCBhcm91bmQgPGVtPicrcmlzaW5nWzBdWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3Jpc2luZ1swXVswXS5zbGljZSgxKSsnPC9lbT4nKTsKICAgIGlmKGhvdHRlc3QpIGxpbmVzLnB1c2goJy4gPHN0cm9uZyBzdHlsZT0iY29sb3I6dmFyKC0taW5rKSI+Jytob3R0ZXN0WzBdKyc8L3N0cm9uZz4gbGVhZHMgbmF0aW9uYWwgYXR0ZW50aW9uJyk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJyBhbG9uZ3NpZGUgPGVtPicrcmlzaW5nWzFdWzBdKyc8L2VtPicpOwogIH0gZWxzZSBpZih0b3AzLmxlbmd0aCl7CiAgICBsaW5lcy5wdXNoKCdOYXRpb25hbCBzaWduYWxzIGFyZSBkaXNwZXJzZWQuIFRvcCBuYXJyYXRpdmVzOiAnK3RvcDMubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGVtPicrblswXSsnPC9lbT4nO30pLmpvaW4oJywgJykpOwogIH0KCiAgaWYobGluZXMubGVuZ3RoKXsKICAgIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytsaW5lcy5qb2luKCcnKSsnLjwvZGl2Pic7CiAgfQoKICAvLyBUYWdzCiAgaWYodEVsKXsKICAgIHZhciB0YWdzPVtdOwogICAgZmFsbGluZy5zbGljZSgwLDEpLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDU5LDE4NCwyMTYsMC4zKTtjb2xvcjojM2JiOGQ4Ij7ihpMgJytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKTsKICAgIH0pOwogICAgcmlzaW5nLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIHRhZ3MucHVzaCgnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjMpO2NvbG9yOiNlMDVhMjgiPuKGkSAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICBpZih0YWdzLmxlbmd0aCkgdEVsLmlubmVySFRNTD10YWdzLmpvaW4oJycpOwogIH0KCiAgaWYobWV0YUVsKXsKICAgIHZhciBzdGF0ZUNvdW50PU9iamVjdC52YWx1ZXMoTElWRSkuZmlsdGVyKGZ1bmN0aW9uKHMpe3JldHVybiBzLmF0dGVudGlvbj4yO30pLmxlbmd0aDsKICAgIG1ldGFFbC50ZXh0Q29udGVudD0nT2JzZXJ2aW5nICcrc3RhdGVDb3VudCsnIGFjdGl2ZSBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKCkudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicse2hvdXI6JzItZGlnaXQnLG1pbnV0ZTonMi1kaWdpdCd9KTsKICB9Cn0KCmZ1bmN0aW9uIHVwZGF0ZUFsbFN0cmlwcygpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBob3R0ZXN0PWVudHJpZXMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCk+KGFbMV0uYXR0ZW50aW9ufHwwKT9iOmE7fSxlbnRyaWVzWzBdKTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsaG90dGVzdFswXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCdBdHRlbnRpb24gJytob3R0ZXN0WzFdLmF0dGVudGlvbi50b0ZpeGVkKDEpKTsKICB2YXIgdG9wQW5nZXJObT1udWxsLHRvcEFuZ2VyUGN0PTA7CiAgZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXsKICAgIHZhciBlPWt2WzFdLmVtb3Rpb25zfHx7fTsKICAgIHZhciBhPWUuYW5nZXJ8fDA7CiAgICBpZihhPjAmJmE8PTEpIGE9TWF0aC5yb3VuZChhKjEwMCk7CiAgICBpZihhPnRvcEFuZ2VyUGN0KXt0b3BBbmdlclBjdD1hO3RvcEFuZ2VyTm09a3ZbMF07fQogIH0pOwogIGlmKHRvcEFuZ2VyTm0mJnRvcEFuZ2VyUGN0PjApewogICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlck5tKTsKICAgIHNldFRleHQoJ3NjLWFuZ2VyLXN1YicsJ0FuZ2VyICcrTWF0aC5yb3VuZCh0b3BBbmdlclBjdCkrJyUgb2Ygc2lnbmFscycpOwogIH0gZWxzZSB7CiAgICAvLyBGYWxsIGJhY2sgdG8gZG9taW5hbnRfZW1vdGlvbj1hbmdlcgogICAgdmFyIGFuZ2VyRG9tPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcic7fSk7CiAgICBpZihhbmdlckRvbS5sZW5ndGgpewogICAgICB2YXIgdG9wQnlBdHQ9YW5nZXJEb20uc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgICAgIHNldFRleHQoJ3NjLWFuZ2VyLXZhbCcsdG9wQnlBdHRbMF0pOwogICAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLCdEb21pbmFudCBlbW90aW9uOiBhbmdlcicpOwogICAgfQogIH0KICB2YXIgY29vbGluZz1lbnRyaWVzLnJlZHVjZShmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS52ZWxvY2l0eXx8MCk8KGFbMV0udmVsb2NpdHl8fDApP2I6YTt9LGVudHJpZXNbMF0pOwogIHNldFRleHQoJ3NjLWNvb2xpbmctdmFsJyxjb29saW5nWzBdKTtzZXRUZXh0KCdzYy1jb29saW5nLXN1YicsJ1ZlbG9jaXR5ICcrY29vbGluZ1sxXS52ZWxvY2l0eS50b0ZpeGVkKDMpKTsKICB2YXIgbmM9e307ZW50cmllcy5mb3JFYWNoKGZ1bmN0aW9uKGt2KXtpZihrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUpbmNba3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlXT0obmNba3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlXXx8MCkrMTt9KTsKICB2YXIgdG49T2JqZWN0LmVudHJpZXMobmMpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pWzBdOwogIGlmKHRuKXtzZXRUZXh0KCdzYy1uYXJyYXRpdmUtdmFsJyx0blswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0blswXS5zbGljZSgxKSk7c2V0VGV4dCgnc2MtbmFycmF0aXZlLXN1YicsJ0RvbWluYW50IGFjcm9zcyAnK3RuWzFdKycgc3RhdGVzJyk7fQp9CmFzeW5jIGZ1bmN0aW9uIGZldGNoRGV0YWlsKG5hbWUpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlLycrZW5jb2RlVVJJQ29tcG9uZW50KG5hbWUpKTsKICAgIGlmKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3IoJ0hUVFAgJytyLnN0YXR1cyk7CiAgICB2YXIgZD1hd2FpdCByLmpzb24oKTsKICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKGQuZW1vdGlvbnN8fHt9KTsKICAgIHZhciBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxkLmRvbWluYW50X2Vtb3Rpb258fG51bGw7CiAgICBTRFtuYW1lXT17YXR0ZW50aW9uOmQuYXR0ZW50aW9uLGRlbHRhOmQuZGVsdGFfMjRoLHZlbG9jaXR5OmQudmVsb2NpdHksZW1vdGlvbnM6ZW1vcyxkb21pbmFudF9lbW90aW9uOmRvbSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmUsCiAgICAgIG5hcnJhdGl2ZXM6KGQubmFycmF0aXZlc3x8W10pLm1hcChmdW5jdGlvbihuKXtyZXR1cm57bmFtZTpuLm5hbWUsdmFsOm4udmFsLGRpcjpuLmRpcnx8J2ZsYXQnfTt9KSwKICAgICAgcmlzaW5nOmQucmlzaW5nfHxbXSxmYWxsaW5nOmQuZmFsbGluZ3x8W10sc3VtbWFyeTpkLnN1bW1hcnl8fERFRkFVTFQuc3VtbWFyeSwKICAgICAgYXJ0aWNsZXM6ZC5hcnRpY2xlc3x8W10sdGltZWxpbmU6ZC50aW1lbGluZXx8REVGQVVMVC50aW1lbGluZSwKICAgICAgbmFycmF0aXZlSGlzdG9yeTpkLm5hcnJhdGl2ZUhpc3Rvcnl8fERFRkFVTFQubmFycmF0aXZlSGlzdG9yeSxzaWduYWxfY291bnQ6ZC5zaWduYWxfY291bnR8fDB9OwogICAgaWYoIUxJVkVbbmFtZV0pTElWRVtuYW1lXT17YXR0ZW50aW9uOmQuYXR0ZW50aW9uLGRlbHRhOmQuZGVsdGFfMjRoLHZlbG9jaXR5OmQudmVsb2NpdHksZG9taW5hbnRfbmFycmF0aXZlOmQuZG9taW5hbnRfbmFycmF0aXZlfTsKICAgIExJVkVbbmFtZV0uZW1vdGlvbnM9ZW1vcztMSVZFW25hbWVdLmRvbWluYW50X2Vtb3Rpb249ZG9tOwogICAgcmV0dXJuIFNEW25hbWVdOwogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tmZXRjaERldGFpbF0nLG5hbWUsZS5tZXNzYWdlKTtyZXR1cm4gU0RbbmFtZV18fERFRkFVTFQ7fQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaFNuYXAoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9zbmFwc2hvdC9kYWlseScpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC5lcnJvcikgcmV0dXJuOwogICAgLy8gdG9wYmFyCiAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xpdmUtY291bnQnKTsKICAgIGlmKGVsJiZkLnRvdGFsX3NpZ25hbHMpIGVsLnRleHRDb250ZW50PWQudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOwogICAgdmFyIG1ldGE9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1tZXRhJyk7CiAgICBpZihtZXRhJiZkLmFzX29mKSBtZXRhLnRleHRDb250ZW50PSczMCBzdGF0ZXMgwrcgdXBkYXRlZCAnK25ldyBEYXRlKGQuYXNfb2YpLnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nKTsKICAgIC8vIHN0YXRzIHN0cmlwCiAgICBzZXRUZXh0KCdzYy1zaWduYWxzLXZhbCcsIGQudG90YWxfc2lnbmFscz9kLnRvdGFsX3NpZ25hbHMudG9Mb2NhbGVTdHJpbmcoKTonLScpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgfWNhdGNoKGUpe30KfQoKZnVuY3Rpb24gc2V0VGV4dChpZCx2YWwpe3ZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoZWwpZWwudGV4dENvbnRlbnQ9dmFsO30KCmZ1bmN0aW9uIHVwZGF0ZVN0cmlwTmFycmF0aXZlKCl7dXBkYXRlQWxsU3RyaXBzKCk7fQpmdW5jdGlvbiB1cGRhdGVTdHJpcEFuZ2VyKCl7fQoKZnVuY3Rpb24gc2VsZWN0SG90dGVzdCgpewogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoU0QpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pWzBdOwogIGlmKHRvcCkgc2VsZWN0Xyh0b3BbMF0pOwp9CmFzeW5jIGZ1bmN0aW9uIGZldGNoSW5zaWdodHMoKXsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9pbnNpZ2h0cycpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoZC5lcnJvcikgcmV0dXJuOwogICAgdmFyIHNpZz1kLnNpZ25hdHVyZTsKICAgIGlmKHNpZyl7CiAgICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgICAgaWYoZWwpZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij48ZW0+JytzaWcuZmFkaW5nLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3NpZy5mYWRpbmcuc2xpY2UoMSkrJzwvZW0+IGZhZGluZyBhcyA8ZW0+JytzaWcucmlzaW5nX3ByaW1hcnkrIjwvZW0+Iisoc2lnLnJpc2luZ19zZWNvbmRhcnk/IiBhbG9uZ3NpZGUgPGVtPiIrc2lnLnJpc2luZ19zZWNvbmRhcnkrIjwvZW0+IjoiIikrIiBhY3Jvc3MgdGhlIG5hdGlvbmFsIGNvbnZlcnNhdGlvbi4gPHN0cm9uZyBzdHlsZT1cImNvbG9yOnZhcigtLWluaylcIj4iK3NpZy5ob3R0ZXN0X3N0YXRlKyI8L3N0cm9uZz4gZG9taW5hdGVzLjwvZGl2PiI7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIGlmKHRFbCYmZC50YWdzKXRFbC5pbm5lckhUTUw9ZC50YWdzLm1hcChmdW5jdGlvbih0KXtyZXR1cm4gJzxzcGFuIGNsYXNzPSJzaS10YWciPicrKHQuZGlyPT09J2Rvd24nPyfihpMgJzon4oaRICcpK3QubGFiZWwrJzwvc3Bhbj4nO30pLmpvaW4oJycpOwogICAgfQogICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgIGlmKHJFbCYmZC5yaXNpbmcmJmQucmlzaW5nLmxlbmd0aClyRWwuaW5uZXJIVE1MPWQucmlzaW5nLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IGNsYXNzPSJuaS1uYW1lIj4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytuLnN0YXRlcy5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKSsnJTtiYWNrZ3JvdW5kOiNlMDVhMjgiPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogICAgaWYoZkVsJiZkLmZhbGxpbmcmJmQuZmFsbGluZy5sZW5ndGgpZkVsLmlubmVySFRNTD1kLmZhbGxpbmcubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKG4uc3RhdGVzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK24uc3RhdGVzLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLG4uc2lnbmFsX3NoYXJlKjMpKyclO2JhY2tncm91bmQ6IzNiYjhkOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgIGlmKGdFbCYmZC5yZWdpb25hbCYmZC5yZWdpb25hbC5sZW5ndGgpZ0VsLmlubmVySFRNTD1kLnJlZ2lvbmFsLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW4iPjxzcGFuIGNsYXNzPSJuaS1uYW1lIj4nK3IucmVnaW9uKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+JytyLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PjxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytyLmhvdHRlc3Rfc3RhdGUrJyDCtyAnK3IudG9wX25hcnJhdGl2ZSsnPC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hGdWxsU25hcHNob3QoKXsKICAvLyBMb2FkIEFMTCBzdGF0ZSBkYXRhIGluIG9uZSByZXF1ZXN0IGZvciBpbnN0YW50IGZpcnN0LWxvYWQKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9mdWxsLXNuYXBzaG90Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLndhcm1pbmdfdXB8fCFkLnN0YXRlc3x8IWQuc3RhdGVzLmxlbmd0aCkgcmV0dXJuIGZhbHNlOwoKICAgIC8vIFBvcHVsYXRlIFNEIGFuZCBMSVZFIGZyb20gZnVsbCBzbmFwc2hvdAogICAgZC5zdGF0ZXMuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgICAgaWYoIXMubmFtZSkgcmV0dXJuOwogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhzLmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxzLmRvbWluYW50X2Vtb3Rpb258fG51bGw7CiAgICAgIHZhciBlbnRyeT1PYmplY3QuYXNzaWduKHt9LHMse2Vtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZGVsdGE6cy5kZWx0YV8yNGh8fDB9KTsKICAgICAgU0Rbcy5uYW1lXT1lbnRyeTsKICAgICAgTElWRVtzLm5hbWVdPXthdHRlbnRpb246cy5hdHRlbnRpb24sZGVsdGE6cy5kZWx0YV8yNGh8fDAsdmVsb2NpdHk6cy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbSxkb21pbmFudF9uYXJyYXRpdmU6cy5kb21pbmFudF9uYXJyYXRpdmUsZW1vdGlvbnM6ZW1vc307CiAgICB9KTsKCiAgICAvLyBVcGRhdGUgc2lnbmFscyBjb3VudAogICAgaWYoZC5zbmFwc2hvdCYmZC5zbmFwc2hvdC50b3RhbF9zaWduYWxzKXsKICAgICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLGQuc25hcHNob3QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpKTsKICAgIH0KCiAgICAvLyBVcGRhdGUgaW5zaWdodHMgZnJvbSBjYWNoZWQgZGF0YQogICAgaWYoZC5pbnNpZ2h0cyYmZC5pbnNpZ2h0cy5zaWduYXR1cmUpewogICAgICB2YXIgc2lnPWQuaW5zaWdodHMuc2lnbmF0dXJlOwogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQuaW5zaWdodHMudGFncyl0RWwuaW5uZXJIVE1MPWQuaW5zaWdodHMudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgICAgaWYockVsJiZkLmluc2lnaHRzLnJpc2luZyYmZC5pbnNpZ2h0cy5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTtyZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6I2UwNWEyOCI+4oaRIHJpc2luZzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NHB4OyI+JytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiNlMDVhMjg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICAgIGlmKGZFbCYmZC5pbnNpZ2h0cy5mYWxsaW5nJiZkLmluc2lnaHRzLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgfQoKICAgIC8vIFJlbmRlciBtYXAgY29sb3JzIGFuZCBzdHJpcHMKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIC8vIExvYWQgaW5zaWdodHMgdG9vCiAgICBidWlsZExvY2FsSW5zaWdodCgpOwogICAgZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7CiAgICAvLyBVc2UgY2FjaGVkIG5hcnJhdGl2ZSBpbnNpZ2h0IGlmIGF2YWlsYWJsZQogICAgaWYoZC5uYXJyYXRpdmVfaW5zaWdodCYmZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0KXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwogICAgICBpZihlbCkgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2QubmFycmF0aXZlX2luc2lnaHQudGV4dCsnPC9kaXY+JzsKICAgICAgaWYodEVsJiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRvcF9uYXJyYXRpdmVzKXsKICAgICAgfQogICAgfQogICAgcmV0dXJuIHRydWU7CiAgfWNhdGNoKGUpewogICAgY29uc29sZS53YXJuKCdbZnVsbC1zbmFwc2hvdF0nLGUubWVzc2FnZSk7CiAgICByZXR1cm4gZmFsc2U7CiAgfQp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKXsKICB0cnl7CiAgICAvLyBUcnkgY2FjaGVkIHZlcnNpb24gZnJvbSBmdWxsLXNuYXBzaG90IGZpcnN0IChhbHJlYWR5IGxvYWRlZCkKICAgIC8vIFRoZW4gY2FsbCBkZWRpY2F0ZWQgZW5kcG9pbnQgZm9yIGZyZXNoIEFJIGFuYWx5c2lzCiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9uYXJyYXRpdmUtaW5zaWdodCcpOwogICAgaWYoIXIub2spIHJldHVybjsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgaWYoIWQudGV4dCkgcmV0dXJuOwoKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLWluc2lnaHQnKTsKICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICB2YXIgbWV0YUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctbWV0YScpOwoKICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC50ZXh0Kyc8L2Rpdj4nOwoKICAgIC8vIFRhZ3MgZnJvbSB0b3AgbmFycmF0aXZlcwogICAgaWYodEVsJiZkLnRvcF9uYXJyYXRpdmVzJiZkLnRvcF9uYXJyYXRpdmVzLmxlbmd0aCl7CiAgICAgIHRFbC5pbm5lckhUTUw9ZC50b3BfbmFycmF0aXZlcy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgICB2YXIgY29sPWk9PT0wPycjZTA1YTI4JzoncmdiYSgxNjAsMTkwLDIzMCwwLjYpJzsKICAgICAgICB2YXIgYXJyb3c9aT09PTA/J+KGkSAnOifCtyAnOwogICAgICAgIHJldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDIyNCw5MCw0MCwwLjIpO2NvbG9yOicrY29sKyciPicrYXJyb3crbi5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLnNsaWNlKDEpKyc8L3NwYW4+JzsKICAgICAgfSkuam9pbignJyk7CiAgICB9CgogICAgaWYobWV0YUVsKXsKICAgICAgdmFyIHQ9bmV3IERhdGUoZC5hc19vZik7CiAgICAgIG1ldGFFbC50ZXh0Q29udGVudD0nU2lnbmFsIGFuYWx5c2lzIMK3ICcrdC50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pKyhkLmZhbGxiYWNrPycgwrcgcGF0dGVybi1iYXNlZCc6JyDCtyBBSSBzeW50aGVzaXplZCcpOwogICAgfQogIH1jYXRjaChlKXtjb25zb2xlLndhcm4oJ1tuYXJyYXRpdmVdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRQb2xsaW5nKCl7CiAgYXdhaXQgUHJvbWlzZS5hbGwoW2ZldGNoQWxsU3RhdGVzKCksZmV0Y2hTbmFwKCldKTsKICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlKTt9KTsKICB2YXIgbj0wOwogIHZhciB0PXNldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7CiAgICBuKys7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTsKICAgIGlmKFNFTCkgcmVuZGVyUGFuZWwoU0VMKTsKICAgIGlmKG4+PTEyKXtjbGVhckludGVydmFsKHQpO3NldEludGVydmFsKGFzeW5jIGZ1bmN0aW9uKCl7YXdhaXQgZmV0Y2hBbGxTdGF0ZXMoKTthd2FpdCBmZXRjaFNuYXAoKTtpZihTRUwpcmVuZGVyUGFuZWwoU0VMKTt9LDEyMDAwMCk7CiAgICAgIHNldEludGVydmFsKGZldGNoSW5zaWdodHMsMzYwMDAwMCk7fQogIH0sMTUwMDApOwp9CgovLyBOQVJSQVRJVkUgREFUQQp2YXIgU0hJRlRTPXsKICAnM20nOlsKICAgIHtmYWRpbmc6J0luZmxhdGlvbicsZmFkaW5nTm90ZTonZWFzaW5nIG5hdGlvbmFsbHknLHJpc2luZzonQm9yZGVyIHNlY3VyaXR5JyxyaXNpbmdOb3RlOidwb3N0LWluY2lkZW50IHN1cmdlJ30sCiAgICB7ZmFkaW5nOidFbGVjdGlvbiByaGV0b3JpYycsZmFkaW5nTm90ZToncG9zdC1jeWNsZSBmYWRlJyxyaXNpbmc6J0dvdmVybmFuY2UgYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3N0ZWFkeSByaXNlJ30sCiAgICB7ZmFkaW5nOidGYXJtZXIgcHJvdGVzdHMnLGZhZGluZ05vdGU6J21vbWVudHVtIGxvc3QnLHJpc2luZzonVW5lbXBsb3ltZW50IGFueGlldHknLHJpc2luZ05vdGU6J3lvdXRoIHNpZ25hbCBzdXJnZSd9LAogIF0sCiAgJzZtJzpbCiAgICB7ZmFkaW5nOidDYXN0ZSBtb2JpbGlzYXRpb24nLGZhZGluZ05vdGU6J3ByZS1lbGVjdGlvbiBwZWFrJyxyaXNpbmc6J0NvcnJ1cHRpb24gYWNjb3VudGFiaWxpdHknLHJpc2luZ05vdGU6J3Bvc3QtY3ljbGUgcHVzaCd9LAogICAge2ZhZGluZzonUmVsaWdpb3VzIG5hdGlvbmFsaXNtJyxmYWRpbmdOb3RlOidwbGF0ZWF1IHBoYXNlJyxyaXNpbmc6J0Vjb25vbWljIGFueGlldHknLHJpc2luZ05vdGU6J2Nvc3Qtb2YtbGl2aW5nJ30sCiAgICB7ZmFkaW5nOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsZmFkaW5nTm90ZToncmliYm9uLWN1dHRpbmcgZG9uZScscmlzaW5nOidMYXcgJiBvcmRlcicscmlzaW5nTm90ZTonY3JpbWUgbmFycmF0aXZlIHJpc2UnfSwKICBdLAogICcxeSc6WwogICAge2ZhZGluZzonUGFuZGVtaWMgcmVjb3ZlcnknLGZhZGluZ05vdGU6J2ZhZGVkIGVhcmx5IHllYXInLHJpc2luZzonSW5mbGF0aW9uJyxyaXNpbmdOb3RlOidkb21pbmF0ZWQgbWlkLXllYXInfSwKICAgIHtmYWRpbmc6J1JlZ2lvbmFsIGlkZW50aXR5JyxmYWRpbmdOb3RlOidsYW5ndWFnZS1sZWQgcGVhaycscmlzaW5nOidTZWN1cml0eSAmIGJvcmRlcnMnLHJpc2luZ05vdGU6J2dlb3BvbGl0aWNhbCBlc2NhbGF0aW9uJ30sCiAgICB7ZmFkaW5nOidHb3Zlcm5hbmNlIG9wdGltaXNtJyxmYWRpbmdOb3RlOidwb2xpY3kgaG9uZXltb29uIGVuZCcscmlzaW5nOidDb3JydXB0aW9uICYgc2NhbXMnLHJpc2luZ05vdGU6J2FjY291bnRhYmlsaXR5IGN5Y2xlJ30sCiAgXSwKfTsKdmFyIFJFR19TSElGVFM9WwogIHtzdGF0ZTonVGFtaWwgTmFkdScsZnJvbTonUmVnaW9uYWwgaWRlbnRpdHknLHRvOidGZWRlcmFsIHJlc291cmNlIGRpc3B1dGVzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonQmloYXInLGZyb206J0VsZWN0aW9uIHJoZXRvcmljJyx0bzonVW5lbXBsb3ltZW50ICYgZXhhbSBzY2FtcycsdGltZTonNiB3a3MnfSwKICB7c3RhdGU6J1dlc3QgQmVuZ2FsJyxmcm9tOidCeXBvbGwgcG9saXRpY3MnLHRvOidMYXcgJiBvcmRlciDCtyBCb3JkZXInLHRpbWU6JzQgd2tzJ30sCiAge3N0YXRlOidSYWphc3RoYW4nLGZyb206J0Zhcm1lciBwcm90ZXN0cycsdG86J0hlYXQgd2F2ZSDCtyBFbnZpcm9ubWVudCcsdGltZTonMiB3a3MnfSwKICB7c3RhdGU6J0thcm5hdGFrYScsZnJvbTonTWluaW5nIGNvbnRyb3ZlcnN5Jyx0bzonTGFuZ3VhZ2Ugc2lnbmFnZSBwb2xpdGljcycsdGltZTonMyB3a3MnfSwKICB7c3RhdGU6J0RlbGhpJyxmcm9tOidNZXRybyBpbmZyYXN0cnVjdHVyZScsdG86J0FpciBxdWFsaXR5IGNyaXNpcycsdGltZTonMTAgZGF5cyd9LAogIHtzdGF0ZTonTWFuaXB1cicsZnJvbTonR292ZXJuYW5jZSAmIGNhYmluZXQnLHRvOidFdGhuaWMgdGVuc2lvbnMgwrcgQUZTUEEnLHRpbWU6JzUgd2tzJ30sCiAge3N0YXRlOidQdW5qYWInLGZyb206J1Bvd2VyIGNyaXNpcycsdG86J0JvcmRlciBzZWN1cml0eSDCtyBEcm9uZXMnLHRpbWU6JzMgd2tzJ30sCl07CnZhciBNT0NLX1I9WwogIHtuYW1lOidCb3JkZXIgc2VjdXJpdHknLHN0YXRlczonSiZLIMK3IFB1bmphYiDCtyBSYWphc3RoYW4nLHBjdDonKzQxJSd9LAogIHtuYW1lOidVbmVtcGxveW1lbnQnLHN0YXRlczonQmloYXIgwrcgVVAgwrcgSmhhcmtoYW5kJyxwY3Q6JysyOCUnfSwKICB7bmFtZTonTGFuZ3VhZ2UgcG9saXRpY3MnLHN0YXRlczonVE4gwrcgS2FybmF0YWthIMK3IE1IJyxwY3Q6JysyMiUnfSwKICB7bmFtZTonRW52aXJvbm1lbnRhbCBjcmlzaXMnLHN0YXRlczonRGVsaGkgwrcgUmFqYXN0aGFuIMK3IEFQJyxwY3Q6JysxOSUnfSwKICB7bmFtZTonRXRobmljIHRlbnNpb25zJyxzdGF0ZXM6J01hbmlwdXIgwrcgQXNzYW0gwrcgV0InLHBjdDonKzE3JSd9LApdOwp2YXIgTU9DS19GPVsKICB7bmFtZTonRWxlY3Rpb24gcmhldG9yaWMnLHN0YXRlczonTmF0aW9uYWwgcG9zdC1jeWNsZScscGN0OictMzglJ30sCiAge25hbWU6J0luZmxhdGlvbiBwcmVzc3VyZScsc3RhdGVzOidFYXNpbmcgbmF0aW9uYWxseScscGN0OictMjQlJ30sCiAge25hbWU6J0Zhcm1lciBwcm90ZXN0cycsc3RhdGVzOidNb21lbnR1bSBsb3N0JyxwY3Q6Jy0xOSUnfSwKICB7bmFtZTonSW5mcmFzdHJ1Y3R1cmUgcHJpZGUnLHN0YXRlczonUmliYm9uLWN1dHRpbmcgZG9uZScscGN0OictMTQlJ30sCiAge25hbWU6J1JlbGlnaW91cyBmZXN0aXZhbHMnLHN0YXRlczonUG9zdC1zZWFzb24gZmFkZScscGN0OictMTElJ30sCl07CgpmdW5jdGlvbiByZW5kZXJTdHJpcChwZXJpb2QpewogIHZhciBkYXRhPVNISUZUU1twZXJpb2RdfHxTSElGVFNbJzNtJ107CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MPWRhdGEubWFwKGZ1bmN0aW9uKHMpewogICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDowO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo4cHg7b3ZlcmZsb3c6aGlkZGVuOyI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjZweCAxMHB4O2JvcmRlci1yaWdodDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhbGwpO21hcmdpbi1ib3R0b206M3B4OyI+ZmFkaW5nPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTtmb250LXdlaWdodDo1MDA7bGluZS1oZWlnaHQ6MS4yOyI+JytzLmZhZGluZysnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweDsiPicrcy5mYWRpbmdOb3RlKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgc3R5bGU9IndpZHRoOjI4cHg7ZmxleC1zaHJpbms6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNDU7Zm9udC1zaXplOjEzcHg7Ij7ihpI8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6OHB4IDEwcHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xNmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1yaXNlKTttYXJnaW4tYm90dG9tOjNweDsiPnJpc2luZzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NTAwO2xpbmUtaGVpZ2h0OjEuMjsiPicrcy5yaXNpbmcrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK3MucmlzaW5nTm90ZSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc3RyaXAtdGFiJykuZm9yRWFjaChmdW5jdGlvbih0YWIpewogIHRhYi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zdHJpcC10YWInKS5mb3JFYWNoKGZ1bmN0aW9uKHQpe3QuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7CiAgICB0YWIuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVuZGVyU3RyaXAodGFiLmRhdGFzZXQucGVyaW9kKTsKICB9KTsKfSk7CgpmdW5jdGlvbiByZW5kZXJNb21lbnR1bSgpewogIC8vIFJlYWQgZnJvbSBTRCAocG9wdWxhdGVkIGJ5IGZldGNoQWxsU3RhdGVzIGZyb20gbGl2ZSBBUEkpCiAgdmFyIG5jPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgbmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKICB2YXIgc29ydGVkPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICB2YXIgcmlzaW5nPXNvcnRlZC5zbGljZSgwLDUpOwogIHZhciBmYWxsaW5nPXNvcnRlZC5zbGljZSgtNSkucmV2ZXJzZSgpOwogIHZhciBteD1yaXNpbmcubGVuZ3RoP3Jpc2luZ1swXVsxXToxMDA7CgogIC8vIFdyaXRlIHRvIHJpc2luZy1saXN0IChtYXRjaGVzIG5hci1yb3cgSFRNTCkKICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogIGlmKHJFbCYmcmlzaW5nLmxlbmd0aCl7CiAgICByRWwuaW5uZXJIVE1MPXJpc2luZy5tYXAoZnVuY3Rpb24obixpKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiNlMDVhMjgiPuKGkSByaXNpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojZTA1YTI4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gZGVjbGluaW5nLWxpc3QKICB2YXIgZkVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkZWNsaW5pbmctbGlzdCcpOwogIGlmKGZFbCYmZmFsbGluZy5sZW5ndGgpewogICAgZkVsLmlubmVySFRNTD1mYWxsaW5nLm1hcChmdW5jdGlvbihuKXsKICAgICAgdmFyIHc9TWF0aC5taW4oMTAwLG5bMV0vbXgqMTAwKTsKICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjVweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuWzBdLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25bMF0uc2xpY2UoMSkrJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gV3JpdGUgdG8gcmVnaW9uYWwtbGlzdCDigJQgdG9wIHN0YXRlIHBlciByZWdpb24gZnJvbSBMSVZFCiAgdmFyIHJlZ2lvbnM9ewogICAgJ05vcnRoJzpbJ0RlbGhpJywnVXR0YXIgUHJhZGVzaCcsJ1B1bmphYicsJ0hhcnlhbmEnLCdIaW1hY2hhbCBQcmFkZXNoJywnVXR0YXJha2hhbmQnLCdKYW1tdSBhbmQgS2FzaG1pciddLAogICAgJ0Vhc3QnOlsnV2VzdCBCZW5nYWwnLCdCaWhhcicsJ0poYXJraGFuZCcsJ09kaXNoYSddLAogICAgJ1dlc3QnOlsnTWFoYXJhc2h0cmEnLCdHdWphcmF0JywnUmFqYXN0aGFuJywnR29hJ10sCiAgICAnU291dGgnOlsnVGFtaWwgTmFkdScsJ0thcm5hdGFrYScsJ0tlcmFsYScsJ0FuZGhyYSBQcmFkZXNoJywnVGVsYW5nYW5hJ10sCiAgICAnTkUnOlsnQXNzYW0nLCdNYW5pcHVyJywnTmFnYWxhbmQnLCdNaXpvcmFtJywnTWVnaGFsYXlhJywnVHJpcHVyYScsJ0FydW5hY2hhbCBQcmFkZXNoJywnU2lra2ltJ10sCiAgICAnQ2VudHJhbCc6WydNYWRoeWEgUHJhZGVzaCcsJ0NoaGF0dGlzZ2FyaCddLAogIH07CiAgdmFyIGdFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVnaW9uYWwtbGlzdCcpOwogIGlmKGdFbCl7CiAgICB2YXIgcmVnSXRlbXM9T2JqZWN0LmVudHJpZXMocmVnaW9ucykubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgdmFyIHJlZ2lvbj1rdlswXSxzdGF0ZXM9a3ZbMV07CiAgICAgIHZhciB0b3A9c3RhdGVzLm1hcChmdW5jdGlvbihzKXtyZXR1cm4ge25hbWU6cyxhdHQ6KExJVkVbc10mJkxJVkVbc10uYXR0ZW50aW9uKXx8MH07fSkKICAgICAgICAuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiLmF0dC1hLmF0dDt9KVswXTsKICAgICAgaWYoIXRvcHx8IXRvcC5hdHQpIHJldHVybiBudWxsOwogICAgICB2YXIgbmFyPShMSVZFW3RvcC5uYW1lXSYmTElWRVt0b3AubmFtZV0uZG9taW5hbnRfbmFycmF0aXZlKXx8J+KAlCc7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzo4cHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmJhc2VsaW5lO21hcmdpbi1ib3R0b206MnB4OyI+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjEyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+JytyZWdpb24rJzwvc3Bhbj4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1hY2NlbnQpIj4nK3RvcC5hdHQudG9GaXhlZCgxKSsnPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+Jyt0b3AubmFtZSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoxcHg7Ij4nK25hcisnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5maWx0ZXIoQm9vbGVhbikuam9pbignJyk7CiAgICBpZihyZWdJdGVtcykgZ0VsLmlubmVySFRNTD1yZWdJdGVtczsKICB9Cn0KCgovLyBTVEFURSBEQVRBCnZhciBTRD17fTsKCnZhciBMSVZFPXt9OwpmdW5jdGlvbiBub3JtYWxpemVFbW90aW9ucyhlKXtpZighZXx8IU9iamVjdC5rZXlzKGUpLmxlbmd0aClyZXR1cm57fTt2YXIgdmFscz1PYmplY3QudmFsdWVzKGUpLHRvdD12YWxzLnJlZHVjZShmdW5jdGlvbihzLHYpe3JldHVybiBzK3Y7fSwwKTtpZih0b3Q8PTApcmV0dXJue307aWYodG90PD0xLjAxKXt2YXIgb3V0PXt9O09iamVjdC5rZXlzKGUpLmZvckVhY2goZnVuY3Rpb24oayl7b3V0W2tdPU1hdGgucm91bmQoZVtrXSoxMDApO30pO3JldHVybiBvdXQ7fXJldHVybiBlO30KZnVuY3Rpb24gZG9taW5hbnRFbW90aW9uKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybiBudWxsO3ZhciBteD0wLGRvbT1udWxsO09iamVjdC5lbnRyaWVzKGUpLmZvckVhY2goZnVuY3Rpb24oa3Ype2lmKGt2WzFdPm14KXtteD1rdlsxXTtkb209a3ZbMF07fX0pO3JldHVybiBkb207fQpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZighZWwpcmV0dXJuO2VsLnRleHRDb250ZW50PXZhbDtpZih2YWwmJnZhbCE9PSctJyl7ZWwuY2xhc3NMaXN0LnJlbW92ZSgnbG9hZGluZycpO319Cgp2YXIgREVGQVVMVD17CiAgYXR0ZW50aW9uOjAsZGVsdGE6MCx2ZWxvY2l0eTowLAogIGVtb3Rpb25zOnt9LGRvbWluYW50X2Vtb3Rpb246bnVsbCxkb21pbmFudF9uYXJyYXRpdmU6bnVsbCwKICBuYXJyYXRpdmVzOltdLHJpc2luZzpbXSxmYWxsaW5nOltdLAogIHN1bW1hcnk6JycsYXJ0aWNsZXM6W10sdGltZWxpbmU6W10sCiAgbmFycmF0aXZlSGlzdG9yeTpbXSxzaWduYWxfY291bnQ6MCwKfTsKCmZ1bmN0aW9uIGcobil7cmV0dXJuIFNEW25dfHxPYmplY3QuYXNzaWduKHt9LERFRkFVTFQpO30KCmZ1bmN0aW9uIGFDKHMpewogIC8vIER5bmFtaWMgc2NhbGU6IGFsd2F5cyBzcHJlYWQgZnVsbCBjb2xvciByYW5nZSBhY3Jvc3MgYWN0dWFsIGRhdGEKICAvLyBHZXQgbWluL21heCBmcm9tIGN1cnJlbnQgU0QgdG8gbm9ybWFsaXplCiAgdmFyIHNjb3Jlcz1PYmplY3QudmFsdWVzKFNEKS5tYXAoZnVuY3Rpb24oZCl7cmV0dXJuIGQuYXR0ZW50aW9ufHwwO30pOwogIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3Jlcyk7CiAgdmFyIG14PU1hdGgubWF4LmFwcGx5KG51bGwsc2NvcmVzKXx8MTsKICAvLyBOb3JtYWxpemUgMC0xCiAgdmFyIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzLW1uKS8obXgtbW4pKSk7CiAgLy8gTWFwIHRvIGNvbG9yIHN0b3BzOiBkYXJrIGJsdWUg4oaSIHRlYWwg4oaSIGFtYmVyIOKGkiBvcmFuZ2Ug4oaSIHJlZAogIGlmKG48MC4xMikgcmV0dXJuICcjMGQxZTMwJzsKICBpZihuPDAuMjUpIHJldHVybiAnIzBlM2Q2YSc7CiAgaWYobjwwLjM4KSByZXR1cm4gJyMwZDVmOTAnOwogIGlmKG48MC41MCkgcmV0dXJuICcjMGU3YWFhJzsKICBpZihuPDAuNjIpIHJldHVybiAnIzFhOTA5MCc7CiAgaWYobjwwLjcyKSByZXR1cm4gJyNjODcwMTAnOwogIGlmKG48MC44MikgcmV0dXJuICcjZDg0MDEwJzsKICBpZihuPDAuOTIpIHJldHVybiAnI2NjMTgwOCc7CiAgcmV0dXJuICcjZmYwMDEwJzsKfQpmdW5jdGlvbiBlQyhlKXsKICB2YXIgbXg9MCxkb209J3ByaWRlJzsKICBmb3IodmFyIGsgaW4gZSl7aWYoZVtrXT5teCl7bXg9ZVtrXTtkb209azt9fQogIHJldHVybiAoe2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9KVtkb21dfHwnIzMzYWFjYyc7Cn0KZnVuY3Rpb24gdkModil7CiAgaWYodj4wLjIpIHJldHVybiAnI2RjMDgxOCc7CiAgaWYodj4wLjEpIHJldHVybiAnI2UwNWEyOCc7CiAgaWYodj4wLjAyKSByZXR1cm4gJyNjYzg4MjInOwogIGlmKHY8LTAuMDUpIHJldHVybiAnIzIyOTliYic7CiAgcmV0dXJuICcjMTUyMDMwJzsKfQoKdmFyIGxheWVyPSdhdHRlbnRpb24nLFNFTD1udWxsLEZBVlM9bmV3IFNldCgpOwoKLy8gTUFQCmZ1bmN0aW9uIHByb2pfKHcsaCxwYWQpewogIHBhZD1wYWR8fDIwOwogIHZhciBtaW5Mb249NjguMSxtYXhMb249OTcuNCxtaW5MYXQ9Ni41LG1heExhdD0zNy4xOwogIHZhciBzY1g9KHctcGFkKjIpLyhtYXhMb24tbWluTG9uKTsKICB2YXIgc2NZPShoLXBhZCoyKS8obWF4TGF0LW1pbkxhdCk7CiAgdmFyIHNjPU1hdGgubWluKHNjWCxzY1kpOwogIHZhciBveD1wYWQrKHctcGFkKjItKG1heExvbi1taW5Mb24pKnNjKS8yOwogIHZhciBveT1wYWQrKGgtcGFkKjItKG1heExhdC1taW5MYXQpKnNjKS8yOwogIHJldHVybiBmdW5jdGlvbihsb24sbGF0KXtyZXR1cm4gW294Kyhsb24tbWluTG9uKSpzYywgb3krKG1heExhdC1sYXQpKnNjXTt9Owp9CmZ1bmN0aW9uIGdlbzJwYXRoKGdlb20scGopewogIHZhciBkPScnOwogIGZ1bmN0aW9uIHJpbmcoY3Mpe3ZhciBzPScnO2NzLmZvckVhY2goZnVuY3Rpb24oYyxpKXt2YXIgcD1waihjWzBdLGNbMV0pO3MrPShpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KTtyZXR1cm4gcysnWic7fQogIGlmKGdlb20udHlwZT09PSdQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7CiAgZWxzZSBpZihnZW9tLnR5cGU9PT0nTXVsdGlQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHApe3AuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pO30pOwogIHJldHVybiBkOwp9CmZ1bmN0aW9uIGN0cihnZW9tKXsKICB2YXIgcHRzPVtdOwogIGZ1bmN0aW9uIGNvbChjKXtpZih0eXBlb2YgY1swXT09PSdudW1iZXInKSBwdHMucHVzaChjKTtlbHNlIGMuZm9yRWFjaChjb2wpO30KICBjb2woZ2VvbS5jb29yZGluYXRlcyk7CiAgaWYoIXB0cy5sZW5ndGgpIHJldHVybiBbMCwwXTsKICByZXR1cm4gW3B0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzBdO30sMCkvcHRzLmxlbmd0aCxwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFsxXTt9LDApL3B0cy5sZW5ndGhdOwp9CmZ1bmN0aW9uIHNOYW1lKHByb3BzKXsKICB2YXIgcmF3PXByb3BzLnN0X25tfHxwcm9wcy5OQU1FXzF8fHByb3BzLm5hbWV8fHByb3BzLk5BTUV8fCcnOwogIHZhciBtYXA9eydMYWRha2gnOidKYW1tdSBhbmQgS2FzaG1pcicsJ0phbW11ICYgS2FzaG1pcic6J0phbW11IGFuZCBLYXNobWlyJywnVXR0YXJhbmNoYWwnOidVdHRhcmFraGFuZCcsJ0FuZGFtYW4gYW5kIE5pY29iYXInOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdBbmRhbWFuICYgTmljb2JhciBJc2xhbmQnOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdOQ1Qgb2YgRGVsaGknOidEZWxoaScsJ1BvbmRpY2hlcnJ5JzonUHVkdWNoZXJyeScsJ0RhZHJhIGFuZCBOYWdhciBIYXZlbGknOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1JywnRGFtYW4gYW5kIERpdSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnfTsKICByZXR1cm4gbWFwW3Jhd118fHJhdzsKfQoKdmFyIGNhY2hlZEdlbz1udWxsOwoKYXN5bmMgZnVuY3Rpb24gbG9hZE1hcChhdHRlbXB0KXsKICBhdHRlbXB0ID0gYXR0ZW1wdHx8MTsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaCgnaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L2doL3VkaXQtMDAxL2luZGlhLW1hcHMtZGF0YUBtYXN0ZXIvdG9wb2pzb24vaW5kaWEuanNvbicpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciB0b3BvPWF3YWl0IHIuanNvbigpOwogICAgY2FjaGVkR2VvPXRvcG9qc29uLmZlYXR1cmUodG9wbyx0b3BvLm9iamVjdHMuc3RhdGVzKTsKICAgIHJlbmRlck1hcChjYWNoZWRHZW8pOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDEwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDMwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDYwMDApOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW21hcF0gbG9hZCBmYWlsZWQgYXR0ZW1wdCAnK2F0dGVtcHQrJzonLGUubWVzc2FnZSk7CiAgICBpZihhdHRlbXB0PDUpewogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7bG9hZE1hcChhdHRlbXB0KzEpO30sIGF0dGVtcHQqMjAwMCk7CiAgICB9IGVsc2UgewogICAgICB2YXIgbWk9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpOwogICAgICBpZihtaSkgbWkuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjojMmEzYTRhO3BhZGRpbmc6NDBweDt0ZXh0LWFsaWduOmNlbnRlcjtmb250LWZhbWlseTptb25vc3BhY2U7Zm9udC1zaXplOjExcHgiPk1hcCB1bmF2YWlsYWJsZSDigJQgcmVmcmVzaCB0byByZXRyeTwvZGl2Pic7CiAgICB9CiAgfQp9CgpmdW5jdGlvbiByZW5kZXJNYXAoc3RhdGVzKXsKICB2YXIgdz04MDAsaD04MDAscGo9cHJval8odyxoLDI4KTsKICB2YXIgc2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1zdGF0ZXMnKTsKICB2YXIgcGc9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1wdWxzZXMnKTsKICB2YXIgZ2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1nbG93Jyk7CiAgc2cuaW5uZXJIVE1MPScnO3BnLmlubmVySFRNTD0nJztnZy5pbm5lckhUTUw9Jyc7CgogIHN0YXRlcy5mZWF0dXJlcy5mb3JFYWNoKGZ1bmN0aW9uKGYpewogICAgaWYoIWYuZ2VvbWV0cnkpIHJldHVybjsKICAgIHZhciBubT1zTmFtZShmLnByb3BlcnRpZXMpLGQ9ZyhubSk7CiAgICB2YXIgcGF0aEVsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdwYXRoJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkJyxnZW8ycGF0aChmLmdlb21ldHJ5LHBqKSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3N0YXRlJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnLG5tKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsJ3JnYmEoMjU1LDI1NSwyNTUsMC4wNyknKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzAuNScpOwogICAgc2cuYXBwZW5kQ2hpbGQocGF0aEVsKTsKCiAgICB2YXIgY3Q9Y3RyKGYuZ2VvbWV0cnkpLGNwPXBqKGN0WzBdLGN0WzFdKTsKCiAgICAvLyBBdG1vc3BoZXJpYyBnbG93IGZvciBoaWdoLWF0dGVudGlvbiBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj02NSl7CiAgICAgIHZhciBnbG93RWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2VsbGlwc2UnKTsKICAgICAgdmFyIGdsb3dSPU1hdGgubWluKDYwLDIwK2QuYXR0ZW50aW9uKjAuNSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7Z2xvd0VsLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgncngnLGdsb3dSKTtnbG93RWwuc2V0QXR0cmlidXRlKCdyeScsZ2xvd1IqMC43KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnb3BhY2l0eScsJzAuMDgnKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsdGVyJywndXJsKCNzdGF0ZUdsb3cpJyk7CiAgICAgIGdsb3dFbC5zdHlsZS5hbmltYXRpb249J2dsb3dQdWxzZSAnKygyLjUrTWF0aC5yYW5kb20oKSkrJ3MgZWFzZS1pbi1vdXQgJysoTWF0aC5yYW5kb20oKSoyKSsncyBpbmZpbml0ZSc7CiAgICAgIGdnLmFwcGVuZENoaWxkKGdsb3dFbCk7CiAgICB9CgogICAgLy8gRHVhbCBwdWxzZSByaW5ncyBmb3IgdmVyeSBob3Qgc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NzIpewogICAgICBbMCwxXS5mb3JFYWNoKGZ1bmN0aW9uKGkpewogICAgICAgIHZhciByaW5nPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdjaXJjbGUnKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtyaW5nLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdwdWxzZS1yaW5nIHAnKyhpKzEpKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJyxhQyhkLmF0dGVudGlvbikpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcxJyk7CiAgICAgICAgcmluZy5zdHlsZS5hbmltYXRpb25EZWxheT0oTWF0aC5yYW5kb20oKSoyLjUpKydzJzsKICAgICAgICBwZy5hcHBlbmRDaGlsZChyaW5nKTsKICAgICAgfSk7CiAgICB9CiAgfSk7CiAgYXBwbHlMYXllcigpOwogIGF0dGFjaEludGVyYWN0aW9ucygpOwp9CgovLyBTaW5nbGUgc291cmNlIG9mIHRydXRoIGZvciBlbW90aW9uIGNvbG9yCi8vIEJvdGggbWFwIGFuZCBwYW5lbCBjYWxsIHRoaXMg4oCUIGd1YXJhbnRlZXMgdGhleSBhbHdheXMgbWF0Y2gKZnVuY3Rpb24gZ2V0RWZmZWN0aXZlRW1vdGlvbihubSl7CiAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogIHZhciBkPVNEW25tXXx8e307CiAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICAvLyAxLiBUcnkgTElWRS5kb21pbmFudF9lbW90aW9uIChzZXQgYnkgL2FwaS9zdGF0ZXMpCiAgdmFyIGRvbT1saXZlLmRvbWluYW50X2Vtb3Rpb258fGQuZG9taW5hbnRfZW1vdGlvbjsKCiAgLy8gMi4gVHJ5IGNvbXB1dGluZyBmcm9tIGVtb3Rpb25zIGJyZWFrZG93bgogIGlmKCFkb20pewogICAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICAgIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyk7CiAgfQoKICAvLyAzLiBGYWxsYmFjazogaW5mZXIgZnJvbSBkb21pbmFudCBuYXJyYXRpdmUgKHNhbWUgbG9naWMgZXZlcnl3aGVyZSkKICBpZighZG9tKXsKICAgIHZhciBucD0obGl2ZS5kb21pbmFudF9uYXJyYXRpdmV8fGQuZG9taW5hbnRfbmFycmF0aXZlfHwnJykudG9Mb3dlckNhc2UoKTsKICAgIGlmKG5wLm1hdGNoKC9ib3JkZXJ8dGVycm9yfHNlY3VyaXR5fGNvbmZsaWN0fGF0dGFja3x3YXJ8aW5maWx0cmF0LykpIGRvbT0nZmVhcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9zY2FtfGNvcnJ1cHR8cHJvdGVzdHxhcnJlc3R8dmlvbGVuY2V8b3V0cmFnZXxjcmltZS8pKSBkb209J2FuZ2VyJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2RldmVsb3B8aW52ZXN0fGdyb3d0aHxsYXVuY2h8aW5hdWd1cnxyZWZvcm18cHJvZ3Jlc3N8Ym9vc3QvKSkgZG9tPSdob3BlJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2N1bHR1cmV8aGVyaXRhZ2V8cHJpZGV8dmljdG9yeXxjZWxlYnJhdHxtZWRhbHxhY2hpZXZlbWVudC8pKSBkb209J3ByaWRlJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2Zsb29kfGRyb3VnaHR8dW5lbXBsb3ltZW50fGluZmxhdGlvbnxzaG9ydGFnZXxjcmlzaXN8Y29uY2Vybi8pKSBkb209J2FueGlldHknOwogICAgZWxzZSBpZigobGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwKT41KSBkb209J2FueGlldHknOyAvLyBhY3RpdmUgc3RhdGUgZGVmYXVsdAogICAgZWxzZSBkb209J2FueGlldHknOyAvLyBnbG9iYWwgZGVmYXVsdAogIH0KCiAgcmV0dXJuIGRvbTsKfQoKLy8gR2V0IGVzdGltYXRlZCBlbW90aW9uIGJyZWFrZG93biAoZm9yIHBhbmVsIGRvbnV0IHdoZW4gcmVhbCBkYXRhIG1pc3NpbmcpCmZ1bmN0aW9uIGdldEVtb3Rpb25CcmVha2Rvd24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgaWYoT2JqZWN0LmtleXMoZW1vcykubGVuZ3RoKSByZXR1cm4ge2Vtb3Rpb25zOmVtb3MsZXN0aW1hdGVkOmZhbHNlfTsKICAvLyBCdWlsZCBza2V3ZWQgZGlzdHJpYnV0aW9uIGZyb20gZWZmZWN0aXZlIGVtb3Rpb24KICB2YXIgZG9tPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogIHZhciBiYXNlPXthbnhpZXR5OjEzLGFuZ2VyOjEzLGhvcGU6MTMscHJpZGU6MTMsZmVhcjoxM307CiAgYmFzZVtkb21dPTQ4OwogIHJldHVybiB7ZW1vdGlvbnM6YmFzZSxlc3RpbWF0ZWQ6dHJ1ZX07Cn0KCmZ1bmN0aW9uIGFwcGx5TGF5ZXIoKXsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLGZpbGw7CiAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKSBmaWxsPWFDKGQuYXR0ZW50aW9uKTsKICAgIGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBkZT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgZmlsbD1lTWFwW2RlXXx8JyMzMzQ0NTUnOwogICAgfQogICAgZWxzZSBmaWxsPXZDKGQudmVsb2NpdHkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGZpbGwpOwogICAgKGZ1bmN0aW9uKCl7CiAgICAgIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiB4LmF0dGVudGlvbnx8MDt9KTsKICAgICAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKSxteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgICAgIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwoZC5hdHRlbnRpb24tbW4pLyhteC1tbikpKTsKICAgICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsbGF5ZXI9PT0nYXR0ZW50aW9uJz9NYXRoLm1heCgwLjMsMC4zK24qMC43KTowLjg1KTsKICAgIH0pKCk7CiAgfSk7Cn0KCmZ1bmN0aW9uIGF0dGFjaEludGVyYWN0aW9ucygpewogIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpewogICAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpOwogICAgICB2YXIgZD1nKG5tKTsKICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgICAgIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgbGF0ZXN0PScnOwogICAgICBpZihkLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlcy5sZW5ndGgpIGxhdGVzdD1kLm5hcnJhdGl2ZXNbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkLm5hcnJhdGl2ZXNbMF0ubmFtZS5zbGljZSgxKTsKICAgICAgZWxzZSBpZihsaXZlLmRvbWluYW50X25hcnJhdGl2ZSkgbGF0ZXN0PWxpdmUuZG9taW5hbnRfbmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2xpdmUuZG9taW5hbnRfbmFycmF0aXZlLnNsaWNlKDEpOwoKICAgICAgdmFyIHJvd3M9Jyc7CiAgICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgICAgIHZhciBhdHQ9bGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwOwogICAgICAgIHZhciBkbHQ9bGl2ZS5kZWx0YXx8ZC5kZWx0YXx8MDsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5BdHRlbnRpb248L3NwYW4+PHN0cm9uZz4nK2F0dC50b0ZpeGVkKDEpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgKGRsdCE9PTA/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaGlmdDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonKyhkbHQ+MD8nI2UwNWEyOCc6JyMzYmI4ZDgnKSsnIj4nKyhkbHQ+MD8nKyc6JycpK2RsdCsnPC9zdHJvbmc+PC9kaXY+JzonJykrCiAgICAgICAgICAobGF0ZXN0Pyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Ub3AgbmFycmF0aXZlPC9zcGFuPjxzdHJvbmc+JytsYXRlc3QrJzwvc3Ryb25nPjwvZGl2Pic6JycpOwogICAgICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBkb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgICAgaWYoZG9tRW1vKXsKICAgICAgICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOmQuZW1vdGlvbnN8fHt9OwogICAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RG9taW5hbnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JytwYWxbZG9tRW1vXSsnIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgICAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vcykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICAgICAgICB2YXIgdG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgICAgICAgaWYodG90PjAmJnRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7fQogICAgICAgICAgcm93cys9ZUwuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4gc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweCI+PHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytwYWxba3ZbMF1dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nK2t2WzBdKyc8L3NwYW4+PHN0cm9uZz4nK01hdGgucm91bmQoa3ZbMV0qMTAwL01hdGgubWF4KDEsdG90KSkrJyU8L3N0cm9uZz48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICAgIH0KICAgICAgfSBlbHNlIHsKICAgICAgICB2YXIgdmVsPWxpdmUudmVsb2NpdHl8fGQudmVsb2NpdHl8fDA7CiAgICAgICAgdmFyIHZlbERpcj12ZWw+MC4xPydSaXNpbmcgZmFzdCc6dmVsPjAuMDI/J1Jpc2luZyc6dmVsPC0wLjA1PydDb29saW5nJzonU3RhYmxlJzsKICAgICAgICB2YXIgdmVsQ29sPXZlbD4wLjAyPycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Nb21lbnR1bTwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRpcmVjdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nK3ZlbERpcisnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfQoKICAgICAgdGlwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0idHQtbiI+JytubSsnPC9kaXY+Jytyb3dzKyhsYXRlc3QmJmxheWVyIT09J2F0dGVudGlvbic/JzxkaXYgY2xhc3M9InR0LW5hciI+PHN0cm9uZz5OYXJyYXRpdmU8L3N0cm9uZz4nK2xhdGVzdCsnPC9kaXY+JzonJyk7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE5MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTUwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0nMSc7CiAgICB9KTsKcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJyxmdW5jdGlvbigpe3RpcC5zdHlsZS5vcGFjaXR5PTA7fSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe3NlbGVjdF8ocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8gU1RBVEUgUEFORUwKZnVuY3Rpb24gc2VsZWN0XyhubSl7CiAgU0VMPW5tOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmNsYXNzTGlzdC50b2dnbGUoJ3NlbGVjdGVkJyxwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk9PT1ubSk7CiAgfSk7CiAgLy8gU2hvdyBsb2FkaW5nIHN0YXRlIGltbWVkaWF0ZWx5IHdpdGggd2hhdGV2ZXIgTElWRSBkYXRhIHdlIGhhdmUKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKHBhbmVsKXsKICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgIHBhbmVsLmlubmVySFRNTD0KICAgICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICAgJzxkaXY+PGRpdiBjbGFzcz0ic3AtZWsiPicrKGxheWVyPT09J2F0dGVudGlvbic/J05hcnJhdGl2ZSBwYW5lbCc6bGF5ZXI9PT0nZW1vdGlvbic/J0Vtb3Rpb25hbCByZWdpc3Rlcic6J01vbWVudHVtIHBhbmVsJykrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoRkFWUy5oYXMobm0pPydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgICAnPC9idXR0b24+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDhlbSI+JysKICAgICAgICAnTG9hZGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKwogICAgICAgIChsaXZlLmF0dGVudGlvbj8nPGJyPjxicj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxOHB4O2NvbG9yOnZhcigtLWluaykiPkF0dGVudGlvbiAnK2xpdmUuYXR0ZW50aW9uLnRvRml4ZWQoMSkrJzwvc3Bhbj4nOicnKSsKICAgICAgICAobGl2ZS5kb21pbmFudF9lbW90aW9uPyc8YnI+PHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JytsaXZlLmRvbWluYW50X2Vtb3Rpb24rJyBzaWduYWwgZG9taW5hbnQ8L3NwYW4+JzonJykrCiAgICAgICc8L2Rpdj4nOwogIH0KICAvLyBGZXRjaCBmdWxsIGRldGFpbCB0aGVuIHJlbmRlcgogIGZldGNoRGV0YWlsKG5tKS50aGVuKGZ1bmN0aW9uKCl7CiAgICBpZihTRUw9PT1ubSl7CiAgICAgIHJlbmRlclBhbmVsKG5tKTsKICAgICAgLy8gVXBkYXRlIGp1c3QgdGhpcyBzdGF0ZSdzIG1hcCBjb2xvciB0byBtYXRjaCB0aGUgcGFuZWwKICAgICAgdmFyIHBhdGg9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJytubSsnIl0nKTsKICAgICAgaWYocGF0aCYmbGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgdmFyIGRvbT1saXZlLmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihsaXZlLmVtb3Rpb25zfHx7fSk7CiAgICAgICAgaWYoZG9tJiZlTWFwW2RvbV0pIHBhdGguc2V0QXR0cmlidXRlKCdmaWxsJyxlTWFwW2RvbV0pOwogICAgICB9IGVsc2UgewogICAgICAgIGFwcGx5TGF5ZXIoKTsKICAgICAgfQogICAgfQogIH0pLmNhdGNoKGZ1bmN0aW9uKGUpewogICAgY29uc29sZS53YXJuKCdbc2VsZWN0XScsZSk7CiAgICBpZihTRUw9PT1ubSkgcmVuZGVyUGFuZWwobm0pOwogIH0pOwp9CgpmdW5jdGlvbiByZW5kZXJQYW5lbChubSl7CiAgdmFyIGQ9ZyhubSk7CiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZighcGFuZWwpIHJldHVybjsKICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgdmFyIGhlYWRlcj0KICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAnPGRpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1layIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsiPicrCiAgICAgICAgICAobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsKICAgICAgICAgIChkLmNvbmZpZGVuY2U/JzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07cGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6JysoZC5jb25maWRlbmNlPT09J0hJR0gnPydyZ2JhKDUxLDIwNCwxMDIsMC4xKSc6ZC5jb25maWRlbmNlPT09J01FRElVTSc/J3JnYmEoMjI0LDkwLDQwLDAuMSknOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpJykrJztjb2xvcjonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/JyMzM2NjNjYnOmQuY29uZmlkZW5jZT09PSdNRURJVU0nPycjZTA1YTI4JzoncmdiYSgyNTUsMjU1LDI1NSwwLjMpJykrJyI+JytkLmNvbmZpZGVuY2UrJyBTSUdOQUw8L3NwYW4+JzonJykrCiAgICAgICAgICAoZC5pc19yZWdpb25hbF9zdG9yeT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKTtjb2xvcjojM2JiOGQ4Ij5SRUdJT05BTCBTUElLRTwvc3Bhbj4nOicnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAnPC9idXR0b24+JysKICAgICc8L2Rpdj4nOwoKICB2YXIgYm9keT0nJzsKCiAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICB2YXIgZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8ndXAnOidkbic7CiAgICB2YXIgbmFycj1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHRsPShkLnRpbWVsaW5lJiZkLnRpbWVsaW5lLmxlbmd0aCk/ZC50aW1lbGluZTpbMCwwLDAsMCwwLDAsMCxkLmF0dGVudGlvbnx8MF07CiAgICB2YXIgdG1uPU1hdGgubWluLmFwcGx5KG51bGwsdGwpLHRteD1NYXRoLm1heC5hcHBseShudWxsLHRsKSx0cj1NYXRoLm1heCgxLHRteC10bW4pOwogICAgdmFyIHR3PTI2MCx0aD02Mix0cD01OwogICAgdmFyIHB0cz10bC5tYXAoZnVuY3Rpb24odixpKXtyZXR1cm5bdHArKGkvKHRsLmxlbmd0aC0xKSkqKHR3LXRwKjIpLHRwKygxLSh2LXRtbikvdHIpKih0aC10cCoyKV07fSk7CiAgICB2YXIgcEQ9cHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybihpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KS5qb2luKCcnKTsKICAgIHZhciBhRD1wRCsnIEwnK3B0c1twdHMubGVuZ3RoLTFdWzBdKycsJysodGgtdHApKycgTCcrcHRzWzBdWzBdKycsJysodGgtdHApKycgWic7CiAgICB2YXIgYWM9YUMoZC5hdHRlbnRpb258fDApOwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdIb3cgaW50ZW5zZWx5ICcrKG5tLnNwbGl0KCcgJylbMF0pKycgaXMgYmVpbmcgZGlzY3Vzc2VkIG5hdGlvbmFsbHkuIFNjb3JlIG9mICcrZC5hdHRlbnRpb24rJyBtZWFucyAnKyhkLmF0dGVudGlvbj42MD8ndmVyeSBoaWdoIOKAlCB0aGlzIHN0YXRlIGRvbWluYXRlcyBuYXRpb25hbCBkaXNjb3Vyc2UnOmQuYXR0ZW50aW9uPjM1PydlbGV2YXRlZCDigJQgY2xlYXJseSBpbiB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uJzpkLmF0dGVudGlvbj4xNT8nbW9kZXJhdGUg4oCUIHNvbWUgbmF0aW9uYWwgY292ZXJhZ2UnOmQuYXR0ZW50aW9uPjU/J2xvdyDigJQgbGltaXRlZCBuYXRpb25hbCBhdHRlbnRpb24nOidtaW5pbWFsIOKAlCBmZXcgc2lnbmFscyBkZXRlY3RlZCcpKycuJysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9Imluc2lnaHQiIHN0eWxlPSInKyhkLmNvbmZpZGVuY2U9PT0iTE9XIj8nYm9yZGVyLWNvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4wNik7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljJzonJykrJyI+JysoKGQuY29uZmlkZW5jZT09PSJMT1ciJiYhZC5zdW1tYXJ5KT8nTGltaXRlZCBzaWduYWxzIGRldGVjdGVkIGZvciAnK25tKycuIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy4nOmQuc3VtbWFyeXx8J0NvbGxlY3Rpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJykrJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKG5hcnJbMF0/bmFyclswXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25hcnJbMF0ubmFtZS5zbGljZSgxKTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAobmFyci5sZW5ndGg/CiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrbmFyci5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHZhciBubj1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPjxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25uKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4Ij7ihpE8L3NwYW4+JzpuLmRpcj09PSdkb3duJz8nIDxzcGFuIHN0eWxlPSJjb2xvcjojM2JiOGQ4O2ZvbnQtc2l6ZTo5cHgiPuKGkzwvc3Bhbj4nOicnKSsnPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdmFsIj4nK3ZhbC50b0ZpeGVkKDEpKyclPC9kaXY+JysKICAgICAgICAgICAgICAnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCx2YWwqMi41KSsnJTtiYWNrZ3JvdW5kOicrKG4uZGlyPT09J3VwJz8nI2UwNWEyOCc6bi5kaXI9PT0nZG93bic/JyMzYmI4ZDgnOicjMzM0NDU1JykrJyI+PC9kaXY+PC9kaXY+PC9kaXY+JzsKICAgICAgICAgIH0pLmpvaW4oJycpKyc8L2Rpdj4nOgogICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb3ctc2lnbmFsIHJlZ2lvbi4gTW9uaXRvcmluZyByZWdpb25hbCBzb3VyY2VzLjwvZGl2PicpKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QXR0ZW50aW9uIOKAlCA4IGRheXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJ0bC13cmFwIj48c3ZnIHZpZXdCb3g9IjAgMCAnK3R3KycgJyt0aCsnIiBzdHlsZT0id2lkdGg6MTAwJTtoZWlnaHQ6MTAwJSI+JysKICAgICAgICAgICc8ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9InRsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnIiB4MT0iMCIgeDI9IjAiIHkxPSIwIiB5Mj0iMSI+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IicrYWMrJyIgc3RvcC1vcGFjaXR5PSIwLjI1Ii8+JysKICAgICAgICAgICAgJzxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAiLz4nKwogICAgICAgICAgJzwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPicrCiAgICAgICAgICAnPHBhdGggZD0iJythRCsnIiBmaWxsPSJ1cmwoI3RsZycrbm0ucmVwbGFjZSgvW15hLXpdL2dpLCcnKSsnKSIgLz4nKwogICAgICAgICAgJzxwYXRoIGQ9IicrcEQrJyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSInK2FjKyciIHN0cm9rZS13aWR0aD0iMS4yIi8+JysKICAgICAgICAgIHB0cy5tYXAoZnVuY3Rpb24ocCxpKXtyZXR1cm4gJzxjaXJjbGUgY3g9IicrcFswXSsnIiBjeT0iJytwWzFdKyciIHI9IicrKGk9PT1wdHMubGVuZ3RoLTE/Mi4yOjEuMikrJyIgZmlsbD0iJythYysnIi8+Jzt9KS5qb2luKCcnKSsKICAgICAgICAnPC9zdmc+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbHMgPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JysoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGg/ZC5hcnRpY2xlcy5sZW5ndGg6MCkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJhcnQtbGlzdCI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLm1hcChmdW5jdGlvbihhKXtyZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIj48ZGl2IGNsYXNzPSJhcnQtc3JjIj4nKyhhLnNyY3x8JycpKyc8L2Rpdj48ZGl2IGNsYXNzPSJhcnQtdHh0Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOgogICAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6NnB4IDAiPk5vIHNpZ25hbHMgY29sbGVjdGVkIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSBpZihsYXllcj09PSdlbW90aW9uJyl7CiAgICAvLyBVc2Ugc2FtZSBmdW5jdGlvbnMgYXMgbWFwIOKAlCBndWFyYW50ZWVkIHRvIG1hdGNoCiAgICB2YXIgbWFwRG9tRW1vPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogICAgdmFyIGJyZWFrZG93bj1nZXRFbW90aW9uQnJlYWtkb3duKG5tKTsKICAgIHZhciBlbW90aW9ucz1icmVha2Rvd24uZW1vdGlvbnM7CiAgICB2YXIgaGFzRW1vcz0hYnJlYWtkb3duLmVzdGltYXRlZDsKICAgIHZhciBlTD1PYmplY3QuZW50cmllcyhlbW90aW9ucyk7CiAgICB2YXIgZVRvdD1lTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKTsKICAgIGlmKGVUb3Q+MCYmZVRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt9CiAgICB2YXIgdG90PU1hdGgubWF4KDEsZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCkpOwogICAgZUwuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICBpZighZUwubGVuZ3RoKXtwYW5lbC5pbm5lckhUTUw9aGVhZGVyKyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4Ij5ObyBlbW90aW9uIGRhdGEgeWV0LjwvZGl2Pic7cmV0dXJuO30KICAgIC8vIGRvbUVtbyA9IHNhbWUgYXMgbWFwIGNvbG9yIChmcm9tIGdldEVmZmVjdGl2ZUVtb3Rpb24pCiAgICB2YXIgZG9tRW1vPW1hcERvbUVtbzsKICAgIC8vIFJlb3JkZXIgZUwgc28gZG9taW5hbnQgc2hvd3MgZmlyc3QKICAgIGVMLnNvcnQoZnVuY3Rpb24oYSxiKXsKICAgICAgaWYoYVswXT09PWRvbUVtbykgcmV0dXJuIC0xOwogICAgICBpZihiWzBdPT09ZG9tRW1vKSByZXR1cm4gMTsKICAgICAgcmV0dXJuIGJbMV0tYVsxXTsKICAgIH0pOwogICAgdmFyIGRvbVBjdD1NYXRoLnJvdW5kKChlTFswXT9lTFswXVsxXToyMCkqMTAwL3RvdCk7CiAgICB2YXIgbmFycjI9ZC5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciB0b3BOYXJTdHI9bmFycjIuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZTt9KS5qb2luKCcgYW5kICcpOwogICAgdmFyIHdoYXRJdD17YW54aWV0eTonVW5jZXJ0YWludHkgYW5kIHVuZWFzZSBpbiAnK25tKyh0b3BOYXJTdHI/Jy4gU2lnbmFsczogJyt0b3BOYXJTdHIrJy4nOicnKSxhbmdlcjonT3V0cmFnZSBhbmQgcHJlc3N1cmUgaW4gJytubSsodG9wTmFyU3RyPycuIERyaXZlbiBieTogJyt0b3BOYXJTdHIrJy4nOicnKSxob3BlOidPcHRpbWlzbSBhbmQgcHJvZ3Jlc3MgaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKSxwcmlkZTonSWRlbnRpdHkgYW5kIGFjaGlldmVtZW50IGluICcrbm0rKHRvcE5hclN0cj8nLiBBcm91bmQ6ICcrdG9wTmFyU3RyKycuJzonJyksZmVhcjonVGhyZWF0IHBlcmNlcHRpb24gaW4gJytubSsodG9wTmFyU3RyPycuIEFyb3VuZDogJyt0b3BOYXJTdHIrJy4nOicnKX07CiAgICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7Y3VtQT1hMjsKICAgICAgdmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICAgIH0pLmpvaW4oJycpOwogICAgdmFyIGVkZXNjPXthbnhpZXR5OidVbmNlcnRhaW50eSwgd29ycnknLGFuZ2VyOidPdXRyYWdlLCBwcm90ZXN0Jyxob3BlOidPcHRpbWlzbSwgcHJvZ3Jlc3MnLHByaWRlOidBY2hpZXZlbWVudCwgaWRlbnRpdHknLGZlYXI6J1RocmVhdCwgaW5zZWN1cml0eSd9OwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdUaGUgZW1vdGlvbmFsIHVuZGVyY3VycmVudCBvZiBzaWduYWxzIGZyb20gJytubSsnLiBXaGF0IHRvbmUgZG9taW5hdGVzIHRoZSBwb2xpdGljYWwgZGlzY291cnNlIOKAlCBvdXRyYWdlLCBob3BlLCBmZWFyLCBvciBhbnhpZXR5PycrCiAgICAnPC9kaXY+JysKICAgICghaGFzRW1vcz8nPGRpdiBzdHlsZT0icGFkZGluZzo2cHggMTFweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO21hcmdpbi1ib3R0b206MTBweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KSI+RXN0aW1hdGVkIGZyb20gc2lnbmFsIGRpcmVjdGlvbiDigJQgbGltaXRlZCBkaXJlY3QgZW1vdGlvbiBkYXRhLjwvZGl2Pic6JycpKwogICAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrcGFsW2RvbUVtb10rJzE0O2JvcmRlcjoxcHggc29saWQgJytwYWxbZG9tRW1vXSsnMzM7bWFyZ2luLWJvdHRvbToxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6JytwYWxbZG9tRW1vXSsnO21hcmdpbi1ib3R0b206NnB4Ij5Eb21pbmFudCBlbW90aW9uPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToyNnB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1pbmspIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHgiPicrZG9tUGN0KyclIMK3ICcrbm0rJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo4cHg7bGluZS1oZWlnaHQ6MS41O2ZvbnQtc3R5bGU6aXRhbGljIj4nK3doYXRJdFtkb21FbW9dKyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RW1vdGlvbmFsIGJyZWFrZG93bjwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE2cHg7Ij4nKwogICAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDc2IDc2IiBzdHlsZT0id2lkdGg6NzJweDtoZWlnaHQ6NzJweDtmbGV4LXNocmluazowIj4nK2FyY3MrJzwvc3ZnPicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjVweDsiPicrCiAgICAgICAgICAgIGVMLm1hcChmdW5jdGlvbihrdil7CiAgICAgICAgICAgICAgdmFyIGs9a3ZbMF0sdj1rdlsxXSxwY3Q9TWF0aC5yb3VuZCh2KjEwMC90b3QpOwogICAgICAgICAgICAgIHJldHVybiAnPGRpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpjZW50ZXI7bWFyZ2luLWJvdHRvbToycHg7Ij4nKwogICAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OyI+PHNwYW4gc3R5bGU9IndpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6MnB4O2JhY2tncm91bmQ6JytwYWxba10rJztkaXNwbGF5OmlubGluZS1ibG9jayI+PC9zcGFuPicrCiAgICAgICAgICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjonKyhrPT09ZG9tRW1vPyd2YXIoLS1pbmspJzondmFyKC0tZGltKScpKyciPicray5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStrLnNsaWNlKDEpKyc8L3NwYW4+PC9kaXY+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2NvbG9yOnZhcigtLWluaykiPicrcGN0KyclPC9zcGFuPicrCiAgICAgICAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrcGN0KyclO2JhY2tncm91bmQ6JytwYWxba10rJztvcGFjaXR5OjAuNztib3JkZXItcmFkaXVzOjFweCI+PC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAgIChrPT09ZG9tRW1vPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MnB4OyI+JytlZGVzY1trXSsnPC9kaXY+JzonJykrCiAgICAgICAgICAgICAgJzwvZGl2Pic7CiAgICAgICAgICAgIH0pLmpvaW4oJycpKwogICAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5TaWduYWwgaGVhZGxpbmVzPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NHB4OyI+JysKICAgICAgICAgICgoZC5hcnRpY2xlcyYmZC5hcnRpY2xlcy5sZW5ndGgpPwogICAgICAgICAgICBkLmFydGljbGVzLnNsaWNlKDAsNSkubWFwKGZ1bmN0aW9uKGEpewogICAgICAgICAgICAgIHZhciBlQ29sb3I9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICAgICAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjZweDtwYWRkaW5nOjZweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ij4nKwogICAgICAgICAgICAgICAgKGEuZW1vdGlvbj8nPHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytlQ29sb3JbYS5lbW90aW9uXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrO21hcmdpbi10b3A6NXB4O2ZsZXgtc2hyaW5rOjAiPjwvc3Bhbj4nOicnKSsKICAgICAgICAgICAgICAgICc8ZGl2PjxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLWRpbSk7bGluZS1oZWlnaHQ6MS40Ij4nKyhhLnR4dHx8YS50aXRsZXx8JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHgiPicrKGEuc3JjfHwnJykrKGEuZW1vdGlvbj8nIMK3ICcrYS5lbW90aW9uOicnKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJyk6CiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo0cHggMCI+Tm8gc2lnbmFscyB5ZXQuPC9kaXY+JykrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwoKICB9IGVsc2UgewogICAgdmFyIHZlbD1kLnZlbG9jaXR5fHwwOwogICAgdmFyIHZlbERpcj12ZWw+MC4xNT8nUmlzaW5nIGZhc3QnOnZlbD4wLjA1PydSaXNpbmcnOnZlbDwtMC4xPydDb29saW5nIGZhc3QnOnZlbDwtMC4wMj8nQ29vbGluZyc6J1N0YWJsZSc7CiAgICB2YXIgdmVsQ29sPXZlbD4wLjA1PycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgIHZhciB2ZWxEZXNjPXsnUmlzaW5nIGZhc3QnOidTaWduYWwgdm9sdW1lIHN1cmdpbmcuJywnUmlzaW5nJzonQXR0ZW50aW9uIGJ1aWxkaW5nLicsJ1N0YWJsZSc6J0JhbGFuY2VkIG1vbWVudHVtLicsJ0Nvb2xpbmcnOidBdHRlbnRpb24gZmFkaW5nLicsJ0Nvb2xpbmcgZmFzdCc6J1NoYXJwIHNpZ25hbCBkZWNheS4nfTsKICAgIHZhciBuYXJyMz1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHJpc2luZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSd1cCc7fSk7CiAgICB2YXIgZmFsbGluZ05hcnM9bmFycjMuZmlsdGVyKGZ1bmN0aW9uKG4pe3JldHVybiBuLmRpcj09PSdkb3duJzt9KTsKICAgIHZhciBjdHg9Jyc7CiAgICBpZih2ZWw+MC4wNSYmcmlzaW5nTmFycy5sZW5ndGgpIGN0eD0nRHJpdmVuIGJ5IHJpc2luZyBzaWduYWxzIGFyb3VuZCA8c3Ryb25nPicrcmlzaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+Lic7CiAgICBlbHNlIGlmKHZlbDwtMC4wNSYmZmFsbGluZ05hcnMubGVuZ3RoKSBjdHg9J1NpZ25hbHMgYXJvdW5kIDxzdHJvbmc+JytmYWxsaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lO30pLmpvaW4oJzwvc3Ryb25nPiBhbmQgPHN0cm9uZz4nKSsnPC9zdHJvbmc+IGxvc2luZyB0cmFjdGlvbi4nOwogICAgZWxzZSBjdHg9J1NpZ25hbCB2b2x1bWUgJysodmVsPjAuMDI/J2J1aWxkaW5nJzonc3RhYmxlJykrJyBpbiAnK25tKycuJzsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnSXMgYXR0ZW50aW9uIGZvciAnK25tKycgZ3Jvd2luZyBvciBmYWRpbmc/IFJpc2luZyBtb21lbnR1bSBtZWFucyBhIG5hcnJhdGl2ZSBpcyBhY2NlbGVyYXRpbmcuIENvb2xpbmcgbWVhbnMgdGhlIHN0b3J5IGlzIGxvc2luZyB0cmFjdGlvbi4nKwogICAgJzwvZGl2PicrCiAgICAnPGRpdiBzdHlsZT0icGFkZGluZzoxNHB4O2JvcmRlci1yYWRpdXM6MTBweDtiYWNrZ3JvdW5kOicrdmVsQ29sKycxNDtib3JkZXI6MXB4IHNvbGlkICcrdmVsQ29sKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3ZlbENvbCsnO21hcmdpbi1ib3R0b206NnB4Ij5TaWduYWwgbW9tZW50dW08L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo4cHg7Ij4nKwogICAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MzJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKSI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjonK3ZlbENvbCsnO2ZvbnQtd2VpZ2h0OjUwMCI+Jyt2ZWxEaXIrJzwvZGl2PicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7Zm9udC1zdHlsZTppdGFsaWM7bGluZS1oZWlnaHQ6MS41Ij4nK3ZlbERlc2NbdmVsRGlyXSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjttYXJnaW4tdG9wOjEwcHg7cGFkZGluZy10b3A6MTBweDtib3JkZXItdG9wOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIj4nK2N0eCsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlZlbG9jaXR5PC9kaXY+PGRpdiBjbGFzcz0ic3MtdmFsIiBzdHlsZT0iZm9udC1zaXplOjE4cHg7Y29sb3I6Jyt2ZWxDb2wrJyI+JysodmVsPjA/JysnOicnKSt2ZWwudG9GaXhlZCgzKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPjI0aCDOtDwvZGl2PjxkaXYgY2xhc3M9InNzLWRlbHRhICcrKGQuZGVsdGE+PTA/J3VwJzonZG4nKSsnIj4nKyhkLmRlbHRhPj0wPycrJzonJykrKGQuZGVsdGF8fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+QXR0ZW50aW9uPC9kaXY+PGRpdiBjbGFzcz0ic3MtbmFyIj4nKyhkLmF0dGVudGlvbnx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAocmlzaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+QWNjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICByaXNpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wNSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojZTA1YTI4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJykrCiAgICAgIChmYWxsaW5nTmFycy5sZW5ndGg/JzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+RGVjZWxlcmF0aW5nPC9kaXY+JysKICAgICAgICBmYWxsaW5nTmFycy5tYXAoZnVuY3Rpb24ocil7cmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47cGFkZGluZzo3cHggMTBweDttYXJnaW4tYm90dG9tOjRweDtib3JkZXItcmFkaXVzOjVweDtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoNTksMTg0LDIxNiwwLjEyKSI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWluaykiPicrci5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK3IubmFtZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtjb2xvcjojM2JiOGQ4Ij4nK3IudmFsLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+Jzt9KS5qb2luKCcnKSsnPC9kaXY+JzonJyk7CiAgfQoKICBwYW5lbC5pbm5lckhUTUw9aGVhZGVyK2JvZHk7Cn0KCgpmdW5jdGlvbiB0b2dnbGVGYXYobm0pewogIGlmKEZBVlMuaGFzKG5tKSkgRkFWUy5kZWxldGUobm0pO2Vsc2UgRkFWUy5hZGQobm0pOwogIHJlbmRlclBhbmVsKFNFTCk7cmVuZGVyRmF2cygpOwp9CmZ1bmN0aW9uIHJlbmRlckZhdnMoKXsKICB2YXIgcm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXYtcm93Jyk7CiAgaWYoIUZBVlMuc2l6ZSl7cm93LmlubmVySFRNTD0nPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+JztyZXR1cm47fQogIHJvdy5pbm5lckhUTUw9QXJyYXkuZnJvbShGQVZTKS5tYXAoZnVuY3Rpb24obm0pewogICAgdmFyIGQ9ZyhubSksZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8nI2UwNWEyOCc6JyMzYmI4ZDgnOwogICAgdmFyIHRvcD1kLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlc1swXT9kLm5hcnJhdGl2ZXNbMF0ubmFtZTon4oCUJzsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0iZmF2LWNhcmQiIG9uY2xpY2s9InNlbGVjdF8oXCcnK25tKydcJykiPicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1oZWFkIj48c3BhbiBjbGFzcz0iZmMtbmFtZSI+JytubSsnPC9zcGFuPjxzcGFuIGNsYXNzPSJmYy1zYyI+JytkLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPk5hcnJhdGl2ZTwvc3Bhbj48c3BhbiBjbGFzcz0idiI+Jyt0b3ArJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0iZmMtcm93Ij48c3Bhbj4yNGg8L3NwYW4+PHNwYW4gY2xhc3M9InYiIHN0eWxlPSJjb2xvcjonK2RDKyciPicrZFMrZC5kZWx0YSsnPC9zcGFuPjwvZGl2PicrCiAgICAnPC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQoKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLmx0YWInKS5mb3JFYWNoKGZ1bmN0aW9uKGMpewogIGMuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oeCl7eC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIGMuY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7bGF5ZXI9Yy5kYXRhc2V0LmxheWVyO2FwcGx5TGF5ZXIoKTsKICB9KTsKfSk7CgpmdW5jdGlvbiB1cGRhdGVDbG9jaygpewogIHZhciBub3c9bmV3IERhdGUoKSxpc3Q9bmV3IERhdGUobm93LmdldFRpbWUoKStub3cuZ2V0VGltZXpvbmVPZmZzZXQoKSo2MDAwMCsxOTgwMDAwMCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Nsb2NrJykudGV4dENvbnRlbnQ9U3RyaW5nKGlzdC5nZXRIb3VycygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0TWludXRlcygpKS5wYWRTdGFydCgyLCcwJykrJzonK1N0cmluZyhpc3QuZ2V0U2Vjb25kcygpKS5wYWRTdGFydCgyLCcwJykrJyBJU1QnOwp9CnNldEludGVydmFsKHVwZGF0ZUNsb2NrLDEwMDApO3VwZGF0ZUNsb2NrKCk7CgovLyBJTklUIOKAlCB3YWl0IGZvciBET00KLy8gaSBidXR0b24gdG9vbHRpcCDigJQgdXNlcyBmaXhlZCBwb3NpdGlvbmluZyBzbyBpdCdzIG5ldmVyIGNsaXBwZWQKKGZ1bmN0aW9uKCl7CiAgdmFyIHRpcD1udWxsOwogIGZ1bmN0aW9uIHNob3dUaXAoZSl7CiAgICBpZighdGlwKXt0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2x0YWItdG9vbHRpcCcpO30KICAgIHZhciB0eHQ9dGhpcy5nZXRBdHRyaWJ1dGUoJ2RhdGEtdGlwJyk7CiAgICBpZighdHh0fHwhdGlwKSByZXR1cm47CiAgICB0aXAudGV4dENvbnRlbnQ9dHh0OwogICAgdGlwLmNsYXNzTGlzdC5hZGQoJ3Zpc2libGUnKTsKICAgIHZhciByZWN0PXRoaXMuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7CiAgICB2YXIgdHc9MjQwOwogICAgdmFyIGxlZnQ9TWF0aC5taW4ocmVjdC5sZWZ0LHdpbmRvdy5pbm5lcldpZHRoLXR3LTEwKTsKICAgIHRpcC5zdHlsZS5sZWZ0PWxlZnQrJ3B4JzsKICAgIHRpcC5zdHlsZS50b3A9KHJlY3QudG9wLTEwLXRpcC5vZmZzZXRIZWlnaHR8fHJlY3QudG9wLTgwKSsncHgnOwogICAgLy8gUmVwb3NpdGlvbiBhZnRlciByZW5kZXIKICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmdW5jdGlvbigpewogICAgICB0aXAuc3R5bGUudG9wPShyZWN0LnRvcC10aXAub2Zmc2V0SGVpZ2h0LTgpKydweCc7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gaGlkZVRpcCgpewogICAgaWYoIXRpcCl7dGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsdGFiLXRvb2x0aXAnKTt9CiAgICBpZih0aXApIHRpcC5jbGFzc0xpc3QucmVtb3ZlKCd2aXNpYmxlJyk7CiAgfQogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlb3ZlcicsZnVuY3Rpb24oZSl7CiAgICBpZihlLnRhcmdldC5jbGFzc0xpc3QuY29udGFpbnMoJ2x0YWItaW5mbycpKSBzaG93VGlwLmNhbGwoZS50YXJnZXQsZSk7CiAgfSk7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2VvdXQnLGZ1bmN0aW9uKGUpewogICAgaWYoZS50YXJnZXQuY2xhc3NMaXN0LmNvbnRhaW5zKCdsdGFiLWluZm8nKSkgaGlkZVRpcCgpOwogIH0pOwp9KSgpOwoKZnVuY3Rpb24gaW5pdCgpewogIHJlbmRlclN0cmlwKCczbScpOwoKICAvLyBMb2FkIG1hcCB3aXRoIHJldHJ5CiAgdmFyIG1hcEF0dGVtcHRzPTA7CiAgZnVuY3Rpb24gdHJ5TG9hZE1hcCgpewogICAgaWYodHlwZW9mIHRvcG9qc29uPT09J3VuZGVmaW5lZCcpewogICAgICBpZihtYXBBdHRlbXB0cysrPDEwKXtzZXRUaW1lb3V0KHRyeUxvYWRNYXAsMzAwKTt9CiAgICAgIHJldHVybjsKICAgIH0KICAgIGxvYWRNYXAoKTsKICB9CiAgdHJ5TG9hZE1hcCgpOwoKICAvLyBMb2FkIGZ1bGwgY2FjaGVkIHNuYXBzaG90IGltbWVkaWF0ZWx5IGZvciBpbnN0YW50IGRhdGEKICBmZXRjaEZ1bGxTbmFwc2hvdCgpLnRoZW4oZnVuY3Rpb24ob2spewogICAgaWYob2spewogICAgICByZW5kZXJNb21lbnR1bSgpOwogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7c3RhcnRQb2xsaW5nKCk7fSwxMDAwKTsKICAgIH0gZWxzZSB7CiAgICAgIHN0YXJ0UG9sbGluZygpOwogICAgfQogIH0pOwoKICAvLyBSZXRyeSBtYXAgaWYgc3RpbGwgZW1wdHkKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSwzMDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmxlbmd0aClsb2FkTWFwKCk7fSw2MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hJbnNpZ2h0cygpLmNhdGNoKGZ1bmN0aW9uKCl7fSk7fSw1MDAwKTsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7ZmV0Y2hOYXJyYXRpdmVJbnNpZ2h0KCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDgwMDApOwp9CmlmKGRvY3VtZW50LnJlYWR5U3RhdGU9PT0nbG9hZGluZycpewogIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ0RPTUNvbnRlbnRMb2FkZWQnLCBpbml0KTsKfSBlbHNlIHsKICAvLyBBbHJlYWR5IGxvYWRlZCDigJQgYnV0IHdhaXQgb25lIHRpY2sgdG8gZW5zdXJlIGFsbCBzY3JpcHRzIHBhcnNlZAogIHNldFRpbWVvdXQoaW5pdCwgMCk7Cn0KCi8vIFJFUExBWSBJTkRJQQp2YXIgUkVQTEFZX1BFUklPRFM9eyc3ZCc6e2RheXM6NyxsYWJlbDonUGFzdCA3IGRheXMnfSwnMzBkJzp7ZGF5czozMCxsYWJlbDonUGFzdCAzMCBkYXlzJ30sJzZtJzp7ZGF5czoxODAsbGFiZWw6J1Bhc3QgNiBtb250aHMnfSwnZWxlY3Rpb24nOntkYXlzOjkwLGxhYmVsOidFbGVjdGlvbiBzZWFzb24gMjAyNCd9fTsKdmFyIHJlcGxheVBlcmlvZD0nN2QnLHJlcGxheVBvcz0wLHJlcGxheVBsYXlpbmc9ZmFsc2UscmVwbGF5VGltZXI9bnVsbCxyZXBsYXlTcGVlZD0xLGxhc3RTbmFwUG9zPS0xOwpmdW5jdGlvbiBmbXREYXRlKGQpe3JldHVybiBkLnRvTG9jYWxlRGF0ZVN0cmluZygnZW4tSU4nLHtkYXk6J251bWVyaWMnLG1vbnRoOidzaG9ydCd9KTt9CmZ1bmN0aW9uIGluaXRSZXBsYXkoKXsKICB2YXIgcD1SRVBMQVlfUEVSSU9EU1tyZXBsYXlQZXJpb2RdLG5vdz1uZXcgRGF0ZSgpLHN0YXJ0PW5ldyBEYXRlKG5vdy1wLmRheXMqODY0MDAwMDApOwogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncnAtZGF0ZXMnKTsKICBpZihlbCllbC5pbm5lckhUTUw9JzxzcGFuPicrZm10RGF0ZShzdGFydCkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjMzKSkrJzwvc3Bhbj48c3Bhbj4nK2ZtdERhdGUobmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3AuZGF5cyo4NjQwMDAwMCowLjY2KSkrJzwvc3Bhbj48c3Bhbj5Ub2RheTwvc3Bhbj4nOwogIHNldFJlcGxheVBvcygwKTsKfQpmdW5jdGlvbiBzZXRSZXBsYXlQb3MocG9zKXsKICByZXBsYXlQb3M9TWF0aC5tYXgoMCxNYXRoLm1pbigxLHBvcykpOwogIHZhciBmaWxsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1maWxsJyksdGh1bWI9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRodW1iJyksZGF0ZUVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdycC1jdXJyZW50LWRhdGUnKTsKICBpZihmaWxsKWZpbGwuc3R5bGUud2lkdGg9KHJlcGxheVBvcyoxMDApKyclJzsKICBpZih0aHVtYil0aHVtYi5zdHlsZS5sZWZ0PShyZXBsYXlQb3MqMTAwKSsnJSc7CiAgdmFyIHA9UkVQTEFZX1BFUklPRFNbcmVwbGF5UGVyaW9kXSxub3c9bmV3IERhdGUoKSxzdGFydD1uZXcgRGF0ZShub3ctcC5kYXlzKjg2NDAwMDAwKSxjdXI9bmV3IERhdGUoc3RhcnQuZ2V0VGltZSgpK3JlcGxheVBvcypwLmRheXMqODY0MDAwMDApOwogIGlmKGRhdGVFbClkYXRlRWwudGV4dENvbnRlbnQ9Zm10RGF0ZShjdXIpKycg4oCUICcrcC5sYWJlbDsKICB2YXIgc2NhbGU9MC4zNStyZXBsYXlQb3MqMC42NTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLHNhPShkLmF0dGVudGlvbnx8MCkqc2NhbGU7CiAgICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbih4KXtyZXR1cm4gKHguYXR0ZW50aW9ufHwwKSpzY2FsZTt9KTsKICAgIHZhciBtbj1NYXRoLm1pbi5hcHBseShudWxsLHNjb3JlcyksbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxLG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLChzYS1tbikvKG14LW1uKSkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGFDKHNhKSk7cC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsTWF0aC5tYXgoMC4yLDAuMituKjAuOCkpOwogIH0pOwogIGlmKE1hdGguYWJzKHJlcGxheVBvcy1sYXN0U25hcFBvcyk+MC4xMil7bGFzdFNuYXBQb3M9cmVwbGF5UG9zO3VwZGF0ZVJlcGxheVNuYXBzaG90KHJlcGxheVBvcyk7fQp9CmZ1bmN0aW9uIHVwZGF0ZVJlcGxheVNuYXBzaG90KHBvcyl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uYXR0ZW50aW9uPjA7fSkubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm57bmFtZTprdlswXSxhdHQ6TWF0aC5yb3VuZCgoa3ZbMV0uYXR0ZW50aW9ufHwwKSooMC4zNStwb3MqMC42NSkpLG5hcjooa3ZbMV0ubmFycmF0aXZlcyYma3ZbMV0ubmFycmF0aXZlc1swXT9rdlsxXS5uYXJyYXRpdmVzWzBdLm5hbWU6J+KAlCcpfTt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGIuYXR0LWEuYXR0O30pLnNsaWNlKDAsNik7CiAgdmFyIHNuYXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXNuYXAtc3RhdGVzJyk7CiAgaWYoIXNuYXApcmV0dXJuOwogIGlmKCF0b3AubGVuZ3RoKXtzbmFwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0icnAtbG9nLWVtcHR5Ij5ObyBzaWduYWwgZGF0YSB5ZXQuPC9kaXY+JztyZXR1cm47fQogIHNuYXAuaW5uZXJIVE1MPXRvcC5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuICc8ZGl2IGNsYXNzPSJycC1zdGF0ZS1jYXJkIj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1uYW1lIj4nK3MubmFtZSsnPC9kaXY+PGRpdiBjbGFzcz0icnAtc3RhdGUtbmFyIj4nK3MubmFyKyc8L2Rpdj48ZGl2IGNsYXNzPSJycC1zdGF0ZS1hdHQiPkF0dGVudGlvbiAnK3MuYXR0Kyc8L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwp9CmZ1bmN0aW9uIHRvZ2dsZVJlcGxheSgpewogIHJlcGxheVBsYXlpbmc9IXJlcGxheVBsYXlpbmc7CiAgdmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpOwogIGlmKHJlcGxheVBsYXlpbmcpe2lmKHJlcGxheVBvcz49MC45OSlzZXRSZXBsYXlQb3MoMCk7aWYoaWNvbilpY29uLnNldEF0dHJpYnV0ZSgncG9pbnRzJywnMywyIDcsMiA3LDggMyw4IE04LDIgMTIsMiAxMiw4IDgsOCcpO3J1blJlcGxheSgpO30KICBlbHNle2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO30KfQpmdW5jdGlvbiBydW5SZXBsYXkoKXsKICBjbGVhckludGVydmFsKHJlcGxheVRpbWVyKTsKICByZXBsYXlUaW1lcj1zZXRJbnRlcnZhbChmdW5jdGlvbigpewogICAgcmVwbGF5UG9zKz0wLjAwMypyZXBsYXlTcGVlZDsKICAgIGlmKHJlcGxheVBvcz49MSl7cmVwbGF5UG9zPTE7c2V0UmVwbGF5UG9zKDEpO3JlcGxheVBsYXlpbmc9ZmFsc2U7dmFyIGljb249ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXBsYXktaWNvbicpO2lmKGljb24paWNvbi5zZXRBdHRyaWJ1dGUoJ3BvaW50cycsJzIsMSA5LDUgMiw5Jyk7Y2xlYXJJbnRlcnZhbChyZXBsYXlUaW1lcik7YXBwbHlMYXllcigpO3JldHVybjt9CiAgICBzZXRSZXBsYXlQb3MocmVwbGF5UG9zKTsKICB9LDYwKTsKfQooZnVuY3Rpb24oKXt2YXIgdHJhY2s9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JwLXRyYWNrJyk7aWYoIXRyYWNrKXJldHVybjt2YXIgZHJhZz1mYWxzZTsKdHJhY2suYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vkb3duJyxmdW5jdGlvbihlKXtkcmFnPXRydWU7dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxmdW5jdGlvbihlKXtpZighZHJhZylyZXR1cm47dmFyIHJlY3Q9dHJhY2suZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7c2V0UmVwbGF5UG9zKChlLmNsaWVudFgtcmVjdC5sZWZ0KS9yZWN0LndpZHRoKTt9KTsKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2V1cCcsZnVuY3Rpb24oKXtpZihkcmFnKXtkcmFnPWZhbHNlO2lmKCFyZXBsYXlQbGF5aW5nKWFwcGx5TGF5ZXIoKTt9fSk7fSkoKTsKZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYnRuKXtidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLGZ1bmN0aW9uKCl7ZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnJwLWJ0bicpLmZvckVhY2goZnVuY3Rpb24oYil7Yi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTtidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7cmVwbGF5UGVyaW9kPWJ0bi5kYXRhc2V0LnBlcmlvZDtyZXBsYXlQb3M9MDtsYXN0U25hcFBvcz0tMTtpbml0UmVwbGF5KCk7fSk7fSk7CmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGJ0bil7YnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe2RvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5ycC1zcGQnKS5mb3JFYWNoKGZ1bmN0aW9uKGIpe2IuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7fSk7YnRuLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO3JlcGxheVNwZWVkPXBhcnNlSW50KGJ0bi5kYXRhc2V0LnNwZCk7fSk7fSk7CmluaXRSZXBsYXkoKTsKc2V0VGltZW91dChmdW5jdGlvbigpewogIC8vIEF1dG8tc2VsZWN0IGhvdHRlc3Qgc3RhdGUgZnJvbSBMSVZFIGRhdGEKICB2YXIgc3JjPU9iamVjdC5rZXlzKExJVkUpLmxlbmd0aD9MSVZFOlNEOwogIHZhciB0b3A9T2JqZWN0LmVudHJpZXMoc3JjKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICBpZih0b3ApewogICAgdmFyIGVsPWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXAtc3RhdGVzIC5zdGF0ZVtkYXRhLW5hbWU9IicrdG9wWzBdKyciXScpOwogICAgaWYoZWwpIHNlbGVjdF8odG9wWzBdKTsKICB9Cn0sMzAwMCk7CnNldFRpbWVvdXQocmVuZGVyRmF2cywyNDAwKTsKPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPgo="

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
