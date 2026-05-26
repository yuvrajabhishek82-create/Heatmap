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

async def compute_db_intelligence():
    """
    Use historical DB data to compute:
    - Dynamic baselines (7-day rolling avg per state)
    - Historical velocity (DB-based, restart-proof)
    - Abnormal movement detection
    - Narrative persistence scoring
    """
    conn = await get_db()
    if not conn:
        return
    try:
        now = datetime.now(timezone.utc)
        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_48h = now - timedelta(hours=48)

        # ── Dynamic baselines: 7-day rolling average signal count per state ──
        rows = await conn.fetch("""
            SELECT state, COUNT(*) as cnt,
                   COUNT(*) FILTER (WHERE published_at > $2) as recent_cnt
            FROM signal_store
            WHERE published_at > $1
            GROUP BY state
        """, cutoff_7d, cutoff_24h)

        for row in rows:
            state = row["state"]
            seven_day_total = row["cnt"]
            daily_avg = seven_day_total / 7.0
            # Blend with hardcoded baseline (60/40) for stability
            hardcoded = STATE_BASELINES.get(state, 5.0)
            dynamic = daily_avg if daily_avg > 0 else hardcoded
            store.dynamic_baselines[state] = round(0.6 * dynamic + 0.4 * hardcoded, 2)

        # ── Historical velocity: last 24h vs previous 24h from DB ────────────
        vel_rows = await conn.fetch("""
            SELECT state,
                   COUNT(*) FILTER (WHERE published_at > $2) as recent,
                   COUNT(*) FILTER (WHERE published_at <= $2 AND published_at > $3) as older
            FROM signal_store
            WHERE published_at > $3
            GROUP BY state
        """, cutoff_7d, cutoff_24h, cutoff_48h)

        for row in vel_rows:
            state = row["state"]
            recent = row["recent"] or 0
            older = row["older"] or 1
            raw_vel = (recent - older) / max(older, 1)
            store.historical_velocity[state] = round(math.tanh(raw_vel * 2), 3)

        # ── Abnormal movement: compare current 24h to 7-day daily average ────
        store.abnormal_states = set()
        for state, baseline in store.dynamic_baselines.items():
            current = len([s for s in store.signals.get(state, [])
                          if (now - s["published_at"]).total_seconds() < 86400])
            if baseline > 0 and current > baseline * 2.5:
                store.abnormal_states.add(state)
                print(f"[intelligence] Abnormal movement: {state} ({current} vs baseline {baseline:.1f})")

        # ── Narrative persistence: how many consecutive days each narrative dominated ──
        cutoff_14d_date = (now - timedelta(days=14)).date()
        nar_rows = await conn.fetch("""
            SELECT state, dominant_narrative, COUNT(*) as days
            FROM daily_snapshots
            WHERE date > $1::date AND dominant_narrative IS NOT NULL
            GROUP BY state, dominant_narrative
            ORDER BY state, days DESC
        """, cutoff_14d_date)

        store.narrative_persistence = {}
        for row in nar_rows:
            state = row["state"]
            if state not in store.narrative_persistence:
                store.narrative_persistence[state] = {}
            store.narrative_persistence[state][row["dominant_narrative"]] = row["days"]

        store.db_intelligence_at = now
        print(f"[intelligence] Updated: {len(store.dynamic_baselines)} baselines, "
              f"{len(store.abnormal_states)} abnormal states")

    except Exception as e:
        print(f"[intelligence] Error: {e}")
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
        # DB-derived intelligence — updated every 6 hours
        self.dynamic_baselines: dict[str, float] = {}       # 7-day rolling avg per state
        self.historical_velocity: dict[str, float] = {}     # DB-based velocity
        self.abnormal_states: set[str] = set()              # states with unusual movement
        self.narrative_persistence: dict[str, dict] = {}    # how long each narrative has held
        self.db_intelligence_at: datetime | None = None

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
    # Refresh DB intelligence every 6 hours
    if (not store.db_intelligence_at or
        (datetime.now(timezone.utc) - store.db_intelligence_at).total_seconds() > 21600):
        asyncio.create_task(compute_db_intelligence())
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

    # Baseline-normalized attention — use dynamic DB baseline if available
    static_baseline = STATE_BASELINES.get(state, 5.0)
    baseline = store.dynamic_baselines.get(state, static_baseline)
    raw_count = len(sigs_48h)
    deviation_ratio = raw_count / max(baseline, 0.1)
    normalized = weighted_volume * (deviation_ratio ** 0.5)
    # Denominator: baseline * 6 means 4x expected volume → ~80 attention
    attention = round(min(95, 100 * math.tanh(normalized / (baseline * 6.0))), 1)
    attention = round(attention * conf_weight, 1)
    # Attention smoothing: only blend if previous score is meaningful (>2)
    prev_score = store.scores.get(state, {}).get("attention", 0)
    if prev_score > 2:
        attention = round(0.7 * attention + 0.3 * prev_score, 1)
    # Boost for abnormal movement
    if state in store.abnormal_states:
        attention = round(min(95, attention * 1.25), 1)

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
    mem_velocity = round(math.tanh(raw_vel * 2), 3)

    # Blend with DB-based velocity if available (more stable, restart-proof)
    db_vel = store.historical_velocity.get(state)
    if db_vel is not None:
        velocity = round(0.6 * mem_velocity + 0.4 * db_vel, 3)
    else:
        velocity = mem_velocity
    if velocity > 0.3 and confidence in ("MEDIUM", "HIGH"):
        attention = round(min(95, attention * 1.15), 1)

    is_regional = deviation_ratio > 2.0 and raw_count < baseline * 1.5

    # ── Cluster-coherent scoring ─────────────────────────────────────────
    # Step 1: Score each signal by weighted importance
    def signal_weight(s):
        decay = 2 ** (-(now - s["published_at"]).total_seconds() / 129_600)
        src_w = get_source_weight(s.get("source", ""))
        intensity = s.get("intensity", 0.5)
        return decay * src_w * intensity

    weighted_sigs = [(s, signal_weight(s)) for s in sigs_48h]
    weighted_sigs.sort(key=lambda x: -x[1])

    # Step 2: Build narrative scores weighted by signal importance
    nar_now: dict[str, float] = {}
    nar_prev_d: dict[str, int] = {}
    for s, w in weighted_sigs:
        for n in s.get("narratives", []):
            nar_now[n] = nar_now.get(n, 0) + w
    for s in sigs_prev:
        for n in s.get("narratives", []):
            nar_prev_d[n] = nar_prev_d.get(n, 0) + 1

    total_nar = max(0.001, sum(nar_now.values()))
    top_narratives = sorted(nar_now.items(), key=lambda kv: -kv[1])[:5]

    # Step 3: Identify dominant narrative cluster
    dominant_nar = top_narratives[0][0] if top_narratives else None

    # Step 4: Extract signals that belong to the dominant narrative cluster
    # This ensures emotion and momentum are derived from the SAME discourse
    if dominant_nar:
        cluster_sigs = [s for s, w in weighted_sigs
                       if dominant_nar in s.get("narratives", [])]
        # Fall back to all signals if cluster is too small
        if len(cluster_sigs) < max(2, len(sigs_48h) // 4):
            cluster_sigs = [s for s, w in weighted_sigs]
    else:
        cluster_sigs = [s for s, w in weighted_sigs]

    # Step 5: Derive EMOTION from the dominant cluster signals
    emo_totals: dict[str, float] = {k: 0.0 for k in EMOTION_KEYWORDS}
    for s in cluster_sigs:
        w = signal_weight(s)
        for k, v in s.get("emotions", {}).items():
            if k in emo_totals:
                emo_totals[k] += v * w  # weight emotion by signal importance
    total_emo = sum(emo_totals.values())
    emotions = {k: round(v / total_emo, 3) for k, v in emo_totals.items()
                if emo_totals[k] > 0} if total_emo > 0 else {}

    # Step 6: Build narrative breakdown with persistence context
    narrative_breakdown = []
    for n, c in top_narratives:
        prev = nar_prev_d.get(n, 0)
        val = round(c / total_nar * 100, 1)
        persistence = store.narrative_persistence.get(state, {}).get(n, 0)
        if persistence >= 3:
            direction = "sustained"
        elif prev == 0 or c > prev * 0.5:
            direction = "up"
        elif c < prev * 0.3:
            direction = "down"
        else:
            direction = "flat"
        narrative_breakdown.append({"name": n, "val": val, "dir": direction})

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

    # Dominant emotion derived from cluster signals — contextually aligned with dominant narrative
    dominant_emotion = max(emotions, key=lambda k: emotions[k]) if emotions else None

    # Derive dominant_narrative first, then apply coherence check
    dominant_narrative = top_narratives[0][0] if top_narratives else None

    # Coherence check: ensure emotion aligns with the dominant narrative
    COHERENT_PAIRS = {
        "border issues":   {"fear", "anxiety", "anger"},
        "law & order":     {"anger", "fear", "anxiety"},
        "corruption":      {"anger", "anxiety"},
        "elections":       {"hope", "pride", "anxiety", "anger"},
        "governance":      {"anger", "anxiety", "hope"},
        "protest":         {"anger", "anxiety"},
        "security":        {"fear", "anxiety", "anger"},
        "unemployment":    {"anxiety", "anger"},
        "farmer issues":   {"anger", "anxiety"},
        "religion":        {"anger", "pride", "fear"},
        "environment":     {"anxiety", "fear"},
        "economy":         {"anxiety", "anger", "hope"},
        "infrastructure":  {"hope", "pride"},
        "education":       {"anxiety", "anger", "hope"},
        "caste":           {"anger", "anxiety", "fear"},
        "nationalism":     {"pride", "anger"},
        "health":          {"anxiety", "fear"},
    }
    if dominant_narrative and dominant_emotion:
        coherent = COHERENT_PAIRS.get(dominant_narrative, set())
        if coherent and dominant_emotion not in coherent:
            coherent_emos = {k: v for k, v in emotions.items() if k in coherent}
            if coherent_emos:
                dominant_emotion = max(coherent_emos, key=lambda k: coherent_emos[k])

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
            await compute_db_intelligence()
            print("[startup] DB intelligence loaded")
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
FRONTEND_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ii8+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wIi8+Cjx0aXRsZT5QdWxzZSBvZiBJbmRpYSDigJQgVGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lcy48L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUZyYXVuY2VzOm9wc3osaXRhbCx3Z2h0QDkuLjE0NCwwLDMwMDs5Li4xNDQsMCw0MDA7OS4uMTQ0LDEsMzAwOzkuLjE0NCwxLDQwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEAzMDA7NDAwJmZhbWlseT1JbnRlcitUaWdodDp3Z2h0QDMwMDs0MDA7NTAwJmRpc3BsYXk9c3dhcCIgcmVsPSJzdHlsZXNoZWV0Ij4KPHN0eWxlPgo6cm9vdHsKICAtLWJnOiMwNTA3MGM7CiAgLS1iZzE6IzA5MGQxNTsKICAtLWJnMjojMGQxMjIwOwogIC0tc3VyZjpyZ2JhKDE0LDIwLDM0LDAuNik7CiAgLS1ib3JkZXI6cmdiYSgxNjAsMTkwLDIzMCwwLjA2KTsKICAtLWJvcmRlcjI6cmdiYSgxNjAsMTkwLDIzMCwwLjEzKTsKICAtLWluazojZGRlNmY1OwogIC0tZGltOiM3YTg4OTk7CiAgLS1mYWludDojM2U0ZDYwOwogIC0tYWNjZW50OiNlMDVhMjg7CiAgLS1hY2NlbnREaW06cmdiYSgyMjQsOTAsNDAsMC4xNSk7CiAgLS1yaXNlOiNlMDVhMjg7CiAgLS1mYWxsOiMzYmI4ZDg7CiAgLS1zZXJpZjonRnJhdW5jZXMnLEdlb3JnaWEsc2VyaWY7CiAgLS1zYW5zOidJbnRlciBUaWdodCcsc3lzdGVtLXVpLHNhbnMtc2VyaWY7CiAgLS1tb25vOidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KaHRtbCxib2R5e2JhY2tncm91bmQ6dmFyKC0tYmcpO2NvbG9yOnZhcigtLWluayk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7b3ZlcmZsb3cteDpoaWRkZW47c2Nyb2xsLWJlaGF2aW9yOnNtb290aH0KCi8qIGF0bW9zcGhlcmljIGJhY2tncm91bmQgKi8KYm9keXsKICBiYWNrZ3JvdW5kOgogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgODAlIDUwJSBhdCA1MCUgLTEwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNTUpIDAlLCB0cmFuc3BhcmVudCA2MCUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNTAlIDQwJSBhdCAxMCUgNjAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMjUpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgcmFkaWFsLWdyYWRpZW50KGVsbGlwc2UgNjAlIDUwJSBhdCA5MCUgMTAwJSwgcmdiYSgxNDAsODAsMjAwLDAuMDIpIDAlLCB0cmFuc3BhcmVudCA1NSUpLAogICAgdmFyKC0tYmcpOwogIG1pbi1oZWlnaHQ6MTAwdmg7Cn0KYm9keTo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7cG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtwb2ludGVyLWV2ZW50czpub25lO3otaW5kZXg6MDsKICBiYWNrZ3JvdW5kLWltYWdlOnVybCgiZGF0YTppbWFnZS9zdmcreG1sO3V0ZjgsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyMDAnIGhlaWdodD0nMjAwJz48ZmlsdGVyIGlkPSduJz48ZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC44NScgbnVtT2N0YXZlcz0nMicvPjxmZUNvbG9yTWF0cml4IHZhbHVlcz0nMCAwIDAgMCAwLjg1IDAgMCAwIDAgMC45IDAgMCAwIDAgMSAwIDAgMCAwLjA0IDAnLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0nMTAwJScgaGVpZ2h0PScxMDAlJyBmaWx0ZXI9J3VybCglMjNuKScvPjwvc3ZnPiIpOwogIG9wYWNpdHk6MC40NTttaXgtYmxlbmQtbW9kZTpzb2Z0LWxpZ2h0Owp9CgovKiBUT1BCQVIgKi8KLnRvcGJhcnsKICBwb3NpdGlvbjpmaXhlZDt0b3A6MDtsZWZ0OjA7cmlnaHQ6MDt6LWluZGV4OjEwMDsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6MTRweCAzNnB4OwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC43NSk7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMjhweCkgc2F0dXJhdGUoMTMwJSk7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouYnJhbmR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDt0ZXh0LWRlY29yYXRpb246bm9uZX0KLmJyYW5kLW1hcmt7d2lkdGg6MjhweDtoZWlnaHQ6MjhweDtib3JkZXItcmFkaXVzOjZweDtiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCgxMzVkZWcsI2UwNWEyOCAwJSwjYjAyODQ4IDEwMCUpO3Bvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowO2JveC1zaGFkb3c6MCAwIDE4cHggcmdiYSgyMjQsOTAsNDAsMC4yMil9Ci5icmFuZC1wdWxzZS1kb3R7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXJ9Ci5icmFuZC1wdWxzZS1kb3Q6OmJlZm9yZXtjb250ZW50OicnO3dpZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjkyKTthbmltYXRpb246YnJhbmRQdWxzZSAzLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouYnJhbmQtcHVsc2UtZG90OjphZnRlcntjb250ZW50OicnO3Bvc2l0aW9uOmFic29sdXRlO3dpZHRoOjE0cHg7aGVpZ2h0OjE0cHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7YW5pbWF0aW9uOmJyYW5kUmlwcGxlIDMuMnMgZWFzZS1vdXQgaW5maW5pdGV9CkBrZXlmcmFtZXMgYnJhbmRQdWxzZXswJSwxMDAle29wYWNpdHk6MC45O3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjQ1O3RyYW5zZm9ybTpzY2FsZSgwLjgyKX19CkBrZXlmcmFtZXMgYnJhbmRSaXBwbGV7MCV7dHJhbnNmb3JtOnNjYWxlKDAuNik7b3BhY2l0eTowLjV9MTAwJXt0cmFuc2Zvcm06c2NhbGUoMS44KTtvcGFjaXR5OjB9fQouYnJhbmQtdGV4dC1ibG9ja3tkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxcHh9Ci5icmFuZC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MTVweDtmb250LXdlaWdodDo0MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTtjb2xvcjp2YXIoLS1pbmspO2xpbmUtaGVpZ2h0OjEuMTtmb250LXN0eWxlOm5vcm1hbH0KLmJyYW5kLXB1bHNlLXdvcmR7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I2U4YzRhMDthbmltYXRpb246cHVsc2VXb3JkIDRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlV29yZHswJSwxMDAle29wYWNpdHk6MX01MCV7b3BhY2l0eTowLjcyfX0KLmJyYW5kLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjF9Ci50b3BiYXItcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoyMHB4fQouc2lnLWhvdmVyLXdyYXB7cG9zaXRpb246cmVsYXRpdmU7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtjdXJzb3I6ZGVmYXVsdH0KLnNpZy1ob3Zlci10aXB7CiAgcG9zaXRpb246YWJzb2x1dGU7dG9wOmNhbGMoMTAwJSArIDEwcHgpO3JpZ2h0OjA7CiAgYmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45Nyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxNHB4O3doaXRlLXNwYWNlOm5vd3JhcDsKICBwb2ludGVyLWV2ZW50czpub25lO29wYWNpdHk6MDt2aXNpYmlsaXR5OmhpZGRlbjsKICB0cmFuc2l0aW9uOm9wYWNpdHkgMC4xOHMsdmlzaWJpbGl0eSAwLjE4czsKICB6LWluZGV4Ojk5OTk7Cn0KLnNpZy1ob3Zlci13cmFwOmhvdmVyIC5zaWctaG92ZXItdGlwe29wYWNpdHk6MTt2aXNpYmlsaXR5OnZpc2libGV9Ci5zaWctaG92ZXItbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NXB4O29wYWNpdHk6MC43fQouc2lnLWhvdmVyLXNvdXJjZXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzowLjA0ZW19Ci5saXZlLWluZGljYXRvcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo3cHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bGV0dGVyLXNwYWNpbmc6MC4wNWVtOwp9Ci5saXZlLWRvdHt3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiM0YWRlODA7Ym94LXNoYWRvdzowIDAgOHB4IHJnYmEoNzQsMjIyLDEyOCwwLjcpO2FuaW1hdGlvbjpsZCAyLjVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGxkezAlLDEwMCV7b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxKX01MCV7b3BhY2l0eTowLjM1O3RyYW5zZm9ybTpzY2FsZSgwLjgpfX0KLmNsb2Nre2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bGV0dGVyLXNwYWNpbmc6MC4wNGVtfQoKLyogSEVSTyAqLwouaGVyb3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgcGFkZGluZzo3MnB4IDM2cHggMDsKICBtYXgtd2lkdGg6MTQ4MHB4O21hcmdpbjowIGF1dG87Cn0KLmhlcm8tZXllYnJvd3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMzJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206MjRweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4fQouaGVyby1leWVicm93OjpiZWZvcmV7Y29udGVudDonJzt3aWR0aDoxNnB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1mYWludCk7b3BhY2l0eTowLjV9Ci5oZXJvLWJyYW5kLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOm5vcm1hbDtmb250LXNpemU6Y2xhbXAoMzZweCw0LjJ2dyw2NHB4KTtsaW5lLWhlaWdodDoxO2xldHRlci1zcGFjaW5nOi0wLjAzZW07Y29sb3I6dmFyKC0taW5rKTttYXJnaW46MH0KLmhlcm8tYnJhbmQtbmFtZSBlbXtmb250LXN0eWxlOml0YWxpYztjb2xvcjojZThjNGEwO2FuaW1hdGlvbjpwdWxzZU5hbWVHbG93IDVzIGVhc2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIHB1bHNlTmFtZUdsb3d7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC43Mn19Ci5oZXJvLXRhZ2xpbmV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZTpjbGFtcCgxNXB4LDEuNXZ3LDIwcHgpO2ZvbnQtd2VpZ2h0OjMwMDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNDtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbjowIDAgMTJweCAwO21heC13aWR0aDo0ODBweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjF9Ci5oZXJvLWRlc2N7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjEzcHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWZhaW50KTtsaW5lLWhlaWdodDoxLjY7bWF4LXdpZHRoOjQwMHB4O21hcmdpbjowIDAgNnB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1zdWItbGluZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDYyLDc3LDk2LDAuNik7bWFyZ2luOjAgMCAyMHB4IDA7cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouaGVyby1wdWxzZS1zaWduYWx7cG9zaXRpb246cmVsYXRpdmU7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtmbGV4LXNocmluazowfQouaHBzLWNvcmV7cG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuOTthbmltYXRpb246aHBzQ29yZSA0cyBlYXNlLWluLW91dCBpbmZpbml0ZX0KQGtleWZyYW1lcyBocHNDb3JlezAlLDEwMCV7b3BhY2l0eTowLjk7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNDt0cmFuc2Zvcm06c2NhbGUoMC43NSl9fQouaHBzLXJpbmd7cG9zaXRpb246YWJzb2x1dGU7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1hY2NlbnQpO2FuaW1hdGlvbjpocHNSaW5nIDRzIGVhc2Utb3V0IGluZmluaXRlfQouaHBzLXJpbmcucjF7aW5zZXQ6MXB4O2FuaW1hdGlvbi1kZWxheTowc30uaHBzLXJpbmcucjJ7aW5zZXQ6LTNweDthbmltYXRpb24tZGVsYXk6MS40cztib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zNSl9CkBrZXlmcmFtZXMgaHBzUmluZ3swJXtvcGFjaXR5OjAuNjt0cmFuc2Zvcm06c2NhbGUoMC43KX0xMDAle29wYWNpdHk6MDt0cmFuc2Zvcm06c2NhbGUoMS42KX19CgovKiBTSUdOQVRVUkUgSU5TSUdIVCAqLwouc2lnbmF0dXJlLWluc2lnaHR7CiAgbWFyZ2luLXRvcDowOwogIHBhZGRpbmc6MTRweCAyMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDEzNWRlZywgcmdiYSgyMjQsOTAsNDAsMC4wNikgMCUsIHJnYmEoNTksMTg0LDIxNiwwLjAzKSAxMDAlKTsKICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpOwogIG1heC13aWR0aDo5MDBweDsKICBwb3NpdGlvbjpyZWxhdGl2ZTtvdmVyZmxvdzpoaWRkZW47Cn0KLnNpZ25hdHVyZS1pbnNpZ2h0OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MnB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgdmFyKC0tYWNjZW50KSwgdHJhbnNwYXJlbnQpOwp9Ci5zaS1sYWJlbHsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206MTBweDsKfQouc2ktdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOmNsYW1wKDE0cHgsMS40dncsMThweCk7CiAgZm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluayk7bGluZS1oZWlnaHQ6MS41O2xldHRlci1zcGFjaW5nOi0wLjAxZW07Cn0KLnNpLXRleHQgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tYWNjZW50KX0KLnNpLXN1YnsKICBtYXJnaW4tdG9wOjEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTsKICBsZXR0ZXItc3BhY2luZzowLjA0ZW07ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTRweDtmbGV4LXdyYXA6d3JhcDsKfQouc2ktdGFnewogIHBhZGRpbmc6MnB4IDhweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTsKICBmb250LXNpemU6OS41cHg7Cn0KCi8qIE5BUlJBVElWRSBTVFJJUCAqLwoKLnN0cmlwLXRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLWZhaW50KTtwYWRkaW5nOjRweCA5cHg7Ym9yZGVyLXJhZGl1czozcHg7Y3Vyc29yOnBvaW50ZXI7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDtib3JkZXI6bm9uZTt0cmFuc2l0aW9uOmFsbCAwLjE1czsKfQouc3RyaXAtdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4xMil9Ci5zdHJpcC10YWI6aG92ZXJ7Y29sb3I6dmFyKC0tZGltKX0KLnN0cmlwLWNvbHsKICBmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1iZzEpO3BhZGRpbmc6MDsKfQouc3RyaXAtY29sLWhlYWR7CiAgcGFkZGluZzoxMHB4IDE2cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwp9Ci5zdHJpcC1jb2wtaGVhZC5mYWRle2NvbG9yOnZhcigtLWZhbGwpfQouc3RyaXAtY29sLWhlYWQucmlzZTJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5zdHJpcC1jb2wtaGVhZC5zaGlmdHtjb2xvcjp2YXIoLS1kaW0pfQouc3RyaXAtY29sLWJvZHl7cGFkZGluZzoxMnB4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OHB4fQouc3RyaXAtaXRlbXsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7Z2FwOjhweDsKfQouc3RyaXAtdG9waWN7Zm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5zdHJpcC1ub3Rle2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjp2YXIoLS1mYWludCl9Ci5zdHJpcC1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTtmb250LXNpemU6MTRweDtmbGV4LXNocmluazowfQoKLyogTUFJTiBMQVlPVVQgKi8KLm1haW57CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDM2MHB4OwogIGdhcDoxNHB4OwogIG1pbi13aWR0aDowOwp9CgovKiBNQVAgKi8KLm1hcC1jYXJkewogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6MTZweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxNnB4KTsKICBvdmVyZmxvdzpoaWRkZW47cG9zaXRpb246cmVsYXRpdmU7Cn0KLm1hcC1jYXJkOjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDowOwogIGJhY2tncm91bmQ6CiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA3MCUgNTAlIGF0IDM1JSAwJSwgcmdiYSgyMjQsOTAsNDAsMC4wNSkgMCUsIHRyYW5zcGFyZW50IDYwJSksCiAgICByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSA1MCUgNDAlIGF0IDgwJSAxMDAlLCByZ2JhKDU5LDE4NCwyMTYsMC4wMykgMCUsIHRyYW5zcGFyZW50IDYwJSk7Cn0KLm1hcC10b3B7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47CiAgcGFkZGluZzoxMnB4IDE4cHggMDsKfQoubWFwLXRpdGxlLWJsb2NrIC5tdHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE3cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW19Ci5tYXAtdGl0bGUtYmxvY2sgLm1ze2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDZlbTttYXJnaW4tdG9wOjJweH0KLmxlZ2VuZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWRpbSl9Ci5sZWdlbmQtYmFyewogIGhlaWdodDozcHg7d2lkdGg6ODBweDtib3JkZXItcmFkaXVzOjJweDsKICBiYWNrZ3JvdW5kOmxpbmVhci1ncmFkaWVudCh0byByaWdodCwjMGUyMDM1LCMxYTU1ODAgMjUlLCM4YTVjMTggNTUlLCNjMDM4MWEgODAlLCNlMDEwMjApOwp9Ci5sYXllci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsKICBwYWRkaW5nOjEwcHggMjBweCA2cHg7Cn0KLmxheWVyLWxhYmVse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtsZXR0ZXItc3BhY2luZzowLjE0ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLmx0YWJze2Rpc3BsYXk6ZmxleDtnYXA6M3B4fQoubHRhYnsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6M3B4IDlweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO3RyYW5zaXRpb246YWxsIDAuMTVzOwp9Ci5sdGFiLmFjdGl2ZXtjb2xvcjp2YXIoLS1pbmspO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wOCk7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMil9Ci5sdGFie2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo1cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6dmlzaWJsZX0KLmx0YWItaW5mb3t3aWR0aDoxM3B4O2hlaWdodDoxM3B4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgxNjAsMTkwLDIzMCwwLjIpO2ZvbnQtc2l6ZTo4cHg7Zm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zdHlsZTppdGFsaWM7Zm9udC13ZWlnaHQ6NjAwO2NvbG9yOnJnYmEoMTYwLDE5MCwyMzAsMC4zNSk7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtjdXJzb3I6aGVscDtmbGV4LXNocmluazowO3RyYW5zaXRpb246YWxsIDAuMTVzO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MTAwfQoubHRhYi1pbmZvOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2NlbnQpO2NvbG9yOnZhcigtLWFjY2VudCl9CiNsdGFiLXRvb2x0aXB7cG9zaXRpb246Zml4ZWQ7YmFja2dyb3VuZDpyZ2JhKDgsMTIsMjAsMC45OCk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMTIpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6MTBweCAxM3B4O2ZvbnQtZmFtaWx5OnZhcigtLXNhbnMpO2ZvbnQtc2l6ZToxMXB4O2ZvbnQtd2VpZ2h0OjMwMDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjt3aWR0aDoyMzBweDt3aGl0ZS1zcGFjZTpub3JtYWw7dGV4dC1hbGlnbjpsZWZ0O2JveC1zaGFkb3c6MCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNik7cG9pbnRlci1ldmVudHM6bm9uZTtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzO3otaW5kZXg6OTk5OTk7ZGlzcGxheTpub25lfQojbHRhYi10b29sdGlwLnZpc2libGV7b3BhY2l0eToxO2Rpc3BsYXk6YmxvY2t9Ci5sdGFiOmhvdmVye2NvbG9yOnZhcigtLWRpbSl9CgoubWFwLXN2Zy13cmFwewogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmc6MTJweCAxNnB4IDE2cHg7Cn0KLm1hcC1pbm5lcntwb3NpdGlvbjpyZWxhdGl2ZTthc3BlY3QtcmF0aW86MS8xO3dpZHRoOjEwMCV9CiNpbmRpYS1tYXB7d2lkdGg6MTAwJTtoZWlnaHQ6MTAwJTtkaXNwbGF5OmJsb2NrO292ZXJmbG93OnZpc2libGV9CgovKiBtYXAgc3RhdGUgc3R5bGVzICovCiNpbmRpYS1tYXAgLnN0YXRlewogIGN1cnNvcjpwb2ludGVyOwogIHRyYW5zaXRpb246ZmlsdGVyIDAuMjVzIGVhc2UsIHN0cm9rZS13aWR0aCAwLjJzIGVhc2UsIHN0cm9rZSAwLjJzIGVhc2U7Cn0KI2luZGlhLW1hcCAuc3RhdGU6aG92ZXJ7CiAgc3Ryb2tlOnJnYmEoMjU1LDI1NSwyNTUsMC43KSAhaW1wb3J0YW50O3N0cm9rZS13aWR0aDoxcHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjI1KSBkcm9wLXNoYWRvdygwIDAgMTBweCByZ2JhKDI1NSwyNTUsMjU1LDAuMikpOwp9CiNpbmRpYS1tYXAgLnN0YXRlLnNlbGVjdGVkewogIHN0cm9rZTpyZ2JhKDI1NSwyNTUsMjU1LDAuOSkgIWltcG9ydGFudDtzdHJva2Utd2lkdGg6MS40cHggIWltcG9ydGFudDsKICBmaWx0ZXI6YnJpZ2h0bmVzcygxLjM1KSBkcm9wLXNoYWRvdygwIDAgMTZweCByZ2JhKDI1NSwyNTUsMjU1LDAuMykpOwp9CgovKiBhbmltYXRlZCBwdWxzZSByaW5ncyAqLwoucHVsc2UtcmluZ3tmaWxsOm5vbmU7cG9pbnRlci1ldmVudHM6bm9uZX0KLnB1bHNlLXJpbmcucDF7YW5pbWF0aW9uOnByIDIuOHMgZWFzZS1vdXQgaW5maW5pdGV9Ci5wdWxzZS1yaW5nLnAye2FuaW1hdGlvbjpwciAyLjhzIGVhc2Utb3V0IDAuOXMgaW5maW5pdGV9CkBrZXlmcmFtZXMgcHJ7CiAgMCV7cjo0O29wYWNpdHk6MC43O3N0cm9rZS13aWR0aDoxLjJ9CiAgMTAwJXtyOjI2O29wYWNpdHk6MDtzdHJva2Utd2lkdGg6MC4yfQp9CgovKiBhdG1vc3BoZXJpYyBnbG93IGJlaGluZCBob3Qgc3RhdGVzICovCi5zdGF0ZS1nbG93e3BvaW50ZXItZXZlbnRzOm5vbmU7ZmlsbDpub25lfQpAa2V5ZnJhbWVzIGdsb3dQdWxzZXswJSwxMDAle29wYWNpdHk6MC4xMn01MCV7b3BhY2l0eTowLjIyfX0KCi5tYXAtdG9vbHRpcHsKICBwb3NpdGlvbjphYnNvbHV0ZTtwb2ludGVyLWV2ZW50czpub25lOwogIGJhY2tncm91bmQ6cmdiYSg1LDcsMTIsMC45NSk7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTJweCk7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtib3JkZXItcmFkaXVzOjlweDsKICBwYWRkaW5nOjEycHggMTRweDtvcGFjaXR5OjA7dHJhbnNpdGlvbjpvcGFjaXR5IDAuMTJzO3otaW5kZXg6OTk5OTttaW4td2lkdGg6MTcwcHg7Cn0KLnR0LW57Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNnB4O2ZvbnQtd2VpZ2h0OjQwMDttYXJnaW4tYm90dG9tOjhweDtjb2xvcjp2YXIoLS1pbmspfQoudHQtcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo0cHh9Ci50dC1yIHN0cm9uZ3tjb2xvcjp2YXIoLS1pbmspfQoudHQtbmFyewogIG1hcmdpbi10b3A6OHB4O3BhZGRpbmctdG9wOjhweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci50dC1uYXIgc3Ryb25ne2NvbG9yOnZhcigtLWRpbSk7ZGlzcGxheTpibG9jazttYXJnaW4tYm90dG9tOjJweH0KCi8qIFNUQVRFIFBBTkVMICovCi5zdGF0ZS1wYW5lbHsKICBiYWNrZ3JvdW5kOnZhcigtLXN1cmYpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjE2cHg7YmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcGFkZGluZzoyMHB4O292ZXJmbG93LXk6YXV0bzttYXgtaGVpZ2h0Ojc4MHB4OwogIG1pbi13aWR0aDowO292ZXJmbG93LXg6aGlkZGVuOwp9Ci5zdGF0ZS1wYW5lbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6M3B4fQouc3RhdGUtcGFuZWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6dmFyKC0tYm9yZGVyMik7Ym9yZGVyLXJhZGl1czoycHh9CgoucGFuZWwtZW1wdHl7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjsKICBoZWlnaHQ6MTAwJTttaW4taGVpZ2h0OjMyMHB4O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MzJweCAyMHB4Owp9Ci5wYW5lbC1lbXB0eSBzdmd7b3BhY2l0eTowLjE1O21hcmdpbi1ib3R0b206MThweH0KLnBhbmVsLWVtcHR5IC5wZS10e2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MThweDtmb250LXN0eWxlOml0YWxpYztjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi1ib3R0b206OHB4fQoucGFuZWwtZW1wdHkgLnBlLXN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDRlbTtsaW5lLWhlaWdodDoxLjd9CgovKiBzdGF0ZSBwYW5lbCBpbnRlcm5hbHMgKi8KLnNwLWhlYWR7CiAgZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7CiAgbWFyZ2luLWJvdHRvbToxNnB4O3BhZGRpbmctYm90dG9tOjE0cHg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQouc3AtZWt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO2NvbG9yOnZhcigtLWZhaW50KTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zcC1uYW1le2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6MjhweDtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtsaW5lLWhlaWdodDoxO2NvbG9yOnZhcigtLWluayl9Ci5mYXYtYnRuewogIGJhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIyKTtjb2xvcjp2YXIoLS1mYWludCk7CiAgd2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjZweDtjdXJzb3I6cG9pbnRlcjsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7dHJhbnNpdGlvbjphbGwgMC4xOHM7cGFkZGluZzowO2ZsZXgtc2hyaW5rOjA7Cn0KLmZhdi1idG46aG92ZXJ7Y29sb3I6dmFyKC0tZGltKTtib3JkZXItY29sb3I6dmFyKC0tZGltKX0KLmZhdi1idG4ub257Y29sb3I6dmFyKC0tYWNjZW50KTtib3JkZXItY29sb3I6cmdiYSgyMjQsOTAsNDAsMC4zKTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpfQouZmF2LWJ0biBzdmd7d2lkdGg6MTNweDtoZWlnaHQ6MTNweH0KCi8qIG5hcnJhdGl2ZSB0aW1lbGluZSDigJQgdGhlIHNpZ25hdHVyZSBmZWF0dXJlICovCi5uYXItdGltZWxpbmV7CiAgbWFyZ2luLWJvdHRvbToxNnB4Owp9Ci5udC1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQoubnQtZmxvd3sKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDowOwogIHBvc2l0aW9uOnJlbGF0aXZlO3BhZGRpbmctbGVmdDoxNnB4Owp9Ci5udC1mbG93OjpiZWZvcmV7CiAgY29udGVudDonJztwb3NpdGlvbjphYnNvbHV0ZTtsZWZ0OjVweDt0b3A6NnB4O2JvdHRvbTo2cHg7d2lkdGg6MXB4OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSx2YXIoLS1hY2NlbnQpLHZhcigtLWJvcmRlcikpO29wYWNpdHk6MC40Owp9Ci5udC1zdGVwewogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0O2dhcDoxMHB4OwogIHBhZGRpbmc6NXB4IDA7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm50LWRvdHsKICB3aWR0aDoxMHB4O2hlaWdodDoxMHB4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXgtc2hyaW5rOjA7CiAgcG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTZweDt0b3A6N3B4OwogIGJvcmRlcjoxLjVweCBzb2xpZCBjdXJyZW50Q29sb3I7YmFja2dyb3VuZDp2YXIoLS1iZyk7Cn0KLm50LXN0ZXAucGFzdCAubnQtZG90e2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC1kb3R7Y29sb3I6dmFyKC0tYWNjZW50KTtib3gtc2hhZG93OjAgMCA4cHggcmdiYSgyMjQsOTAsNDAsMC40KX0KLm50LXN0ZXAuY3VycmVudCAubnQtZG90e2NvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDEwcHggcmdiYSgyMjQsOTAsNDAsMC41KX0KLm50LWNvbnRlbnR7ZmxleDoxfQoubnQtdG9waWN7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjUwMDtsaW5lLWhlaWdodDoxLjN9Ci5udC1zdGVwLnBhc3QgLm50LXRvcGlje2NvbG9yOnZhcigtLWZhaW50KX0KLm50LXN0ZXAucmVjZW50IC5udC10b3BpY3tjb2xvcjp2YXIoLS1kaW0pfQoubnQtd2hlbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KCi8qIGluc2lnaHQgYmxvY2sgKi8KLmluc2lnaHR7CiAgbWFyZ2luLWJvdHRvbToxNHB4OwogIHBhZGRpbmc6MTJweCAxNHB4IDEycHggMTZweDsKICBib3JkZXItbGVmdDoxLjVweCBzb2xpZCB2YXIoLS1hY2NlbnQpOwogIGJhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMyk7Ym9yZGVyLXJhZGl1czowIDhweCA4cHggMDsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjEzLjVweDtmb250LXN0eWxlOml0YWxpYzsKICBjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNTU7Zm9udC13ZWlnaHQ6MzAwOwp9CgovKiBjb21wYWN0IHNjb3JlIHN0cmlwICovCi5zY29yZS1zdHJpcHsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxNnB4OwogIHBhZGRpbmc6OHB4IDEycHg7Ym9yZGVyLXJhZGl1czo3cHg7CiAgYmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBtYXJnaW4tYm90dG9tOjE0cHg7Cn0KLnNzLWl0ZW17ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4fQouc3MtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjE1ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KX0KLnNzLXZhbHtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6MzAwO2xldHRlci1zcGFjaW5nOi0wLjAyZW07Y29sb3I6dmFyKC0taW5rKX0KLnNzLWRlbHRhe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6MnB4IDdweDtib3JkZXItcmFkaXVzOjNweH0KLnNzLWRlbHRhLnVwe2NvbG9yOiNlMDYwMzA7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjEpfQouc3MtZGVsdGEuZG57Y29sb3I6IzNiYjhkODtiYWNrZ3JvdW5kOnJnYmEoNTksMTg0LDIxNiwwLjEpfQouc3MtZGl2aWRlcnt3aWR0aDoxcHg7aGVpZ2h0OjMycHg7YmFja2dyb3VuZDp2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjB9Ci5zcy1uYXJ7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMH0KCi5zcC1zZWN0aW9ue21hcmdpbi1ib3R0b206MTRweH0KLnNwLXNlYy10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo5cHg7Cn0KCi8qIG5hcnJhdGl2ZXMgKi8KLm5hci1saXN0e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjZweH0KLm5hci1pdGVtMntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvO2dhcDo2cHg7YWxpZ24taXRlbXM6Y2VudGVyfQoubmktbGFiZWx7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1pbmspfQoubmktdmFse2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KX0KLm5pLXRyYWNre2dyaWQtY29sdW1uOjEvLTE7aGVpZ2h0OjEuNXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA0KTtib3JkZXItcmFkaXVzOjFweDtvdmVyZmxvdzpoaWRkZW47bWFyZ2luLXRvcDotM3B4fQoubmktZmlsbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweDt0cmFuc2l0aW9uOndpZHRoIDAuN3N9CgovKiBtb3ZlbWVudCAqLwoubXYtZ3JpZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjdweH0KLm12LWJsb2Nre2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czo3cHg7cGFkZGluZzo5cHh9Ci5tdi1oe2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xNGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5tdi1ibG9jay51cCAubXYtaHtjb2xvcjp2YXIoLS1yaXNlKX0KLm12LWJsb2NrLmRuIC5tdi1oe2NvbG9yOnZhcigtLWZhbGwpfQoubXYtaXR7Zm9udC1zaXplOjEwLjVweDtwYWRkaW5nOjRweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y29sb3I6dmFyKC0tZmFpbnQpfQoubXYtaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLm12LWl0IHN0cm9uZ3tjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtd2VpZ2h0OjUwMDtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMXB4fQoubXYtaXQgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4fQoKLyogZW1vdGlvbiAqLwouZW0tcm93e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHh9Ci5lbS1kb251dHt3aWR0aDo3NnB4O2hlaWdodDo3NnB4O2ZsZXgtc2hyaW5rOjB9Ci5lbS1sZWd7ZmxleDoxO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjRweH0KLmVtLWl0ZW17ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4fQouZW0tc3d7d2lkdGg6NnB4O2hlaWdodDo2cHg7Ym9yZGVyLXJhZGl1czoycHg7ZmxleC1zaHJpbms6MH0KLmVtLW57ZmxleDoxO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKX0KLmVtLXB7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjkuNXB4O2NvbG9yOnZhcigtLWluayl9CgovKiB0aW1lbGluZSBjaGFydCAqLwoudGwtd3JhcHtoZWlnaHQ6NzJweH0KCi8qIGFydGljbGVzICovCi5hcnQtbGlzdHtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo1cHh9Ci5hcnQtaXRlbXsKICBkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjdweCA5cHg7Ym9yZGVyLXJhZGl1czo2cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAxKTsKICB0cmFuc2l0aW9uOmFsbCAwLjEyczsKfQouYXJ0LWl0ZW06aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDIpO2JvcmRlci1jb2xvcjp2YXIoLS1ib3JkZXIyKX0KLmFydC1zcmN7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjcuNXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tZmFpbnQpO2ZsZXgtc2hyaW5rOjA7d2lkdGg6NDRweDtwYWRkaW5nLXRvcDoxcHh9Ci5hcnQtdHh0e2ZvbnQtc2l6ZToxMXB4O2xpbmUtaGVpZ2h0OjEuNDtjb2xvcjp2YXIoLS1kaW0pfQoKLyogTkFSUkFUSVZFIElOVEVMTElHRU5DRSBST1cgKi8KLm5hci1yb3d7CiAgcG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxOwogIG1heC13aWR0aDoxNDgwcHg7bWFyZ2luOjAgYXV0bzsKICBwYWRkaW5nOjAgMzZweCAyOHB4OwogIGRpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyIDFmcjtnYXA6MThweDsKfQoubmFyLWNhcmR7CiAgYmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxNHB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDE0cHgpO292ZXJmbG93OmhpZGRlbjsKICBkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOwp9Ci5uYy1oZWFkewogIHBhZGRpbmc6MTZweCAyMHB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7ZmxleC1zaHJpbms6MDsKfQoubmMtYm9keXtwYWRkaW5nOjhweCAyMHB4IDE2cHg7ZmxleDoxO292ZXJmbG93LXk6YXV0bzt9Ci5uYy10aXRsZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE2cHg7Zm9udC13ZWlnaHQ6NDAwO2xldHRlci1zcGFjaW5nOi0wLjAxZW07Y29sb3I6dmFyKC0taW5rKX0KLm5jLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDVlbTttYXJnaW4tdG9wOjJweH0KLm5jLWJvZHl7cGFkZGluZzoxM3B4IDE2cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MH0KCi5tb20taXR7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4OwogIHBhZGRpbmc6N3B4IDA7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKfQoubW9tLWl0OmZpcnN0LW9mLXR5cGV7Ym9yZGVyLXRvcDpub25lO3BhZGRpbmctdG9wOjB9Ci5tb20tcmt7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjp2YXIoLS1mYWludCk7d2lkdGg6MTNweDtmbGV4LXNocmluazowfQoubW9tLWluZntmbGV4OjF9Ci5tb20tbm17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5tb20tc3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjFweH0KLm1vbS1wY3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2ZvbnQtd2VpZ2h0OjQwMDtmbGV4LXNocmluazowfQoubW9tLXBjLnJ7Y29sb3I6dmFyKC0tcmlzZSl9Ci5tb20tcGMuZntjb2xvcjp2YXIoLS1mYWxsKX0KLm1vbS10cntoZWlnaHQ6MS41cHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpO2JvcmRlci1yYWRpdXM6MXB4O21hcmdpbjozcHggMCAwO292ZXJmbG93OmhpZGRlbn0KLm1vbS1mbHtoZWlnaHQ6MTAwJTtib3JkZXItcmFkaXVzOjFweH0KCi5yZWctaXR7CiAgZGlzcGxheTpmbGV4O2dhcDo5cHg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDsKICBwYWRkaW5nOjhweCAwO2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjpvcGFjaXR5IDAuMTVzOwp9Ci5yZWctaXQ6Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmU7cGFkZGluZy10b3A6MH0KLnJlZy1pdDpob3ZlcntvcGFjaXR5OjAuNzV9Ci5yZWctYmFkZ2V7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjA3ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIHBhZGRpbmc6MnB4IDZweDtib3JkZXItcmFkaXVzOjNweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuMDcpO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNCk7CiAgY29sb3I6dmFyKC0tYWNjZW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4O3doaXRlLXNwYWNlOm5vd3JhcDsKfQoucmVnLWZse2ZsZXg6MTtmb250LXNpemU6MTEuNXB4O2xpbmUtaGVpZ2h0OjEuNX0KLnJlZy1mcm9te2NvbG9yOnZhcigtLWZhaW50KX0KLnJlZy1hcnJ7Y29sb3I6dmFyKC0tYWNjZW50KTtvcGFjaXR5OjAuNTttYXJnaW46MCA0cHh9Ci5yZWctdG97Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo1MDB9Ci5yZWctdG17Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTtmbGV4LXNocmluazowO21hcmdpbi10b3A6MnB4fQoKLyogRkFWUyAqLwouZmF2c3sKICBwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjE7CiAgbWF4LXdpZHRoOjE0ODBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAzNnB4IDQwcHg7Cn0KLmZhdnMtbGFiZWx7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWJvdHRvbToxMHB4fQouZmF2cy1yb3d7ZGlzcGxheTpmbGV4O2dhcDoxMHB4O292ZXJmbG93LXg6YXV0bztwYWRkaW5nLWJvdHRvbTozcHh9Ci5mYXZzLXJvdzo6LXdlYmtpdC1zY3JvbGxiYXJ7aGVpZ2h0OjJweH0KLmZhdnMtcm93Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWJvcmRlcjIpO2JvcmRlci1yYWRpdXM6MXB4fQouZmF2LWNhcmR7CiAgZmxleDowIDAgMTkwcHg7YmFja2dyb3VuZDp2YXIoLS1zdXJmKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MTJweDtjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOmFsbCAwLjE4czsKfQouZmF2LWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMjIpO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC4wMil9Ci5mYy1oZWFke2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjdweH0KLmZjLW5hbWV7Zm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxNXB4O2ZvbnQtd2VpZ2h0OjQwMDtjb2xvcjp2YXIoLS1pbmspfQouZmMtc2N7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpfQouZmMtcm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2Vlbjtmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDozcHh9Ci5mYy1yb3cgLnZ7Y29sb3I6dmFyKC0tZGltKTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OS41cHh9Ci5mYXZzLWVtcHR5e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWZhaW50KTtmb250LXN0eWxlOml0YWxpYztwYWRkaW5nOjRweCAwfQoKLyogRk9PVCAqLwouZm9vdHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjQ4cHggMzZweCA2MHB4O21heC13aWR0aDo1ODBweDttYXJnaW46MCBhdXRvO3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MX0KLmZvb3QtbmFtZXtmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE1cHg7Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6dmFyKC0tZGltKTtsZXR0ZXItc3BhY2luZzotMC4wMWVtO21hcmdpbi1ib3R0b206MTRweH0KLmZvb3QtbGluZXtmb250LWZhbWlseTp2YXIoLS1zYW5zKTtmb250LXNpemU6MTJweDtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0tZmFpbnQpO2xpbmUtaGVpZ2h0OjEuODttYXJnaW4tYm90dG9tOjEycHh9Ci5mb290LXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7bGV0dGVyLXNwYWNpbmc6MC4yZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnJnYmEoNjIsNzcsOTYsMC41KX0KCi8qIGFuaW1hdGlvbnMgKi8KQGtleWZyYW1lcyBmYWRlVXB7ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoubWFwLWNhcmQsLnN0YXRlLXBhbmVsLC5uYXItY2FyZCwuc2lnbmF0dXJlLWluc2lnaHR7YW5pbWF0aW9uOmZhZGVVcCAwLjU1cyBjdWJpYy1iZXppZXIoLjIsLjgsLjIsMSkgYmFja3dhcmRzfQoubmFyLWNhcmQ6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjA3c30KLm5hci1jYXJkOm50aC1jaGlsZCgzKXthbmltYXRpb24tZGVsYXk6MC4xNHN9Ci5zaWduYXR1cmUtaW5zaWdodHthbmltYXRpb24tZGVsYXk6MC4wNXN9CgpAbWVkaWEobWF4LXdpZHRoOjExMDBweCl7CiAgLm1haW57Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAuc3RhdGUtcGFuZWx7bWF4LWhlaWdodDpub25lfQogIC5uYXItcm93e2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9Cn0KCi8qIOKUgOKUgCBXSEFUIElORElBIElTIFJFQUNUSU5HIFRPIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwoud2lyLXNlY3Rpb257CiAgZmxleDoxO21pbi13aWR0aDowOwogIHBhZGRpbmc6MDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxNHB4OwogIGJhY2tncm91bmQ6dmFyKC0tc3VyZik7CiAgYmFja2Ryb3AtZmlsdGVyOmJsdXIoMTZweCk7CiAgcG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuOwogIGRpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Cn0KLndpci1oZWFkZXJ7CiAgcGFkZGluZzoxOHB4IDIycHggMTRweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Cn0KLndpci10aXRsZXsKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuM2VtOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO29wYWNpdHk6MC44NTsKfQoud2lyLWxpdmV7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIGZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMWVtOwp9Ci53aXItbGl2ZS1kb3R7CiAgd2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDojMzlmZjE0OwogIGJveC1zaGFkb3c6MCAwIDZweCByZ2JhKDU3LDI1NSwyMCwwLjYpOwogIGFuaW1hdGlvbjp3aXJMaXZlUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyB3aXJMaXZlUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjF9NTAle29wYWNpdHk6MC4zfX0KLndpci1zaWduYWxze2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47ZmxleDoxO292ZXJmbG93OmhpZGRlbn0KLndpci1zaWduYWx7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjA7CiAgcGFkZGluZzoxM3B4IDIycHg7CiAgYm9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzNSk7CiAgb3BhY2l0eTowOwogIGFuaW1hdGlvbjp3aXJGYWRlSW4gMC42cyBlYXNlIGZvcndhcmRzOwogIHBvc2l0aW9uOnJlbGF0aXZlO2N1cnNvcjpkZWZhdWx0OwogIHRyYW5zaXRpb246YmFja2dyb3VuZCAwLjE1czsKfQoud2lyLXNpZ25hbDpob3ZlcntiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMil9Ci53aXItc2lnbmFsOmxhc3QtY2hpbGR7Ym9yZGVyLWJvdHRvbTpub25lfQpAa2V5ZnJhbWVzIHdpckZhZGVJbntmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNsYXRlWCgtNnB4KX10b3tvcGFjaXR5OjE7dHJhbnNmb3JtOm5vbmV9fQoud2lyLXNpZ25hbC1iYXJ7CiAgd2lkdGg6MnB4O2JvcmRlci1yYWRpdXM6MXB4O2ZsZXgtc2hyaW5rOjA7CiAgbWFyZ2luLXJpZ2h0OjE0cHg7bWFyZ2luLXRvcDo0cHg7CiAgYWxpZ24tc2VsZjpzdHJldGNoO21pbi1oZWlnaHQ6MTZweDsKICBvcGFjaXR5OjAuNjsKfQoud2lyLXNpZ25hbC1jb250ZW50e2ZsZXg6MTttaW4td2lkdGg6MH0KLndpci1zaWduYWwtdGV4dHsKICBmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjE0LjVweDtmb250LXdlaWdodDozMDA7CiAgY29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjU7bGV0dGVyLXNwYWNpbmc6LTAuMDFlbTsKfQoud2lyLXNpZ25hbC10ZXh0IGVte2ZvbnQtc3R5bGU6aXRhbGljO2NvbG9yOmluaGVyaXQ7b3BhY2l0eTowLjh9Ci53aXItc2lnbmFsLW1ldGF7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi10b3A6NHB4Owp9Ci53aXItc2lnbmFsLXRhZ3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6N3B4O2xldHRlci1zcGFjaW5nOjAuMTRlbTsKICB0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7b3BhY2l0eTowLjQ1Owp9Ci53aXItc2lnbmFsLWxvY3sKICBmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpOwp9Ci53aXItbG9hZGluZ3sKICBkaXNwbGF5OmZsZXg7Z2FwOjZweDtwYWRkaW5nOjIwcHggMjJweDthbGlnbi1pdGVtczpjZW50ZXI7Cn0KLndpci1kb3R7d2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjQpO2FuaW1hdGlvbjp3aXJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZX0KLndpci1kb3Q6bnRoLWNoaWxkKDIpe2FuaW1hdGlvbi1kZWxheTowLjJzfQoud2lyLWRvdDpudGgtY2hpbGQoMyl7YW5pbWF0aW9uLWRlbGF5OjAuNHN9CkBrZXlmcmFtZXMgd2lyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KCi8qIOKUgOKUgCBTVEFUUyBTVFJJUCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KI3N0YXRzLXN0cmlwe292ZXJmbG93OmhpZGRlbn0KLnN0YXQtY2VsbHsKICBmbGV4OjE7bWluLXdpZHRoOjA7cGFkZGluZzoxNHB4IDE4cHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICBqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoycHg7Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCByZ2JhKDE2MCwxOTAsMjMwLDAuMDcpOwp9Ci5zdGF0LWNlbGw6bGFzdC1jaGlsZHtib3JkZXItcmlnaHQ6bm9uZX0KLnN0YXQtZGl2e3dpZHRoOjFweDtiYWNrZ3JvdW5kOnJnYmEoMTYwLDE5MCwyMzAsMC4wNyk7ZmxleC1zaHJpbms6MDttYXJnaW46OHB4IDB9Ci5zYy1sYWJlbHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCk7d2hpdGUtc3BhY2U6bm93cmFwfQouc2MtdmFse2ZvbnQtZmFtaWx5OnZhcigtLXNlcmlmKTtmb250LXNpemU6Y2xhbXAoMTVweCwxLjZ2dywyMnB4KTtmb250LXdlaWdodDozMDA7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjE7d2hpdGUtc3BhY2U6bm93cmFwO292ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzfQouc2MtdmFsLXNte2ZvbnQtc2l6ZTpjbGFtcCgxM3B4LDEuMnZ3LDE2cHgpIWltcG9ydGFudH0KLnNjLXN1Yntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OC41cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4O3doaXRlLXNwYWNlOm5vd3JhcDtvdmVyZmxvdzpoaWRkZW47dGV4dC1vdmVyZmxvdzplbGxpcHNpc30KLnNjLWhvdmVyYWJsZXtwb3NpdGlvbjpyZWxhdGl2ZTtjdXJzb3I6ZGVmYXVsdH0KLnNjLXRvb2x0aXB7CiAgZGlzcGxheTpub25lO3Bvc2l0aW9uOmFic29sdXRlO2JvdHRvbTpjYWxjKDEwMCUgKyA4cHgpO2xlZnQ6NTAlOwogIHRyYW5zZm9ybTp0cmFuc2xhdGVYKC01MCUpOwogIGJhY2tncm91bmQ6cmdiYSg4LDEyLDIwLDAuOTcpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxMnB4IDE0cHg7d2lkdGg6MjAwcHg7CiAgZm9udC1mYW1pbHk6dmFyKC0tc2Fucyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjU7CiAgei1pbmRleDo5OTk5O3BvaW50ZXItZXZlbnRzOm5vbmU7d2hpdGUtc3BhY2U6bm9ybWFsO3RleHQtYWxpZ246bGVmdDsKICBib3gtc2hhZG93OjAgOHB4IDI0cHggcmdiYSgwLDAsMCwwLjUpOwp9Ci5zYy1ob3ZlcmFibGU6aG92ZXIgLnNjLXRvb2x0aXB7ZGlzcGxheTpibG9ja30KLnNjLXRpcC10aXRsZXtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1hY2NlbnQpO21hcmdpbi1ib3R0b206NnB4fQouc2MtdGlwLXJvd3tkaXNwbGF5OmZsZXg7Z2FwOjZweDttYXJnaW4tYm90dG9tOjRweDtmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pfQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PgoKPGRpdiBpZD0ibHRhYi10b29sdGlwIj48L2Rpdj4KCjwhLS0gTE9BREVSIC0tPgo8ZGl2IGlkPSJhcHAtbG9hZGVyIiBzdHlsZT0icG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDt6LWluZGV4Ojk5OTk4O2JhY2tncm91bmQ6IzA2MDkxMDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO3RyYW5zaXRpb246b3BhY2l0eSAwLjhzIGVhc2UsdmlzaWJpbGl0eSAwLjhzIGVhc2U7Ij4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt3aWR0aDo2NHB4O2hlaWdodDo2NHB4O21hcmdpbi1ib3R0b206MzZweCI+CiAgICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDoyNHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6I2UwNWEyODthbmltYXRpb246bGRyUHVsc2UgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuNCk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGUiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JvcmRlcjoxcHggc29saWQgcmdiYSgyMjQsOTAsNDAsMC4xNSk7YW5pbWF0aW9uOmxkclJpbmcgMnMgZWFzZS1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuNXMiPjwvZGl2PgogICAgPGRpdiBzdHlsZT0icG9zaXRpb246YWJzb2x1dGU7aW5zZXQ6LTEwcHg7Ym9yZGVyLXJhZGl1czo1MCU7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyNCw5MCw0MCwwLjA3KTthbmltYXRpb246bGRyUmluZyAycyBlYXNlLW91dCBpbmZpbml0ZTthbmltYXRpb24tZGVsYXk6MXMiPjwvZGl2PgogIDwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidQbGF5ZmFpciBEaXNwbGF5JyxHZW9yZ2lhLHNlcmlmO2ZvbnQtc2l6ZTpjbGFtcCgyOHB4LDV2dyw0MnB4KTtmb250LXdlaWdodDozMDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTtjb2xvcjojZjBlY2U0O2xpbmUtaGVpZ2h0OjE7bWFyZ2luLWJvdHRvbToxMHB4Ij4KICAgIDxlbSBzdHlsZT0iY29sb3I6I2U4YzRhMDtmb250LXN0eWxlOml0YWxpYyI+UHVsc2U8L2VtPiBvZiBJbmRpYQogIDwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMXB4O2xldHRlci1zcGFjaW5nOjAuMjhlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjQpO21hcmdpbi1ib3R0b206MjhweCI+VGhlIG1vdmVtZW50IGJlbmVhdGggdGhlIGhlYWRsaW5lczwvZGl2PgogIDxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OidDb3VyaWVyIE5ldycsbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB4O2xldHRlci1zcGFjaW5nOjAuMThlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNjAsMTkwLDIzMCwwLjI1KTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMHB4Ij4KICAgIDxzcGFuPk5vdCBuZXdzPC9zcGFuPjxzcGFuIHN0eWxlPSJvcGFjaXR5OjAuMyI+wrc8L3NwYW4+PHNwYW4+Tm90IHByZWRpY3Rpb248L3NwYW4+PHNwYW4gc3R5bGU9Im9wYWNpdHk6MC4zIj7Ctzwvc3Bhbj4KICAgIDxzcGFuPkp1c3QgPHNwYW4gc3R5bGU9ImNvbG9yOiMzOWZmMTQ7dGV4dC1zaGFkb3c6MCAwIDEwcHggcmdiYSg1NywyNTUsMjAsMC41KTthbmltYXRpb246bGRyR2xvdyAycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+b2JzZXJ2YXRpb248L3NwYW4+PC9zcGFuPgogIDwvZGl2PgogIDxkaXYgc3R5bGU9Im1hcmdpbi10b3A6NDhweDtkaXNwbGF5OmZsZXg7Z2FwOjZweCI+CiAgICA8c3BhbiBzdHlsZT0id2lkdGg6NHB4O2hlaWdodDo0cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjUpO2FuaW1hdGlvbjpsZHJEb3QgMS4ycyBlYXNlLWluLW91dCBpbmZpbml0ZSI+PC9zcGFuPgogICAgPHNwYW4gc3R5bGU9IndpZHRoOjRweDtoZWlnaHQ6NHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6cmdiYSgyMjQsOTAsNDAsMC41KTthbmltYXRpb246bGRyRG90IDEuMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7YW5pbWF0aW9uLWRlbGF5OjAuMnMiPjwvc3Bhbj4KICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo0cHg7aGVpZ2h0OjRweDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnJnYmEoMjI0LDkwLDQwLDAuNSk7YW5pbWF0aW9uOmxkckRvdCAxLjJzIGVhc2UtaW4tb3V0IGluZmluaXRlO2FuaW1hdGlvbi1kZWxheTowLjRzIj48L3NwYW4+CiAgPC9kaXY+CjwvZGl2Pgo8c3R5bGU+CkBrZXlmcmFtZXMgbGRyUHVsc2V7MCUsMTAwJXtvcGFjaXR5OjE7dHJhbnNmb3JtOnNjYWxlKDEpfTUwJXtvcGFjaXR5OjAuNTt0cmFuc2Zvcm06c2NhbGUoMC44KX19CkBrZXlmcmFtZXMgbGRyUmluZ3swJXt0cmFuc2Zvcm06c2NhbGUoMC44KTtvcGFjaXR5OjAuNn0xMDAle3RyYW5zZm9ybTpzY2FsZSgxLjUpO29wYWNpdHk6MH19CkBrZXlmcmFtZXMgbGRyR2xvd3swJSwxMDAle3RleHQtc2hhZG93OjAgMCAxMHB4IHJnYmEoNTcsMjU1LDIwLDAuNSl9NTAle3RleHQtc2hhZG93OjAgMCAyMnB4IHJnYmEoNTcsMjU1LDIwLDAuOSksMCAwIDQwcHggcmdiYSg1NywyNTUsMjAsMC4zKX19CkBrZXlmcmFtZXMgbGRyRG90ezAlLDgwJSwxMDAle3RyYW5zZm9ybTpzY2FsZSgwLjYpO29wYWNpdHk6MC4zfTQwJXt0cmFuc2Zvcm06c2NhbGUoMSk7b3BhY2l0eToxfX0KPC9zdHlsZT4KCjxkaXYgY2xhc3M9InRvcGJhciI+CiAgPGRpdiBjbGFzcz0iYnJhbmQiPgogICAgPGRpdiBjbGFzcz0iYnJhbmQtbWFyayI+PHNwYW4gY2xhc3M9ImJyYW5kLXB1bHNlLWRvdCI+PC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0iYnJhbmQtdGV4dC1ibG9jayI+CiAgICAgIDxzcGFuIGNsYXNzPSJicmFuZC1uYW1lIj48ZW0gY2xhc3M9ImJyYW5kLXB1bHNlLXdvcmQiPlB1bHNlPC9lbT4gb2YgSW5kaWE8L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJicmFuZC10YWdsaW5lIj5UaGUgbW92ZW1lbnQgYmVuZWF0aCB0aGUgaGVhZGxpbmVzLjwvc3Bhbj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9InRvcGJhci1yIj4KICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci13cmFwIj4KICAgICAgPGRpdiBjbGFzcz0ibGl2ZS1pbmRpY2F0b3IiPgogICAgICAgIDxzcGFuIGNsYXNzPSJsaXZlLWRvdCI+PC9zcGFuPgogICAgICAgIDxzcGFuIGlkPSJsaXZlLWNvdW50Ij7igKY8L3NwYW4+IHNpZ25hbHMKICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci10aXAiPgogICAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci1sYWJlbCI+T2JzZXJ2ZWQgZnJvbTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InNpZy1ob3Zlci1zb3VyY2VzIj5yZWdpb25hbCBtZWRpYSDCtyBwdWJsaWMgZGlzY3Vzc2lvbiDCtyBpbmRlcGVuZGVudCByZXBvcnRpbmcgwrcgc29jaWFsIHNpZ25hbHM8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNsb2NrIiBpZD0iY2xvY2siPi0tOi0tOi0tIElTVDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjwhLS0gSEVSTyAtLT4KPHNlY3Rpb24gY2xhc3M9Imhlcm8iIHN0eWxlPSJwYWRkaW5nLXRvcDo4MHB4O3BhZGRpbmctYm90dG9tOjI0cHg7cG9zaXRpb246cmVsYXRpdmU7b3ZlcmZsb3c6aGlkZGVuIj4KICA8ZGl2IHN0eWxlPSJwb3NpdGlvbjphYnNvbHV0ZTt3aWR0aDo2MDBweDtoZWlnaHQ6MzUwcHg7dG9wOi02MHB4O2xlZnQ6LTgwcHg7YmFja2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSBhdCA0MCUgNTAlLHJnYmEoMjI0LDkwLDQwLDAuMDUpIDAlLHRyYW5zcGFyZW50IDY1JSk7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjA7YW5pbWF0aW9uOmFtYmllbnRTaGlmdCAxMnMgZWFzZS1pbi1vdXQgaW5maW5pdGUgYWx0ZXJuYXRlIj48L2Rpdj4KICA8c3R5bGU+QGtleWZyYW1lcyBhbWJpZW50U2hpZnR7MCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoMCl9MTAwJXt0cmFuc2Zvcm06dHJhbnNsYXRlWCgyNHB4KSB0cmFuc2xhdGVZKC0xMnB4KX19PC9zdHlsZT4KICA8ZGl2IGNsYXNzPSJoZXJvLWV5ZWJyb3ciIHN0eWxlPSJwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjEiPkNvbGxlY3RpdmUgYXR0ZW50aW9uICZtaWRkb3Q7IEluZGlhPC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1icmFuZC1ibG9jayIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjE4cHg7bWFyZ2luLWJvdHRvbToxNnB4O3Bvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MSI+CiAgICA8ZGl2IGNsYXNzPSJoZXJvLXB1bHNlLXNpZ25hbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJocHMtY29yZSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMSI+PC9zcGFuPjxzcGFuIGNsYXNzPSJocHMtcmluZyByMiI+PC9zcGFuPgogICAgPC9kaXY+CiAgICA8aDEgY2xhc3M9Imhlcm8tYnJhbmQtbmFtZSI+PGVtPlB1bHNlPC9lbT4gb2YgSW5kaWE8L2gxPgogIDwvZGl2PgogIDxwIGNsYXNzPSJoZXJvLXRhZ2xpbmUiPlRoZSBtb3ZlbWVudCBiZW5lYXRoIHRoZSBoZWFkbGluZXMuPC9wPgogIDxwIGNsYXNzPSJoZXJvLWRlc2MiPk9ic2VydmUgaG93IEluZGlhJ3MgbmFycmF0aXZlcyBhbmQgcHVibGljIGF0dGVudGlvbiBzaGlmdCBpbiByZWFsIHRpbWUuPC9wPgogIDxwIGNsYXNzPSJoZXJvLXN1Yi1saW5lIj5PYnNlcnZpbmcgSW5kaWEgaW4gbW90aW9uLjwvcD4KCiAgPCEtLSBMSVZFIFNUQVRTIFNUUklQIC0tPgo8ZGl2IGlkPSJzdGF0cy1zdHJpcCIgc3R5bGU9InBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MjtiYWNrZ3JvdW5kOnJnYmEoOSwxMywyMSwwLjkpO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMTYwLDE5MCwyMzAsMC4wOCk7cGFkZGluZzowIDM2cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOnN0cmV0Y2g7Ij4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5TaWduYWxzIHRyYWNrZWQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwgc2MtdmFsLXNtIiBpZD0ic2Mtc2lnbmFscy12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLXNpZ25hbHMtc3ViIj5sb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiIG9uY2xpY2s9InNlbGVjdEhvdHRlc3QoKSIgc3R5bGU9ImN1cnNvcjpwb2ludGVyIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+SGlnaGVzdCBhdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1ob3R0ZXN0LXZhbCI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2Mtc3ViIiBpZD0ic2MtaG90dGVzdC1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1ob3R0ZXN0LXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5QZWFrIGFuZ2VyIHN0YXRlPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIiBpZD0ic2MtYW5nZXItdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1hbmdlci1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1hbmdlci10aXAiPjwvZGl2PjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtZGl2Ij48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWNlbGwgc2MtaG92ZXJhYmxlIj48ZGl2IGNsYXNzPSJzYy1sYWJlbCI+RmFzdGVzdCByaXNpbmc8L2Rpdj48ZGl2IGNsYXNzPSJzYy12YWwiIGlkPSJzYy1yaXNpbmctdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1yaXNpbmctc3ViIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy10b29sdGlwIiBpZD0ic2MtcmlzaW5nLXRpcCI+PC9kaXY+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1kaXYiPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQtY2VsbCBzYy1ob3ZlcmFibGUiPjxkaXYgY2xhc3M9InNjLWxhYmVsIj5Ub3AgbmFycmF0aXZlPC9kaXY+PGRpdiBjbGFzcz0ic2MtdmFsIHNjLXZhbC1zbSIgaWQ9InNjLW5hci12YWwiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXN1YiIgaWQ9InNjLW5hci1zdWIiPuKAlDwvZGl2PjxkaXYgY2xhc3M9InNjLXRvb2x0aXAiIGlkPSJzYy1uYXItdGlwIj48L2Rpdj48L2Rpdj4KICA8ZGl2IGNsYXNzPSJzdGF0LWRpdiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdC1jZWxsIHNjLWhvdmVyYWJsZSI+PGRpdiBjbGFzcz0ic2MtbGFiZWwiPkxlYXN0IGFjdGl2ZTwvZGl2PjxkaXYgY2xhc3M9InNjLXZhbCIgaWQ9InNjLWNvb2wtdmFsIj7igJQ8L2Rpdj48ZGl2IGNsYXNzPSJzYy1zdWIiIGlkPSJzYy1jb29sLXN1YiI+4oCUPC9kaXY+PGRpdiBjbGFzcz0ic2MtdG9vbHRpcCIgaWQ9InNjLWNvb2wtdGlwIj48L2Rpdj48L2Rpdj4KPC9kaXY+CgogIDwhLS0gU0lHTkFUVVJFIElOU0lHSFQgKyBOQVJSQVRJVkUgU1RSSVAgc2lkZSBieSBzaWRlIC0tPgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6MThweDthbGlnbi1pdGVtczpzdHJldGNoO21hcmdpbi10b3A6MTZweDttYXJnaW4tYm90dG9tOjA7bWF4LXdpZHRoOjE0ODBweDttYXJnaW4tbGVmdDphdXRvO21hcmdpbi1yaWdodDphdXRvO3BhZGRpbmc6MCAzNnB4OyI+CiAgICA8ZGl2IGNsYXNzPSJ3aXItc2VjdGlvbiI+CiAgICAgIDxkaXYgY2xhc3M9Indpci1oZWFkZXIiPgogICAgICAgIDxkaXYgY2xhc3M9Indpci10aXRsZSI+V2hhdCBJbmRpYSBpcyByZWFjdGluZyB0bzwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Indpci1saXZlIj48c3BhbiBjbGFzcz0id2lyLWxpdmUtZG90Ij48L3NwYW4+bGl2ZSBzaWduYWxzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ3aXItc2lnbmFscyIgaWQ9Indpci1zaWduYWxzIj4KICAgICAgICA8ZGl2IGNsYXNzPSJ3aXItbG9hZGluZyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0id2lyLWRvdCI+PC9zcGFuPjxzcGFuIGNsYXNzPSJ3aXItZG90Ij48L3NwYW4+PHNwYW4gY2xhc3M9Indpci1kb3QiPjwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MCAwIDM2MHB4O2JhY2tncm91bmQ6dmFyKC0tc3VyZik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTRweDtiYWNrZHJvcC1maWx0ZXI6Ymx1cigxMnB4KTtvdmVyZmxvdzpoaWRkZW47ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsiPgogICAgICA8IS0tIGhlYWRlciAtLT4KICAgICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2ZsZXgtc2hyaW5rOjA7Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjIyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWZhaW50KSI+UmVjZW50IG5hcnJhdGl2ZSBzaGlmdHM8L3NwYW4+CiAgICAgIDwvZGl2PgogICAgICA8IS0tIHNoaWZ0cyBsaXN0IC0tPgogICAgICA8ZGl2IHN0eWxlPSJmbGV4OjE7b3ZlcmZsb3c6aGlkZGVuO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47anVzdGlmeS1jb250ZW50OmNlbnRlcjtwYWRkaW5nOjEwcHggMTRweDtnYXA6NnB4OyIgaWQ9InNoaWZ0LWxpc3QiPjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCgo8IS0tIE1BSU46IE1BUCArIFNUQVRFIFBBTkVMIC0tPgo8ZGl2IGNsYXNzPSJtYWluIj4KCiAgPGRpdiBjbGFzcz0ibWFwLWNhcmQiPgogICAgPGRpdiBjbGFzcz0ibWFwLXRvcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC10aXRsZS1ibG9jayI+CiAgICAgICAgPGRpdiBjbGFzcz0ibXQiPkluZGlhICZtZGFzaDsgY29sbGVjdGl2ZSBhdHRlbnRpb248L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtcyIgaWQ9Im1hcC1tZXRhIj4zMCBzdGF0ZXMgJm1pZGRvdDsgbGl2ZSBzaWduYWwgY29tcG9zaXRlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJsZWdlbmQiPjxzcGFuPnF1aWV0PC9zcGFuPjxkaXYgY2xhc3M9ImxlZ2VuZC1iYXIiPjwvZGl2PjxzcGFuPmFjdGl2ZTwvc3Bhbj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibGF5ZXItcm93Ij4KICAgICAgPHNwYW4gY2xhc3M9ImxheWVyLWxhYmVsIj5WaWV3PC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJsdGFicyI+CiAgICAgICAgPHNwYW4gY2xhc3M9Imx0YWIiIGRhdGEtbGF5ZXI9ImF0dGVudGlvbiI+QXR0ZW50aW9uIDxzcGFuIGNsYXNzPSJsdGFiLWluZm8iIGRhdGEtdGlwPSJXaGljaCBzdGF0ZXMgYXJlIHJlY2VpdmluZyB0aGUgbW9zdCBwdWJsaWMgZm9jdXMuIEhpZ2ggYXR0ZW50aW9uID0gY29uY2VudHJhdGVkIG5ld3MgY292ZXJhZ2UgYW5kIHBvbGl0aWNhbCBhY3Rpdml0eS4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIGFjdGl2ZSIgZGF0YS1sYXllcj0iZW1vdGlvbiI+RW1vdGlvbiA8c3BhbiBjbGFzcz0ibHRhYi1pbmZvIiBkYXRhLXRpcD0iVGhlIGRvbWluYW50IGVtb3Rpb25hbCB0b25lIOKAlCBhbnhpb3VzLCBhbmdyeSwgaG9wZWZ1bCwgcHJvdWQgb3IgZmVhcmZ1bC4gUmV2ZWFscyB0aGUgcHN5Y2hvbG9naWNhbCB1bmRlcmN1cnJlbnQgb2YgcG9saXRpY2FsIGF0dGVudGlvbi4iPmk8L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJsdGFiIiBkYXRhLWxheWVyPSJ2ZWxvY2l0eSI+TW9tZW50dW0gPHNwYW4gY2xhc3M9Imx0YWItaW5mbyIgZGF0YS10aXA9IklzIGF0dGVudGlvbiByaXNpbmcgb3IgZmFsbGluZz8gUmlzaW5nID0gbmFycmF0aXZlIGFjY2VsZXJhdGluZy4gQ29vbGluZyA9IGxvc2luZyB0cmFjdGlvbi4gU2hvd3Mgc3RhdGVzIGVudGVyaW5nIG9yIGV4aXRpbmcgYSBwb2xpdGljYWwgY3ljbGUuIj5pPC9zcGFuPjwvc3Bhbj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im1hcC1zdmctd3JhcCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcC1pbm5lciI+CiAgICAgICAgPHN2ZyBpZD0iaW5kaWEtbWFwIiB2aWV3Qm94PSIwIDAgODAwIDgwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgbWVldCI+CiAgICAgICAgICA8ZGVmcz4KICAgICAgICAgICAgPHJhZGlhbEdyYWRpZW50IGlkPSJhbWJHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9InJnYmEoMjI0LDkwLDQwLDAuMDQpIi8+CiAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICAgICAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgICAgICAgICA8ZmlsdGVyIGlkPSJzdGF0ZUdsb3ciIHg9Ii0zMCUiIHk9Ii0zMCUiIHdpZHRoPSIxNjAlIiBoZWlnaHQ9IjE2MCUiPgogICAgICAgICAgICAgIDxmZUdhdXNzaWFuQmx1ciBpbj0iU291cmNlR3JhcGhpYyIgc3RkRGV2aWF0aW9uPSI4IiByZXN1bHQ9ImJsdXIiLz4KICAgICAgICAgICAgICA8ZmVDb21wb3NpdGUgaW49IlNvdXJjZUdyYXBoaWMiIGluMj0iYmx1ciIgb3BlcmF0b3I9Im92ZXIiLz4KICAgICAgICAgICAgPC9maWx0ZXI+CiAgICAgICAgICA8L2RlZnM+CiAgICAgICAgICA8cmVjdCB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgZmlsbD0idXJsKCNhbWJHbG93KSIvPgogICAgICAgICAgPGcgaWQ9Im1hcC1nbG93Ij48L2c+CiAgICAgICAgICA8ZyBpZD0ibWFwLXN0YXRlcyI+PC9nPgogICAgICAgICAgPGcgaWQ9Im1hcC1wdWxzZXMiPjwvZz4KICAgICAgICA8L3N2Zz4KICAgICAgICA8ZGl2IGNsYXNzPSJtYXAtdG9vbHRpcCIgaWQ9InRvb2x0aXAiPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFNUQVRFIFBBTkVMIC0tPgogIDxkaXYgY2xhc3M9InN0YXRlLXBhbmVsIiBpZD0ic3RhdGUtZGV0YWlsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLWVtcHR5Ij4KICAgICAgPHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMSI+CiAgICAgICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgOHY0TTEyIDE2aC4wMSIvPgogICAgICA8L3N2Zz4KICAgICAgPGRpdiBjbGFzcz0icGUtdCI+U2VsZWN0IGEgc3RhdGU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0icGUtcyI+Q2xpY2sgYW55IHJlZ2lvbiBvbiB0aGUgbWFwPGJyLz50byBvcGVuIGl0cyBuYXJyYXRpdmUgcGFuZWwuPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCjwvZGl2PgoKPCEtLSBOQVJSQVRJVkUgUk9XIC0tPgo8ZGl2IGNsYXNzPSJuYXItcm93IiBpZD0ibmFyLXJvdyIgc3R5bGU9Im9wYWNpdHk6MDt0cmFuc2l0aW9uOm9wYWNpdHkgMC41cyBlYXNlIj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgcmlzZTIiPjwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9Im5jLXRpdGxlIj5SaXNpbmcgbmFycmF0aXZlczwvc3Bhbj4KICAgICAgPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1sZWZ0OmF1dG8iPmdhaW5pbmcgdHJhY3Rpb248L3NwYW4+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9Im5jLWJvZHkiIGlkPSJyaXNpbmctbGlzdCI+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvYWRpbmcuLi48L2Rpdj48L2Rpdj4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJuYXItY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJuYy1oZWFkIiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4OyI+CiAgICAgIDxzcGFuIGNsYXNzPSJuYy1kb3QgZmFsbCI+PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0ibmMtdGl0bGUiPkRlY2xpbmluZyBuYXJyYXRpdmVzPC9zcGFuPgogICAgICA8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLWxlZnQ6YXV0byI+bG9zaW5nIHRyYWN0aW9uPC9zcGFuPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJuYy1ib2R5IiBpZD0iZGVjbGluaW5nLWxpc3QiPjxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Mb2FkaW5nLi4uPC9kaXY+PC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSBGQVZTIC0tPgo8c2VjdGlvbiBjbGFzcz0iZmF2cyI+CiAgPGRpdiBjbGFzcz0iZmF2cy1sYWJlbCI+VHJhY2tlZCBzdGF0ZXM8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmYXZzLXJvdyIgaWQ9ImZhdi1yb3ciPgogICAgPGRpdiBjbGFzcz0iZmF2cy1lbXB0eSI+Tm8gc3RhdGVzIHRyYWNrZWQuIEJvb2ttYXJrIGFueSBzdGF0ZSBwYW5lbCB0byBmb2xsb3cgaXRzIG5hcnJhdGl2ZSBldm9sdXRpb24uPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjxkaXYgY2xhc3M9ImZvb3QiPgogIDxkaXYgY2xhc3M9ImZvb3QtbmFtZSI+UHVsc2Ugb2YgSW5kaWE8L2Rpdj4KICA8ZGl2IGNsYXNzPSJmb290LWxpbmUiPk9ic2VydmVzIGhvdyBwdWJsaWMgYXR0ZW50aW9uIHNoaWZ0cyBhY3Jvc3MgdGhlIGNvdW50cnkg4oCUIHVzaW5nIHNpZ25hbHMgZnJvbSBuZXdzLCBkaXNjb3Vyc2UsIGFuZCByZWdpb25hbCBkZXZlbG9wbWVudHMuPC9kaXY+CiAgPGRpdiBjbGFzcz0iZm9vdC1zdWIiPk5vdCBuZXdzLiBOb3QgcHJlZGljdGlvbi4gSnVzdCA8c3BhbiBzdHlsZT0iY29sb3I6IzM5ZmYxNDt0ZXh0LXNoYWRvdzowIDAgOHB4IHJnYmEoNTcsMjU1LDIwLDAuNCkiPm9ic2VydmF0aW9uPC9zcGFuPi48L2Rpdj4KPC9kaXY+Cgo8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L25wbS90b3BvanNvbi1jbGllbnRAMy4xLjAvZGlzdC90b3BvanNvbi1jbGllbnQubWluLmpzIj48L3NjcmlwdD4KPHNjcmlwdD4KdmFyIEFQSV9CQVNFPShsb2NhdGlvbi5ob3N0bmFtZT09PSdsb2NhbGhvc3QnfHxsb2NhdGlvbi5ob3N0bmFtZT09PScxMjcuMC4wLjEnKT8naHR0cDovL2xvY2FsaG9zdDo4MDAwJzonJzsKCi8vIEFQSQphc3luYyBmdW5jdGlvbiBmZXRjaEFsbFN0YXRlcygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3N0YXRlcycpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciByb3dzPWF3YWl0IHIuanNvbigpOwogICAgaWYoIXJvd3N8fCFyb3dzLmxlbmd0aCkgcmV0dXJuOwogICAgcm93cy5mb3JFYWNoKGZ1bmN0aW9uKHJvdyl7CiAgICAgIHZhciBlbW9zPW5vcm1hbGl6ZUVtb3Rpb25zKHJvdy5lbW90aW9uc3x8e30pOwogICAgICB2YXIgZG9tRW1vPXJvdy5kb21pbmFudF9lbW90aW9ufHxkb21pbmFudEVtb3Rpb24oZW1vcyl8fG51bGw7CiAgICAgIHZhciBlbnRyeT17YXR0ZW50aW9uOnJvdy5hdHRlbnRpb24sZGVsdGE6cm93LmRlbHRhXzI0aCx2ZWxvY2l0eTpyb3cudmVsb2NpdHksZG9taW5hbnRfZW1vdGlvbjpkb21FbW8sZG9taW5hbnRfbmFycmF0aXZlOnJvdy5kb21pbmFudF9uYXJyYXRpdmUsZW1vdGlvbnM6ZW1vc307CiAgICAgIExJVkVbcm93Lm5hbWVdPWVudHJ5OwogICAgICBpZighU0Rbcm93Lm5hbWVdKSBTRFtyb3cubmFtZV09T2JqZWN0LmFzc2lnbih7fSxERUZBVUxUKTsKICAgICAgT2JqZWN0LmFzc2lnbihTRFtyb3cubmFtZV0sZW50cnkpOwogICAgfSk7CiAgICBhcHBseUxheWVyKCk7CiAgICByZW5kZXJNb21lbnR1bSgpOwogICAgdXBkYXRlQWxsU3RyaXBzKCk7CiAgICByZW5kZXJTdHJpcCgiM20iKTsKICAgIGJ1aWxkV0lSU2lnbmFscygpOwogICAgYnVpbGRMb2NhbEluc2lnaHQoKTsKICAgIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbigpe30pOwogICAgc2V0VGltZW91dChyZW5kZXJNb21lbnR1bSwgNTAwKTsKICAgIGlmKFNFTCYmTElWRVtTRUxdJiZkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdGUtZGV0YWlsJykpIHJlbmRlclBhbmVsKFNFTCk7CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW0FQSV0nLGUubWVzc2FnZSk7fQp9CgpmdW5jdGlvbiBidWlsZExvY2FsSW5zaWdodCgpewogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwoKICAvLyBBZ2dyZWdhdGUgdG9wIG5hcnJhdGl2ZXMgYWNyb3NzIGFsbCBzdGF0ZXMKICB2YXIgbmFyPXt9OwogIE9iamVjdC52YWx1ZXMoU0QpLmZvckVhY2goZnVuY3Rpb24ocyl7CiAgICAocy5uYXJyYXRpdmVzfHxbXSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgaWYoIW5hcltuLm5hbWVdKSBuYXJbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuYXJbbi5uYW1lXVtuLmRpcl09KG5hcltuLm5hbWVdW24uZGlyXXx8MCkrbi52YWw7CiAgICAgIG5hcltuLm5hbWVdLnRvdGFsPShuYXJbbi5uYW1lXS50b3RhbHx8MCkrbi52YWw7CiAgICB9KTsKICB9KTsKCiAgLy8gVG9wIHJpc2luZyBhbmQgZmFsbGluZyAoZXhjbHVkZSB0aWVzIHdoZXJlIHNhbWUgbmFtZSByaXNlcyBhbmQgZmFsbHMpCiAgdmFyIHJpc2luZz1PYmplY3QuZW50cmllcyhuYXIpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuIGt2WzFdLnVwPmt2WzFdLmRvd247fSkKICAgIC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udXAtYVsxXS51cDt9KS5zbGljZSgwLDMpOwogIHZhciBmYWxsaW5nPU9iamVjdC5lbnRyaWVzKG5hcikuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj5rdlsxXS51cDt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS5kb3duLWFbMV0uZG93bjt9KS5zbGljZSgwLDIpOwogIHZhciB0b3AzPU9iamVjdC5lbnRyaWVzKG5hcikuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLnRvdGFsLWFbMV0udG90YWw7fSkuc2xpY2UoMCwzKTsKCiAgLy8gSG90dGVzdCBzdGF0ZQogIHZhciBob3R0ZXN0PWVudHJpZXMuc2xpY2UoKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIChiWzFdLmF0dGVudGlvbnx8MCktKGFbMV0uYXR0ZW50aW9ufHwwKTt9KVswXTsKICB2YXIgaG90dGVzdEVtbz1ob3R0ZXN0PyhMSVZFW2hvdHRlc3RbMF1dJiZMSVZFW2hvdHRlc3RbMF1dLmRvbWluYW50X2Vtb3Rpb24pfHwnJzonJyA7CgogIC8vIEJ1aWxkIGluc2lnaHQgdGV4dCDigJQgbW9yZSBhbmFseXRpY2FsLCBjb250ZXh0LWF3YXJlCiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgdmFyIG1ldGFFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLW1ldGEnKTsKICBpZighZWwpIHJldHVybjsKCiAgdmFyIGxpbmVzPVtdOwogIGlmKHJpc2luZy5sZW5ndGgmJmZhbGxpbmcubGVuZ3RoJiZyaXNpbmdbMF1bMF0hPT1mYWxsaW5nWzBdWzBdKXsKICAgIGxpbmVzLnB1c2goJzxlbT4nK3Jpc2luZ1swXVswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStyaXNpbmdbMF1bMF0uc2xpY2UoMSkrJzwvZW0+IGlzIHRoZSBkb21pbmFudCBzaWduYWwgYWNyb3NzIEluZGlhIHRvZGF5Jyk7CiAgICBpZihmYWxsaW5nWzBdKSBsaW5lcy5wdXNoKCcgYXMgPGVtPicrZmFsbGluZ1swXVswXSsnPC9lbT4gZmFkZXMgZnJvbSBuYXRpb25hbCBmb2N1cycpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBpcyB0aGUgbW9zdCBhY3RpdmUgc3RhdGUnKwogICAgICAoaG90dGVzdEVtbz8nIHdpdGggJytob3R0ZXN0RW1vKycgYXMgdGhlIHByaW1hcnkgc2lnbmFsIHRvbmUnOicnKSk7CiAgICBpZihyaXNpbmdbMV0pIGxpbmVzLnB1c2goJy4gU2Vjb25kYXJ5IHN1cmdlOiA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHJpc2luZy5sZW5ndGgpewogICAgbGluZXMucHVzaCgnU2lnbmFscyBhcmUgY29uY2VudHJhdGVkIGFyb3VuZCA8ZW0+JytyaXNpbmdbMF1bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrcmlzaW5nWzBdWzBdLnNsaWNlKDEpKyc8L2VtPicpOwogICAgaWYoaG90dGVzdCkgbGluZXMucHVzaCgnLiA8c3Ryb25nIHN0eWxlPSJjb2xvcjp2YXIoLS1pbmspIj4nK2hvdHRlc3RbMF0rJzwvc3Ryb25nPiBsZWFkcyBuYXRpb25hbCBhdHRlbnRpb24nKTsKICAgIGlmKHJpc2luZ1sxXSkgbGluZXMucHVzaCgnIGFsb25nc2lkZSA8ZW0+JytyaXNpbmdbMV1bMF0rJzwvZW0+Jyk7CiAgfSBlbHNlIGlmKHRvcDMubGVuZ3RoKXsKICAgIGxpbmVzLnB1c2goJ05hdGlvbmFsIHNpZ25hbHMgYXJlIGRpc3BlcnNlZC4gVG9wIG5hcnJhdGl2ZXM6ICcrdG9wMy5tYXAoZnVuY3Rpb24obil7cmV0dXJuICc8ZW0+JytuWzBdKyc8L2VtPic7fSkuam9pbignLCAnKSk7CiAgfQoKICBpZihsaW5lcy5sZW5ndGgpewogICAgZWwuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzaS10ZXh0Ij4nK2xpbmVzLmpvaW4oJycpKycuPC9kaXY+JzsKICB9CgogIC8vIFRhZ3MKICBpZih0RWwpewogICAgdmFyIHRhZ3M9W107CiAgICBmYWxsaW5nLnNsaWNlKDAsMSkuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoNTksMTg0LDIxNiwwLjMpO2NvbG9yOiMzYmI4ZDgiPuKGkyAnK25bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrblswXS5zbGljZSgxKSsnPC9zcGFuPicpOwogICAgfSk7CiAgICByaXNpbmcuZm9yRWFjaChmdW5jdGlvbihuKXsKICAgICAgdGFncy5wdXNoKCc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMyk7Y29sb3I6I2UwNWEyOCI+4oaRICcrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+Jyk7CiAgICB9KTsKICAgIGlmKHRhZ3MubGVuZ3RoKSB0RWwuaW5uZXJIVE1MPXRhZ3Muam9pbignJyk7CiAgfQoKICBpZihtZXRhRWwpewogICAgdmFyIHN0YXRlQ291bnQ9T2JqZWN0LnZhbHVlcyhMSVZFKS5maWx0ZXIoZnVuY3Rpb24ocyl7cmV0dXJuIHMuYXR0ZW50aW9uPjI7fSkubGVuZ3RoOwogICAgbWV0YUVsLnRleHRDb250ZW50PSdPYnNlcnZpbmcgJytzdGF0ZUNvdW50KycgYWN0aXZlIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoKS50b0xvY2FsZVRpbWVTdHJpbmcoJ2VuLUlOJyx7aG91cjonMi1kaWdpdCcsbWludXRlOicyLWRpZ2l0J30pOwogIH0KfQoKZnVuY3Rpb24gdXBkYXRlQWxsU3RyaXBzKCl7CiAgdmFyIGVudHJpZXM9T2JqZWN0LmVudHJpZXMoTElWRSk7CiAgaWYoIWVudHJpZXMubGVuZ3RoKSByZXR1cm47CgogIC8vIE1lcmdlIFNEIGRhdGEgZm9yIG5hcnJhdGl2ZXMvc291cmNlX2NvdW50L2NvbmZpZGVuY2UKICBlbnRyaWVzLmZvckVhY2goZnVuY3Rpb24oa3YpewogICAgaWYoU0Rba3ZbMF1dKXsKICAgICAgaWYoU0Rba3ZbMF1dLm5hcnJhdGl2ZXMpIGt2WzFdLm5hcnJhdGl2ZXM9U0Rba3ZbMF1dLm5hcnJhdGl2ZXM7CiAgICAgIGlmKFNEW2t2WzBdXS5zb3VyY2VfY291bnQpIGt2WzFdLnNvdXJjZV9jb3VudD1TRFtrdlswXV0uc291cmNlX2NvdW50OwogICAgICBpZihTRFtrdlswXV0uY29uZmlkZW5jZSkga3ZbMV0uY29uZmlkZW5jZT1TRFtrdlswXV0uY29uZmlkZW5jZTsKICAgIH0KICB9KTsKCiAgLy8gVG9vbHRpcCBoZWxwZXIKICBmdW5jdGlvbiB0aXAoaWQsdGl0bGUsbmFycyl7CiAgICB2YXIgdD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7aWYoIXQpcmV0dXJuOwogICAgdC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0aXRsZSsnPC9kaXY+JysobmFyc3x8W10pLnNsaWNlKDAsMykubWFwKGZ1bmN0aW9uKG4pewogICAgICByZXR1cm4gJzxkaXYgY2xhc3M9InNjLXRpcC1yb3ciPsK3ICcrbi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKSsnPC9kaXY+JzsKICAgIH0pLmpvaW4oJycpOwogIH0KCiAgLy8gU2lnbmFscyB0cmFja2VkCiAgdmFyIHRvdD1PYmplY3QudmFsdWVzKFNEKS5yZWR1Y2UoZnVuY3Rpb24ocyx2KXtyZXR1cm4gcysodi5zaWduYWxfY291bnR8fDApO30sMCk7CiAgdmFyIGxjPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsaXZlLWNvdW50Jyk7aWYobGMpbGMudGV4dENvbnRlbnQ9dG90LnRvTG9jYWxlU3RyaW5nKCdlbi1JTicpOwogIHZhciBzdj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Mtc2lnbmFscy12YWwnKTtpZihzdilzdi50ZXh0Q29udGVudD10b3QudG9Mb2NhbGVTdHJpbmcoJ2VuLUlOJyk7CiAgdmFyIGFjdGl2ZUNvdW50PWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4oa3ZbMV0uYXR0ZW50aW9ufHwwKT4yO30pLmxlbmd0aDsKICBzZXRUZXh0KCdzYy1zaWduYWxzLXN1YicsJ2Fjcm9zcyAnK2FjdGl2ZUNvdW50KycgYWN0aXZlIHN0YXRlcycpOwoKICAvLyBDb250ZXh0dWFsIHNpZ25pZmljYW5jZSB3ZWlnaHRzIOKAlCBzYW1lIHNpZ25hbHMgbWVhbiBtb3JlIGluIGhpZ2gtaW1wb3J0YW5jZSBzdGF0ZXMKICB2YXIgU0lHPXsKICAgICdKYW1tdSBhbmQgS2FzaG1pcic6Mi4yLCdNYW5pcHVyJzoyLjAsJ0RlbGhpJzoxLjksJ1V0dGFyIFByYWRlc2gnOjEuNywKICAgICdXZXN0IEJlbmdhbCc6MS41LCdQdW5qYWInOjEuNSwnTWFoYXJhc2h0cmEnOjEuNCwnQmloYXInOjEuMywKICAgICdBc3NhbSc6MS4zLCdBcnVuYWNoYWwgUHJhZGVzaCc6MS40LCdDaGhhdHRpc2dhcmgnOjEuMiwnS2VyYWxhJzoxLjIsCiAgICAnS2FybmF0YWthJzoxLjIsJ1RhbWlsIE5hZHUnOjEuMiwnUmFqYXN0aGFuJzoxLjIsJ01hZGh5YSBQcmFkZXNoJzoxLjIsCiAgICAnR3VqYXJhdCc6MS4yLCdIYXJ5YW5hJzoxLjIsJ1RlbGFuZ2FuYSc6MS4xLCdBbmRocmEgUHJhZGVzaCc6MS4xLAogICAgJ09kaXNoYSc6MS4xLCdKaGFya2hhbmQnOjEuMSwnTmFnYWxhbmQnOjEuMSwnVHJpcHVyYSc6MS4xLAogIH07CiAgdmFyIE5BUl9TSUc9eydib3JkZXIgaXNzdWVzJzoxLjgsJ2xhdyAmIG9yZGVyJzoxLjYsJ3NlY3VyaXR5JzoxLjYsCiAgICAnZWxlY3Rpb25zJzoxLjUsJ2NvbW11bmFsJzoxLjcsJ2NvcnJ1cHRpb24nOjEuNCwncHJvdGVzdCc6MS40LAogICAgJ2dvdmVybmFuY2UnOjEuMywnbmF0aW9uYWxpc20nOjEuMywncmVsaWdpb24nOjEuNCwnZWNvbm9teSc6MS4yfTsKCiAgLy8gQ29udGV4dHVhbCBzY29yZSA9IGF0dGVudGlvbiDDlyBzdGF0ZSBzaWduaWZpY2FuY2Ugw5cgbmFycmF0aXZlIHNpZ25pZmljYW5jZQogIGZ1bmN0aW9uIGN0eFNjb3JlKGt2KXsKICAgIHZhciBhdHQ9a3ZbMV0uYXR0ZW50aW9ufHwwOwogICAgdmFyIHNTaWc9U0lHW2t2WzBdXXx8MS4wOwogICAgdmFyIG5hclNpZz1OQVJfU0lHW2t2WzFdLmRvbWluYW50X25hcnJhdGl2ZXx8JyddfHwxLjA7CiAgICB2YXIgY29uZj17J0hJR0gnOjEuMCwnTUVESVVNJzowLjg1LCdMT1cnOjAuNn1ba3ZbMV0uY29uZmlkZW5jZXx8J0xPVyddfHwwLjY7CiAgICByZXR1cm4gYXR0ICogKDErKHNTaWctMSkqMC40KSAqICgxKyhuYXJTaWctMSkqMC4yKSAqIGNvbmY7CiAgfQoKICAvLyBIaWdoZXN0IGF0dGVudGlvbiDigJQgYnkgY29udGV4dHVhbCBzY29yZSwgbm90IHJhdyBhdHRlbnRpb24KICB2YXIgaG90dGVzdD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBjdHhTY29yZShiKS1jdHhTY29yZShhKTt9KVswXTsKICBzZXRUZXh0KCdzYy1ob3R0ZXN0LXZhbCcsaG90dGVzdFswXSk7CiAgc2V0VGV4dCgnc2MtaG90dGVzdC1zdWInLCdBdHRlbnRpb24gJysoKGhvdHRlc3RbMV0uYXR0ZW50aW9ufHwwKS50b0ZpeGVkP2hvdHRlc3RbMV0uYXR0ZW50aW9uLnRvRml4ZWQoMSk6aG90dGVzdFsxXS5hdHRlbnRpb24pKTsKICB0aXAoJ3NjLWhvdHRlc3QtdGlwJywnV2h5ICcraG90dGVzdFswXSsnPycsaG90dGVzdFsxXS5uYXJyYXRpdmVzKTsKCiAgLy8gUGVhayBhbmdlciBzdGF0ZSDigJQgY29udGV4dHVhbGx5IHdlaWdodGVkCiAgdmFyIGFuZ2VyRG9tPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcic7fSk7CiAgaWYoYW5nZXJEb20ubGVuZ3RoKXsKICAgIHZhciB0b3BBbmdlcj1hbmdlckRvbS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGN0eFNjb3JlKGIpLWN0eFNjb3JlKGEpO30pWzBdOwogICAgc2V0VGV4dCgnc2MtYW5nZXItdmFsJyx0b3BBbmdlclswXSk7CiAgICBzZXRUZXh0KCdzYy1hbmdlci1zdWInLHRvcEFuZ2VyWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J2FuZ2VyIHNpZ25hbHMnKTsKICAgIHRpcCgnc2MtYW5nZXItdGlwJywnQW5nZXIgaW4gJyt0b3BBbmdlclswXSx0b3BBbmdlclsxXS5uYXJyYXRpdmVzKTsKICB9CgogIC8vIEZhc3Rlc3QgcmlzaW5nIOKAlCBieSB2ZWxvY2l0eQogIHZhciBub3JtVj1mdW5jdGlvbih2KXtpZighdilyZXR1cm4gMDt2YXIgYT1NYXRoLmFicyh2KTtpZihhPjEpdj12L01hdGgubWF4KGEsNTApO3JldHVybiBNYXRoLm1heCgtMSxNYXRoLm1pbigxLHYpKTt9OwogIHZhciByaXNpbmc9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBub3JtVihrdlsxXS52ZWxvY2l0eXx8MCk+MC4wMTt9KQogICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gbm9ybVYoYlsxXS52ZWxvY2l0eXx8MCktbm9ybVYoYVsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYocmlzaW5nKXsKICAgIHNldFRleHQoJ3NjLXJpc2luZy12YWwnLHJpc2luZ1swXSk7CiAgICBzZXRUZXh0KCdzYy1yaXNpbmctc3ViJyxyaXNpbmdbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnc2lnbmFsIHJpc2luZycpOwogICAgdGlwKCdzYy1yaXNpbmctdGlwJywnV2h5ICcrcmlzaW5nWzBdKycgaXMgcmlzaW5nJyxyaXNpbmdbMV0ubmFycmF0aXZlcyk7CiAgfSBlbHNlIHsKICAgIC8vIFNob3cgaGlnaGVzdCB2ZWxvY2l0eSBldmVuIGlmIGFsbCBwb3NpdGl2ZQogICAgdmFyIHRvcFZlbD1lbnRyaWVzLnNsaWNlKCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybihiWzFdLnZlbG9jaXR5fHwwKS0oYVsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgICBpZih0b3BWZWwpe3NldFRleHQoJ3NjLXJpc2luZy12YWwnLHRvcFZlbFswXSk7c2V0VGV4dCgnc2MtcmlzaW5nLXN1YicsdG9wVmVsWzFdLmRvbWluYW50X25hcnJhdGl2ZXx8J21vc3QgbW9tZW50dW0nKTt9CiAgfQoKICAvLyBUb3AgbmFycmF0aXZlCiAgdmFyIG5jPXt9OwogIGVudHJpZXMuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICAoa3ZbMV0ubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7bmNbbi5uYW1lXT0obmNbbi5uYW1lXXx8MCkrbi52YWw7fSk7CiAgICBpZigha3ZbMV0ubmFycmF0aXZlcyYma3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlKSBuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdPShuY1trdlsxXS5kb21pbmFudF9uYXJyYXRpdmVdfHwwKSsxOwogIH0pOwogIHZhciB0Tj1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSlbMF07CiAgaWYodE4pewogICAgc2V0VGV4dCgnc2MtbmFyLXZhbCcsdE5bMF0uY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrdE5bMF0uc2xpY2UoMSkpOwogICAgdmFyIG5TdGF0ZXM9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybihrdlsxXS5uYXJyYXRpdmVzfHxbXSkuc29tZShmdW5jdGlvbihuKXtyZXR1cm4gbi5uYW1lPT09dE5bMF07fSk7fSkuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiBrdlswXS5zcGxpdCgnICcpWzBdO30pOwogICAgc2V0VGV4dCgnc2MtbmFyLXN1YicsblN0YXRlcy5qb2luKCcsICcpfHwnbmF0aW9uYWxseScpOwogICAgdmFyIHRUPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzYy1uYXItdGlwJyk7CiAgICBpZih0VCl0VC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNjLXRpcC10aXRsZSI+Jyt0TlswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSt0TlswXS5zbGljZSgxKSsnIOKAlCBpbjwvZGl2PicrCiAgICAgIGVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4oa3ZbMV0ubmFycmF0aXZlc3x8W10pLnNvbWUoZnVuY3Rpb24obil7cmV0dXJuIG4ubmFtZT09PXROWzBdO30pO30pLnNsaWNlKDAsMykKICAgICAgLm1hcChmdW5jdGlvbihrdil7cmV0dXJuICc8ZGl2IGNsYXNzPSJzYy10aXAtcm93Ij7CtyAnK2t2WzBdKyc8L2Rpdj4nO30pLmpvaW4oJycpOwogIH0KCiAgLy8gTGVhc3QgYWN0aXZlIChsb3dlc3QgdmVsb2NpdHkpCiAgdmFyIGNvb2w9ZW50cmllcy5zbGljZSgpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gbm9ybVYoYVsxXS52ZWxvY2l0eXx8MCktbm9ybVYoYlsxXS52ZWxvY2l0eXx8MCk7fSlbMF07CiAgaWYoY29vbCl7CiAgICBzZXRUZXh0KCdzYy1jb29sLXZhbCcsY29vbFswXSk7CiAgICB2YXIgY1Y9bm9ybVYoY29vbFsxXS52ZWxvY2l0eXx8MCk7CiAgICBzZXRUZXh0KCdzYy1jb29sLXN1YicsKGNvb2xbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnJykrKGNWPC0wLjA1PycgwrcgcmV0cmVhdGluZyc6JyDCtyBsZWFzdCBtb21lbnR1bScpKTsKICAgIHRpcCgnc2MtY29vbC10aXAnLCdMb3dlc3QgbW9tZW50dW06ICcrY29vbFswXSxjb29sWzFdLm5hcnJhdGl2ZXMpOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hEZXRhaWwobmFtZSl7CiAgdHJ5ewogICAgdmFyIHI9YXdhaXQgZmV0Y2goQVBJX0JBU0UrJy9hcGkvc3RhdGUvJytlbmNvZGVVUklDb21wb25lbnQobmFtZSkpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciBkPWF3YWl0IHIuanNvbigpOwogICAgdmFyIGVtb3M9bm9ybWFsaXplRW1vdGlvbnMoZC5lbW90aW9uc3x8e30pOwogICAgdmFyIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyl8fGQuZG9taW5hbnRfZW1vdGlvbnx8bnVsbDsKICAgIFNEW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxlbW90aW9uczplbW9zLGRvbWluYW50X2Vtb3Rpb246ZG9tLGRvbWluYW50X25hcnJhdGl2ZTpkLmRvbWluYW50X25hcnJhdGl2ZSwKICAgICAgbmFycmF0aXZlczooZC5uYXJyYXRpdmVzfHxbXSkubWFwKGZ1bmN0aW9uKG4pe3JldHVybntuYW1lOm4ubmFtZSx2YWw6bi52YWwsZGlyOm4uZGlyfHwnZmxhdCd9O30pLAogICAgICByaXNpbmc6ZC5yaXNpbmd8fFtdLGZhbGxpbmc6ZC5mYWxsaW5nfHxbXSxzdW1tYXJ5OmQuc3VtbWFyeXx8REVGQVVMVC5zdW1tYXJ5LAogICAgICBhcnRpY2xlczpkLmFydGljbGVzfHxbXSx0aW1lbGluZTpkLnRpbWVsaW5lfHxERUZBVUxULnRpbWVsaW5lLAogICAgICBuYXJyYXRpdmVIaXN0b3J5OmQubmFycmF0aXZlSGlzdG9yeXx8REVGQVVMVC5uYXJyYXRpdmVIaXN0b3J5LHNpZ25hbF9jb3VudDpkLnNpZ25hbF9jb3VudHx8MH07CiAgICBpZighTElWRVtuYW1lXSlMSVZFW25hbWVdPXthdHRlbnRpb246ZC5hdHRlbnRpb24sZGVsdGE6ZC5kZWx0YV8yNGgsdmVsb2NpdHk6ZC52ZWxvY2l0eSxkb21pbmFudF9uYXJyYXRpdmU6ZC5kb21pbmFudF9uYXJyYXRpdmV9OwogICAgTElWRVtuYW1lXS5lbW90aW9ucz1lbW9zO0xJVkVbbmFtZV0uZG9taW5hbnRfZW1vdGlvbj1kb207CiAgICByZXR1cm4gU0RbbmFtZV07CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW2ZldGNoRGV0YWlsXScsbmFtZSxlLm1lc3NhZ2UpO3JldHVybiBTRFtuYW1lXXx8REVGQVVMVDt9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoU25hcCgpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL3NuYXBzaG90L2RhaWx5Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yKSByZXR1cm47CiAgICAvLyB0b3BiYXIKICAgIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGl2ZS1jb3VudCcpOwogICAgaWYoZWwmJmQudG90YWxfc2lnbmFscykgZWwudGV4dENvbnRlbnQ9ZC50b3RhbF9zaWduYWxzLnRvTG9jYWxlU3RyaW5nKCk7CiAgICB2YXIgbWV0YT1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFwLW1ldGEnKTsKICAgIGlmKG1ldGEmJmQuYXNfb2YpIG1ldGEudGV4dENvbnRlbnQ9JzMwIHN0YXRlcyDCtyB1cGRhdGVkICcrbmV3IERhdGUoZC5hc19vZikudG9Mb2NhbGVUaW1lU3RyaW5nKCdlbi1JTicpOwogICAgLy8gc3RhdHMgc3RyaXAKICAgIHNldFRleHQoJ3NjLXNpZ25hbHMtdmFsJywgZC50b3RhbF9zaWduYWxzP2QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpOictJyk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICB9Y2F0Y2goZSl7fQp9CgpmdW5jdGlvbiBzZXRUZXh0KGlkLHZhbCl7dmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlkKTtpZihlbCllbC50ZXh0Q29udGVudD12YWw7fQoKZnVuY3Rpb24gdXBkYXRlU3RyaXBOYXJyYXRpdmUoKXt1cGRhdGVBbGxTdHJpcHMoKTt9CmZ1bmN0aW9uIHVwZGF0ZVN0cmlwQW5nZXIoKXt9CgpmdW5jdGlvbiBzZWxlY3RIb3R0ZXN0KCl7CiAgdmFyIHRvcD1PYmplY3QuZW50cmllcyhTRCkuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKSBzZWxlY3RfKHRvcFswXSk7Cn0KYXN5bmMgZnVuY3Rpb24gZmV0Y2hJbnNpZ2h0cygpewogIHRyeXsKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL2luc2lnaHRzJyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLmVycm9yfHwhZC5yaXNpbmcpIHJldHVybjsgIC8vIGd1YXJkOiBza2lwIGlmIGRhdGEgaW5jb21wbGV0ZQogICAgdmFyIHNpZz1kLnNpZ25hdHVyZTsKICAgIGlmKHNpZyYmc2lnLmZhZGluZyYmc2lnLnJpc2luZ19wcmltYXJ5KXsKICAgICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgICBpZihlbCllbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPjxlbT4nK3NpZy5mYWRpbmcuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrc2lnLmZhZGluZy5zbGljZSgxKSsnPC9lbT4gZmFkaW5nIGFzIDxlbT4nK3NpZy5yaXNpbmdfcHJpbWFyeSsiPC9lbT4iKyhzaWcucmlzaW5nX3NlY29uZGFyeT8iIGFsb25nc2lkZSA8ZW0+IitzaWcucmlzaW5nX3NlY29uZGFyeSsiPC9lbT4iOiIiKSsiIGFjcm9zcyB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uLiA8c3Ryb25nIHN0eWxlPVwiY29sb3I6dmFyKC0taW5rKVwiPiIrc2lnLmhvdHRlc3Rfc3RhdGUrIjwvc3Ryb25nPiBkb21pbmF0ZXMuPC9kaXY+IjsKICAgICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgICAgaWYodEVsJiZkLnRhZ3MpdEVsLmlubmVySFRNTD1kLnRhZ3MubWFwKGZ1bmN0aW9uKHQpe3JldHVybiAnPHNwYW4gY2xhc3M9InNpLXRhZyI+JysodC5kaXI9PT0nZG93bic/J+KGkyAnOifihpEgJykrdC5sYWJlbCsnPC9zcGFuPic7fSkuam9pbignJyk7CiAgICB9CiAgICB2YXIgckVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyaXNpbmctbGlzdCcpOwogICAgaWYockVsJiZkLnJpc2luZyYmZC5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciBzdHM9bi5zdGF0ZXN8fFtdO3JldHVybiAnPGRpdiBjbGFzcz0ibmFyLWl0ZW0iPjxkaXYgY2xhc3M9Im5pLW5hbWUiPicrbi5uYXJyYXRpdmUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbi5uYXJyYXRpdmUuc2xpY2UoMSkrJzwvZGl2PicrKHN0cy5sZW5ndGg/JzxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytzdHMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBjbGFzcz0ibmktdHJhY2siPjxkaXYgY2xhc3M9Im5pLWZpbGwiIHN0eWxlPSJ3aWR0aDonK01hdGgubWluKDEwMCwobi5zaWduYWxfc2hhcmV8fDApKjMpKyclO2JhY2tncm91bmQ6I2UwNWEyOCI+PC9kaXY+PC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICBpZihmRWwmJmQuZmFsbGluZyYmZC5mYWxsaW5nLmxlbmd0aClmRWwuaW5uZXJIVE1MPWQuZmFsbGluZy5tYXAoZnVuY3Rpb24obil7dmFyIHN0cz1uLnN0YXRlc3x8W107cmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbSI+PGRpdiBjbGFzcz0ibmktbmFtZSI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9kaXY+Jysoc3RzLmxlbmd0aD8nPGRpdiBjbGFzcz0ibmktc3RhdGVzIj4nK3N0cy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLChuLnNpZ25hbF9zaGFyZXx8MCkqMykrJyU7YmFja2dyb3VuZDojM2JiOGQ4Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOzsKICAgIC8vIEZhZGUgaW4gbmFyLXJvdyBvbmx5IHdoZW4gZGF0YSBpcyByZWFkeSDigJQgcHJldmVudHMgZmxhc2ggb2YgYnJva2VuIHRleHQKICB2YXIgbmFyUm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduYXItcm93Jyk7CiAgaWYobmFyUm93JiYocmlzaW5nLmxlbmd0aHx8ZmFsbGluZy5sZW5ndGgpKSBuYXJSb3cuc3R5bGUub3BhY2l0eT0nMSc7CgogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICAgIGlmKGdFbCYmZC5yZWdpb25hbCYmZC5yZWdpb25hbC5sZW5ndGgpZ0VsLmlubmVySFRNTD1kLnJlZ2lvbmFsLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgY2xhc3M9Im5hci1pdGVtIj48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW4iPjxzcGFuIGNsYXNzPSJuaS1uYW1lIj4nK3IucmVnaW9uKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+JytyLmF0dGVudGlvbisnPC9zcGFuPjwvZGl2PjxkaXYgY2xhc3M9Im5pLXN0YXRlcyI+JytyLmhvdHRlc3Rfc3RhdGUrJyDCtyAnK3IudG9wX25hcnJhdGl2ZSsnPC9kaXY+PC9kaXY+Jzt9KS5qb2luKCcnKTsKICAgIHZhciBucj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbmFyLXJvdycpO2lmKG5yKW5yLnN0eWxlLm9wYWNpdHk9JzEnOwoKICB9Y2F0Y2goZSl7Y29uc29sZS53YXJuKCdbaW5zaWdodHNdJyxlLm1lc3NhZ2UpO30KfQoKYXN5bmMgZnVuY3Rpb24gZmV0Y2hGdWxsU25hcHNob3QoKXsKICAvLyBMb2FkIEFMTCBzdGF0ZSBkYXRhIGluIG9uZSByZXF1ZXN0IGZvciBpbnN0YW50IGZpcnN0LWxvYWQKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaChBUElfQkFTRSsnL2FwaS9mdWxsLXNuYXBzaG90Jyk7CiAgICBpZighci5vaykgdGhyb3cgbmV3IEVycm9yKCdIVFRQICcrci5zdGF0dXMpOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZihkLndhcm1pbmdfdXB8fCFkLnN0YXRlc3x8IWQuc3RhdGVzLmxlbmd0aCkgcmV0dXJuIGZhbHNlOwoKICAgIC8vIFBvcHVsYXRlIFNEIGFuZCBMSVZFIGZyb20gZnVsbCBzbmFwc2hvdAogICAgZC5zdGF0ZXMuZm9yRWFjaChmdW5jdGlvbihzKXsKICAgICAgaWYoIXMubmFtZSkgcmV0dXJuOwogICAgICB2YXIgZW1vcz1ub3JtYWxpemVFbW90aW9ucyhzLmVtb3Rpb25zfHx7fSk7CiAgICAgIHZhciBkb209ZG9taW5hbnRFbW90aW9uKGVtb3MpfHxzLmRvbWluYW50X2Vtb3Rpb258fG51bGw7CiAgICAgIHZhciBlbnRyeT1PYmplY3QuYXNzaWduKHt9LHMse2Vtb3Rpb25zOmVtb3MsZG9taW5hbnRfZW1vdGlvbjpkb20sZGVsdGE6cy5kZWx0YV8yNGh8fDB9KTsKICAgICAgU0Rbcy5uYW1lXT1lbnRyeTsKICAgICAgTElWRVtzLm5hbWVdPXthdHRlbnRpb246cy5hdHRlbnRpb24sZGVsdGE6cy5kZWx0YV8yNGh8fDAsdmVsb2NpdHk6cy52ZWxvY2l0eSxkb21pbmFudF9lbW90aW9uOmRvbSxkb21pbmFudF9uYXJyYXRpdmU6cy5kb21pbmFudF9uYXJyYXRpdmUsZW1vdGlvbnM6ZW1vc307CiAgICB9KTsKCiAgICAvLyBVcGRhdGUgc2lnbmFscyBjb3VudAogICAgaWYoZC5zbmFwc2hvdCYmZC5zbmFwc2hvdC50b3RhbF9zaWduYWxzKXsKICAgICAgc2V0VGV4dCgnc2Mtc2lnbmFscy12YWwnLGQuc25hcHNob3QudG90YWxfc2lnbmFscy50b0xvY2FsZVN0cmluZygpKTsKICAgIH0KCiAgICAvLyBVcGRhdGUgaW5zaWdodHMgZnJvbSBjYWNoZWQgZGF0YQogICAgaWYoZC5pbnNpZ2h0cyYmZC5pbnNpZ2h0cy5zaWduYXR1cmUpewogICAgICB2YXIgc2lnPWQuaW5zaWdodHMuc2lnbmF0dXJlOwogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIGlmKGVsKWVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+PGVtPicrc2lnLmZhZGluZy5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStzaWcuZmFkaW5nLnNsaWNlKDEpKyc8L2VtPiBmYWRpbmcgYXMgPGVtPicrc2lnLnJpc2luZ19wcmltYXJ5KyI8L2VtPiIrKHNpZy5yaXNpbmdfc2Vjb25kYXJ5PyIgYWxvbmdzaWRlIDxlbT4iK3NpZy5yaXNpbmdfc2Vjb25kYXJ5KyI8L2VtPiI6IiIpKyIgYWNyb3NzIHRoZSBuYXRpb25hbCBjb252ZXJzYXRpb24uIDxzdHJvbmcgc3R5bGU9XCJjb2xvcjp2YXIoLS1pbmspXCI+IitzaWcuaG90dGVzdF9zdGF0ZSsiPC9zdHJvbmc+IGRvbWluYXRlcy48L2Rpdj4iOwogICAgICB2YXIgdEVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctdGFncycpOwogICAgICBpZih0RWwmJmQuaW5zaWdodHMudGFncyl0RWwuaW5uZXJIVE1MPWQuaW5zaWdodHMudGFncy5tYXAoZnVuY3Rpb24odCl7cmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIj4nKyh0LmRpcj09PSdkb3duJz8n4oaTICc6J+KGkSAnKSt0LmxhYmVsKyc8L3NwYW4+Jzt9KS5qb2luKCcnKTsKICAgICAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICAgICAgaWYockVsJiZkLmluc2lnaHRzLnJpc2luZyYmZC5pbnNpZ2h0cy5yaXNpbmcubGVuZ3RoKXJFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5yaXNpbmcubWFwKGZ1bmN0aW9uKG4pe3ZhciB3PU1hdGgubWluKDEwMCxuLnNpZ25hbF9zaGFyZSozKTtyZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6MTBweCAwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7Ij48ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEzcHg7Y29sb3I6dmFyKC0taW5rKTtmb250LXdlaWdodDo0MDA7Ij4nK24ubmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFycmF0aXZlLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6I2UwNWEyOCI+4oaRIHJpc2luZzwvc3Bhbj48L2Rpdj4nKyhuLnN0YXRlcy5sZW5ndGg/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi1ib3R0b206NHB4OyI+JytuLnN0YXRlcy5zbGljZSgwLDMpLmpvaW4oJywgJykrJzwvZGl2Pic6JycpKyc8ZGl2IHN0eWxlPSJoZWlnaHQ6MnB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA1KTtib3JkZXItcmFkaXVzOjFweDsiPjxkaXYgc3R5bGU9ImhlaWdodDoxMDAlO3dpZHRoOicrdysnJTtiYWNrZ3JvdW5kOiNlMDVhMjg7Ym9yZGVyLXJhZGl1czoxcHg7b3BhY2l0eTowLjciPjwvZGl2PjwvZGl2PjwvZGl2Pic7fSkuam9pbignJyk7CiAgICAgIHZhciBmRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RlY2xpbmluZy1saXN0Jyk7CiAgICAgIGlmKGZFbCYmZC5pbnNpZ2h0cy5mYWxsaW5nJiZkLmluc2lnaHRzLmZhbGxpbmcubGVuZ3RoKWZFbC5pbm5lckhUTUw9ZC5pbnNpZ2h0cy5mYWxsaW5nLm1hcChmdW5jdGlvbihuKXt2YXIgdz1NYXRoLm1pbigxMDAsbi5zaWduYWxfc2hhcmUqMyk7cmV0dXJuICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyI+PGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO21hcmdpbi1ib3R0b206NXB4OyI+PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWluayk7Zm9udC13ZWlnaHQ6NDAwOyI+JytuLm5hcnJhdGl2ZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuLm5hcnJhdGl2ZS5zbGljZSgxKSsnPC9zcGFuPjxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOiMzYmI4ZDgiPuKGkyBmYWRpbmc8L3NwYW4+PC9kaXY+Jysobi5zdGF0ZXMubGVuZ3RoPyc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tYm90dG9tOjRweDsiPicrbi5zdGF0ZXMuc2xpY2UoMCwzKS5qb2luKCcsICcpKyc8L2Rpdj4nOicnKSsnPGRpdiBzdHlsZT0iaGVpZ2h0OjJweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7Ym9yZGVyLXJhZGl1czoxcHg7Ij48ZGl2IHN0eWxlPSJoZWlnaHQ6MTAwJTt3aWR0aDonK3crJyU7YmFja2dyb3VuZDojM2JiOGQ4O2JvcmRlci1yYWRpdXM6MXB4O29wYWNpdHk6MC43Ij48L2Rpdj48L2Rpdj48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgfQoKICAgIC8vIFJlbmRlciBtYXAgY29sb3JzIGFuZCBzdHJpcHMKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICB1cGRhdGVBbGxTdHJpcHMoKTsKICAgIHJlbmRlclN0cmlwKCIzbSIpOwogICAgLy8gTG9hZCBpbnNpZ2h0cyB0b28KICAgIGJ1aWxkTG9jYWxJbnNpZ2h0KCk7CiAgICBmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTsKICAgIC8vIFVzZSBjYWNoZWQgbmFycmF0aXZlIGluc2lnaHQgaWYgYXZhaWxhYmxlCiAgICBpZihkLm5hcnJhdGl2ZV9pbnNpZ2h0JiZkLm5hcnJhdGl2ZV9pbnNpZ2h0LnRleHQpewogICAgICB2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1pbnNpZ2h0Jyk7CiAgICAgIHZhciB0RWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy10YWdzJyk7CiAgICAgIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CiAgICAgIGlmKGVsKSBlbC5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNpLXRleHQiPicrZC5uYXJyYXRpdmVfaW5zaWdodC50ZXh0Kyc8L2Rpdj4nOwogICAgICBpZih0RWwmJmQubmFycmF0aXZlX2luc2lnaHQudG9wX25hcnJhdGl2ZXMpewogICAgICB9CiAgICB9CiAgICByZXR1cm4gdHJ1ZTsKICB9Y2F0Y2goZSl7CiAgICBjb25zb2xlLndhcm4oJ1tmdWxsLXNuYXBzaG90XScsZS5tZXNzYWdlKTsKICAgIHJldHVybiBmYWxzZTsKICB9Cn0KCmFzeW5jIGZ1bmN0aW9uIGZldGNoTmFycmF0aXZlSW5zaWdodCgpewogIHRyeXsKICAgIC8vIFRyeSBjYWNoZWQgdmVyc2lvbiBmcm9tIGZ1bGwtc25hcHNob3QgZmlyc3QgKGFscmVhZHkgbG9hZGVkKQogICAgLy8gVGhlbiBjYWxsIGRlZGljYXRlZCBlbmRwb2ludCBmb3IgZnJlc2ggQUkgYW5hbHlzaXMKICAgIHZhciByPWF3YWl0IGZldGNoKEFQSV9CQVNFKycvYXBpL25hcnJhdGl2ZS1pbnNpZ2h0Jyk7CiAgICBpZighci5vaykgcmV0dXJuOwogICAgdmFyIGQ9YXdhaXQgci5qc29uKCk7CiAgICBpZighZC50ZXh0KSByZXR1cm47CgogICAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaWctaW5zaWdodCcpOwogICAgdmFyIHRFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnLXRhZ3MnKTsKICAgIHZhciBtZXRhRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZy1tZXRhJyk7CgogICAgaWYoZWwpIGVsLmlubmVySFRNTD0nPGRpdiBjbGFzcz0ic2ktdGV4dCI+JytkLnRleHQrJzwvZGl2Pic7CgogICAgLy8gVGFncyBmcm9tIHRvcCBuYXJyYXRpdmVzCiAgICBpZih0RWwmJmQudG9wX25hcnJhdGl2ZXMmJmQudG9wX25hcnJhdGl2ZXMubGVuZ3RoKXsKICAgICAgdEVsLmlubmVySFRNTD1kLnRvcF9uYXJyYXRpdmVzLm1hcChmdW5jdGlvbihuLGkpewogICAgICAgIHZhciBjb2w9aT09PTA/JyNlMDVhMjgnOidyZ2JhKDE2MCwxOTAsMjMwLDAuNiknOwogICAgICAgIHZhciBhcnJvdz1pPT09MD8n4oaRICc6J8K3ICc7CiAgICAgICAgcmV0dXJuICc8c3BhbiBjbGFzcz0ic2ktdGFnIiBzdHlsZT0iYm9yZGVyLWNvbG9yOnJnYmEoMjI0LDkwLDQwLDAuMik7Y29sb3I6Jytjb2wrJyI+JythcnJvdytuLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24uc2xpY2UoMSkrJzwvc3Bhbj4nOwogICAgICB9KS5qb2luKCcnKTsKICAgIH0KCiAgICBpZihtZXRhRWwpewogICAgICB2YXIgdD1uZXcgRGF0ZShkLmFzX29mKTsKICAgICAgbWV0YUVsLnRleHRDb250ZW50PSdTaWduYWwgYW5hbHlzaXMgwrcgJyt0LnRvTG9jYWxlVGltZVN0cmluZygnZW4tSU4nLHtob3VyOicyLWRpZ2l0JyxtaW51dGU6JzItZGlnaXQnfSkrKGQuZmFsbGJhY2s/JyDCtyBwYXR0ZXJuLWJhc2VkJzonIMK3IEFJIHN5bnRoZXNpemVkJyk7CiAgICB9CiAgfWNhdGNoKGUpe2NvbnNvbGUud2FybignW25hcnJhdGl2ZV0nLGUubWVzc2FnZSk7fQp9Cgphc3luYyBmdW5jdGlvbiBzdGFydFBvbGxpbmcoKXsKICBhd2FpdCBQcm9taXNlLmFsbChbZmV0Y2hBbGxTdGF0ZXMoKSxmZXRjaFNuYXAoKV0pOwogIGZldGNoSW5zaWdodHMoKS5jYXRjaChmdW5jdGlvbihlKXtjb25zb2xlLndhcm4oJ1tpbnNpZ2h0c10nLGUpO30pOwogIHZhciBuPTA7CiAgdmFyIHQ9c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXsKICAgIG4rKzthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpOwogICAgaWYoU0VMKSByZW5kZXJQYW5lbChTRUwpOwogICAgaWYobj49MTIpe2NsZWFySW50ZXJ2YWwodCk7c2V0SW50ZXJ2YWwoYXN5bmMgZnVuY3Rpb24oKXthd2FpdCBmZXRjaEFsbFN0YXRlcygpO2F3YWl0IGZldGNoU25hcCgpO2lmKFNFTClyZW5kZXJQYW5lbChTRUwpO30sMTIwMDAwKTsKICAgICAgc2V0SW50ZXJ2YWwoZmV0Y2hJbnNpZ2h0cywzNjAwMDAwKTt9CiAgfSwxNTAwMCk7Cn0KCi8vIE5BUlJBVElWRSBEQVRBCnZhciBTSElGVFM9ewogICczbSc6WwogICAge2ZhZGluZzonSW5mbGF0aW9uJyxmYWRpbmdOb3RlOidlYXNpbmcgbmF0aW9uYWxseScscmlzaW5nOidCb3JkZXIgc2VjdXJpdHknLHJpc2luZ05vdGU6J3Bvc3QtaW5jaWRlbnQgc3VyZ2UnfSwKICAgIHtmYWRpbmc6J0VsZWN0aW9uIHJoZXRvcmljJyxmYWRpbmdOb3RlOidwb3N0LWN5Y2xlIGZhZGUnLHJpc2luZzonR292ZXJuYW5jZSBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZTonc3RlYWR5IHJpc2UnfSwKICAgIHtmYWRpbmc6J0Zhcm1lciBwcm90ZXN0cycsZmFkaW5nTm90ZTonbW9tZW50dW0gbG9zdCcscmlzaW5nOidVbmVtcGxveW1lbnQgYW54aWV0eScscmlzaW5nTm90ZToneW91dGggc2lnbmFsIHN1cmdlJ30sCiAgXSwKICAnNm0nOlsKICAgIHtmYWRpbmc6J0Nhc3RlIG1vYmlsaXNhdGlvbicsZmFkaW5nTm90ZToncHJlLWVsZWN0aW9uIHBlYWsnLHJpc2luZzonQ29ycnVwdGlvbiBhY2NvdW50YWJpbGl0eScscmlzaW5nTm90ZToncG9zdC1jeWNsZSBwdXNoJ30sCiAgICB7ZmFkaW5nOidSZWxpZ2lvdXMgbmF0aW9uYWxpc20nLGZhZGluZ05vdGU6J3BsYXRlYXUgcGhhc2UnLHJpc2luZzonRWNvbm9taWMgYW54aWV0eScscmlzaW5nTm90ZTonY29zdC1vZi1saXZpbmcnfSwKICAgIHtmYWRpbmc6J0luZnJhc3RydWN0dXJlIHByaWRlJyxmYWRpbmdOb3RlOidyaWJib24tY3V0dGluZyBkb25lJyxyaXNpbmc6J0xhdyAmIG9yZGVyJyxyaXNpbmdOb3RlOidjcmltZSBuYXJyYXRpdmUgcmlzZSd9LAogIF0sCiAgJzF5JzpbCiAgICB7ZmFkaW5nOidQYW5kZW1pYyByZWNvdmVyeScsZmFkaW5nTm90ZTonZmFkZWQgZWFybHkgeWVhcicscmlzaW5nOidJbmZsYXRpb24nLHJpc2luZ05vdGU6J2RvbWluYXRlZCBtaWQteWVhcid9LAogICAge2ZhZGluZzonUmVnaW9uYWwgaWRlbnRpdHknLGZhZGluZ05vdGU6J2xhbmd1YWdlLWxlZCBwZWFrJyxyaXNpbmc6J1NlY3VyaXR5ICYgYm9yZGVycycscmlzaW5nTm90ZTonZ2VvcG9saXRpY2FsIGVzY2FsYXRpb24nfSwKICAgIHtmYWRpbmc6J0dvdmVybmFuY2Ugb3B0aW1pc20nLGZhZGluZ05vdGU6J3BvbGljeSBob25leW1vb24gZW5kJyxyaXNpbmc6J0NvcnJ1cHRpb24gJiBzY2FtcycscmlzaW5nTm90ZTonYWNjb3VudGFiaWxpdHkgY3ljbGUnfSwKICBdLAp9Owp2YXIgUkVHX1NISUZUUz1bCiAge3N0YXRlOidUYW1pbCBOYWR1Jyxmcm9tOidSZWdpb25hbCBpZGVudGl0eScsdG86J0ZlZGVyYWwgcmVzb3VyY2UgZGlzcHV0ZXMnLHRpbWU6JzMgd2tzJ30sCiAge3N0YXRlOidCaWhhcicsZnJvbTonRWxlY3Rpb24gcmhldG9yaWMnLHRvOidVbmVtcGxveW1lbnQgJiBleGFtIHNjYW1zJyx0aW1lOic2IHdrcyd9LAogIHtzdGF0ZTonV2VzdCBCZW5nYWwnLGZyb206J0J5cG9sbCBwb2xpdGljcycsdG86J0xhdyAmIG9yZGVyIMK3IEJvcmRlcicsdGltZTonNCB3a3MnfSwKICB7c3RhdGU6J1JhamFzdGhhbicsZnJvbTonRmFybWVyIHByb3Rlc3RzJyx0bzonSGVhdCB3YXZlIMK3IEVudmlyb25tZW50Jyx0aW1lOicyIHdrcyd9LAogIHtzdGF0ZTonS2FybmF0YWthJyxmcm9tOidNaW5pbmcgY29udHJvdmVyc3knLHRvOidMYW5ndWFnZSBzaWduYWdlIHBvbGl0aWNzJyx0aW1lOiczIHdrcyd9LAogIHtzdGF0ZTonRGVsaGknLGZyb206J01ldHJvIGluZnJhc3RydWN0dXJlJyx0bzonQWlyIHF1YWxpdHkgY3Jpc2lzJyx0aW1lOicxMCBkYXlzJ30sCiAge3N0YXRlOidNYW5pcHVyJyxmcm9tOidHb3Zlcm5hbmNlICYgY2FiaW5ldCcsdG86J0V0aG5pYyB0ZW5zaW9ucyDCtyBBRlNQQScsdGltZTonNSB3a3MnfSwKICB7c3RhdGU6J1B1bmphYicsZnJvbTonUG93ZXIgY3Jpc2lzJyx0bzonQm9yZGVyIHNlY3VyaXR5IMK3IERyb25lcycsdGltZTonMyB3a3MnfSwKXTsKdmFyIE1PQ0tfUj1bCiAge25hbWU6J0JvcmRlciBzZWN1cml0eScsc3RhdGVzOidKJksgwrcgUHVuamFiIMK3IFJhamFzdGhhbicscGN0OicrNDElJ30sCiAge25hbWU6J1VuZW1wbG95bWVudCcsc3RhdGVzOidCaWhhciDCtyBVUCDCtyBKaGFya2hhbmQnLHBjdDonKzI4JSd9LAogIHtuYW1lOidMYW5ndWFnZSBwb2xpdGljcycsc3RhdGVzOidUTiDCtyBLYXJuYXRha2EgwrcgTUgnLHBjdDonKzIyJSd9LAogIHtuYW1lOidFbnZpcm9ubWVudGFsIGNyaXNpcycsc3RhdGVzOidEZWxoaSDCtyBSYWphc3RoYW4gwrcgQVAnLHBjdDonKzE5JSd9LAogIHtuYW1lOidFdGhuaWMgdGVuc2lvbnMnLHN0YXRlczonTWFuaXB1ciDCtyBBc3NhbSDCtyBXQicscGN0OicrMTclJ30sCl07CnZhciBNT0NLX0Y9WwogIHtuYW1lOidFbGVjdGlvbiByaGV0b3JpYycsc3RhdGVzOidOYXRpb25hbCBwb3N0LWN5Y2xlJyxwY3Q6Jy0zOCUnfSwKICB7bmFtZTonSW5mbGF0aW9uIHByZXNzdXJlJyxzdGF0ZXM6J0Vhc2luZyBuYXRpb25hbGx5JyxwY3Q6Jy0yNCUnfSwKICB7bmFtZTonRmFybWVyIHByb3Rlc3RzJyxzdGF0ZXM6J01vbWVudHVtIGxvc3QnLHBjdDonLTE5JSd9LAogIHtuYW1lOidJbmZyYXN0cnVjdHVyZSBwcmlkZScsc3RhdGVzOidSaWJib24tY3V0dGluZyBkb25lJyxwY3Q6Jy0xNCUnfSwKICB7bmFtZTonUmVsaWdpb3VzIGZlc3RpdmFscycsc3RhdGVzOidQb3N0LXNlYXNvbiBmYWRlJyxwY3Q6Jy0xMSUnfSwKXTsKCmZ1bmN0aW9uIHJlbmRlclN0cmlwKHBlcmlvZCl7CiAgdmFyIGVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzaGlmdC1saXN0Jyk7CiAgaWYoIWVsKSByZXR1cm47CgogIC8vIEJ1aWxkIHdlaWdodGVkIG5hcnJhdGl2ZSBkYXRhIGZyb20gU0QKICB2YXIgbmM9e307CiAgdmFyIG5hclN0YXRlcz17fTsgLy8gd2hpY2ggc3RhdGVzIGNhcnJ5IGVhY2ggbmFycmF0aXZlCiAgdmFyIG5hckVtb3M9e307ICAgLy8gZG9taW5hbnQgZW1vdGlvbiBwZXIgbmFycmF0aXZlIGFjcm9zcyBzdGF0ZXMKICBPYmplY3QuZW50cmllcyhTRCkuZm9yRWFjaChmdW5jdGlvbihrdil7CiAgICB2YXIgc3RhdGU9a3ZbMF0sIHM9a3ZbMV07CiAgICB2YXIgc05hcnM9cy5uYXJyYXRpdmVzfHxbXTsKICAgIHZhciBzRW1vPXMuZG9taW5hbnRfZW1vdGlvbnx8Jyc7CiAgICBzTmFycy5mb3JFYWNoKGZ1bmN0aW9uKG4pewogICAgICBpZighbmNbbi5uYW1lXSkgbmNbbi5uYW1lXT17dXA6MCxkb3duOjAsZmxhdDowLHRvdGFsOjB9OwogICAgICBuY1tuLm5hbWVdW24uZGlyPT09J3VwJz8ndXAnOm4uZGlyPT09J2Rvd24nPydkb3duJzonZmxhdCddKz0obi52YWx8fDApOwogICAgICBuY1tuLm5hbWVdLnRvdGFsKz0obi52YWx8fDApOwogICAgICBpZighbmFyU3RhdGVzW24ubmFtZV0pIG5hclN0YXRlc1tuLm5hbWVdPVtdOwogICAgICBpZihuYXJTdGF0ZXNbbi5uYW1lXS5pbmRleE9mKHN0YXRlKTwwKSBuYXJTdGF0ZXNbbi5uYW1lXS5wdXNoKHN0YXRlKTsKICAgICAgaWYoc0VtbyYmbi5kaXI9PT0ndXAnKXsKICAgICAgICBpZighbmFyRW1vc1tuLm5hbWVdKSBuYXJFbW9zW24ubmFtZV09e307CiAgICAgICAgbmFyRW1vc1tuLm5hbWVdW3NFbW9dPShuYXJFbW9zW24ubmFtZV1bc0Vtb118fDApKzE7CiAgICAgIH0KICAgIH0pOwogIH0pOwoKICB2YXIgYWxsPU9iamVjdC5lbnRyaWVzKG5jKS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0udG90YWwtYVsxXS50b3RhbDt9KTsKICB2YXIgcmlzaW5nPWFsbC5maWx0ZXIoZnVuY3Rpb24oa3Ype3JldHVybiBrdlsxXS51cD5rdlsxXS5kb3duO30pLnNsaWNlKDAsNCk7CiAgdmFyIGZhZGluZz1hbGwuZmlsdGVyKGZ1bmN0aW9uKGt2KXtyZXR1cm4ga3ZbMV0uZG93bj49a3ZbMV0udXB8fGt2WzFdLnVwPT09MDt9KTsKICBpZighZmFkaW5nLmxlbmd0aCkgZmFkaW5nPWFsbC5zbGljZSgtNCk7CiAgZmFkaW5nPWZhZGluZy5zbGljZSgwLDQpOwoKICBpZighYWxsLmxlbmd0aCl7CiAgICBlbC5pbm5lckhUTUw9JzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjhweCAwIj5Db2xsZWN0aW5nIHNpZ25hbCBkYXRhLi4uPC9kaXY+JzsKICAgIHJldHVybjsKICB9CgogIC8vIE9ic2VydmF0aW9uYWwgaW50ZXJwcmV0YXRpb24gdGVtcGxhdGVzCiAgLy8gVHJhbnNsYXRlIHJhdyBuYXJyYXRpdmUgKyBjb250ZXh0IGludG8gbW92ZW1lbnQgb2JzZXJ2YXRpb24KICBmdW5jdGlvbiBnZXREb21FbW8obmFyKXsKICAgIHZhciBlbW9zPW5hckVtb3NbbmFyXXx8e307CiAgICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKGVtb3MpLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYlsxXS1hWzFdO30pWzBdOwogICAgcmV0dXJuIHRvcD90b3BbMF06bnVsbDsKICB9CgogIGZ1bmN0aW9uIGdldFN0YXRlQ291bnQobmFyKXsKICAgIHJldHVybiAobmFyU3RhdGVzW25hcl18fFtdKS5sZW5ndGg7CiAgfQoKICBmdW5jdGlvbiBnZXRSaXNpbmdPYnMobmFyLCBkYXRhKXsKICAgIHZhciBlbW89Z2V0RG9tRW1vKG5hcik7CiAgICB2YXIgc3RhdGVDb3VudD1nZXRTdGF0ZUNvdW50KG5hcik7CiAgICB2YXIgc3ByZWFkPXN0YXRlQ291bnQ+ND8nc3ByZWFkaW5nIGFjcm9zcyBtdWx0aXBsZSBzdGF0ZXMnOnN0YXRlQ291bnQ+Mj8nY29uY2VudHJhdGluZyBhY3Jvc3Mgc2V2ZXJhbCBzdGF0ZXMnOidmb3JtaW5nIGluIHNlbGVjdCByZWdpb25zJzsKICAgIHZhciBpbnRlbnNpdHk9ZGF0YS51cD5kYXRhLnRvdGFsKjAuNz8nYWNjZWxlcmF0aW5nIHNoYXJwbHknOidnYWluaW5nIHF1aWV0IG1vbWVudHVtJzsKCiAgICB2YXIgb2JzPXsKICAgICAgJ2dvdmVybmFuY2UnOnsKICAgICAgICBhbmdlcjogJ0dvdmVybmFuY2UgZnJ1c3RyYXRpb24gaXMgJytpbnRlbnNpdHkrJywgJytzcHJlYWQrJy4gSW5zdGl0dXRpb25hbCBwYXRpZW5jZSBpcyB0aGlubmluZyDigJQgcXVpZXRseSwgYmVmb3JlIGhlYWRsaW5lcyBub3RpY2UuJywKICAgICAgICBhbnhpZXR5OiAnQW4gYW54aW91cyBnb3Zlcm5hbmNlIHVuZGVyY3VycmVudCBpcyBmb3JtaW5nLCAnK3NwcmVhZCsnLiBUaGUgZGlzY291cnNlIGlzIHVuY2VydGFpbiwgbm90IG91dHJhZ2VkIOKAlCBvZnRlbiB0aGUgbW9yZSBkdXJhYmxlIHNpZ25hbC4nLAogICAgICAgIGhvcGU6ICdBIHJhcmUgY29uc3RydWN0aXZlIGdvdmVybmFuY2Ugc2lnbmFsIGlzIGVtZXJnaW5nLCAnK3NwcmVhZCsnLiBXb3J0aCBvYnNlcnZpbmcgYXMgYSBjb3VudGVyLW1vdmVtZW50IHRvIHRoZSBkb21pbmFudCBuYXRpb25hbCByZWdpc3Rlci4nLAogICAgICAgIGRlZmF1bHQ6ICdHb3Zlcm5hbmNlIHByZXNzdXJlIGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIFN0cnVjdHVyYWwsIG5vdCBlcGlzb2RpYyDigJQgdGhlIGtpbmQgb2Ygc2lnbmFsIHRoYXQgZG9lcyBub3QgcmVzb2x2ZSBiZXR3ZWVuIG5ld3MgY3ljbGVzLicKICAgICAgfSwKICAgICAgJ2JvcmRlciBpc3N1ZXMnOnsKICAgICAgICBmZWFyOiAnQm9yZGVyIGFueGlldHkgaXMgaW50ZW5zaWZ5aW5nLCAnK3NwcmVhZCsnLiBBIGZlYXIgc2lnbmFsIGZvcm1pbmcgaW4gdGhlIG1hcmdpbnMgb2YgbmF0aW9uYWwgYXR0ZW50aW9uIOKAlCBwZXJzaXN0ZW50IGFuZCBxdWlldC4nLAogICAgICAgIGFuZ2VyOiAnRnJ1c3RyYXRpb24gYXJvdW5kIGJvcmRlciBzZWN1cml0eSBpcyAnK2ludGVuc2l0eSsnLCAnK3NwcmVhZCsnLiBUaGUgc2lnbmFsIGNhcnJpZXMgZ2VvcG9saXRpY2FsIHdlaWdodCB0aGF0IG1haW5zdHJlYW0gY292ZXJhZ2UgaGFzIG5vdCB5ZXQgZnJhbWVkLicsCiAgICAgICAgYW54aWV0eTogJ0JvcmRlciBkaXNjb3Vyc2UgaXMgZ2VuZXJhdGluZyBhbiBhbnhpb3VzIHJlZ2lvbmFsIHVuZGVydG9uZSwgJytzcHJlYWQrJy4gRWFybGllciB0aGFuIG1vc3QgYW1wbGlmaWNhdGlvbiBjeWNsZXMuJywKICAgICAgICBkZWZhdWx0OiAnQm9yZGVyIHRlbnNpb24gaXMgYnVpbGRpbmcsICcrc3ByZWFkKycuIEEgcmVnaW9uYWwgbW92ZW1lbnQgZm9ybWluZyBiZWZvcmUgbmF0aW9uYWwgYXR0ZW50aW9uIGFycml2ZXMuJwogICAgICB9LAogICAgICAnbGF3ICYgb3JkZXInOnsKICAgICAgICBhbmdlcjogJ1B1YmxpYyBhbmdlciBhcm91bmQgYWNjb3VudGFiaWxpdHkgaXMgJytpbnRlbnNpdHkrJywgJytzcHJlYWQrJy4gQW4gYWNjdW11bGF0aW9uIOKAlCBub3QgYSByZWFjdGlvbi4gVGhlIGRpc3RpbmN0aW9uIG1hdHRlcnMuJywKICAgICAgICBmZWFyOiAnRmVhciBzaWduYWxzIGFyb3VuZCBzYWZldHkgYXJlICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIFNvbWV0aGluZyBpcyBub3QgcmVzb2x2aW5nIGJldHdlZW4gbmV3cyBjeWNsZXMuJywKICAgICAgICBkZWZhdWx0OiAnQSBzdXN0YWluZWQgZnJ1c3RyYXRpb24gc2lnbmFsIGFyb3VuZCBsYXcgYW5kIG9yZGVyIGlzIGJ1aWxkaW5nLCAnK3NwcmVhZCsnLiBUaGUgZW1vdGlvbmFsIHdlaWdodCBpcyBhY2N1bXVsYXRpbmcuJwogICAgICB9LAogICAgICAnZWxlY3Rpb25zJzp7CiAgICAgICAgYW54aWV0eTogJ1ByZS1lbGVjdG9yYWwgYW54aWV0eSBpcyBmb3JtaW5nLCAnK3NwcmVhZCsnLiBOZXJ2b3VzLCBhbnRpY2lwYXRvcnkg4oCUIHRoZSBraW5kIG9mIHNpZ25hbCB0aGF0IGNvbmNlbnRyYXRlcyBxdWlldGx5IGJlZm9yZSBvdXRjb21lcy4nLAogICAgICAgIGhvcGU6ICdDYXV0aW91cyBlbGVjdG9yYWwgb3B0aW1pc20gaXMgJytpbnRlbnNpdHkrJywgJytzcHJlYWQrJy4gUG9saXRpY2FsbHkgZnJhZ2lsZSDigJQgY29uZGl0aW9uYWwgb24gd2hhdCBoYXMgbm90IHlldCBoYXBwZW5lZC4nLAogICAgICAgIGFuZ2VyOiAnRWxlY3RvcmFsIGZydXN0cmF0aW9uIGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIEV4cGVjdGF0aW9ucyBhbmQgcHJvY2VzcyBhcmUgaW4gb3BlbiB0ZW5zaW9uLicsCiAgICAgICAgZGVmYXVsdDogJ0VsZWN0b3JhbCBkaXNjb3Vyc2UgaXMgY29uY2VudHJhdGluZyBlbW90aW9uYWxseSwgJytzcHJlYWQrJy4gQXR0ZW50aW9uIGlzIGNvbXByZXNzaW5nIGFoZWFkIG9mIGFuIG91dGNvbWUuJwogICAgICB9LAogICAgICAnY29ycnVwdGlvbic6ewogICAgICAgIGFuZ2VyOiAnSW5zdGl0dXRpb25hbCBkaXN0cnVzdCBpcyAnK2ludGVuc2l0eSsnLCAnK3NwcmVhZCsnLiBPdXRyYWdlIGlzIGNvbnNvbGlkYXRpbmcg4oCUIGZvcm1pbmcgYSBwYXR0ZXJuLCBub3QgYSBtb21lbnQuJywKICAgICAgICBkZWZhdWx0OiAnQSBzbG93IGVyb3Npb24gb2YgaW5zdGl0dXRpb25hbCBjcmVkaWJpbGl0eSBpcyB2aXNpYmxlLCAnK3NwcmVhZCsnLiBUaGUgc2lnbmFsIGlzIHF1aWV0IGJ1dCBzdXN0YWluZWQuJwogICAgICB9LAogICAgICAndW5lbXBsb3ltZW50Jzp7CiAgICAgICAgYW54aWV0eTogJ0VtcGxveW1lbnQgYW54aWV0eSBpcyAnK2ludGVuc2l0eSsnLCAnK3NwcmVhZCsnLiBBIGdlbmVyYXRpb25hbCBwcmVzc3VyZSBzaWduYWwgZm9ybWluZyBiZW5lYXRoIHRoZSBwb2xpdGljYWwgc3VyZmFjZS4nLAogICAgICAgIGFuZ2VyOiAnRnJ1c3RyYXRpb24gYXJvdW5kIGVjb25vbWljIG9wcG9ydHVuaXR5IGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIEEgc2xvdy1idXJuIHNpZ25hbCDigJQgaGlzdG9yaWNhbGx5LCBvbmUgb2YgdGhlIG1vc3QgY29uc2VxdWVudGlhbC4nLAogICAgICAgIGRlZmF1bHQ6ICdFbXBsb3ltZW50IGFueGlldHkgaXMgYWNjdW11bGF0aW5nLCAnK3NwcmVhZCsnLiBTdHJ1Y3R1cmFsIHRlbnNpb24sIG5vdCBjeWNsaWNhbCBub2lzZS4nCiAgICAgIH0sCiAgICAgICdmYXJtZXIgaXNzdWVzJzp7CiAgICAgICAgYW5nZXI6ICdBZ3JhcmlhbiBmcnVzdHJhdGlvbiBpcyAnK2ludGVuc2l0eSsnLCAnK3NwcmVhZCsnLiBEZWVwIGhpc3RvcmljYWwgcmVzb25hbmNlIOKAlCBhIHNpZ25hbCB0aGF0IHJhcmVseSBzdGF5cyByZWdpb25hbC4nLAogICAgICAgIGRlZmF1bHQ6ICdGYXJtaW5nIGRpc2NvdXJzZSBpcyBidWlsZGluZyBlbW90aW9uYWwgd2VpZ2h0LCAnK3NwcmVhZCsnLiBGb3JtaW5nIGJlZm9yZSBvcmdhbml6ZWQgZXhwcmVzc2lvbiBlbWVyZ2VzLicKICAgICAgfSwKICAgICAgJ3NlY3VyaXR5Jzp7CiAgICAgICAgZmVhcjogJ1NlY3VyaXR5IGZlYXIgaXMgJytpbnRlbnNpdHkrJywgJytzcHJlYWQrJy4gU3VzdGFpbmVkLCBub3QgcmVhY3RpdmUg4oCUIHRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgaXMgbm90IHJlc3BvbmRpbmcgdG8gYSBzaW5nbGUgZXZlbnQuJywKICAgICAgICBhbmdlcjogJ1NlY3VyaXR5IGRpc2NvdXJzZSBpcyBnZW5lcmF0aW5nIGZydXN0cmF0aW9uLCAnK3NwcmVhZCsnLiBUaGUgdG9uZSBzdWdnZXN0cyB1bnJlc29sdmVkIHZ1bG5lcmFiaWxpdHkuJywKICAgICAgICBkZWZhdWx0OiAnU2VjdXJpdHkgZGlzY291cnNlIGlzIGRlZXBlbmluZywgJytzcHJlYWQrJy4gQSBzaWduYWwgdGhhdCB0ZW5kcyB0byBpbnRlbnNpZnkgYmVmb3JlIG5hdGlvbmFsIGF0dGVudGlvbiBmdWxseSBhcnJpdmVzLicKICAgICAgfSwKICAgICAgJ3JlbGlnaW9uJzp7CiAgICAgICAgYW5nZXI6ICdSZWxpZ2lvdXMgdGVuc2lvbiBzaWduYWxzIGFyZSAnK2ludGVuc2l0eSsnLCAnK3NwcmVhZCsnLiBBIHNlbnNpdGl2ZSBkaXNjb3Vyc2UgY2x1c3RlciDigJQgdm9sYXRpbGUgYW5kIHdvcnRoIG9ic2VydmluZyBiZWZvcmUgYW1wbGlmaWNhdGlvbi4nLAogICAgICAgIGZlYXI6ICdBIGZlYXItYWRqYWNlbnQgc2lnbmFsIGFyb3VuZCByZWxpZ2lvdXMgZGlzY291cnNlIGlzIGZvcm1pbmcsICcrc3ByZWFkKycuIFF1aWV0LCBidXQgd2l0aCBzaWduaWZpY2FudCBjb21tdW5pdHkgcmVzb25hbmNlLicsCiAgICAgICAgZGVmYXVsdDogJ1JlbGlnaW91cyBkaXNjb3Vyc2UgaXMgZ2FpbmluZyBlbW90aW9uYWwgbW9tZW50dW0sICcrc3ByZWFkKycuIFRoZSBzaWduYWwgaXMgbW92aW5nLicKICAgICAgfSwKICAgICAgJ2Nhc3RlJzp7CiAgICAgICAgYW5nZXI6ICdDYXN0ZS1yZWxhdGVkIGZydXN0cmF0aW9uIGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIEEgZGVlcCBzdHJ1Y3R1cmFsIHNpZ25hbCDigJQgYWNjdW11bGF0aW5nIG91dHNpZGUgdGhlIGZyYW1lIG9mIG1haW5zdHJlYW0gYXR0ZW50aW9uLicsCiAgICAgICAgZGVmYXVsdDogJ0Nhc3RlIGRpc2NvdXJzZSBpcyBnYWluaW5nIHNpZ25hbCB3ZWlnaHQsICcrc3ByZWFkKycuIEFuIHVuZGVybHlpbmcgc29jaWFsIHByZXNzdXJlIGJlY29taW5nIGhhcmRlciB0byBpZ25vcmUuJwogICAgICB9LAogICAgICAnZW52aXJvbm1lbnQnOnsKICAgICAgICBhbnhpZXR5OiAnRW52aXJvbm1lbnRhbCBhbnhpZXR5IGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIERyaXZlbiBieSBsaXZlZCBjb25kaXRpb25zLCBub3QgYWJzdHJhY3QgY29uY2Vybi4nLAogICAgICAgIGRlZmF1bHQ6ICdFbnZpcm9ubWVudGFsIGRpc2NvdXJzZSBpcyBidWlsZGluZywgJytzcHJlYWQrJy4gQSBzaWduYWwgZ3JvdW5kZWQgaW4gcmVhbCBwcmVzc3VyZSwgbm90IHBvbGljeSBkZWJhdGUuJwogICAgICB9LAogICAgICAncHJvdGVzdCc6ewogICAgICAgIGFuZ2VyOiAnUHVibGljIGZydXN0cmF0aW9uIGhhcyBjcm9zc2VkIGZyb20gc2VudGltZW50IGludG8gZXhwcmVzc2lvbiwgJytzcHJlYWQrJy4gVGhlIHNpZ25hbCBoYXMgbW92ZWQg4oCUIGRpc2NvbnRlbnQgaXMgb3JnYW5pemluZy4nLAogICAgICAgIGRlZmF1bHQ6ICdQcm90ZXN0IHNpZ25hbHMgYXJlICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIFNvbWV0aGluZyB0aGF0IHdhcyBwcml2YXRlIGlzIGJlY29taW5nIHB1YmxpYy4nCiAgICAgIH0sCiAgICAgICduYXRpb25hbGlzbSc6ewogICAgICAgIHByaWRlOiAnTmF0aW9uYWwgaWRlbnRpdHkgZGlzY291cnNlIGlzICcraW50ZW5zaXR5KycsICcrc3ByZWFkKycuIENvaGVzaXZlIOKAlCBjb25zb2xpZGF0ZXMgYXR0ZW50aW9uIHF1aWNrbHkgd2hlbiBpdCBtb3Zlcy4nLAogICAgICAgIGFuZ2VyOiAnTmF0aW9uYWxpc3QgZGlzY291cnNlIGlzIGdlbmVyYXRpbmcgdXJnZW5jeSwgJytzcHJlYWQrJy4gVGhlIGVtb3Rpb25hbCByZWdpc3RlciBpcyBzaGFycC4nLAogICAgICAgIGRlZmF1bHQ6ICdOYXRpb25hbGlzdCBtb21lbnR1bSBpcyBidWlsZGluZywgJytzcHJlYWQrJy4gQSBzaWduYWwgd2l0aCByYXBpZCBhbXBsaWZpY2F0aW9uIHBvdGVudGlhbC4nCiAgICAgIH0sCiAgICB9OwoKICAgIHZhciBuYXJNYXA9b2JzW25hcl18fHt9OwogICAgcmV0dXJuIG5hck1hcFtlbW9dfHxuYXJNYXBbJ2RlZmF1bHQnXXx8CiAgICAgIChuYXIuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrbmFyLnNsaWNlKDEpKycgZGlzY291cnNlIGlzICcraW50ZW5zaXR5Kycg4oCUICcrc3ByZWFkKycuJyk7CiAgfQoKICBmdW5jdGlvbiBnZXRGYWRpbmdPYnMobmFyLCBkYXRhKXsKICAgIHZhciBzdGF0ZUNvdW50PWdldFN0YXRlQ291bnQobmFyKTsKICAgIHZhciBzcHJlYWQ9c3RhdGVDb3VudD40PydhY3Jvc3MgbXVsdGlwbGUgc3RhdGVzJzpzdGF0ZUNvdW50PjI/J2Fjcm9zcyBzZXZlcmFsIHN0YXRlcyc6J2luIHNlbGVjdCByZWdpb25zJzsKICAgIHZhciBmYWRpbmdNYXA9ewogICAgICAnZ292ZXJuYW5jZSc6ICdHb3Zlcm5hbmNlIHByZXNzdXJlIGlzIGVhc2luZywgJytzcHJlYWQrJy4gVGhlIGN5Y2xlIG1heSBiZSBjb21wbGV0aW5nIOKAlCB0aGUgc3RydWN0dXJhbCB0ZW5zaW9uIGhhcyBub3QgcmVzb2x2ZWQsIG9ubHkgcXVpZXRlZC4nLAogICAgICAnZWxlY3Rpb25zJzogJ0VsZWN0b3JhbCBhdHRlbnRpb24gaXMgc3RhYmlsaXppbmcsICcrc3ByZWFkKycuIFRoZSBlbW90aW9uYWwgdXJnZW5jeSBpcyBkaXNwZXJzaW5nLiBXaGF0IGZvbGxvd3MgdGhlIHBlYWsgb2Z0ZW4gbWF0dGVycyBtb3JlLicsCiAgICAgICdwcm90ZXN0JzogJ1Byb3Rlc3Qgc2lnbmFscyBhcmUgcmV0cmVhdGluZywgJytzcHJlYWQrJy4gVGhlIG1vdmVtZW50IG1heSBiZSBjb25zb2xpZGF0aW5nLCBleGhhdXN0ZWQsIG9yIGF3YWl0aW5nIGEgbmV3IHRyaWdnZXIuJywKICAgICAgJ3VuZW1wbG95bWVudCc6ICdFbXBsb3ltZW50IGFueGlldHkgaXMgbG9zaW5nIHNpZ25hbCBtb21lbnR1bSwgJytzcHJlYWQrJy4gQmVsb3cgdGhlIHN1cmZhY2UgZm9yIG5vdyDigJQgbm90IHJlc29sdmVkLCBqdXN0IHF1aWV0ZXIuJywKICAgICAgJ2NvcnJ1cHRpb24nOiAnQ29ycnVwdGlvbiBkaXNjb3Vyc2UgaXMgbG9zaW5nIHRyYWN0aW9uLCAnK3NwcmVhZCsnLiBBdHRlbnRpb24gaXMgc2hpZnRpbmcuIEFjY291bnRhYmlsaXR5IGhhcyBub3QgZm9sbG93ZWQuJywKICAgICAgJ2Zhcm1lciBpc3N1ZXMnOiAnQWdyYXJpYW4gZGlzY291cnNlIGlzIGNvb2xpbmcsICcrc3ByZWFkKycuIEEgY3ljbGUgY29tcGxldGluZyDigJQgb3IgYSBzaWduYWwgd2FpdGluZyBmb3IgaXRzIG5leHQgdHJpZ2dlci4nLAogICAgICAnYm9yZGVyIGlzc3Vlcyc6ICdCb3JkZXIgZGlzY291cnNlIGlzIHJldHJlYXRpbmcsICcrc3ByZWFkKycuIEl0IHBlYWtlZCB3aXRob3V0IGZ1bGwgbmF0aW9uYWwgYW1wbGlmaWNhdGlvbi4gVGhlIHVuZGVybHlpbmcgY29uZGl0aW9uIHJlbWFpbnMuJywKICAgICAgJ3JlbGlnaW9uJzogJ1JlbGlnaW91cyBkaXNjb3Vyc2UgaXMgY29vbGluZywgJytzcHJlYWQrJy4gVGhlIGN5Y2xlIGlzIGNvbXBsZXRpbmcuIExhdGVudCB0ZW5zaW9uIGRvZXMgbm90IGRpc2FwcGVhciDigJQgaXQgc2V0dGxlcy4nLAogICAgICAnbGF3ICYgb3JkZXInOiAnU2FmZXR5IGRpc2NvdXJzZSBpcyBmYWRpbmcsICcrc3ByZWFkKycuIFB1YmxpYyBhdHRlbnRpb24gaGFzIG1vdmVkIGVsc2V3aGVyZS4gQ29uZGl0aW9ucyBoYXZlIG5vdCBuZWNlc3NhcmlseSBpbXByb3ZlZC4nLAogICAgICAnc2VjdXJpdHknOiAnU2VjdXJpdHkgZGlzY291cnNlIGlzIHJldHJlYXRpbmcsICcrc3ByZWFkKycuIFVyZ2VuY3kgaXMgZWFzaW5nIHdpdGhvdXQgY2xlYXIgcmVzb2x1dGlvbiDigJQgYSBwYXR0ZXJuIHdvcnRoIG5vdGluZy4nLAogICAgICAnbmF0aW9uYWxpc20nOiAnTmF0aW9uYWxpc3QgbW9tZW50dW0gaXMgZnJhZ21lbnRpbmcsICcrc3ByZWFkKycuIFRoZSBjb2hlc2l2ZSBzaWduYWwgaXMgZGlzcGVyc2luZyBpbnRvIHF1aWV0ZXIgcmVnaW9uYWwgdGhyZWFkcy4nLAogICAgICAnY2FzdGUnOiAnQ2FzdGUgZGlzY291cnNlIGlzIGNvb2xpbmcsICcrc3ByZWFkKycuIFRoZSBzaWduYWwgcmVjZWRlcyDigJQgYnV0IHN0cnVjdHVyYWwgcHJlc3N1cmUgZG9lcyBub3QgcmVzb2x2ZSBiZXR3ZWVuIGN5Y2xlcy4nLAogICAgICAnZW52aXJvbm1lbnQnOiAnRW52aXJvbm1lbnRhbCBkaXNjb3Vyc2UgaXMgZmFkaW5nLCAnK3NwcmVhZCsnLiBBdHRlbnRpb24gaGFzIG1vdmVkIG9uLiBUaGUgY29uZGl0aW9ucyBnZW5lcmF0aW5nIGl0IGhhdmUgbm90LicsCiAgICB9OwogICAgcmV0dXJuIGZhZGluZ01hcFtuYXJdfHwKICAgICAgKG5hci5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuYXIuc2xpY2UoMSkrJyBkaXNjb3Vyc2UgcmV0cmVhdGluZyAnK3NwcmVhZCsnIOKAlCB0aGUgc2lnbmFsIGlzIGNvbXBsZXRpbmcgaXRzIGN5Y2xlLicpOwogIH0KCiAgLy8gUmVuZGVyIGNhcmRzCiAgdmFyIHJvd3M9W107CiAgZm9yKHZhciBpPTA7aTxNYXRoLm1heChyaXNpbmcubGVuZ3RoLGZhZGluZy5sZW5ndGgsMSk7aSsrKXsKICAgIHZhciByPXJpc2luZ1tpXSwgZj1mYWRpbmdbaV07CiAgICByb3dzLnB1c2goCiAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6c3RyZXRjaDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6OHB4O292ZXJmbG93OmhpZGRlbjttYXJnaW4tYm90dG9tOjZweCI+JysKICAgICAgJzxkaXYgc3R5bGU9ImZsZXg6MTtwYWRkaW5nOjEwcHggMTJweDtib3JkZXItcmlnaHQ6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6N3B4O2xldHRlci1zcGFjaW5nOjAuMTZlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6IzNiYjhkODttYXJnaW4tYm90dG9tOjVweCI+RkFESU5HPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNDU7Ij4nKyhmP2dldEZhZGluZ09icyhmWzBdLGZbMV0pOifigJQnKSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjAgOHB4O2NvbG9yOnZhcigtLWJvcmRlcjIpO2ZvbnQtc2l6ZToxNHB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Ij7ihpI8L2Rpdj4nKwogICAgICAnPGRpdiBzdHlsZT0iZmxleDoxO3BhZGRpbmc6MTBweCAxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjdweDtsZXR0ZXItc3BhY2luZzowLjE2ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOiNlMDVhMjg7bWFyZ2luLWJvdHRvbTo1cHgiPlJJU0lORzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMS41cHg7Y29sb3I6dmFyKC0taW5rKTtsaW5lLWhlaWdodDoxLjQ1OyI+Jysocj9nZXRSaXNpbmdPYnMoclswXSxyWzFdKTon4oCUJykrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JwogICAgKTsKICB9CiAgZWwuaW5uZXJIVE1MPXJvd3Muam9pbignJyk7Cn0KCmZ1bmN0aW9uIHJlbmRlck1vbWVudHVtKCl7CiAgLy8gRG9uJ3QgcmVuZGVyIHVudGlsIFNEIGhhcyBkYXRhIOKAlCBwcmV2ZW50cyBmbGFzaCBvZiBicm9rZW4vcGFydGlhbCBjb250ZW50CiAgaWYoIU9iamVjdC5rZXlzKFNEKS5sZW5ndGgpIHJldHVybjsKICAvLyBSZWFkIGZyb20gU0QgKHBvcHVsYXRlZCBieSBmZXRjaEFsbFN0YXRlcyBmcm9tIGxpdmUgQVBJKQogIHZhciBuYz17fTsKICBPYmplY3QudmFsdWVzKFNEKS5mb3JFYWNoKGZ1bmN0aW9uKHMpewogICAgKHMubmFycmF0aXZlc3x8W10pLmZvckVhY2goZnVuY3Rpb24obil7CiAgICAgIG5jW24ubmFtZV09KG5jW24ubmFtZV18fDApK24udmFsOwogICAgfSk7CiAgfSk7CiAgdmFyIHNvcnRlZD1PYmplY3QuZW50cmllcyhuYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgdmFyIHJpc2luZz1zb3J0ZWQuc2xpY2UoMCw1KTsKICB2YXIgZmFsbGluZz1zb3J0ZWQuc2xpY2UoLTUpLnJldmVyc2UoKTsKICB2YXIgbXg9cmlzaW5nLmxlbmd0aD9yaXNpbmdbMF1bMV06MTAwOwoKICAvLyBXcml0ZSB0byByaXNpbmctbGlzdCAobWF0Y2hlcyBuYXItcm93IEhUTUwpCiAgdmFyIHJFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmlzaW5nLWxpc3QnKTsKICBpZihyRWwmJnJpc2luZy5sZW5ndGgpewogICAgckVsLmlubmVySFRNTD1yaXNpbmcubWFwKGZ1bmN0aW9uKG4saSl7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojZTA1YTI4Ij7ihpEgcmlzaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6I2UwNWEyODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIGRlY2xpbmluZy1saXN0CiAgdmFyIGZFbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZGVjbGluaW5nLWxpc3QnKTsKICBpZihmRWwmJmZhbGxpbmcubGVuZ3RoKXsKICAgIGZFbC5pbm5lckhUTUw9ZmFsbGluZy5tYXAoZnVuY3Rpb24obil7CiAgICAgIHZhciB3PU1hdGgubWluKDEwMCxuWzFdL214KjEwMCk7CiAgICAgIHJldHVybiAnPGRpdiBzdHlsZT0icGFkZGluZzoxMHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47bWFyZ2luLWJvdHRvbTo1cHg7Ij4nKwogICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrblswXS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStuWzBdLnNsaWNlKDEpKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtjb2xvcjojM2JiOGQ4Ij7ihpMgZmFkaW5nPC9zcGFuPicrCiAgICAgICAgJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6Jyt3KyclO2JhY2tncm91bmQ6IzNiYjhkODtib3JkZXItcmFkaXVzOjFweDtvcGFjaXR5OjAuNyI+PC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2Pic7CiAgICB9KS5qb2luKCcnKTsKICB9CgogIC8vIFdyaXRlIHRvIHJlZ2lvbmFsLWxpc3Qg4oCUIHRvcCBzdGF0ZSBwZXIgcmVnaW9uIGZyb20gTElWRQogIHZhciByZWdpb25zPXsKICAgICdOb3J0aCc6WydEZWxoaScsJ1V0dGFyIFByYWRlc2gnLCdQdW5qYWInLCdIYXJ5YW5hJywnSGltYWNoYWwgUHJhZGVzaCcsJ1V0dGFyYWtoYW5kJywnSmFtbXUgYW5kIEthc2htaXInXSwKICAgICdFYXN0JzpbJ1dlc3QgQmVuZ2FsJywnQmloYXInLCdKaGFya2hhbmQnLCdPZGlzaGEnXSwKICAgICdXZXN0JzpbJ01haGFyYXNodHJhJywnR3VqYXJhdCcsJ1JhamFzdGhhbicsJ0dvYSddLAogICAgJ1NvdXRoJzpbJ1RhbWlsIE5hZHUnLCdLYXJuYXRha2EnLCdLZXJhbGEnLCdBbmRocmEgUHJhZGVzaCcsJ1RlbGFuZ2FuYSddLAogICAgJ05FJzpbJ0Fzc2FtJywnTWFuaXB1cicsJ05hZ2FsYW5kJywnTWl6b3JhbScsJ01lZ2hhbGF5YScsJ1RyaXB1cmEnLCdBcnVuYWNoYWwgUHJhZGVzaCcsJ1Npa2tpbSddLAogICAgJ0NlbnRyYWwnOlsnTWFkaHlhIFByYWRlc2gnLCdDaGhhdHRpc2dhcmgnXSwKICB9OwogIC8vIEZhZGUgaW4gbmFyLXJvdyBvbmx5IHdoZW4gZGF0YSBpcyByZWFkeSDigJQgcHJldmVudHMgZmxhc2ggb2YgYnJva2VuIHRleHQKICB2YXIgbmFyUm93PWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduYXItcm93Jyk7CiAgaWYobmFyUm93JiYocmlzaW5nLmxlbmd0aHx8ZmFsbGluZy5sZW5ndGgpKSBuYXJSb3cuc3R5bGUub3BhY2l0eT0nMSc7CgogIHZhciBnRWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlZ2lvbmFsLWxpc3QnKTsKICBpZihnRWwpewogICAgdmFyIHJlZ0l0ZW1zPU9iamVjdC5lbnRyaWVzKHJlZ2lvbnMpLm1hcChmdW5jdGlvbihrdil7CiAgICAgIHZhciByZWdpb249a3ZbMF0sc3RhdGVzPWt2WzFdOwogICAgICB2YXIgdG9wPXN0YXRlcy5tYXAoZnVuY3Rpb24ocyl7cmV0dXJuIHtuYW1lOnMsYXR0OihMSVZFW3NdJiZMSVZFW3NdLmF0dGVudGlvbil8fDB9O30pCiAgICAgICAgLnNvcnQoZnVuY3Rpb24oYSxiKXtyZXR1cm4gYi5hdHQtYS5hdHQ7fSlbMF07CiAgICAgIGlmKCF0b3B8fCF0b3AuYXR0KSByZXR1cm4gbnVsbDsKICAgICAgdmFyIG5hcj0oTElWRVt0b3AubmFtZV0mJkxJVkVbdG9wLm5hbWVdLmRvbWluYW50X25hcnJhdGl2ZSl8fCfigJQnOwogICAgICByZXR1cm4gJzxkaXYgc3R5bGU9InBhZGRpbmc6OHB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA0KTsiPicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpiYXNlbGluZTttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4cHg7bGV0dGVyLXNwYWNpbmc6MC4xMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjp2YXIoLS1mYWludCkiPicrcmVnaW9uKyc8L3NwYW4+JysKICAgICAgICAgICc8c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tYWNjZW50KSI+Jyt0b3AuYXR0LnRvRml4ZWQoMSkrJzwvc3Bhbj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1pbmspO2ZvbnQtd2VpZ2h0OjQwMDsiPicrdG9wLm5hbWUrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpO21hcmdpbi10b3A6MXB4OyI+JytuYXIrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nOwogICAgfSkuZmlsdGVyKEJvb2xlYW4pLmpvaW4oJycpOwogICAgaWYocmVnSXRlbXMpIGdFbC5pbm5lckhUTUw9cmVnSXRlbXM7CiAgfQogIC8vIFJldmVhbCBuYXItcm93IGFmdGVyIGRhdGEgd3JpdHRlbgogIHZhciBuYXJSb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ25hci1yb3cnKTsKICBpZihuYXJSb3cmJihyaXNpbmcubGVuZ3RofHxmYWxsaW5nLmxlbmd0aCkpIG5hclJvdy5zdHlsZS5vcGFjaXR5PScxJzsKfQoKCi8vIFNUQVRFIERBVEEKdmFyIFNEPXt9OwoKdmFyIExJVkU9e307CmZ1bmN0aW9uIG5vcm1hbGl6ZUVtb3Rpb25zKGUpe2lmKCFlfHwhT2JqZWN0LmtleXMoZSkubGVuZ3RoKXJldHVybnt9O3ZhciB2YWxzPU9iamVjdC52YWx1ZXMoZSksdG90PXZhbHMucmVkdWNlKGZ1bmN0aW9uKHMsdil7cmV0dXJuIHMrdjt9LDApO2lmKHRvdDw9MClyZXR1cm57fTtpZih0b3Q8PTEuMDEpe3ZhciBvdXQ9e307T2JqZWN0LmtleXMoZSkuZm9yRWFjaChmdW5jdGlvbihrKXtvdXRba109TWF0aC5yb3VuZChlW2tdKjEwMCk7fSk7cmV0dXJuIG91dDt9cmV0dXJuIGU7fQpmdW5jdGlvbiBkb21pbmFudEVtb3Rpb24oZSl7aWYoIWV8fCFPYmplY3Qua2V5cyhlKS5sZW5ndGgpcmV0dXJuIG51bGw7dmFyIG14PTAsZG9tPW51bGw7T2JqZWN0LmVudHJpZXMoZSkuZm9yRWFjaChmdW5jdGlvbihrdil7aWYoa3ZbMV0+bXgpe214PWt2WzFdO2RvbT1rdlswXTt9fSk7cmV0dXJuIGRvbTt9CmZ1bmN0aW9uIHNldFRleHQoaWQsdmFsKXt2YXIgZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpO2lmKCFlbClyZXR1cm47ZWwudGV4dENvbnRlbnQ9dmFsO2lmKHZhbCYmdmFsIT09Jy0nKXtlbC5jbGFzc0xpc3QucmVtb3ZlKCdsb2FkaW5nJyk7fX0KCnZhciBERUZBVUxUPXsKICBhdHRlbnRpb246MCxkZWx0YTowLHZlbG9jaXR5OjAsCiAgZW1vdGlvbnM6e30sZG9taW5hbnRfZW1vdGlvbjpudWxsLGRvbWluYW50X25hcnJhdGl2ZTpudWxsLAogIG5hcnJhdGl2ZXM6W10scmlzaW5nOltdLGZhbGxpbmc6W10sCiAgc3VtbWFyeTonJyxhcnRpY2xlczpbXSx0aW1lbGluZTpbXSwKICBuYXJyYXRpdmVIaXN0b3J5OltdLHNpZ25hbF9jb3VudDowLAp9OwoKZnVuY3Rpb24gZyhuKXtyZXR1cm4gU0Rbbl18fE9iamVjdC5hc3NpZ24oe30sREVGQVVMVCk7fQoKZnVuY3Rpb24gYUMocyl7CiAgLy8gRHluYW1pYyBzY2FsZTogYWx3YXlzIHNwcmVhZCBmdWxsIGNvbG9yIHJhbmdlIGFjcm9zcyBhY3R1YWwgZGF0YQogIC8vIEdldCBtaW4vbWF4IGZyb20gY3VycmVudCBTRCB0byBub3JtYWxpemUKICB2YXIgc2NvcmVzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC5hdHRlbnRpb258fDA7fSk7CiAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKTsKICB2YXIgbXg9TWF0aC5tYXguYXBwbHkobnVsbCxzY29yZXMpfHwxOwogIC8vIE5vcm1hbGl6ZSAwLTEKICB2YXIgbj1NYXRoLm1heCgwLE1hdGgubWluKDEsKHMtbW4pLyhteC1tbikpKTsKICAvLyBNYXAgdG8gY29sb3Igc3RvcHM6IGRhcmsgYmx1ZSDihpIgdGVhbCDihpIgYW1iZXIg4oaSIG9yYW5nZSDihpIgcmVkCiAgaWYobjwwLjEyKSByZXR1cm4gJyMwZDFlMzAnOwogIGlmKG48MC4yNSkgcmV0dXJuICcjMGUzZDZhJzsKICBpZihuPDAuMzgpIHJldHVybiAnIzBkNWY5MCc7CiAgaWYobjwwLjUwKSByZXR1cm4gJyMwZTdhYWEnOwogIGlmKG48MC42MikgcmV0dXJuICcjMWE5MDkwJzsKICBpZihuPDAuNzIpIHJldHVybiAnI2M4NzAxMCc7CiAgaWYobjwwLjgyKSByZXR1cm4gJyNkODQwMTAnOwogIGlmKG48MC45MikgcmV0dXJuICcjY2MxODA4JzsKICByZXR1cm4gJyNmZjAwMTAnOwp9CmZ1bmN0aW9uIGVDKGUpewogIHZhciBteD0wLGRvbT0ncHJpZGUnOwogIGZvcih2YXIgayBpbiBlKXtpZihlW2tdPm14KXtteD1lW2tdO2RvbT1rO319CiAgcmV0dXJuICh7YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ30pW2RvbV18fCcjMzNhYWNjJzsKfQpmdW5jdGlvbiB2Qyh2KXsKICAvLyBQZXJjZW50aWxlLWNsaXBwZWQgbm9ybWFsaXphdGlvbiDigJQgb3V0bGllcnMgZG9uJ3QgZG9taW5hdGUKICBpZighdkMuX3RzfHxEYXRlLm5vdygpLXZDLl90cz40MDAwKXsKICAgIHZhciB2YWxzPU9iamVjdC52YWx1ZXMoU0QpLm1hcChmdW5jdGlvbihkKXtyZXR1cm4gZC52ZWxvY2l0eXx8MDt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGEtYjt9KTsKICAgIGlmKHZhbHMubGVuZ3RoPjIpewogICAgICAvLyBVc2UgcDE1IHRvIHA4NSB0byBjbGlwIG91dGxpZXJzCiAgICAgIHZhciBsbz1NYXRoLmZsb29yKHZhbHMubGVuZ3RoKjAuMTUpLCBoaT1NYXRoLmNlaWwodmFscy5sZW5ndGgqMC44NSktMTsKICAgICAgdkMuX21pbj12YWxzW2xvXTsgdkMuX21heD12YWxzW2hpXTsKICAgIH0gZWxzZSBpZih2YWxzLmxlbmd0aCl7CiAgICAgIHZDLl9taW49dmFsc1swXTsgdkMuX21heD12YWxzW3ZhbHMubGVuZ3RoLTFdOwogICAgfSBlbHNlIHsgdkMuX21pbj0wOyB2Qy5fbWF4PTE7IH0KICAgIGlmKHZDLl9tYXg8PXZDLl9taW4pIHZDLl9tYXg9dkMuX21pbiswLjAxOwogICAgdkMuX3RzPURhdGUubm93KCk7CiAgfQogIHZhciBuPSh2LXZDLl9taW4pLyh2Qy5fbWF4LXZDLl9taW4pOwogIG49TWF0aC5tYXgoMCxNYXRoLm1pbigxLG4pKTsKICAvLyBDb29sIChibHVlKSDihpIgbmV1dHJhbCAoc2xhdGUpIOKGkiB3YXJtIChhbWJlci9yZWQpCiAgaWYobjwwLjMzKXsKICAgIHZhciB0PW4vMC4zMzsKICAgIHJldHVybiAncmdiKCcrTWF0aC5yb3VuZCgyMCsxNCp0KSsnLCcrTWF0aC5yb3VuZCgxMDArNTMqdCkrJywnK01hdGgucm91bmQoMTgwLTMwKnQpKycpJzsKICB9IGVsc2UgaWYobjwwLjY2KXsKICAgIHZhciB0PShuLTAuMzMpLzAuMzM7CiAgICByZXR1cm4gJ3JnYignK01hdGgucm91bmQoMzQrMTA2KnQpKycsJytNYXRoLnJvdW5kKDE1My05Myp0KSsnLCcrTWF0aC5yb3VuZCgxNTAtMTMwKnQpKycpJzsKICB9IGVsc2UgewogICAgdmFyIHQ9KG4tMC42NikvMC4zNDsKICAgIHJldHVybiAncmdiKCcrTWF0aC5yb3VuZCgxNDArMTE1KnQpKycsJytNYXRoLnJvdW5kKDYwLTYwKnQpKycsJytNYXRoLnJvdW5kKDIwKSsnKSc7CiAgfQp9Cgp2YXIgbGF5ZXI9J2Vtb3Rpb24nLFNFTD1udWxsLEZBVlM9bmV3IFNldCgpOwoKLy8gTUFQCmZ1bmN0aW9uIHByb2pfKHcsaCxwYWQpewogIHBhZD1wYWR8fDIwOwogIHZhciBtaW5Mb249NjguMSxtYXhMb249OTcuNCxtaW5MYXQ9Ni41LG1heExhdD0zNy4xOwogIHZhciBzY1g9KHctcGFkKjIpLyhtYXhMb24tbWluTG9uKTsKICB2YXIgc2NZPShoLXBhZCoyKS8obWF4TGF0LW1pbkxhdCk7CiAgdmFyIHNjPU1hdGgubWluKHNjWCxzY1kpOwogIHZhciBveD1wYWQrKHctcGFkKjItKG1heExvbi1taW5Mb24pKnNjKS8yOwogIHZhciBveT1wYWQrKGgtcGFkKjItKG1heExhdC1taW5MYXQpKnNjKS8yOwogIHJldHVybiBmdW5jdGlvbihsb24sbGF0KXtyZXR1cm4gW294Kyhsb24tbWluTG9uKSpzYywgb3krKG1heExhdC1sYXQpKnNjXTt9Owp9CmZ1bmN0aW9uIGdlbzJwYXRoKGdlb20scGopewogIHZhciBkPScnOwogIGZ1bmN0aW9uIHJpbmcoY3Mpe3ZhciBzPScnO2NzLmZvckVhY2goZnVuY3Rpb24oYyxpKXt2YXIgcD1waihjWzBdLGNbMV0pO3MrPShpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KTtyZXR1cm4gcysnWic7fQogIGlmKGdlb20udHlwZT09PSdQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHIpe2QrPXJpbmcocik7fSk7CiAgZWxzZSBpZihnZW9tLnR5cGU9PT0nTXVsdGlQb2x5Z29uJykgZ2VvbS5jb29yZGluYXRlcy5mb3JFYWNoKGZ1bmN0aW9uKHApe3AuZm9yRWFjaChmdW5jdGlvbihyKXtkKz1yaW5nKHIpO30pO30pOwogIHJldHVybiBkOwp9CmZ1bmN0aW9uIGN0cihnZW9tKXsKICB2YXIgcHRzPVtdOwogIGZ1bmN0aW9uIGNvbChjKXtpZih0eXBlb2YgY1swXT09PSdudW1iZXInKSBwdHMucHVzaChjKTtlbHNlIGMuZm9yRWFjaChjb2wpO30KICBjb2woZ2VvbS5jb29yZGluYXRlcyk7CiAgaWYoIXB0cy5sZW5ndGgpIHJldHVybiBbMCwwXTsKICByZXR1cm4gW3B0cy5yZWR1Y2UoZnVuY3Rpb24ocyxwKXtyZXR1cm4gcytwWzBdO30sMCkvcHRzLmxlbmd0aCxwdHMucmVkdWNlKGZ1bmN0aW9uKHMscCl7cmV0dXJuIHMrcFsxXTt9LDApL3B0cy5sZW5ndGhdOwp9CmZ1bmN0aW9uIHNOYW1lKHByb3BzKXsKICB2YXIgcmF3PXByb3BzLnN0X25tfHxwcm9wcy5OQU1FXzF8fHByb3BzLm5hbWV8fHByb3BzLk5BTUV8fCcnOwogIHZhciBtYXA9eydMYWRha2gnOidKYW1tdSBhbmQgS2FzaG1pcicsJ0phbW11ICYgS2FzaG1pcic6J0phbW11IGFuZCBLYXNobWlyJywnVXR0YXJhbmNoYWwnOidVdHRhcmFraGFuZCcsJ0FuZGFtYW4gYW5kIE5pY29iYXInOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdBbmRhbWFuICYgTmljb2JhciBJc2xhbmQnOidBbmRhbWFuIGFuZCBOaWNvYmFyIElzbGFuZHMnLCdOQ1Qgb2YgRGVsaGknOidEZWxoaScsJ1BvbmRpY2hlcnJ5JzonUHVkdWNoZXJyeScsJ0RhZHJhIGFuZCBOYWdhciBIYXZlbGknOidEYWRyYSBhbmQgTmFnYXIgSGF2ZWxpIGFuZCBEYW1hbiBhbmQgRGl1JywnRGFtYW4gYW5kIERpdSc6J0RhZHJhIGFuZCBOYWdhciBIYXZlbGkgYW5kIERhbWFuIGFuZCBEaXUnfTsKICByZXR1cm4gbWFwW3Jhd118fHJhdzsKfQoKdmFyIGNhY2hlZEdlbz1udWxsOwoKYXN5bmMgZnVuY3Rpb24gbG9hZE1hcChhdHRlbXB0KXsKICBhdHRlbXB0ID0gYXR0ZW1wdHx8MTsKICB0cnl7CiAgICB2YXIgcj1hd2FpdCBmZXRjaCgnaHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L2doL3VkaXQtMDAxL2luZGlhLW1hcHMtZGF0YUBtYXN0ZXIvdG9wb2pzb24vaW5kaWEuanNvbicpOwogICAgaWYoIXIub2spIHRocm93IG5ldyBFcnJvcignSFRUUCAnK3Iuc3RhdHVzKTsKICAgIHZhciB0b3BvPWF3YWl0IHIuanNvbigpOwogICAgY2FjaGVkR2VvPXRvcG9qc29uLmZlYXR1cmUodG9wbyx0b3BvLm9iamVjdHMuc3RhdGVzKTsKICAgIHJlbmRlck1hcChjYWNoZWRHZW8pOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDEwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDMwMDApOwogICAgc2V0VGltZW91dChhcHBseUxheWVyLDYwMDApOwogIH1jYXRjaChlKXsKICAgIGNvbnNvbGUud2FybignW21hcF0gbG9hZCBmYWlsZWQgYXR0ZW1wdCAnK2F0dGVtcHQrJzonLGUubWVzc2FnZSk7CiAgICBpZihhdHRlbXB0PDUpewogICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7bG9hZE1hcChhdHRlbXB0KzEpO30sIGF0dGVtcHQqMjAwMCk7CiAgICB9IGVsc2UgewogICAgICB2YXIgbWk9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm1hcC1pbm5lcicpOwogICAgICBpZihtaSkgbWkuaW5uZXJIVE1MPSc8ZGl2IHN0eWxlPSJjb2xvcjojMmEzYTRhO3BhZGRpbmc6NDBweDt0ZXh0LWFsaWduOmNlbnRlcjtmb250LWZhbWlseTptb25vc3BhY2U7Zm9udC1zaXplOjExcHgiPk1hcCB1bmF2YWlsYWJsZSDigJQgcmVmcmVzaCB0byByZXRyeTwvZGl2Pic7CiAgICB9CiAgfQp9CgpmdW5jdGlvbiByZW5kZXJNYXAoc3RhdGVzKXsKICB2YXIgdz04MDAsaD04MDAscGo9cHJval8odyxoLDI4KTsKICB2YXIgc2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1zdGF0ZXMnKTsKICB2YXIgcGc9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1wdWxzZXMnKTsKICB2YXIgZ2c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21hcC1nbG93Jyk7CiAgc2cuaW5uZXJIVE1MPScnO3BnLmlubmVySFRNTD0nJztnZy5pbm5lckhUTUw9Jyc7CgogIHN0YXRlcy5mZWF0dXJlcy5mb3JFYWNoKGZ1bmN0aW9uKGYpewogICAgaWYoIWYuZ2VvbWV0cnkpIHJldHVybjsKICAgIHZhciBubT1zTmFtZShmLnByb3BlcnRpZXMpLGQ9ZyhubSk7CiAgICB2YXIgcGF0aEVsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdwYXRoJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkJyxnZW8ycGF0aChmLmdlb21ldHJ5LHBqKSk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdjbGFzcycsJ3N0YXRlJyk7CiAgICBwYXRoRWwuc2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnLG5tKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsJ3JnYmEoMjU1LDI1NSwyNTUsMC4wNyknKTsKICAgIHBhdGhFbC5zZXRBdHRyaWJ1dGUoJ3N0cm9rZS13aWR0aCcsJzAuNScpOwogICAgc2cuYXBwZW5kQ2hpbGQocGF0aEVsKTsKCiAgICB2YXIgY3Q9Y3RyKGYuZ2VvbWV0cnkpLGNwPXBqKGN0WzBdLGN0WzFdKTsKCiAgICAvLyBBdG1vc3BoZXJpYyBnbG93IGZvciBoaWdoLWF0dGVudGlvbiBzdGF0ZXMKICAgIGlmKGQuYXR0ZW50aW9uPj02NSl7CiAgICAgIHZhciBnbG93RWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudE5TKCdodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZycsJ2VsbGlwc2UnKTsKICAgICAgdmFyIGdsb3dSPU1hdGgubWluKDYwLDIwK2QuYXR0ZW50aW9uKjAuNSk7CiAgICAgIGdsb3dFbC5zZXRBdHRyaWJ1dGUoJ2N4JyxjcFswXSk7Z2xvd0VsLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgncngnLGdsb3dSKTtnbG93RWwuc2V0QXR0cmlidXRlKCdyeScsZ2xvd1IqMC43KTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsbCcsYUMoZC5hdHRlbnRpb24pKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnb3BhY2l0eScsJzAuMDgnKTsKICAgICAgZ2xvd0VsLnNldEF0dHJpYnV0ZSgnZmlsdGVyJywndXJsKCNzdGF0ZUdsb3cpJyk7CiAgICAgIGdsb3dFbC5zdHlsZS5hbmltYXRpb249J2dsb3dQdWxzZSAnKygyLjUrTWF0aC5yYW5kb20oKSkrJ3MgZWFzZS1pbi1vdXQgJysoTWF0aC5yYW5kb20oKSoyKSsncyBpbmZpbml0ZSc7CiAgICAgIGdnLmFwcGVuZENoaWxkKGdsb3dFbCk7CiAgICB9CgogICAgLy8gRHVhbCBwdWxzZSByaW5ncyBmb3IgdmVyeSBob3Qgc3RhdGVzCiAgICBpZihkLmF0dGVudGlvbj49NzIpewogICAgICBbMCwxXS5mb3JFYWNoKGZ1bmN0aW9uKGkpewogICAgICAgIHZhciByaW5nPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnROUygnaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLCdjaXJjbGUnKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY3gnLGNwWzBdKTtyaW5nLnNldEF0dHJpYnV0ZSgnY3knLGNwWzFdKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnY2xhc3MnLCdwdWxzZS1yaW5nIHAnKyhpKzEpKTsKICAgICAgICByaW5nLnNldEF0dHJpYnV0ZSgnc3Ryb2tlJyxhQyhkLmF0dGVudGlvbikpOwogICAgICAgIHJpbmcuc2V0QXR0cmlidXRlKCdzdHJva2Utd2lkdGgnLCcxJyk7CiAgICAgICAgcmluZy5zdHlsZS5hbmltYXRpb25EZWxheT0oTWF0aC5yYW5kb20oKSoyLjUpKydzJzsKICAgICAgICBwZy5hcHBlbmRDaGlsZChyaW5nKTsKICAgICAgfSk7CiAgICB9CiAgfSk7CiAgYXBwbHlMYXllcigpOwogIGF0dGFjaEludGVyYWN0aW9ucygpOwp9CgovLyBTaW5nbGUgc291cmNlIG9mIHRydXRoIGZvciBlbW90aW9uIGNvbG9yCi8vIEJvdGggbWFwIGFuZCBwYW5lbCBjYWxsIHRoaXMg4oCUIGd1YXJhbnRlZXMgdGhleSBhbHdheXMgbWF0Y2gKZnVuY3Rpb24gZ2V0RWZmZWN0aXZlRW1vdGlvbihubSl7CiAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogIHZhciBkPVNEW25tXXx8e307CiAgdmFyIGVNYXA9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwoKICAvLyAxLiBUcnkgTElWRS5kb21pbmFudF9lbW90aW9uIChzZXQgYnkgL2FwaS9zdGF0ZXMpCiAgdmFyIGRvbT1saXZlLmRvbWluYW50X2Vtb3Rpb258fGQuZG9taW5hbnRfZW1vdGlvbjsKCiAgLy8gMi4gVHJ5IGNvbXB1dGluZyBmcm9tIGVtb3Rpb25zIGJyZWFrZG93bgogIGlmKCFkb20pewogICAgdmFyIGVtb3M9bGl2ZS5lbW90aW9ucyYmT2JqZWN0LmtleXMobGl2ZS5lbW90aW9ucykubGVuZ3RoP2xpdmUuZW1vdGlvbnM6KGQuZW1vdGlvbnN8fHt9KTsKICAgIGRvbT1kb21pbmFudEVtb3Rpb24oZW1vcyk7CiAgfQoKICAvLyAzLiBGYWxsYmFjazogaW5mZXIgZnJvbSBkb21pbmFudCBuYXJyYXRpdmUgKHNhbWUgbG9naWMgZXZlcnl3aGVyZSkKICBpZighZG9tKXsKICAgIHZhciBucD0obGl2ZS5kb21pbmFudF9uYXJyYXRpdmV8fGQuZG9taW5hbnRfbmFycmF0aXZlfHwnJykudG9Mb3dlckNhc2UoKTsKICAgIGlmKG5wLm1hdGNoKC9ib3JkZXJ8dGVycm9yfHNlY3VyaXR5fGNvbmZsaWN0fGF0dGFja3x3YXJ8aW5maWx0cmF0LykpIGRvbT0nZmVhcic7CiAgICBlbHNlIGlmKG5wLm1hdGNoKC9zY2FtfGNvcnJ1cHR8cHJvdGVzdHxhcnJlc3R8dmlvbGVuY2V8b3V0cmFnZXxjcmltZS8pKSBkb209J2FuZ2VyJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2RldmVsb3B8aW52ZXN0fGdyb3d0aHxsYXVuY2h8aW5hdWd1cnxyZWZvcm18cHJvZ3Jlc3N8Ym9vc3QvKSkgZG9tPSdob3BlJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2N1bHR1cmV8aGVyaXRhZ2V8cHJpZGV8dmljdG9yeXxjZWxlYnJhdHxtZWRhbHxhY2hpZXZlbWVudC8pKSBkb209J3ByaWRlJzsKICAgIGVsc2UgaWYobnAubWF0Y2goL2Zsb29kfGRyb3VnaHR8dW5lbXBsb3ltZW50fGluZmxhdGlvbnxzaG9ydGFnZXxjcmlzaXN8Y29uY2Vybi8pKSBkb209J2FueGlldHknOwogICAgZWxzZSBpZigobGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwKT41KSBkb209J2FueGlldHknOyAvLyBhY3RpdmUgc3RhdGUgZGVmYXVsdAogICAgZWxzZSBkb209J2FueGlldHknOyAvLyBnbG9iYWwgZGVmYXVsdAogIH0KCiAgcmV0dXJuIGRvbTsKfQoKLy8gR2V0IGVzdGltYXRlZCBlbW90aW9uIGJyZWFrZG93biAoZm9yIHBhbmVsIGRvbnV0IHdoZW4gcmVhbCBkYXRhIG1pc3NpbmcpCmZ1bmN0aW9uIGdldEVtb3Rpb25CcmVha2Rvd24obm0pewogIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICB2YXIgZD1TRFtubV18fHt9OwogIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOihkLmVtb3Rpb25zfHx7fSk7CiAgaWYoT2JqZWN0LmtleXMoZW1vcykubGVuZ3RoKSByZXR1cm4ge2Vtb3Rpb25zOmVtb3MsZXN0aW1hdGVkOmZhbHNlfTsKICAvLyBCdWlsZCBza2V3ZWQgZGlzdHJpYnV0aW9uIGZyb20gZWZmZWN0aXZlIGVtb3Rpb24KICB2YXIgZG9tPWdldEVmZmVjdGl2ZUVtb3Rpb24obm0pOwogIHZhciBiYXNlPXthbnhpZXR5OjEzLGFuZ2VyOjEzLGhvcGU6MTMscHJpZGU6MTMsZmVhcjoxM307CiAgYmFzZVtkb21dPTQ4OwogIHJldHVybiB7ZW1vdGlvbnM6YmFzZSxlc3RpbWF0ZWQ6dHJ1ZX07Cn0KCmZ1bmN0aW9uIGFwcGx5TGF5ZXIoKXsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgdmFyIG5tPXAuZ2V0QXR0cmlidXRlKCdkYXRhLW5hbWUnKSxkPWcobm0pLGZpbGw7CiAgICBpZihsYXllcj09PSdhdHRlbnRpb24nKSBmaWxsPWFDKGQuYXR0ZW50aW9uKTsKICAgIGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgIHZhciBkZT1nZXRFZmZlY3RpdmVFbW90aW9uKG5tKTsKICAgICAgZmlsbD1lTWFwW2RlXXx8JyMzMzQ0NTUnOwogICAgfQogICAgZWxzZSBmaWxsPXZDKGQudmVsb2NpdHkpOwogICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwnLGZpbGwpOwogICAgKGZ1bmN0aW9uKCl7CiAgICAgIHZhciBzY29yZXM9T2JqZWN0LnZhbHVlcyhTRCkubWFwKGZ1bmN0aW9uKHgpe3JldHVybiB4LmF0dGVudGlvbnx8MDt9KTsKICAgICAgdmFyIG1uPU1hdGgubWluLmFwcGx5KG51bGwsc2NvcmVzKSxteD1NYXRoLm1heC5hcHBseShudWxsLHNjb3Jlcyl8fDE7CiAgICAgIHZhciBuPU1hdGgubWF4KDAsTWF0aC5taW4oMSwoZC5hdHRlbnRpb24tbW4pLyhteC1tbikpKTsKICAgICAgcC5zZXRBdHRyaWJ1dGUoJ2ZpbGwtb3BhY2l0eScsbGF5ZXI9PT0nYXR0ZW50aW9uJz9NYXRoLm1heCgwLjMsMC4zK24qMC43KTowLjg1KTsKICAgIH0pKCk7CiAgfSk7Cn0KCmZ1bmN0aW9uIGF0dGFjaEludGVyYWN0aW9ucygpewogIHZhciB0aXA9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rvb2x0aXAnKTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcjbWFwLXN0YXRlcyAuc3RhdGUnKS5mb3JFYWNoKGZ1bmN0aW9uKHApewogICAgcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLGZ1bmN0aW9uKGUpewogICAgICB2YXIgbm09cC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpOwogICAgICB2YXIgZD1nKG5tKTsKICAgICAgdmFyIGxpdmU9TElWRVtubV18fHt9OwogICAgICB2YXIgdGlwPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b29sdGlwJyk7CiAgICAgIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogICAgICB2YXIgbGF0ZXN0PScnOwogICAgICBpZihkLm5hcnJhdGl2ZXMmJmQubmFycmF0aXZlcy5sZW5ndGgpIGxhdGVzdD1kLm5hcnJhdGl2ZXNbMF0ubmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkLm5hcnJhdGl2ZXNbMF0ubmFtZS5zbGljZSgxKTsKICAgICAgZWxzZSBpZihsaXZlLmRvbWluYW50X25hcnJhdGl2ZSkgbGF0ZXN0PWxpdmUuZG9taW5hbnRfbmFycmF0aXZlLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2xpdmUuZG9taW5hbnRfbmFycmF0aXZlLnNsaWNlKDEpOwoKICAgICAgdmFyIHJvd3M9Jyc7CiAgICAgIGlmKGxheWVyPT09J2F0dGVudGlvbicpewogICAgICAgIHZhciBhdHQ9bGl2ZS5hdHRlbnRpb258fGQuYXR0ZW50aW9ufHwwOwogICAgICAgIHZhciBkbHQ9bGl2ZS5kZWx0YXx8ZC5kZWx0YXx8MDsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5BdHRlbnRpb248L3NwYW4+PHN0cm9uZz4nK2F0dC50b0ZpeGVkKDEpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgKGRsdCE9PTA/JzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPjI0aCBzaGlmdDwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonKyhkbHQ+MD8nI2UwNWEyOCc6JyMzYmI4ZDgnKSsnIj4nKyhkbHQ+MD8nKyc6JycpK2RsdCsnPC9zdHJvbmc+PC9kaXY+JzonJykrCiAgICAgICAgICAobGF0ZXN0Pyc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Ub3AgbmFycmF0aXZlPC9zcGFuPjxzdHJvbmc+JytsYXRlc3QrJzwvc3Ryb25nPjwvZGl2Pic6JycpOwogICAgICB9IGVsc2UgaWYobGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBkb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICAgICAgaWYoZG9tRW1vKXsKICAgICAgICAgIHZhciBlbW9zPWxpdmUuZW1vdGlvbnMmJk9iamVjdC5rZXlzKGxpdmUuZW1vdGlvbnMpLmxlbmd0aD9saXZlLmVtb3Rpb25zOmQuZW1vdGlvbnN8fHt9OwogICAgICAgICAgcm93cz0nPGRpdiBjbGFzcz0idHQtciI+PHNwYW4+RG9taW5hbnQ8L3NwYW4+PHN0cm9uZyBzdHlsZT0iY29sb3I6JytwYWxbZG9tRW1vXSsnIj4nK2RvbUVtby5jaGFyQXQoMCkudG9VcHBlckNhc2UoKStkb21FbW8uc2xpY2UoMSkrJzwvc3Ryb25nPjwvZGl2Pic7CiAgICAgICAgICB2YXIgZUw9T2JqZWN0LmVudHJpZXMoZW1vcykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiBiWzFdLWFbMV07fSk7CiAgICAgICAgICB2YXIgdG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgICAgICAgaWYodG90PjAmJnRvdDw9MS4wMSl7ZUw9ZUwubWFwKGZ1bmN0aW9uKGt2KXtyZXR1cm5ba3ZbMF0sTWF0aC5yb3VuZChrdlsxXSoxMDApXTt9KTt0b3Q9ZUwucmVkdWNlKGZ1bmN0aW9uKHMsa3Ype3JldHVybiBzK2t2WzFdO30sMCk7fQogICAgICAgICAgcm93cys9ZUwuc2xpY2UoMCwzKS5tYXAoZnVuY3Rpb24oa3Ype3JldHVybiAnPGRpdiBjbGFzcz0idHQtciI+PHNwYW4gc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweCI+PHNwYW4gc3R5bGU9IndpZHRoOjVweDtoZWlnaHQ6NXB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6JytwYWxba3ZbMF1dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2siPjwvc3Bhbj4nK2t2WzBdKyc8L3NwYW4+PHN0cm9uZz4nK01hdGgucm91bmQoa3ZbMV0qMTAwL01hdGgubWF4KDEsdG90KSkrJyU8L3N0cm9uZz48L2Rpdj4nO30pLmpvaW4oJycpOwogICAgICAgIH0KICAgICAgfSBlbHNlIHsKICAgICAgICB2YXIgdmVsPWxpdmUudmVsb2NpdHl8fGQudmVsb2NpdHl8fDA7CiAgICAgICAgdmFyIHZlbERpcj12ZWw+MC4xPydSaXNpbmcgZmFzdCc6dmVsPjAuMDI/J1Jpc2luZyc6dmVsPC0wLjA1PydDb29saW5nJzonU3RhYmxlJzsKICAgICAgICB2YXIgdmVsQ29sPXZlbD4wLjAyPycjZTA1YTI4Jzp2ZWw8LTAuMDI/JyMzYmI4ZDgnOicjNTU2Njc3JzsKICAgICAgICByb3dzPSc8ZGl2IGNsYXNzPSJ0dC1yIj48c3Bhbj5Nb21lbnR1bTwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L3N0cm9uZz48L2Rpdj4nKwogICAgICAgICAgJzxkaXYgY2xhc3M9InR0LXIiPjxzcGFuPkRpcmVjdGlvbjwvc3Bhbj48c3Ryb25nIHN0eWxlPSJjb2xvcjonK3ZlbENvbCsnIj4nK3ZlbERpcisnPC9zdHJvbmc+PC9kaXY+JzsKICAgICAgfQoKICAgICAgdGlwLmlubmVySFRNTD0nPGRpdiBjbGFzcz0idHQtbiI+JytubSsnPC9kaXY+Jytyb3dzKyhsYXRlc3QmJmxheWVyIT09J2F0dGVudGlvbic/JzxkaXYgY2xhc3M9InR0LW5hciI+PHN0cm9uZz5OYXJyYXRpdmU8L3N0cm9uZz4nK2xhdGVzdCsnPC9kaXY+JzonJyk7CiAgICAgIHZhciByZWN0PWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tYXAtaW5uZXInKS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICAgICAgdGlwLnN0eWxlLmxlZnQ9TWF0aC5taW4oZS5jbGllbnRYLXJlY3QubGVmdCsxNCxyZWN0LndpZHRoLTE5MCkrJ3B4JzsKICAgICAgdGlwLnN0eWxlLnRvcD1NYXRoLm1pbihlLmNsaWVudFktcmVjdC50b3ArMTQscmVjdC5oZWlnaHQtMTUwKSsncHgnOwogICAgICB0aXAuc3R5bGUub3BhY2l0eT0nMSc7CiAgICB9KTsKcC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJyxmdW5jdGlvbigpe3RpcC5zdHlsZS5vcGFjaXR5PTA7fSk7CiAgICBwLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJyxmdW5jdGlvbigpe3NlbGVjdF8ocC5nZXRBdHRyaWJ1dGUoJ2RhdGEtbmFtZScpKTt9KTsKICB9KTsKfQoKLy8gU1RBVEUgUEFORUwKZnVuY3Rpb24gc2VsZWN0XyhubSl7CiAgU0VMPW5tOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJyNtYXAtc3RhdGVzIC5zdGF0ZScpLmZvckVhY2goZnVuY3Rpb24ocCl7CiAgICBwLmNsYXNzTGlzdC50b2dnbGUoJ3NlbGVjdGVkJyxwLmdldEF0dHJpYnV0ZSgnZGF0YS1uYW1lJyk9PT1ubSk7CiAgfSk7CiAgLy8gU2hvdyBsb2FkaW5nIHN0YXRlIGltbWVkaWF0ZWx5IHdpdGggd2hhdGV2ZXIgTElWRSBkYXRhIHdlIGhhdmUKICB2YXIgcGFuZWw9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXRlLWRldGFpbCcpOwogIGlmKHBhbmVsKXsKICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgIHBhbmVsLmlubmVySFRNTD0KICAgICAgJzxkaXYgY2xhc3M9InNwLWhlYWQiPicrCiAgICAgICAgJzxkaXY+PGRpdiBjbGFzcz0ic3AtZWsiPicrKGxheWVyPT09J2F0dGVudGlvbic/J05hcnJhdGl2ZSBwYW5lbCc6bGF5ZXI9PT0nZW1vdGlvbic/J0Vtb3Rpb25hbCByZWdpc3Rlcic6J01vbWVudHVtIHBhbmVsJykrJzwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNwLW5hbWUiPicrbm0rJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxidXR0b24gY2xhc3M9ImZhdi1idG4gJysoRkFWUy5oYXMobm0pPydvbic6JycpKyciIGRhdGEtbm09Iicrbm0rJyIgb25jbGljaz0idG9nZ2xlRmF2KHRoaXMuZGF0YXNldC5ubSkiIHRpdGxlPSJUcmFjayI+JysKICAgICAgICAgICc8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iJysoRkFWUy5oYXMobm0pPydjdXJyZW50Q29sb3InOidub25lJykrJyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNMTkgMjFsLTctNS03IDVWNWEyIDIgMCAwIDEgMi0yaDEwYTIgMiAwIDAgMSAyIDJ6Ii8+PC9zdmc+JysKICAgICAgICAnPC9idXR0b24+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjIwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZmFpbnQpO2xldHRlci1zcGFjaW5nOjAuMDhlbSI+JysKICAgICAgICAnTG9hZGluZyBzaWduYWxzIGZvciAnK25tKycuLi4nKwogICAgICAgIChsaXZlLmF0dGVudGlvbj8nPGJyPjxicj48c3BhbiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tc2VyaWYpO2ZvbnQtc2l6ZToxOHB4O2NvbG9yOnZhcigtLWluaykiPkF0dGVudGlvbiAnK2xpdmUuYXR0ZW50aW9uLnRvRml4ZWQoMSkrJzwvc3Bhbj4nOicnKSsKICAgICAgICAobGl2ZS5kb21pbmFudF9lbW90aW9uPyc8YnI+PHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KSI+JytsaXZlLmRvbWluYW50X2Vtb3Rpb24rJyBzaWduYWwgZG9taW5hbnQ8L3NwYW4+JzonJykrCiAgICAgICc8L2Rpdj4nOwogIH0KICAvLyBGZXRjaCBmdWxsIGRldGFpbCB0aGVuIHJlbmRlcgogIGZldGNoRGV0YWlsKG5tKS50aGVuKGZ1bmN0aW9uKCl7CiAgICBpZihTRUw9PT1ubSl7CiAgICAgIHJlbmRlclBhbmVsKG5tKTsKICAgICAgLy8gVXBkYXRlIGp1c3QgdGhpcyBzdGF0ZSdzIG1hcCBjb2xvciB0byBtYXRjaCB0aGUgcGFuZWwKICAgICAgdmFyIHBhdGg9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21hcC1zdGF0ZXMgLnN0YXRlW2RhdGEtbmFtZT0iJytubSsnIl0nKTsKICAgICAgaWYocGF0aCYmbGF5ZXI9PT0nZW1vdGlvbicpewogICAgICAgIHZhciBsaXZlPUxJVkVbbm1dfHx7fTsKICAgICAgICB2YXIgZU1hcD17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgdmFyIGRvbT1saXZlLmRvbWluYW50X2Vtb3Rpb258fGRvbWluYW50RW1vdGlvbihsaXZlLmVtb3Rpb25zfHx7fSk7CiAgICAgICAgaWYoZG9tJiZlTWFwW2RvbV0pIHBhdGguc2V0QXR0cmlidXRlKCdmaWxsJyxlTWFwW2RvbV0pOwogICAgICB9IGVsc2UgewogICAgICAgIGFwcGx5TGF5ZXIoKTsKICAgICAgfQogICAgfQogIH0pLmNhdGNoKGZ1bmN0aW9uKGUpewogICAgY29uc29sZS53YXJuKCdbc2VsZWN0XScsZSk7CiAgICBpZihTRUw9PT1ubSkgcmVuZGVyUGFuZWwobm0pOwogIH0pOwp9CgpmdW5jdGlvbiByZW5kZXJQYW5lbChubSl7CiAgdmFyIGQ9ZyhubSk7CiAgdmFyIHBhbmVsPWRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0ZS1kZXRhaWwnKTsKICBpZighcGFuZWwpIHJldHVybjsKICB2YXIgcGFsPXthbnhpZXR5OicjODg0NGNjJyxhbmdlcjonI2RkMjI0NCcsaG9wZTonIzMzY2M2NicscHJpZGU6JyMzM2FhY2MnLGZlYXI6JyNjYzg4MzMnfTsKCiAgdmFyIGhlYWRlcj0KICAgICc8ZGl2IGNsYXNzPSJzcC1oZWFkIj4nKwogICAgICAnPGRpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcC1layIgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDsiPicrCiAgICAgICAgICAobGF5ZXI9PT0nYXR0ZW50aW9uJz8nTmFycmF0aXZlIHBhbmVsJzpsYXllcj09PSdlbW90aW9uJz8nRW1vdGlvbmFsIHJlZ2lzdGVyJzonTW9tZW50dW0gcGFuZWwnKSsKICAgICAgICAgIChkLmNvbmZpZGVuY2U/JzxzcGFuIHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6Ny41cHg7bGV0dGVyLXNwYWNpbmc6MC4xZW07cGFkZGluZzoycHggNnB4O2JvcmRlci1yYWRpdXM6M3B4O2JhY2tncm91bmQ6JysoZC5jb25maWRlbmNlPT09J0hJR0gnPydyZ2JhKDUxLDIwNCwxMDIsMC4xKSc6ZC5jb25maWRlbmNlPT09J01FRElVTSc/J3JnYmEoMjI0LDkwLDQwLDAuMSknOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpJykrJztjb2xvcjonKyhkLmNvbmZpZGVuY2U9PT0nSElHSCc/JyMzM2NjNjYnOmQuY29uZmlkZW5jZT09PSdNRURJVU0nPycjZTA1YTI4JzoncmdiYSgyNTUsMjU1LDI1NSwwLjMpJykrJyI+JytkLmNvbmZpZGVuY2UrJyBTSUdOQUw8L3NwYW4+JzonJykrCiAgICAgICAgICAoZC5pc19yZWdpb25hbF9zdG9yeT8nPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo3LjVweDtsZXR0ZXItc3BhY2luZzowLjFlbTtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7YmFja2dyb3VuZDpyZ2JhKDU5LDE4NCwyMTYsMC4xKTtjb2xvcjojM2JiOGQ4Ij5SRUdJT05BTCBTUElLRTwvc3Bhbj4nOicnKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3AtbmFtZSI+JytubSsnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8YnV0dG9uIGNsYXNzPSJmYXYtYnRuICcrKEZBVlMuaGFzKG5tKT8nb24nOicnKSsnIiBkYXRhLW5tPSInK25tKyciIG9uY2xpY2s9InRvZ2dsZUZhdih0aGlzLmRhdGFzZXQubm0pIiB0aXRsZT0iVHJhY2siPicrCiAgICAgICAgJzxzdmcgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSInKyhGQVZTLmhhcyhubSk/J2N1cnJlbnRDb2xvcic6J25vbmUnKSsnIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik0xOSAyMWwtNy01LTcgNVY1YTIgMiAwIDAgMSAyLTJoMTBhMiAyIDAgMCAxIDIgMnoiLz48L3N2Zz4nKwogICAgICAnPC9idXR0b24+JysKICAgICc8L2Rpdj4nOwoKICB2YXIgYm9keT0nJzsKCiAgaWYobGF5ZXI9PT0nYXR0ZW50aW9uJyl7CiAgICB2YXIgZFM9ZC5kZWx0YT49MD8nKyc6JycsZEM9ZC5kZWx0YT49MD8ndXAnOidkbic7CiAgICB2YXIgbmFycj1kLm5hcnJhdGl2ZXN8fFtdOwogICAgdmFyIHRsPShkLnRpbWVsaW5lJiZkLnRpbWVsaW5lLmxlbmd0aCk/ZC50aW1lbGluZTpbMCwwLDAsMCwwLDAsMCxkLmF0dGVudGlvbnx8MF07CiAgICB2YXIgdG1uPU1hdGgubWluLmFwcGx5KG51bGwsdGwpLHRteD1NYXRoLm1heC5hcHBseShudWxsLHRsKSx0cj1NYXRoLm1heCgxLHRteC10bW4pOwogICAgdmFyIHR3PTI2MCx0aD02Mix0cD01OwogICAgdmFyIHB0cz10bC5tYXAoZnVuY3Rpb24odixpKXtyZXR1cm5bdHArKGkvKHRsLmxlbmd0aC0xKSkqKHR3LXRwKjIpLHRwKygxLSh2LXRtbikvdHIpKih0aC10cCoyKV07fSk7CiAgICB2YXIgcEQ9cHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybihpPT09MD8nTSc6J0wnKStwWzBdLnRvRml4ZWQoMSkrJywnK3BbMV0udG9GaXhlZCgxKTt9KS5qb2luKCcnKTsKICAgIHZhciBhRD1wRCsnIEwnK3B0c1twdHMubGVuZ3RoLTFdWzBdKycsJysodGgtdHApKycgTCcrcHRzWzBdWzBdKycsJysodGgtdHApKycgWic7CiAgICB2YXIgYWM9YUMoZC5hdHRlbnRpb258fDApOwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdIb3cgaW50ZW5zZWx5ICcrKG5tLnNwbGl0KCcgJylbMF0pKycgaXMgYmVpbmcgZGlzY3Vzc2VkIG5hdGlvbmFsbHkuIFNjb3JlIG9mICcrZC5hdHRlbnRpb24rJyBtZWFucyAnKyhkLmF0dGVudGlvbj42MD8ndmVyeSBoaWdoIOKAlCB0aGlzIHN0YXRlIGRvbWluYXRlcyBuYXRpb25hbCBkaXNjb3Vyc2UnOmQuYXR0ZW50aW9uPjM1PydlbGV2YXRlZCDigJQgY2xlYXJseSBpbiB0aGUgbmF0aW9uYWwgY29udmVyc2F0aW9uJzpkLmF0dGVudGlvbj4xNT8nbW9kZXJhdGUg4oCUIHNvbWUgbmF0aW9uYWwgY292ZXJhZ2UnOmQuYXR0ZW50aW9uPjU/J2xvdyDigJQgbGltaXRlZCBuYXRpb25hbCBhdHRlbnRpb24nOidtaW5pbWFsIOKAlCBmZXcgc2lnbmFscyBkZXRlY3RlZCcpKycuJysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9Imluc2lnaHQiIHN0eWxlPSInKyhkLmNvbmZpZGVuY2U9PT0iTE9XIj8nYm9yZGVyLWNvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4wNik7Y29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtc3R5bGU6aXRhbGljJzonJykrJyI+JysoKGQuY29uZmlkZW5jZT09PSJMT1ciJiYhZC5zdW1tYXJ5KT8nTGltaXRlZCBzaWduYWxzIGRldGVjdGVkIGZvciAnK25tKycuIE1vbml0b3JpbmcgcmVnaW9uYWwgc291cmNlcy4nOmQuc3VtbWFyeXx8J0NvbGxlY3Rpbmcgc2lnbmFscyBmb3IgJytubSsnLi4uJykrJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzY29yZS1zdHJpcCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPkF0dGVudGlvbjwvZGl2PjxkaXYgY2xhc3M9InNzLXZhbCI+JysoZC5hdHRlbnRpb258fDApKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIHNoaWZ0PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJytkQysnIj4nK2RTKyhkLmRlbHRhfHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtZGl2aWRlciI+PC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0ic3MtaXRlbSI+PGRpdiBjbGFzcz0ic3MtbGFiZWwiPlRvcCBuYXJyYXRpdmU8L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKG5hcnJbMF0/bmFyclswXS5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK25hcnJbMF0ubmFtZS5zbGljZSgxKTon4oCUJykrJzwvZGl2PjwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5OYXJyYXRpdmUgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAobmFyci5sZW5ndGg/CiAgICAgICAgICAnPGRpdiBjbGFzcz0ibmFyLWxpc3QiPicrbmFyci5tYXAoZnVuY3Rpb24obil7CiAgICAgICAgICAgIHZhciBubj1uLm5hbWU/bi5uYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK24ubmFtZS5zbGljZSgxKTpuLm5hbWU7CiAgICAgICAgICAgIHZhciB2YWw9dHlwZW9mIG4udmFsPT09J251bWJlcic/bi52YWw6MDsKICAgICAgICAgICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJuYXItaXRlbTIiPjxkaXYgY2xhc3M9Im5pLWxhYmVsIj4nK25uKyhuLmRpcj09PSd1cCc/JyA8c3BhbiBzdHlsZT0iY29sb3I6I2UwNWEyODtmb250LXNpemU6OXB4IiB0aXRsZT0iZ2FpbmluZyB0cmFjdGlvbiI+4oaRPC9zcGFuPic6bi5kaXI9PT0nZG93bic/JyA8c3BhbiBzdHlsZT0iY29sb3I6IzNiYjhkODtmb250LXNpemU6OXB4IiB0aXRsZT0icmV0cmVhdGluZyI+4oaTPC9zcGFuPic6JycpKyc8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS12YWwiPicrdmFsLnRvRml4ZWQoMSkrJyU8L2Rpdj4nKwogICAgICAgICAgICAgICc8ZGl2IGNsYXNzPSJuaS10cmFjayI+PGRpdiBjbGFzcz0ibmktZmlsbCIgc3R5bGU9IndpZHRoOicrTWF0aC5taW4oMTAwLHZhbCoyLjUpKyclO2JhY2tncm91bmQ6Jysobi5kaXI9PT0ndXAnPycjZTA1YTI4JzpuLmRpcj09PSdkb3duJz8nIzNiYjhkOCc6JyMzMzQ0NTUnKSsnIj48L2Rpdj48L2Rpdj48L2Rpdj4nOwogICAgICAgICAgfSkuam9pbignJykrJzwvZGl2Pic6CiAgICAgICAgICAnPGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O3BhZGRpbmc6OHB4IDAiPkxvdy1zaWduYWwgcmVnaW9uLiBNb25pdG9yaW5nIHJlZ2lvbmFsIHNvdXJjZXMuPC9kaXY+JykrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BdHRlbnRpb24g4oCUIDggZGF5czwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InRsLXdyYXAiPjxzdmcgdmlld0JveD0iMCAwICcrdHcrJyAnK3RoKyciIHN0eWxlPSJ3aWR0aDoxMDAlO2hlaWdodDoxMDAlIj4nKwogICAgICAgICAgJzxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0idGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKyciIHgxPSIwIiB4Mj0iMCIgeTE9IjAiIHkyPSIxIj4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iJythYysnIiBzdG9wLW9wYWNpdHk9IjAuMjUiLz4nKwogICAgICAgICAgICAnPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSInK2FjKyciIHN0b3Atb3BhY2l0eT0iMCIvPicrCiAgICAgICAgICAnPC9saW5lYXJHcmFkaWVudD48L2RlZnM+JysKICAgICAgICAgICc8cGF0aCBkPSInK2FEKyciIGZpbGw9InVybCgjdGxnJytubS5yZXBsYWNlKC9bXmEtel0vZ2ksJycpKycpIiAvPicrCiAgICAgICAgICAnPHBhdGggZD0iJytwRCsnIiBmaWxsPSJub25lIiBzdHJva2U9IicrYWMrJyIgc3Ryb2tlLXdpZHRoPSIxLjIiLz4nKwogICAgICAgICAgcHRzLm1hcChmdW5jdGlvbihwLGkpe3JldHVybiAnPGNpcmNsZSBjeD0iJytwWzBdKyciIGN5PSInK3BbMV0rJyIgcj0iJysoaT09PXB0cy5sZW5ndGgtMT8yLjI6MS4yKSsnIiBmaWxsPSInK2FjKyciLz4nO30pLmpvaW4oJycpKwogICAgICAgICc8L3N2Zz48L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNwLXNlY3Rpb24iPjxkaXYgY2xhc3M9InNwLXNlYy10aXRsZSI+U2lnbmFscyA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tZmFpbnQpIj4nKyhkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aD9kLmFydGljbGVzLmxlbmd0aDowKSsnPC9zcGFuPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9ImFydC1saXN0Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMubWFwKGZ1bmN0aW9uKGEpewogICAgICAgICAgICAgIHZhciB0eHQ9YS50eHR8fGEudGl0bGV8fCcnOwogICAgICAgICAgICAgIHZhciBzcmM9YS5zcmN8fCcnOwogICAgICAgICAgICAgIC8vIFNraXAgZW1wdHkgb3IgdmVyeSBzaG9ydCB0aXRsZXMKICAgICAgICAgICAgICBpZih0eHQubGVuZ3RoPDI1KSByZXR1cm4gbnVsbDsKICAgICAgICAgICAgICAvLyBTa2lwIFlvdVR1YmUgZW50aXJlbHkKICAgICAgICAgICAgICBpZihzcmMuaW5kZXhPZigneW91dHViZScpPj0wKSByZXR1cm4gbnVsbDsKICAgICAgICAgICAgICAvLyBTa2lwIG5vaXNlIGtleXdvcmRzCiAgICAgICAgICAgICAgdmFyIHRsPXR4dC50b0xvd2VyQ2FzZSgpOwogICAgICAgICAgICAgIHZhciBub2lzZUt3PVsndHJ1bXAnLCd1a3JhaW5lJywncnVzc2lhJywnZ2F6YScsJ3JlY2lwZScsJ2hvcm9zY29wZScsJ2NlbGVicml0eScsJ2JveCBvZmZpY2UnLCdtdXNpYyB2aWRlbycsJ2xpdmUgc2NvcmUnLCdjcmlja2V0IHNjb3JlJywnd2F0Y2g6JywncGhvdG9zOicsJ2JyZWFraW5nOiddOwogICAgICAgICAgICAgIGlmKG5vaXNlS3cuc29tZShmdW5jdGlvbihrKXtyZXR1cm4gdGwuaW5kZXhPZihrKT49MDt9KSkgcmV0dXJuIG51bGw7CiAgICAgICAgICAgICAgLy8gU291cmNlIGxhYmVsIOKAlCBiYWNrZW5kIGFscmVhZHkgY2xlYW5lZCB0aGlzCiAgICAgICAgICAgICAgLy8gSWYgZW1wdHkgKG5hdGlvbmFsIG1lZGlhIGhpZGRlbiksIHNob3cgbm8gc291cmNlIGxhYmVsCiAgICAgICAgICAgICAgdmFyIHNyY0h0bWw9c3JjPyc8ZGl2IGNsYXNzPSJhcnQtc3JjIj4nK3NyYysnPC9kaXY+JzonJzsKICAgICAgICAgICAgICByZXR1cm4gJzxkaXYgY2xhc3M9ImFydC1pdGVtIj4nK3NyY0h0bWwrJzxkaXYgY2xhc3M9ImFydC10eHQiPicrdHh0Kyc8L2Rpdj48L2Rpdj4nOwogICAgICAgICAgICB9KS5maWx0ZXIoQm9vbGVhbikuam9pbignJyk6CiAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7cGFkZGluZzo2cHggMCI+Tm8gc2lnbmFscyBjb2xsZWN0ZWQgeWV0LjwvZGl2PicpKwogICAgICAgICc8L2Rpdj4nKwogICAgICAnPC9kaXY+JzsKCiAgfSBlbHNlIGlmKGxheWVyPT09J2Vtb3Rpb24nKXsKICAgIC8vIFVzZSBzYW1lIGZ1bmN0aW9ucyBhcyBtYXAg4oCUIGd1YXJhbnRlZWQgdG8gbWF0Y2gKICAgIHZhciBtYXBEb21FbW89Z2V0RWZmZWN0aXZlRW1vdGlvbihubSk7CiAgICB2YXIgYnJlYWtkb3duPWdldEVtb3Rpb25CcmVha2Rvd24obm0pOwogICAgdmFyIGVtb3Rpb25zPWJyZWFrZG93bi5lbW90aW9uczsKICAgIHZhciBoYXNFbW9zPSFicmVha2Rvd24uZXN0aW1hdGVkOwogICAgdmFyIGVMPU9iamVjdC5lbnRyaWVzKGVtb3Rpb25zKTsKICAgIHZhciBlVG90PWVMLnJlZHVjZShmdW5jdGlvbihzLGt2KXtyZXR1cm4gcytrdlsxXTt9LDApOwogICAgaWYoZVRvdD4wJiZlVG90PD0xLjAxKXtlTD1lTC5tYXAoZnVuY3Rpb24oa3Ype3JldHVybltrdlswXSxNYXRoLnJvdW5kKGt2WzFdKjEwMCldO30pO30KICAgIHZhciB0b3Q9TWF0aC5tYXgoMSxlTC5yZWR1Y2UoZnVuY3Rpb24ocyxrdil7cmV0dXJuIHMra3ZbMV07fSwwKSk7CiAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuIGJbMV0tYVsxXTt9KTsKICAgIGlmKCFlTC5sZW5ndGgpe3BhbmVsLmlubmVySFRNTD1oZWFkZXIrJzxkaXYgc3R5bGU9InBhZGRpbmc6MjBweDtjb2xvcjp2YXIoLS1mYWludCk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHgiPk5vIGVtb3Rpb24gZGF0YSB5ZXQuPC9kaXY+JztyZXR1cm47fQogICAgLy8gZG9tRW1vID0gc2FtZSBhcyBtYXAgY29sb3IgKGZyb20gZ2V0RWZmZWN0aXZlRW1vdGlvbikKICAgIHZhciBkb21FbW89bWFwRG9tRW1vOwoKICAgIC8vIENvbnRleHR1YWwgZW1vdGlvbiByZWFzb24g4oCUIHN0YXRlLXNwZWNpZmljLCBuYXJyYXRpdmUtZHJpdmVuCiAgICAvLyBTdGF0ZS1zcGVjaWZpYyBvdmVycmlkZXMgZm9yIGNvbnRleHR1YWwgYWNjdXJhY3kKICAgIHZhciBfc3RhdGVFbW9Db250ZXh0PXsKICAgICAgJ0phbW11IGFuZCBLYXNobWlyJzoge2FuZ2VyOidTZWN1cml0eSBpbmNpZGVudHMgYW5kIHBvbGl0aWNhbCB0ZW5zaW9ucyBydW5uaW5nIGhpZ2gnLCBmZWFyOidPbmdvaW5nIHNlY3VyaXR5IHNpdHVhdGlvbiBrZWVwaW5nIGFueGlldHkgZWxldmF0ZWQnLCBhbnhpZXR5OidVbmNlcnRhaW50eSBhcm91bmQgcG9saXRpY2FsIHN0YXR1cyBhbmQgc2VjdXJpdHknfSwKICAgICAgJ01hbmlwdXInOiAgICAgICAgICAge2FuZ2VyOidFdGhuaWMgY29uZmxpY3QgYW5kIHZpb2xlbmNlIGRyaXZpbmcgaW50ZW5zZSBwdWJsaWMgZnJ1c3RyYXRpb24nLCBmZWFyOidPbmdvaW5nIGNvbW11bmFsIGNvbmZsaWN0IG1ha2luZyBjb21tdW5pdGllcyBmZWFyZnVsJywgYW54aWV0eTonUHJvbG9uZ2VkIGV0aG5pYyB0ZW5zaW9ucyBjcmVhdGluZyBkZWVwIHVuY2VydGFpbnR5J30sCiAgICAgICdQdW5qYWInOiAgICAgICAgICAgIHthbmdlcjonUG9saXRpY2FsIGRpc3B1dGVzIGFuZCBhZ3JhcmlhbiBkaXN0cmVzcyBmdWVsbGluZyBmcnVzdHJhdGlvbicsIGFueGlldHk6J0Vjb25vbWljIHByZXNzdXJlcyBvbiBmYXJtaW5nIGNvbW11bml0aWVzIGNyZWF0aW5nIGNvbmNlcm4nfSwKICAgICAgJ1dlc3QgQmVuZ2FsJzogICAgICAge2FuZ2VyOidQb2xpdGljYWwgdmlvbGVuY2UgYW5kIHBhcnR5IHJpdmFscnkgZ2VuZXJhdGluZyBvdXRyYWdlJywgYW54aWV0eTonUG9saXRpY2FsIHVuY2VydGFpbnR5IGFuZCBnb3Zlcm5hbmNlIGNvbmNlcm5zJ30sCiAgICAgICdVdHRhciBQcmFkZXNoJzogICAgIHthbmdlcjonTGF3IGFuZCBvcmRlciBpbmNpZGVudHMgYW5kIHBvbGl0aWNhbCBkaXNwdXRlcyBkcml2aW5nIGFuZ2VyJywgYW54aWV0eTonRWNvbm9taWMgY29uY2VybnMgYW5kIGdvdmVybmFuY2UgZ2FwcyBjcmVhdGluZyB1bmVhc2UnfSwKICAgICAgJ0RlbGhpJzogICAgICAgICAgICAge2FuZ2VyOidHb3Zlcm5hbmNlIGRpc3B1dGVzIGFuZCBwb2xpdGljYWwgY2xhc2hlcyBkcml2aW5nIGZydXN0cmF0aW9uJywgYW54aWV0eTonQWlyIHF1YWxpdHksIGdvdmVybmFuY2UgYW5kIHBvbGl0aWNhbCB1bmNlcnRhaW50eSd9LAogICAgfTsKICAgIC8vIFVzZSBzdGF0ZS1zcGVjaWZpYyBjb250ZXh0IGlmIGF2YWlsYWJsZSBmb3IgZG9taW5hbnQgZW1vdGlvbgogICAgdmFyIF9zdGF0ZVNwZWNpZmljPShfc3RhdGVFbW9Db250ZXh0W25tXXx8e30pW2RvbUVtb118fG51bGw7CgogICAgdmFyIF9lbW9SZWFzb25zPXsKICAgICAgYW5nZXI6ewogICAgICAgICdib3JkZXIgaXNzdWVzJzogICAnQm9yZGVyIHRlbnNpb25zIGFuZCBzZWN1cml0eSBpbmNpZGVudHMgZnVlbGxpbmcgcHVibGljIGZydXN0cmF0aW9uJywKICAgICAgICAnbGF3ICYgb3JkZXInOiAgICAgJ0NyaW1lIGFuZCBsYXcgZW5mb3JjZW1lbnQgaW5jaWRlbnRzIGdlbmVyYXRpbmcgc3Ryb25nIHB1YmxpYyBhbmdlcicsCiAgICAgICAgJ2NvcnJ1cHRpb24nOiAgICAgICdTY2FtIGV4cG9zdXJlIGFuZCBnb3Zlcm5hbmNlIGZhaWx1cmVzIGZ1ZWxsaW5nIG91dHJhZ2UnLAogICAgICAgICdlbGVjdGlvbnMnOiAgICAgICAnRWxlY3RvcmFsIGRpc3B1dGVzIGFuZCBwb2xpdGljYWwgcml2YWxyaWVzIGludGVuc2lmeWluZyBwdWJsaWMgYW5nZXInLAogICAgICAgICdwcm90ZXN0JzogICAgICAgICAnQWN0aXZlIHN0cmVldCBwcm90ZXN0cyBhbmQgYWdpdGF0aW9ucyBkcml2aW5nIGRpc2NvdXJzZScsCiAgICAgICAgJ2dvdmVybmFuY2UnOiAgICAgICdBZG1pbmlzdHJhdGl2ZSBmYWlsdXJlcyBhbmQgcG9saWN5IGRpc3B1dGVzIGRyYXdpbmcgYW5nZXInLAogICAgICAgICdjYXN0ZSc6ICAgICAgICAgICAnQ2FzdGUgZGlzY3JpbWluYXRpb24gaW5jaWRlbnRzIHN0b2tpbmcgY29tbXVuaXR5IHRlbnNpb25zJywKICAgICAgICAncmVsaWdpb24nOiAgICAgICAgJ0NvbW11bmFsIHRlbnNpb25zIGdlbmVyYXRpbmcgc3Ryb25nIGVtb3Rpb25hbCByZWFjdGlvbnMnLAogICAgICAgICdmYXJtZXIgaXNzdWVzJzogICAnQWdyYXJpYW4gZGlzdHJlc3MgZHJpdmluZyBmYXJtZXIgYWdpdGF0aW9uJywKICAgICAgICAnc2VjdXJpdHknOiAgICAgICAgJ1NlY3VyaXR5IGluY2lkZW50cyBmdWVsbGluZyBmZWFyIGFuZCBhbmdlcicsCiAgICAgIH0sCiAgICAgIGFueGlldHk6ewogICAgICAgICdlY29ub215JzogICAgICAgICAnRWNvbm9taWMgdW5jZXJ0YWludHkgY3JlYXRpbmcgd2lkZXNwcmVhZCBhcHByZWhlbnNpb24nLAogICAgICAgICdpbmZsYXRpb24nOiAgICAgICAnUmlzaW5nIHByaWNlcyBlcm9kaW5nIGhvdXNlaG9sZCBjb25maWRlbmNlJywKICAgICAgICAndW5lbXBsb3ltZW50JzogICAgJ0pvYiBtYXJrZXQgY29uY2VybnMgZ2VuZXJhdGluZyBhbnhpZXR5IGFjcm9zcyB0aGUgc3RhdGUnLAogICAgICAgICdib3JkZXIgaXNzdWVzJzogICAnQm9yZGVyIHRlbnNpb25zIGNyZWF0aW5nIHNlY3VyaXR5IGFueGlldHknLAogICAgICAgICdlbnZpcm9ubWVudCc6ICAgICAnRW52aXJvbm1lbnRhbCBjcmlzaXMgdHJpZ2dlcmluZyBwdWJsaWMgY29uY2VybicsCiAgICAgICAgJ2Zhcm1lciBpc3N1ZXMnOiAgICdDcm9wIGRpc3RyZXNzIGFuZCBtb25zb29uIHVuY2VydGFpbnR5IGNyZWF0aW5nIGFueGlldHknLAogICAgICAgICdoZWFsdGgnOiAgICAgICAgICAnSGVhbHRoIGVtZXJnZW5jeSBzaWduYWxzIGVsZXZhdGluZyBwdWJsaWMgY29uY2VybicsCiAgICAgICAgJ2dvdmVybmFuY2UnOiAgICAgICdQb2xpY3kgdW5jZXJ0YWludHkgZ2VuZXJhdGluZyBpbnN0aXR1dGlvbmFsIGFueGlldHknLAogICAgICAgICdzZWN1cml0eSc6ICAgICAgICAnU2VjdXJpdHkgc2l0dWF0aW9uIGNyZWF0aW5nIHVuZGVybHlpbmcgZmVhcicsCiAgICAgIH0sCiAgICAgIGhvcGU6ewogICAgICAgICdlbGVjdGlvbnMnOiAgICAgICAnRWxlY3RvcmFsIG1vbWVudHVtIGdlbmVyYXRpbmcgb3B0aW1pc20gZm9yIHBvbGl0aWNhbCBjaGFuZ2UnLAogICAgICAgICdlY29ub215JzogICAgICAgICAnRWNvbm9taWMgaW5kaWNhdG9ycyBzaG93aW5nIGVhcmx5IHJlY292ZXJ5IHNpZ25hbHMnLAogICAgICAgICdnb3Zlcm5hbmNlJzogICAgICAnUG9saWN5IGFubm91bmNlbWVudHMgY3JlYXRpbmcgY2F1dGlvdXMgb3B0aW1pc20nLAogICAgICAgICdpbmZyYXN0cnVjdHVyZSc6ICAnSW5mcmFzdHJ1Y3R1cmUgZGV2ZWxvcG1lbnQgZ2VuZXJhdGluZyBkZXZlbG9wbWVudCBob3BlcycsCiAgICAgICAgJ2VkdWNhdGlvbic6ICAgICAgICdFZHVjYXRpb24gcmVmb3JtcyBidWlsZGluZyBleHBlY3RhdGlvbnMgZm9yIGNoYW5nZScsCiAgICAgIH0sCiAgICAgIGZlYXI6ewogICAgICAgICdzZWN1cml0eSc6ICAgICAgICAnU2VjdXJpdHkgaW5jaWRlbnRzIGNyZWF0aW5nIGZlYXIgYWNyb3NzIGNvbW11bml0aWVzJywKICAgICAgICAnYm9yZGVyIGlzc3Vlcyc6ICAgJ0JvcmRlciBzaXR1YXRpb24gZ2VuZXJhdGluZyBmZWFyIG9mIGVzY2FsYXRpb24nLAogICAgICAgICdsYXcgJiBvcmRlcic6ICAgICAnQ3JpbWUgcGF0dGVybnMgY3JlYXRpbmcgcHVibGljIHNhZmV0eSBjb25jZXJucycsCiAgICAgICAgJ2hlYWx0aCc6ICAgICAgICAgICdEaXNlYXNlIHNpZ25hbHMgZ2VuZXJhdGluZyBwdWJsaWMgaGVhbHRoIGFueGlldHknLAogICAgICAgICdlbnZpcm9ubWVudCc6ICAgICAnRW52aXJvbm1lbnRhbCB0aHJlYXRzIGNyZWF0aW5nIGZlYXIgb2YgZGlzYXN0ZXInLAogICAgICAgICdyZWxpZ2lvbic6ICAgICAgICAnQ29tbXVuYWwgdGVuc2lvbnMgY3JlYXRpbmcgZmVhciBvZiB2aW9sZW5jZScsCiAgICAgIH0sCiAgICAgIHByaWRlOnsKICAgICAgICAnbmF0aW9uYWxpc20nOiAgICAgJ05hdGlvbmFsIHNlbnRpbWVudCBhbmQgcGF0cmlvdGljIGRpc2NvdXJzZSBhdCBoaWdoIGludGVuc2l0eScsCiAgICAgICAgJ2VsZWN0aW9ucyc6ICAgICAgICdFbGVjdG9yYWwgbW9tZW50dW0gZ2VuZXJhdGluZyBzdHJvbmcgY29tbXVuaXR5IHByaWRlJywKICAgICAgICAncmVsaWdpb24nOiAgICAgICAgJ0N1bHR1cmFsIGFuZCByZWxpZ2lvdXMgY2VsZWJyYXRpb25zIGRyaXZpbmcgcHJpZGUgc2lnbmFscycsCiAgICAgICAgJ2luZnJhc3RydWN0dXJlJzogICdEZXZlbG9wbWVudCBtaWxlc3RvbmVzIGdlbmVyYXRpbmcgcmVnaW9uYWwgcHJpZGUnLAogICAgICB9LAogICAgfTsKICAgIHZhciBfZW1vQ3R4PScnOwogICAgdmFyIF9zZD1TRFtubV18fHt9OwogICAgdmFyIF9kb21OYXI9X3NkLmRvbWluYW50X25hcnJhdGl2ZXx8Jyc7CiAgICAvLyBVc2Ugc3RhdGUtc3BlY2lmaWMgY29udGV4dCBmaXJzdCwgdGhlbiBuYXJyYXRpdmUtYmFzZWQsIHRoZW4gZ2VuZXJpYwogICAgaWYoX3N0YXRlU3BlY2lmaWMpewogICAgICBfZW1vQ3R4PV9zdGF0ZVNwZWNpZmljOwogICAgfSBlbHNlIGlmKGRvbUVtbyYmX2Vtb1JlYXNvbnNbZG9tRW1vXSl7CiAgICAgIF9lbW9DdHg9X2Vtb1JlYXNvbnNbZG9tRW1vXVtfZG9tTmFyXXx8X2Vtb1JlYXNvbnNbZG9tRW1vXVtPYmplY3Qua2V5cyhfZW1vUmVhc29uc1tkb21FbW9dKVswXV18fCcnOwogICAgfQogICAgLy8gRmFsbGJhY2sgY29udGV4dCBmcm9tIHNpZ25hbCBhcnRpY2xlcwogICAgaWYoIV9lbW9DdHgmJl9zZC5hcnRpY2xlcyYmX3NkLmFydGljbGVzLmxlbmd0aCl7CiAgICAgIHZhciBfdG9wQXJ0PV9zZC5hcnRpY2xlc1swXTsKICAgICAgaWYoX3RvcEFydCYmX3RvcEFydC50eHQpIF9lbW9DdHg9J1NpZ25hbHMgY29uY2VudHJhdGVkIGFyb3VuZDogJytfdG9wQXJ0LnR4dC5zbGljZSgwLDgwKTsKICAgIH0KCiAgICAvLyBSZW9yZGVyIGVMIHNvIGRvbWluYW50IHNob3dzIGZpcnN0CiAgICBlTC5zb3J0KGZ1bmN0aW9uKGEsYil7CiAgICAgIGlmKGFbMF09PT1kb21FbW8pIHJldHVybiAtMTsKICAgICAgaWYoYlswXT09PWRvbUVtbykgcmV0dXJuIDE7CiAgICAgIHJldHVybiBiWzFdLWFbMV07CiAgICB9KTsKICAgIHZhciBkb21QY3Q9TWF0aC5yb3VuZCgoZUxbMF0/ZUxbMF1bMV06MjApKjEwMC90b3QpOwogICAgdmFyIG5hcnIyPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgdG9wTmFyU3RyPW5hcnIyLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiBuLm5hbWU7fSkuam9pbignIGFuZCAnKTsKICAgIHZhciB3aGF0SXQ9e2FueGlldHk6J0EgZGlmZnVzZSB1bmVhc2UgaXMgcnVubmluZyB0aHJvdWdoIHNpZ25hbHMgZnJvbSAnK25tKyh0b3BOYXJTdHI/JywgY29uY2VudHJhdGVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBTaWduYWxzIGF0IHRoaXMgc3RhZ2UgdGVuZCB0byBiZSBsb2NhbGx5IGFic29yYmVkIGJlZm9yZSB3aWRlbmluZy4nOicuJyAgKSxhbmdlcjonRnJ1c3RyYXRpb24gc2lnbmFscyBhcmUgZWxldmF0ZWQgaW4gJytubSsodG9wTmFyU3RyPycsIHBhcnRpY3VsYXJseSBhcm91bmQgJyt0b3BOYXJTdHIrJy4gVGhlIHRvbmUgc3VnZ2VzdHMgcHJlc3N1cmUgYnVpbGRpbmcgcmF0aGVyIHRoYW4gYSBzaW5nbGUgZXZlbnQuJzonLiBUaGUgZW1vdGlvbmFsIHJlZ2lzdGVyIGlzIG5vdGljZWFibHkgdGVuc2UuJyksaG9wZTonQW4gdW51c3VhbGx5IG9wdGltaXN0aWMgc2lnbmFsIHJlZ2lzdGVyIGZyb20gJytubSsodG9wTmFyU3RyPycsIG9yaWVudGVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBXb3J0aCB3YXRjaGluZyDigJQgcG9zaXRpdmUgc2lnbmFscyBhdCB0aGlzIGRlbnNpdHkgYXJlIHJlbGF0aXZlbHkgcmFyZS4nOicuIEEgc2lnbmFsIHdvcnRoIG1vbml0b3JpbmcuJykscHJpZGU6J1N0cm9uZyBpZGVudGl0eSBzaWduYWxzIGluICcrbm0rKHRvcE5hclN0cj8nLCBjZW50cmVkIGFyb3VuZCAnK3RvcE5hclN0cisnLiBSZWdpb25hbGx5IGNvbmNlbnRyYXRlZCBhbmQgZW1vdGlvbmFsbHkgZGVuc2UuJzonLiBMb2NhbGx5IGNvbmNlbnRyYXRlZCwgZW1vdGlvbmFsbHkgc3Ryb25nLicpLGZlYXI6J0FwcHJlaGVuc2lvbiBzaWduYWxzIGluICcrbm0rKHRvcE5hclN0cj8nLCBhcm91bmQgJyt0b3BOYXJTdHIrJy4gVGhlc2UgdGVuZCB0byBpbnRlbnNpZnkgYmVmb3JlIGFjaGlldmluZyB3aWRlciB2aXNpYmlsaXR5Lic6Jy4gVGhlIHJlZ2lzdGVyIGNhcnJpZXMgYW4gZWRnZSB0aGF0IHRlbmRzIHRvIHByZWNlZGUgbGFyZ2VyIGN5Y2xlcy4nKX07CiAgICB2YXIgY3VtQT0tTWF0aC5QSS8yLGN4PTM4LGN5PTM4LFI9MzMscmk9MjA7CiAgICB2YXIgYXJjcz1lTC5tYXAoZnVuY3Rpb24oa3YpewogICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLGZyPXYvdG90LGExPWN1bUEsYTI9Y3VtQStmcipNYXRoLlBJKjI7Y3VtQT1hMjsKICAgICAgdmFyIGxnPShhMi1hMSk+TWF0aC5QST8xOjA7CiAgICAgIHZhciB4MT1jeCtNYXRoLmNvcyhhMSkqUix5MT1jeStNYXRoLnNpbihhMSkqUix4Mj1jeCtNYXRoLmNvcyhhMikqUix5Mj1jeStNYXRoLnNpbihhMikqUjsKICAgICAgdmFyIHgzPWN4K01hdGguY29zKGEyKSpyaSx5Mz1jeStNYXRoLnNpbihhMikqcmkseDQ9Y3grTWF0aC5jb3MoYTEpKnJpLHk0PWN5K01hdGguc2luKGExKSpyaTsKICAgICAgcmV0dXJuICc8cGF0aCBkPSJNJyt4MS50b0ZpeGVkKDEpKycsJyt5MS50b0ZpeGVkKDEpKycgQScrUisnLCcrUisnIDAgJytsZysnIDEgJyt4Mi50b0ZpeGVkKDEpKycsJyt5Mi50b0ZpeGVkKDEpKycgTCcreDMudG9GaXhlZCgxKSsnLCcreTMudG9GaXhlZCgxKSsnIEEnK3JpKycsJytyaSsnIDAgJytsZysnIDAgJyt4NC50b0ZpeGVkKDEpKycsJyt5NC50b0ZpeGVkKDEpKycgWiIgZmlsbD0iJytwYWxba10rJyIgb3BhY2l0eT0iMC45Ii8+JzsKICAgIH0pLmpvaW4oJycpOwogICAgdmFyIGVkZXNjPXthbnhpZXR5OidEaWZmdXNlIHVuZWFzZSwgd29ycnkgc2lnbmFscycsYW5nZXI6J0ZydXN0cmF0aW9uLCBwcmVzc3VyZSBzaWduYWxzJyxob3BlOidPcHRpbWlzbSwgZm9yd2FyZCBtb21lbnR1bScscHJpZGU6J0lkZW50aXR5LCByZWdpb25hbCBhc3NlcnRpb24nLGZlYXI6J0FwcHJlaGVuc2lvbiwgdGhyZWF0IHBlcmNlcHRpb24nfTsKICAgIGJvZHkrPQogICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjlweDtsZXR0ZXItc3BhY2luZzowLjA4ZW07Y29sb3I6dmFyKC0tZmFpbnQpO3BhZGRpbmc6OHB4IDAgNHB4IDA7bGluZS1oZWlnaHQ6MS42Ij4nKwogICAgICAnVGhlIGVtb3Rpb25hbCByZWdpc3RlciBvZiBzaWduYWxzIGZyb20gJytubSsnIOKAlCB3aGF0IHRvbmUgcnVucyB0aHJvdWdoIHRoZSBkaXNjb3Vyc2UgYW5kIGhvdyBjb25jZW50cmF0ZWQgaXQgaXMuJysKICAgICc8L2Rpdj4nKwogICAgKCFoYXNFbW9zPyc8ZGl2IHN0eWxlPSJwYWRkaW5nOjZweCAxMXB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjAyKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7bWFyZ2luLWJvdHRvbToxMHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo5cHg7Y29sb3I6dmFyKC0tZmFpbnQpIj5Fc3RpbWF0ZWQgZnJvbSBzaWduYWwgZGlyZWN0aW9uIOKAlCBsaW1pdGVkIGRpcmVjdCBlbW90aW9uIGRhdGEuPC9kaXY+JzonJykrCiAgICAgICc8ZGl2IHN0eWxlPSJwYWRkaW5nOjE0cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6JytwYWxbZG9tRW1vXSsnMTQ7Ym9yZGVyOjFweCBzb2xpZCAnK3BhbFtkb21FbW9dKyczMzttYXJnaW4tYm90dG9tOjEycHg7Ij4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OHB4O2xldHRlci1zcGFjaW5nOjAuMmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjonK3BhbFtkb21FbW9dKyc7bWFyZ2luLWJvdHRvbTo2cHgiPkRvbWluYW50IGVtb3Rpb248L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjI2cHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluaykiPicrZG9tRW1vLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2RvbUVtby5zbGljZSgxKSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjRweCI+Jytkb21QY3QrJyUgwrcgJytubSsnPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tdG9wOjhweDtsaW5lLWhlaWdodDoxLjU7Zm9udC1zdHlsZTppdGFsaWMiPicrKF9lbW9DdHh8fHdoYXRJdFtkb21FbW9dfHwnJykrJzwvZGl2PicrCiAgICAgICc8L2Rpdj4nKwogICAgICAnPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5FbW90aW9uYWwgYnJlYWtkb3duPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTZweDsiPicrCiAgICAgICAgICAnPHN2ZyB2aWV3Qm94PSIwIDAgNzYgNzYiIHN0eWxlPSJ3aWR0aDo3MnB4O2hlaWdodDo3MnB4O2ZsZXgtc2hyaW5rOjAiPicrYXJjcysnPC9zdmc+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6NXB4OyI+JysKICAgICAgICAgICAgZUwubWFwKGZ1bmN0aW9uKGt2KXsKICAgICAgICAgICAgICB2YXIgaz1rdlswXSx2PWt2WzFdLHBjdD1NYXRoLnJvdW5kKHYqMTAwL3RvdCk7CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2FsaWduLWl0ZW1zOmNlbnRlcjttYXJnaW4tYm90dG9tOjJweDsiPicrCiAgICAgICAgICAgICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo2cHg7Ij48c3BhbiBzdHlsZT0id2lkdGg6N3B4O2hlaWdodDo3cHg7Ym9yZGVyLXJhZGl1czoycHg7YmFja2dyb3VuZDonK3BhbFtrXSsnO2Rpc3BsYXk6aW5saW5lLWJsb2NrIj48L3NwYW4+JysKICAgICAgICAgICAgICAgICAgJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTEuNXB4O2NvbG9yOicrKGs9PT1kb21FbW8/J3ZhcigtLWluayknOid2YXIoLS1kaW0pJykrJyI+JytrLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2suc2xpY2UoMSkrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAgICAgICAgICAgICAnPHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0taW5rKSI+JytwY3QrJyU8L3NwYW4+JysKICAgICAgICAgICAgICAgICc8L2Rpdj4nKwogICAgICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImhlaWdodDoycHg7YmFja2dyb3VuZDpyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpO2JvcmRlci1yYWRpdXM6MXB4OyI+PGRpdiBzdHlsZT0iaGVpZ2h0OjEwMCU7d2lkdGg6JytwY3QrJyU7YmFja2dyb3VuZDonK3BhbFtrXSsnO29wYWNpdHk6MC43O2JvcmRlci1yYWRpdXM6MXB4Ij48L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICAgKGs9PT1kb21FbW8/JzxkaXYgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZTo4LjVweDtjb2xvcjp2YXIoLS1mYWludCk7bWFyZ2luLXRvcDoycHg7Ij4nK2VkZXNjW2tdKyc8L2Rpdj4nOicnKSsKICAgICAgICAgICAgICAnPC9kaXY+JzsKICAgICAgICAgICAgfSkuam9pbignJykrCiAgICAgICAgICAnPC9kaXY+JysKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJzcC1zZWN0aW9uIj48ZGl2IGNsYXNzPSJzcC1zZWMtdGl0bGUiPlNpZ25hbCBoZWFkbGluZXM8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo0cHg7Ij4nKwogICAgICAgICAgKChkLmFydGljbGVzJiZkLmFydGljbGVzLmxlbmd0aCk/CiAgICAgICAgICAgIGQuYXJ0aWNsZXMuc2xpY2UoMCw1KS5tYXAoZnVuY3Rpb24oYSl7CiAgICAgICAgICAgICAgdmFyIGVDb2xvcj17YW54aWV0eTonIzg4NDRjYycsYW5nZXI6JyNkZDIyNDQnLGhvcGU6JyMzM2NjNjYnLHByaWRlOicjMzNhYWNjJyxmZWFyOicjY2M4ODMzJ307CiAgICAgICAgICAgICAgcmV0dXJuICc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6ZmxleC1zdGFydDtnYXA6NnB4O3BhZGRpbmc6NnB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjAzKTsiPicrCiAgICAgICAgICAgICAgICAoYS5lbW90aW9uPyc8c3BhbiBzdHlsZT0id2lkdGg6NXB4O2hlaWdodDo1cHg7Ym9yZGVyLXJhZGl1czo1MCU7YmFja2dyb3VuZDonK2VDb2xvclthLmVtb3Rpb25dKyc7ZGlzcGxheTppbmxpbmUtYmxvY2s7bWFyZ2luLXRvcDo1cHg7ZmxleC1zaHJpbms6MCI+PC9zcGFuPic6JycpKwogICAgICAgICAgICAgICAgJzxkaXY+PGRpdiBzdHlsZT0iZm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtsaW5lLWhlaWdodDoxLjQiPicrKGEudHh0fHxhLnRpdGxlfHwnJykrJzwvZGl2PicrCiAgICAgICAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjguNXB4O2NvbG9yOnZhcigtLWZhaW50KTttYXJnaW4tdG9wOjJweCI+JysoYS5zcmN8fCcnKSsoYS5lbW90aW9uPycgwrcgJythLmVtb3Rpb246JycpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICAgICAgICc8L2Rpdj4nOwogICAgICAgICAgICB9KS5qb2luKCcnKToKICAgICAgICAgICAgJzxkaXYgc3R5bGU9ImNvbG9yOnZhcigtLWZhaW50KTtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTBweDtwYWRkaW5nOjRweCAwIj5ObyBzaWduYWxzIHlldC48L2Rpdj4nKSsKICAgICAgICAnPC9kaXY+JysKICAgICAgJzwvZGl2Pic7CgogIH0gZWxzZSB7CiAgICB2YXIgdmVsPWQudmVsb2NpdHl8fDA7CiAgICB2YXIgdmVsRGlyPXZlbD4wLjE1PydSaXNpbmcgZmFzdCc6dmVsPjAuMDU/J1Jpc2luZyc6dmVsPC0wLjE/J0Nvb2xpbmcgZmFzdCc6dmVsPC0wLjAyPydDb29saW5nJzonU3RhYmxlJzsKICAgIHZhciB2ZWxDb2w9dmVsPjAuMDU/JyNlMDVhMjgnOnZlbDwtMC4wMj8nIzNiYjhkOCc6JyM1NTY2NzcnOwogICAgdmFyIG5hcnIzPWQubmFycmF0aXZlc3x8W107CiAgICB2YXIgcmlzaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J3VwJzt9KTsKICAgIHZhciBmYWxsaW5nTmFycz1uYXJyMy5maWx0ZXIoZnVuY3Rpb24obil7cmV0dXJuIG4uZGlyPT09J2Rvd24nO30pOwogICAgdmFyIHRvcE5hcj1uYXJyMy5sZW5ndGg/bmFycjNbMF0ubmFtZTonJzsKICAgIHZhciBkb21FbW9Nb209ZC5kb21pbmFudF9lbW90aW9ufHwnJzsKICAgIHZhciBzaWdDb3VudD1kLnNpZ25hbF9jb3VudHx8MDsKICAgIHZhciBzcmNDb3VudD1kLnNvdXJjZV9jb3VudHx8MTsKICAgIHZhciB0b3BBcnRzPShkLmFydGljbGVzfHxbXSkuc2xpY2UoMCwyKS5tYXAoZnVuY3Rpb24oYSl7cmV0dXJuIGEudHh0fHxhLnRpdGxlfHwnJ30pLmZpbHRlcihCb29sZWFuKTsKCiAgICAvLyBCdWlsZCBhIHJpY2gsIHN0YXRlLXNwZWNpZmljIGludGVycHJldGF0aW9uIG9mIHdoYXQgdGhlIG1vbWVudHVtIG1lYW5zCiAgICBmdW5jdGlvbiBidWlsZE1vbWVudHVtU3RvcnkoKXsKICAgICAgdmFyIGxpbmVzPVtdOwoKICAgICAgLy8gTGluZSAxOiBXaGF0IGlzIGRyaXZpbmcgdGhlIG1vdmVtZW50IOKAlCB0aGUgV0hZCiAgICAgIGlmKHZlbD4wLjA1KXsKICAgICAgICBpZihyaXNpbmdOYXJzLmxlbmd0aCl7CiAgICAgICAgICB2YXIgbmFyTmFtZXM9cmlzaW5nTmFycy5zbGljZSgwLDIpLm1hcChmdW5jdGlvbihuKXtyZXR1cm4gJzxlbT4nK24ubmFtZSsnPC9lbT4nO30pLmpvaW4oJyBhbmQgJyk7CiAgICAgICAgICBsaW5lcy5wdXNoKG5tKycgaXMgYXR0cmFjdGluZyBhY2NlbGVyYXRpbmcgYXR0ZW50aW9uIGFyb3VuZCAnK25hck5hbWVzKycuJysKICAgICAgICAgICAgKHNyY0NvdW50PjM/JyBTaWduYWxzIGFyZSBhcnJpdmluZyBmcm9tICcrc3JjQ291bnQrJyBkaXN0aW5jdCBzb3VyY2UgdHlwZXMg4oCUIHJlZ2lvbmFsIHByZXNzLCBwdWJsaWMgZGlzY291cnNlLCBhbmQgYnJvYWRlciBtZWRpYSDigJQgc3VnZ2VzdGluZyB0aGlzIGlzIG5vdCBhIGxvY2FsaXNlZCBmbGFyZSBidXQgYSB3aWRlbmluZyBzdG9yeS4nOgogICAgICAgICAgICAnIENvdmVyYWdlIGlzIHN0aWxsIGNvbnNvbGlkYXRpbmcgYWNyb3NzIHNvdXJjZXMuJykpOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICBsaW5lcy5wdXNoKCdTaWduYWwgdm9sdW1lIGluICcrbm0rJyBpcyBjbGltYmluZyDigJQgJytzaWdDb3VudCsnIHNpZ25hbHMgdHJhY2tlZCBpbiB0aGUgbGFzdCA0OCBob3VycywgdXAgc2lnbmlmaWNhbnRseSBmcm9tIHRoZSBwcmV2aW91cyB3aW5kb3cuJyk7CiAgICAgICAgfQogICAgICB9IGVsc2UgaWYodmVsPC0wLjA1KXsKICAgICAgICBpZihmYWxsaW5nTmFycy5sZW5ndGgpewogICAgICAgICAgdmFyIG5hck5hbWVzPWZhbGxpbmdOYXJzLnNsaWNlKDAsMikubWFwKGZ1bmN0aW9uKG4pe3JldHVybiAnPGVtPicrbi5uYW1lKyc8L2VtPic7fSkuam9pbignIGFuZCAnKTsKICAgICAgICAgIGxpbmVzLnB1c2goJ0NvdmVyYWdlIG9mICcrbmFyTmFtZXMrJyBpbiAnK25tKycgaXMgY29udHJhY3RpbmcuIFRoZSBkaXNjb3Vyc2UgY3ljbGUgYXJvdW5kIHRoaXMgbmFycmF0aXZlIGFwcGVhcnMgdG8gaGF2ZSBwYXNzZWQgaXRzIHBlYWsg4oCUIHNpZ25hbCBpbnRlbnNpdHkgaXMgZGVjbGluaW5nIGFuZCBzb3VyY2VzIGFyZSBtb3Zpbmcgb24uJyk7CiAgICAgICAgfSBlbHNlIHsKICAgICAgICAgIGxpbmVzLnB1c2gobm0rJyBpcyBlbnRlcmluZyBhIHF1aWV0ZXIgcGhhc2UuIEFmdGVyIHJlY2VudCBhY3Rpdml0eSwgc2lnbmFsIHZvbHVtZSBpcyByZXRyZWF0aW5nIOKAlCBuYXRpb25hbCBhdHRlbnRpb24gaXMgbGlrZWx5IHNoaWZ0aW5nIHRvIG90aGVyIHN0b3JpZXMuJyk7CiAgICAgICAgfQogICAgICB9IGVsc2UgewogICAgICAgIGxpbmVzLnB1c2gobm0rJyBpcyBob2xkaW5nIGEgc3RlYWR5IHNpZ25hbCBiYXNlbGluZS4gJytzaWdDb3VudCsnIHNpZ25hbHMgdHJhY2tlZCDigJQgY29uc2lzdGVudCBwcmVzZW5jZSBpbiBuYXRpb25hbCBkaXNjb3Vyc2Ugd2l0aG91dCBhIGRvbWluYW50IGFjY2VsZXJhdGlvbiBldmVudC4nKTsKICAgICAgfQoKICAgICAgLy8gTGluZSAyOiBXaGF0IHRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgdGVsbHMgdXMgYWJvdXQgdGhlIFdIWQogICAgICBpZihkb21FbW9Nb20mJnZlbD4wLjAyKXsKICAgICAgICB2YXIgZW1vQ3R4PXsKICAgICAgICAgIGFuZ2VyOiAnVGhlIGRvbWluYW50IGVtb3Rpb25hbCByZWdpc3RlciBpcyBhbmdlciDigJQgdGhlIG1vbWVudHVtIGhlcmUgaXMgZHJpdmVuIGJ5IHB1YmxpYyBmcnVzdHJhdGlvbiwgbm90IHJvdXRpbmUgY292ZXJhZ2UuIFRoaXMgcGF0dGVybiB0eXBpY2FsbHkgaW5kaWNhdGVzIGEgZ292ZXJuYW5jZSBvciBsYXctYW5kLW9yZGVyIHRyaWdnZXIgdGhhdCBpcyBnZW5lcmF0aW5nIHJlYWN0aXZlIGRpc2NvdXJzZS4nLAogICAgICAgICAgYW54aWV0eTogJ1RoZSB1bmRlcmx5aW5nIHNpZ25hbCB0b25lIGlzIGFueGlvdXMg4oCUIG1vbWVudHVtIGlzIGJ1aWxkaW5nIGFyb3VuZCB1bmNlcnRhaW50eSByYXRoZXIgdGhhbiBhIHNpbmdsZSBldmVudC4gRWNvbm9taWMgcHJlc3N1cmUsIHBvbGljeSBhbWJpZ3VpdHksIG9yIGFuIHVucmVzb2x2ZWQgY3Jpc2lzIGlzIGxpa2VseSBzdXN0YWluaW5nIHRoZSBhdHRlbnRpb24uJywKICAgICAgICAgIGhvcGU6ICdUaGUgc2lnbmFsIHRvbmUgc2tld3Mgb3B0aW1pc3RpYyDigJQgbW9tZW50dW0gaXMgYmVpbmcgZHJpdmVuIGJ5IGEgZGV2ZWxvcG1lbnQsIGFubm91bmNlbWVudCwgb3IgaW5pdGlhdGl2ZSB0aGF0IGlzIGdlbmVyYXRpbmcgcG9zaXRpdmUgcmVnaW9uYWwgYXR0ZW50aW9uLicsCiAgICAgICAgICBmZWFyOiAnRmVhciBpcyB0aGUgZG9taW5hbnQgc2lnbmFsIHJlZ2lzdGVyIOKAlCBtb21lbnR1bSBpcyBidWlsZGluZyBhcm91bmQgYSBzZWN1cml0eSwgc2FmZXR5LCBvciB0aHJlYXQtcmVsYXRlZCBzdG9yeS4gVGhlIGFjY2VsZXJhdGlvbiBoZXJlIHdhcnJhbnRzIGNsb3NlIHdhdGNoaW5nLicsCiAgICAgICAgICBwcmlkZTogJ1ByaWRlIHNpZ25hbHMgYXJlIGRyaXZpbmcgdGhlIG1vbWVudHVtIOKAlCBhbiBhY2hpZXZlbWVudCwgcmVjb2duaXRpb24sIG9yIGN1bHR1cmFsIGV2ZW50IGlzIGdlbmVyYXRpbmcgc3VzdGFpbmVkIHBvc2l0aXZlIGF0dGVudGlvbiBpbiAnK25tKycuJwogICAgICAgIH07CiAgICAgICAgaWYoZW1vQ3R4W2RvbUVtb01vbV0pIGxpbmVzLnB1c2goZW1vQ3R4W2RvbUVtb01vbV0pOwogICAgICB9CgogICAgICAvLyBMaW5lIDM6IFdoYXQgdG8gd2F0Y2gg4oCUIGZvcndhcmQtbG9va2luZyBpbnRlcnByZXRhdGlvbgogICAgICBpZih2ZWw+MC4xNSl7CiAgICAgICAgbGluZXMucHVzaCgnQXQgdGhpcyBhY2NlbGVyYXRpb24gcmF0ZSwgJytubSsnIGlzIGxpa2VseSB0byBiZWNvbWUgYSBuYXRpb25hbCBhdHRlbnRpb24gZm9jYWwgcG9pbnQgd2l0aGluIHRoZSBuZXh0IDI04oCTNDggaG91cnMuIFN0YXRlcyByZWFjaGluZyB0aGlzIHZlbG9jaXR5IHRocmVzaG9sZCB0eXBpY2FsbHkgYXR0cmFjdCBtYWluc3RyZWFtIG1lZGlhIGFtcGxpZmljYXRpb24gc2hvcnRseSBhZnRlci4nKTsKICAgICAgfSBlbHNlIGlmKHZlbD4wLjA1KXsKICAgICAgICBsaW5lcy5wdXNoKCdJZiBzaWduYWwgbW9tZW50dW0gaG9sZHMsIHRoaXMgc3RvcnkgaGFzIHRoZSB0cmFqZWN0b3J5IHRvIGJyZWFrIGludG8gYnJvYWRlciBuYXRpb25hbCBjb252ZXJzYXRpb24uIE1vbml0b3JpbmcgdGhlIG5leHQgaW5nZXN0IGN5Y2xlIHdpbGwgaW5kaWNhdGUgd2hldGhlciB0aGUgYWNjZWxlcmF0aW9uIGlzIHN1c3RhaW5lZCBvciBwbGF0ZWF1aW5nLicpOwogICAgICB9IGVsc2UgaWYodmVsPC0wLjEpewogICAgICAgIGxpbmVzLnB1c2goJ1RoZSBhdHRlbnRpb24gY3ljbGUgZm9yICcrbm0rJyBhcHBlYXJzIHRvIGJlIGNvbXBsZXRpbmcuIFRoaXMgaXMgdHlwaWNhbCBwb3N0LXBlYWsgYmVoYXZpb3VyIOKAlCB1bmxlc3MgYSBuZXcgdHJpZ2dlciBlbWVyZ2VzLCBzaWduYWwgdm9sdW1lIHdpbGwgbGlrZWx5IHN0YWJpbGlzZSBhdCBiYXNlbGluZSB3aXRoaW4gdGhlIG5leHQgY3ljbGUuJyk7CiAgICAgIH0gZWxzZSBpZih2ZWw8LTAuMDIpewogICAgICAgIGxpbmVzLnB1c2goJ01vbWVudHVtIGlzIHJldHJlYXRpbmcsIGJ1dCBub3QgY29sbGFwc2VkLiBBIHNlY29uZGFyeSB0cmlnZ2VyIGNvdWxkIHJlLWlnbml0ZSBjb3ZlcmFnZSDigJQgd29ydGggd2F0Y2hpbmcgZm9yIGZvbGxvdy11cCBkZXZlbG9wbWVudHMuJyk7CiAgICAgIH0KCiAgICAgIHJldHVybiBsaW5lcy5qb2luKCcgJyk7CiAgICB9CiAgICB2YXIgY3R4PWJ1aWxkTW9tZW50dW1TdG9yeSgpOwogICAgYm9keSs9CiAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6OXB4O2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS1mYWludCk7cGFkZGluZzo4cHggMCA0cHggMDtsaW5lLWhlaWdodDoxLjYiPicrCiAgICAgICdTaWduYWwgdmVsb2NpdHkgZm9yICcrbm0rJyDigJQgd2hldGhlciBhdHRlbnRpb24gaXMgYnVpbGRpbmcsIGhvbGRpbmcsIG9yIGJlZ2lubmluZyB0byByZXRyZWF0IGZyb20gdGhlIGN1cnJlbnQgY3ljbGUuJysKICAgICc8L2Rpdj4nKwogICAgJzxkaXYgc3R5bGU9InBhZGRpbmc6MTRweDtib3JkZXItcmFkaXVzOjEwcHg7YmFja2dyb3VuZDonK3ZlbENvbCsnMTQ7Ym9yZGVyOjFweCBzb2xpZCAnK3ZlbENvbCsnMzM7bWFyZ2luLWJvdHRvbToxMnB4OyI+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjhweDtsZXR0ZXItc3BhY2luZzowLjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6Jyt2ZWxDb2wrJzttYXJnaW4tYm90dG9tOjZweCI+U2lnbmFsIG1vbWVudHVtPC9kaXY+JysKICAgICAgICAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lO2dhcDoxMHB4O21hcmdpbi1ib3R0b206OHB4OyI+JysKICAgICAgICAgICc8ZGl2IHN0eWxlPSJmb250LWZhbWlseTp2YXIoLS1zZXJpZik7Zm9udC1zaXplOjMycHg7Zm9udC13ZWlnaHQ6MzAwO2NvbG9yOnZhcigtLWluaykiPicrKHZlbD4wPycrJzonJykrdmVsLnRvRml4ZWQoMykrJzwvZGl2PicrCiAgICAgICAgICAnPGRpdiBzdHlsZT0iZm9udC1zaXplOjE0cHg7Y29sb3I6Jyt2ZWxDb2wrJztmb250LXdlaWdodDo1MDAiPicrdmVsRGlyKyc8L2Rpdj4nKwogICAgICAgICc8L2Rpdj4nKwogICAgICAgICc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1kaW0pO2xpbmUtaGVpZ2h0OjEuNjU7bWFyZ2luLXRvcDo2cHgiPicrY3R4Kyc8L2Rpdj4nKwogICAgICAnPC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9InNjb3JlLXN0cmlwIj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+VmVsb2NpdHk8L2Rpdj48ZGl2IGNsYXNzPSJzcy12YWwiIHN0eWxlPSJmb250LXNpemU6MThweDtjb2xvcjonK3ZlbENvbCsnIj4nKyh2ZWw+MD8nKyc6JycpK3ZlbC50b0ZpeGVkKDMpKyc8L2Rpdj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1kaXZpZGVyIj48L2Rpdj4nKwogICAgICAgICc8ZGl2IGNsYXNzPSJzcy1pdGVtIj48ZGl2IGNsYXNzPSJzcy1sYWJlbCI+MjRoIM60PC9kaXY+PGRpdiBjbGFzcz0ic3MtZGVsdGEgJysoZC5kZWx0YT49MD8ndXAnOidkbicpKyciPicrKGQuZGVsdGE+PTA/JysnOicnKSsoZC5kZWx0YXx8MCkrJzwvZGl2PjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWRpdmlkZXIiPjwvZGl2PicrCiAgICAgICAgJzxkaXYgY2xhc3M9InNzLWl0ZW0iPjxkaXYgY2xhc3M9InNzLWxhYmVsIj5BdHRlbnRpb248L2Rpdj48ZGl2IGNsYXNzPSJzcy1uYXIiPicrKGQuYXR0ZW50aW9ufHwwKSsnPC9kaXY+PC9kaXY+JysKICAgICAgJzwvZGl2PicrCiAgICAgIChyaXNpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5BY2NlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIHJpc2luZ05hcnMubWFwKGZ1bmN0aW9uKHIpe3JldHVybiAnPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO3BhZGRpbmc6N3B4IDEwcHg7bWFyZ2luLWJvdHRvbTo0cHg7Ym9yZGVyLXJhZGl1czo1cHg7YmFja2dyb3VuZDpyZ2JhKDIyNCw5MCw0MCwwLjA1KTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjI0LDkwLDQwLDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiNlMDVhMjgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKSsKICAgICAgKGZhbGxpbmdOYXJzLmxlbmd0aD8nPGRpdiBjbGFzcz0ic3Atc2VjdGlvbiI+PGRpdiBjbGFzcz0ic3Atc2VjLXRpdGxlIj5EZWNlbGVyYXRpbmc8L2Rpdj4nKwogICAgICAgIGZhbGxpbmdOYXJzLm1hcChmdW5jdGlvbihyKXtyZXR1cm4gJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjtwYWRkaW5nOjdweCAxMHB4O21hcmdpbi1ib3R0b206NHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6cmdiYSg1OSwxODQsMjE2LDAuMDUpO2JvcmRlcjoxcHggc29saWQgcmdiYSg1OSwxODQsMjE2LDAuMTIpIj48c3BhbiBzdHlsZT0iZm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0taW5rKSI+JytyLm5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkrci5uYW1lLnNsaWNlKDEpKyc8L3NwYW4+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiMzYmI4ZDgiPicrci52YWwudG9GaXhlZCgxKSsnJTwvc3Bhbj48L2Rpdj4nO30pLmpvaW4oJycpKyc8L2Rpdj4nOicnKTsKICB9CgogIHBhbmVsLmlubmVySFRNTD1oZWFkZXIrYm9keTsKfQoKCmZ1bmN0aW9uIHRvZ2dsZUZhdihubSl7CiAgaWYoRkFWUy5oYXMobm0pKSBGQVZTLmRlbGV0ZShubSk7ZWxzZSBGQVZTLmFkZChubSk7CiAgcmVuZGVyUGFuZWwoU0VMKTtyZW5kZXJGYXZzKCk7Cn0KZnVuY3Rpb24gcmVuZGVyRmF2cygpewogIHZhciByb3c9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Zhdi1yb3cnKTsKICBpZighRkFWUy5zaXplKXtyb3cuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJmYXZzLWVtcHR5Ij5ObyBzdGF0ZXMgdHJhY2tlZC4gQm9va21hcmsgYW55IHN0YXRlIHBhbmVsIHRvIGZvbGxvdyBpdHMgbmFycmF0aXZlIGV2b2x1dGlvbi48L2Rpdj4nO3JldHVybjt9CiAgcm93LmlubmVySFRNTD1BcnJheS5mcm9tKEZBVlMpLm1hcChmdW5jdGlvbihubSl7CiAgICB2YXIgZD1nKG5tKSxkUz1kLmRlbHRhPj0wPycrJzonJyxkQz1kLmRlbHRhPj0wPycjZTA1YTI4JzonIzNiYjhkOCc7CiAgICB2YXIgdG9wPWQubmFycmF0aXZlcyYmZC5uYXJyYXRpdmVzWzBdP2QubmFycmF0aXZlc1swXS5uYW1lOifigJQnOwogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJmYXYtY2FyZCIgb25jbGljaz0ic2VsZWN0XyhcJycrbm0rJ1wnKSI+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLWhlYWQiPjxzcGFuIGNsYXNzPSJmYy1uYW1lIj4nK25tKyc8L3NwYW4+PHNwYW4gY2xhc3M9ImZjLXNjIj4nK2QuYXR0ZW50aW9uKyc8L3NwYW4+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9ImZjLXJvdyI+PHNwYW4+TmFycmF0aXZlPC9zcGFuPjxzcGFuIGNsYXNzPSJ2Ij4nK3RvcCsnPC9zcGFuPjwvZGl2PicrCiAgICAgICc8ZGl2IGNsYXNzPSJmYy1yb3ciPjxzcGFuPjI0aDwvc3Bhbj48c3BhbiBjbGFzcz0idiIgc3R5bGU9ImNvbG9yOicrZEMrJyI+JytkUytkLmRlbHRhKyc8L3NwYW4+PC9kaXY+JysKICAgICc8L2Rpdj4nOwogIH0pLmpvaW4oJycpOwp9Cgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24oYyl7CiAgYy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsZnVuY3Rpb24oKXsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5sdGFiJykuZm9yRWFjaChmdW5jdGlvbih4KXt4LmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpO30pOwogICAgYy5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtsYXllcj1jLmRhdGFzZXQubGF5ZXI7YXBwbHlMYXllcigpOwogIH0pOwp9KTsKCmZ1bmN0aW9uIHVwZGF0ZUNsb2NrKCl7CiAgdmFyIG5vdz1uZXcgRGF0ZSgpLGlzdD1uZXcgRGF0ZShub3cuZ2V0VGltZSgpK25vdy5nZXRUaW1lem9uZU9mZnNldCgpKjYwMDAwKzE5ODAwMDAwKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2xvY2snKS50ZXh0Q29udGVudD1TdHJpbmcoaXN0LmdldEhvdXJzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRNaW51dGVzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnOicrU3RyaW5nKGlzdC5nZXRTZWNvbmRzKCkpLnBhZFN0YXJ0KDIsJzAnKSsnIElTVCc7Cn0Kc2V0SW50ZXJ2YWwodXBkYXRlQ2xvY2ssMTAwMCk7dXBkYXRlQ2xvY2soKTsKCmZ1bmN0aW9uIG5vcm1WKHYpewogIGlmKCF2KSByZXR1cm4gMDsKICB2YXIgYT1NYXRoLmFicyh2KTsKICBpZihhPjEpIHY9di9NYXRoLm1heChhLDUwKTsKICByZXR1cm4gTWF0aC5tYXgoLTEsTWF0aC5taW4oMSx2KSk7Cn0KCmZ1bmN0aW9uIGJ1aWxkV0lSU2lnbmFscygpewogIHZhciBwYWw9e2FueGlldHk6JyM4ODQ0Y2MnLGFuZ2VyOicjZGQyMjQ0Jyxob3BlOicjMzNjYzY2JyxwcmlkZTonIzMzYWFjYycsZmVhcjonI2NjODgzMyd9OwogIHZhciBlbnRyaWVzPU9iamVjdC5lbnRyaWVzKExJVkUpLmZpbHRlcihmdW5jdGlvbihrdil7cmV0dXJuKGt2WzFdLmF0dGVudGlvbnx8MCk+MDt9KS5zb3J0KGZ1bmN0aW9uKGEsYil7cmV0dXJuKGJbMV0uYXR0ZW50aW9ufHwwKS0oYVsxXS5hdHRlbnRpb258fDApO30pOwogIGlmKCFlbnRyaWVzLmxlbmd0aCkgcmV0dXJuOwogIHZhciBzaWduYWxzPVtdLHVzZWROPVtdLHVzZWRTPVtdOwogIGZ1bmN0aW9uIHVzZWQobixzKXtyZXR1cm4gdXNlZE4uaW5kZXhPZihuKT49MHx8dXNlZFMuaW5kZXhPZihzKT49MDt9CiAgZnVuY3Rpb24gdXNlKG4scyl7aWYobil1c2VkTi5wdXNoKG4pO2lmKHMpdXNlZFMucHVzaChzKTt9CgogIHZhciBlbW9PYnM9ewogICAgYW5nZXI6ewogICAgICBnb3Zlcm5hbmNlOidHb3Zlcm5hbmNlIGZydXN0cmF0aW9uIGlzIG5vdCBkaXNwZXJzaW5nIOKAlCBpdCBpcyBjb25zb2xpZGF0aW5nLiBJbiByZWdpb25zIHdoZXJlIHRoaXMgc2lnbmFsIHBlcnNpc3RzLCBwdWJsaWMgdHJ1c3QgaW4gaW5zdGl0dXRpb25hbCBtZWNoYW5pc21zIGlzIHF1aWV0bHkgZXJvZGluZy4nLAogICAgICAnbGF3ICYgb3JkZXInOidUaGUgYW5nZXIgaGVyZSBpcyBub3QgcmVhY3RpdmUgbm9pc2UuIEl0IGlzIHRoZSBhY2N1bXVsYXRpb24gb2YgdW5yZXNvbHZlZCBleHBlY3RhdGlvbnMgYXJvdW5kIHNhZmV0eSBhbmQgYWNjb3VudGFiaWxpdHkg4oCUIGEgc2lnbmFsIHRoYXQgdGVuZHMgdG8gZGVlcGVuIGJlZm9yZSBpdCBzdXJmYWNlcyBuYXRpb25hbGx5LicsCiAgICAgIGVsZWN0aW9uczonRWxlY3RvcmFsIHRlbnNpb24gaXMgZ2VuZXJhdGluZyBzb21ldGhpbmcgbW9yZSBkdXJhYmxlIHRoYW4gb3V0cmFnZSDigJQgYSBzdXN0YWluZWQgcXVlc3Rpb25pbmcgb2YgcHJvY2Vzcy4gVGhpcyBraW5kIG9mIHNpZ25hbCBmb3JtcyBxdWlldGx5IGJlZm9yZSBpdCBiZWNvbWVzIHZpc2libGUuJywKICAgICAgY29ycnVwdGlvbjonV2hhdCBpcyByZWdpc3RlcmluZyBoZXJlIGlzIG5vdCBqdXN0IHNjYW5kYWwgZmF0aWd1ZS4gSXQgaXMgYSBzbG93IHdpdGhkcmF3YWwgb2YgaW5zdGl0dXRpb25hbCBjcmVkaWJpbGl0eSDigJQgaGFyZGVyIHRvIHJldmVyc2UgdGhhbiBhbnkgc2luZ2xlIGNvbnRyb3ZlcnN5LicsCiAgICAgIHByb3Rlc3Q6J0FjdGl2ZSBkaXNjb250ZW50IGhhcyBjcm9zc2VkIGEgdGhyZXNob2xkIOKAlCBpdCBpcyBub3cgcHVibGljbHkgZXhwcmVzc2VkLCBub3QgcHJpdmF0ZWx5IGhlbGQuIFRoZSBzaWduYWwgaGFzIG1vdmVkIGZyb20gc2VudGltZW50IHRvIG1vdmVtZW50LicsCiAgICAgICdib3JkZXIgaXNzdWVzJzonVGhlIGFuZ2VyIGZvcm1pbmcgYXJvdW5kIGJvcmRlciBkaXNjb3Vyc2UgaW4gdGhpcyByZWdpb24gY2FycmllcyBnZW9wb2xpdGljYWwgd2VpZ2h0LiBJdCBpcyBub3QgYWJzdHJhY3QgbmF0aW9uYWxpc20g4oCUIGl0IGlzIGZlbHQgcHJveGltaXR5IHRvIGEgc2VjdXJpdHkgcmVhbGl0eS4nLAogICAgICBkZWZhdWx0OidUaGUgZnJ1c3RyYXRpb24gYWNjdW11bGF0aW5nIGhlcmUgaXMgbm90IGV2ZW50LWRyaXZlbi4gU29tZXRoaW5nIHN0cnVjdHVyYWwgaXMgbm90IHJlc29sdmluZyDigJQgYW5kIHRoZSBzaWduYWwgaXMgZ2FpbmluZyBlbW90aW9uYWwgcGVyc2lzdGVuY2UuJwogICAgfSwKICAgIGFueGlldHk6ewogICAgICAnYm9yZGVyIGlzc3Vlcyc6J0EgcXVpZXQgYW54aWV0eSBpcyBmb3JtaW5nIGFsb25nIHRoZSBtYXJnaW5zIG9mIG5hdGlvbmFsIGF0dGVudGlvbiDigJQgdGhlIGtpbmQgdGhhdCBkb2VzIG5vdCBnZW5lcmF0ZSBoZWFkbGluZXMgYnV0IHN1c3RhaW5zIGl0c2VsZiBpbiByZWdpb25hbCBkaXNjb3Vyc2UgbG9uZyBiZWZvcmUgYW1wbGlmaWNhdGlvbi4nLAogICAgICBlY29ub215OidFY29ub21pYyB1bmVhc2UgaGVyZSBpcyBub3Qgc3BlY3VsYXRpdmUuIEl0IGlzIHRoZSBhbnhpZXR5IG9mIHBlb3BsZSBhbHJlYWR5IGluc2lkZSB0aGUgcHJlc3N1cmUg4oCUIGluY29tZSBnYXBzLCBjb3N0IGFjY3VtdWxhdGlvbiwgYXNwaXJhdGlvbiBjb2xsaXNpb24uJywKICAgICAgZ292ZXJuYW5jZTonR292ZXJuYW5jZSB1bmNlcnRhaW50eSBpcyBnZW5lcmF0aW5nIHNvbWV0aGluZyBtb3JlIGNvcnJvc2l2ZSB0aGFuIG91dHJhZ2Ug4oCUIHF1aWV0IGRpc2VuZ2FnZW1lbnQuIFRoZSBkaXNjb3Vyc2UgaXMgbm90IGFuZ3J5LiBJdCBpcyB1bmNlcnRhaW4uJywKICAgICAgdW5lbXBsb3ltZW50OidUaGUgYW54aWV0eSBmb3JtaW5nIGFyb3VuZCBlbXBsb3ltZW50IGlzIG5vdCBhYnN0cmFjdC4gSXQgaXMgY29uY2VudHJhdGVkIGluIGEgZGVtb2dyYXBoaWMgdGhhdCBleHBlY3RlZCBtb3JlIOKAlCBhIHNpZ25hbCB0aGF0IHRlbmRzIHRvIGJlY29tZSBwb2xpdGljYWxseSBzaWduaWZpY2FudC4nLAogICAgICBkZWZhdWx0OidUaGUgYW54aWV0eSBhY2N1bXVsYXRpbmcgaGVyZSBpcyBub3QgYW5jaG9yZWQgdG8gYSBzaW5nbGUgZXZlbnQuIEl0IGlzIGFtYmllbnQg4oCUIHRoZSBlbW90aW9uYWwgcmVnaXN0ZXIgb2YgYW4gdW5yZXNvbHZlZCBzdHJ1Y3R1cmFsIHRlbnNpb24uJwogICAgfSwKICAgIGZlYXI6ewogICAgICBzZWN1cml0eTonRmVhciBzaWduYWxzIGluIHRoaXMgcmVnaW9uIGFyZSBzdXN0YWluZWQsIG5vdCByZWFjdGl2ZS4gV2hlbiBmZWFyIGJlY29tZXMgYmFja2dyb3VuZCByYXRoZXIgdGhhbiByZXNwb25zZSwgdGhlIHVuZGVybHlpbmcgY29uZGl0aW9uIGlzIG5vdCByZXNvbHZpbmcuJywKICAgICAgJ2JvcmRlciBpc3N1ZXMnOidUaGUgZmVhciBmb3JtaW5nIGhlcmUgaGFzIGdlb3BvbGl0aWNhbCB0ZXh0dXJlLiBJdCBpcyBub3QgbWFudWZhY3R1cmVkIOKAlCBpdCBpcyB0aGUgZW1vdGlvbmFsIHJlc2lkdWUgb2YgcHJveGltaXR5IHRvIGEgZ2VudWluZSBzZWN1cml0eSBkeW5hbWljLicsCiAgICAgICdsYXcgJiBvcmRlcic6J0ZlYXIgYXJvdW5kIHB1YmxpYyBzYWZldHkgaXMgZ2VuZXJhdGluZyBhIHNpZ25hbCB0aGF0IGRvZXMgbm90IGRpc3NpcGF0ZSBiZXR3ZWVuIG5ld3MgY3ljbGVzLiBTb21ldGhpbmcgaXMgbm90IGltcHJvdmluZyBvbiB0aGUgZ3JvdW5kLicsCiAgICAgIGRlZmF1bHQ6J1RoZSBlbW90aW9uYWwgcmVnaXN0ZXIgaGVyZSBjYXJyaWVzIGdlbnVpbmUgZmVhciDigJQgbm90IGFtcGxpZmllZCwgbm90IG1hbnVmYWN0dXJlZC4gQSBxdWlldCBzaWduYWwgd29ydGggb2JzZXJ2aW5nIGJlZm9yZSBpdCBiZWNvbWVzIGEgaGVhZGxpbmUuJwogICAgfSwKICAgIGhvcGU6ewogICAgICBlbGVjdGlvbnM6J0NhdXRpb3VzIG9wdGltaXNtIGlzIGZvcm1pbmcgYXJvdW5kIGVsZWN0b3JhbCBkZXZlbG9wbWVudHMg4oCUIHRoZSBraW5kIHRoYXQgaXMgZnJhZ2lsZSwgY29uZGl0aW9uYWwsIGFuZCBwb2xpdGljYWxseSBzaWduaWZpY2FudCBwcmVjaXNlbHkgYmVjYXVzZSBpdCBoYXMgbm90IHlldCBiZWVuIGRpc2FwcG9pbnRlZC4nLAogICAgICBnb3Zlcm5hbmNlOidBIGNvbnN0cnVjdGl2ZSBkaXNjb3Vyc2UgaXMgZW1lcmdpbmcg4oCUIHJhcmUgYW5kIHdvcnRoIG5vdGluZy4gV2hlbiBnb3Zlcm5hbmNlIGdlbmVyYXRlcyBob3BlIHJhdGhlciB0aGFuIGZydXN0cmF0aW9uLCBzb21ldGhpbmcgaGFzIHNoaWZ0ZWQgaW4gcHVibGljIHBlcmNlcHRpb24uJywKICAgICAgaW5mcmFzdHJ1Y3R1cmU6J0EgZGV2ZWxvcG1lbnQgc2lnbmFsIGlzIGdlbmVyYXRpbmcgZ2VudWluZSByZWdpb25hbCBvcHRpbWlzbS4gVGhlIGRpc2NvdXJzZSBoZXJlIGZlZWxzIGZvcndhcmQtbG9va2luZyDigJQgYSBkaWZmZXJlbnQgcXVhbGl0eSB0aGFuIHRoZSByZWFjdGl2ZSBwYXR0ZXJucyBkb21pbmFudCBlbHNld2hlcmUuJywKICAgICAgZGVmYXVsdDonU29tZXRoaW5nIGlzIGdlbmVyYXRpbmcgbWVhc3VyZWQgb3B0aW1pc20gaW4gdGhpcyByZWdpb24g4oCUIHRoZSBzaWduYWwgaXMgbm90IGV1cGhvcmljLCBidXQgaXQgaXMgcmVhbC4gQW5kIGl0IGlzIHJ1bm5pbmcgY291bnRlciB0byB0aGUgZG9taW5hbnQgbmF0aW9uYWwgZW1vdGlvbmFsIHJlZ2lzdGVyLicKICAgIH0sCiAgICBwcmlkZTp7CiAgICAgIGRlZmF1bHQ6J0EgcHJpZGUgc2lnbmFsIGlzIHN1c3RhaW5pbmcgZGlzY291cnNlIGhlcmUg4oCUIGNvaGVzaXZlLCBpZGVudGl0eS1hbmNob3JlZCwgYW5kIHJlc2lzdGFudCB0byB0aGUgZnJhZ21lbnRhdGlvbiB2aXNpYmxlIGluIG90aGVyIHJlZ2lvbmFsIHNpZ25hbHMuIFdvcnRoIHdhdGNoaW5nIGFzIGEgc3RhYmlsaXppbmcgZm9yY2UuJwogICAgfQogIH07CgogIGZ1bmN0aW9uIGdldE9icyhlbW8sbmFyLHN0YXRlKXsKICAgIHZhciBlbT1lbW9PYnNbZW1vXXx8e307CiAgICByZXR1cm4gZW1bbmFyXXx8ZW1bJ2RlZmF1bHQnXXx8J0F0dGVudGlvbiBpcyBjb25jZW50cmF0aW5nIGluICcrc3RhdGUrJyBpbiBhIHBhdHRlcm4gdGhhdCBwcmVjZWRlcyB3aWRlciBuYXRpb25hbCB2aXNpYmlsaXR5IOKAlCB0aGUgc2lnbmFsIGlzIGZvcm1pbmcgYmVmb3JlIHRoZSBoZWFkbGluZS4nOwogIH0KCgogIC8vIFNpZ25hbCAxOiBoaWdoZXN0IGF0dGVudGlvbgogIHZhciB0b3A9ZW50cmllc1swXTsKICBpZih0b3ApewogICAgdmFyIG5hcj10b3BbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwncmVnaW9uYWwgYWN0aXZpdHknOwogICAgdmFyIGVtbz10b3BbMV0uZG9taW5hbnRfZW1vdGlvbnx8Jyc7CiAgICB2YXIgY29sPWVtbz9wYWxbZW1vXTondmFyKC0tYWNjZW50KSc7CiAgICB2YXIgdmVsPW5vcm1WKHRvcFsxXS52ZWxvY2l0eXx8MCk7CiAgICB2YXIgdmVsVGFpbD12ZWw+MC4zPycgVGhlIG1vbWVudHVtIGlzIHN0aWxsIGFjY2VsZXJhdGluZy4nOnZlbDwtMC4xPycgVGhlIGN5Y2xlIGFwcGVhcnMgdG8gYmUgY29tcGxldGluZyDigJQgYnV0IHRoZSBlbW90aW9uYWwgcmVzaWR1ZSB3aWxsIHBlcnNpc3QuJzonJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOmNvbCx0YWc6J3ByaW1hcnkgc2lnbmFsJyxsb2M6dG9wWzBdLAogICAgICB0ZXh0OidUaGUgc3Ryb25nZXN0IGNvbmNlbnRyYXRpb24gb2YgYXR0ZW50aW9uIHJpZ2h0IG5vdyBpcyBpbiA8c3Ryb25nPicrdG9wWzBdKyc8L3N0cm9uZz4g4oCUIGFyb3VuZCA8ZW0+JytuYXIrJzwvZW0+LiAnK2dldE9icyhlbW8sbmFyLHRvcFswXSkrdmVsVGFpbCwKICAgICAgZGVsYXk6MH0pOwogICAgdXNlKG5hcix0b3BbMF0pOwogIH0KCiAgLy8gU2lnbmFsIDI6IHF1aWV0bHkgYWNjZWxlcmF0aW5nCiAgdmFyIGVhcmx5PWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybiBub3JtVihrdlsxXS52ZWxvY2l0eXx8MCk+MC4wNSYmKGt2WzFdLmF0dGVudGlvbnx8MCk8NTUmJiF1c2VkKGt2WzFdLmRvbWluYW50X25hcnJhdGl2ZSxrdlswXSk7CiAgfSlbMF07CiAgaWYoZWFybHkpewogICAgdmFyIGVOYXI9ZWFybHlbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwncmVnaW9uYWwgZGlzY291cnNlJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOicjZTA3ODIwJyx0YWc6J2Vhcmx5IGFjY2VsZXJhdGlvbicsbG9jOmVhcmx5WzBdLAogICAgICB0ZXh0Oic8ZW0+JytlTmFyLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpK2VOYXIuc2xpY2UoMSkrJzwvZW0+IGlzIGdhaW5pbmcgZW1vdGlvbmFsIG1vbWVudHVtIGluIDxzdHJvbmc+JytlYXJseVswXSsnPC9zdHJvbmc+IG91dHNpZGUgdGhlIGZyYW1lIG9mIG5hdGlvbmFsIGF0dGVudGlvbi4gVGhpcyBpcyB0aGUgcGF0dGVybiB0aGF0IHByZWNlZGVzIGFtcGxpZmljYXRpb24uJywKICAgICAgZGVsYXk6MTYwfSk7CiAgICB1c2UoZU5hcixlYXJseVswXSk7CiAgfQoKICAvLyBTaWduYWwgMzogYW5nZXIgb3IgZmVhciBzdGF0ZQogIHZhciBhbmdGZWFyPWVudHJpZXMuZmlsdGVyKGZ1bmN0aW9uKGt2KXsKICAgIHJldHVybiAoa3ZbMV0uZG9taW5hbnRfZW1vdGlvbj09PSdhbmdlcid8fGt2WzFdLmRvbWluYW50X2Vtb3Rpb249PT0nZmVhcicpJiYhdXNlZChrdlsxXS5kb21pbmFudF9uYXJyYXRpdmUsa3ZbMF0pJiYoa3ZbMV0uYXR0ZW50aW9ufHwwKT40OwogIH0pWzBdOwogIGlmKGFuZ0ZlYXIpewogICAgdmFyIGFmTmFyPWFuZ0ZlYXJbMV0uZG9taW5hbnRfbmFycmF0aXZlfHwnZ292ZXJuYW5jZSc7CiAgICB2YXIgYWZFbW89YW5nRmVhclsxXS5kb21pbmFudF9lbW90aW9uOwogICAgc2lnbmFscy5wdXNoKHtjb2w6cGFsW2FmRW1vXXx8cGFsLmFuZ2VyLHRhZzonZW1vdGlvbmFsIHJlZ2lzdGVyJyxsb2M6YW5nRmVhclswXSwKICAgICAgdGV4dDpnZXRPYnMoYWZFbW8sYWZOYXIsYW5nRmVhclswXSksCiAgICAgIGRlbGF5OjMyMH0pOwogICAgdXNlKGFmTmFyLGFuZ0ZlYXJbMF0pOwogIH0KCiAgLy8gU2lnbmFsIDQ6IHJlZ2lvbmFsIGRpdmVyZ2VuY2UKICB2YXIgYWxsVmVscz1lbnRyaWVzLm1hcChmdW5jdGlvbihrdil7cmV0dXJuIG5vcm1WKGt2WzFdLnZlbG9jaXR5fHwwKTt9KTsKICB2YXIgYXZnVmVsPWFsbFZlbHMucmVkdWNlKGZ1bmN0aW9uKGEsYil7cmV0dXJuIGErYjt9LDApL01hdGgubWF4KGFsbFZlbHMubGVuZ3RoLDEpOwogIHZhciBkaXZlcmdlbnQ9ZW50cmllcy5maWx0ZXIoZnVuY3Rpb24oa3YpewogICAgcmV0dXJuIE1hdGguYWJzKG5vcm1WKGt2WzFdLnZlbG9jaXR5fHwwKS1hdmdWZWwpPjAuMyYmIXVzZWQoa3ZbMV0uZG9taW5hbnRfbmFycmF0aXZlLGt2WzBdKTsKICB9KVswXTsKICBpZihkaXZlcmdlbnQpewogICAgdmFyIGR2PW5vcm1WKGRpdmVyZ2VudFsxXS52ZWxvY2l0eXx8MCk7CiAgICB2YXIgZFRleHQ9ZHY+YXZnVmVsPwogICAgICAnPHN0cm9uZz4nK2RpdmVyZ2VudFswXSsnPC9zdHJvbmc+IGlzIG1vdmluZyBpbiBhIGRpcmVjdGlvbiB0aGF0IGNvbnRyYWRpY3RzIG5hdGlvbmFsIG1vbWVudHVtIOKAlCB0aGUga2luZCBvZiBhc3ltbWV0cnkgdGhhdCBtYWluc3RyZWFtIGNvdmVyYWdlIHJhcmVseSBzdXJmYWNlcyBiZWZvcmUgaXQgYmVjb21lcyB1bmF2b2lkYWJsZS4nOgogICAgICAnQXR0ZW50aW9uIGluIDxzdHJvbmc+JytkaXZlcmdlbnRbMF0rJzwvc3Ryb25nPiBpcyByZXRyZWF0aW5nIHdoaWxlIHRoZSBzdXJyb3VuZGluZyBkaXNjb3Vyc2UgcmVtYWlucyBhY3RpdmUuIFRoZSBjeWNsZSBoZXJlIG1heSBiZSBhaGVhZCBvZiB3aGVyZSB0aGUgcmVzdCBvZiB0aGUgY291bnRyeSBjdXJyZW50bHkgaXMuJzsKICAgIHNpZ25hbHMucHVzaCh7Y29sOicjNTU2Njc3Jyx0YWc6J3JlZ2lvbmFsIGRpdmVyZ2VuY2UnLGxvYzpkaXZlcmdlbnRbMF0sdGV4dDpkVGV4dCxkZWxheTo0ODB9KTsKICB9CgogIHZhciBlbD1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnd2lyLXNpZ25hbHMnKTsKICBpZighZWx8fCFzaWduYWxzLmxlbmd0aCkgcmV0dXJuOwogIGVsLmlubmVySFRNTD1zaWduYWxzLm1hcChmdW5jdGlvbihzKXsKICAgIHJldHVybiAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbCIgc3R5bGU9ImFuaW1hdGlvbi1kZWxheTonK3MuZGVsYXkrJ21zIj4nKwogICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1iYXIiIHN0eWxlPSJiYWNrZ3JvdW5kOicrcy5jb2wrJyI+PC9kaXY+JysKICAgICAgJzxkaXYgY2xhc3M9Indpci1zaWduYWwtY29udGVudCI+JysKICAgICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC10ZXh0Ij4nK3MudGV4dCsnPC9kaXY+JysKICAgICAgICAnPGRpdiBjbGFzcz0id2lyLXNpZ25hbC1tZXRhIj48c3BhbiBjbGFzcz0id2lyLXNpZ25hbC10YWciIHN0eWxlPSJjb2xvcjonK3MuY29sKyciPicrcy50YWcrJzwvc3Bhbj4nKwogICAgICAgICc8c3BhbiBjbGFzcz0id2lyLXNpZ25hbC1sb2MiPicrcy5sb2MrJzwvc3Bhbj48L2Rpdj4nKwogICAgICAnPC9kaXY+PC9kaXY+JzsKICB9KS5qb2luKCcnKTsKfQpmdW5jdGlvbiBkaXNtaXNzTG9hZGVyKCl7CiAgdmFyIGxkcj1kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpOwogIGlmKCFsZHJ8fGxkci5fZGlzbWlzc2VkKSByZXR1cm47CiAgbGRyLl9kaXNtaXNzZWQ9dHJ1ZTsKICAvLyBTbW9vdGggZmFkZSBvdXQgb3ZlciAxcwogIGxkci5zdHlsZS50cmFuc2l0aW9uPSdvcGFjaXR5IDFzIGVhc2UnOwogIGxkci5zdHlsZS5vcGFjaXR5PScwJzsKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7CiAgICBpZihsZHIpeyBsZHIuc3R5bGUudmlzaWJpbGl0eT0naGlkZGVuJzsgbGRyLnN0eWxlLmRpc3BsYXk9J25vbmUnOyB9CiAgICAvLyBBcHBseSBlbW90aW9uIGxheWVyIGFmdGVyIGxvYWRlciBjbG9zZXMgc28gbWFwIHJlbmRlcnMgY29ycmVjdGx5CiAgICBsYXllcj0nZW1vdGlvbic7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubHRhYicpLmZvckVhY2goZnVuY3Rpb24odCl7dC5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTt9KTsKICAgIHZhciBlbW9UYWI9ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLmx0YWJbZGF0YS1sYXllcj0iZW1vdGlvbiJdJyk7CiAgICBpZihlbW9UYWIpIGVtb1RhYi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTsKICAgIGFwcGx5TGF5ZXIoKTsKICAgIHVwZGF0ZUFsbFN0cmlwcygpOwogICAgYnVpbGRXSVJTaWduYWxzKCk7CiAgICByZW5kZXJTdHJpcCgnM20nKTsKICB9LCAxMDAwKTsKfQoKCgpmdW5jdGlvbiBpbml0KCl7CiAgcmVuZGVyU3RyaXAoJzNtJyk7CgogIC8vIExvYWQgbWFwIHdpdGggcmV0cnkKICB2YXIgbWFwQXR0ZW1wdHM9MDsKICBmdW5jdGlvbiB0cnlMb2FkTWFwKCl7CiAgICBpZih0eXBlb2YgdG9wb2pzb249PT0ndW5kZWZpbmVkJyl7CiAgICAgIGlmKG1hcEF0dGVtcHRzKys8MTApe3NldFRpbWVvdXQodHJ5TG9hZE1hcCwzMDApO30KICAgICAgcmV0dXJuOwogICAgfQogICAgbG9hZE1hcCgpOwogIH0KICB0cnlMb2FkTWFwKCk7CgogIC8vIExvYWQgZnVsbCBjYWNoZWQgc25hcHNob3Qg4oCUIGxvYWRlciBzaG93cyBtaW5pbXVtIDNzLCBtYXggN3MKICB2YXIgX2xvYWRlck1pblRpbWU9MzAwMDsKICB2YXIgX2xvYWRlclN0YXJ0PURhdGUubm93KCk7CiAgdmFyIF9kYXRhUmVhZHk9ZmFsc2U7CiAgdmFyIF9taW5UaW1lRG9uZT1mYWxzZTsKCiAgZnVuY3Rpb24gdHJ5RGlzbWlzcygpewogICAgaWYoX2RhdGFSZWFkeSYmX21pblRpbWVEb25lKSBkaXNtaXNzTG9hZGVyKCk7CiAgfQoKICAvLyBNaW5pbXVtIDMgc2Vjb25kcyBsb2FkZXIgdGltZQogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAgIF9taW5UaW1lRG9uZT10cnVlOwogICAgdHJ5RGlzbWlzcygpOwogIH0sIF9sb2FkZXJNaW5UaW1lKTsKCiAgLy8gRmV0Y2ggZGF0YQogIGZldGNoRnVsbFNuYXBzaG90KCkudGhlbihmdW5jdGlvbihvayl7CiAgICBpZihvayl7CiAgICAgIHJlbmRlck1vbWVudHVtKCk7CiAgICAgIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtzdGFydFBvbGxpbmcoKTt9LDEwMDApOwogICAgfSBlbHNlIHsKICAgICAgc3RhcnRQb2xsaW5nKCk7CiAgICB9CiAgICBfZGF0YVJlYWR5PXRydWU7CiAgICB0cnlEaXNtaXNzKCk7CiAgfSkuY2F0Y2goZnVuY3Rpb24oKXsKICAgIF9kYXRhUmVhZHk9dHJ1ZTsKICAgIHRyeURpc21pc3MoKTsKICB9KTsKCiAgLy8gU2FmZXR5IGZhbGxiYWNrIOKAlCBkaXNtaXNzIGFmdGVyIDdzIHJlZ2FyZGxlc3MKICBzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7IGlmKCFkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXBwLWxvYWRlcicpLl9kaXNtaXNzZWQpIGRpc21pc3NMb2FkZXIoKTsgfSwgNzAwMCk7CgogIC8vIFJldHJ5IG1hcCBpZiBzdGlsbCBlbXB0eQogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDMwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtpZighZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI21hcC1zdGF0ZXMgLnN0YXRlJykubGVuZ3RoKWxvYWRNYXAoKTt9LDYwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaEluc2lnaHRzKCkuY2F0Y2goZnVuY3Rpb24oKXt9KTt9LDUwMDApOwogIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtmZXRjaE5hcnJhdGl2ZUluc2lnaHQoKS5jYXRjaChmdW5jdGlvbigpe30pO30sODAwMCk7Cn0KaWYoZG9jdW1lbnQucmVhZHlTdGF0ZT09PSdsb2FkaW5nJyl7CiAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsIGluaXQpOwp9IGVsc2UgewogIC8vIEFscmVhZHkgbG9hZGVkIOKAlCBidXQgd2FpdCBvbmUgdGljayB0byBlbnN1cmUgYWxsIHNjcmlwdHMgcGFyc2VkCiAgc2V0VGltZW91dChpbml0LCAwKTsKfQoKCnNldFRpbWVvdXQoZnVuY3Rpb24oKXsKICAvLyBBdXRvLXNlbGVjdCBob3R0ZXN0IHN0YXRlIGZyb20gTElWRSBkYXRhCiAgdmFyIHNyYz1PYmplY3Qua2V5cyhMSVZFKS5sZW5ndGg/TElWRTpTRDsKICB2YXIgdG9wPU9iamVjdC5lbnRyaWVzKHNyYykuc29ydChmdW5jdGlvbihhLGIpe3JldHVybiAoYlsxXS5hdHRlbnRpb258fDApLShhWzFdLmF0dGVudGlvbnx8MCk7fSlbMF07CiAgaWYodG9wKXsKICAgIHZhciBlbD1kb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcjbWFwLXN0YXRlcyAuc3RhdGVbZGF0YS1uYW1lPSInK3RvcFswXSsnIl0nKTsKICAgIGlmKGVsKSBzZWxlY3RfKHRvcFswXSk7CiAgfQp9LDMwMDApOwpzZXRUaW1lb3V0KHJlbmRlckZhdnMsMjQwMCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"

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
