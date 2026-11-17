"""Talent radar: live software-engineering internship postings and skills.

The research desk's counterpart on the hiring side. Two open job APIs are
queried (Arbeitnow and Remotive, both keyless), postings are normalised into one
schema, filtered to software-engineering internships, and the free-text
descriptions are mined against a skill taxonomy to produce a ranked list of
required skills — which can then be compared with a candidate's own profile.

Like ``market_data``, every request is retried, cached in memory and on disk,
and degrades gracefully: if a source is unavailable the other one still works.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

from market_data import CACHE_DIR, MarketDataError, http_get

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api?page={page}"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs?search={query}&limit=50"

TALENT_TTL = 6 * 3600  # hours
_CACHE: Dict[str, tuple] = {}

SWE_PATTERN = re.compile(
    r"software|engineer|developer|programmer|full[\s-]?stack|front[\s-]?end|"
    r"back[\s-]?end|data\s+(?:engineer|scientist|analyst)|machine\s+learning|"
    r"\bml\b|\bai\b|devops|site\s+reliability|cloud|platform|mobile|ios|android|"
    r"quant|python|java|golang|rust|sde|swe|qa\s+engineer|security\s+engineer",
    re.IGNORECASE,
)
INTERN_PATTERN = re.compile(
    r"\bintern(?:ship|s)?\b|\bco[\s-]?op\b|praktik|werkstudent|trainee|"
    r"graduate\s+(?:program|scheme)|junior",
    re.IGNORECASE,
)
SENIOR_PATTERN = re.compile(r"\bsenior\b|\bstaff\b|\bprincipal\b|\blead\b|\bhead\b", re.IGNORECASE)

# skill -> regex. Ordered roughly by how commonly they appear on SWE job ads.
SKILL_TAXONOMY: Dict[str, str] = {
    "Python": r"\bpython\b",
    "JavaScript": r"\bjavascript\b|\bjs\b",
    "TypeScript": r"\btypescript\b",
    "Java": r"\bjava\b",
    "C++": r"\bc\+\+",
    "C#": r"\bc#|\.net\b",
    "Go": r"\bgolang\b|\bgo\b(?=,|\s+developer|\s+engineer)",
    "Rust": r"\brust\b",
    "SQL": r"\bsql\b",
    "React": r"\breact(?:\.js|js)?\b",
    "Node.js": r"\bnode(?:\.js|js)?\b",
    "Django": r"\bdjango\b",
    "Flask": r"\bflask\b",
    "Spring": r"\bspring\s*(?:boot)?\b",
    "REST APIs": r"\brest(?:ful)?\s*api|\bapis?\b",
    "GraphQL": r"\bgraphql\b",
    "AWS": r"\baws\b|amazon web services",
    "GCP": r"\bgcp\b|google cloud",
    "Azure": r"\bazure\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "Terraform": r"\bterraform\b",
    "CI/CD": r"\bci/?cd\b|continuous integration|github actions|jenkins",
    "Git": r"\bgit\b|github|gitlab",
    "Linux": r"\blinux\b|\bunix\b",
    "PostgreSQL": r"\bpostgres(?:ql)?\b",
    "MongoDB": r"\bmongo(?:db)?\b",
    "Redis": r"\bredis\b",
    "Kafka": r"\bkafka\b",
    "Spark": r"\bspark\b|pyspark",
    "Airflow": r"\bairflow\b|\bdagster\b",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "scikit-learn": r"\bscikit[\s-]?learn\b|sklearn",
    "PyTorch": r"\bpytorch\b",
    "TensorFlow": r"\btensorflow\b",
    "Machine Learning": r"\bmachine learning\b|\bml\b",
    "LLMs": r"\bllms?\b|large language model|openai|langchain",
    "Data Structures": r"\bdata structures?\b",
    "Algorithms": r"\balgorithms?\b",
    "Unit Testing": r"\bunit test|\bpytest\b|\bjunit\b|\btdd\b|test automation",
    "Agile": r"\bagile\b|\bscrum\b|\bkanban\b",
    "Kubernetes Operators": r"\boperators?\b(?=.*kubernetes)",
    "Security": r"\bsecurity\b|\boauth\b|\bjwt\b",
    "Tableau": r"\btableau\b|\bpower\s?bi\b",
    "Excel": r"\bexcel\b",
    "Figma": r"\bfigma\b",
}
_COMPILED_SKILLS = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in SKILL_TAXONOMY.items()}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cache_get(key: str, ttl: float):
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None


def _cache_set(key: str, value):
    _CACHE[key] = (time.time(), value)
    return value


def _plain_text(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw or "")
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def extract_skills(text: str) -> List[str]:
    """Return the taxonomy skills mentioned in a block of text."""
    if not text:
        return []
    return [name for name, pattern in _COMPILED_SKILLS.items() if pattern.search(text)]


def _classify(title: str, job_types: Iterable[str], tags: Iterable[str], description: str) -> Dict[str, bool]:
    """Decide whether a posting is a software-engineering internship.

    The role test is deliberately restricted to the title and tags: matching on
    the description lets in unrelated jobs that merely mention a "platform" or
    "AI", which is how a marketing internship ends up in a SWE skills report.
    """
    type_text = " ".join(job_types or [])
    tag_text = " ".join(tags or [])
    haystack = f"{title} {type_text} {tag_text}"
    return {
        "is_internship": bool(INTERN_PATTERN.search(haystack)),
        "is_senior": bool(SENIOR_PATTERN.search(title)),
        "is_swe": bool(SWE_PATTERN.search(f"{title} {tag_text}")),
    }


def _normalise(
    *,
    source: str,
    title: str,
    company: str,
    location: str,
    url: str,
    tags: Iterable[str],
    job_types: Iterable[str],
    description: str,
    published: str,
) -> Optional[Dict[str, object]]:
    title = (title or "").strip()
    text = _plain_text(description)
    if not title:
        return None
    flags = _classify(title, job_types, tags, text)
    if flags["is_senior"] or not flags["is_swe"]:
        return None
    skills = extract_skills(f"{title} {text}")
    return {
        "source": source,
        "title": title,
        "company": (company or "").strip() or "Unknown",
        "location": (location or "Not specified").strip(),
        "url": url,
        "tags": ", ".join(t for t in (tags or []) if t)[:160],
        "job_types": ", ".join(t for t in (job_types or []) if t)[:120],
        "published": published or "",
        "is_internship": flags["is_internship"],
        "skills": skills,
        "description": text[:1200],
    }


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def fetch_arbeitnow(pages: int = 6) -> List[Dict[str, object]]:
    postings: List[Dict[str, object]] = []
    for page in range(1, max(1, pages) + 1):
        try:
            raw = http_get(ARBEITNOW_URL.format(page=page), timeout=20, retries=2)
            payload = json.loads(raw)
        except (MarketDataError, json.JSONDecodeError):
            break
        rows = payload.get("data") or []
        if not rows:
            break
        for job in rows:
            item = _normalise(
                source="Arbeitnow",
                title=job.get("title", ""),
                company=job.get("company_name", ""),
                location=job.get("location", ""),
                url=job.get("url", ""),
                tags=job.get("tags") or [],
                job_types=job.get("job_types") or [],
                description=job.get("description", ""),
                published=_iso(job.get("created_at")),
            )
            if item:
                postings.append(item)
    return postings


def fetch_remotive(queries: Optional[List[str]] = None) -> List[Dict[str, object]]:
    queries = queries or [
        "software intern",
        "developer intern",
        "engineering intern",
        "intern",
    ]
    postings: List[Dict[str, object]] = []
    for query in queries:
        try:
            raw = http_get(
                REMOTIVE_URL.format(query=query.replace(" ", "%20")), timeout=20, retries=2
            )
            payload = json.loads(raw)
        except (MarketDataError, json.JSONDecodeError):
            continue
        for job in payload.get("jobs") or []:
            item = _normalise(
                source="Remotive",
                title=job.get("title", ""),
                company=job.get("company_name", ""),
                location=job.get("candidate_required_location", ""),
                url=job.get("url", ""),
                tags=job.get("tags") or [],
                job_types=[job.get("job_type", "")],
                description=job.get("description", ""),
                published=(job.get("publication_date", "") or "")[:10],
            )
            if item:
                postings.append(item)
    return postings


def _iso(value) -> str:
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(value or "")[:10]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def scan(
    pages: int = 10,
    internships_only: bool = True,
    use_cache: bool = True,
) -> Dict[str, object]:
    """Scan live postings and aggregate the required skills.

    Returns ``{postings, skills, fetched_at, sources, error}`` where ``skills`` is a
    DataFrame of ``skill / postings / share / sources``.
    """
    key = f"talent:{pages}:{internships_only}"
    if use_cache:
        cached = _cache_get(key, TALENT_TTL)
        if cached is not None:
            return cached

    disk_path = os.path.join(CACHE_DIR, "talent.json")
    if use_cache and os.path.exists(disk_path):
        if time.time() - os.path.getmtime(disk_path) < TALENT_TTL:
            try:
                with open(disk_path, "r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                return _cache_set(key, _rebuild(stored))
            except Exception:
                pass

    error = ""
    try:
        postings = fetch_arbeitnow(pages=pages) + fetch_remotive()
    except Exception as exc:  # defensive: never break the page
        postings, error = [], str(exc)

    if internships_only:
        filtered = [p for p in postings if p["is_internship"]]
        if len(filtered) >= 5:
            postings = filtered

    # de-duplicate on (title, company)
    seen = set()
    unique = []
    for job in postings:
        fingerprint = (str(job["title"]).lower()[:80], str(job["company"]).lower()[:60])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(job)

    skills = skill_table(unique)
    result = {
        "postings": pd.DataFrame(unique),
        "skills": skills,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources": sorted({str(p["source"]) for p in unique}),
        "error": error,
    }

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(disk_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "postings": unique,
                    "fetched_at": result["fetched_at"],
                    "sources": result["sources"],
                },
                handle,
            )
    except OSError:
        pass

    return _cache_set(key, result)


def _rebuild(stored: Dict[str, object]) -> Dict[str, object]:
    postings = stored.get("postings") or []
    return {
        "postings": pd.DataFrame(postings),
        "skills": skill_table(postings),
        "fetched_at": stored.get("fetched_at", ""),
        "sources": stored.get("sources", []),
        "error": "",
    }


def skill_table(postings: List[Dict[str, object]]) -> pd.DataFrame:
    """Rank skills by how many postings require them."""
    if not postings:
        return pd.DataFrame(columns=["skill", "postings", "share", "sources"])

    total = len(postings)
    counts: Dict[str, int] = {}
    by_source: Dict[str, set] = {}
    for job in postings:
        for skill in job.get("skills") or []:
            counts[skill] = counts.get(skill, 0) + 1
            by_source.setdefault(skill, set()).add(str(job.get("source", "")))

    rows = [
        {
            "skill": skill,
            "postings": count,
            "share": round(count / total * 100, 1),
            "sources": ", ".join(sorted(s for s in by_source.get(skill, set()) if s)),
        }
        for skill, count in counts.items()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["skill", "postings", "share", "sources"])
    return frame.sort_values(["postings", "skill"], ascending=[False, True]).reset_index(drop=True)


def gap_analysis(skills: pd.DataFrame, my_skills: Iterable[str], top_n: int = 12) -> Dict[str, object]:
    """Compare the most-demanded skills against a candidate's own skill list."""
    mine = {s.strip().lower() for s in (my_skills or []) if s and s.strip()}
    if skills is None or skills.empty:
        return {"covered": [], "missing": [], "coverage": 0.0, "rows": pd.DataFrame()}

    top = skills.head(top_n)
    covered, missing, rows = [], [], []
    for _, row in top.iterrows():
        has = row["skill"].lower() in mine
        (covered if has else missing).append(row["skill"])
        rows.append(
            {
                "skill": row["skill"],
                "share": row["share"],
                "postings": row["postings"],
                "you": "✓" if has else "—",
            }
        )
    coverage = len(covered) / len(top) * 100 if len(top) else 0.0
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(coverage, 1),
        "rows": pd.DataFrame(rows),
    }
