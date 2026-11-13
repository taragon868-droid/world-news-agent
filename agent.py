"""Ripple news agent.

Fetches world news from RSS feeds, groups duplicate coverage into stories, asks a free LLM
which ones are major (directly affect people's lives in a country, or reach other countries),
and writes the result to data/stories.json for the website.
"""
import datetime as dt
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

import feedparser
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
STORIES_FILE = DATA / "stories.json"
STATE_FILE = DATA / "state.json"

FEEDS = {
    "BBC News": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "DW": "https://rss.dw.com/rdf/rss-en-world",
    "France 24": "https://www.france24.com/en/rss",
    "NPR": "https://feeds.npr.org/1004/rss.xml",
    "UN News": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    "WHO": "https://www.who.int/rss-feeds/news-english.xml",
}
OFFICIAL = {"UN News", "WHO"}  # one official source is enough to consider a story

# Providers are tried in order; if one is rate limited the next one takes over.
PROVIDERS = [
    {
        "name": "gemini",
        "key": os.getenv("GEMINI_API_KEY"),
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "prefer": ["flash"],
    },
    {
        "name": "groq",
        "key": os.getenv("GROQ_API_KEY"),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "prefer": ["llama-3.3-70b", "llama-3.1-8b", "gpt-oss-120b", "qwen"],
    },
]

MAX_AGE_HOURS = 36      # ignore older articles
KEEP_DAYS = 7           # how long stories stay on the site
MAX_SCORED = 12         # deep-scoring calls per run (protects free limits)
MIN_IMPORTANCE = 50     # 0-100 needed to publish
MIN_CONFIDENCE = 5      # 0-10 needed to publish
PAUSE = 5               # seconds between model calls

NOW = dt.datetime.now(dt.timezone.utc)


# ---------- model access ----------
def discover_model(p):
    """Ask the provider which models exist and pick the best match (model names change often)."""
    try:
        r = requests.get(
            p["url"].rsplit("/chat/completions", 1)[0] + "/models",
            headers={"Authorization": f"Bearer {p['key']}"},
            timeout=30,
        )
        ids = [m["id"].replace("models/", "") for m in r.json().get("data", [])]
    except Exception as e:
        print("  could not list models:", e)
        return None
    bad = ("whisper", "guard", "tts", "embed", "image", "live", "audio", "compound", "orpheus", "playai", "lite", "vision", "robotics", "computer", "safeguard")
    ids = [i for i in ids if not any(b in i for b in bad)]
    version = lambda i: [int(n) for n in re.findall(r"\d+", i)]
    for pref in p["prefer"]:
        found = sorted((i for i in ids if pref in i), key=version, reverse=True)
        if found:
            return found[0]
    return None


def llm(system, prompt):
    last = "no API keys set (add GEMINI_API_KEY and/or GROQ_API_KEY as GitHub secrets)"
    for p in PROVIDERS:
        if not p["key"]:
            continue
        for attempt in range(2):
            try:
                r = requests.post(
                    p["url"],
                    headers={"Authorization": f"Bearer {p['key']}"},
                    json={
                        "model": p["model"],
                        "temperature": 0.2,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=90,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                last = f"{p['name']} ({p['model']}) returned {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 404) and attempt == 0:
                    new = discover_model(p)
                    if new and new != p["model"]:
                        print(f"  model {p['model']} unavailable, switching to {new}")
                        p["model"] = new
                        continue
            except Exception as e:  # network errors, bad JSON, etc.
                last = f"{p['name']} failed: {e}"
            break
        print("  provider problem ->", last)
    raise RuntimeError(last)


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        return json.loads(m.group(0)) if m else None


# ---------- fetching and clustering ----------
STOP = set(
    "the and for with from after over into new says say will has have had not but about amid "
    "against what why how who its are was were been this that than more most year".split()
)


def tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def clean(text, n=400):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()[:n]


def fetch_articles():
    articles = []
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url, agent="RippleNewsAgent/1.0")
            if not feed.entries:
                print(f"  no entries from {source} (feed URL may have changed)")
            for e in feed.entries:
                p = e.get("published_parsed") or e.get("updated_parsed")
                when = dt.datetime(*p[:6], tzinfo=dt.timezone.utc) if p else NOW
                if (NOW - when).total_seconds() > MAX_AGE_HOURS * 3600:
                    continue
                title = clean(e.get("title"), 200)
                link = e.get("link")
                if title and link:
                    articles.append(
                        {
                            "source": source,
                            "title": title,
                            "summary": clean(e.get("summary")),
                            "url": link,
                            "when": when,
                            "tok": tokens(title),
                        }
                    )
        except Exception as ex:
            print(f"  could not read {source}: {ex}")
    return articles


def cluster(articles):
    clusters = []
    for a in sorted(articles, key=lambda x: x["when"]):
        best, best_score = None, 0
        for c in clusters:
            for b in c["articles"]:
                shared = len(a["tok"] & b["tok"])
                small = min(len(a["tok"]), len(b["tok"])) or 1
                score = shared / small
                if shared >= 3 and score >= 0.5 and score > best_score:
                    best, best_score = c, score
        if best:
            best["articles"].append(a)
        else:
            clusters.append({"articles": [a]})
    return clusters


def sources_of(c):
    return {a["source"] for a in c["articles"]}


# ---------- judging ----------
TRIAGE_SYSTEM = "You are a news editor for a site that publishes only globally significant news. Reply with JSON only."
SCORE_SYSTEM = (
    "You are a careful news analyst. Judge whether a story is major: it directly affects the lives "
    "of many people in a country, or a state's decision affects other countries. Use only the "
    "provided article text; never invent facts. Write in your own words. Reply with JSON only."
)


def triage(clusters):
    keep = []
    for i in range(0, len(clusters), 40):
        chunk = clusters[i : i + 40]
        lines = "\n".join(f"{n}: {c['articles'][0]['title']}" for n, c in enumerate(chunk))
        prompt = (
            "Numbered headlines follow. Return a JSON list of the numbers of stories that could be "
            "MAJOR: conflict or security, big laws or policy, economy/trade/energy, health emergencies, "
            "disasters, migration, major technology regulation, anything with effects across borders. "
            "Exclude sports, celebrity, routine crime, local news and opinion.\n\n" + lines
        )
        ids = parse_json(llm(TRIAGE_SYSTEM, prompt)) or []
        keep += [chunk[n] for n in ids if isinstance(n, int) and 0 <= n < len(chunk)]
        time.sleep(PAUSE)
    return keep


def score(c):
    arts = c["articles"][:6]
    text = "\n\n".join(f"[{a['source']}] {a['title']}\n{a['summary']}" for a in arts)
    prompt = (
        "Analyse this story and return JSON with exactly these keys:\n"
        '{"major": true/false,\n'
        ' "scores": {"scale": 0-10 (people affected), "severity": 0-10 (lives, money, rights, safety), '
        '"cross_border": 0-10 (effect on other countries), "durability": 0-10 (matters next month)},\n'
        ' "confidence": 0-10 (how well the articles confirm it),\n'
        ' "headline": neutral headline in your own words,\n'
        ' "summary": 2-3 sentence summary in your own words,\n'
        ' "country": main country affected or "Global",\n'
        ' "regions": [other countries or regions affected],\n'
        ' "inside": 1-2 sentences on how people in that country are directly affected,\n'
        ' "beyond": 1-2 sentences on effects on other countries, or "" if none,\n'
        ' "category": one of "Conflict","Politics & law","Economy & trade","Health","Climate & disasters","Energy","Technology","Migration","Other"}\n\n'
        + text
    )
    r = parse_json(llm(SCORE_SYSTEM, prompt))
    if not isinstance(r, dict) or not r.get("major"):
        return None
    s = r.get("scores", {})
    importance = round(
        10 * (0.3 * s.get("scale", 0) + 0.3 * s.get("severity", 0) + 0.25 * s.get("cross_border", 0) + 0.15 * s.get("durability", 0))
    )
    if importance < MIN_IMPORTANCE or r.get("confidence", 0) < MIN_CONFIDENCE:
        return None
    return {
        "id": hashlib.sha1(arts[0]["url"].encode()).hexdigest()[:12],
        "headline": r.get("headline", arts[0]["title"]),
        "summary": r.get("summary", ""),
        "country": r.get("country", "Global"),
        "regions": r.get("regions", []),
        "inside": r.get("inside", ""),
        "beyond": r.get("beyond", ""),
        "category": r.get("category", "Other"),
        "importance": importance,
        "confidence": r.get("confidence"),
        "scores": s,
        "sources": [{"name": a["source"], "title": a["title"], "url": a["url"]} for a in arts],
        "published": min(a["when"] for a in arts).isoformat(),
        "updated": NOW.isoformat(),
    }


# ---------- state ----------
def load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    DATA.mkdir(exist_ok=True)
    stories = load(STORIES_FILE, {"stories": []})["stories"]
    seen = load(STATE_FILE, {"seen": {}})["seen"]
    cutoff = NOW - dt.timedelta(days=KEEP_DAYS)
    stories = [s for s in stories if dt.datetime.fromisoformat(s["updated"]) > cutoff]
    seen = {u: t for u, t in seen.items() if dt.datetime.fromisoformat(t) > cutoff}

    print("Fetching feeds...")
    articles = fetch_articles()
    clusters = cluster(articles)
    # Candidates: new stories covered by 2+ independent sources, or by an official source.
    fresh = [
        c for c in clusters
        if not any(a["url"] in seen for a in c["articles"])
        and (len(sources_of(c)) >= 2 or sources_of(c) & OFFICIAL)
    ]
    print(f"{len(articles)} articles -> {len(clusters)} stories -> {len(fresh)} new candidates")

    try:
        picked = triage(fresh)
        picked.sort(key=lambda c: len(sources_of(c)), reverse=True)
        print(f"Triage kept {len(picked)}")
        for c in picked[:MAX_SCORED]:
            story = score(c)
            for a in c["articles"]:
                seen[a["url"]] = NOW.isoformat()
            if story:
                print("  PUBLISH:", story["importance"], story["headline"])
                stories.append(story)
            time.sleep(PAUSE)
    except RuntimeError as e:
        print("Stopping early:", e)

    stories.sort(key=lambda s: s["updated"], reverse=True)
    STORIES_FILE.write_text(
        json.dumps({"updated": NOW.isoformat(), "stories": stories}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    STATE_FILE.write_text(json.dumps({"seen": seen}), encoding="utf-8")
    print(f"Saved {len(stories)} stories")


if __name__ == "__main__":
    main()
