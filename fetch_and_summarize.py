#!/usr/bin/env python3
"""Fetch AI news from RSS feeds, summarize with Mistral, email the harvest."""

import collections
import html
import itertools
import json
import os
import re
import smtplib
import sys
import time
from datetime import date
from email.mime.text import MIMEText
from urllib.parse import urlparse

import feedparser
import requests
from dotenv import load_dotenv
from mistralai.client import Mistral

# domain is used to validate article links before they are hidden
# behind clickable text in the email
FEEDS = {
    "TechCrunch AI": {
        "feed": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "domain": "techcrunch.com",
    },
    "VentureBeat AI": {
        # the category/ai feed froze on a fixed set of old articles;
        # the site feed is chronological and effectively all AI anyway
        "feed": "https://venturebeat.com/feed/",
        "domain": "venturebeat.com",
    },
    "The Rundown AI": {
        "feed": "https://rss.beehiiv.com/feeds/2R3C6Bt5wj.xml",
        "domain": "therundown.ai",
        # each daily post bundles several stories - split them apart
        "split_issue": True,
    },
}

REQUIRED_ENV = ("MISTRAL_API_KEY", "GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO")

ARTICLES_PER_FEED = 4
MAX_ENTRY_AGE_DAYS = 3  # ignore entries older than this, see entry_is_fresh
MAX_STORY_CHARS = 1500  # cap per-story text sent to the model
MISTRAL_MODEL = "mistral-small-latest"
# duplicate spotting is harder than summarizing, so it gets a bigger model
DUP_MODEL = "mistral-medium-latest"
HTTP_TIMEOUT = 15
# some news sites 403 requests without a browser-like user agent
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

TAG_RE = re.compile(r"<[^>]+>")

# remembers what was already sent; on github actions this file is
# carried between runs by the actions cache
SEEN_FILE = "seen_titles.txt"
SEEN_MAX = 500  # cap so the file doesn't grow forever


def safe_link(url, domain):
    """A link may only be hidden behind clickable text if it is https
    and points at the source's own domain (or a subdomain). Keeps a
    compromised feed from smuggling foreign urls behind a trusted name."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (host == domain or host.endswith("." + domain))


def entry_is_fresh(entry):
    """Ignore entries older than a few days. A feed that re-lists old
    items (venturebeat's frozen category feed did exactly that) must
    not be able to push stale news. Entries without a date pass."""
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if not published:
        return True
    return time.mktime(published) >= time.time() - MAX_ENTRY_AGE_DAYS * 86400


def split_issue(entry, source, domain):
    """The Rundown's daily post bundles several stories with ads and
    tool guides in between. Split the post body on its headings and
    keep only blocks with a 'Why it matters' section - real news
    stories always have one, ads and guides never do. The individual
    stories have no urls of their own, so they share the post's url."""
    link = entry.get("link", "")
    if not safe_link(link, domain):
        print(f"[warn] {source}: skipping issue with suspect url: {link}")
        return []
    body = entry.get("content", [{}])[0].get("value", "") or entry.get("summary", "")
    stories = []
    # split keeps the heading texts at odd indexes, block bodies follow
    parts = re.split(r"<h[34][^>]*>(.*?)</h[34]>", body, flags=re.S)
    for i in range(1, len(parts) - 1, 2):
        text = re.sub(r"\s+", " ", TAG_RE.sub(" ", parts[i + 1])).strip()
        if "Why it matters" not in text:
            continue
        # drop image credits and other lead-in before the story text
        if "The Rundown:" in text:
            text = text[text.index("The Rundown:"):]
        stories.append({
            "source": source,
            "title": TAG_RE.sub("", parts[i]).strip(),
            "url": link,
            "description": text[:MAX_STORY_CHARS],
        })
    return stories


def fetch_articles():
    """Collect recent entries from all feeds.

    A failing feed is logged and skipped so one dead site doesn't
    kill the whole run.
    """
    articles = []
    for source, cfg in FEEDS.items():
        try:
            # download ourselves: feedparser's own fetching has no timeout
            resp = requests.get(cfg["feed"], timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[warn] {source}: fetch failed: {e}")
            continue
        feed = feedparser.parse(resp.content)
        if feed.bozo and not feed.entries:
            print(f"[warn] {source}: feed did not parse, skipping")
            continue
        for entry in feed.entries[:ARTICLES_PER_FEED]:
            if not entry_is_fresh(entry):
                continue
            if cfg.get("split_issue"):
                articles += split_issue(entry, source, cfg["domain"])
                continue
            link = entry.get("link", "")
            # drop suspect links before spending a summarization call on them
            if not safe_link(link, cfg["domain"]):
                print(f"[warn] {source}: skipping entry with suspect url: {link}")
                continue
            articles.append({
                "source": source,
                "title": entry.get("title", "").strip(),
                "url": link,
                # descriptions often contain embedded HTML, strip to plain text
                "description": TAG_RE.sub("", entry.get("summary", "")).strip(),
            })
    return articles


def build_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)


def clean_summary(text):
    """Models sometimes ignore formatting instructions - strip any
    markdown bold markers and a leading 'Summary:' label."""
    text = text.strip().replace("**", "").replace("__", "")
    if text.lower().startswith("summary:"):
        text = text[len("summary:"):].strip()
    return text


def summarize(client, article):
    """Two-sentence summary of one article. Falls back to the raw feed
    description if the API call fails, so the article isn't dropped."""
    prompt = (
        "Summarize this AI news item in 2 concise sentences. "
        "Plain text only: no markdown, no headings, no 'Summary:' label.\n"
        f"Title: {article['title']}\n"
        f"Description: {article['description']}"
    )
    try:
        resp = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
    except Exception as e:
        print(f"[warn] summarization failed for '{article['title']}': {e}")
        text = article["description"][:300] or "(no summary available)"
    return clean_summary(text)


SIG_DESC_CHARS = 200  # description prefix that feeds a story signature
SIG_MIN_WORD = 4  # shorter words are noise unless they carry a digit
# tuned against real harvests: genuine cross-source pairs score a
# containment of 0.33 and up, the closest merely-related pair sits at
# 0.23, so the cut sits in the gap between them
DUP_MIN_SHARED = 4
DUP_MIN_CONTAINMENT = 0.30
DUP_SNIPPET_CHARS = 300  # per-story text sent to the duplicate model
DUP_MAX_DROP_RATIO = 0.5  # refuse a verdict that would delete half the harvest
DUP_RETRIES = 2
# The model may lower the lexical bar but not ignore it. Left to itself
# it paired two stories with no words in common at all, and another two
# whose only link was 'openai' and 'models' - words half the harvest
# shares on any given day. So a pair it proposes still has to clear a
# containment just under the lexical one, on terms that are rare enough
# that day to mean something.
DUP_LLM_MIN_CONTAINMENT = 0.25
DUP_LLM_MIN_DISTINCT = 2
DUP_COMMON_DF_RATIO = 0.25  # a term in more of the day's stories is filler

# words too common in news copy to say anything about which story it is
SIG_STOPWORDS = frozenset("""
about after again against along already also although always among another
around because become been before behind being below better between both
build built came come could does doing done down during each early even
every first from further gave general getting give given going gone good
great half hand have having here high hold home however into just keep
kept know known large last late later least less like little long made make
many maybe more most much must near need never next once only open other
others ours over part past people perhaps place plus rather really right
said same says seen several shall should show shown since some soon still
such take taken tell than that their them then there these they thing think
this those though three through thus time today together took toward under
until upon used uses using very want week well went were what when where
which while with within without work would year years your
""".split())

# keep internal hyphens and dots so mai-cyber-1-flash and 2.8t survive
SIG_PUNCT_RE = re.compile(r"[^\w.\-]+")
# every rundown story body opens with this, it distinguishes nothing
RUNDOWN_LEAD_RE = re.compile(r"^\s*the rundown:\s*", re.I)


def story_signature(article):
    """The set of words that identify which story an item is about.

    Title plus the start of the description, lowercased, with emoji,
    punctuation and filler words removed. Short words are dropped
    unless they contain a digit, because that is where the model
    names live - k3, 2.8t, mai-cyber-1-flash.
    """
    desc = RUNDOWN_LEAD_RE.sub("", html.unescape(article.get("description", "")))
    text = html.unescape(article.get("title", "")) + " " + desc[:SIG_DESC_CHARS]
    tokens = set()
    for raw in SIG_PUNCT_RE.sub(" ", text.lower()).split():
        token = raw.strip(".-")
        if not token or token in SIG_STOPWORDS:
            continue
        if len(token) >= SIG_MIN_WORD or any(c.isdigit() for c in token):
            tokens.add(token)
    return tokens


def lexical_duplicates(articles):
    """Pair up stories whose signatures overlap enough to be the same
    news. Deliberately strict: it should never merge two real stories,
    the model pass afterwards is what catches the subtler pairs.

    Returns {index to drop: index it duplicates}, keeping the earliest
    listed item of each group.
    """
    signatures = [story_signature(a) for a in articles]
    drops = {}
    for i, j in itertools.combinations(range(len(articles)), 2):
        shared = signatures[i] & signatures[j]
        smaller = min(len(signatures[i]), len(signatures[j])) or 1
        if len(shared) < DUP_MIN_SHARED or len(shared) / smaller < DUP_MIN_CONTAINMENT:
            continue
        keep, drop = i, j
        while keep in drops:  # fold into the group's surviving item
            keep = drops[keep]
        if drop == keep or drop in drops:
            continue
        drops[drop] = keep
        print(f"[dup/lexical] '{articles[drop]['title']}' duplicates "
              f"'{articles[keep]['title']}' "
              f"(shared={len(shared)}, overlap={len(shared) / smaller:.2f}, "
              f"terms={sorted(shared)})")
    return drops


def parse_duplicate_groups(reply, count):
    """Read the model's JSON verdict into groups of 1-based indexes.

    Strictly json, never a digit scrape: headlines are full of numbers
    ('Kimi K3', 'GPT-5.6') and scraping them deletes real articles. The
    old prompt asked for numbers to drop and the model would return
    every member of a group, taking the story out altogether.
    Raises ValueError on anything malformed so the caller can retry.
    """
    text = reply.strip()
    if text.startswith("```"):  # models like to fence their json
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise ValueError(f"expected an object with a 'groups' list, got {payload!r}")
    groups = []
    for group in payload["groups"]:
        if not isinstance(group, list):
            raise ValueError(f"group is not a list: {group!r}")
        members = []
        for n in group:
            if not isinstance(n, int) or isinstance(n, bool):
                raise ValueError(f"group member is not an integer: {n!r}")
            if not 1 <= n <= count:
                raise ValueError(f"index {n} outside 1..{count}")
            if n not in members:
                members.append(n)
        if len(members) > 1:
            groups.append(sorted(members))
    return groups


def llm_duplicates(client, articles):
    """Ask the model which of the remaining stories are the same news.

    Returns {index to drop: index it duplicates}, or {} if the model
    never answers usably - a failure here must cost us nothing but a
    duplicate in the email.
    """
    listing = "\n\n".join(
        f"{i}. {a['title']}\n{a['description'][:DUP_SNIPPET_CHARS]}"
        for i, a in enumerate(articles, 1)
    )
    prompt = (
        "Below are today's AI news stories, each numbered. Different outlets "
        "sometimes cover the same underlying news. Group the numbers that "
        "report the same news event.\n\n"
        "Answer with JSON and nothing else, in exactly this shape:\n"
        '{"groups": [[2, 7], [4, 5]]}\n'
        "Each inner list holds every number covering one news event. Only "
        "group stories about the same event - stories that merely share a "
        "company or a theme are different stories. If nothing is duplicated, "
        'answer {"groups": []}.\n\n'
        + listing
    )
    reply = ""
    for attempt in range(DUP_RETRIES + 1):
        try:
            resp = client.chat.complete(
                model=DUP_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = resp.choices[0].message.content or ""
            groups = parse_duplicate_groups(reply, len(articles))
            break
        except Exception as e:
            print(f"[warn] duplicate check attempt {attempt + 1} failed: {e}")
            if reply:
                print(f"[warn] raw reply was: {reply!r}")
            if attempt == DUP_RETRIES:
                print("[warn] duplicate check gave up, keeping every story")
                return {}
            time.sleep(2 ** attempt)

    signatures = [story_signature(a) for a in articles]
    # how many of today's stories each term shows up in - a term in lots
    # of them says nothing about which story this is. Two stories is the
    # fewest a shared term can appear in, so it always counts as rare.
    seen_in = collections.Counter(t for sig in signatures for t in sig)
    common_at = max(2, len(articles) * DUP_COMMON_DF_RATIO)

    drops = {}
    for group in groups:
        keep = group[0] - 1
        for n in group[1:]:
            drop = n - 1
            if drop in drops:
                continue
            shared = signatures[keep] & signatures[drop]
            smaller = min(len(signatures[keep]), len(signatures[drop])) or 1
            distinct = sorted(t for t in shared if seen_in[t] <= common_at)
            overlap = len(shared) / smaller
            if len(distinct) < DUP_LLM_MIN_DISTINCT or overlap < DUP_LLM_MIN_CONTAINMENT:
                print(f"[dup/model] ignoring unsupported pair: "
                      f"'{articles[drop]['title']}' vs "
                      f"'{articles[keep]['title']}' "
                      f"(overlap={overlap:.2f}, distinctive={distinct})")
                continue
            drops[drop] = keep
            print(f"[dup/model] '{articles[drop]['title']}' duplicates "
                  f"'{articles[keep]['title']}' "
                  f"(overlap={overlap:.2f}, distinctive={distinct})")
    return drops


def drop_duplicate_stories(client, articles):
    """Different sites cover the same story under different headlines.
    A lexical pass catches the obvious repeats without an api call, then
    the model looks over what is left. Every decision is logged, and on
    any api or parsing problem nothing is dropped."""
    if len(articles) < 2:
        return articles
    print(f"[dup] checking {len(articles)} stories for cross-source duplicates")
    # a pass that wants to delete half the harvest has misfired; a real
    # day never has that much repetition. Discard that pass rather than
    # send a nearly empty email
    limit = len(articles) * DUP_MAX_DROP_RATIO

    dropped = lexical_duplicates(articles)
    if not dropped:
        print("[dup/lexical] no duplicates found")
    elif len(dropped) > limit:
        print(f"[warn] lexical pass wanted to drop {len(dropped)} of "
              f"{len(articles)} stories, that is too many - ignoring it")
        dropped = {}

    survivors = [i for i in range(len(articles)) if i not in dropped]
    if len(survivors) > 1:
        offered = [articles[i] for i in survivors]
        model_drops = llm_duplicates(client, offered)
        if not model_drops:
            print("[dup/model] no further duplicates found")
        proposed = dict(dropped)
        for drop, keep in model_drops.items():
            proposed[survivors[drop]] = survivors[keep]
        if len(proposed) > limit:
            print(f"[warn] duplicate check wanted to drop {len(proposed)} of "
                  f"{len(articles)} stories, that is too many - ignoring the "
                  f"model pass")
        else:
            dropped = proposed

    kept = [a for i, a in enumerate(articles) if i not in dropped]
    print(f"[dup] dropped {len(dropped)} duplicate(s), {len(kept)} stories remain")
    return kept


def title_key(title):
    """First six words of the title, lowercased, punctuation stripped.
    Close-enough fingerprint to catch reposts with minor title edits."""
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return " ".join(words[:6])


def load_seen():
    try:
        with open(SEEN_FILE) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def save_seen(keys):
    with open(SEEN_FILE, "w") as f:
        f.write("\n".join(keys[-SEEN_MAX:]) + "\n")


def build_html(items):
    """Assemble the email body. The source label links to the article
    (urls were validated by safe_link at fetch time); everything from
    feeds or the model is escaped so it can't inject markup."""
    parts = []
    for a in items:
        label = html.escape(a["source"].upper()) + ":"
        summary = html.escape(a["summary"])
        href = html.escape(a["url"], quote=True)
        parts.append(f'<p><a href="{href}">{label}</a> {summary}</p>')
    return "<html><body>" + "\n".join(parts) + "</body></html>"


def send_email(html_body):
    """Send via gmail smtp over ssl. GMAIL_TO may hold several
    comma-separated addresses - they are delivered as bcc so
    recipients don't see each other. To add a subscriber, just
    append their address to GMAIL_TO."""
    user = os.environ["GMAIL_USER"]
    recipients = [r.strip() for r in os.environ["GMAIL_TO"].split(",") if r.strip()]

    msg = MIMEText(html_body, "html")
    msg["Subject"] = f"AI Daily News Harvest — {date.today():%d %b %Y}"
    msg["From"] = user
    msg["To"] = user  # real recipients are bcc'd

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(user, os.environ["GMAIL_APP_PASSWORD"])
        server.sendmail(user, recipients, msg.as_string())


if __name__ == "__main__":
    load_dotenv()
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing env vars: {', '.join(missing)}")

    items = fetch_articles()
    if not items:
        sys.exit("no articles fetched from any feed")

    # skip anything already sent on a previous day
    seen = load_seen()
    fresh = [a for a in items if title_key(a["title"]) not in seen]
    if len(fresh) < len(items):
        print(f"skipped {len(items) - len(fresh)} previously sent article(s)")
    if not fresh:
        print("nothing new today, no email sent")
        sys.exit()

    client = build_client()
    # duplicates dropped here are still remembered below, so the same
    # story can't come back tomorrow through the other site's feed
    candidates = fresh
    fresh = drop_duplicate_stories(client, fresh)
    if not fresh:
        print("nothing new today, no email sent")
        sys.exit()

    for a in fresh:
        a["summary"] = summarize(client, a)

    send_email(build_html(fresh))
    # only remember articles after the send succeeded, so a failed
    # run retries them tomorrow instead of losing them
    save_seen(seen + [title_key(a["title"]) for a in candidates])
    print(f"sent {len(fresh)} articles")
