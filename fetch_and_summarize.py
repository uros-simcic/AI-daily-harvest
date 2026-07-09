#!/usr/bin/env python3
"""Fetch AI news from RSS feeds, summarize with Mistral, email the harvest."""

import html
import os
import re
import smtplib
import sys
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
        "feed": "https://venturebeat.com/category/ai/feed/",
        "domain": "venturebeat.com",
    },
    "The Rundown AI": {
        "feed": "https://rss.beehiiv.com/feeds/2R3C6Bt5wj.xml",
        "domain": "therundown.ai",
    },
}

REQUIRED_ENV = ("MISTRAL_API_KEY", "GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO")

ARTICLES_PER_FEED = 4
MISTRAL_MODEL = "mistral-small-latest"
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
    for a in fresh:
        a["summary"] = summarize(client, a)

    send_email(build_html(fresh))
    # only remember articles after the send succeeded, so a failed
    # run retries them tomorrow instead of losing them
    save_seen(seen + [title_key(a["title"]) for a in fresh])
    print(f"sent {len(fresh)} articles")
