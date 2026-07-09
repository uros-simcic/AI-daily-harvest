#!/usr/bin/env python3
"""Fetch AI news from RSS feeds, summarize with Mistral, email the harvest."""

import os
import re
import sys

import feedparser
import requests
from dotenv import load_dotenv
from mistralai.client import Mistral

FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "The Rundown AI": "https://rss.beehiiv.com/feeds/2R3C6Bt5wj.xml",
}

ARTICLES_PER_FEED = 4
MISTRAL_MODEL = "mistral-small-latest"
HTTP_TIMEOUT = 15
# some news sites 403 requests without a browser-like user agent
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

TAG_RE = re.compile(r"<[^>]+>")


def fetch_articles():
    """Collect recent entries from all feeds.

    A failing feed is logged and skipped so one dead site doesn't
    kill the whole run.
    """
    articles = []
    for source, url in FEEDS.items():
        try:
            # download ourselves: feedparser's own fetching has no timeout
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[warn] {source}: fetch failed: {e}")
            continue
        feed = feedparser.parse(resp.content)
        if feed.bozo and not feed.entries:
            print(f"[warn] {source}: feed did not parse, skipping")
            continue
        for entry in feed.entries[:ARTICLES_PER_FEED]:
            articles.append({
                "source": source,
                "title": entry.get("title", "").strip(),
                "url": entry.get("link", ""),
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


if __name__ == "__main__":
    load_dotenv()
    items = fetch_articles()
    if not items:
        sys.exit("no articles fetched from any feed")
    client = build_client()
    for a in items:
        print(f"{a['source'].upper()}: {summarize(client, a)}\n{a['url']}\n")
