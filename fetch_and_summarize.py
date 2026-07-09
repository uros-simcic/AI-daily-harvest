#!/usr/bin/env python3
"""Fetch AI news from RSS feeds, summarize with Mistral, email the harvest."""

import re
import sys

import feedparser
import requests

FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "The Rundown AI": "https://rss.beehiiv.com/feeds/2R3C6Bt5wj.xml",
}

ARTICLES_PER_FEED = 4
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


if __name__ == "__main__":
    items = fetch_articles()
    if not items:
        sys.exit("no articles fetched from any feed")
    for a in items:
        print(f"[{a['source']}] {a['title']}\n{a['url']}\n")
