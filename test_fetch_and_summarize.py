#!/usr/bin/env python3
"""Regression checks for feed fetching and link validation."""

import unittest

from fetch_and_summarize import looks_like_feed, safe_link, unwrap_article_url


class SafeLinkTests(unittest.TestCase):
    def test_techcrunch_https(self):
        self.assertTrue(safe_link(
            "https://techcrunch.com/2026/09/16/example/",
            ["techcrunch.com"],
        ))

    def test_rundown_beehiiv_host(self):
        self.assertTrue(safe_link(
            "https://therundownai.beehiiv.com/p/zuck-sits-out-the-ai-slowdown",
            ["therundown.ai", "therundownai.beehiiv.com"],
        ))

    def test_rundown_canonical_host_still_ok(self):
        self.assertTrue(safe_link(
            "https://www.therundown.ai/p/example",
            ["therundown.ai", "therundownai.beehiiv.com"],
        ))

    def test_rejects_foreign_host(self):
        self.assertFalse(safe_link(
            "https://evil.example/phish",
            ["therundown.ai", "therundownai.beehiiv.com"],
        ))

    def test_rejects_http(self):
        self.assertFalse(safe_link("http://techcrunch.com/x", ["techcrunch.com"]))


class UnwrapTests(unittest.TestCase):
    def test_bing_apiclick_extracts_venturebeat(self):
        wrapped = (
            "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&"
            "url=https%3a%2f%2fventurebeat.com%2ftechnology%2fclaude-docs"
            "&c=1"
        )
        self.assertEqual(
            unwrap_article_url(wrapped),
            "https://venturebeat.com/technology/claude-docs",
        )
        self.assertTrue(safe_link(unwrap_article_url(wrapped), ["venturebeat.com"]))

    def test_plain_url_unchanged(self):
        url = "https://venturebeat.com/ai/hello"
        self.assertEqual(unwrap_article_url(url), url)


class LooksLikeFeedTests(unittest.TestCase):
    def test_rss_ok(self):
        body = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        self.assertTrue(looks_like_feed(body, "application/rss+xml; charset=UTF-8"))

    def test_html_challenge_rejected(self):
        body = b'<!DOCTYPE html><html lang="en"><head><title>Vercel Security Checkpoint</title>'
        self.assertFalse(looks_like_feed(body, "text/html; charset=utf-8"))


if __name__ == "__main__":
    unittest.main()
