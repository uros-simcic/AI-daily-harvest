#!/usr/bin/env python3
"""Regression checks for feed fetching and link validation."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import fetch_and_summarize as harvest

from fetch_and_summarize import (
    DUP_MODEL,
    MIN_REQUEST_GAP,
    is_rate_limited,
    looks_like_feed,
    parse_summaries,
    safe_link,
    unwrap_article_url,
)


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


class SummaryParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(
            parse_summaries(takeaways("one", "two"), 2),
            ["one", "two"],
        )

    def test_fenced_json(self):
        reply = "```json\n" + takeaways("a") + "\n```"
        self.assertEqual(parse_summaries(reply, 1), ["a"])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            parse_summaries(takeaways("only-one"), 2)


    def test_skip_results_keep_input_alignment(self):
        reply = json.dumps({"items": [
            {"status": "skip", "reason": "promotion"},
            {"status": "keep", "takeaway": "Cached inputs and generated outputs have separate pricing."},
        ]})
        self.assertEqual(parse_summaries(reply, 2), [None, "Cached inputs and generated outputs have separate pricing."])

    def test_invalid_items_and_overlong_takeaways_are_rejected(self):
        cases = [None, [], {}, {"status": "skip", "reason": "other"},
                 {"status": "keep", "takeaway": "**"},
                 {"status": "keep", "takeaway": "word " * 56}]
        for item in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    parse_summaries(json.dumps({"items": [item]}), 1)


class RateLimitTests(unittest.TestCase):
    def test_mistral_429_body(self):
        err = Exception(
            'API error occurred: Status 429. Body: {"type":"rate_limited","code":"1300"}'
        )
        self.assertTrue(is_rate_limited(err))

    def test_other_errors_not_rate_limits(self):
        self.assertFalse(is_rate_limited(Exception("timeout connecting to api")))

    def test_gap_clears_one_request_per_second(self):
        self.assertGreaterEqual(MIN_REQUEST_GAP, 2.0)

    def test_duplicate_model_is_pinned(self):
        self.assertFalse(DUP_MODEL.endswith("-latest"))


def takeaways(*texts):
    return json.dumps({"items": [{"status": "keep", "takeaway": text} for text in texts]})


def completion(text):
    return SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=text)),
    ])


class GenerationFailureTests(unittest.TestCase):
    def setUp(self):
        self.article = {
            "title": "Founder Summit tickets",
            "description": "Register before prices increase at the door.",
        }

    def test_failed_single_generation_does_not_return_feed_copy(self):
        output = io.StringIO()
        with patch.object(harvest, "chat_complete", side_effect=RuntimeError("provider detail")):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "digest was not sent"):
                    harvest.summarize(Mock(), self.article)
        self.assertNotIn("provider detail", output.getvalue())
        self.assertNotIn(self.article["title"], output.getvalue())

    def test_empty_single_generation_does_not_return_feed_copy(self):
        with patch.object(harvest, "chat_complete", return_value=completion("  ")):
            with self.assertRaisesRegex(RuntimeError, "digest was not sent"):
                harvest.summarize(Mock(), self.article)

    def test_persistent_batch_rate_limit_does_not_start_individual_requests(self):
        output = io.StringIO()
        with patch.object(harvest, "chat_complete", side_effect=RuntimeError("429 provider detail")):
            with patch.object(harvest, "summarize") as individual:
                with contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(RuntimeError, "remains rate-limited"):
                        harvest.summarize_all(Mock(), [self.article, self.article])
                individual.assert_not_called()
        self.assertNotIn("provider detail", output.getvalue())

    def test_malformed_batch_can_still_recover_with_valid_individual_output(self):
        with patch.object(harvest, "chat_complete", side_effect=[
            completion("not JSON"), completion(takeaways("A supported explanation.")),
        ]) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                result = harvest.summarize_all(Mock(), [self.article])
        self.assertEqual(result, ["A supported explanation."])
        self.assertEqual(request.call_count, 2)


class HarvestEntryPointTests(unittest.TestCase):
    def run_harvest(self, replies, error=None, preview=False, all_skipped=False):
        """Run the real script using fake feeds/provider/mail and isolated state."""
        script = Path(harvest.__file__).resolve()
        articles = [
            ("Alpha inference memory research", "A supported source about inference memory."),
            ("Beta robot actuator release", "A different source about robot actuators."),
        ]
        rss = ("<rss version='2.0'><channel>" + "".join(
            f"<item><title>{title}</title>"
            f"<link>https://techcrunch.com/example-{i}/</link>"
            f"<description>{description}</description></item>"
            for i, (title, description) in enumerate(articles)
        ) + "</channel></rss>").encode()
        empty_rss = b"<rss version='2.0'><channel></channel></rss>"

        def feed_response(url, **kwargs):
            return SimpleNamespace(
                status_code=200, headers={"content-type": "application/rss+xml"},
                content=rss if url == harvest.FEEDS["TechCrunch AI"]["feed"] else empty_rss,
            )

        provider = Mock()
        generation_replies = Mock(side_effect=replies)

        def respond(*, model, messages):
            if model == harvest.DUP_MODEL:
                return completion('{"groups": []}')
            return generation_replies()

        provider.chat.complete.side_effect = respond
        original_dir = Path.cwd()
        initial_seen = b"an older article already delivered\n"
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / harvest.SEEN_FILE
            state.write_bytes(initial_seen)
            try:
                os.chdir(directory)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, {
                        "MISTRAL_API_KEY": "test-only",
                        "GMAIL_USER": "sender@example.invalid",
                        "GMAIL_APP_PASSWORD": "test-only",
                        "GMAIL_TO": "recipient@example.invalid",
                        "HARVEST_PREVIEW": "1" if preview else "0",
                    }, clear=True))
                    stack.enter_context(patch("dotenv.load_dotenv"))
                    stack.enter_context(patch("requests.get", side_effect=feed_response))
                    stack.enter_context(patch("mistralai.client.Mistral", return_value=provider))
                    stack.enter_context(patch("time.sleep"))
                    smtp = stack.enter_context(patch("smtplib.SMTP_SSL"))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error):
                            runpy.run_path(str(script), run_name="__main__")
                        smtp.assert_not_called()
                        self.assertEqual(state.read_bytes(), initial_seen)
                    else:
                        if preview or all_skipped:
                            with self.assertRaises(SystemExit) as stopped:
                                runpy.run_path(str(script), run_name="__main__")
                            self.assertIsNone(stopped.exception.code)
                            smtp.assert_not_called()
                            self.assertEqual(state.read_bytes(), initial_seen)
                            return generation_replies.call_count
                        runpy.run_path(str(script), run_name="__main__")
                        mail = smtp.return_value.__enter__.return_value
                        mail.sendmail.assert_called_once()
                        self.assertEqual(state.read_text().splitlines(), [
                            initial_seen.decode().strip(),
                            *[harvest.title_key(title) for title, _ in articles],
                        ])
            finally:
                os.chdir(original_dir)
        return generation_replies.call_count

    def test_batch_429_exhausts_only_one_retry_budget_without_send_or_state_change(self):
        attempts = harvest.RATE_LIMIT_RETRIES + 1
        calls = self.run_harvest(
            [RuntimeError("429 rate limited")] * attempts,
            error="remains rate-limited",
        )
        self.assertEqual(calls, attempts)

    def test_partial_generation_then_failure_sends_nothing_and_preserves_history(self):
        calls = self.run_harvest([
            completion("not JSON"),
            completion(takeaways("A supported explanation for the first item.")),
            RuntimeError("provider unavailable"),
        ], error="digest was not sent")
        self.assertEqual(calls, 3)

    def test_empty_individual_result_sends_nothing_and_preserves_history(self):
        calls = self.run_harvest([
            completion("not JSON"), completion("  "),
        ], error="digest was not sent")
        self.assertEqual(calls, 2)

    def test_preview_never_sends_or_updates_history(self):
        self.assertEqual(self.run_harvest([completion(takeaways(
            "Tool approval lets a person inspect a proposed change before execution.",
            "Benchmark rankings depend on the tasks and weights in their evaluations.",
        ))], preview=True), 1)

    def test_promotions_and_insufficient_sources_send_nothing(self):
        reply = json.dumps({"items": [
            {"status": "skip", "reason": "promotion"},
            {"status": "skip", "reason": "insufficient_source"},
        ]})
        self.assertEqual(self.run_harvest([completion(reply)], all_skipped=True), 1)

    def test_valid_batch_still_sends_and_remembers_articles(self):
        calls = self.run_harvest([
            completion(takeaways(
                "A supported explanation for the first item.",
                "A supported explanation for the second item.",
            )),
        ])
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
