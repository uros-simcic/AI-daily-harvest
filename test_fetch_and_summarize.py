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


class RundownSectionTests(unittest.TestCase):
    def split(self, body, link="https://therundownai.beehiiv.com/p/example"):
        return harvest.split_issue(
            {"link": link, "content": [{"value": body}]},
            "The Rundown AI", ["therundown.ai", "therundownai.beehiiv.com"])

    def test_news_with_why_it_matters_is_kept(self):
        stories = self.split("<h6>RESEARCH</h6><h4>New model</h4>"
                             "<p>The Rundown: Evidence. Why it matters: A limitation.</p>")
        self.assertEqual([s["title"] for s in stories], ["New model"])

    def test_editorial_teaching_does_not_need_why_it_matters(self):
        for category in ["AI TRAINING", "ROWAN\u2019S CORNER", "NATE'S NOTEBOOK"]:
            with self.subTest(category=category):
                stories = self.split(f"<h6>{category}</h6><h4>Practical lesson</h4>"
                                     "<p>Compare the formula with a known result.</p>")
                self.assertEqual(len(stories), 1)
                self.assertIn("known result", stories[0]["description"])

    def test_sponsors_are_excluded_even_with_why_it_matters(self):
        for category in ["TOGETHER WITH ACME", "PRESENTED BY ACME", "SPONSORED BY ACME"]:
            with self.subTest(category=category):
                self.assertEqual(self.split(f"<h6>{category}</h6><h4>Buy this</h4>"
                                            "<p>Why it matters: Register now.</p>"), [])

    def test_unknown_sections_are_not_treated_as_teaching(self):
        self.assertEqual(self.split("<h6>OTHER</h6><h4>Link roundup</h4>"
                                    "<p>Unclassified material.</p>"), [])

    def test_sections_do_not_leak_into_previous_story(self):
        stories = self.split("<h6>GOOGLE</h6><h4>News</h4>"
                             "<p>Why it matters: Supported detail.</p>"
                             "<h6>COMMUNITY</h6><p>Unrelated community text.</p>"
                             "<h3>Everything else</h3><p>Unrelated roundup.</p>")
        self.assertEqual(len(stories), 1)
        self.assertNotIn("Unrelated", stories[0]["description"])

    def test_unsafe_issue_link_is_rejected(self):
        self.assertEqual(self.split("<h6>AI TRAINING</h6><h4>Lesson</h4>"
                                    "<p>Useful text.</p>", "https://example.com/story"), [])


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

    def test_accepts_a_16_word_lesson(self):
        lesson = " ".join(["word"] * 16)
        self.assertEqual(parse_summaries(takeaways(lesson), 1), [lesson])

    def test_invalid_items_and_overlong_takeaways_are_rejected(self):
        cases = [None, [], {}, {"status": "skip", "reason": "other"},
                 {"status": "keep", "takeaway": "**"},
                 {"status": "keep", "takeaway": "word " * 17}]
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

    def test_repair_only_unresolved_and_keep_source_alignment(self):
        stories = [dict(self.article, title=name) for name in ("A", "B", "C")]
        first = {"items": [{"status": "skip", "reason": "promotion"},
                           {"status": "keep", "takeaway": "word " * 17},
                           {"status": "keep", "takeaway": "Valid first draft."}]}
        repaired = {"items": [{"status": "skip", "reason": "promotion"},
                              {"status": "keep", "takeaway": "Register before prices increase.",
                               "evidence": self.article["description"]}]}
        with patch.object(harvest, "chat_complete", side_effect=[
            completion(json.dumps(first)), completion(json.dumps(repaired))
        ]) as request:
            self.assertEqual(harvest.summarize_all(Mock(), stories),
                             [None, "Register before prices increase.", "Valid first draft."])
        inputs = json.loads(request.call_args.args[2][1]["content"])
        self.assertEqual([x["title"] for x in inputs], ["A", "B"])
        self.assertEqual(inputs[1]["previous_draft"], "word " * 17)

    def test_fabricated_repair_evidence_is_excluded(self):
        first = completion(takeaways("Valid first draft.", "word " * 17))
        repair = completion(json.dumps({"items": [{"status": "keep", "takeaway": "New.",
                                                  "evidence": "Invented source."}]}))
        with patch.object(harvest, "chat_complete", side_effect=[first, repair]):
            self.assertEqual(harvest.summarize_all(Mock(), [self.article] * 2),
                             ["Valid first draft.", None])

    def test_repair_outage_preserves_valid_first_draft(self):
        with patch.object(harvest, "chat_complete", side_effect=[
            completion(takeaways("Valid first draft.", "word " * 17)), RuntimeError("offline")
        ]):
            self.assertEqual(harvest.summarize_all(Mock(), [self.article] * 2),
                             ["Valid first draft.", None])


class HarvestEntryPointTests(unittest.TestCase):
    def run_harvest(self, replies, error=None, preview=False, all_skipped=False, partial=False):
        """Run the real script using fake feeds/provider/mail and isolated state."""
        script = Path(harvest.__file__).resolve()
        articles = [
            ("Alpha inference memory research", "A supported source about inference memory. " * 20),
            ("Beta robot actuator release", "A different source about robot actuators. " * 20),
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

        def respond(*, model, messages, max_tokens):
            self.assertEqual(max_tokens, 2048)
            if messages[0]["content"] == harvest.GROUNDING_INSTRUCTIONS:
                count = len(json.loads(messages[1]["content"]))
                return completion(json.dumps({"supported": [True] * count}))
            if messages[0]["content"] not in (harvest.TAKEAWAY_INSTRUCTIONS, harvest.REPAIR_INSTRUCTIONS):
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
                        "GMAIL_TO": ("recipient@example.invalid,rejected@example.invalid"
                                     if partial else "recipient@example.invalid"),
                        "HARVEST_PREVIEW": "1" if preview else "0",
                    }, clear=True))
                    stack.enter_context(patch("dotenv.load_dotenv"))
                    stack.enter_context(patch("requests.get", side_effect=feed_response))
                    stack.enter_context(patch("mistralai.client.Mistral", return_value=provider))
                    stack.enter_context(patch("time.sleep"))
                    smtp = stack.enter_context(patch("smtplib.SMTP_SSL"))
                    smtp.return_value.__enter__.return_value.sendmail.return_value = (
                        {"rejected@example.invalid": (550, b"refused")} if partial else {})
                    printed = io.StringIO()
                    stack.enter_context(contextlib.redirect_stdout(printed))
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
                        if partial:
                            self.assertIn("partial_delivery", printed.getvalue())
                            self.assertNotIn("sent 2 articles", printed.getvalue())
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
            RuntimeError("provider unavailable"),
        ], error="digest was not sent")
        self.assertEqual(calls, 2)

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
        self.assertEqual(self.run_harvest([completion(reply), completion(reply)], all_skipped=True), 2)

    def test_partial_batch_reports_partial_and_preserves_existing_history_policy(self):
        self.assertEqual(self.run_harvest([completion(takeaways(
            "A supported explanation for the first item.",
            "A supported explanation for the second item.",
        ))], partial=True), 1)

    def test_valid_batch_still_sends_and_remembers_articles(self):
        calls = self.run_harvest([
            completion(takeaways(
                "A supported explanation for the first item.",
                "A supported explanation for the second item.",
            )),
        ])
        self.assertEqual(calls, 1)


class GroundingTests(unittest.TestCase):
    def test_rejected_and_skipped_items_keep_alignment(self):
        articles = [{"title": "Test", "description": "Source"}] * 3
        with patch.object(harvest, "chat_complete", return_value=completion(
                '{"supported": [false, true]}')):
            self.assertEqual(harvest.validate_takeaways(
                Mock(), articles, ["Unsupported", None, "Supported"]),
                [None, None, "Supported"])

    def test_no_candidates_avoids_provider_call(self):
        with patch.object(harvest, "chat_complete") as call:
            self.assertEqual(harvest.validate_takeaways(
                Mock(), [{"title": "Test", "description": "Source"}], [None]), [None])
            call.assert_not_called()

    def test_malformed_verdicts_stop_delivery(self):
        for reply in ('{}', '{"supported": []}', '{"supported": ["true"]}', 'bad'):
            with self.subTest(reply=reply), patch.object(
                    harvest, "chat_complete", return_value=completion(reply)):
                with self.assertRaisesRegex(RuntimeError, "digest was not sent"):
                    harvest.validate_takeaways(Mock(),
                        [{"title": "Test", "description": "Source"}], ["Lesson"])


class SourceEvidenceTests(unittest.TestCase):
    def test_article_parser_excludes_navigation_and_scripts(self):
        parser = harvest.ArticleTextParser()
        parser.feed('<nav><p>Wrong</p></nav><div class="entry-content"><p>Evidence '
                    '<strong>here</strong>.</p><script>Bad</script></div><p>Outside</p>')
        self.assertEqual(" ".join("".join(parser.parts).split()), "Evidence here.")

    def test_rejects_invented_chip_and_benchmark(self):
        article = {"title": "GPU study", "description": "Blackwell trained the models."}
        self.assertTrue(harvest.has_unsupported_identifiers("It used A100 chips.", article))
        self.assertTrue(harvest.has_unsupported_identifiers("It was 19% faster.", article))
        self.assertFalse(harvest.has_unsupported_identifiers("Blackwell was used.", article))

    def test_preserves_supported_number(self):
        self.assertFalse(harvest.has_unsupported_identifiers("It was 19% faster.",
            {"title": "Study", "description": "The model was 19% faster."}))

    def test_unavailable_thin_article_keeps_feed_text(self):
        article = {"source": "TechCrunch AI", "title": "Study",
                   "url": "https://techcrunch.com/study", "description": "Thin source"}
        with patch.object(harvest.requests, "get", side_effect=harvest.requests.ConnectionError):
            self.assertEqual(harvest.enrich_sources([article]), [article])

    def test_long_feed_avoids_article_request(self):
        article = {"source": "TechCrunch AI", "title": "Study",
                   "url": "https://techcrunch.com/study", "description": "Evidence " * 100}
        with patch.object(harvest.requests, "get") as request:
            self.assertEqual(harvest.enrich_sources([article]), [article])
            request.assert_not_called()

    def test_numeric_rejection_precedes_ai_check(self):
        with patch.object(harvest, "chat_complete") as request:
            self.assertEqual(harvest.validate_takeaways(Mock(),
                [{"title": "Study", "description": "No figures supplied"}], ["19% faster"]), [None])
            request.assert_not_called()


class SmtpOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {
            "GMAIL_USER": "sender@example.invalid", "GMAIL_APP_PASSWORD": "test-only",
            "GMAIL_TO": "one@example.invalid,two@example.invalid",
        }).start()
        self.smtp = patch.object(harvest.smtplib, "SMTP_SSL").start()
        self.mail = self.smtp.return_value.__enter__.return_value

    def test_full_acceptance_is_counted(self):
        self.mail.sendmail.return_value = {}
        self.assertEqual(harvest.send_email("<p>Lesson</p>"), (2, 0))
        self.mail.sendmail.assert_called_once()

    def test_partial_refusals_are_counted(self):
        self.mail.sendmail.return_value = {"two@example.invalid": (550, b"refused")}
        self.assertEqual(harvest.send_email("<p>Lesson</p>"), (1, 1))

    def test_timeout_is_uncertain_and_does_not_expose_details(self):
        self.mail.sendmail.side_effect = TimeoutError("one@example.invalid test-only")
        with self.assertRaisesRegex(RuntimeError, "smtp_uncertain") as stopped:
            harvest.send_email("<p>Lesson</p>")
        self.assertNotIn("example.invalid", str(stopped.exception))
        self.assertNotIn("test-only", str(stopped.exception))
        self.mail.sendmail.assert_called_once()

    def test_all_refusals_are_sanitized(self):
        self.mail.sendmail.side_effect = harvest.smtplib.SMTPRecipientsRefused({
            "one@example.invalid": (550, b"private server details")})
        with self.assertRaisesRegex(RuntimeError, "smtp_rejected") as stopped:
            harvest.send_email("<p>Lesson</p>")
        self.assertNotIn("example.invalid", str(stopped.exception))

    def test_login_failure_is_not_sent(self):
        self.mail.login.side_effect = harvest.smtplib.SMTPAuthenticationError(535, b"private")
        with self.assertRaisesRegex(RuntimeError, "smtp_not_sent"):
            harvest.send_email("<p>Lesson</p>")
        self.mail.sendmail.assert_not_called()

    def test_cleanup_failure_preserves_confirmed_acceptance(self):
        self.mail.sendmail.return_value = {}
        self.smtp.return_value.__exit__.side_effect = harvest.smtplib.SMTPServerDisconnected("private")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(harvest.send_email("<p>Lesson</p>"), (2, 0))

    def test_data_rejection_is_definite(self):
        self.mail.sendmail.side_effect = harvest.smtplib.SMTPDataError(550, b"private")
        with self.assertRaisesRegex(RuntimeError, "smtp_rejected"):
            harvest.send_email("<p>Lesson</p>")


if __name__ == "__main__":
    unittest.main()
