# AI Daily Harvest

[![Daily News Harvest](https://github.com/uros-simcic/AI-daily-harvest/actions/workflows/daily-news-harvest.yml/badge.svg)](https://github.com/uros-simcic/AI-daily-harvest/actions/workflows/daily-news-harvest.yml)

AI news, delivered daily: articles from TechCrunch, VentureBeat and
The Rundown, fetched via RSS, turned into short teaching takeaways by Mistral,
and delivered as one email every morning. Runs for free on GitHub Actions —
no server needed.

## How it works

1. **Fetch** — pulls the latest entries from each source's RSS feed.
   Feeds are downloaded with a timeout and parsed with feedparser; a dead
   feed is skipped, the others still deliver. VentureBeat's own feed is
   blocked from GitHub Actions, so that source falls back to Bing's
   VentureBeat RSS (the emailed links still point at venturebeat.com).
   The Rundown's posts are accepted from both therundown.ai and
   Beehiiv, which is where those issues are published now.
2. **Filter** — article links must be https on the source's own domain,
   anything else is dropped. Articles already sent on a previous day are
   skipped (fingerprint: first six words of the title), so reposts with
   small title edits don't show up twice.
3. **Prepare** — generate eligible takeaways as described below, then deduplicate.
   For deduplication, the three sources often run the same story under
   different headlines. A word-overlap pass catches the obvious repeats
   with no API call at all; a single call to a larger model then reviews
   whatever is left, and its verdict counts only for pairs the wording
   already supports. The first listed copy of a story is the one kept.
   If the check fails nothing is dropped, and a pass that would delete
   half the harvest is thrown out rather than mail a near-empty email.
4. **Teach** — Mistral turns each supported source into a concrete teaching
   point in one short sentence of at most 28 words. Promotions and sources too thin
   to support a useful explanation are excluded. Eligible stories then
   pass through the duplicate checks above before sending.
   Calls retain the 35-second spacing and bounded 429 retries.
   Persistent batch rate limiting stops the run without individual calls.
   Other unusable batch responses may be retried per item. Any generation
   failure stops the digest before sending or saving article history;
   raw feed descriptions are never substituted. If every item is excluded,
   the run reports `no_eligible_content` and sends no email. Excluded items
   are not marked sent and may be reconsidered on a later run.
5. **Send** — a single email via Gmail. Each entry is a clickable source
   label followed by the teaching takeaway. Recipients are BCC'd, so a small
   subscriber list works out of the box.

## Setup

```bash
git clone git@github.com:uros-simcic/AI-daily-harvest.git
cd AI-daily-harvest
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your values
```

| Variable             | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| `MISTRAL_API_KEY`    | from [console.mistral.ai](https://console.mistral.ai)              |
| `GMAIL_USER`         | Gmail address that sends the email                                 |
| `GMAIL_APP_PASSWORD` | app password from [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) — requires 2-step verification, never your real password |
| `GMAIL_TO`           | recipient address(es), comma-separated                             |

Test it:

```bash
python fetch_and_summarize.py
```

Running it again right away should print `nothing new today` — already
sent articles are remembered in `seen_titles.txt`.

## Daily schedule

The workflow in `.github/workflows/daily-news-harvest.yml` runs every day
at 05:19 UTC (GitHub cron is best-effort, so the actual start can drift).
Changes to that workflow file must land as a merge commit by the repo
owner. Squash or a GitHub App merge rebinds the schedule actor and
the daily job stops.
Add the same four variables as repository secrets under
*Settings → Secrets and variables → Actions*, then trigger a first run
manually from the *Actions* tab to check everything works.

The sent-articles list is carried between runs by the Actions cache.

## Adding a subscriber

Append the address to `GMAIL_TO` (comma-separated) in your `.env` and in
the repository secret. Recipients are BCC'd and never see each other.

## See also

[yt-weekly-review](https://github.com/uros-simcic/yt-weekly-review) —
YouTube channels, reviewed weekly: three timestamped key takes per new
video, one email every Sunday.

## License

[MIT](LICENSE)

## Preview without sending

Select the manual workflow preview input to generate takeaways without SMTP or updating sent history. Locally, set HARVEST_PREVIEW=1; only MISTRAL_API_KEY is required. Preview HTML appears in workflow output. A provider failure stops the run instead of substituting feed descriptions. Takeaways are limited to 28 words and must be grounded in the supplied source; unsuitable items are skipped.
