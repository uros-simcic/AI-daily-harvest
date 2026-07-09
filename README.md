# AI Daily Harvest

AI news, delivered daily: articles from TechCrunch, VentureBeat and
The Rundown, fetched via RSS, summarized to two sentences each by Mistral,
and delivered as one email every morning. Runs for free on GitHub Actions —
no server needed.

## How it works

1. **Fetch** — pulls the latest entries from each source's RSS feed.
   Feeds are downloaded with a timeout and parsed with feedparser; a dead
   feed is skipped, the others still deliver.
2. **Filter** — article links must be https on the source's own domain,
   anything else is dropped. Articles already sent on a previous day are
   skipped (fingerprint: first six words of the title), so reposts with
   small title edits don't show up twice.
3. **Summarize** — one Mistral call per article turns title + feed
   description into a two-sentence summary. If a call fails, the feed's
   own description is used instead.
4. **Send** — a single email via Gmail. Each entry is a clickable source
   label followed by the summary. Recipients are BCC'd, so a small
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
at 09:00 UTC (GitHub cron is best-effort, so the actual start can drift).
Add the same four variables as repository secrets under
*Settings → Secrets and variables → Actions*, then trigger a first run
manually from the *Actions* tab to check everything works.

The sent-articles list is carried between runs by the Actions cache.

## Adding a subscriber

Append the address to `GMAIL_TO` (comma-separated) in your `.env` and in
the repository secret. Recipients are BCC'd and never see each other.

## License

[MIT](LICENSE)
