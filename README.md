# AI Daily Harvest

[![Daily News Harvest](https://github.com/uros-simcic/AI-daily-harvest/actions/workflows/daily-news-harvest.yml/badge.svg)](https://github.com/uros-simcic/AI-daily-harvest/actions/workflows/daily-news-harvest.yml)

AI news from TechCrunch, VentureBeat and The Rundown, delivered as short,
practical lessons in one daily email.

The app fetches recent articles, prepares the source text, generates teaching
takeaways with Mistral, checks factual support and removes duplicate stories.
Each takeaway is limited to 16 words. Duplicate checks use word overlap and
the same Mistral model.

Runs for free on GitHub Actions.

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

Select the manual workflow preview input to generate takeaways for review.
Locally, set `HARVEST_PREVIEW=1`; only `MISTRAL_API_KEY` is required.
Preview HTML appears in workflow output.

### SMTP outcomes

Logs report SMTP acceptance counts and distinguish partial refusal, rejection
and uncertain delivery. Article history is updated after SMTP accepts at least
one recipient. Exclusion logs identify why individual articles were omitted.
