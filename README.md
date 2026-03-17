# KhunDech 1.0.0

> This project is vibe coding 99.99%.

KhunDech is a Discord AI assistant bot focused on automation, financial analysis, and self-learning workflows.

## What this project can do

- Chat in Discord with Gemini-powered responses and memory context.
- Manage recurring automation tasks (cron-style) with natural language and commands.
- Fetch and analyze SET stock financial data (`!setfinancial`, `!setbalance`).
- Fetch global stock data with yfinance (`!stock <TICKER>`).
- Run auto-learning jobs that continuously build knowledge files.
- Separate channels for main bot messages, cron/task outputs, and learning outputs.
- Execute Docker/runtime diagnostics from bot commands (`!runtime`, `!docker`, `!compose`).
- Perform controlled self-improvement operations (`!improve`, `!upgrade`).

## Project structure

- `source/bot.py` — main Discord bot entrypoint and command routing.
- `source/khundech/` — core modules (config, persistence, scheduler, finance, self-learning, etc.).
- `source/data/` — runtime data (tasks, memory, history, knowledge files).
- `docker-compose.yml` — deployment configuration.
- `.env` / `.env.example` — secret and environment configuration.

## Tech stack

- Python 3.12
- discord.py
- google-genai (Gemini)
- pandas
- yfinance
- Docker / Docker Compose

## Environment setup

1. Copy `.env.example` to `.env`.
2. Fill all values in `.env`:
   - `DISCORD_TOKEN`
   - `GEMINI_API_KEY`
   - `ALLOWED_USER_ID`
   - `MAIN_CHANNEL_ID`
   - `CRON_CHANNEL_ID`
   - `LEARNING_CHANNEL_ID`
   - `AI_MODEL`
   - `DISCORD_APP_ID`
   - `GUILD_ID`

> Important: Never commit `.env` to GitHub.

## Run with Docker

```bash
docker compose up --build -d
docker compose logs --tail 120 khundech
```

## Common bot capabilities (examples)

- `how many auto lern we have`
- `add more autolern ...`
- `how many cronjob we have`
- `!setfinancial AOT`
- `!stock TSLA`
- `!autolearn`
- `!klearn set_financial`

## Security notes

- Keep tokens and API keys only in `.env`.
- Rotate secrets immediately if they were ever exposed.
- `.gitignore` is configured to prevent committing environment secrets.

## Version

Current release: **KhunDech 1.0.0**

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) file for details.

You are free to:
- Use this software for any purpose
- Redistribute it
- Modify it

Under the condition that:
- Any modifications or distributions must also be licensed under GPLv3
- Include a copy of the license in your distribution
- Include a statement of changes you made
- Disclose the source code
