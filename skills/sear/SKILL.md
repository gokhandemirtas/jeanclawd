---
name: sear
description: "Use the local Docker-hosted SearXNG metasearch API for explicit web searches. Trigger when the user starts a request with `sear:`, for example `sear: marvel heroes between 1950 to now`."
---

# SearXNG Search

Use this opt-in skill for searches through the local SearXNG instance. The
runtime strips `sear:`, calls the JSON API, removes duplicate URLs, ranks
results by score, and returns source material for answer synthesis.

Examples:

- `sear: marvel heroes between 1950 to now`
- `sear: latest local planning applications`
- `sear: compare current electric vehicle charging standards`

The service is started with:

```sh
docker compose -f searxng/docker-compose.yml up -d
```

Set `SEARXNG_URL` when the service is not at the default local URL.
