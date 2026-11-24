# Contributing

Thanks for your interest in Motion Lab. Contributions should keep the project local-first, make behavior testable without a camera where practical, and avoid adding data collection or network services without discussion.

## Before you start

- Open an issue to discuss a substantial change. For small fixes, a pull request is welcome.
- Do not include personal data, private recordings, credentials, or generated datasets.
- Explain any privacy, network, storage, or model-download behavior your change introduces.

## Development

Requirements: Python 3.10–3.12 and uv.

```bash
uv sync --group dev --extra vision --extra web --extra ml
uv run pytest -q
```

Tests should use synthetic or mocked inputs and run without a webcam or downloaded model when possible. Describe any manual hardware checks separately. Keep changes focused and update relevant documentation.

## Pull requests

Include the problem, approach, privacy/storage/network impact, and verification performed. Link related issues and note any known limitations. A maintainer will review before merging.
