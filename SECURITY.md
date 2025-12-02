# Security Policy

This project values responsible disclosure and welcomes reports of security vulnerabilities. Please follow the guidance below to help us investigate and remediate issues efficiently.

## Supported Versions

We actively support and patch the following versions:

- Python runtime: 3.10, 3.11, 3.12
- App components: latest `main` branch of this repository

Older Python versions (<3.10) and forks or derivatives are out of scope.

## Reporting a Vulnerability

- Preferred: Open a private security advisory on GitHub (Security → Advisories → Report a vulnerability) for this repository.
- Alternative: Open a GitHub Issue and mark it clearly with the `security` label. Avoid sharing exploit details publicly.

Please include:

- A clear description of the issue and impact
- Steps to reproduce (minimal PoC script or instructions)
- Affected files, config, and environment details (OS, Python version)
- Potential remediation ideas if known

We will acknowledge receipt within 5 business days and aim to provide a remediation plan or fix within 30 days for high-severity issues.

## Scope and Components

This repository contains:

- `collector.py`: background data collector (GraphQL API → SQLite)
- `app.py`: Streamlit dashboard (consumer)
- `database.py`: SQLite connection, WAL mode, retry logic
- `utils.py`: calculations and business logic

Security-relevant areas include API handling, database interactions, process management, configuration, and Streamlit serving.

## Security Best Practices (Project-Specific)

- Database

  - SQLite runs in WAL mode to allow concurrent reads/writes. Ensure the database file permissions are restricted to the local user account.
  - Always use the `@retry_db_op` decorator for DB operations to mitigate locking issues.
  - Do not expose the SQLite file over shared/network drives without proper access controls.

- Secrets & Config

  - Do not hardcode API tokens or secrets in the repo. Use environment variables or a local, ignored `.env` file.
  - Review `config.py` and shell scripts (`run.ps1`, `run_collector.ps1`) for any sensitive values; avoid committing them.

- Network & API

  - Collector uses a GraphQL API. Validate inputs, handle timeouts, and avoid blindly trusting remote data.
  - Implement strict error handling and logging via `logging` (not `print`). Logs must not contain secrets.

- Streamlit App

  - If serving beyond localhost, place behind a reverse proxy and enable TLS.
  - Limit publicly exposed endpoints and avoid enabling any unsafe Streamlit features.

- Dependencies

  - Keep `requirements.txt` up to date and pin versions when feasible.
  - Run `pip install --upgrade --requirement requirements.txt` regularly and check for known CVEs.

- Windows Execution (scripts)
  - PowerShell scripts (`run.ps1`, `run_collector.ps1`) should be reviewed before execution; avoid running from untrusted directories.
  - Use standard PowerShell execution policies and signed scripts in production environments.

## Vulnerability Classification

We generally assess severity using CVSS v3.1 as a reference and prioritize:

- High/Critical: RCE, data corruption, unauthorized write access, credential disclosure
- Medium: DoS, significant info leak, unauthorized read access
- Low: Non-sensitive info leaks, minor misconfigurations

## Patching and Release Process

- Fixes are developed in a private branch when appropriate, then merged to `main`.
- A changelog entry will note security-impacting changes.
- If warranted, we will publish a GitHub Security Advisory with mitigation guidance.

## Responsible Disclosure

If you discover a vulnerability, please do not publicly disclose it until a fix is available. Coordinate with us via a private security advisory or security-labeled issue. We appreciate sanitized PoCs that demonstrate impact without risking user data.

## Code of Conduct

Reports must be respectful and constructive. Any harmful, exploitative, or abusive behavior will not be tolerated.

---

Thank you for helping keep the project and its users safe.
