# Set up

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run a Cursor agent

The `run_agent.py` script uses the [Cursor Python SDK](https://cursor.com/docs/sdk/python). It requires **Python 3.10+** (separate from the project venv if that venv is older).

```bash
export CURSOR_API_KEY="cursor_..."
pip install -r requirements-sdk.txt
python3.10 run_agent.py "Summarize what this repository does"
```

Useful flags:

- `--cwd /path/to/repo` — local workspace (default: current directory)
- `--cloud https://github.com/org/repo` — run in Cursor cloud
- `--resume <agent-id>` — continue an existing agent
- `--no-stream` — wait for the final result only