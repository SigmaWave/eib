#!/usr/bin/env python3
"""Run a Cursor agent against this workspace via the Cursor SDK."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cursor_sdk import (
    Agent,
    AgentOptions,
    CloudAgentOptions,
    CloudRepository,
    CursorAgentError,
    LocalAgentOptions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Cursor agent and run a prompt against this workspace."
    )
    parser.add_argument("prompt", help="Prompt to send to the agent")
    parser.add_argument(
        "--model",
        default="composer-2.5",
        help="Model id (default: composer-2.5)",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory for a local agent (default: current directory)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CURSOR_API_KEY"),
        help="Cursor API key (default: CURSOR_API_KEY env var)",
    )
    parser.add_argument(
        "--cloud",
        metavar="REPO_URL",
        help="Run in the cloud against REPO_URL instead of locally",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="Git ref for cloud runs (default: main)",
    )
    parser.add_argument(
        "--auto-pr",
        action="store_true",
        help="Open a pull request when the cloud run finishes",
    )
    parser.add_argument(
        "--resume",
        metavar="AGENT_ID",
        help="Resume an existing agent instead of creating a new one",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the final result without streaming assistant text",
    )
    return parser.parse_args()


def build_agent_options(args: argparse.Namespace) -> AgentOptions:
    if not args.api_key:
        print(
            "Missing API key. Set CURSOR_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.cloud:
        return AgentOptions(
            model=args.model,
            api_key=args.api_key,
            cloud=CloudAgentOptions(
                repos=[CloudRepository(url=args.cloud, starting_ref=args.ref)],
                auto_create_pr=args.auto_pr,
                skip_reviewer_request=True,
            ),
        )

    return AgentOptions(
        model=args.model,
        api_key=args.api_key,
        local=LocalAgentOptions(cwd=str(args.cwd.resolve())),
    )


def stream_run(run) -> None:
    for message in run.messages():
        if message.type != "assistant":
            continue
        for block in message.message.content:
            if block.type == "text":
                print(block.text, end="", flush=True)
    print()


def run_prompt(args: argparse.Namespace) -> int:
    options = build_agent_options(args)

    try:
        if args.resume:
            agent_cm = Agent.resume(
                args.resume,
                AgentOptions(api_key=args.api_key),
            )
        else:
            agent_cm = Agent.create(options)

        with agent_cm as agent:
            run = agent.send(args.prompt)
            print(f"agent={agent.agent_id} run={run.id}", file=sys.stderr)

            if not args.no_stream:
                stream_run(run)

            result = run.wait()
    except CursorAgentError as err:
        print(
            f"startup failed: {err.message} (retryable={err.is_retryable})",
            file=sys.stderr,
        )
        return 1

    if result.status == "error":
        print(f"run failed: {result.id}", file=sys.stderr)
        return 2

    if args.no_stream and result.result:
        print(result.result)

    return 0


def main() -> None:
    sys.exit(run_prompt(parse_args()))


if __name__ == "__main__":
    main()
