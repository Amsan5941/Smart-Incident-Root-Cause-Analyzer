#!/usr/bin/env python3
"""
Slack Bot Integration for Incident Root Cause Analyzer
-------------------------------------------------------
Slash command: /analyze-incident [paste logs here]
Direct mention: @IncidentBot analyze [paste logs]

Deploy: Set SLACK_BOT_TOKEN + SLACK_SIGNING_SECRET environment variables.
Install slack-bolt: pip install slack-bolt

Usage:
    python slack_bot.py               # Start bot server (port 3000)
    python slack_bot.py --test        # Test with sample incident
"""

import argparse
import json
import logging
import os
import re
import sys

import httpx

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ANALYZER_API_URL = os.environ.get("ANALYZER_API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# API Client
# ─────────────────────────────────────────────────────────────────────────────

def analyze_incident(logs: str, metrics: str = "", error_trace: str = "", service: str = "") -> dict:
    """Call the analyzer API and return the result."""
    response = httpx.post(
        f"{ANALYZER_API_URL}/analyze",
        json={"logs": logs, "metrics": metrics, "error_trace": error_trace, "service": service or None},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def parse_incident_text(text: str) -> dict:
    """
    Parse user-pasted incident text into structured fields.
    Supports key=value sections or plain log dump.
    """
    result = {"logs": "", "metrics": "", "error_trace": "", "service": ""}

    # Try to extract sections by headers
    sections = re.split(r"\n(?=SERVICE:|LOGS:|METRICS:|ERROR TRACE:|STACK TRACE:)", text, flags=re.IGNORECASE)
    for section in sections:
        if re.match(r"SERVICE:", section, re.IGNORECASE):
            result["service"] = section.split(":", 1)[1].strip().split("\n")[0]
        elif re.match(r"LOGS:", section, re.IGNORECASE):
            result["logs"] = section.split(":", 1)[1].strip()
        elif re.match(r"METRICS:", section, re.IGNORECASE):
            result["metrics"] = section.split(":", 1)[1].strip()
        elif re.match(r"(ERROR TRACE:|STACK TRACE:)", section, re.IGNORECASE):
            result["error_trace"] = re.split(r":", section, 1)[1].strip()

    # If no sections found, treat the whole text as logs
    if not result["logs"]:
        result["logs"] = text.strip()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Slack Block Kit message builder
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_EMOJI = {
    "database_n_plus_one": ":repeat:",
    "database_connection_pool": ":swimming_man:",
    "memory_leak": ":droplet:",
    "cpu_hot_loop": ":fire:",
    "disk_full": ":floppy_disk:",
    "database_missing_index": ":card_index:",
    "kubernetes_oom": ":boom:",
    "cache_stampede": ":elephant:",
    "database_deadlock": ":lock:",
    "service_timeout_cascade": ":hourglass:",
    "certificate_expiry": ":closed_lock_with_key:",
    "rate_limit_breach": ":no_entry:",
    "config_error": ":gear:",
    "network_partition": ":broken_heart:",
    "application_error": ":bug:",
    "unknown": ":question:",
}


def build_analysis_blocks(result: dict, user_id: str = None) -> list:
    """Build Slack Block Kit payload from analysis result."""
    confidence = result.get("confidence", 0)
    confidence_pct = int(confidence * 100)
    category = result.get("category", "unknown")
    emoji = CATEGORY_EMOJI.get(category, ":mag:")

    # Confidence bar
    bar_filled = "█" * (confidence_pct // 10)
    bar_empty = "░" * (10 - confidence_pct // 10)
    conf_bar = f"{bar_filled}{bar_empty} {confidence_pct}%"

    # Fix steps (first 4 max)
    fix_steps = result.get("fix_steps", [])[:4]
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(fix_steps))

    # Similar incidents
    similar = result.get("similar_incidents", [])[:3]
    similar_text = ""
    if similar:
        similar_text = "\n".join(
            f"• `{s.get('incident_id')}` — {s.get('root_cause', '')[:60]}"
            for s in similar
        )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji} Incident Root Cause Analysis"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Root Cause:*\n>{result.get('root_cause', 'Unknown')}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Confidence:*\n`{conf_bar}`"},
                {"type": "mrkdwn", "text": f"*Category:*\n`{category}`"},
            ],
        },
    ]

    if steps_text:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Resolution Steps:*\n{steps_text}"},
        })

    if similar_text:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Similar Past Incidents:*\n{similar_text}"},
        })

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    f"Request ID: `{result.get('request_id', 'n/a')}` | "
                    f"Model: `{result.get('model_used', 'n/a')}` | "
                    f"⚡ {result.get('inference_time_ms', 0)}ms"
                    + (f" | Requested by <@{user_id}>" if user_id else "")
                ),
            }
        ],
    })

    # Feedback buttons
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "✅ Correct"},
                "style": "primary",
                "action_id": f"feedback_correct_{result.get('request_id')}",
                "value": json.dumps({"request_id": result.get("request_id"), "correct": True}),
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "❌ Incorrect"},
                "style": "danger",
                "action_id": f"feedback_incorrect_{result.get('request_id')}",
                "value": json.dumps({"request_id": result.get("request_id"), "correct": False}),
            },
        ],
    })

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Slack Bolt App (requires slack-bolt library)
# ─────────────────────────────────────────────────────────────────────────────

def create_slack_app():
    try:
        from slack_bolt import App
        from slack_bolt.adapter.flask import SlackRequestHandler
    except ImportError:
        logger.error("slack-bolt not installed. Run: pip install slack-bolt flask")
        sys.exit(1)

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

    if not bot_token or not signing_secret:
        logger.error("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required")
        sys.exit(1)

    app = App(token=bot_token, signing_secret=signing_secret)

    @app.command("/analyze-incident")
    def handle_slash_command(ack, command, say, logger):
        ack()  # Acknowledge within 3 seconds
        text = command.get("text", "").strip()
        user_id = command.get("user_id")

        if not text:
            say("Please paste incident logs after the command:\n`/analyze-incident [paste logs here]`")
            return

        say(f"<@{user_id}> Analyzing incident... ⏳")

        try:
            fields = parse_incident_text(text)
            result = analyze_incident(**fields)
            blocks = build_analysis_blocks(result, user_id)
            say(blocks=blocks)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            say(f"❌ Analysis failed: {str(e)[:200]}")

    @app.event("app_mention")
    def handle_mention(event, say, logger):
        text = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
        user_id = event.get("user")

        if not text:
            say("Mention me with incident logs to analyze:\n`@IncidentBot [paste logs here]`")
            return

        say(f"<@{user_id}> Analyzing incident... ⏳")
        try:
            fields = parse_incident_text(text)
            result = analyze_incident(**fields)
            blocks = build_analysis_blocks(result, user_id)
            say(blocks=blocks, thread_ts=event.get("ts"))
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            say(f"❌ Analysis failed: {str(e)[:200]}", thread_ts=event.get("ts"))

    @app.action(re.compile(r"feedback_(correct|incorrect)_.+"))
    def handle_feedback(ack, action, respond, logger):
        ack()
        try:
            data = json.loads(action.get("value", "{}"))
            request_id = data.get("request_id")
            correct = data.get("correct", True)
            if request_id:
                httpx.post(
                    f"{ANALYZER_API_URL}/feedback/{request_id}",
                    json={"score": 5 if correct else 1, "correct": correct},
                    timeout=10,
                )
            respond(f"Thank you for the feedback! {'✅' if correct else '❌'}")
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    sample_text = """SERVICE: checkout-service

LOGS:
[2024-01-15 14:23:45] ERROR checkout-service: Slow request detected (path=/api/checkout duration=45000ms)
[2024-01-15 14:23:45] WARN db-pool: 248 queries executed in single request
[2024-01-15 14:23:45] DEBUG ORM: SELECT * FROM orders WHERE id=1; (2ms)
[2024-01-15 14:23:45] DEBUG ORM: SELECT * FROM orders WHERE id=2; (2ms)

METRICS:
DB CPU: 98%, Connection pool: 100%, P99 latency: 45000ms

ERROR TRACE:
TimeoutError: request exceeded 45000ms
  at checkout-service/handlers/checkout.py:142"""

    print("Running test analysis with sample incident...\n")
    fields = parse_incident_text(sample_text)
    print(f"Parsed fields: {list(fields.keys())}")

    result = analyze_incident(**fields)

    print(f"\nAnalysis result:")
    print(f"  Root cause:     {result.get('root_cause')}")
    print(f"  Confidence:     {result.get('confidence', 0):.0%}")
    print(f"  Category:       {result.get('category')}")
    print(f"  Inference time: {result.get('inference_time_ms')}ms")
    print(f"\n  Fix steps:")
    for i, step in enumerate(result.get("fix_steps", []), 1):
        print(f"    {i}. {step}")

    blocks = build_analysis_blocks(result)
    print(f"\nBlock Kit preview: {len(blocks)} blocks generated")


def main():
    parser = argparse.ArgumentParser(description="Slack bot for incident analysis")
    parser.add_argument("--test", action="store_true", help="Run test mode (no Slack required)")
    parser.add_argument("--port", type=int, default=3000, help="Port for Slack bot server")
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    app = create_slack_app()
    logger.info(f"Starting Slack bot on port {args.port}...")
    from flask import Flask, request, make_response
    from slack_bolt.adapter.flask import SlackRequestHandler

    flask_app = Flask(__name__)
    handler = SlackRequestHandler(app)

    @flask_app.route("/slack/events", methods=["POST"])
    def slack_events():
        return handler.handle(request)

    flask_app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
