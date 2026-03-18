#!/usr/bin/env python3
"""
Grafana Webhook Integration Test Utility
----------------------------------------
Simulates Grafana webhook payloads for testing the /grafana/webhook endpoint.

Usage:
    python grafana_webhook.py --api-url http://localhost:8000 --scenario db_cpu
    python grafana_webhook.py --api-url http://localhost:8000 --scenario memory_leak
    python grafana_webhook.py --api-url http://localhost:8000 --list-scenarios
"""

import argparse
import json
import sys

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Scenario Library
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "db_cpu": {
        "title": "High DB CPU Alert",
        "message": "Database CPU has been above 90% for 5 consecutive minutes",
        "ruleName": "db_cpu_high",
        "state": "alerting",
        "service": "checkout-service",
        "tags": {"team": "backend", "service": "checkout-service", "env": "production"},
        "evalMatches": [
            {"metric": "db_cpu_percent", "value": 97.3},
            {"metric": "active_connections", "value": 100},
            {"metric": "query_duration_p99_ms", "value": 45000},
        ],
        "logs": "[2024-01-15 14:23:45] ERROR checkout-service: SELECT * FROM orders WHERE user_id=1; (2ms)\n"
                "[2024-01-15 14:23:45] DEBUG ORM: 248 queries executed in single request\n"
                "[2024-01-15 14:23:45] WARN db-pool: Connection pool 100% utilised",
        "metrics": "DB CPU: 97%, Query count/req: 248, P99 latency: 45000ms, Pool: 100/100",
        "error_trace": "TimeoutError: request exceeded 45000ms\n  at checkout-service/handlers/checkout.py:142",
    },
    "memory_leak": {
        "title": "Memory Leak Detected — Payment Service",
        "message": "Heap usage growing unbounded, OOMKill imminent",
        "ruleName": "memory_growth_rate_high",
        "state": "alerting",
        "service": "payment-service",
        "tags": {"team": "payments", "service": "payment-service", "env": "production"},
        "evalMatches": [
            {"metric": "heap_used_mb", "value": 7400},
            {"metric": "heap_limit_mb", "value": 8192},
            {"metric": "gc_pause_ms", "value": 2300},
            {"metric": "pod_restarts", "value": 8},
        ],
        "logs": "[2024-01-15 10:15:00] WARN payment-service: Heap usage at 7400MB (limit: 8192MB)\n"
                "[2024-01-15 10:15:01] WARN payment-service: Full GC triggered (pause: 2300ms)\n"
                "[2024-01-15 10:15:02] ERROR k8s: Pod payment-service-7d9f4b OOMKilled (exit code 137)\n"
                "[2024-01-15 10:15:03] INFO k8s: Pod restarting (restart #8)",
        "metrics": "Heap: 7400/8192MB (90%), GC time: 35% of CPU, Restart count: 8/hour, Growth: +150MB/hr",
        "error_trace": "java.lang.OutOfMemoryError: Java heap space\n"
                       "  at payment.cache.InMemoryCache.put(InMemoryCache.java:89)\n"
                       "  at payment.handlers.RequestHandler.process(RequestHandler.java:234)",
    },
    "disk_full": {
        "title": "Disk Full — Production Node",
        "message": "/var/log partition at 99% capacity, writes failing",
        "ruleName": "disk_usage_critical",
        "state": "alerting",
        "service": "api-gateway",
        "tags": {"service": "api-gateway", "env": "production", "node": "prod-node-03"},
        "evalMatches": [
            {"metric": "disk_used_percent", "value": 99.1},
            {"metric": "disk_available_mb", "value": 120},
        ],
        "logs": "[2024-01-16 08:30:00] ERROR api-gateway: Write failed: No space left on device (/var/log)\n"
                "[2024-01-16 08:30:01] ERROR postgres: could not write to pg_wal: No space left on device\n"
                "[2024-01-16 08:30:02] CRIT alertmanager: DISK_FULL on prod-node-03",
        "metrics": "Disk /var/log: 99.1% used, Available: 120MB, Log growth: +1.5GB/day, WAL files: 25GB",
        "error_trace": "OSError: [Errno 28] No space left on device: '/var/log/api-gateway.log.2024-01-16'\n"
                       "  at api-gateway/logging/file_handler.py:78",
    },
    "circuit_breaker": {
        "title": "Circuit Breaker OPEN — Stripe API",
        "message": "Payment gateway timeout cascade, 95% error rate on /checkout",
        "ruleName": "downstream_error_rate_high",
        "state": "alerting",
        "service": "payment-service",
        "tags": {"service": "payment-service", "downstream": "stripe-api"},
        "evalMatches": [
            {"metric": "stripe_api_error_rate", "value": 98},
            {"metric": "checkout_p99_ms", "value": 30000},
            {"metric": "thread_pool_saturation", "value": 100},
        ],
        "logs": "[2024-01-17 16:45:00] ERROR payment-service: Timeout calling stripe-api after 30000ms\n"
                "[2024-01-17 16:45:01] WARN payment-service: Circuit breaker HALF-OPEN for stripe-api\n"
                "[2024-01-17 16:45:02] ERROR payment-service: stripe-api health check failed (10 consecutive)\n"
                "[2024-01-17 16:45:03] WARN payment-service: Circuit breaker OPEN – failing fast",
        "metrics": "Stripe error rate: 98%, Thread pool: 100% saturated, Checkout success: 2%, CB: OPEN",
        "error_trace": "ConnectTimeout: failed to connect to stripe-api:443 after 30000ms\n"
                       "  at payment-service/clients/stripe.py:89\n"
                       "  Retry attempts: 3/3 exhausted",
    },
    "cache_stampede": {
        "title": "Cache Stampede — Homepage",
        "message": "Cache TTL expired, 10000x DB load spike",
        "ruleName": "cache_hit_rate_critical",
        "state": "alerting",
        "service": "recommendation-service",
        "tags": {"service": "recommendation-service"},
        "evalMatches": [
            {"metric": "cache_hit_rate", "value": 0.02},
            {"metric": "db_qps", "value": 15000},
            {"metric": "api_error_rate", "value": 0.65},
        ],
        "logs": "[2024-01-18 12:00:00] INFO redis: Cache miss for key 'homepage_recommendations' (TTL expired)\n"
                "[2024-01-18 12:00:00] WARN recommendation-service: 4200 concurrent cache misses\n"
                "[2024-01-18 12:00:01] ERROR postgres: connection pool exhausted\n"
                "[2024-01-18 12:00:01] ERROR recommendation-service: latency spike to 25000ms",
        "metrics": "Cache hit rate: 2% (was 98%), DB QPS: 15000 (normal: 50), Error rate: 65%, Recovery: 45s",
        "error_trace": "CacheStampedeEvent: key='homepage_recommendations' expired\n"
                       "  4200 goroutines racing to rebuild cache\n"
                       "  DBError: connection pool exhausted during rebuild",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Webhook Sender
# ─────────────────────────────────────────────────────────────────────────────

def send_webhook(api_url: str, scenario_name: str, timeout: int = 60) -> dict:
    scenario = SCENARIOS[scenario_name]
    endpoint = f"{api_url.rstrip('/')}/grafana/webhook"
    print(f"\n{'='*60}")
    print(f"Sending Grafana webhook: {scenario_name}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Title: {scenario['title']}")
    print(f"{'='*60}")

    response = httpx.post(endpoint, json=scenario, timeout=timeout)
    response.raise_for_status()
    result = response.json()

    print(f"\nRESPONSE (HTTP {response.status_code}):")
    print(f"  Request ID:     {result.get('request_id')}")
    print(f"  Root Cause:     {result.get('root_cause')}")
    print(f"  Confidence:     {result.get('confidence', 0):.0%}")
    print(f"  Category:       {result.get('category')}")
    print(f"  Inference time: {result.get('inference_time_ms')}ms")
    print(f"\n  Fix Steps:")
    for i, step in enumerate(result.get("fix_steps", []), 1):
        print(f"    {i}. {step}")

    similar = result.get("similar_incidents", [])
    if similar:
        print(f"\n  Similar Incidents:")
        for s in similar:
            print(f"    - [{s.get('incident_id')}] {s.get('service')}: {s.get('root_cause', '')[:60]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Grafana webhook integration tester")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Scenario to send")
    parser.add_argument("--all", action="store_true", help="Run all scenarios sequentially")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for name, scenario in SCENARIOS.items():
            print(f"  {name:<20} {scenario['title']}")
        return

    if args.all:
        results = {}
        for name in SCENARIOS:
            try:
                results[name] = send_webhook(args.api_url, name, args.timeout)
            except Exception as e:
                print(f"  ERROR: {e}")
                results[name] = {"error": str(e)}
        print(f"\n{'='*60}")
        print(f"All scenarios complete ({len(results)} total)")
        return

    if not args.scenario:
        parser.error("Provide --scenario or --all")

    try:
        send_webhook(args.api_url, args.scenario, args.timeout)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
