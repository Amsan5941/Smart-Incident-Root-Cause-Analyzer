#!/usr/bin/env python3
"""
Synthetic Incident Data Generator
Generates 800+ realistic production incident training examples for fine-tuning.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

random.seed(42)

SERVICES = [
    "checkout-service", "payment-service", "auth-service", "api-gateway",
    "user-service", "inventory-service", "notification-service", "search-service",
    "recommendation-service", "order-service", "cart-service", "product-service",
]

ENVIRONMENTS = ["production", "staging"]
SEVERITIES = ["critical", "high", "medium"]

# ─────────────────────────────────────────────────────────────────────────────
# Incident templates: each template returns (logs, metrics, error_trace, root_cause, resolution_steps, category)
# ─────────────────────────────────────────────────────────────────────────────

def n_plus_one_query(service: str, ts: str) -> dict:
    table = random.choice(["orders", "users", "products", "reviews", "carts"])
    query_time = random.randint(3000, 90000)
    query_count = random.randint(50, 500)
    cpu = random.randint(75, 99)
    return {
        "logs": f"""[{ts}] ERROR {service}: Slow request detected (path=/api/checkout duration={query_time}ms)
[{ts}] WARN  db-pool: {query_count} queries executed in single request
[{ts}] DEBUG ORM: SELECT * FROM {table} WHERE id=1; (2ms)
[{ts}] DEBUG ORM: SELECT * FROM {table} WHERE id=2; (2ms)
[{ts}] DEBUG ORM: SELECT * FROM {table} WHERE id=3; (2ms)
... ({query_count - 3} more similar queries)
[{ts}] ERROR {service}: Request timeout after {query_time}ms""",
        "metrics": f"DB CPU: {cpu}%, Connection pool: {random.randint(90, 100)}% utilised, "
                   f"Query rate: {query_count * 10}/min, P99 latency: {query_time}ms, "
                   f"Active connections: {random.randint(95, 100)}/100",
        "error_trace": f"TimeoutError: request exceeded {query_time}ms\n"
                       f"  at {service}/handlers/checkout.py:142\n"
                       f"  at orm/query.py:89 (execute_query)\n"
                       f"  at orm/relations.py:234 (lazy_load)",
        "root_cause": f"N+1 query problem in {service}: ORM lazy-loading {table} records individually "
                      f"({query_count} queries instead of 1 JOIN)",
        "resolution_steps": [
            f"Replace lazy-loading with eager JOIN: `queryset.select_related('{table}')`",
            "Add database query count assertion in integration tests",
            "Set ORM strict mode to raise on N+1 in non-production environments",
            "Add query count metric to request tracing",
        ],
        "category": "database_n_plus_one",
    }


def connection_pool_exhaustion(service: str, ts: str) -> dict:
    pool_size = random.choice([10, 20, 50, 100])
    wait_time = random.randint(5000, 30000)
    rps = random.randint(200, 2000)
    return {
        "logs": f"""[{ts}] ERROR {service}: Could not acquire database connection after {wait_time}ms
[{ts}] ERROR db-pool: Pool exhausted (size={pool_size}, waiting={random.randint(50, 200)})
[{ts}] WARN  {service}: Rejecting request – no DB connection available
[{ts}] ERROR {service}: HTTP 503 Service Unavailable (connection pool timeout)
[{ts}] INFO  {service}: RPS={rps}, active_connections={pool_size}/{pool_size}""",
        "metrics": f"DB connections: {pool_size}/{pool_size} (100%), "
                   f"Request queue depth: {random.randint(100, 500)}, "
                   f"Error rate: {random.randint(40, 80)}%, RPS: {rps}, "
                   f"Connection wait time P99: {wait_time}ms",
        "error_trace": f"PoolTimeoutError: QueuePool limit of size {pool_size} overflow 0 reached, "
                       f"connection timed out, timeout {wait_time / 1000:.1f}s\n"
                       f"  at sqlalchemy/pool/impl.py:298\n"
                       f"  at {service}/db/session.py:45\n"
                       f"  at {service}/repositories/base.py:112",
        "root_cause": f"Database connection pool exhaustion in {service}: pool size ({pool_size}) is too "
                      f"small for current load ({rps} RPS). Long-running transactions may be holding connections.",
        "resolution_steps": [
            f"Increase connection pool size: `POOL_SIZE={pool_size * 3}` (monitor DB max_connections)",
            "Audit long-running transactions and add statement_timeout",
            "Implement connection pool monitoring alert at 80% utilisation",
            "Consider PgBouncer connection pooler to reduce DB-side connections",
            "Add circuit breaker to fail fast when pool is exhausted",
        ],
        "category": "database_connection_pool",
    }


def memory_leak(service: str, ts: str) -> dict:
    heap_mb = random.randint(2000, 7800)
    restart_count = random.randint(3, 15)
    return {
        "logs": f"""[{ts}] WARN  {service}: Heap usage at {heap_mb}MB (threshold: 6144MB)
[{ts}] WARN  {service}: GC pause time: {random.randint(500, 3000)}ms (full GC triggered)
[{ts}] ERROR {service}: OutOfMemoryError – Java heap space
[{ts}] ERROR k8s: Pod {service}-7d9f4b-xk2p1 OOMKilled (exit code 137)
[{ts}] INFO  k8s: Restarting pod {service}-7d9f4b-xk2p1 (restart #{restart_count})
[{ts}] WARN  {service}: Container restarted {restart_count} times in last 1h""",
        "metrics": f"Heap used: {heap_mb}MB / 8192MB ({heap_mb * 100 // 8192}%), "
                   f"GC time: {random.randint(15, 45)}% of CPU, "
                   f"Pod restarts: {restart_count}/hour, "
                   f"Memory growth rate: +{random.randint(50, 200)}MB/hour",
        "error_trace": f"java.lang.OutOfMemoryError: Java heap space\n"
                       f"  at java.util.Arrays.copyOf(Arrays.java:3210)\n"
                       f"  at {service}.cache.InMemoryCache.put(InMemoryCache.java:89)\n"
                       f"  at {service}.handlers.RequestHandler.process(RequestHandler.java:234)\n"
                       f"Caused by: unbounded cache growth (no eviction policy)",
        "root_cause": f"Memory leak in {service}: InMemoryCache has no eviction policy, "
                      f"accumulating objects indefinitely. Cache grows at ~{random.randint(50, 200)}MB/hour "
                      f"until OOMKill.",
        "resolution_steps": [
            "Add TTL and max-size eviction to InMemoryCache (e.g., Caffeine or Guava Cache)",
            "Set JVM heap dump on OOM: -XX:+HeapDumpOnOutOfMemoryError",
            "Add memory growth rate alert (>50MB/hour triggers page)",
            "Profile heap with async-profiler to confirm leak source",
            "Increase pod memory limit as temporary mitigation",
        ],
        "category": "memory_leak",
    }


def high_cpu_hot_loop(service: str, ts: str) -> dict:
    cpu = random.randint(92, 100)
    thread = random.choice(["worker-1", "event-loop", "scheduler-thread", "consumer-0"])
    return {
        "logs": f"""[{ts}] WARN  {service}: CPU usage {cpu}% sustained for {random.randint(5, 30)} minutes
[{ts}] WARN  {service}: Thread {thread} spinning – no yield detected
[{ts}] ERROR {service}: Health check timeout (liveness probe failed)
[{ts}] WARN  k8s: Pod {service} not responding to liveness probe
[{ts}] INFO  k8s: Killing unresponsive pod, scheduling replacement
[{ts}] WARN  {service}: Request queue backing up ({random.randint(500, 5000)} pending)""",
        "metrics": f"CPU: {cpu}%, Single-thread CPU: 100%, "
                   f"Request queue: {random.randint(500, 5000)}, "
                   f"Throughput: dropped {random.randint(60, 90)}%, "
                   f"Context switches: {random.randint(100, 500)}/sec (very low – thread blocked)",
        "error_trace": f"Thread dump – {thread} state: RUNNABLE for >300s\n"
                       f"  at {service}.processing.DataProcessor.parseEvent(DataProcessor.java:445)\n"
                       f"  at {service}.processing.DataProcessor.processLoop(DataProcessor.java:312)\n"
                       f"  at java.util.regex.Pattern.compile(Pattern.java:1891)\n"
                       f"  [catastrophic backtracking in regex]",
        "root_cause": f"CPU hot-loop in {service}: catastrophic regex backtracking in DataProcessor.parseEvent(). "
                      f"Pathological input triggers O(2^n) regex evaluation, pinning {thread} at 100% CPU.",
        "resolution_steps": [
            "Replace backtracking regex with linear-time alternative or possessive quantifiers",
            "Add regex timeout using Pattern.matcher with timeout (Java 21+)",
            "Validate and sanitise input before regex application",
            "Add CPU spike alert: >90% for >2 minutes triggers page",
            "Consider ReDoS testing in CI pipeline",
        ],
        "category": "cpu_hot_loop",
    }


def disk_full(service: str, ts: str) -> dict:
    partition = random.choice(["/var/log", "/data", "/", "/var/lib/postgresql"])
    usage = random.randint(97, 100)
    return {
        "logs": f"""[{ts}] ERROR {service}: Write failed: No space left on device ({partition})
[{ts}] ERROR postgres: could not write to file "pg_wal/000000010000001A00000056": No space left on device
[{ts}] WARN  system: Disk {partition} at {usage}% capacity
[{ts}] ERROR {service}: Failed to write audit log – disk full
[{ts}] ERROR {service}: Database write transaction aborted (ENOSPC)
[{ts}] CRIT  alertmanager: DISK_FULL alert fired for node {service}-node-01""",
        "metrics": f"Disk {partition}: {usage}% used, "
                   f"Available: {random.randint(100, 500)}MB, "
                   f"Write IOPS: 0 (blocked), "
                   f"Log file growth: +{random.randint(500, 2000)}MB/day, "
                   f"WAL files: {random.randint(10, 50)}GB",
        "error_trace": f"OSError: [Errno 28] No space left on device: '{partition}/app.log.{ts[:10]}'\n"
                       f"  at {service}/logging/file_handler.py:78\n"
                       f"  at logging/handlers.py:120 (emit)\n"
                       f"PostgreSQL FATAL: could not write to file in pg_wal",
        "root_cause": f"Disk exhaustion on {partition}: unrotated log files and PostgreSQL WAL accumulation "
                      f"consumed all available disk space. No disk usage alert was configured below 95%.",
        "resolution_steps": [
            f"Immediate: `find {partition} -name '*.log' -mtime +7 -delete` to free space",
            "Configure logrotate with daily rotation and 7-day retention",
            "Set disk alert at 80% (warning) and 90% (critical)",
            "If WAL: check replication slots blocking WAL cleanup (`pg_replication_slots`)",
            "Increase volume size or add log shipping to object storage",
        ],
        "category": "disk_full",
    }


def slow_query_missing_index(service: str, ts: str) -> dict:
    table = random.choice(["events", "audit_logs", "transactions", "sessions", "metrics"])
    scan_rows = random.randint(5_000_000, 50_000_000)
    query_time = random.randint(10000, 120000)
    return {
        "logs": f"""[{ts}] WARN  postgres: slow query detected ({query_time}ms):
                  SELECT * FROM {table} WHERE created_at > '2024-01-01' AND user_id = 12345
[{ts}] WARN  {service}: Database query exceeded threshold ({query_time}ms > 1000ms)
[{ts}] INFO  postgres: sequential scan on {table} ({scan_rows:,} rows examined)
[{ts}] WARN  {service}: P99 latency degraded to {query_time}ms
[{ts}] ERROR {service}: Request timeout for user 12345""",
        "metrics": f"Query time P99: {query_time}ms (normal: 50ms), "
                   f"Table scan rows: {scan_rows:,}, "
                   f"DB CPU: {random.randint(60, 90)}%, "
                   f"Buffer cache hit rate: {random.randint(20, 50)}% (normally >95%)",
        "error_trace": f"QueryTimeout: query exceeded {query_time}ms\n"
                       f"QUERY PLAN:\n"
                       f"  Seq Scan on {table} (cost=0.00..{scan_rows // 100:.2f} rows={scan_rows})\n"
                       f"    Filter: ((created_at > '2024-01-01') AND (user_id = 12345))\n"
                       f"    Rows Removed by Filter: {scan_rows - 100:,}\n"
                       f"  Planning time: 2.3ms, Execution time: {query_time}ms",
        "root_cause": f"Missing composite index on {table}(user_id, created_at): full sequential scan of "
                      f"{scan_rows:,} rows. Recent table growth exceeded threshold where seq scan becomes "
                      f"costlier than index scan.",
        "resolution_steps": [
            f"CREATE INDEX CONCURRENTLY idx_{table}_user_created ON {table}(user_id, created_at DESC)",
            "Run EXPLAIN ANALYZE to confirm index is used post-creation",
            "Add slow query log alert (>500ms triggers investigation)",
            "Schedule regular ANALYZE on high-growth tables",
            "Consider partitioning large tables by created_at",
        ],
        "category": "database_missing_index",
    }


def kubernetes_oom_kill(service: str, ts: str) -> dict:
    memory_limit = random.choice(["512Mi", "1Gi", "2Gi", "4Gi"])
    pod_name = f"{service}-{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:5]}"
    return {
        "logs": f"""[{ts}] WARN  kubelet: {pod_name} is using high memory
[{ts}] ERROR kubelet: OOMKilling process {service} (pid {random.randint(1000, 9999)})
              container {service} in pod {pod_name}: usage exceeds limit {memory_limit}
[{ts}] WARN  k8s: Pod {pod_name} OOMKilled (exit code 137)
[{ts}] INFO  k8s: Back-off restarting failed container (CrashLoopBackOff)
[{ts}] WARN  k8s: Pod {pod_name} restart count: {random.randint(5, 20)}
[{ts}] ERROR alertmanager: CrashLoopBackOff alert – {service}""",
        "metrics": f"Pod memory: {memory_limit} limit exceeded, "
                   f"OOMKill count: {random.randint(5, 20)}/hour, "
                   f"Node memory pressure: true, "
                   f"Container restart count: {random.randint(5, 20)}, "
                   f"Node allocatable memory: {random.randint(60, 90)}% used",
        "error_trace": f"OOMKilled: container {service} exceeded memory limit {memory_limit}\n"
                       f"  kernel: [oom_kill_process] Out of memory: Kill process {random.randint(1000, 9999)} "
                       f"({service}) score {random.randint(800, 999)} or sacrifice child\n"
                       f"  kubectl describe pod {pod_name}: Last State: OOMKilled",
        "root_cause": f"Kubernetes OOMKill in {service}: memory limit ({memory_limit}) set too low for "
                      f"workload. Recent traffic increase or data processing job caused memory spike beyond limit.",
        "resolution_steps": [
            f"Increase memory limit in deployment: `resources.limits.memory: {memory_limit.replace('Gi', 'Gi').replace('512Mi', '1Gi')}`",
            "Set memory request = 70% of limit to ensure proper scheduling",
            "Add JVM/runtime memory flags to cap heap within container limit",
            "Implement horizontal pod autoscaling (HPA) for memory-heavy workloads",
            "Profile memory usage with `kubectl top pod --containers`",
        ],
        "category": "kubernetes_oom",
    }


def cache_stampede(service: str, ts: str) -> dict:
    cache_key = random.choice(["product_catalog", "user_session", "homepage_data", "price_list"])
    ttl = random.choice([60, 300, 600, 3600])
    return {
        "logs": f"""[{ts}] INFO  redis: Cache miss for key '{cache_key}' (TTL expired)
[{ts}] WARN  {service}: {random.randint(500, 5000)} concurrent cache misses for '{cache_key}'
[{ts}] ERROR postgres: connection pool exhausted (stampede origin)
[{ts}] WARN  {service}: DB query queue depth: {random.randint(200, 2000)} (normal: <10)
[{ts}] ERROR {service}: Request latency spike to {random.randint(5000, 30000)}ms
[{ts}] INFO  redis: Cache rebuilt for '{cache_key}' after {random.randint(2, 15)}s""",
        "metrics": f"Cache hit rate: dropped from 98% to 0% at {ts}, "
                   f"DB QPS spike: {random.randint(1000, 10000)}x normal, "
                   f"Redis memory: stable (miss not eviction), "
                   f"API error rate: {random.randint(30, 70)}% during stampede, "
                   f"Recovery time: {random.randint(10, 60)}s",
        "error_trace": f"CacheStampedeEvent: key='{cache_key}' expired simultaneously\n"
                       f"  {random.randint(500, 5000)} goroutines racing to rebuild cache\n"
                       f"  DBError: connection pool exhausted during rebuild\n"
                       f"  at {service}/cache/manager.go:156 (GetOrSet)\n"
                       f"  at {service}/handlers/catalog.go:89",
        "root_cause": f"Cache stampede on key '{cache_key}': when TTL ({ttl}s) expires, thousands of "
                      f"concurrent requests simultaneously attempt DB reload, overwhelming the database. "
                      f"No mutex or probabilistic early expiration implemented.",
        "resolution_steps": [
            "Implement mutex/single-flight pattern: only one goroutine rebuilds cache",
            f"Use probabilistic early expiration (XFetch algorithm) for hot keys",
            "Set cache-aside with stale-while-revalidate: serve stale data during refresh",
            "Add cache warming job before TTL expiry for predictable keys",
            f"Monitor cache hit rate; alert if drops below 90%",
        ],
        "category": "cache_stampede",
    }


def deadlock(service: str, ts: str) -> dict:
    table1 = random.choice(["orders", "inventory"])
    table2 = random.choice(["payments", "reservations"])
    return {
        "logs": f"""[{ts}] ERROR postgres: deadlock detected
[{ts}] ERROR postgres: Process {random.randint(1000,9999)} waits for ShareLock on transaction {random.randint(10000,99999)}
[{ts}] ERROR postgres: Process {random.randint(1000,9999)} waits for ShareLock on transaction {random.randint(10000,99999)}
[{ts}] ERROR {service}: Transaction rolled back due to deadlock (attempt {random.randint(1,5)}/3)
[{ts}] WARN  {service}: Deadlock rate: {random.randint(10, 100)}/min (normal: 0)
[{ts}] ERROR {service}: Order processing failing with deadlock errors""",
        "metrics": f"Deadlock rate: {random.randint(10, 100)}/minute, "
                   f"Transaction rollback rate: {random.randint(20, 60)}%, "
                   f"Lock wait time: {random.randint(1000, 30000)}ms, "
                   f"DB CPU: {random.randint(60, 90)}%, "
                   f"Active locks: {random.randint(100, 500)}",
        "error_trace": f"DeadlockDetected: deadlock found when waiting for lock\n"
                       f"DETAIL: Process A: UPDATE {table1} → waiting for {table2}\n"
                       f"        Process B: UPDATE {table2} → waiting for {table1}\n"
                       f"  at {service}/services/order_service.py:234 (process_order)\n"
                       f"  at {service}/services/payment_service.py:156 (reserve_payment)",
        "root_cause": f"Database deadlock in {service}: order processing acquires locks on {table1} then "
                      f"{table2}, while payment processing acquires {table2} then {table1}. Inconsistent "
                      f"lock acquisition order after recent refactor.",
        "resolution_steps": [
            f"Enforce consistent lock ordering: always acquire {table1} lock before {table2}",
            "Use SELECT FOR UPDATE SKIP LOCKED for queue-style processing",
            "Implement retry with exponential backoff for deadlock errors",
            "Add deadlock monitoring alert (>1/minute triggers investigation)",
            "Consider optimistic locking with version columns",
        ],
        "category": "database_deadlock",
    }


def service_timeout_cascade(service: str, ts: str) -> dict:
    downstream = random.choice(["payment-gateway", "shipping-api", "tax-service", "identity-provider"])
    timeout_ms = random.choice([5000, 10000, 30000])
    return {
        "logs": f"""[{ts}] ERROR {service}: Timeout calling {downstream} after {timeout_ms}ms
[{ts}] WARN  {service}: Circuit breaker HALF-OPEN for {downstream}
[{ts}] ERROR {service}: {downstream} health check failed ({random.randint(3, 10)} consecutive failures)
[{ts}] WARN  {service}: Circuit breaker OPEN for {downstream} – failing fast
[{ts}] ERROR {service}: Request queue backing up ({random.randint(100, 2000)} pending)
[{ts}] CRIT  {service}: Cascading failure – upstream services timing out""",
        "metrics": f"Downstream {downstream} error rate: {random.randint(80, 100)}%, "
                   f"Circuit breaker: OPEN, "
                   f"Thread pool: {random.randint(90, 100)}% saturated (waiting on {downstream}), "
                   f"Request success rate: {random.randint(5, 20)}%, "
                   f"Bulkhead queue depth: {random.randint(100, 1000)}",
        "error_trace": f"ConnectTimeout: failed to connect to {downstream}:{random.randint(8000, 9999)} "
                       f"after {timeout_ms}ms\n"
                       f"  at {service}/clients/{downstream.replace('-','_')}.py:89 (call)\n"
                       f"  at resilience4j/circuit_breaker.py:234\n"
                       f"  Caused by: {downstream} TCP connect timeout\n"
                       f"  Retry attempts: 3/3 exhausted",
        "root_cause": f"Cascading timeout failure: {downstream} is down/slow, causing {service} threads "
                      f"to block waiting for responses. Thread pool saturation then causes failure cascade "
                      f"to upstream callers. Circuit breaker opened after {random.randint(3, 10)} failures.",
        "resolution_steps": [
            f"Investigate {downstream} independently (separate on-call rotation if external)",
            "Verify circuit breaker is correctly shedding load (check breaker state)",
            "Implement bulkhead isolation: dedicated thread pool for {downstream} calls",
            f"Add fallback: return cached/default response when {downstream} is unavailable",
            "Review timeout values – 30s may be too long if downstream is unresponsive",
        ],
        "category": "service_timeout_cascade",
    }


def certificate_expiry(service: str, ts: str) -> dict:
    domain = f"api.{service.replace('-service', '')}.company.com"
    days_remaining = random.randint(0, 3)
    return {
        "logs": f"""[{ts}] ERROR {service}: TLS handshake failed: certificate expired for {domain}
[{ts}] WARN  {service}: SSL certificate for {domain} expires in {days_remaining} day(s)
[{ts}] ERROR nginx: SSL_do_handshake() failed (SSL: error:14094416:SSL routines)
[{ts}] ERROR {service}: HTTPS requests failing with SSL_ERROR_BAD_CERT_DOMAIN
[{ts}] CRIT  monitoring: Certificate expiry alert for {domain}
[{ts}] ERROR {service}: All HTTPS traffic to {domain} rejected""",
        "metrics": f"HTTPS error rate: 100%, "
                   f"Certificate days remaining: {days_remaining}, "
                   f"Affected endpoints: all (TLS termination failure), "
                   f"HTTP fallback traffic: 0 (HSTS enforced), "
                   f"Affected users: 100%",
        "error_trace": f"SSLCertificateError: certificate for {domain} expired {days_remaining} days ago\n"
                       f"  Certificate subject: CN={domain}\n"
                       f"  Certificate expiry: {ts[:10]} 23:59:59 UTC\n"
                       f"  at ssl/tls_handshake.py:234\n"
                       f"  Error code: CERTIFICATE_VERIFY_FAILED",
        "root_cause": f"TLS certificate expiry for {domain}: certificate expired {days_remaining} day(s) ago. "
                      f"Certificate rotation was not automated and manual renewal process failed due to no "
                      f"alert configured before expiry.",
        "resolution_steps": [
            f"Immediate: renew certificate for {domain} via cert-manager or Let's Encrypt",
            "Automate certificate renewal: deploy cert-manager in Kubernetes",
            "Add certificate expiry monitoring: alert at 30, 14, 7, and 1 day",
            "Document manual renewal runbook as backup procedure",
            "Consider wildcard certificate to reduce surface area",
        ],
        "category": "certificate_expiry",
    }


def rate_limit_breach(service: str, ts: str) -> dict:
    external_api = random.choice(["stripe-api", "sendgrid", "twilio", "google-maps", "aws-ses"])
    limit = random.choice([100, 1000, 10000])
    return {
        "logs": f"""[{ts}] ERROR {service}: Rate limit exceeded for {external_api} (HTTP 429)
[{ts}] WARN  {service}: {external_api} rate limit: {limit}/{random.choice(['minute','hour'])} reached
[{ts}] ERROR {service}: Retry-After: {random.randint(10, 3600)} seconds
[{ts}] WARN  {service}: Queuing requests – {random.randint(100, 5000)} items in backlog
[{ts}] ERROR {service}: Email delivery failing – {external_api} rejected all requests
[{ts}] WARN  {service}: Falling back to alternative provider""",
        "metrics": f"{external_api} error rate: {random.randint(80, 100)}% (HTTP 429), "
                   f"API call rate: {limit * 2}/min (2x limit), "
                   f"Queue backlog: {random.randint(100, 5000)}, "
                   f"Delivery success rate: {random.randint(0, 20)}%, "
                   f"Cost spike: {random.randint(200, 500)}% above normal",
        "error_trace": f"RateLimitError: 429 Too Many Requests from {external_api}\n"
                       f"  X-RateLimit-Limit: {limit}\n"
                       f"  X-RateLimit-Remaining: 0\n"
                       f"  Retry-After: {random.randint(10, 3600)}\n"
                       f"  at {service}/clients/{external_api.replace('-', '_')}.py:67",
        "root_cause": f"Rate limit breach against {external_api}: recent feature deployment triggered "
                      f"notification fan-out without rate limiting, sending {limit * 2}/min against a "
                      f"{limit}/min limit. No rate limiting or token bucket implemented.",
        "resolution_steps": [
            f"Implement token bucket rate limiter for {external_api} client",
            "Add exponential backoff with jitter on 429 responses",
            "Batch notifications where possible to reduce API call volume",
            "Monitor API quota usage and alert at 80% of limit",
            f"Consider upgrading {external_api} plan if usage is legitimately higher",
        ],
        "category": "rate_limit_breach",
    }


def config_error_deployment(service: str, ts: str) -> dict:
    config_key = random.choice(["DATABASE_URL", "REDIS_URL", "API_KEY", "SECRET_KEY", "AWS_REGION"])
    return {
        "logs": f"""[{ts}] ERROR {service}: Failed to start: missing required config '{config_key}'
[{ts}] ERROR {service}: KeyError: '{config_key}' not found in environment
[{ts}] WARN  k8s: Pod {service}-deploy failing to initialise
[{ts}] ERROR k8s: CrashLoopBackOff – {service} (exit code 1)
[{ts}] INFO  {service}: Deployment v2.3.1 rolled back to v2.3.0
[{ts}] WARN  deploy: Rollback triggered after {random.randint(2, 10)} failed pod starts""",
        "metrics": f"Pod start success rate: 0%, "
                   f"CrashLoopBackOff count: {random.randint(3, 10)}, "
                   f"Deployment rollback triggered: true, "
                   f"Downtime: {random.randint(2, 15)} minutes, "
                   f"Error type: configuration (not code)",
        "error_trace": f"ConfigurationError: Required environment variable '{config_key}' is not set\n"
                       f"  at {service}/config/settings.py:34 (validate_required)\n"
                       f"  at {service}/app.py:12 (startup)\n"
                       f"  SystemExit: 1\n"
                       f"Deployment history: v2.3.0 (stable) → v2.3.1 (failed, missing config)",
        "root_cause": f"Deployment failure in {service}: new version v2.3.1 requires '{config_key}' "
                      f"environment variable that was not added to the Kubernetes secret/ConfigMap during "
                      f"deployment. Config change was not paired with infrastructure update.",
        "resolution_steps": [
            f"Add '{config_key}' to Kubernetes Secret/ConfigMap in all environments",
            "Implement startup config validation that fails fast with clear error messages",
            "Add required env vars to deployment checklist / PR template",
            "Use config schema validation in CI pipeline before deployment",
            "Track config changes alongside code changes in same PR",
        ],
        "category": "config_error",
    }


def network_partition(service: str, ts: str) -> dict:
    region = random.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
    affected_az = random.choice(["AZ-a", "AZ-b", "AZ-c"])
    return {
        "logs": f"""[{ts}] ERROR {service}: Cannot reach peers in {affected_az} ({region})
[{ts}] WARN  consul: Health check failing for nodes in {affected_az}
[{ts}] ERROR {service}: Split-brain detected – quorum lost
[{ts}] WARN  haproxy: Removing {affected_az} nodes from load balancer pool
[{ts}] ERROR {service}: Write operations failing – insufficient quorum ({affected_az} unreachable)
[{ts}] CRIT  alertmanager: NETWORK_PARTITION alert – {region}/{affected_az}""",
        "metrics": f"Nodes unreachable in {affected_az}: {random.randint(3, 10)}, "
                   f"Packet loss to {affected_az}: 100%, "
                   f"Read success rate: 100% (reads from healthy AZs), "
                   f"Write success rate: 0% (quorum requires {affected_az}), "
                   f"Replication lag: {random.randint(30, 600)}s and growing",
        "error_trace": f"NetworkPartitionError: unable to reach quorum ({affected_az} unreachable)\n"
                       f"  ping {affected_az}-node-01: 100% packet loss\n"
                       f"  consul: Node health CRITICAL for {random.randint(3, 10)} nodes\n"
                       f"  raft: election timeout – cannot form quorum without {affected_az}",
        "root_cause": f"Network partition in {region}: {affected_az} availability zone is isolated due to "
                      f"network infrastructure failure. Services requiring write quorum are failing; "
                      f"read-only operations continue on healthy AZs.",
        "resolution_steps": [
            f"Contact cloud provider regarding {region}/{affected_az} network issue",
            "Failover writes to healthy AZs if cluster supports it",
            "Monitor replication lag for when {affected_az} reconnects",
            "Review multi-AZ deployment: ensure writes can succeed with N-1 AZs",
            "Add runbook for AZ partition (automated or manual failover procedure)",
        ],
        "category": "network_partition",
    }


def high_error_rate_bug(service: str, ts: str) -> dict:
    error_type = random.choice(["NullPointerException", "KeyError", "AttributeError", "TypeError", "ValueError"])
    endpoint = random.choice(["/api/checkout", "/api/search", "/api/orders", "/api/users", "/api/products"])
    error_rate = random.randint(40, 95)
    return {
        "logs": f"""[{ts}] ERROR {service}: Unhandled {error_type} in handler for {endpoint}
[{ts}] ERROR {service}: {error_type}: 'price' attribute missing in Product object
[{ts}] WARN  {service}: Error rate on {endpoint}: {error_rate}% (threshold: 5%)
[{ts}] INFO  {service}: Deployment v{random.randint(2,5)}.{random.randint(0,9)}.{random.randint(0,9)} "
              deployed {random.randint(2, 30)} minutes ago
[{ts}] ERROR {service}: {random.randint(100, 10000)} errors in last 5 minutes
[{ts}] WARN  alertmanager: HIGH_ERROR_RATE alert for {service}/{endpoint}""",
        "metrics": f"Error rate {endpoint}: {error_rate}%, "
                   f"HTTP 500 rate: {error_rate}%, "
                   f"Affected users: {random.randint(100, 10000)}, "
                   f"Deployment age: {random.randint(2, 30)} minutes (recent change), "
                   f"Previous error rate: <1%",
        "error_trace": f"{error_type}: 'price' attribute missing\n"
                       f"  File \"{service}/handlers/product.py\", line {random.randint(50, 300)}, in handle_{endpoint.split('/')[-1]}\n"
                       f"    total = item.price * item.quantity\n"
                       f"AttributeError: 'Product' object has no attribute 'price'\n"
                       f"  Note: Product schema updated in v2.3.1 – 'price' renamed to 'unit_price'",
        "root_cause": f"Breaking schema change in {service}: recent deployment renamed 'price' to 'unit_price' "
                      f"in Product model but handler code on {endpoint} still references old field name. "
                      f"No backward compatibility or migration handled.",
        "resolution_steps": [
            f"Immediate: rollback to previous version (`kubectl rollout undo deployment/{service}`)",
            "Fix handler to use new field name 'unit_price' (or add compatibility shim)",
            "Add schema change tests: verify all consumers updated before deploying",
            "Use feature flags for breaking changes to enable gradual rollout",
            "Add contract testing between producer/consumer services",
        ],
        "category": "application_error",
    }


# ─────────────────────────────────────────────────────────────────────────────
# All template generators
# ─────────────────────────────────────────────────────────────────────────────
GENERATORS = [
    n_plus_one_query,
    connection_pool_exhaustion,
    memory_leak,
    high_cpu_hot_loop,
    disk_full,
    slow_query_missing_index,
    kubernetes_oom_kill,
    cache_stampede,
    deadlock,
    service_timeout_cascade,
    certificate_expiry,
    rate_limit_breach,
    config_error_deployment,
    network_partition,
    high_error_rate_bug,
]


def make_timestamp(days_back: int = 365) -> str:
    base = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    base = base.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
    )
    return base.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_incident(idx: int) -> dict[str, Any]:
    service = random.choice(SERVICES)
    ts = make_timestamp()
    generator = random.choice(GENERATORS)
    data = generator(service, ts)
    return {
        "incident_id": f"INC-{idx:05d}",
        "timestamp": ts,
        "service": service,
        "severity": random.choice(SEVERITIES),
        "environment": random.choice(ENVIRONMENTS),
        "logs": data["logs"],
        "metrics": data["metrics"],
        "error_trace": data["error_trace"],
        "root_cause": data["root_cause"],
        "resolution_steps": data["resolution_steps"],
        "category": data["category"],
        # Instruction-tuned training format
        "instruction": "You are an expert SRE. Analyze the following production incident and identify the root cause, confidence level, and step-by-step resolution.",
        "input": f"SERVICE: {service}\nSEVERITY: {data.get('severity', 'high')}\n\nLOGS:\n{data['logs']}\n\nMETRICS:\n{data['metrics']}\n\nERROR TRACE:\n{data['error_trace']}",
        "output": f"ROOT CAUSE: {data['root_cause']}\n\nCONFIDENCE: {random.randint(78, 97)}%\n\nRESOLUTION STEPS:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(data["resolution_steps"])),
    }


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic incident training data")
    parser.add_argument("--count", type=int, default=900, help="Number of incidents to generate")
    parser.add_argument("--output", type=str, default="data/training_incidents.jsonl")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction for test set")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    incidents = [make_incident(i + 1) for i in range(args.count)]
    random.shuffle(incidents)

    split_idx = int(len(incidents) * (1 - args.test_split))
    train_incidents = incidents[:split_idx]
    test_incidents = incidents[split_idx:]

    train_path = args.output
    test_path = args.output.replace(".jsonl", "_test.jsonl")

    with open(train_path, "w") as f:
        for inc in train_incidents:
            f.write(json.dumps(inc) + "\n")

    with open(test_path, "w") as f:
        for inc in test_incidents:
            f.write(json.dumps(inc) + "\n")

    # Category distribution
    from collections import Counter
    cats = Counter(inc["category"] for inc in incidents)

    print(f"Generated {len(incidents)} incidents")
    print(f"  Train: {len(train_incidents)} → {train_path}")
    print(f"  Test:  {len(test_incidents)}  → {test_path}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {count:>4}")


if __name__ == "__main__":
    main()
