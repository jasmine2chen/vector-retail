"""
scripts/healthcheck.py
Simple health check script for deployment validation.
Exit code 0 = healthy, 1 = unhealthy.

Usage: python scripts/healthcheck.py
"""
import sys
import os
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def check_http() -> bool:
    """Check if the /health HTTP endpoint is responding."""
    port = os.getenv("PORT", "8080")
    url = f"http://127.0.0.1:{port}/health"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        if resp.status == 200:
            print(f"OK: /health returned 200")
            return True
        print(f"FAIL: /health returned {resp.status}")
        return False
    except urllib.error.URLError:
        print("WARN: HTTP not reachable (may be in CLI mode)")
        return True  # Non-fatal if running as CLI


def check_env() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("FAIL: ANTHROPIC_API_KEY not set")
        return False
    print(f"OK: ANTHROPIC_API_KEY set (length={len(key)})")
    return True


def check_imports() -> bool:
    try:
        import numpy, pydantic, structlog, tenacity, yfinance
        print(f"OK: Core packages importable (numpy={numpy.__version__})")
        return True
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False


def check_policy_config() -> bool:
    from pathlib import Path
    config = Path(__file__).parent.parent / "config" / "policy_rules.json"
    if config.exists():
        import json
        with open(config) as f:
            rules = json.load(f)
        print(f"OK: Policy config v{rules.get('version', '?')} loaded")
        return True
    print("FAIL: config/policy_rules.json not found")
    return False


def check_audit_chain() -> bool:
    from vector_retail.core.audit import AuditTrail
    trail = AuditTrail("healthcheck", "system")
    trail.record("health", "check", "started")
    trail.record("health", "check", "complete")
    ok = trail.verify_chain_integrity()
    print(f"{'OK' if ok else 'FAIL'}: Audit chain integrity")
    return ok


def main():
    checks = [check_http, check_env, check_imports, check_policy_config, check_audit_chain]
    results = [c() for c in checks]
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Health check: {passed}/{total} passed")
    print(f"Status: {'HEALTHY' if all(results) else 'UNHEALTHY'}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
