#!/usr/bin/env python3
"""

NOTE (FIX-043): the fixture auth.py contains a deliberate bare `except:`
clause (and other intentional code smells) so the built-in analyzer flags
them as findings during demo runs. This is the only place in the codebase
where such a pattern is acceptable — production code must use specific
exception types (lint-enforced via ruff E722 with this file exempted).
Create the demo fixture repository ('demo-web-shop') with intentional issues.

This synthetic repo gives the pipeline realistic material:
- security findings (SQLi string concat, eval, hardcoded secret, weak hash)
- bugs (bare except, mutable default args)
- code smells (high complexity, long functions, deep nesting)
- SATD comments (TODO/FIXME)
- git history: multiple commits, authors, hotspots (auth.py churns most)
- cross-file call graph (api → auth → utils), docs, issues fixture
"""
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

AUTH_PY = '''"""Authentication module for demo-web-shop."""
import hashlib
import sqlite3

# TODO(security): move this to environment config before launch
SECRET_KEY = "super-secret-key-9f8e7d6c5b4a"

FAILED_LOGINS = {}
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


def hash_password(password):
    """Hash a password. FIXME: md5 is weak — migrate to bcrypt."""
    return hashlib.md5(password.encode()).hexdigest()


def validate_credentials(username, password):
    """Validate user credentials against the database."""
    conn = sqlite3.connect("shop.db")
    cursor = conn.cursor()
    # FIXME: string concatenation — SQL injection risk
    query = "SELECT id, password_hash, is_active FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return False
    return row[1] == hash_password(password) and row[2] == 1


def login(username, password, max_retries=3, session_data={}):
    """Login a user with lockout logic and retry handling.

    Long, complex function on purpose — exercises complexity findings.
    """
    if username is None or password is None:
        return None
    if username == "":
        return None
    attempts = FAILED_LOGINS.get(username, 0)
    if attempts >= MAX_ATTEMPTS:
        last_attempt = session_data.get("last_attempt")
        if last_attempt is not None:
            elapsed = datetime.now() - last_attempt
            if elapsed.total_seconds() < LOCKOUT_SECONDS:
                return None
            else:
                FAILED_LOGINS[username] = 0
                attempts = 0
        else:
            return None
    ok = False
    for attempt in range(max_retries):
        try:
            ok = validate_credentials(username, password)
            break
        except sqlite3.OperationalError:
            if attempt == max_retries - 1:
                raise
        except:
            # XXX: silently swallowing errors
            break
    if ok:
        FAILED_LOGINS.pop(username, None)
        session_data["user"] = username
        session_data["last_attempt"] = None
        return create_session(username)
    else:
        FAILED_LOGINS[username] = attempts + 1
        session_data["last_attempt"] = datetime.now()
        return None


def create_session(username):
    """Create session token for user."""
    # TODO: use signed JWT sessions
    token = username + ":" + SECRET_KEY
    return token


def check_permission(user, action):
    """Very nested permission logic (deep nesting smell)."""
    if user:
        if user.get("role"):
            if action in ("read", "list"):
                if user["role"] in ("admin", "user", "viewer"):
                    return True
                else:
                    if user["role"] == "guest":
                        return action == "list"
                    else:
                        return False
            else:
                if user["role"] == "admin":
                    if action in ("write", "delete", "read", "list"):
                        return True
                    else:
                        return False
                else:
                    return False
        else:
            return False
    else:
        return False


def evaluate_rule(rule_expr, context):
    """Dynamic rule evaluation. HACK: eval used for legacy rules."""
    if rule_expr.startswith("legacy:"):
        return eval(rule_expr[len("legacy:"):], {}, context)
    return bool(context.get(rule_expr))
'''

CART_PY = '''"""Shopping cart module."""
from demo_shop.utils import calculate_total, format_currency


class Cart:
    """Shopping cart with items and totals."""

    def __init__(self, items=None):
        self.items = items or []

    def add_item(self, product, quantity=1):
        """Add product to cart."""
        for entry in self.items:
            if entry["product"].sku == product.sku:
                entry["quantity"] += quantity
                return
        self.items.append({"product": product, "quantity": quantity})

    def remove_item(self, sku):
        """Remove item by SKU."""
        self.items = [e for e in self.items if e["product"].sku != sku]

    def total(self):
        """Cart total."""
        return calculate_total(self.items)

    def summary(self):
        """Human-readable summary."""
        return format_currency(self.total())
'''

UTILS_PY = '''"""Shared utilities."""
import json


def calculate_total(items):
    """Sum item prices."""
    total = 0
    for entry in items:
        total += entry["product"].price * entry["quantity"]
    return total


def format_currency(amount):
    """Format a currency amount."""
    return "$" + f"{amount:,.2f}"


def parse_config(raw):
    """Parse JSON config, tolerating bad input."""
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def slugify(text):
    """Convert text to slug."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    return "".join(out).strip("-")
'''

API_PY = '''"""HTTP API layer for demo-web-shop."""
from demo_shop.auth import check_permission, evaluate_rule, login
from demo_shop.cart import Cart
from demo_shop.models import Product
from demo_shop.utils import parse_config

CONFIG = parse_config("{}")


def handle_login(request):
    """POST /login handler."""
    body = request.get("body", {})
    token = login(body.get("username"), body.get("password"))
    if token is None:
        return {"status": 401, "error": "invalid credentials or locked out"}
    return {"status": 200, "token": token}


def handle_add_to_cart(request):
    """POST /cart handler."""
    user = request.get("user")
    if not check_permission(user, "write"):
        return {"status": 403, "error": "forbidden"}
    product = Product.from_dict(request["body"]["product"])
    cart = Cart(request.get("cart_items", []))
    cart.add_item(product, request["body"].get("quantity", 1))
    return {"status": 200, "cart": cart.summary()}


def handle_discount(request):
    """Apply discount rule."""
    rule = request["body"]["rule"]
    context = request["body"].get("context", {})
    return {"status": 200, "applied": evaluate_rule(rule, context)}
'''

MODELS_PY = '''"""Data models for demo-web-shop."""


class Product:
    """A purchasable product."""

    def __init__(self, sku, name, price, category="general"):
        self.sku = sku
        self.name = name
        self.price = price
        self.category = category

    @classmethod
    def from_dict(cls, data):
        """Build product from dict."""
        return cls(data["sku"], data["name"], data["price"], data.get("category", "general"))

    def to_dict(self):
        """Serialize product."""
        return {"sku": self.sku, "name": self.name,
                "price": self.price, "category": self.category}


class User:
    """A shop user with a role."""

    def __init__(self, username, role="user"):
        self.username = username
        self.role = role

    def to_dict(self):
        """Serialize user."""
        return {"username": self.username, "role": self.role}
'''

TEST_API_PY = '''"""Tests for the demo API."""
from demo_shop.api import handle_add_to_cart, handle_login
from demo_shop.models import User


def test_login_rejects_unknown_user():
    result = handle_login({"body": {"username": "ghost", "password": "x"}})
    assert result["status"] == 401


def test_add_to_cart_requires_permission():
    user = User("alice", "viewer")
    result = handle_add_to_cart({"user": user.to_dict(), "body": {}})
    assert result["status"] == 403
'''

README_MD = '''# demo-web-shop

A tiny demonstration web-shop backend used as a RepoIntel AI fixture.

## Architecture

- `demo_shop/auth.py` — authentication, sessions, permissions
- `demo_shop/cart.py` — shopping cart
- `demo_shop/models.py` — Product/User models
- `demo_shop/api.py` — HTTP handlers
- `demo_shop/utils.py` — helpers

## Known issues

See issues tracker. The auth module historically concentrates most defects
and change activity (hotspot).
'''

DOCS_ARCH_MD = '''# Architecture Notes

The system is a layered design:

1. **API layer** (`api.py`) validates permissions via `auth.py`.
2. **Auth layer** (`auth.py`) talks to SQLite and issues session tokens.
3. **Domain layer** (`cart.py`, `models.py`) holds business rules.
4. **Utility layer** (`utils.py`) provides shared helpers.

Login flow: `handle_login` → `login` → `validate_credentials` → `hash_password`.
Lockout policy: MAX_ATTEMPTS failures lock the account for LOCKOUT_SECONDS.
'''

ISSUES_FIXTURE = {
    "issues": [
        {"number": 1, "title": "Login slow when database under load", "body": "Login takes 4+ seconds during checkout spikes. Suspect the retry loop and lockout check in auth.login.",
         "labels": ["performance", "auth"], "state": "open"},
        {"number": 2, "title": "SQL injection in user lookup", "body": "validate_credentials concatenates username into the SQL query. Parameterize it.",
         "labels": ["security", "bug"], "state": "open"},
        {"number": 3, "title": "Passwords hashed with MD5", "body": "hash_password uses MD5. Migrate to bcrypt with rehash-on-login.",
         "labels": ["security", "tech-debt"], "state": "open"},
        {"number": 4, "title": "eval() in rule engine crashes on bad rules", "body": "evaluate_rule uses eval for legacy rules and raises on malformed input.",
         "labels": ["bug", "security"], "state": "open"},
    ],
    "pull_requests": [
        {"number": 10, "title": "Add lockout to login", "body": "Implements MAX_ATTEMPTS lockout.",
         "reviewers": ["reviewer-a"], "state": "merged"},
        {"number": 11, "title": "Refactor cart totals", "body": "Extracts calculate_total to utils.",
         "reviewers": ["reviewer-b"], "state": "merged"},
    ],
}


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def commit(repo: Path, date: datetime, author: str, email: str, message: str, files: dict[str, str]):
    for rel_path, content in files.items():
        target = repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    run(["git", "add", "-A"], repo)
    env_date = date.strftime("%Y-%m-%dT%H:%M:%S")
    run(["git", "-c", f"user.name={author}", "-c", f"user.email={email}",
         "commit", "-m", message, "--date", env_date], repo)


def main(repo_path: str = "data/repos/demo-web-shop") -> Path:
    repo = Path(repo_path).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    if (repo / ".git").exists():
        return repo

    run(["git", "init", "-b", "main"], repo)
    base = datetime(2026, 1, 6, 10, 0, 0)
    pkg = "demo_shop"

    # commit 1: skeleton + utils + models
    commit(repo, base, "Ada Lovelace", "ada@shop.dev", "Initial skeleton: models + utils", {
        f"{pkg}/__init__.py": "",
        f"{pkg}/models.py": MODELS_PY,
        f"{pkg}/utils.py": UTILS_PY,
        "README.md": README_MD,
    })
    # commit 2: auth v1
    commit(repo, base + timedelta(days=3), "Grace Hopper", "grace@shop.dev", "Add authentication module", {
        f"{pkg}/auth.py": AUTH_PY,
    })
    # commit 3: cart + api
    commit(repo, base + timedelta(days=6), "Ada Lovelace", "ada@shop.dev", "Add cart and API layer", {
        f"{pkg}/cart.py": CART_PY,
        f"{pkg}/api.py": API_PY,
    })
    # commit 4: docs
    commit(repo, base + timedelta(days=9), "Linus T", "linus@shop.dev", "Document architecture", {
        "docs/ARCHITECTURE.md": DOCS_ARCH_MD,
    })
    # commit 5: tests (auth churn again — hotspot)
    commit(repo, base + timedelta(days=12), "Grace Hopper", "grace@shop.dev", "Add API tests; touch auth", {
        "tests/__init__.py": "",
        "tests/test_api.py": TEST_API_PY,
    })
    # commit 6: auth churn 3 (hotspot confirmed)
    commit(repo, base + timedelta(days=15), "Grace Hopper", "grace@shop.dev", "Tweak lockout constants in auth", {
        f"{pkg}/auth.py": AUTH_PY.replace("MAX_ATTEMPTS = 5", "MAX_ATTEMPTS = 5  # tuned after incident 12"),
    })
    # commit 7: fixture issues
    fixture_dir = repo / ".repointel"
    fixture_dir.mkdir(exist_ok=True)
    (fixture_dir / "issues.json").write_text(json.dumps(ISSUES_FIXTURE, indent=2), encoding="utf-8")
    commit(repo, base + timedelta(days=16), "Ada Lovelace", "ada@shop.dev", "Add local issue fixtures", {
        ".repointel/issues.json": (fixture_dir / "issues.json").read_text(),
    })
    print(f"Demo repo created at {repo}")
    return repo


if __name__ == "__main__":
    main()
