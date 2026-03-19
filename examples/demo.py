#!/usr/bin/env python3
"""
holo-code-gen demo: Generate 5 different code units from specs.

Run with: python3 examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo_code_gen import (
    CodeSpec, InputParam, OutputParam, Example,
    HoloCodeGenerator,
)

gen = HoloCodeGenerator()

SEPARATOR = "=" * 60


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ------------------------------------------------------------------ #
# 1. Python function: calculate_compound_interest                      #
# ------------------------------------------------------------------ #
section("1. Python Function — calculate_compound_interest")

spec1 = CodeSpec(
    name="calculate_compound_interest",
    template="function",
    language="python",
    description="Calculate compound interest given principal, rate, time, and compounding frequency.",
    inputs=[
        InputParam("principal", "float", "Initial investment amount"),
        InputParam("rate", "float", "Annual interest rate as a decimal (e.g., 0.05 for 5%)"),
        InputParam("time", "float", "Time in years"),
        InputParam("n", "int", "Compounding frequency per year", default=12),
    ],
    output=OutputParam("float", "Final amount after compound interest"),
    constraints=["rate must be >= 0", "principal must be > 0", "n must be >= 1"],
    examples=[
        Example(
            inputs={"principal": 1000.0, "rate": 0.05, "time": 1.0, "n": 12},
            expected_output=1051.16,
            description="$1000 at 5% for 1 year, monthly compounding",
        ),
    ],
    tags=["finance", "math"],
)

result1 = gen.generate(spec1)
print(result1)


# ------------------------------------------------------------------ #
# 2. TypeScript class — UserSessionManager                            #
# ------------------------------------------------------------------ #
section("2. TypeScript Class — UserSessionManager")

spec2 = CodeSpec(
    name="UserSessionManager",
    template="class",
    language="typescript",
    description="Manages active user sessions with TTL-based expiry.",
    inputs=[
        InputParam("maxSessions", "number", "Maximum concurrent sessions allowed"),
        InputParam("ttlSeconds", "number", "Session time-to-live in seconds"),
    ],
    output=OutputParam("boolean", "Whether the operation succeeded"),
    constraints=["maxSessions must be > 0", "ttlSeconds must be >= 60"],
    examples=[
        Example(
            inputs={"userId": "u_123", "token": "abc"},
            expected_output=True,
            description="create session",
        ),
        Example(
            inputs={"token": "abc"},
            expected_output=True,
            description="invalidate session",
        ),
    ],
    tags=["auth", "session"],
)

result2 = gen.generate(spec2)
print(result2)


# ------------------------------------------------------------------ #
# 3. Python REST endpoint — create_user                               #
# ------------------------------------------------------------------ #
section("3. Python REST Endpoint — create_user")

spec3 = CodeSpec(
    name="create_user",
    template="rest_endpoint",
    language="python",
    description="Create a new user account. Returns 201 with user ID on success.",
    inputs=[
        InputParam("username", "str", "Unique username (3-32 chars, alphanumeric)"),
        InputParam("email", "str", "Valid email address"),
        InputParam("password", "str", "Password (min 8 chars, hashed before storage)"),
    ],
    output=OutputParam("dict", "Created user data with id, username, email"),
    constraints=[
        "username must be unique",
        "email must be valid format",
        "password is never stored in plaintext",
    ],
    tags=["api", "users"],
    dependencies=["hashlib"],
)

result3 = gen.generate(spec3)
print(result3)


# ------------------------------------------------------------------ #
# 4. TypeScript data model — ProductListing                           #
# ------------------------------------------------------------------ #
section("4. TypeScript Data Model — ProductListing")

spec4 = CodeSpec(
    name="ProductListing",
    template="data_model",
    language="typescript",
    description="Represents a product listing in an e-commerce catalog.",
    inputs=[
        InputParam("id", "string", "Unique product identifier"),
        InputParam("name", "string", "Product display name"),
        InputParam("price", "number", "Price in USD cents"),
        InputParam("category", "string", "Product category slug"),
        InputParam("inStock", "boolean", "Whether the product is available", required=False),
        InputParam("tags", "string[]", "Searchable tags", required=False),
    ],
    constraints=["price must be >= 0", "id must be globally unique"],
    tags=["ecommerce", "catalog"],
)

result4 = gen.generate(spec4)
print(result4)


# ------------------------------------------------------------------ #
# 5. Python test suite — validate_email                               #
# ------------------------------------------------------------------ #
section("5. Python Test Suite — validate_email")

spec5 = CodeSpec(
    name="validate_email",
    template="test_suite",
    language="python",
    description="Test suite for the validate_email function.",
    inputs=[
        InputParam("email", "str", "Email address to validate"),
    ],
    output=OutputParam("bool", "True if valid email format"),
    examples=[
        Example(
            inputs={"email": "user@example.com"},
            expected_output=True,
            description="valid email",
        ),
        Example(
            inputs={"email": "not-an-email"},
            expected_output=False,
            description="missing at-sign",
        ),
        Example(
            inputs={"email": "@nodomain.com"},
            expected_output=False,
            description="empty local part",
        ),
        Example(
            inputs={"email": "user@"},
            expected_output=False,
            description="missing domain",
        ),
    ],
    tags=["validation", "email"],
)

result5 = gen.generate(spec5)
print(result5)


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
section("Summary")

results = [result1, result2, result3, result4, result5]
names = [r.spec.name for r in results]
statuses = ["✅ PASS" if r.success else "❌ FAIL" for r in results]
for name, status in zip(names, statuses):
    print(f"  {status}  {name}")

total_pass = sum(1 for r in results if r.success)
print(f"\n{total_pass}/{len(results)} passed validation")
