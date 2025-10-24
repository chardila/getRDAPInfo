# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive RDAP (Registration Data Access Protocol) client that queries information about multiple internet resource types: domains, IP addresses (IPv4/IPv6), Autonomous System Numbers (ASNs), and entities/object tags. The client automatically detects resource types and intelligently selects appropriate RDAP servers using IANA's bootstrap registries.

## Core Architecture

**Single-file design**: All functionality is contained in `rdap_client.py`.

**Multi-layer architecture**:
1. **Detection Layer** (`detect_resource_type`): Automatically identifies resource type (domain, IPv4, IPv6, ASN, entity) based on query format
2. **Bootstrap Layer** (`load_iana_bootstrap`): Fetches and caches IANA RDAP bootstrap files for each resource type
3. **Rate-Limit Tracking** (`RateLimitTracker`): Tracks rate-limited servers and their cooldown periods
4. **Routing Layer** (`get_rdap_server` + type-specific functions): Determines which RDAP server to use with smart matching and rate-limit filtering
5. **Query Layer** (`rdap_query` + `rdap_lookup` + `rdap_bulk_lookup`): Performs RDAP lookups with metrics tracking
6. **Metrics Layer** (`RDAPMetrics`): Collects performance statistics (latency, success rates, server stats)

**Multi-bootstrap support**:
- Five bootstrap types: `dns`, `ipv4`, `ipv6`, `asn`, `object-tags`
- Each bootstrap file cached separately in `iana_rdap_cache/` directory
- Bootstrap URLs defined in `IANA_BOOTSTRAP_URLS` dictionary (lines 16-22)

**Smart resource-specific routing**:
- **DNS** (`get_rdap_server_for_dns`): Matches by TLD extraction, filters rate-limited servers
- **IP** (`get_rdap_server_for_ip`): Finds **most specific network match** (longest prefix) using `ipaddress` module, filters rate-limited servers
- **ASN** (`get_rdap_server_for_asn`): Finds **smallest matching range** (most specific), parses string ranges like "1-1876", filters rate-limited servers
- **Entity** (`get_rdap_server_for_entity`): Matches by registry tag extraction, filters rate-limited servers

**Key mechanisms**:
- **Persistent rate-limit tracking**: Stores timestamp per server base URL, skips for 300 seconds (configurable)
- **Smart matching**: Always selects most specific match (smallest network/range) rather than first match
- **Server filtering**: All routing functions filter out currently rate-limited servers before selection
- **Caching**: Each bootstrap type cached separately with 24-hour TTL
- **Server rotation**: Randomly selects from available (non-rate-limited) RDAP servers
- **Retry logic**: Max retry counter (default: 5) with intelligent backoff
- **Metrics collection**: Tracks every query with timestamp, server, latency, success/failure
- **Parallel execution**: ThreadPoolExecutor for bulk lookups with configurable worker count

## Running the Code

**Execute the client**:
```bash
python3 rdap_client.py              # Sequential lookups
python3 rdap_client.py --parallel   # Parallel bulk lookups
python3 rdap_client.py -v           # Verbose (DEBUG logging)
python3 rdap_client.py -q           # Quiet (ERROR logging only)
```

**Single query**:
```python
from rdap_client import rdap_lookup

# Unified interface - auto-detects resource type
result = rdap_lookup("example.com")     # Domain
result = rdap_lookup("8.8.8.8")         # IPv4
result = rdap_lookup("AS15169")         # ASN
result = rdap_lookup("GOGL-ARIN")       # Entity
```

**Bulk parallel queries**:
```python
from rdap_client import rdap_bulk_lookup, metrics

queries = ["example.com", "8.8.8.8", "AS15169"]
results = rdap_bulk_lookup(queries, max_workers=5)

# Print metrics after bulk operation
metrics.print_summary()
```

**Advanced usage** (manual control):
```python
from rdap_client import (
    load_iana_bootstrap,
    rdap_query,
    detect_resource_type,
    rate_limit_tracker,
    metrics
)

# Detect resource type
rtype, normalized = detect_resource_type("8.8.8.8")

# Load specific bootstrap
bootstrap = load_iana_bootstrap("ipv4")

# Query with explicit parameters
result = rdap_query("8.8.8.8", "ipv4", bootstrap, max_retries=3)

# Check if server is rate-limited
if rate_limit_tracker.is_rate_limited("https://rdap.arin.net"):
    print("Server currently blocked")

# Access metrics
summary = metrics.get_summary()
```

## Dependencies

Standard library plus `requests`. Install with:
```bash
pip install requests
```

## Important Implementation Details

### Rate Limiting & Server Selection
- **RateLimitTracker** (lines 35-71): Maintains dictionary of rate-limited servers with timestamps
- **Cooldown period**: 300 seconds (5 minutes) - configurable via `RATE_LIMIT_BACKOFF`
- **Server filtering**: All `get_rdap_server_for_*` functions call `rate_limit_tracker.get_available_servers()` before selection
- **Fallback behavior**: If all servers are rate-limited, picks one anyway (will retry with backoff)

### Smart Matching Logic
- **IP matching** (lines 237-272): Collects ALL matching networks, sorts by `prefixlen` (descending) to get most specific
- **ASN matching** (lines 274-319): Parses string ranges ("1-1876"), collects all matches, sorts by range size (ascending)
- **ASN format handling**: Supports string ranges ("1-1876"), arrays ([1, 1876]), and single integers

### Metrics & Logging
- **RDAPMetrics** (lines 74-160): Tracks every query with server, resource_type, success, latency, status_code
- **Global instances**: `rate_limit_tracker` and `metrics` are module-level singletons (lines 163-165)
- **Logging**: Configured at module level (lines 27-32), adjustable via command-line flags or programmatically

### Parallel Execution
- **rdap_bulk_lookup** (lines 402-437): Uses ThreadPoolExecutor with configurable worker count
- **Thread safety**: RateLimitTracker has `_lock` placeholder for future thread-safe operations
- **Result aggregation**: Returns dict mapping query -> {success, data/error}

### Other Details
- **Retry logic**: Explicit `retry_count` parameter (max 5 retries) prevents stack overflow
- **Cache structure**: Bootstrap files in `iana_rdap_cache/` with format `iana_rdap_{type}.json`
- **Resource detection** (lines 197-220): Uses regex and `ipaddress` module. Entity tag requires "HANDLE-REGISTRY" format
- **RDAP URL construction** (lines 351-361): Different paths per type (domain/, ip/, autnum/, entity/)
- **Metrics recording** (lines 382, 377, 389): Called on every query attempt, success or failure
