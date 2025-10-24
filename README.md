# RDAP Client

A Python client for querying domain registration information using the Registration Data Access Protocol (RDAP).

## What is RDAP?

RDAP (Registration Data Access Protocol) is the modern successor to WHOIS. It provides standardized, machine-readable access to domain registration data through a RESTful API. Unlike WHOIS, RDAP offers:
- Structured JSON responses
- Internationalization support
- Better security and access control
- Standardized query formats

## Features

- **Multi-resource support**: Query domains, IP addresses (IPv4/IPv6), Autonomous System Numbers (ASNs), and entities/object tags
- **Automatic resource detection**: Intelligently detects the type of resource you're querying
- **Automatic server discovery**: Uses IANA's RDAP bootstrap registries to find the correct RDAP server for any resource
- **Smart matching**: Uses most-specific prefix/range matching for IPs and ASNs (not just first match)
- **Smart caching**: Caches bootstrap data locally for 24 hours to minimize network requests
- **Persistent rate-limit tracking**: Tracks rate-limited servers per base URL and automatically avoids them for 5 minutes
- **Load distribution**: Randomly selects from available (non-rate-limited) RDAP servers
- **Retry logic**: Automatically handles rate limiting and server failures with intelligent retry
- **Performance metrics**: Collects and reports latency, success rates, and per-server statistics
- **Parallel lookups**: Supports bulk queries with ThreadPoolExecutor for high-throughput operations
- **Comprehensive logging**: Configurable logging levels for debugging and monitoring
- **Simple API**: Easy to use as a standalone script or import as a module

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd getRDAPInfo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### As a Script

Run the client directly to query multiple resource types:

```bash
# Sequential lookups (default)
python3 rdap_client.py

# Parallel bulk lookups
python3 rdap_client.py --parallel

# Quiet mode (errors only)
python3 rdap_client.py -q

# Verbose mode (debug logging)
python3 rdap_client.py -v
```

By default, it queries several examples (domain, IPv4, IPv6, ASN) and prints a metrics summary at the end.

### As a Module

Import and use in your own Python code:

```python
from rdap_client import rdap_lookup

# Automatic resource type detection - just pass any RDAP resource
result = rdap_lookup("example.com")           # Domain
result = rdap_lookup("8.8.8.8")               # IPv4 address
result = rdap_lookup("2001:4860:4860::8888")  # IPv6 address
result = rdap_lookup("AS15169")               # ASN (with or without "AS" prefix)
result = rdap_lookup("GOGL-ARIN")             # Entity/object tag

# Access the results (structure varies by resource type)
print(result.get('handle'))       # Resource handle
print(result.get('entities'))     # Related entities
print(result.get('nameservers'))  # DNS servers (domains only)
```

### Bulk Parallel Lookups

For high-throughput operations, use the bulk lookup function:

```python
from rdap_client import rdap_bulk_lookup, metrics

# Query multiple resources in parallel
queries = [
    "example.com",
    "google.com",
    "8.8.8.8",
    "1.1.1.1",
    "AS15169",
    "AS13335"
]

results = rdap_bulk_lookup(queries, max_workers=5)

# Process results
for query, result in results.items():
    if result['success']:
        data = result['data']
        print(f"{query}: {data.get('handle', 'N/A')}")
    else:
        print(f"{query}: Error - {result['error']}")

# View performance metrics
metrics.print_summary()
```

### Advanced Usage

For more control, use the lower-level functions:

```python
from rdap_client import (
    load_iana_bootstrap,
    rdap_query,
    detect_resource_type,
    rate_limit_tracker,
    metrics
)

# Manually detect resource type
resource_type, normalized_query = detect_resource_type("8.8.8.8")
print(f"Resource type: {resource_type}")  # "ipv4"

# Load specific bootstrap data
bootstrap_data = load_iana_bootstrap("ipv4")

# Query with explicit resource type
result = rdap_query("8.8.8.8", "ipv4", bootstrap_data)

# Check rate-limit status
if rate_limit_tracker.is_rate_limited("https://rdap.arin.net"):
    print("Server is currently rate-limited")

# Get metrics summary
summary = metrics.get_summary()
print(f"Success rate: {summary['success_rate']:.2f}%")
```

### Example Output

```json
{
  "objectClassName": "domain",
  "handle": "2336799_DOMAIN_COM-VRSN",
  "ldhName": "EXAMPLE.COM",
  "links": [...],
  "status": ["client delete prohibited", "client transfer prohibited", ...],
  "entities": [...],
  "nameservers": [...]
}
```

## How It Works

1. **Detect**: Automatically identifies the resource type (domain, IPv4, IPv6, ASN, or entity) based on the query format
2. **Bootstrap**: Loads the appropriate IANA bootstrap file for that resource type (cached locally for 24 hours)
3. **Route**: Finds matching RDAP servers from the bootstrap data:
   - **Domains**: Matches by TLD (.com, .org, etc.)
   - **IP addresses**: Matches by IP prefix/network range
   - **ASNs**: Matches by AS number range
   - **Entities**: Matches by registry tag (ARIN, RIPE, etc.)
4. **Query**: Constructs the appropriate RDAP URL and sends an HTTP request
5. **Retry**: If rate-limited (429) or server error occurs, automatically retries with backoff (max 5 retries)

## Configuration

All configuration constants are defined at the top of `rdap_client.py`:

- **Cache TTL**: `CACHE_TTL` (default: 86400 seconds / 24 hours)
- **Cache location**: `CACHE_DIR` (default: `iana_rdap_cache/` directory)
- **Bootstrap URLs**: `IANA_BOOTSTRAP_URLS` dictionary for custom bootstrap sources
- **Rate limit backoff**: `RATE_LIMIT_BACKOFF` (default: 300 seconds / 5 minutes)
- **HTTP timeout**: 10 seconds per request
- **Max retries**: 5 retries per query (configurable via `max_retries` parameter)
- **Retry delay**: 2-second delay after rate limiting before retry
- **Parallel workers**: Configurable via `max_workers` parameter in `rdap_bulk_lookup()` (default: 5)

### Logging Configuration

Set logging level programmatically or via command-line flags:

```python
import logging
from rdap_client import logger

# Set to DEBUG for detailed information
logger.setLevel(logging.DEBUG)

# Set to WARNING to see only warnings and errors
logger.setLevel(logging.WARNING)
```

Command-line flags:
- `-v` or `--verbose`: DEBUG level
- `-q` or `--quiet`: ERROR level only
- Default: INFO level

## Requirements

- Python 3.6+
- `requests` library
- `urllib3` v1.x (for LibreSSL compatibility on macOS)

**Note for macOS users**: If you see SSL warnings about LibreSSL, the requirements.txt pins urllib3 to v1.x for compatibility.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
