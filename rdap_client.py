import ipaddress
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

IANA_BOOTSTRAP_URLS = {
    "dns": "https://data.iana.org/rdap/dns.json",
    "ipv4": "https://data.iana.org/rdap/ipv4.json",
    "ipv6": "https://data.iana.org/rdap/ipv6.json",
    "asn": "https://data.iana.org/rdap/asn.json",
    "object-tags": "https://data.iana.org/rdap/object-tags.json"
}
CACHE_DIR = "iana_rdap_cache"
CACHE_TTL = 86400  # refresh daily
RATE_LIMIT_BACKOFF = 300  # seconds to avoid a rate-limited server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimitTracker:
    """Track rate-limited servers and their cooldown periods."""

    def __init__(self, backoff_seconds: int = RATE_LIMIT_BACKOFF):
        self.backoff_seconds = backoff_seconds
        self.rate_limited: Dict[str, datetime] = {}
        self._lock = None  # For thread safety if needed

    def mark_rate_limited(self, server_url: str):
        """Mark a server as rate-limited."""
        base_url = self._get_base_url(server_url)
        self.rate_limited[base_url] = datetime.now()
        logger.warning(f"Server {base_url} rate-limited until {datetime.now() + timedelta(seconds=self.backoff_seconds)}")

    def is_rate_limited(self, server_url: str) -> bool:
        """Check if a server is currently rate-limited."""
        base_url = self._get_base_url(server_url)
        if base_url not in self.rate_limited:
            return False

        elapsed = (datetime.now() - self.rate_limited[base_url]).total_seconds()
        if elapsed > self.backoff_seconds:
            # Cooldown period expired
            del self.rate_limited[base_url]
            return False

        return True

    def get_available_servers(self, server_urls: List[str]) -> List[str]:
        """Filter out rate-limited servers from a list."""
        return [url for url in server_urls if not self.is_rate_limited(url)]

    @staticmethod
    def _get_base_url(url: str) -> str:
        """Extract base URL (scheme + netloc) from a full URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"


class RDAPMetrics:
    """Collect and report metrics about RDAP queries."""

    def __init__(self):
        self.queries: List[Dict] = []
        self.server_stats: Dict[str, Dict] = defaultdict(lambda: {
            'queries': 0,
            'successes': 0,
            'failures': 0,
            'rate_limits': 0,
            'total_latency': 0.0
        })

    def record_query(self, server: str, resource_type: str, query: str,
                     success: bool, latency: float, status_code: Optional[int] = None):
        """Record a query attempt."""
        self.queries.append({
            'timestamp': datetime.now(),
            'server': server,
            'resource_type': resource_type,
            'query': query,
            'success': success,
            'latency': latency,
            'status_code': status_code
        })

        base_url = RateLimitTracker._get_base_url(server)
        stats = self.server_stats[base_url]
        stats['queries'] += 1
        stats['total_latency'] += latency

        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

        if status_code == 429:
            stats['rate_limits'] += 1

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_queries = len(self.queries)
        if total_queries == 0:
            return {'total_queries': 0}

        successful = sum(1 for q in self.queries if q['success'])
        total_latency = sum(q['latency'] for q in self.queries)

        return {
            'total_queries': total_queries,
            'successful': successful,
            'failed': total_queries - successful,
            'success_rate': successful / total_queries * 100,
            'avg_latency': total_latency / total_queries,
            'servers_used': len(self.server_stats),
            'server_stats': dict(self.server_stats)
        }

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        if summary['total_queries'] == 0:
            print("No queries recorded yet.")
            return

        print("\n" + "="*60)
        print("RDAP METRICS SUMMARY")
        print("="*60)
        print(f"Total Queries:    {summary['total_queries']}")
        print(f"Successful:       {summary['successful']}")
        print(f"Failed:           {summary['failed']}")
        print(f"Success Rate:     {summary['success_rate']:.2f}%")
        print(f"Avg Latency:      {summary['avg_latency']:.3f}s")
        print(f"Servers Used:     {summary['servers_used']}")

        print("\nPer-Server Statistics:")
        print("-" * 60)
        for server, stats in summary['server_stats'].items():
            if stats['queries'] > 0:
                avg_lat = stats['total_latency'] / stats['queries']
                srv_success_rate = stats['successes'] / stats['queries'] * 100
                print(f"\n{server}")
                print(f"  Queries: {stats['queries']}, Success: {stats['successes']}, "
                      f"Failed: {stats['failures']}, Rate-limits: {stats['rate_limits']}")
                print(f"  Success Rate: {srv_success_rate:.2f}%, Avg Latency: {avg_lat:.3f}s")
        print("="*60 + "\n")


# Global instances
rate_limit_tracker = RateLimitTracker()
metrics = RDAPMetrics()

def load_iana_bootstrap(bootstrap_type="dns"):
    """Load the IANA RDAP bootstrap file for a specific type (refresh if stale)."""
    if bootstrap_type not in IANA_BOOTSTRAP_URLS:
        raise ValueError(f"Invalid bootstrap type: {bootstrap_type}")

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(CACHE_DIR, f"iana_rdap_{bootstrap_type}.json")

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < CACHE_TTL:
            with open(cache_file, "r") as f:
                return json.load(f)

    print(f"Fetching latest IANA RDAP bootstrap for {bootstrap_type}...")
    resp = requests.get(IANA_BOOTSTRAP_URLS[bootstrap_type], timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    return data

def detect_resource_type(query):
    """Detect the type of RDAP resource based on the query string."""
    query = query.strip()

    # Check if it's an IP address
    try:
        ip = ipaddress.ip_address(query)
        return "ipv6" if ip.version == 6 else "ipv4", query
    except ValueError:
        pass

    # Check if it's an IP network (CIDR notation)
    try:
        network = ipaddress.ip_network(query, strict=False)
        return "ipv6" if network.version == 6 else "ipv4", str(network)
    except ValueError:
        pass

    # Check if it's an ASN (AS12345 or just 12345)
    asn_match = re.match(r'^(?:AS)?(\d+)$', query, re.IGNORECASE)
    if asn_match:
        return "asn", asn_match.group(1)

    # Check if it's an entity/object tag (format varies, but typically alphanumeric with hyphens)
    # Entity tags often contain a registry tag like "ABC123-ARIN"
    if re.match(r'^[A-Z0-9]+-[A-Z]+$', query, re.IGNORECASE):
        return "object-tags", query

    # Default to domain/DNS
    return "dns", query

def get_rdap_server_for_dns(domain, bootstrap_data):
    """Return available RDAP server URLs for the given domain (filtered by rate limits)."""
    tld = domain.strip().split(".")[-1].lower()
    for service in bootstrap_data["services"]:
        tlds, urls = service
        if tld in tlds:
            # Filter out rate-limited servers
            available = rate_limit_tracker.get_available_servers(urls)
            if available:
                return random.choice(available)
            # If all servers are rate-limited, return any (will be handled in retry logic)
            logger.warning(f"All servers for {tld} are rate-limited, using fallback")
            return random.choice(urls)
    return None

def get_rdap_server_for_ip(ip_str, bootstrap_data):
    """Return RDAP server URL for the given IP, using most specific network match."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Try as network
        ip = ipaddress.ip_network(ip_str, strict=False).network_address

    # Find all matching networks and select the most specific (smallest prefix)
    matches = []
    for service in bootstrap_data["services"]:
        prefixes, urls = service
        for prefix in prefixes:
            try:
                network = ipaddress.ip_network(prefix)
                if ip in network:
                    matches.append((network, urls))
            except ValueError:
                continue

    if not matches:
        return None

    # Sort by prefix length (descending) to get most specific match
    matches.sort(key=lambda x: x[0].prefixlen, reverse=True)
    most_specific_network, urls = matches[0]

    logger.debug(f"IP {ip} matched to network {most_specific_network} (most specific of {len(matches)} matches)")

    # Filter out rate-limited servers
    available = rate_limit_tracker.get_available_servers(urls)
    if available:
        return random.choice(available)

    logger.warning(f"All servers for {most_specific_network} are rate-limited, using fallback")
    return random.choice(urls)

def get_rdap_server_for_asn(asn, bootstrap_data):
    """Return RDAP server URL for the given ASN, using most specific range match."""
    asn_num = int(asn)

    # Find all matching ranges and select the most specific (smallest range)
    matches = []
    for service in bootstrap_data["services"]:
        ranges, urls = service
        for asn_range in ranges:
            # Parse range - can be string "start-end" or array [start, end] or single int
            if isinstance(asn_range, str):
                # String format: "start-end"
                if '-' in asn_range:
                    start, end = map(int, asn_range.split('-'))
                else:
                    # Single ASN as string
                    start = end = int(asn_range)
            elif isinstance(asn_range, list) and len(asn_range) == 2:
                # Array format: [start, end]
                start, end = asn_range
            elif isinstance(asn_range, int):
                # Single ASN as int
                start = end = asn_range
            else:
                continue

            if start <= asn_num <= end:
                range_size = end - start + 1
                matches.append((range_size, [start, end], urls))

    if not matches:
        return None

    # Sort by range size (ascending) to get most specific match
    matches.sort(key=lambda x: x[0])
    range_size, asn_range, urls = matches[0]

    logger.debug(f"ASN {asn_num} matched to range {asn_range} (most specific of {len(matches)} matches)")

    # Filter out rate-limited servers
    available = rate_limit_tracker.get_available_servers(urls)
    if available:
        return random.choice(available)

    logger.warning(f"All servers for ASN range {asn_range} are rate-limited, using fallback")
    return random.choice(urls)

def get_rdap_server_for_entity(entity_tag, bootstrap_data):
    """Return RDAP server URL for the given entity/object tag (filtered by rate limits)."""
    # Entity tags are typically in format "HANDLE-REGISTRY"
    # Extract the registry tag (part after the last hyphen)
    parts = entity_tag.split("-")
    if len(parts) >= 2:
        registry_tag = parts[-1].upper()
        for service in bootstrap_data["services"]:
            tags, urls = service
            if registry_tag in [t.upper() for t in tags]:
                # Filter out rate-limited servers
                available = rate_limit_tracker.get_available_servers(urls)
                if available:
                    return random.choice(available)
                logger.warning(f"All servers for registry {registry_tag} are rate-limited, using fallback")
                return random.choice(urls)
    return None

def get_rdap_server(query, resource_type, bootstrap_data):
    """Route to the appropriate server selection function based on resource type."""
    if resource_type == "dns":
        return get_rdap_server_for_dns(query, bootstrap_data)
    elif resource_type in ("ipv4", "ipv6"):
        return get_rdap_server_for_ip(query, bootstrap_data)
    elif resource_type == "asn":
        return get_rdap_server_for_asn(query, bootstrap_data)
    elif resource_type == "object-tags":
        return get_rdap_server_for_entity(query, bootstrap_data)
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")

def rdap_query(query, resource_type, bootstrap_data, retry_count=0, max_retries=5):
    """Query RDAP for a resource with metrics tracking and rate-limit handling."""
    if retry_count >= max_retries:
        logger.error(f"Max retries ({max_retries}) exceeded for {query}")
        raise ValueError(f"Max retries exceeded for {query}")

    server = get_rdap_server(query, resource_type, bootstrap_data)
    if not server:
        raise ValueError(f"No RDAP server found for {query} (type: {resource_type})")

    # Construct the appropriate RDAP URL based on resource type
    if resource_type == "dns":
        path = f"domain/{query}"
    elif resource_type in ("ipv4", "ipv6"):
        path = f"ip/{query}"
    elif resource_type == "asn":
        path = f"autnum/{query}"
    elif resource_type == "object-tags":
        path = f"entity/{query}"
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")

    rdap_url = f"{server.rstrip('/')}/{path}"
    logger.info(f"Querying {rdap_url} (attempt {retry_count + 1}/{max_retries})")

    start_time = time.time()
    status_code = None

    try:
        resp = requests.get(rdap_url, timeout=10)
        status_code = resp.status_code
        latency = time.time() - start_time

        if resp.status_code == 429:
            logger.warning(f"Rate-limited by {server}, marking and rotating...")
            rate_limit_tracker.mark_rate_limited(server)
            metrics.record_query(server, resource_type, query, False, latency, status_code)
            time.sleep(2)
            return rdap_query(query, resource_type, bootstrap_data, retry_count + 1, max_retries)

        resp.raise_for_status()
        metrics.record_query(server, resource_type, query, True, latency, status_code)
        logger.info(f"Successfully queried {server} in {latency:.3f}s")
        return resp.json()

    except requests.RequestException as e:
        latency = time.time() - start_time
        logger.warning(f"Error querying {server}: {e}")
        metrics.record_query(server, resource_type, query, False, latency, status_code)
        # Rotate to another server
        return rdap_query(query, resource_type, bootstrap_data, retry_count + 1, max_retries)

def rdap_lookup(query):
    """Unified RDAP lookup function that auto-detects resource type."""
    resource_type, normalized_query = detect_resource_type(query)
    logger.info(f"Detected resource type: {resource_type}")

    bootstrap_data = load_iana_bootstrap(resource_type)
    return rdap_query(normalized_query, resource_type, bootstrap_data)


def rdap_bulk_lookup(queries: List[str], max_workers: int = 5) -> Dict[str, Dict]:
    """
    Perform parallel RDAP lookups for multiple queries.

    Args:
        queries: List of queries (domains, IPs, ASNs, entities)
        max_workers: Maximum number of concurrent threads

    Returns:
        Dictionary mapping queries to their results (or error info)
    """
    results = {}

    def lookup_single(query: str) -> Tuple[str, Dict]:
        """Helper function to lookup a single query."""
        try:
            result = rdap_lookup(query)
            return query, {'success': True, 'data': result}
        except Exception as e:
            logger.error(f"Failed to lookup {query}: {e}")
            return query, {'success': False, 'error': str(e)}

    logger.info(f"Starting bulk lookup of {len(queries)} queries with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_query = {executor.submit(lookup_single, q): q for q in queries}

        # Collect results as they complete
        for future in as_completed(future_to_query):
            query, result = future.result()
            results[query] = result

    logger.info(f"Bulk lookup completed: {sum(1 for r in results.values() if r['success'])}/{len(queries)} successful")

    return results

if __name__ == "__main__":
    import sys

    # Set logging level based on -v flag
    if "-v" in sys.argv or "--verbose" in sys.argv:
        logger.setLevel(logging.DEBUG)
    elif "-q" in sys.argv or "--quiet" in sys.argv:
        logger.setLevel(logging.ERROR)

    print("RDAP Client - Enhanced with metrics, rate-limiting, and parallel lookups\n")

    # Examples of querying different resource types
    examples = [
        "example.com",      # Domain (DNS)
        "8.8.8.8",          # IPv4 address
        "2001:4860:4860::8888",  # IPv6 address
        "AS15169",          # ASN (Google)
        # "GOGL-ARIN",      # Entity/object tag (uncomment to test)
    ]

    # Demo: Sequential lookups
    if "--parallel" not in sys.argv:
        print("Running sequential lookups...")
        for query in examples:
            print(f"\n{'='*60}")
            print(f"Looking up: {query}")
            print('='*60)
            try:
                result = rdap_lookup(query)
                # Print only essential info to avoid overwhelming output
                if 'handle' in result:
                    print(f"Handle: {result['handle']}")
                if 'ldhName' in result:
                    print(f"Name: {result['ldhName']}")
                if 'objectClassName' in result:
                    print(f"Type: {result['objectClassName']}")
                print(f"(Full response available in result)")
            except Exception as e:
                print(f"Error: {e}")

    # Demo: Parallel bulk lookups
    else:
        print("\nRunning parallel bulk lookups...")
        bulk_results = rdap_bulk_lookup(examples, max_workers=3)

        print(f"\n{'='*60}")
        print("Bulk Lookup Results:")
        print('='*60)
        for query, result in bulk_results.items():
            if result['success']:
                data = result['data']
                handle = data.get('handle', 'N/A')
                obj_class = data.get('objectClassName', 'N/A')
                print(f"✓ {query:30} -> {obj_class} ({handle})")
            else:
                print(f"✗ {query:30} -> Error: {result['error']}")

    # Print metrics summary
    print("\n")
    metrics.print_summary()
