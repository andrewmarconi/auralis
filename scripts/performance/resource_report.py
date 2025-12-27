#!/usr/bin/env python3
"""
Resource Usage Report Generator

Generates comprehensive performance report by querying Prometheus metrics
and analyzing resource utilization trends.

Task: T098 [US3]

Usage:
    python scripts/performance/resource_report.py --prometheus-url http://localhost:9090
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from tabulate import tabulate


class PrometheusClient:
    """Query Prometheus for metrics data."""

    def __init__(self, base_url: str):
        """
        Initialize Prometheus client.

        Args:
            base_url: Prometheus server URL (e.g., http://localhost:9090)
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"

    def query(self, query: str) -> Optional[Dict]:
        """
        Execute instant query against Prometheus.

        Args:
            query: PromQL query string

        Returns:
            Query result dict or None on error
        """
        try:
            response = requests.get(
                f"{self.api_url}/query",
                params={"query": query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus: {e}", file=sys.stderr)
            return None

    def query_range(self, query: str, start: datetime, end: datetime, step: str = "1m") -> Optional[Dict]:
        """
        Execute range query against Prometheus.

        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query resolution (e.g., "1m", "5m")

        Returns:
            Query result dict or None on error
        """
        try:
            response = requests.get(
                f"{self.api_url}/query_range",
                params={
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "step": step
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus range: {e}", file=sys.stderr)
            return None


class ResourceReport:
    """Generate resource utilization report from Prometheus metrics."""

    def __init__(self, prometheus_url: str):
        """
        Initialize report generator.

        Args:
            prometheus_url: Prometheus server URL
        """
        self.client = PrometheusClient(prometheus_url)
        self.report_data = {}

    def collect_synthesis_metrics(self) -> Dict:
        """Collect synthesis performance metrics."""
        metrics = {}

        # P50, P95, P99 latency
        for quantile in [0.50, 0.95, 0.99]:
            query = f'histogram_quantile({quantile}, rate(auralis_synthesis_latency_seconds_bucket[5m])) * 1000'
            result = self.client.query(query)
            if result and result['status'] == 'success' and result['data']['result']:
                value = float(result['data']['result'][0]['value'][1])
                metrics[f'p{int(quantile*100)}_latency_ms'] = round(value, 2)

        # Phrase generation rate
        query = 'auralis_phrase_generation_rate_hz'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['phrase_rate_hz'] = round(value, 3)

        # Total synthesis operations
        query = 'auralis_synthesis_total'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['total_operations'] = int(value)

        return metrics

    def collect_memory_metrics(self) -> Dict:
        """Collect memory usage metrics."""
        metrics = {}

        # Current RSS memory
        query = 'auralis_memory_usage_mb{type="rss"}'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['rss_mb'] = round(value, 2)

        # Python memory
        query = 'auralis_memory_usage_mb{type="python"}'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['python_mb'] = round(value, 2)

        # Memory growth rate (MB/hour)
        query = 'rate(auralis_memory_usage_mb{type="rss"}[1h]) * 3600'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['growth_rate_mb_per_hour'] = round(value, 2)

        return metrics

    def collect_gpu_metrics(self) -> Dict:
        """Collect GPU memory metrics."""
        metrics = {}

        # GPU allocated memory
        query = 'auralis_gpu_memory_allocated_mb'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            for series in result['data']['result']:
                device = series['metric']['device']
                value = float(series['value'][1])
                metrics[f'{device}_allocated_mb'] = round(value, 2)

        # GPU reserved memory (CUDA only)
        query = 'auralis_gpu_memory_reserved_mb{device="cuda"}'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['cuda_reserved_mb'] = round(value, 2)

        return metrics

    def collect_cpu_metrics(self) -> Dict:
        """Collect CPU utilization metrics."""
        metrics = {}

        # Current CPU usage
        query = 'auralis_cpu_usage_percent'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['current_percent'] = round(value, 2)

        # Average CPU over 5 minutes
        query = 'avg_over_time(auralis_cpu_usage_percent[5m])'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            value = float(result['data']['result'][0]['value'][1])
            metrics['avg_5m_percent'] = round(value, 2)

        return metrics

    def collect_gc_metrics(self) -> Dict:
        """Collect garbage collection statistics."""
        metrics = {}

        # Collection counts per generation
        for gen in [0, 1, 2]:
            query = f'auralis_gc_collections_total{{generation="{gen}"}}'
            result = self.client.query(query)
            if result and result['status'] == 'success' and result['data']['result']:
                value = float(result['data']['result'][0]['value'][1])
                metrics[f'gen{gen}_collections'] = int(value)

        # Collection rate (collections/sec)
        for gen in [0, 1, 2]:
            query = f'rate(auralis_gc_collections_total{{generation="{gen}"}}[5m])'
            result = self.client.query(query)
            if result and result['status'] == 'success' and result['data']['result']:
                value = float(result['data']['result'][0]['value'][1])
                metrics[f'gen{gen}_rate_per_sec'] = round(value, 4)

        return metrics

    def check_success_criteria(self) -> Dict:
        """Validate success criteria from spec.md."""
        criteria = {}

        # SC-001: 99% of chunks delivered within 50ms
        query = 'histogram_quantile(0.99, rate(auralis_synthesis_latency_seconds_bucket[5m])) * 1000'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            p99_ms = float(result['data']['result'][0]['value'][1])
            criteria['SC-001'] = {
                'description': '99% chunks <50ms',
                'target': '<50ms',
                'actual': f'{p99_ms:.2f}ms',
                'passed': p99_ms < 50.0
            }

        # SC-003: CPU utilization reduced by 30% (requires baseline comparison)
        query = 'avg_over_time(auralis_cpu_usage_percent[5m])'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            cpu_avg = float(result['data']['result'][0]['value'][1])
            criteria['SC-003'] = {
                'description': 'CPU <50% avg',
                'target': '<50%',
                'actual': f'{cpu_avg:.2f}%',
                'passed': cpu_avg < 50.0
            }

        # SC-004: Memory growth <10MB/hour
        query = 'rate(auralis_memory_usage_mb{type="rss"}[1h]) * 3600'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            growth = float(result['data']['result'][0]['value'][1])
            criteria['SC-004'] = {
                'description': 'Memory growth <10MB/hr',
                'target': '<10MB/hr',
                'actual': f'{growth:.2f}MB/hr',
                'passed': growth < 10.0
            }

        # SC-008: P95 jitter <20ms
        query = 'histogram_quantile(0.95, rate(auralis_synthesis_latency_seconds_bucket[5m])) * 1000'
        result = self.client.query(query)
        if result and result['status'] == 'success' and result['data']['result']:
            p95_ms = float(result['data']['result'][0]['value'][1])
            criteria['SC-008'] = {
                'description': 'P95 jitter <20ms',
                'target': '<20ms',
                'actual': f'{p95_ms:.2f}ms',
                'passed': p95_ms < 20.0
            }

        return criteria

    def generate_report(self) -> str:
        """Generate formatted performance report."""
        output = []
        output.append("=" * 80)
        output.append("AURALIS RESOURCE UTILIZATION REPORT")
        output.append("=" * 80)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")

        # Synthesis Performance
        output.append("SYNTHESIS PERFORMANCE")
        output.append("-" * 80)
        synth_metrics = self.collect_synthesis_metrics()
        if synth_metrics:
            table_data = [[k, v] for k, v in synth_metrics.items()]
            output.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
        else:
            output.append("  No synthesis metrics available")
        output.append("")

        # Memory Usage
        output.append("MEMORY USAGE")
        output.append("-" * 80)
        mem_metrics = self.collect_memory_metrics()
        if mem_metrics:
            table_data = [[k, v] for k, v in mem_metrics.items()]
            output.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
        else:
            output.append("  No memory metrics available")
        output.append("")

        # GPU Usage
        output.append("GPU USAGE")
        output.append("-" * 80)
        gpu_metrics = self.collect_gpu_metrics()
        if gpu_metrics:
            table_data = [[k, v] for k, v in gpu_metrics.items()]
            output.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
        else:
            output.append("  No GPU metrics available")
        output.append("")

        # CPU Usage
        output.append("CPU USAGE")
        output.append("-" * 80)
        cpu_metrics = self.collect_cpu_metrics()
        if cpu_metrics:
            table_data = [[k, v] for k, v in cpu_metrics.items()]
            output.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
        else:
            output.append("  No CPU metrics available")
        output.append("")

        # Garbage Collection
        output.append("GARBAGE COLLECTION")
        output.append("-" * 80)
        gc_metrics = self.collect_gc_metrics()
        if gc_metrics:
            table_data = [[k, v] for k, v in gc_metrics.items()]
            output.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
        else:
            output.append("  No GC metrics available")
        output.append("")

        # Success Criteria
        output.append("SUCCESS CRITERIA VALIDATION")
        output.append("-" * 80)
        criteria = self.check_success_criteria()
        if criteria:
            table_data = [
                [
                    criterion_id,
                    data['description'],
                    data['target'],
                    data['actual'],
                    "✓ PASS" if data['passed'] else "✗ FAIL"
                ]
                for criterion_id, data in criteria.items()
            ]
            output.append(tabulate(
                table_data,
                headers=["ID", "Description", "Target", "Actual", "Status"],
                tablefmt="simple"
            ))

            # Summary
            passed = sum(1 for data in criteria.values() if data['passed'])
            total = len(criteria)
            output.append("")
            output.append(f"Summary: {passed}/{total} criteria passed ({passed/total*100:.1f}%)")
        else:
            output.append("  No success criteria data available")
        output.append("")

        output.append("=" * 80)

        return "\n".join(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate Auralis resource utilization report')
    parser.add_argument(
        '--prometheus-url',
        default='http://localhost:9090',
        help='Prometheus server URL (default: http://localhost:9090)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: print to stdout)'
    )

    args = parser.parse_args()

    # Generate report
    reporter = ResourceReport(args.prometheus_url)
    report = reporter.generate_report()

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
