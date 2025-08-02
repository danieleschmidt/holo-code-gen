#!/usr/bin/env python3
"""
Automated metrics collection script for Holo-Code-Gen project.

This script collects various project metrics including:
- Code quality metrics
- Test coverage
- Security vulnerabilities
- Performance metrics
- Development velocity
- Photonic-specific metrics
"""

import json
import os
import subprocess
import sys
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and updates project metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        self.current_metrics = self.load_current_metrics()
    
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from JSON file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metrics file not found: {self.metrics_file}")
            return {}
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save updated metrics to JSON file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {self.metrics_file}")
    
    def run_command(self, command: str, capture_output: bool = True) -> Optional[str]:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip() if capture_output else None
            else:
                logger.warning(f"Command failed: {command}")
                logger.warning(f"Error: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error running command '{command}': {e}")
            return None
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        logger.info("Collecting code quality metrics...")
        
        metrics = {}
        
        # Lines of code
        loc_output = self.run_command("find holo_code_gen -name '*.py' -exec wc -l {} + | tail -1")
        if loc_output:
            try:
                metrics["lines_of_code"] = {
                    "current": int(loc_output.split()[0]),
                    "last_updated": datetime.datetime.now().isoformat()
                }
            except (ValueError, IndexError):
                logger.warning("Could not parse lines of code")
        
        # Test coverage (if pytest-cov is available)
        coverage_output = self.run_command("python -m pytest --cov=holo_code_gen --cov-report=term-missing tests/ --tb=no -q")
        if coverage_output and "%" in coverage_output:
            try:
                # Extract coverage percentage
                for line in coverage_output.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        coverage_str = line.split()[-1].replace('%', '')
                        metrics["test_coverage"] = {
                            "current": float(coverage_str),
                            "last_updated": datetime.datetime.now().isoformat()
                        }
                        break
            except (ValueError, IndexError):
                logger.warning("Could not parse test coverage")
        
        # Code complexity (if radon is available)
        complexity_output = self.run_command("python -m radon cc holo_code_gen -a")
        if complexity_output:
            try:
                # Extract average complexity
                for line in complexity_output.split('\n'):
                    if 'Average complexity:' in line:
                        complexity = float(line.split(':')[1].strip().split()[0])
                        if "code_complexity" not in metrics:
                            metrics["code_complexity"] = {}
                        metrics["code_complexity"]["cyclomatic_complexity"] = {
                            "current": complexity,
                            "last_updated": datetime.datetime.now().isoformat()
                        }
                        break
            except (ValueError, IndexError):
                logger.warning("Could not parse code complexity")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        
        metrics = {}
        
        # Safety check for known vulnerabilities
        safety_output = self.run_command("python -m safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vulnerability_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                
                for vuln in safety_data:
                    severity = vuln.get("advisory", "").lower()
                    if "critical" in severity:
                        vulnerability_counts["critical"] += 1
                    elif "high" in severity:
                        vulnerability_counts["high"] += 1
                    elif "medium" in severity:
                        vulnerability_counts["medium"] += 1
                    else:
                        vulnerability_counts["low"] += 1
                
                metrics["vulnerabilities"] = vulnerability_counts
                
                # Calculate security score
                total_vulns = sum(vulnerability_counts.values())
                if total_vulns == 0:
                    security_score = 100
                else:
                    # Weighted scoring: critical=10, high=5, medium=2, low=1
                    weighted_score = (
                        vulnerability_counts["critical"] * 10 +
                        vulnerability_counts["high"] * 5 +
                        vulnerability_counts["medium"] * 2 +
                        vulnerability_counts["low"] * 1
                    )
                    security_score = max(0, 100 - weighted_score)
                
                metrics["security_score"] = {
                    "current": security_score,
                    "last_updated": datetime.datetime.now().isoformat()
                }
                
            except json.JSONDecodeError:
                logger.warning("Could not parse safety check output")
        
        # Check for outdated dependencies
        outdated_output = self.run_command("python -m pip list --outdated --format=json")
        if outdated_output:
            try:
                outdated_data = json.loads(outdated_output)
                if "dependency_health" not in metrics:
                    metrics["dependency_health"] = {}
                metrics["dependency_health"]["outdated_dependencies"] = {
                    "current": len(outdated_data),
                    "last_updated": datetime.datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                logger.warning("Could not parse outdated packages output")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        logger.info("Collecting performance metrics...")
        
        metrics = {}
        
        # Run basic performance test if available
        perf_test_script = self.project_root / "scripts" / "performance_test.py"
        if perf_test_script.exists():
            perf_output = self.run_command(f"python {perf_test_script}")
            if perf_output:
                try:
                    # Parse performance test output
                    perf_data = json.loads(perf_output)
                    for key, value in perf_data.items():
                        metrics[key] = {
                            "current": value,
                            "last_updated": datetime.datetime.now().isoformat()
                        }
                except json.JSONDecodeError:
                    logger.warning("Could not parse performance test output")
        
        return metrics
    
    def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development velocity metrics."""
        logger.info("Collecting development metrics...")
        
        metrics = {}
        
        # Git metrics
        try:
            # Commits in the last week
            commits_week = self.run_command(
                "git log --since='1 week ago' --oneline | wc -l"
            )
            if commits_week:
                if "velocity" not in metrics:
                    metrics["velocity"] = {}
                metrics["velocity"]["commits_per_week"] = {
                    "current": int(commits_week),
                    "last_updated": datetime.datetime.now().isoformat()
                }
            
            # Recent commit activity
            commit_output = self.run_command(
                "git log --since='1 month ago' --pretty=format:'%ad' --date=short | sort | uniq -c"
            )
            if commit_output:
                daily_commits = len(commit_output.strip().split('\n'))
                if "velocity" not in metrics:
                    metrics["velocity"] = {}
                metrics["velocity"]["active_days_per_month"] = {
                    "current": daily_commits,
                    "last_updated": datetime.datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.warning(f"Error collecting git metrics: {e}")
        
        return metrics
    
    def collect_photonic_specific_metrics(self) -> Dict[str, Any]:
        """Collect photonic-specific metrics."""
        logger.info("Collecting photonic-specific metrics...")
        
        metrics = {}
        
        # Count photonic components
        template_dir = self.project_root / "holo_code_gen" / "templates"
        if template_dir.exists():
            component_files = list(template_dir.glob("**/*.py"))
            if "template_library" not in metrics:
                metrics["template_library"] = {}
            metrics["template_library"]["components_count"] = {
                "current": len(component_files),
                "last_updated": datetime.datetime.now().isoformat()
            }
        
        # Test photonic library integrity
        integrity_output = self.run_command(
            "python -c 'from holo_code_gen.templates import IMECLibrary; IMECLibrary.verify_integrity(); print(\"OK\")'"
        )
        if integrity_output and "OK" in integrity_output:
            if "template_library" not in metrics:
                metrics["template_library"] = {}
            metrics["template_library"]["verification_coverage"] = {
                "current": 100,
                "last_updated": datetime.datetime.now().isoformat()
            }
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        logger.info("Starting comprehensive metrics collection...")
        
        updated_metrics = self.current_metrics.copy()
        
        # Ensure metrics structure exists
        if "metrics" not in updated_metrics:
            updated_metrics["metrics"] = {}
        
        # Collect different metric categories
        metric_categories = [
            ("code_quality", self.collect_code_quality_metrics),
            ("security", self.collect_security_metrics),
            ("performance", self.collect_performance_metrics),
            ("development", self.collect_development_metrics),
            ("photonic_specific", self.collect_photonic_specific_metrics),
        ]
        
        for category, collector_func in metric_categories:
            try:
                new_metrics = collector_func()
                if new_metrics:
                    if category not in updated_metrics["metrics"]:
                        updated_metrics["metrics"][category] = {}
                    
                    # Merge new metrics with existing ones
                    for key, value in new_metrics.items():
                        updated_metrics["metrics"][category][key] = value
                
            except Exception as e:
                logger.error(f"Error collecting {category} metrics: {e}")
        
        # Update tracking information
        if "tracking" not in updated_metrics:
            updated_metrics["tracking"] = {}
        
        updated_metrics["tracking"]["last_collection"] = datetime.datetime.now().isoformat()
        updated_metrics["tracking"]["next_collection"] = (
            datetime.datetime.now() + datetime.timedelta(days=1)
        ).isoformat()
        
        return updated_metrics
    
    def generate_report(self, output_format: str = "json") -> str:
        """Generate a metrics report in the specified format."""
        logger.info(f"Generating metrics report in {output_format} format...")
        
        if output_format == "json":
            return json.dumps(self.current_metrics, indent=2)
        
        elif output_format == "markdown":
            report = "# Project Metrics Report\n\n"
            report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
            
            if "metrics" in self.current_metrics:
                for category, metrics in self.current_metrics["metrics"].items():
                    report += f"## {category.replace('_', ' ').title()}\n\n"
                    
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and "current" in metric_data:
                            current = metric_data["current"]
                            target = metric_data.get("target", "N/A")
                            unit = metric_data.get("unit", "")
                            
                            report += f"- **{metric_name.replace('_', ' ').title()}**: {current} {unit}"
                            if target != "N/A":
                                report += f" (Target: {target} {unit})"
                            report += "\n"
                    
                    report += "\n"
            
            return report
        
        elif output_format == "prometheus":
            # Generate Prometheus metrics format
            prometheus_metrics = []
            
            if "metrics" in self.current_metrics:
                for category, metrics in self.current_metrics["metrics"].items():
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and "current" in metric_data:
                            metric_full_name = f"holo_code_gen_{category}_{metric_name}"
                            value = metric_data["current"]
                            
                            if isinstance(value, (int, float)):
                                prometheus_metrics.append(
                                    f"# HELP {metric_full_name} {metric_name.replace('_', ' ').title()}\n"
                                    f"# TYPE {metric_full_name} gauge\n"
                                    f"{metric_full_name} {value}\n"
                                )
            
            return "\n".join(prometheus_metrics)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Main function to run metrics collection."""
    parser = argparse.ArgumentParser(description="Collect project metrics for Holo-Code-Gen")
    parser.add_argument(
        "--project-root", 
        type=Path, 
        default=Path.cwd(),
        help="Path to project root directory"
    )
    parser.add_argument(
        "--output-format", 
        choices=["json", "markdown", "prometheus"],
        default="json",
        help="Output format for metrics report"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for metrics report"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect metrics, don't generate report"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing metrics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(args.project_root)
        
        if not args.report_only:
            # Collect metrics
            updated_metrics = collector.collect_all_metrics()
            collector.save_metrics(updated_metrics)
            logger.info("Metrics collection completed successfully")
        
        if not args.collect_only:
            # Generate report
            report = collector.generate_report(args.output_format)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {args.output_file}")
            else:
                print(report)
        
        return 0
    
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())