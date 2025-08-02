#!/usr/bin/env python3
"""
Repository health monitoring script for Holo-Code-Gen.

This script analyzes the overall health of the repository by checking:
- Code quality metrics
- Test coverage and reliability
- Security posture
- Documentation completeness
- Development velocity
- Technical debt indicators
- Photonic-specific health metrics
"""

import json
import os
import subprocess
import sys
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthMetric:
    """Represents a single health metric with scoring."""
    
    def __init__(self, name: str, description: str, weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = weight
        self.score = None
        self.status = None
        self.details = {}
        self.recommendations = []
    
    def calculate_score(self, value: float, target: float, threshold_good: float = 0.8, threshold_warning: float = 0.6) -> float:
        """Calculate normalized score (0-100) based on value vs target."""
        if target == 0:
            return 100 if value == 0 else 0
        
        ratio = value / target if target > 0 else (target / value if value > 0 else 1)
        ratio = min(ratio, 1.0)  # Cap at 100%
        
        score = ratio * 100
        
        if score >= threshold_good * 100:
            self.status = "excellent"
        elif score >= threshold_warning * 100:
            self.status = "good"
        elif score >= 0.4 * 100:
            self.status = "warning"
        else:
            self.status = "critical"
        
        self.score = score
        return score


class RepositoryHealthAnalyzer:
    """Analyzes repository health across multiple dimensions."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics = {}
        self.overall_score = None
        self.analysis_timestamp = datetime.datetime.now()
    
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
                logger.debug(f"Command failed: {command}")
                return None
        except Exception as e:
            logger.debug(f"Error running command '{command}': {e}")
            return None
    
    def analyze_code_quality(self) -> HealthMetric:
        """Analyze code quality metrics."""
        metric = HealthMetric(
            "Code Quality",
            "Overall code quality including complexity, style, and maintainability",
            weight=2.0
        )
        
        quality_scores = []
        
        # Check code complexity
        complexity_output = self.run_command("python -m radon cc holo_code_gen -a")
        if complexity_output:
            try:
                for line in complexity_output.split('\n'):
                    if 'Average complexity:' in line:
                        complexity = float(line.split(':')[1].strip().split()[0])
                        # Lower complexity is better (target: 6, max acceptable: 10)
                        complexity_score = max(0, 100 - (complexity - 6) * 20)
                        quality_scores.append(complexity_score)
                        metric.details["cyclomatic_complexity"] = complexity
                        break
            except (ValueError, IndexError):
                pass
        
        # Check code style compliance
        style_output = self.run_command("python -m ruff check holo_code_gen --statistics")
        if style_output:
            # Count style violations
            violation_count = len(style_output.split('\n')) if style_output else 0
            # Target: 0 violations, warning at 10, critical at 50
            style_score = max(0, 100 - violation_count * 2)
            quality_scores.append(style_score)
            metric.details["style_violations"] = violation_count
        
        # Check type coverage
        type_output = self.run_command("python -m mypy holo_code_gen --show-error-codes")
        if type_output:
            type_errors = len([line for line in type_output.split('\n') if 'error:' in line])
            type_score = max(0, 100 - type_errors * 5)
            quality_scores.append(type_score)
            metric.details["type_errors"] = type_errors
        
        # Calculate overall code quality score
        if quality_scores:
            avg_score = sum(quality_scores) / len(quality_scores)
            metric.score = avg_score
            if avg_score >= 80:
                metric.status = "excellent"
            elif avg_score >= 60:
                metric.status = "good"
            elif avg_score >= 40:
                metric.status = "warning"
            else:
                metric.status = "critical"
        else:
            metric.score = 50  # Default if no data
            metric.status = "unknown"
        
        # Add recommendations
        if metric.details.get("cyclomatic_complexity", 0) > 8:
            metric.recommendations.append("Consider refactoring complex functions to reduce cyclomatic complexity")
        if metric.details.get("style_violations", 0) > 10:
            metric.recommendations.append("Run 'ruff format .' to fix style violations")
        if metric.details.get("type_errors", 0) > 5:
            metric.recommendations.append("Add type annotations to improve type safety")
        
        return metric
    
    def analyze_test_coverage(self) -> HealthMetric:
        """Analyze test coverage and quality."""
        metric = HealthMetric(
            "Test Coverage",
            "Test coverage percentage and test quality metrics",
            weight=2.0
        )
        
        # Run coverage analysis
        coverage_output = self.run_command(
            "python -m pytest --cov=holo_code_gen --cov-report=term-missing tests/ --tb=no -q"
        )
        
        if coverage_output:
            try:
                for line in coverage_output.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        coverage_str = line.split()[-1].replace('%', '')
                        coverage = float(coverage_str)
                        metric.calculate_score(coverage, 90)  # Target: 90% coverage
                        metric.details["coverage_percentage"] = coverage
                        break
            except (ValueError, IndexError):
                metric.score = 0
                metric.status = "critical"
                metric.details["coverage_percentage"] = 0
        
        # Count test files
        test_files = list(self.project_root.glob("tests/**/*.py"))
        test_count = len([f for f in test_files if f.name.startswith("test_")])
        metric.details["test_files"] = test_count
        
        # Check for test types
        has_unit_tests = any("unit" in str(f) for f in test_files)
        has_integration_tests = any("integration" in str(f) for f in test_files)
        has_performance_tests = any("performance" in str(f) for f in test_files)
        
        metric.details["test_types"] = {
            "unit": has_unit_tests,
            "integration": has_integration_tests,
            "performance": has_performance_tests
        }
        
        # Add recommendations
        if metric.score and metric.score < 80:
            metric.recommendations.append("Increase test coverage to at least 80%")
        if not has_integration_tests:
            metric.recommendations.append("Add integration tests for critical workflows")
        if not has_performance_tests:
            metric.recommendations.append("Add performance tests for photonic simulations")
        
        return metric
    
    def analyze_security_posture(self) -> HealthMetric:
        """Analyze security-related metrics."""
        metric = HealthMetric(
            "Security Posture",
            "Security vulnerability assessment and best practices compliance",
            weight=3.0
        )
        
        security_score = 100  # Start with perfect score and deduct
        
        # Check for known vulnerabilities
        safety_output = self.run_command("python -m safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vuln_count = len(safety_data)
                security_score -= vuln_count * 10  # -10 points per vulnerability
                metric.details["vulnerabilities"] = vuln_count
            except json.JSONDecodeError:
                pass
        
        # Run security linting
        bandit_output = self.run_command("python -m bandit -r holo_code_gen -f json")
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                issues = bandit_data.get("results", [])
                high_issues = sum(1 for issue in issues if issue.get("issue_severity") == "HIGH")
                medium_issues = sum(1 for issue in issues if issue.get("issue_severity") == "MEDIUM")
                
                security_score -= high_issues * 15  # -15 points per high severity issue
                security_score -= medium_issues * 5  # -5 points per medium severity issue
                
                metric.details["security_issues"] = {
                    "high": high_issues,
                    "medium": medium_issues,
                    "total": len(issues)
                }
            except json.JSONDecodeError:
                pass
        
        # Check for security best practices
        has_security_md = (self.project_root / "SECURITY.md").exists()
        has_dependabot = (self.project_root / ".github" / "dependabot.yml").exists()
        
        if not has_security_md:
            security_score -= 5
        if not has_dependabot:
            security_score -= 5
        
        metric.details["security_practices"] = {
            "security_policy": has_security_md,
            "dependabot_config": has_dependabot
        }
        
        metric.score = max(0, security_score)
        if metric.score >= 90:
            metric.status = "excellent"
        elif metric.score >= 70:
            metric.status = "good"
        elif metric.score >= 50:
            metric.status = "warning"
        else:
            metric.status = "critical"
        
        # Add recommendations
        vuln_count = metric.details.get("vulnerabilities", 0)
        if vuln_count > 0:
            metric.recommendations.append(f"Fix {vuln_count} known security vulnerabilities")
        
        security_issues = metric.details.get("security_issues", {})
        if security_issues.get("high", 0) > 0:
            metric.recommendations.append("Address high-severity security issues found by Bandit")
        
        if not has_security_md:
            metric.recommendations.append("Create SECURITY.md with vulnerability reporting process")
        
        return metric
    
    def analyze_documentation_health(self) -> HealthMetric:
        """Analyze documentation completeness and quality."""
        metric = HealthMetric(
            "Documentation Health",
            "Documentation coverage, quality, and maintenance",
            weight=1.5
        )
        
        doc_score = 0
        
        # Check for essential documentation files
        essential_docs = {
            "README.md": 25,
            "CONTRIBUTING.md": 15,
            "CHANGELOG.md": 10,
            "LICENSE": 10,
            "docs/": 20,
            "examples/": 20
        }
        
        for doc_path, points in essential_docs.items():
            if (self.project_root / doc_path).exists():
                doc_score += points
        
        metric.details["essential_docs_score"] = doc_score
        
        # Check README quality
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_quality = 0
            
            # Check for key sections
            if "installation" in readme_content.lower():
                readme_quality += 20
            if "usage" in readme_content.lower() or "quick start" in readme_content.lower():
                readme_quality += 20
            if "example" in readme_content.lower():
                readme_quality += 20
            if "api" in readme_content.lower() or "documentation" in readme_content.lower():
                readme_quality += 20
            if "contributing" in readme_content.lower():
                readme_quality += 20
            
            metric.details["readme_quality"] = readme_quality
            doc_score = (doc_score + readme_quality) / 2  # Average with essential docs
        
        # Check for inline documentation
        py_files = list(self.project_root.glob("holo_code_gen/**/*.py"))
        documented_functions = 0
        total_functions = 0
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                # Count functions and classes
                functions = re.findall(r'^\s*def\s+\w+\s*\(', content, re.MULTILINE)
                classes = re.findall(r'^\s*class\s+\w+\s*[\(:]', content, re.MULTILINE)
                total_functions += len(functions) + len(classes)
                
                # Count documented functions (with docstrings)
                doc_functions = re.findall(r'def\s+\w+\s*\([^)]*\):\s*"""', content)
                doc_classes = re.findall(r'class\s+\w+\s*[\(:].*?"""', content, re.DOTALL)
                documented_functions += len(doc_functions) + len(doc_classes)
            except Exception:
                continue
        
        if total_functions > 0:
            doc_coverage = (documented_functions / total_functions) * 100
            metric.details["inline_doc_coverage"] = doc_coverage
            doc_score = (doc_score + doc_coverage) / 2  # Average with other scores
        
        metric.score = min(100, doc_score)
        if metric.score >= 80:
            metric.status = "excellent"
        elif metric.score >= 60:
            metric.status = "good"
        elif metric.score >= 40:
            metric.status = "warning"
        else:
            metric.status = "critical"
        
        # Add recommendations
        missing_docs = [doc for doc, _ in essential_docs.items() 
                       if not (self.project_root / doc).exists()]
        if missing_docs:
            metric.recommendations.append(f"Add missing documentation: {', '.join(missing_docs)}")
        
        if metric.details.get("inline_doc_coverage", 0) < 70:
            metric.recommendations.append("Add docstrings to functions and classes")
        
        return metric
    
    def analyze_development_velocity(self) -> HealthMetric:
        """Analyze development activity and velocity."""
        metric = HealthMetric(
            "Development Velocity",
            "Development activity, commit frequency, and project momentum",
            weight=1.0
        )
        
        velocity_scores = []
        
        # Check recent commit activity
        recent_commits = self.run_command("git log --since='30 days ago' --oneline | wc -l")
        if recent_commits:
            try:
                commit_count = int(recent_commits)
                # Target: 20+ commits per month, good: 10+, warning: 5+
                if commit_count >= 20:
                    velocity_scores.append(100)
                elif commit_count >= 10:
                    velocity_scores.append(75)
                elif commit_count >= 5:
                    velocity_scores.append(50)
                else:
                    velocity_scores.append(25)
                
                metric.details["commits_last_30_days"] = commit_count
            except ValueError:
                pass
        
        # Check contributor activity
        contributors = self.run_command("git log --since='90 days ago' --format='%ae' | sort | uniq | wc -l")
        if contributors:
            try:
                contributor_count = int(contributors)
                # More contributors = better velocity
                velocity_scores.append(min(100, contributor_count * 25))
                metric.details["active_contributors"] = contributor_count
            except ValueError:
                pass
        
        # Check issue/PR activity (if GitHub CLI is available)
        issue_activity = self.run_command("gh issue list --state all --limit 100 --json createdAt")
        if issue_activity:
            try:
                issues = json.loads(issue_activity)
                recent_issues = sum(1 for issue in issues 
                                  if (datetime.datetime.now() - 
                                      datetime.datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))).days <= 30)
                velocity_scores.append(min(100, recent_issues * 10))
                metric.details["recent_issues"] = recent_issues
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Calculate overall velocity score
        if velocity_scores:
            metric.score = sum(velocity_scores) / len(velocity_scores)
        else:
            metric.score = 50  # Default if no data
        
        if metric.score >= 80:
            metric.status = "excellent"
        elif metric.score >= 60:
            metric.status = "good"
        elif metric.score >= 40:
            metric.status = "warning"
        else:
            metric.status = "critical"
        
        # Add recommendations
        if metric.details.get("commits_last_30_days", 0) < 10:
            metric.recommendations.append("Increase development activity with more frequent commits")
        if metric.details.get("active_contributors", 0) < 2:
            metric.recommendations.append("Encourage more contributors to participate in development")
        
        return metric
    
    def analyze_photonic_specific_health(self) -> HealthMetric:
        """Analyze photonic-specific health metrics."""
        metric = HealthMetric(
            "Photonic Platform Health",
            "Health of photonic simulation and design capabilities",
            weight=2.5
        )
        
        photonic_scores = []
        
        # Check photonic library integrity
        library_check = self.run_command(
            "python -c 'from holo_code_gen.templates import IMECLibrary; "
            "IMECLibrary.verify_integrity(); print(\"OK\")'"
        )
        if library_check and "OK" in library_check:
            photonic_scores.append(100)
            metric.details["library_integrity"] = True
        else:
            photonic_scores.append(0)
            metric.details["library_integrity"] = False
        
        # Check for essential photonic dependencies
        photonic_deps = ["gdstk", "numpy", "scipy"]
        available_deps = 0
        
        for dep in photonic_deps:
            dep_check = self.run_command(f"python -c 'import {dep}; print(\"OK\")'")
            if dep_check and "OK" in dep_check:
                available_deps += 1
        
        dependency_score = (available_deps / len(photonic_deps)) * 100
        photonic_scores.append(dependency_score)
        metric.details["photonic_dependencies"] = {
            "available": available_deps,
            "total": len(photonic_deps),
            "percentage": dependency_score
        }
        
        # Check for photonic-specific tests
        photonic_test_files = list(self.project_root.glob("tests/**/test_*photonic*.py"))
        photonic_test_files.extend(list(self.project_root.glob("tests/**/test_*template*.py")))
        photonic_test_files.extend(list(self.project_root.glob("tests/**/test_*simulation*.py")))
        
        if photonic_test_files:
            photonic_scores.append(100)
            metric.details["photonic_tests"] = len(photonic_test_files)
        else:
            photonic_scores.append(50)
            metric.details["photonic_tests"] = 0
        
        # Check for example photonic designs
        example_files = list(self.project_root.glob("examples/**/*.py"))
        if example_files:
            photonic_scores.append(100)
            metric.details["example_designs"] = len(example_files)
        else:
            photonic_scores.append(0)
            metric.details["example_designs"] = 0
        
        # Calculate overall photonic health
        metric.score = sum(photonic_scores) / len(photonic_scores)
        
        if metric.score >= 85:
            metric.status = "excellent"
        elif metric.score >= 70:
            metric.status = "good"
        elif metric.score >= 50:
            metric.status = "warning"
        else:
            metric.status = "critical"
        
        # Add recommendations
        if not metric.details.get("library_integrity", False):
            metric.recommendations.append("Fix photonic library integrity issues")
        if metric.details.get("photonic_dependencies", {}).get("percentage", 0) < 100:
            metric.recommendations.append("Install missing photonic simulation dependencies")
        if metric.details.get("photonic_tests", 0) == 0:
            metric.recommendations.append("Add photonic-specific unit and integration tests")
        if metric.details.get("example_designs", 0) == 0:
            metric.recommendations.append("Create example photonic circuit designs")
        
        return metric
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive repository health analysis."""
        logger.info("Starting comprehensive repository health analysis...")
        
        # Analyze each health dimension
        self.metrics = {
            "code_quality": self.analyze_code_quality(),
            "test_coverage": self.analyze_test_coverage(),
            "security_posture": self.analyze_security_posture(),
            "documentation_health": self.analyze_documentation_health(),
            "development_velocity": self.analyze_development_velocity(),
            "photonic_health": self.analyze_photonic_specific_health()
        }
        
        # Calculate overall health score
        total_weighted_score = 0
        total_weight = 0
        
        for metric in self.metrics.values():
            if metric.score is not None:
                total_weighted_score += metric.score * metric.weight
                total_weight += metric.weight
        
        if total_weight > 0:
            self.overall_score = total_weighted_score / total_weight
        else:
            self.overall_score = 0
        
        # Determine overall status
        if self.overall_score >= 85:
            overall_status = "excellent"
        elif self.overall_score >= 70:
            overall_status = "good"
        elif self.overall_score >= 50:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        # Compile results
        results = {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "repository_path": str(self.project_root),
            "overall_score": round(self.overall_score, 2),
            "overall_status": overall_status,
            "metrics": {},
            "summary": {
                "total_metrics": len(self.metrics),
                "excellent": sum(1 for m in self.metrics.values() if m.status == "excellent"),
                "good": sum(1 for m in self.metrics.values() if m.status == "good"),
                "warning": sum(1 for m in self.metrics.values() if m.status == "warning"),
                "critical": sum(1 for m in self.metrics.values() if m.status == "critical")
            },
            "recommendations": []
        }
        
        # Add metric details
        for name, metric in self.metrics.items():
            results["metrics"][name] = {
                "name": metric.name,
                "description": metric.description,
                "score": round(metric.score, 2) if metric.score else None,
                "status": metric.status,
                "weight": metric.weight,
                "details": metric.details,
                "recommendations": metric.recommendations
            }
            
            # Collect high-priority recommendations
            if metric.status in ["critical", "warning"]:
                results["recommendations"].extend(metric.recommendations)
        
        logger.info(f"Analysis completed. Overall score: {self.overall_score:.1f}/100 ({overall_status})")
        
        return results
    
    def generate_health_report(self, results: Dict[str, Any], format_type: str = "markdown") -> str:
        """Generate a health report in the specified format."""
        if format_type == "markdown":
            return self._generate_markdown_report(results)
        elif format_type == "json":
            return json.dumps(results, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown health report."""
        report = f"# Repository Health Report\n\n"
        report += f"**Generated:** {results['analysis_timestamp']}\n"
        report += f"**Repository:** {results['repository_path']}\n"
        report += f"**Overall Score:** {results['overall_score']}/100\n"
        report += f"**Status:** {results['overall_status'].upper()}\n\n"
        
        # Status indicators
        status_icons = {
            "excellent": "🟢",
            "good": "🟡",
            "warning": "🟠",
            "critical": "🔴"
        }
        
        # Summary
        summary = results['summary']
        report += "## Summary\n\n"
        report += f"- 🟢 Excellent: {summary['excellent']}\n"
        report += f"- 🟡 Good: {summary['good']}\n"
        report += f"- 🟠 Warning: {summary['warning']}\n"
        report += f"- 🔴 Critical: {summary['critical']}\n\n"
        
        # Detailed metrics
        report += "## Detailed Analysis\n\n"
        
        for metric_key, metric_data in results['metrics'].items():
            icon = status_icons.get(metric_data['status'], "⚪")
            report += f"### {icon} {metric_data['name']}\n\n"
            report += f"**Score:** {metric_data['score']}/100\n"
            report += f"**Status:** {metric_data['status'].title()}\n"
            report += f"**Weight:** {metric_data['weight']}\n"
            report += f"**Description:** {metric_data['description']}\n\n"
            
            if metric_data['details']:
                report += "**Details:**\n"
                for key, value in metric_data['details'].items():
                    report += f"- {key.replace('_', ' ').title()}: {value}\n"
                report += "\n"
            
            if metric_data['recommendations']:
                report += "**Recommendations:**\n"
                for rec in metric_data['recommendations']:
                    report += f"- {rec}\n"
                report += "\n"
        
        # Top recommendations
        if results['recommendations']:
            report += "## Priority Recommendations\n\n"
            for i, rec in enumerate(results['recommendations'][:10], 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        # Health trends (placeholder for future implementation)
        report += "## Health Trends\n\n"
        report += "*Health trend analysis will be available after multiple health checks.*\n\n"
        
        return report


def main():
    """Main function to run repository health analysis."""
    parser = argparse.ArgumentParser(description="Analyze repository health for Holo-Code-Gen")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root directory"
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for health report"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for health report"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON file"
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
        analyzer = RepositoryHealthAnalyzer(args.project_root)
        results = analyzer.run_comprehensive_analysis()
        
        # Generate report
        report = analyzer.generate_health_report(results, args.output_format)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
            logger.info(f"Health report saved to {args.output_file}")
        else:
            print(report)
        
        # Save detailed results if requested
        if args.save_results:
            results_file = args.project_root / "reports" / f"health_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detailed results saved to {results_file}")
        
        # Exit with appropriate code based on health status
        if results['overall_status'] in ['critical']:
            return 2
        elif results['overall_status'] in ['warning']:
            return 1
        else:
            return 0
    
    except Exception as e:
        logger.error(f"Health analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())