#!/usr/bin/env python3
"""
Automated maintenance script for Holo-Code-Gen project.

This script performs various automated maintenance tasks including:
- Repository cleanup
- Dependency updates
- Security scanning
- Performance monitoring
- Health checks
- Report generation
"""

import json
import os
import shutil
import subprocess
import sys
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MaintenanceTask:
    """Base class for maintenance tasks."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success = False
        self.error_message = None
        self.execution_time = None
    
    def execute(self, project_root: Path) -> bool:
        """Execute the maintenance task."""
        raise NotImplementedError
    
    def log_result(self):
        """Log the task execution result."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        logger.info(f"{status}: {self.name}")
        if self.error_message:
            logger.error(f"Error: {self.error_message}")
        if self.execution_time:
            logger.info(f"Execution time: {self.execution_time:.2f}s")


class CleanupTask(MaintenanceTask):
    """Clean up temporary files and build artifacts."""
    
    def __init__(self):
        super().__init__(
            "Repository Cleanup",
            "Remove temporary files, build artifacts, and cached data"
        )
    
    def execute(self, project_root: Path) -> bool:
        """Execute cleanup tasks."""
        start_time = datetime.datetime.now()
        
        try:
            cleanup_patterns = [
                "**/__pycache__",
                "**/*.pyc",
                "**/*.pyo",
                "**/*.pyd",
                ".pytest_cache",
                ".coverage",
                ".mypy_cache",
                ".ruff_cache",
                "htmlcov",
                "dist",
                "build",
                "*.egg-info",
                ".tox",
                ".nox",
                "*.log",
                "temp",
                "tmp"
            ]
            
            total_size_freed = 0
            files_removed = 0
            
            for pattern in cleanup_patterns:
                for path in project_root.glob(pattern):
                    if path.exists():
                        if path.is_file():
                            total_size_freed += path.stat().st_size
                            path.unlink()
                            files_removed += 1
                        elif path.is_dir():
                            dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                            total_size_freed += dir_size
                            shutil.rmtree(path)
                            files_removed += 1
            
            # Clean Docker artifacts if Docker is available
            try:
                subprocess.run(
                    ["docker", "system", "prune", "-f"], 
                    capture_output=True, 
                    check=True
                )
                logger.info("Docker system cleanup completed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.debug("Docker cleanup skipped (Docker not available)")
            
            self.success = True
            size_mb = total_size_freed / (1024 * 1024)
            logger.info(f"Cleanup completed: {files_removed} items removed, {size_mb:.2f} MB freed")
            
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success


class DependencyUpdateTask(MaintenanceTask):
    """Check for and report dependency updates."""
    
    def __init__(self):
        super().__init__(
            "Dependency Analysis",
            "Check for outdated dependencies and security vulnerabilities"
        )
    
    def execute(self, project_root: Path) -> bool:
        """Execute dependency analysis."""
        start_time = datetime.datetime.now()
        
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["python", "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                logger.info(f"Found {len(outdated_packages)} outdated packages")
                
                # Check for security vulnerabilities
                security_result = subprocess.run(
                    ["python", "-m", "safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                
                if security_result.returncode == 0:
                    vulnerabilities = json.loads(security_result.stdout)
                    if vulnerabilities:
                        logger.warning(f"Found {len(vulnerabilities)} security vulnerabilities")
                        for vuln in vulnerabilities[:5]:  # Show first 5
                            logger.warning(f"- {vuln.get('package_name', 'Unknown')}: {vuln.get('advisory', 'No details')}")
                    else:
                        logger.info("No security vulnerabilities found")
                
                self.success = True
                
            else:
                self.error_message = result.stderr
                self.success = False
        
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success


class SecurityScanTask(MaintenanceTask):
    """Perform security scanning."""
    
    def __init__(self):
        super().__init__(
            "Security Scanning",
            "Run security analysis tools on the codebase"
        )
    
    def execute(self, project_root: Path) -> bool:
        """Execute security scanning."""
        start_time = datetime.datetime.now()
        
        try:
            # Run Bandit security scanner
            bandit_result = subprocess.run(
                ["python", "-m", "bandit", "-r", "holo_code_gen", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if bandit_result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    issues = bandit_data.get("results", [])
                    
                    # Count issues by severity
                    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                    for issue in issues:
                        severity = issue.get("issue_severity", "LOW")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    logger.info(f"Security scan completed: {len(issues)} total issues")
                    for severity, count in severity_counts.items():
                        if count > 0:
                            logger.info(f"- {severity}: {count}")
                    
                    if severity_counts.get("HIGH", 0) > 0:
                        logger.warning("High severity security issues found!")
                    
                except json.JSONDecodeError:
                    logger.warning("Could not parse Bandit output")
                
                self.success = True
            else:
                self.error_message = bandit_result.stderr
                self.success = False
        
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success


class HealthCheckTask(MaintenanceTask):
    """Perform application health checks."""
    
    def __init__(self):
        super().__init__(
            "Health Checks",
            "Verify application components and dependencies"
        )
    
    def execute(self, project_root: Path) -> bool:
        """Execute health checks."""
        start_time = datetime.datetime.now()
        
        try:
            health_checks = []
            
            # Check Python import
            try:
                import_result = subprocess.run(
                    ["python", "-c", "import holo_code_gen; print('OK')"],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                health_checks.append(("Python Import", import_result.returncode == 0))
            except Exception:
                health_checks.append(("Python Import", False))
            
            # Check photonic library integrity
            try:
                library_result = subprocess.run(
                    ["python", "-c", "from holo_code_gen.templates import IMECLibrary; IMECLibrary.verify_integrity(); print('OK')"],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                health_checks.append(("Photonic Library", library_result.returncode == 0))
            except Exception:
                health_checks.append(("Photonic Library", False))
            
            # Check essential dependencies
            essential_deps = ["numpy", "scipy", "torch", "networkx", "pydantic"]
            for dep in essential_deps:
                try:
                    dep_result = subprocess.run(
                        ["python", "-c", f"import {dep}; print('OK')"],
                        capture_output=True,
                        text=True,
                        cwd=project_root
                    )
                    health_checks.append((f"Dependency: {dep}", dep_result.returncode == 0))
                except Exception:
                    health_checks.append((f"Dependency: {dep}", False))
            
            # Report results
            passed_checks = sum(1 for _, status in health_checks if status)
            total_checks = len(health_checks)
            
            logger.info(f"Health checks: {passed_checks}/{total_checks} passed")
            
            for check_name, status in health_checks:
                status_icon = "✅" if status else "❌"
                logger.info(f"{status_icon} {check_name}")
            
            self.success = passed_checks == total_checks
            if not self.success:
                self.error_message = f"Only {passed_checks}/{total_checks} health checks passed"
        
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success


class PerformanceMonitoringTask(MaintenanceTask):
    """Monitor application performance."""
    
    def __init__(self):
        super().__init__(
            "Performance Monitoring",
            "Run performance benchmarks and collect metrics"
        )
    
    def execute(self, project_root: Path) -> bool:
        """Execute performance monitoring."""
        start_time = datetime.datetime.now()
        
        try:
            # Run basic performance test
            perf_script = project_root / "scripts" / "performance_test.py"
            
            if perf_script.exists():
                perf_result = subprocess.run(
                    ["python", str(perf_script)],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                
                if perf_result.returncode == 0:
                    logger.info("Performance benchmark completed")
                    logger.info(perf_result.stdout)
                else:
                    logger.warning("Performance benchmark failed")
                    logger.warning(perf_result.stderr)
            else:
                # Basic import performance test
                import_time_result = subprocess.run(
                    ["python", "-c", 
                     "import time; start=time.time(); import holo_code_gen; "
                     "print(f'Import time: {time.time()-start:.3f}s')"],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                
                if import_time_result.returncode == 0:
                    logger.info(f"Basic performance check: {import_time_result.stdout.strip()}")
            
            self.success = True
        
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success


class ReportGenerationTask(MaintenanceTask):
    """Generate maintenance report."""
    
    def __init__(self, tasks: List[MaintenanceTask]):
        super().__init__(
            "Report Generation",
            "Generate comprehensive maintenance report"
        )
        self.tasks = tasks
    
    def execute(self, project_root: Path) -> bool:
        """Generate maintenance report."""
        start_time = datetime.datetime.now()
        
        try:
            report_data = {
                "maintenance_run": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "project_root": str(project_root),
                    "total_tasks": len(self.tasks),
                    "successful_tasks": sum(1 for task in self.tasks if task.success),
                    "failed_tasks": sum(1 for task in self.tasks if not task.success)
                },
                "task_results": []
            }
            
            for task in self.tasks:
                task_data = {
                    "name": task.name,
                    "description": task.description,
                    "success": task.success,
                    "execution_time": task.execution_time,
                    "error_message": task.error_message
                }
                report_data["task_results"].append(task_data)
            
            # Save JSON report
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_report_path = reports_dir / f"maintenance_report_{timestamp}.json"
            
            with open(json_report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Generate markdown report
            md_report_path = reports_dir / f"maintenance_report_{timestamp}.md"
            self._generate_markdown_report(report_data, md_report_path)
            
            logger.info(f"Maintenance report saved to {json_report_path}")
            logger.info(f"Markdown report saved to {md_report_path}")
            
            self.success = True
        
        except Exception as e:
            self.error_message = str(e)
            self.success = False
        
        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return self.success
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], output_path: Path):
        """Generate markdown format report."""
        with open(output_path, 'w') as f:
            f.write("# Automated Maintenance Report\n\n")
            
            # Summary
            run_info = report_data["maintenance_run"]
            f.write(f"**Generated:** {run_info['timestamp']}\n")
            f.write(f"**Project:** {run_info['project_root']}\n")
            f.write(f"**Total Tasks:** {run_info['total_tasks']}\n")
            f.write(f"**Successful:** {run_info['successful_tasks']}\n")
            f.write(f"**Failed:** {run_info['failed_tasks']}\n\n")
            
            # Task Results
            f.write("## Task Results\n\n")
            
            for task in report_data["task_results"]:
                status_icon = "✅" if task["success"] else "❌"
                f.write(f"### {status_icon} {task['name']}\n\n")
                f.write(f"**Description:** {task['description']}\n")
                f.write(f"**Status:** {'Success' if task['success'] else 'Failed'}\n")
                
                if task["execution_time"]:
                    f.write(f"**Execution Time:** {task['execution_time']:.2f}s\n")
                
                if task["error_message"]:
                    f.write(f"**Error:** {task['error_message']}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            failed_tasks = [task for task in report_data["task_results"] if not task["success"]]
            
            if failed_tasks:
                f.write("### Failed Tasks\n\n")
                for task in failed_tasks:
                    f.write(f"- **{task['name']}**: {task['error_message'] or 'Unknown error'}\n")
                f.write("\n")
            else:
                f.write("All maintenance tasks completed successfully! 🎉\n\n")
            
            f.write("### Next Steps\n\n")
            f.write("1. Review any failed tasks and address underlying issues\n")
            f.write("2. Monitor system performance and security metrics\n")
            f.write("3. Schedule next maintenance run\n")
            f.write("4. Update dependencies if security vulnerabilities were found\n")


class AutomatedMaintenance:
    """Main class for coordinating automated maintenance tasks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks = []
    
    def add_task(self, task: MaintenanceTask):
        """Add a maintenance task to the queue."""
        self.tasks.append(task)
    
    def run_all_tasks(self) -> bool:
        """Run all registered maintenance tasks."""
        logger.info(f"Starting automated maintenance for {self.project_root}")
        logger.info(f"Scheduled tasks: {len(self.tasks)}")
        
        start_time = datetime.datetime.now()
        
        for i, task in enumerate(self.tasks, 1):
            logger.info(f"Running task {i}/{len(self.tasks)}: {task.name}")
            task.execute(self.project_root)
            task.log_result()
        
        # Generate final report
        report_task = ReportGenerationTask(self.tasks)
        report_task.execute(self.project_root)
        report_task.log_result()
        
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        successful_tasks = sum(1 for task in self.tasks if task.success)
        
        logger.info(f"Maintenance completed in {total_time:.2f}s")
        logger.info(f"Results: {successful_tasks}/{len(self.tasks)} tasks successful")
        
        return successful_tasks == len(self.tasks)


def main():
    """Main function to run automated maintenance."""
    parser = argparse.ArgumentParser(description="Run automated maintenance for Holo-Code-Gen")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root directory"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["cleanup", "dependencies", "security", "health", "performance", "all"],
        default=["all"],
        help="Maintenance tasks to run"
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
        maintenance = AutomatedMaintenance(args.project_root)
        
        # Add requested tasks
        if "all" in args.tasks:
            task_classes = [
                CleanupTask,
                DependencyUpdateTask,
                SecurityScanTask,
                HealthCheckTask,
                PerformanceMonitoringTask
            ]
        else:
            task_mapping = {
                "cleanup": CleanupTask,
                "dependencies": DependencyUpdateTask,
                "security": SecurityScanTask,
                "health": HealthCheckTask,
                "performance": PerformanceMonitoringTask
            }
            task_classes = [task_mapping[task] for task in args.tasks if task != "all"]
        
        for task_class in task_classes:
            maintenance.add_task(task_class())
        
        # Run maintenance
        success = maintenance.run_all_tasks()
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())