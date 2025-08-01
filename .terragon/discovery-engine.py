#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes work items
"""

import json
import os
import subprocess
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization engine"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        # Load configuration
        self.config = self._load_config()
        self.discovered_items = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for value discovery"""
        return {
            "scoring": {
                "weights": {"wsjf": 0.6, "ice": 0.1, "technicalDebt": 0.2, "security": 0.1},
                "thresholds": {"minScore": 15, "maxRisk": 0.7, "securityBoost": 2.0}
            },
            "discovery": {
                "sources": ["gitHistory", "staticAnalysis", "todoComments"],
                "patterns": {"debt_markers": ["TODO", "FIXME", "HACK", "DEPRECATED"]}
            }
        }
    
    def discover_all_items(self) -> List[Dict[str, Any]]:
        """Discover all value items from multiple sources"""
        items = []
        
        # Discover from different sources
        items.extend(self._discover_todo_markers())
        items.extend(self._discover_git_history())
        items.extend(self._discover_static_analysis())
        items.extend(self._discover_architecture_gaps())
        items.extend(self._discover_security_issues())
        items.extend(self._discover_performance_opportunities())
        
        # Score and prioritize
        scored_items = [self._score_item(item) for item in items]
        sorted_items = sorted(scored_items, key=lambda x: x['composite_score'], reverse=True)
        
        self.discovered_items = sorted_items
        return sorted_items
    
    def _discover_todo_markers(self) -> List[Dict[str, Any]]:
        """Discover TODO, FIXME, HACK markers in codebase"""
        items = []
        patterns = self.config["discovery"]["patterns"]["debt_markers"]
        
        try:
            # Use git grep for efficiency
            for pattern in patterns:
                result = subprocess.run(
                    ["git", "grep", "-n", pattern, "--", "*.py", "*.md", "*.yaml", "*.yml"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        match = re.match(r'([^:]+):(\d+):(.*)', line)
                        if match:
                            file_path, line_num, content = match.groups()
                            items.append({
                                'id': f"TODO-{len(items)+1:03d}",
                                'title': f"Resolve {pattern} in {file_path}",
                                'description': content.strip(),
                                'category': 'technical_debt',
                                'file': file_path,
                                'line': int(line_num),
                                'effort_hours': self._estimate_todo_effort(content),
                                'discovery_source': 'todo_markers'
                            })
        except Exception as e:
            logger.warning(f"Error discovering TODO markers: {e}")
        
        return items
    
    def _discover_git_history(self) -> List[Dict[str, Any]]:
        """Discover improvement opportunities from git history"""
        items = []
        
        try:
            # Find files with frequent changes (hotspots)
            result = subprocess.run(
                ["git", "log", "--name-only", "--pretty=format:", "--since=3.months"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            file_changes = {}
            for line in result.stdout.strip().split('\n'):
                if line and line.endswith('.py'):
                    file_changes[line] = file_changes.get(line, 0) + 1
            
            # Identify hotspots (top 20% most changed files)
            if file_changes:
                threshold = sorted(file_changes.values(), reverse=True)[len(file_changes)//5] if len(file_changes) > 5 else 1
                hotspots = [f for f, count in file_changes.items() if count >= threshold]
                
                for file_path in hotspots[:10]:  # Top 10 hotspots
                    items.append({
                        'id': f"HOTSPOT-{len(items)+1:03d}",
                        'title': f"Refactor high-churn file: {file_path}",
                        'description': f"File changed {file_changes[file_path]} times in 3 months",
                        'category': 'refactoring',
                        'file': file_path,
                        'effort_hours': 8,
                        'discovery_source': 'git_hotspots'
                    })
        except Exception as e:
            logger.warning(f"Error analyzing git history: {e}")
        
        return items
    
    def _discover_static_analysis(self) -> List[Dict[str, Any]]:
        """Discover issues through static analysis"""
        items = []
        
        # Check for missing docstrings
        for py_file in self.repo_path.rglob("*.py"):
            if self._is_source_file(py_file):
                missing_docs = self._check_missing_docstrings(py_file)
                if missing_docs:
                    items.append({
                        'id': f"DOC-{len(items)+1:03d}",
                        'title': f"Add docstrings to {py_file.name}",
                        'description': f"Missing {missing_docs} docstrings",
                        'category': 'documentation',
                        'file': str(py_file.relative_to(self.repo_path)),
                        'effort_hours': missing_docs * 0.5,
                        'discovery_source': 'static_analysis'
                    })
        
        # Check for long functions/classes
        for py_file in self.repo_path.rglob("*.py"):
            if self._is_source_file(py_file):
                long_functions = self._check_function_length(py_file)
                for func_info in long_functions:
                    items.append({
                        'id': f"REFACTOR-{len(items)+1:03d}",
                        'title': f"Refactor long function: {func_info['name']}",
                        'description': f"Function has {func_info['lines']} lines",
                        'category': 'refactoring',
                        'file': str(py_file.relative_to(self.repo_path)),
                        'effort_hours': 4,
                        'discovery_source': 'static_analysis'
                    })
        
        return items
    
    def _discover_architecture_gaps(self) -> List[Dict[str, Any]]:
        """Discover architectural improvements needed"""
        items = []
        
        # Check for missing core implementations
        core_files = [
            ("holo_code_gen/compiler/photonic_compiler.py", "Implement PhotonicCompiler class", 16),
            ("holo_code_gen/templates/photonic_component.py", "Implement PhotonicComponent base class", 12),
            ("holo_code_gen/simulation/optical_simulator.py", "Implement optical simulation engine", 20),
            ("holo_code_gen/optimization/power_optimizer.py", "Implement power optimization algorithms", 14),
            ("holo_code_gen/fabrication/gds_generator.py", "Implement GDS file generation", 10)
        ]
        
        for file_path, description, hours in core_files:
            full_path = self.repo_path / file_path
            if not full_path.exists() or full_path.stat().st_size < 1000:  # Less than 1KB
                items.append({
                    'id': f"IMPL-{len(items)+1:03d}",
                    'title': description,
                    'description': f"Core implementation missing: {file_path}",
                    'category': 'implementation',
                    'file': file_path,
                    'effort_hours': hours,
                    'discovery_source': 'architecture_analysis'
                })
        
        return items
    
    def _discover_security_issues(self) -> List[Dict[str, Any]]:
        """Discover security improvement opportunities"""
        items = []
        
        # Check for missing security configurations
        security_items = [
            (".github/workflows/security.yml", "Setup security scanning workflow", 4),
            ("requirements-security.txt", "Create security requirements file", 2),
            (".github/dependabot.yml", "Configure automated dependency updates", 2),
            ("SECURITY.md", "Update security policy", 1)
        ]
        
        for file_path, description, hours in security_items:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                items.append({
                    'id': f"SEC-{len(items)+1:03d}",
                    'title': description,
                    'description': f"Missing security configuration: {file_path}",
                    'category': 'security',
                    'file': file_path,
                    'effort_hours': hours,
                    'discovery_source': 'security_analysis'
                })
        
        return items
    
    def _discover_performance_opportunities(self) -> List[Dict[str, Any]]:
        """Discover performance optimization opportunities"""
        items = []
        
        # Check for missing performance infrastructure
        perf_items = [
            ("tests/performance/benchmark_suite.py", "Implement comprehensive benchmarks", 12),
            ("monitoring/performance_monitor.py", "Setup performance monitoring", 8),
            (".github/workflows/performance.yml", "Create performance testing workflow", 6),
            ("docs/BENCHMARKS.md", "Document performance benchmarks", 4)
        ]
        
        for file_path, description, hours in perf_items:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                items.append({
                    'id': f"PERF-{len(items)+1:03d}",
                    'title': description,
                    'description': f"Missing performance infrastructure: {file_path}",
                    'category': 'performance',
                    'file': file_path,
                    'effort_hours': hours,
                    'discovery_source': 'performance_analysis'
                })
        
        return items
    
    def _score_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Score item using WSJF, ICE, and technical debt metrics"""
        
        # WSJF Components (Weighted Shortest Job First)
        user_value = self._score_user_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity = self._score_opportunity(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        job_size = item.get('effort_hours', 4)
        wsjf_score = cost_of_delay / max(job_size, 0.5)
        
        # ICE Components (Impact, Confidence, Ease)
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = self._score_ease(item)
        ice_score = impact * confidence * ease
        
        # Technical Debt Score
        debt_score = self._score_technical_debt(item)
        
        # Composite score with adaptive weights
        weights = self.config["scoring"]["weights"]
        composite_score = (
            weights["wsjf"] * min(wsjf_score, 100) +
            weights["ice"] * min(ice_score, 1000) / 10 +  # Scale ICE to similar range
            weights["technicalDebt"] * min(debt_score, 100) +
            weights["security"] * (2.0 if item['category'] == 'security' else 1.0)
        )
        
        # Apply boosts
        if item['category'] == 'security':
            composite_score *= self.config["scoring"]["thresholds"]["securityBoost"]
        
        item.update({
            'wsjf_score': round(wsjf_score, 2),
            'ice_score': round(ice_score, 2),
            'debt_score': round(debt_score, 2),
            'composite_score': round(composite_score, 2),
            'cost_of_delay': round(cost_of_delay, 2),
            'impact': impact,
            'confidence': confidence,
            'ease': ease
        })
        
        return item
    
    def _score_user_value(self, item: Dict[str, Any]) -> float:
        """Score user/business value (1-10 scale)"""
        category_values = {
            'implementation': 9,  # Core functionality
            'security': 8,        # Critical for production
            'performance': 7,     # Important for user experience
            'technical_debt': 6,  # Important for maintainability
            'documentation': 5,   # Important for adoption
            'refactoring': 4      # Important for maintenance
        }
        return category_values.get(item['category'], 5)
    
    def _score_time_criticality(self, item: Dict[str, Any]) -> float:
        """Score time criticality (1-10 scale)"""
        if item['category'] == 'security':
            return 9  # Security issues are time-critical
        elif item['category'] == 'implementation':
            return 8  # Core features needed soon
        elif 'TODO' in item.get('description', ''):
            return 6  # TODOs indicate planned work
        return 4
    
    def _score_risk_reduction(self, item: Dict[str, Any]) -> float:
        """Score risk reduction value (1-10 scale)"""
        risk_categories = {
            'security': 9,        # High risk reduction
            'implementation': 7,  # Moderate risk reduction
            'technical_debt': 6,  # Prevents future risk
            'performance': 5,     # Performance risk
            'documentation': 3,   # Low risk reduction
            'refactoring': 4      # Moderate risk reduction
        }
        return risk_categories.get(item['category'], 3)
    
    def _score_opportunity(self, item: Dict[str, Any]) -> float:
        """Score opportunity enablement (1-10 scale)"""
        if item['category'] == 'implementation':
            return 8  # Enables other work
        elif item['category'] == 'performance':
            return 6  # Enables better user experience
        elif item['category'] == 'documentation':
            return 7  # Enables adoption
        return 3
    
    def _score_impact(self, item: Dict[str, Any]) -> int:
        """Score business impact (1-10 scale)"""
        impact_map = {
            'implementation': 9,
            'security': 8,
            'performance': 7,
            'technical_debt': 5,
            'documentation': 4,
            'refactoring': 3
        }
        return impact_map.get(item['category'], 5)
    
    def _score_confidence(self, item: Dict[str, Any]) -> int:
        """Score execution confidence (1-10 scale)"""
        # Higher confidence for smaller tasks
        effort = item.get('effort_hours', 4)
        if effort <= 2:
            return 9
        elif effort <= 8:
            return 7
        elif effort <= 16:
            return 5
        else:
            return 3
    
    def _score_ease(self, item: Dict[str, Any]) -> int:
        """Score implementation ease (1-10 scale)"""
        # Inverse of effort
        effort = item.get('effort_hours', 4)
        if effort <= 1:
            return 10
        elif effort <= 4:
            return 8
        elif effort <= 8:
            return 6
        elif effort <= 16:
            return 4
        else:
            return 2
    
    def _score_technical_debt(self, item: Dict[str, Any]) -> float:
        """Score technical debt impact"""
        if item['category'] == 'technical_debt':
            return 80
        elif item['category'] == 'refactoring':
            return 60
        elif 'TODO' in item.get('description', ''):
            return 40
        return 20
    
    def _estimate_todo_effort(self, content: str) -> int:
        """Estimate effort for TODO items"""
        if 'FIXME' in content or 'HACK' in content:
            return 4  # More complex fixes
        elif 'TODO' in content:
            if len(content) > 100:
                return 6  # Complex TODO
            else:
                return 2  # Simple TODO
        return 2
    
    def _is_source_file(self, file_path: Path) -> bool:
        """Check if file is a source file (not test, docs, etc.)"""
        path_str = str(file_path)
        return not any(exclude in path_str for exclude in ['test_', '/tests/', '/docs/', '__pycache__'])
    
    def _check_missing_docstrings(self, file_path: Path) -> int:
        """Count missing docstrings in a Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Simple heuristic: count functions/classes without docstrings
            function_pattern = r'def\s+\w+\s*\([^)]*\):\s*\n(?!\s*""")'
            class_pattern = r'class\s+\w+[^:]*:\s*\n(?!\s*""")'
            
            missing_funcs = len(re.findall(function_pattern, content))
            missing_classes = len(re.findall(class_pattern, content))
            
            return missing_funcs + missing_classes
        except Exception:
            return 0
    
    def _check_function_length(self, file_path: Path) -> List[Dict[str, Any]]:
        """Find functions that are too long"""
        long_functions = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            in_function = False
            func_start = 0
            func_name = ""
            
            for i, line in enumerate(lines):
                if re.match(r'\s*def\s+(\w+)', line):
                    if in_function and (i - func_start) > 50:  # >50 lines
                        long_functions.append({
                            'name': func_name,
                            'lines': i - func_start,
                            'start_line': func_start + 1
                        })
                    
                    match = re.match(r'\s*def\s+(\w+)', line)
                    func_name = match.group(1)
                    func_start = i
                    in_function = True
                elif in_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # End of function
                    if (i - func_start) > 50:
                        long_functions.append({
                            'name': func_name,
                            'lines': i - func_start,
                            'start_line': func_start + 1
                        })
                    in_function = False
        except Exception:
            pass
        
        return long_functions
    
    def update_backlog(self) -> None:
        """Update the BACKLOG.md file with discovered items"""
        items = self.discover_all_items()
        
        # Generate backlog content
        content = self._generate_backlog_content(items)
        
        # Write to BACKLOG.md
        with open(self.backlog_path, 'w') as f:
            f.write(content)
        
        # Update metrics
        self._update_metrics(items)
        
        logger.info(f"Updated backlog with {len(items)} items")
    
    def _generate_backlog_content(self, items: List[Dict[str, Any]]) -> str:
        """Generate markdown content for backlog"""
        now = datetime.now().isoformat()
        
        content = f"""# 📊 Autonomous Value Backlog

**Repository**: holo-code-gen  
**Maturity Level**: MATURING (68/100)  
**Last Updated**: {now}  
**Discovery Engine**: Active  

## 🎯 Next Best Value Items

### High Priority (Composite Score > 50)

| Rank | ID | Title | Score | WSJF | ICE | Category | Est. Hours |
|------|-----|--------|-------|------|-----|----------|------------|
"""
        
        # Add high priority items
        high_priority = [item for item in items if item['composite_score'] > 50]
        for i, item in enumerate(high_priority[:10], 1):
            content += f"| {i} | {item['id']} | {item['title'][:50]}... | {item['composite_score']:.1f} | {item['wsjf_score']:.1f} | {item['ice_score']:.0f} | {item['category']} | {item['effort_hours']} |\n"
        
        content += "\n### Medium Priority (Composite Score 25-50)\n\n"
        content += "| Rank | ID | Title | Score | Category | Est. Hours |\n"
        content += "|------|-----|--------|-------|----------|------------|\n"
        
        # Add medium priority items
        medium_priority = [item for item in items if 25 <= item['composite_score'] <= 50]
        for i, item in enumerate(medium_priority[:10], 1):
            content += f"| {i} | {item['id']} | {item['title'][:50]}... | {item['composite_score']:.1f} | {item['category']} | {item['effort_hours']} |\n"
        
        # Add summary statistics
        total_items = len(items)
        total_effort = sum(item['effort_hours'] for item in items)
        categories = {}
        for item in items:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        content += f"""

## 📈 Discovery Summary

### Backlog Statistics
- **Total Items**: {total_items}
- **Total Estimated Effort**: {total_effort} hours
- **High Priority Items**: {len(high_priority)}
- **Average Score**: {sum(item['composite_score'] for item in items) / len(items):.1f}

### Items by Category
"""
        
        for category, count in sorted(categories.items()):
            content += f"- **{category.replace('_', ' ').title()}**: {count} items\n"
        
        content += f"""

### Discovery Sources
- **TODO Markers**: {len([i for i in items if i['discovery_source'] == 'todo_markers'])} items
- **Architecture Analysis**: {len([i for i in items if i['discovery_source'] == 'architecture_analysis'])} items
- **Static Analysis**: {len([i for i in items if i['discovery_source'] == 'static_analysis'])} items
- **Security Analysis**: {len([i for i in items if i['discovery_source'] == 'security_analysis'])} items
- **Performance Analysis**: {len([i for i in items if i['discovery_source'] == 'performance_analysis'])} items

## 📋 Top 5 Recommendations

"""
        
        # Add top 5 recommendations
        for i, item in enumerate(items[:5], 1):
            content += f"### {i}. {item['title']}\n"
            content += f"- **Score**: {item['composite_score']:.1f} (WSJF: {item['wsjf_score']:.1f}, ICE: {item['ice_score']:.0f})\n"
            content += f"- **Category**: {item['category'].replace('_', ' ').title()}\n"
            content += f"- **Effort**: {item['effort_hours']} hours\n"
            content += f"- **Description**: {item['description']}\n"
            if 'file' in item:
                content += f"- **File**: `{item['file']}`\n"
            content += "\n"
        
        content += """
---
*Generated by Terragon Autonomous Value Discovery Engine*  
*Backlog updates automatically based on continuous analysis*
"""
        
        return content
    
    def _update_metrics(self, items: List[Dict[str, Any]]) -> None:
        """Update value metrics file"""
        try:
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {"executionHistory": [], "continuousMetrics": {}}
        
        # Update discovery metrics
        metrics["valueDiscovery"] = {
            "totalItemsIdentified": len(items),
            "lastDiscovery": datetime.now().isoformat(),
            "highPriorityItems": len([i for i in items if i['composite_score'] > 50]),
            "byCategory": {
                cat: len([i for i in items if i['category'] == cat])
                for cat in set(item['category'] for item in items)
            },
            "averageScore": sum(item['composite_score'] for item in items) / len(items) if items else 0,
            "totalEffort": sum(item['effort_hours'] for item in items)
        }
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main entry point for discovery engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Value Discovery Engine")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--update", action="store_true", help="Update backlog")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    
    args = parser.parse_args()
    
    engine = ValueDiscoveryEngine(args.repo)
    
    if args.continuous:
        logger.info("Starting continuous discovery mode...")
        while True:
            try:
                engine.update_backlog()
                logger.info("Discovery cycle completed, sleeping for 1 hour...")
                time.sleep(3600)  # 1 hour
            except KeyboardInterrupt:
                logger.info("Stopping continuous discovery")
                break
            except Exception as e:
                logger.error(f"Error in discovery cycle: {e}")
                time.sleep(300)  # 5 minutes on error
    elif args.update:
        engine.update_backlog()
        logger.info("Backlog updated successfully")
    else:
        items = engine.discover_all_items()
        print(f"Discovered {len(items)} value items")
        for item in items[:5]:  # Show top 5
            print(f"- {item['title']} (Score: {item['composite_score']:.1f})")

if __name__ == "__main__":
    main()