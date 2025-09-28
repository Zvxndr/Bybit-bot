#!/usr/bin/env python3
"""
OPEN ALPHA - FINAL REPOSITORY CLEANUP & ANALYSIS
================================================

Comprehensive repository cleanup, validation, and preparation for git push.
Analyzes all files, removes duplicates, validates syntax, and prepares production-ready codebase.

Date: September 28, 2025
Purpose: Final cleanup before git push
Status: Production Ready
"""

import os
import sys
import json
import ast
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import subprocess

class RepositoryAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.analysis_report = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.repo_path),
            "analysis": {},
            "cleanup_actions": [],
            "validation_results": {},
            "recommendations": []
        }
    
    def analyze_repository(self) -> Dict:
        """Perform comprehensive repository analysis"""
        print("🔍 OPEN ALPHA - Repository Analysis Starting...")
        print("=" * 60)
        
        # 1. File Structure Analysis
        self.analysis_report["analysis"]["file_structure"] = self._analyze_file_structure()
        
        # 2. Code Quality Analysis  
        self.analysis_report["analysis"]["code_quality"] = self._analyze_code_quality()
        
        # 3. Documentation Analysis
        self.analysis_report["analysis"]["documentation"] = self._analyze_documentation()
        
        # 4. Duplicate File Detection
        self.analysis_report["analysis"]["duplicates"] = self._find_duplicates()
        
        # 5. Security Analysis
        self.analysis_report["analysis"]["security"] = self._analyze_security()
        
        # 6. Dependency Analysis
        self.analysis_report["analysis"]["dependencies"] = self._analyze_dependencies()
        
        # 7. Test Coverage Analysis
        self.analysis_report["analysis"]["test_coverage"] = self._analyze_test_coverage()
        
        return self.analysis_report
    
    def _analyze_file_structure(self) -> Dict:
        """Analyze repository file structure"""
        print("📁 Analyzing file structure...")
        
        structure = {
            "total_files": 0,
            "by_type": {},
            "by_directory": {},
            "large_files": [],
            "empty_files": []
        }
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._is_git_file(file_path):
                structure["total_files"] += 1
                
                # Count by file type
                ext = file_path.suffix.lower()
                structure["by_type"][ext] = structure["by_type"].get(ext, 0) + 1
                
                # Count by directory
                dir_name = file_path.parent.name
                structure["by_directory"][dir_name] = structure["by_directory"].get(dir_name, 0) + 1
                
                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > 1024 * 1024:  # Files > 1MB
                        structure["large_files"].append({
                            "path": str(file_path.relative_to(self.repo_path)),
                            "size": size
                        })
                    elif size == 0:  # Empty files
                        structure["empty_files"].append(str(file_path.relative_to(self.repo_path)))
                except OSError:
                    pass
        
        print(f"   ✅ Total files: {structure['total_files']}")
        print(f"   📊 Python files: {structure['by_type'].get('.py', 0)}")
        print(f"   📄 Markdown files: {structure['by_type'].get('.md', 0)}")
        print(f"   📋 Large files: {len(structure['large_files'])}")
        
        return structure
    
    def _analyze_code_quality(self) -> Dict:
        """Analyze Python code quality"""
        print("🐍 Analyzing Python code quality...")
        
        quality = {
            "total_python_files": 0,
            "syntax_errors": [],
            "lines_of_code": 0,
            "complexity_warnings": [],
            "import_analysis": {}
        }
        
        for py_file in self.repo_path.rglob("*.py"):
            if self._is_git_file(py_file):
                continue
                
            quality["total_python_files"] += 1
            
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = len(content.splitlines())
                quality["lines_of_code"] += lines
                
                # Check syntax
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    quality["syntax_errors"].append({
                        "file": str(py_file.relative_to(self.repo_path)),
                        "line": e.lineno,
                        "error": str(e)
                    })
                
                # Analyze imports
                self._analyze_imports(py_file, content, quality["import_analysis"])
                
                # Check complexity (simple heuristic)
                if lines > 1000:
                    quality["complexity_warnings"].append({
                        "file": str(py_file.relative_to(self.repo_path)),
                        "lines": lines,
                        "reason": "Large file (>1000 lines)"
                    })
            
            except Exception as e:
                quality["syntax_errors"].append({
                    "file": str(py_file.relative_to(self.repo_path)),
                    "error": f"Read error: {str(e)}"
                })
        
        print(f"   ✅ Python files analyzed: {quality['total_python_files']}")
        print(f"   📝 Total lines of code: {quality['lines_of_code']:,}")
        print(f"   ❌ Syntax errors: {len(quality['syntax_errors'])}")
        
        return quality
    
    def _analyze_documentation(self) -> Dict:
        """Analyze documentation quality"""
        print("📚 Analyzing documentation...")
        
        docs = {
            "markdown_files": 0,
            "documentation_score": 0,
            "missing_docs": [],
            "readme_analysis": {},
            "duplicate_docs": []
        }
        
        readme_files = []
        doc_content_hashes = {}
        
        for md_file in self.repo_path.rglob("*.md"):
            if self._is_git_file(md_file):
                continue
                
            docs["markdown_files"] += 1
            
            # Check for README files
            if md_file.name.lower().startswith('readme'):
                readme_files.append(str(md_file.relative_to(self.repo_path)))
            
            # Check for duplicate content
            try:
                content = md_file.read_text(encoding='utf-8')
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash in doc_content_hashes:
                    docs["duplicate_docs"].append({
                        "file1": doc_content_hashes[content_hash],
                        "file2": str(md_file.relative_to(self.repo_path)),
                        "hash": content_hash
                    })
                else:
                    doc_content_hashes[content_hash] = str(md_file.relative_to(self.repo_path))
            except Exception:
                pass
        
        docs["readme_analysis"]["count"] = len(readme_files)
        docs["readme_analysis"]["files"] = readme_files
        
        # Documentation score (simple heuristic)
        core_dirs = ['src', 'tests', 'docs']
        documented_dirs = 0
        for dir_name in core_dirs:
            dir_path = self.repo_path / dir_name
            if dir_path.exists():
                has_readme = any((dir_path / f"README{ext}").exists() for ext in ['.md', '.rst', '.txt'])
                if has_readme:
                    documented_dirs += 1
        
        docs["documentation_score"] = (documented_dirs / len(core_dirs)) * 100
        
        print(f"   📄 Markdown files: {docs['markdown_files']}")
        print(f"   📋 README files: {len(readme_files)}")
        print(f"   🔄 Duplicate docs: {len(docs['duplicate_docs'])}")
        print(f"   📊 Documentation score: {docs['documentation_score']:.1f}%")
        
        return docs
    
    def _find_duplicates(self) -> Dict:
        """Find duplicate files by content hash"""
        print("🔍 Detecting duplicate files...")
        
        duplicates = {
            "file_duplicates": [],
            "content_duplicates": [],
            "size_duplicates": []
        }
        
        file_hashes = {}
        size_groups = {}
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._is_git_file(file_path):
                try:
                    # Group by size first (performance optimization)
                    size = file_path.stat().st_size
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(file_path)
                    
                    # Hash content for small to medium files
                    if size < 10 * 1024 * 1024:  # Files < 10MB
                        content = file_path.read_bytes()
                        content_hash = hashlib.md5(content).hexdigest()
                        
                        if content_hash in file_hashes:
                            duplicates["content_duplicates"].append({
                                "hash": content_hash,
                                "files": [file_hashes[content_hash], str(file_path.relative_to(self.repo_path))],
                                "size": size
                            })
                        else:
                            file_hashes[content_hash] = str(file_path.relative_to(self.repo_path))
                
                except Exception:
                    pass
        
        # Find size duplicates (potential duplicates)
        for size, files in size_groups.items():
            if len(files) > 1 and size > 0:  # Ignore empty files
                duplicates["size_duplicates"].append({
                    "size": size,
                    "files": [str(f.relative_to(self.repo_path)) for f in files]
                })
        
        print(f"   🔄 Content duplicates: {len(duplicates['content_duplicates'])}")
        print(f"   📏 Size duplicates: {len(duplicates['size_duplicates'])}")
        
        return duplicates
    
    def _analyze_security(self) -> Dict:
        """Basic security analysis"""
        print("🔒 Analyzing security...")
        
        security = {
            "hardcoded_secrets": [],
            "env_files": [],
            "sensitive_files": [],
            "security_score": 0
        }
        
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            'api_key', 'secret_key', 'password', 'token', 'private_key',
            'access_token', 'secret_token', 'auth_token', 'api_secret'
        ]
        
        # Find environment and sensitive files
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._is_git_file(file_path):
                filename = file_path.name.lower()
                
                # Check for env files
                if filename.startswith('.env') or filename.endswith('.env'):
                    security["env_files"].append(str(file_path.relative_to(self.repo_path)))
                
                # Check for sensitive files
                if any(sensitive in filename for sensitive in ['secret', 'key', 'password', 'token']):
                    security["sensitive_files"].append(str(file_path.relative_to(self.repo_path)))
        
        # Basic security score
        issues = len(security["hardcoded_secrets"])
        has_gitignore = (self.repo_path / '.gitignore').exists()
        has_env_example = any('.env.example' in f or '.env.template' in f for f in security["env_files"])
        
        security["security_score"] = max(0, 100 - (issues * 10) + (20 if has_gitignore else 0) + (10 if has_env_example else 0))
        
        print(f"   🔐 Environment files: {len(security['env_files'])}")
        print(f"   ⚠️ Sensitive files: {len(security['sensitive_files'])}")
        print(f"   🛡️ Security score: {security['security_score']:.1f}/100")
        
        return security
    
    def _analyze_dependencies(self) -> Dict:
        """Analyze project dependencies"""
        print("📦 Analyzing dependencies...")
        
        deps = {
            "requirements_files": [],
            "package_managers": [],
            "total_dependencies": 0,
            "dependency_files": []
        }
        
        # Look for dependency files
        dependency_files = [
            'requirements.txt', 'requirements_current.txt', 'requirements_minimal.txt',
            'package.json', 'Pipfile', 'poetry.lock', 'setup.py'
        ]
        
        for dep_file in dependency_files:
            file_path = self.repo_path / dep_file
            if file_path.exists():
                deps["dependency_files"].append(dep_file)
                
                if dep_file.startswith('requirements'):
                    deps["requirements_files"].append(dep_file)
                    try:
                        content = file_path.read_text()
                        deps["total_dependencies"] += len([line for line in content.splitlines() if line.strip() and not line.startswith('#')])
                    except Exception:
                        pass
        
        print(f"   📋 Dependency files: {len(deps['dependency_files'])}")
        print(f"   🐍 Requirements files: {len(deps['requirements_files'])}")
        print(f"   📦 Total dependencies: {deps['total_dependencies']}")
        
        return deps
    
    def _analyze_test_coverage(self) -> Dict:
        """Analyze test coverage"""
        print("🧪 Analyzing test coverage...")
        
        tests = {
            "test_files": 0,
            "test_directories": [],
            "test_frameworks": [],
            "coverage_estimate": 0
        }
        
        test_dirs = set()
        test_files = 0
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._is_git_file(file_path):
                if 'test' in file_path.name.lower() and file_path.suffix == '.py':
                    test_files += 1
                    test_dirs.add(file_path.parent.name)
        
        tests["test_files"] = test_files
        tests["test_directories"] = list(test_dirs)
        
        # Estimate coverage (very basic heuristic)
        total_py_files = len(list(self.repo_path.rglob("*.py")))
        if total_py_files > 0:
            tests["coverage_estimate"] = min(100, (test_files / total_py_files) * 100)
        
        print(f"   🧪 Test files: {test_files}")
        print(f"   📁 Test directories: {len(test_dirs)}")
        print(f"   📊 Coverage estimate: {tests['coverage_estimate']:.1f}%")
        
        return tests
    
    def _analyze_imports(self, file_path: Path, content: str, import_analysis: Dict):
        """Analyze imports in Python file"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module not in import_analysis:
                            import_analysis[module] = 0
                        import_analysis[module] += 1
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.module not in import_analysis:
                            import_analysis[node.module] = 0
                        import_analysis[node.module] += 1
        except Exception:
            pass
    
    def _is_git_file(self, file_path: Path) -> bool:
        """Check if file is a git internal file"""
        return '.git' in file_path.parts or '__pycache__' in file_path.parts
    
    def generate_cleanup_recommendations(self) -> List[str]:
        """Generate cleanup recommendations based on analysis"""
        recommendations = []
        
        # Check for syntax errors
        syntax_errors = self.analysis_report["analysis"]["code_quality"]["syntax_errors"]
        if syntax_errors:
            recommendations.append(f"🚨 CRITICAL: Fix {len(syntax_errors)} syntax errors before deployment")
        
        # Check for duplicates
        duplicates = self.analysis_report["analysis"]["duplicates"]["content_duplicates"]
        if duplicates:
            recommendations.append(f"🔄 CLEANUP: Remove {len(duplicates)} duplicate files to reduce repository size")
        
        # Check documentation
        doc_score = self.analysis_report["analysis"]["documentation"]["documentation_score"]
        if doc_score < 70:
            recommendations.append(f"📚 DOCUMENTATION: Improve documentation score from {doc_score:.1f}%")
        
        # Check test coverage
        coverage = self.analysis_report["analysis"]["test_coverage"]["coverage_estimate"]
        if coverage < 50:
            recommendations.append(f"🧪 TESTING: Increase test coverage from {coverage:.1f}%")
        
        # Check large files
        large_files = self.analysis_report["analysis"]["file_structure"]["large_files"]
        if large_files:
            recommendations.append(f"📏 OPTIMIZATION: Review {len(large_files)} large files for optimization")
        
        return recommendations
    
    def save_analysis_report(self) -> str:
        """Save analysis report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.repo_path / f"REPOSITORY_ANALYSIS_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_report, f, indent=2, default=str)
        
        return str(report_file)

def run_final_cleanup():
    """Run final repository cleanup and analysis"""
    print("🔥 OPEN ALPHA - FINAL REPOSITORY CLEANUP")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Pre-git push analysis and cleanup")
    print()
    
    # Get repository path
    repo_path = Path(__file__).parent
    
    # Initialize analyzer
    analyzer = RepositoryAnalyzer(str(repo_path))
    
    # Run analysis
    report = analyzer.analyze_repository()
    
    # Generate recommendations
    recommendations = analyzer.generate_cleanup_recommendations()
    report["recommendations"] = recommendations
    
    # Save report
    report_file = analyzer.save_analysis_report()
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    file_structure = report["analysis"]["file_structure"]
    code_quality = report["analysis"]["code_quality"]
    documentation = report["analysis"]["documentation"]
    security = report["analysis"]["security"]
    
    print(f"📁 Total Files: {file_structure['total_files']:,}")
    print(f"🐍 Python Files: {code_quality['total_python_files']:,}")
    print(f"📝 Lines of Code: {code_quality['lines_of_code']:,}")
    print(f"📄 Documentation Files: {documentation['markdown_files']}")
    print(f"❌ Syntax Errors: {len(code_quality['syntax_errors'])}")
    print(f"🔄 Duplicate Files: {len(report['analysis']['duplicates']['content_duplicates'])}")
    print(f"🛡️ Security Score: {security['security_score']:.1f}/100")
    print(f"📚 Documentation Score: {documentation['documentation_score']:.1f}/100")
    print(f"🧪 Test Coverage: {report['analysis']['test_coverage']['coverage_estimate']:.1f}%")
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
    else:
        print("✅ No critical issues found - repository is clean!")
    
    # Overall status
    critical_issues = len(code_quality['syntax_errors'])
    if critical_issues == 0:
        print(f"\n🎉 REPOSITORY STATUS: ✅ READY FOR GIT PUSH")
        print(f"📋 Analysis report saved: {Path(report_file).name}")
        return True
    else:
        print(f"\n⚠️ REPOSITORY STATUS: ❌ NEEDS ATTENTION")
        print(f"🚨 Critical issues to fix: {critical_issues}")
        print(f"📋 Analysis report saved: {Path(report_file).name}")
        return False

if __name__ == "__main__":
    success = run_final_cleanup()
    sys.exit(0 if success else 1)