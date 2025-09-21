#!/usr/bin/env python3
"""
Risk Management Consolidation Script

This script analyzes the three separate risk management systems and creates
a unified, consolidated system preserving all unique features.

Current Systems:
- src/bot/risk_management/ (Australian-focused, tax-aware, 7 files)
- src/bot/risk/ (Advanced algorithms, portfolio analysis, 5 files)  
- src/bot/dynamic_risk/ (Volatility/correlation analysis, 4 files)

Consolidation Strategy:
1. Use risk_management/ as the base (most complete)
2. Integrate unique algorithms from risk/ (Kelly, Risk Parity, etc.)
3. Add dynamic features from dynamic_risk/ (GARCH, correlation analysis)
4. Create unified interface maintaining backward compatibility
"""

import os
import shutil
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class RiskSystemAnalysis:
    """Analysis results for a risk management system"""
    path: str
    classes: List[str]
    unique_features: List[str]
    dependencies: List[str]
    lines_of_code: int
    
class RiskConsolidationManager:
    """Manages the consolidation of multiple risk management systems"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.systems = {
            'risk_management': self.repo_root / 'src/bot/risk_management',
            'risk': self.repo_root / 'src/bot/risk', 
            'dynamic_risk': self.repo_root / 'src/bot/dynamic_risk'
        }
        
        # Archive directory for backup
        self.archive_dir = self.repo_root / 'archive/risk_systems_backup'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_systems(self) -> Dict[str, RiskSystemAnalysis]:
        """Analyze all risk management systems"""
        analyses = {}
        
        for name, path in self.systems.items():
            if path.exists():
                analysis = self._analyze_single_system(name, path)
                analyses[name] = analysis
                print(f"\n📊 {name.upper()} Analysis:")
                print(f"  📁 Path: {path}")
                print(f"  📝 Files: {len(list(path.glob('*.py')))}")
                print(f"  📏 Lines: {analysis.lines_of_code}")
                print(f"  🏗️ Classes: {len(analysis.classes)}")
                print(f"  ⭐ Unique Features: {len(analysis.unique_features)}")
                
        return analyses
    
    def _analyze_single_system(self, name: str, path: Path) -> RiskSystemAnalysis:
        """Analyze a single risk management system"""
        classes = []
        unique_features = []
        dependencies = []
        total_lines = 0
        
        for py_file in path.glob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_lines += len(content.splitlines())
                    
                # Parse AST to find classes and functions
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, ast.FunctionDef):
                            if node.name.startswith('_calculate') or node.name.startswith('_analyze'):
                                unique_features.append(f"{py_file.stem}::{node.name}")
                except SyntaxError:
                    pass
                    
                # Find imports to understand dependencies
                imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', content)
                for imp in imports:
                    dep = imp[0] or imp[1]
                    if dep and not dep.startswith('.'):
                        dependencies.append(dep)
                        
            except Exception as e:
                print(f"  ⚠️ Error analyzing {py_file}: {e}")
                
        return RiskSystemAnalysis(
            path=str(path),
            classes=classes,
            unique_features=unique_features,
            dependencies=list(set(dependencies)),
            lines_of_code=total_lines
        )
    
    def identify_consolidation_plan(self, analyses: Dict[str, RiskSystemAnalysis]) -> Dict[str, str]:
        """Create consolidation plan mapping features to target modules"""
        
        plan = {
            # Core risk management (from risk_management/)
            'core_risk_manager': 'risk_management/australian_risk_manager.py',
            'portfolio_controller': 'risk_management/portfolio_risk_controller.py',
            
            # Advanced algorithms (from risk/)
            'position_sizing': 'risk/position_sizing.py',  
            'portfolio_analysis': 'risk/portfolio_analysis.py',
            'real_time_monitoring': 'risk/real_time_monitoring.py',
            'dynamic_adjustment': 'risk/dynamic_adjustment.py',
            
            # Dynamic features (from dynamic_risk/)
            'volatility_monitoring': 'dynamic_risk/volatility_monitor.py',
            'correlation_analysis': 'dynamic_risk/correlation_analysis.py', 
            'dynamic_hedging': 'dynamic_risk/dynamic_hedging.py'
        }
        
        print(f"\n🎯 Consolidation Plan:")
        for feature, source in plan.items():
            print(f"  📦 {feature} ← {source}")
            
        return plan
    
    def create_unified_structure(self) -> None:
        """Create the new unified risk management structure"""
        
        # New unified structure
        unified_path = self.repo_root / 'src/bot/risk'
        
        # Create new structure
        structure = {
            'core': 'Core risk management and position sizing',
            'portfolio': 'Portfolio analysis and monitoring', 
            'dynamic': 'Dynamic risk adjustment and hedging',
            'monitoring': 'Real-time monitoring and alerts',
            'analysis': 'Risk analysis and metrics',
            'legacy': 'Archived components for reference'
        }
        
        print(f"\n🏗️ Creating Unified Structure at {unified_path}:")
        
        for subdir, description in structure.items():
            subdir_path = unified_path / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py with description
            init_file = subdir_path / '__init__.py'
            with open(init_file, 'w') as f:
                f.write(f'"""{description}"""\n')
            
            print(f"  📁 {subdir}/ - {description}")
        
        return unified_path
    
    def backup_existing_systems(self) -> None:
        """Backup existing systems before consolidation"""
        print(f"\n💾 Creating Backups:")
        
        for name, path in self.systems.items():
            if path.exists():
                backup_path = self.archive_dir / name
                
                # Remove existing backup
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                
                # Create new backup
                shutil.copytree(path, backup_path)
                print(f"  ✅ {name} → {backup_path}")
    
    def generate_migration_script(self, unified_path: Path) -> str:
        """Generate migration script for import updates"""
        
        migration_script = f'''
"""
Auto-generated migration script for risk management consolidation
Run this script to update all imports across the codebase
"""

import os
import re
from pathlib import Path

# Import mapping for migration
IMPORT_MIGRATIONS = {{
    # Old risk_management imports
    'from bot.risk_management.australian_risk_manager import': 'from bot.risk.core.australian_risk import',
    'from bot.risk_management.portfolio_risk_controller import': 'from bot.risk.monitoring.portfolio_monitor import',
    'from bot.risk_management import': 'from bot.risk.core import',
    
    # Old risk imports  
    'from bot.risk.position_sizing import': 'from bot.risk.core.position_sizing import',
    'from bot.risk.portfolio_analysis import': 'from bot.risk.portfolio.analysis import',
    'from bot.risk.real_time_monitoring import': 'from bot.risk.monitoring.real_time import',
    
    # Old dynamic_risk imports
    'from bot.dynamic_risk.volatility_monitor import': 'from bot.risk.dynamic.volatility import',
    'from bot.dynamic_risk.correlation_analysis import': 'from bot.risk.dynamic.correlation import',
    'from bot.dynamic_risk.dynamic_hedging import': 'from bot.risk.dynamic.hedging import',
}}

def update_imports_in_file(file_path: Path):
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply import migrations
        for old_import, new_import in IMPORT_MIGRATIONS.items():
            content = content.replace(old_import, new_import)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated: {{file_path}}")
            return True
        return False
        
    except Exception as e:
        print(f"  ❌ Error updating {{file_path}}: {{e}}")
        return False

def migrate_all_imports(repo_root: str = "{self.repo_root}"):
    """Migrate all imports across the codebase"""
    repo_path = Path(repo_root)
    updated_files = 0
    
    print("🔄 Migrating imports across codebase...")
    
    # Find all Python files
    for py_file in repo_path.rglob("*.py"):
        # Skip archived/backup files
        if 'archive' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        if update_imports_in_file(py_file):
            updated_files += 1
    
    print(f"✅ Migration complete! Updated {{updated_files}} files.")

if __name__ == "__main__":
    migrate_all_imports()
'''
        
        script_path = unified_path / 'migrate_imports.py'
        with open(script_path, 'w') as f:
            f.write(migration_script)
            
        print(f"  📝 Migration script: {script_path}")
        return str(script_path)
    
    def execute_consolidation(self) -> bool:
        """Execute the full consolidation process"""
        try:
            print("🚀 Starting Risk Management Consolidation...")
            print("=" * 60)
            
            # Step 1: Analyze existing systems
            analyses = self.analyze_systems()
            
            # Step 2: Create consolidation plan  
            plan = self.identify_consolidation_plan(analyses)
            
            # Step 3: Backup existing systems
            self.backup_existing_systems()
            
            # Step 4: Create unified structure
            unified_path = self.create_unified_structure()
            
            # Step 5: Generate migration script
            migration_script = self.generate_migration_script(unified_path)
            
            print(f"\n✅ Consolidation Setup Complete!")
            print(f"📁 Unified Structure: {unified_path}")
            print(f"💾 Backups: {self.archive_dir}")
            print(f"🔄 Migration Script: {migration_script}")
            
            print(f"\n📋 Next Steps:")
            print(f"  1. Review the unified structure at {unified_path}")
            print(f"  2. Move unique features from each system to appropriate modules")
            print(f"  3. Run the migration script to update imports")
            print(f"  4. Test the consolidated system")
            print(f"  5. Remove redundant directories")
            
            return True
            
        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            return False

def main():
    """Main consolidation entry point"""
    repo_root = os.getcwd()
    
    print("🔍 Risk Management Consolidation Analysis")
    print("=" * 50)
    
    consolidator = RiskConsolidationManager(repo_root)
    success = consolidator.execute_consolidation()
    
    if success:
        print("\n🎉 Risk Management Consolidation Ready!")
        print("Review the output above and proceed with manual consolidation.")
    else:
        print("\n💥 Consolidation failed. Check the errors above.")

if __name__ == "__main__":
    main()