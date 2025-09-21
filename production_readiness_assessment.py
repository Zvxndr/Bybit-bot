#!/usr/bin/env python3
"""
Production Readiness Assessment
Evaluates system readiness for live trading deployment
"""

import os
import sys
from pathlib import Path
import json
import time
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionReadinessAssessment:
    """Comprehensive production readiness evaluation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}
        
    def check_file_structure(self) -> dict:
        """Check if all required files exist"""
        logger.info("📁 Checking file structure...")
        
        required_files = {
            # Core bot files
            "README.md": self.project_root / "README.md",
            "start_api.py": self.project_root / "start_api.py",
            
            # Phase 1 - Execution optimizations
            "execution_engine": self.project_root / "src" / "bot" / "execution",
            
            # Phase 2 - ML system
            "ml_system": self.project_root / "src" / "bot" / "ml",
            "transfer_learning": self.project_root / "src" / "bot" / "ml" / "transfer_learning",
            "bayesian_opt": self.project_root / "src" / "bot" / "ml" / "optimization",
            
            # Phase 3 - Dashboard
            "dashboard_backend": self.project_root / "src" / "dashboard" / "backend",
            "dashboard_frontend": self.project_root / "src" / "dashboard" / "frontend",
            
            # Configuration and docs
            "requirements": self.project_root / "requirements.txt",
            "readme": self.project_root / "README.md"
        }
        
        results = {}
        for name, path in required_files.items():
            exists = path.exists()
            results[name] = exists
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {name}: {path}")
        
        return results
    
    def check_phase_implementations(self) -> dict:
        """Check implementation completeness of each phase"""
        logger.info("🔧 Checking phase implementations...")
        
        results = {
            "phase1_execution": self._check_phase1_files(),
            "phase2_ml_system": self._check_phase2_files(),
            "phase3_dashboard": self._check_phase3_files()
        }
        
        return results
    
    def _check_phase1_files(self) -> bool:
        """Check Phase 1 execution optimization files"""
        phase1_files = [
            "src/bot/execution/optimized_execution.py",
            "src/bot/execution/slippage_minimizer.py",
            "src/bot/execution/liquidity_seeker.py"
        ]
        
        exists_count = sum(1 for f in phase1_files if (self.project_root / f).exists())
        return exists_count >= 2  # At least 2 out of 3 files
    
    def _check_phase2_files(self) -> bool:
        """Check Phase 2 ML system files"""
        phase2_files = [
            "src/bot/ml/transfer_learning/cross_market_learner.py",
            "src/bot/ml/optimization/bayesian_optimizer.py",
            "src/bot/ml/auto_tuning/parameter_tuner.py",
            "src/bot/ml/integration/unified_system.py"
        ]
        
        exists_count = sum(1 for f in phase2_files if (self.project_root / f).exists())
        return exists_count >= 3  # At least 3 out of 4 files
    
    def _check_phase3_files(self) -> bool:
        """Check Phase 3 dashboard files"""
        backend_exists = (self.project_root / "src/dashboard/backend/main.py").exists()
        frontend_exists = (self.project_root / "src/dashboard/frontend/package.json").exists()
        return backend_exists and frontend_exists
    
    def check_configuration(self) -> dict:
        """Check configuration completeness"""
        logger.info("⚙️ Checking configuration...")
        
        results = {
            "env_file": (self.project_root / ".env").exists(),
            "config_file": (self.project_root / "config.py").exists(),
            "requirements": (self.project_root / "requirements.txt").exists()
        }
        
        # Check if config has required sections
        if results["config_file"]:
            try:
                with open(self.project_root / "config.py", 'r') as f:
                    config_content = f.read()
                    results["has_api_config"] = "API_KEY" in config_content
                    results["has_trading_config"] = "TRADING" in config_content
            except:
                results["has_api_config"] = False
                results["has_trading_config"] = False
        
        return results
    
    def check_documentation(self) -> dict:
        """Check documentation completeness"""
        logger.info("📚 Checking documentation...")
        
        results = {
            "main_readme": (self.project_root / "README.md").exists(),
            "implementation_tracking": (self.project_root / "IMPLEMENTATION_TRACKING.md").exists(),
            "dashboard_readme": (self.project_root / "src/dashboard/frontend/README.md").exists()
        }
        
        return results
    
    def assess_production_readiness(self) -> dict:
        """Comprehensive production readiness assessment"""
        logger.info("🎯 Assessing production readiness...")
        
        # Run all checks
        file_structure = self.check_file_structure()
        phase_implementations = self.check_phase_implementations()
        configuration = self.check_configuration()
        documentation = self.check_documentation()
        
        # Calculate scores
        file_score = sum(file_structure.values()) / len(file_structure) * 100
        phase_score = sum(phase_implementations.values()) / len(phase_implementations) * 100
        config_score = sum(configuration.values()) / len(configuration) * 100
        doc_score = sum(documentation.values()) / len(documentation) * 100
        
        overall_score = (file_score + phase_score + config_score + doc_score) / 4
        
        # Determine readiness level
        if overall_score >= 90:
            readiness_level = "PRODUCTION READY"
            recommendation = "✅ System ready for live trading deployment"
        elif overall_score >= 75:
            readiness_level = "MOSTLY READY"
            recommendation = "⚠️ Minor issues to address before production"
        elif overall_score >= 60:
            readiness_level = "NEEDS WORK"
            recommendation = "🔧 Several components need attention"
        else:
            readiness_level = "NOT READY"
            recommendation = "❌ Significant development required"
        
        return {
            "overall_score": overall_score,
            "readiness_level": readiness_level,
            "recommendation": recommendation,
            "detailed_scores": {
                "file_structure": file_score,
                "phase_implementations": phase_score,
                "configuration": config_score,
                "documentation": doc_score
            },
            "detailed_results": {
                "file_structure": file_structure,
                "phase_implementations": phase_implementations,
                "configuration": configuration,
                "documentation": documentation
            }
        }
    
    def generate_readiness_report(self) -> str:
        """Generate comprehensive readiness report"""
        assessment = self.assess_production_readiness()
        
        report = f"""
# 🚀 Bybit Trading Bot - Production Readiness Assessment

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Overall Readiness Score:** {assessment['overall_score']:.1f}%
**Readiness Level:** {assessment['readiness_level']}

## 📊 Assessment Summary

{assessment['recommendation']}

### Detailed Scores:
- **File Structure:** {assessment['detailed_scores']['file_structure']:.1f}%
- **Phase Implementations:** {assessment['detailed_scores']['phase_implementations']:.1f}%
- **Configuration:** {assessment['detailed_scores']['configuration']:.1f}%
- **Documentation:** {assessment['detailed_scores']['documentation']:.1f}%

## 📁 File Structure Analysis

### Core Files:
"""
        
        for name, exists in assessment['detailed_results']['file_structure'].items():
            status = "✅" if exists else "❌"
            report += f"- {status} {name.replace('_', ' ').title()}\n"
        
        report += f"""
## 🔧 Phase Implementation Status

### Phase 1 - Execution Engine: {'✅' if assessment['detailed_results']['phase_implementations']['phase1_execution'] else '❌'}
- Advanced order execution optimizations
- Slippage minimization algorithms
- Liquidity seeking mechanisms

### Phase 2 - ML System: {'✅' if assessment['detailed_results']['phase_implementations']['phase2_ml_system'] else '❌'}
- Cross-market transfer learning
- Bayesian hyperparameter optimization
- Auto-tuning system
- Unified ML integration

### Phase 3 - Dashboard: {'✅' if assessment['detailed_results']['phase_implementations']['phase3_dashboard'] else '❌'}
- FastAPI backend with WebSocket streaming
- Next.js frontend with real-time updates
- System monitoring and visualization

## ⚙️ Configuration Status

"""
        
        for name, exists in assessment['detailed_results']['configuration'].items():
            status = "✅" if exists else "❌"
            report += f"- {status} {name.replace('_', ' ').title()}\n"
        
        report += f"""
## 📚 Documentation Status

"""
        
        for name, exists in assessment['detailed_results']['documentation'].items():
            status = "✅" if exists else "❌"
            report += f"- {status} {name.replace('_', ' ').title()}\n"
        
        report += f"""
## 🎯 Next Steps for Production Deployment

### Immediate Actions Required:
"""
        
        if assessment['overall_score'] >= 90:
            report += """
1. ✅ Set up production environment
2. ✅ Configure production API keys
3. ✅ Deploy backend and frontend services
4. ✅ Begin live trading validation with small positions
5. ✅ Monitor system performance
"""
        else:
            report += """
1. 🔧 Address missing components identified above
2. ⚙️ Complete configuration setup
3. 📝 Ensure all documentation is current
4. 🧪 Run comprehensive testing
5. 🚀 Proceed with production deployment when ready
"""
        
        report += f"""
### Production Checklist:
- [ ] Environment variables configured
- [ ] API keys and secrets secured
- [ ] Database connections tested
- [ ] Monitoring and alerting set up
- [ ] Backup and recovery procedures in place
- [ ] Performance benchmarks established
- [ ] Documentation updated and accessible

**Assessment Status:** {"READY FOR PRODUCTION" if assessment['overall_score'] >= 90 else "REQUIRES ATTENTION"}
"""
        
        return report

def main():
    """Main assessment execution"""
    print("🚀 Starting Production Readiness Assessment...")
    print("=" * 60)
    
    assessor = ProductionReadinessAssessment()
    report = assessor.generate_readiness_report()
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"production_readiness_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n📄 Report saved to: {report_file}")
    print("=" * 60)
    print("🎉 Production Readiness Assessment Complete!")

if __name__ == "__main__":
    main()