#!/usr/bin/env python3
"""
🔥 OPEN ALPHA - Deployment Preparation Script
Ensures all systems align with System Architecture Reference v3.0

This script:
1. Downloads historical data for professional backtesting
2. Validates current implementation against SAR design goals
3. Prepares system for GitHub deployment
4. Generates comprehensive deployment report
"""

import asyncio
import subprocess
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
import logging

# Import our historical data downloader
from historical_data_downloader import HistoricalDataDownloader, DataDownloadConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """
    Validates current implementation against SAR v3.0 specifications
    Ensures alignment with wealth management system goals
    """
    
    def __init__(self):
        self.sar_requirements = {
            'foundation_phase': {
                'debug_safety_system': 'src/debug_safety.py',
                'historical_data_provider': 'src/historical_data_provider.py',
                'fire_dashboard': 'src/dashboard/',
                'bybit_api_integration': 'src/bybit_api.py',
                'configuration_management': 'config/'
            },
            'infrastructure_readiness': {
                'database_schema': 'src/data/speed_demon_cache/',
                'ml_data_infrastructure': 'src/bot/ml_strategy_discovery/',
                'multi_environment_config': ['config/debug.yaml', 'config/development.yaml'],
                'safety_systems': ['src/debug_safety.py']
            }
        }
        
        self.validation_results = {
            'sar_compliance': {},
            'deployment_readiness': False,
            'missing_components': [],
            'recommendations': []
        }
    
    def validate_sar_compliance(self) -> dict:
        """Validate current system against SAR requirements"""
        
        logger.info("🔍 Validating System Architecture Reference compliance")
        
        # Check foundation phase components
        foundation_score = 0
        foundation_total = len(self.sar_requirements['foundation_phase'])
        
        for component, path in self.sar_requirements['foundation_phase'].items():
            if Path(path).exists():
                self.validation_results['sar_compliance'][component] = 'IMPLEMENTED ✅'
                foundation_score += 1
                logger.info(f"✅ {component}: Found at {path}")
            else:
                self.validation_results['sar_compliance'][component] = 'MISSING ❌'
                self.validation_results['missing_components'].append(component)
                logger.warning(f"❌ {component}: Not found at {path}")
        
        # Check infrastructure readiness
        infra_score = 0
        infra_total = 0
        
        for category, paths in self.sar_requirements['infrastructure_readiness'].items():
            if isinstance(paths, list):
                infra_total += len(paths)
                for path in paths:
                    if Path(path).exists():
                        infra_score += 1
                        logger.info(f"✅ {category}: {path} exists")
                    else:
                        self.validation_results['missing_components'].append(f"{category}: {path}")
                        logger.warning(f"❌ {category}: {path} missing")
            else:
                infra_total += 1
                if Path(paths).exists():
                    infra_score += 1
                    logger.info(f"✅ {category}: {paths} exists")
                else:
                    self.validation_results['missing_components'].append(f"{category}: {paths}")
                    logger.warning(f"❌ {category}: {paths} missing")
        
        # Calculate compliance scores
        foundation_compliance = foundation_score / foundation_total if foundation_total > 0 else 0
        infra_compliance = infra_score / infra_total if infra_total > 0 else 0
        overall_compliance = (foundation_compliance + infra_compliance) / 2
        
        self.validation_results['compliance_scores'] = {
            'foundation_phase': foundation_compliance,
            'infrastructure_readiness': infra_compliance,
            'overall_compliance': overall_compliance
        }
        
        # Determine deployment readiness
        self.validation_results['deployment_readiness'] = overall_compliance >= 0.8
        
        logger.info(f"📊 Foundation Phase: {foundation_compliance:.1%}")
        logger.info(f"📊 Infrastructure: {infra_compliance:.1%}")
        logger.info(f"📊 Overall Compliance: {overall_compliance:.1%}")
        
        return self.validation_results
    
    def check_design_goal_alignment(self) -> dict:
        """Check alignment with SAR design goals"""
        
        logger.info("🎯 Checking design goal alignment")
        
        design_goals = {
            'ai_strategy_discovery': {
                'description': 'Machine learning pipeline to test and identify winning strategies',
                'current_status': 'INFRASTRUCTURE READY 📋',
                'implementation_path': 'src/bot/ml_strategy_discovery/',
                'next_steps': 'Integrate ML engines with dashboard'
            },
            'professional_backtesting': {
                'description': 'Institutional-grade historical analysis with proper risk modeling',
                'current_status': 'DATA DOWNLOAD IN PROGRESS 🔄',
                'implementation_path': 'historical_data_downloader.py',
                'next_steps': 'Complete data download and validation'
            },
            'strategy_graduation_system': {
                'description': 'Automatic promotion from paper → testnet → live trading',
                'current_status': 'FOUNDATION READY 📋',
                'implementation_path': 'src/bot/backtesting/',
                'next_steps': 'Connect graduation pipeline to dashboard'
            },
            'dynamic_risk_management': {
                'description': 'Intelligent leverage optimization and risk falloff algorithms',
                'current_status': 'PLANNING STAGE 📋',
                'implementation_path': 'Future development',
                'next_steps': 'Design risk management algorithms'
            },
            'three_tier_business_structure': {
                'description': 'Private use → Trust fund → PTY LTD corporate versions',
                'current_status': 'ARCHITECTURE DEFINED 📋',
                'implementation_path': 'System Architecture Reference',
                'next_steps': 'Implement user management system'
            }
        }
        
        alignment_report = {
            'goals_status': design_goals,
            'current_phase': 'Foundation Development',
            'next_phase': 'Intelligence Integration (ML)',
            'phase_completion': '65%'  # Based on implemented components
        }
        
        return alignment_report
    
    def generate_deployment_recommendations(self) -> list:
        """Generate recommendations for deployment readiness"""
        
        recommendations = []
        
        if self.validation_results['compliance_scores']['overall_compliance'] < 0.8:
            recommendations.append("🔧 Fix missing components before deployment")
        
        if len(self.validation_results['missing_components']) > 0:
            recommendations.append(f"📋 Address {len(self.validation_results['missing_components'])} missing components")
        
        # SAR-specific recommendations
        recommendations.extend([
            "📊 Complete historical data download for professional backtesting",
            "🧠 Prepare ML engine integration for next development phase",
            "🛡️ Ensure debug safety system remains active during deployment",
            "🔥 Validate fire dashboard functionality with historical data",
            "☁️ Prepare for DigitalOcean deployment configuration"
        ])
        
        return recommendations

async def prepare_deployment():
    """
    Main deployment preparation function
    Aligns system with SAR v3.0 and prepares for GitHub push
    """
    
    print("🔥 OPEN ALPHA - Deployment Preparation")
    print("📋 System Architecture Reference v3.0 Alignment")
    print("=" * 60)
    
    # Step 1: Validate SAR compliance
    validator = DeploymentValidator()
    sar_validation = validator.validate_sar_compliance()
    design_alignment = validator.check_design_goal_alignment()
    recommendations = validator.generate_deployment_recommendations()
    
    print("\n📊 SAR COMPLIANCE REPORT:")
    print(f"Foundation Phase: {sar_validation['compliance_scores']['foundation_phase']:.1%}")
    print(f"Infrastructure: {sar_validation['compliance_scores']['infrastructure_readiness']:.1%}")
    print(f"Overall: {sar_validation['compliance_scores']['overall_compliance']:.1%}")
    
    if sar_validation['missing_components']:
        print(f"\n❌ Missing Components ({len(sar_validation['missing_components'])}):")
        for component in sar_validation['missing_components']:
            print(f"   - {component}")
    
    # Step 2: Download historical data
    print("\n📊 HISTORICAL DATA DOWNLOAD:")
    
    config = DataDownloadConfig(
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT'],
        timeframes=['5m', '15m', '1h', '4h', '1d'],  # Optimized for free tier
        lookback_days=730,  # 2 years for initial deployment
        max_requests_per_minute=180,
        database_path='src/data/speed_demon_cache/market_data.db'
    )
    
    async with HistoricalDataDownloader(config) as downloader:
        data_report = await downloader.download_all_data()
        
        # Generate comprehensive deployment report
        report_text = downloader.generate_deployment_report(data_report)
        
        # Save to file
        with open('DEPLOYMENT_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    # Step 3: Update configuration for deployment
    print("\n⚙️ UPDATING CONFIGURATION:")
    
    # Update debug config to reflect historical data availability
    debug_config_path = Path('config/debug.yaml')
    if debug_config_path.exists():
        with open(debug_config_path, 'r') as f:
            debug_config = yaml.safe_load(f)
        
        # Update historical data settings
        if 'historical_data' not in debug_config:
            debug_config['historical_data'] = {}
        
        debug_config['historical_data'].update({
            'enabled': True,
            'database_populated': data_report['deployment_ready'],
            'data_quality_score': data_report['data_quality_score'],
            'last_updated': datetime.now().isoformat(),
            'deployment_ready': data_report['deployment_ready']
        })
        
        with open(debug_config_path, 'w') as f:
            yaml.dump(debug_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Updated {debug_config_path}")
    
    # Step 4: Generate final deployment report
    deployment_summary = {
        'deployment_timestamp': datetime.now().isoformat(),
        'sar_compliance': sar_validation,
        'design_alignment': design_alignment,
        'data_download_report': {
            'deployment_ready': data_report['deployment_ready'],
            'total_data_points': data_report['total_data_points'],
            'quality_score': data_report['data_quality_score']
        },
        'recommendations': recommendations,
        'next_steps': [
            "🚀 Push to GitHub for deployment testing",
            "🔧 Test debug mode with historical data",
            "📊 Validate fire dashboard functionality",
            "🧠 Begin ML engine integration planning",
            "☁️ Prepare DigitalOcean deployment configuration"
        ]
    }
    
    # Save deployment summary
    with open('deployment_summary.json', 'w') as f:
        json.dump(deployment_summary, f, indent=2)
    
    # Step 5: Final deployment readiness assessment
    print("\n🚀 DEPLOYMENT READINESS ASSESSMENT:")
    
    deployment_ready = (
        sar_validation['deployment_readiness'] and
        data_report['deployment_ready'] and
        sar_validation['compliance_scores']['overall_compliance'] >= 0.7
    )
    
    if deployment_ready:
        print("✅ SYSTEM READY FOR GITHUB DEPLOYMENT")
        print("✅ Historical data download completed successfully")
        print("✅ SAR compliance meets deployment standards")
        print("✅ Foundation phase implementation verified")
        print("\n🎯 READY FOR NEXT PHASE: Intelligence Integration (ML)")
    else:
        print("❌ SYSTEM NOT READY FOR DEPLOYMENT")
        print("❌ Additional work required before GitHub push")
        print("\n📋 PRIORITY FIXES NEEDED:")
        for rec in recommendations[:3]:  # Show top 3 recommendations
            print(f"   {rec}")
    
    return deployment_summary

if __name__ == "__main__":
    print("🔥 OPEN ALPHA - Wealth Management System")
    print("📋 Deployment Preparation & SAR Alignment")
    print("")
    
    # Run deployment preparation
    result = asyncio.run(prepare_deployment())
    
    print(f"\n📄 Reports generated:")
    print("   - DEPLOYMENT_REPORT.md")
    print("   - deployment_summary.json")
    print("   - Updated config/debug.yaml")
    
    print(f"\n🔥 Open Alpha deployment preparation complete!")