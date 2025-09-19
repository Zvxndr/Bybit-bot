"""
Comprehensive Tax System Example

This example demonstrates the complete Phase 8 tax and reporting system including:
- Advanced tax calculation with wash sale detection
- Professional tax reporting and compliance
- Sophisticated tax optimization and loss harvesting
- Multi-jurisdiction support
- Real-time tax impact analysis
- Comprehensive audit trails
"""

import sys
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import tax system components
from src.bot.tax import (
    TaxSystem, create_tax_system, create_demo_tax_system,
    AccountingMethod, TransactionType, Transaction,
    ReportType, ReportFormat, ReportConfiguration, JurisdictionCompliance,
    OptimizationStrategy, OptimizationObjective, RiskTolerance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tax_system_example.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ComprehensiveTaxExample:
    """Comprehensive example of the tax system capabilities."""
    
    def __init__(self):
        self.tax_system = None
        self.sample_transactions = []
        
    def run_complete_example(self):
        """Run the complete tax system example."""
        print("🚀 Starting Comprehensive Tax System Example")
        print("=" * 60)
        
        try:
            # 1. Initialize tax system
            self.initialize_tax_system()
            
            # 2. Load sample trading data
            self.create_sample_trading_data()
            
            # 3. Process transactions and analyze tax impact
            self.process_transactions_with_analysis()
            
            # 4. Generate comprehensive tax reports
            self.generate_tax_reports()
            
            # 5. Run tax optimization analysis
            asyncio.run(self.run_tax_optimization())
            
            # 6. Demonstrate real-time tax impact analysis
            self.demonstrate_real_time_analysis()
            
            # 7. Show advanced portfolio analysis
            self.show_advanced_portfolio_analysis()
            
            # 8. Export tax data for external use
            self.export_tax_data()
            
            print("\n✅ Comprehensive tax system example completed successfully!")
            
        except Exception as e:
            logger.error(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize_tax_system(self):
        """Initialize the tax system with professional configuration."""
        print("\n📊 Initializing Tax System")
        print("-" * 30)
        
        # Create tax system with FIFO accounting (most common)
        self.tax_system = create_tax_system(
            jurisdiction="US",
            accounting_method="FIFO",
            db_path="comprehensive_tax_example.db"
        )
        
        status = self.tax_system.get_system_status()
        
        print(f"✅ Tax System Initialized")
        print(f"   📍 Jurisdiction: {status['configuration']['jurisdiction']}")
        print(f"   📋 Accounting Method: {status['configuration']['accounting_method']}")
        print(f"   🔄 Wash Sale Detection: {status['configuration']['wash_sale_enabled']}")
        
        logger.info("Tax system initialized successfully")
    
    def create_sample_trading_data(self):
        """Create realistic sample trading data for demonstration."""
        print("\n📈 Creating Sample Trading Data")
        print("-" * 30)
        
        # Realistic crypto trading scenario over 2024
        base_date = datetime(2024, 1, 1)
        
        self.sample_transactions = [
            # Initial BTC purchases
            Transaction(
                id="btc_buy_1",
                timestamp=base_date + timedelta(days=10),
                transaction_type=TransactionType.BUY,
                asset="BTC",
                quantity=Decimal('2.0'),
                price_per_unit=Decimal('42000'),
                total_value=Decimal('84000'),
                fees=Decimal('100'),
                exchange="Coinbase"
            ),
            
            # ETH purchase
            Transaction(
                id="eth_buy_1",
                timestamp=base_date + timedelta(days=15),
                transaction_type=TransactionType.BUY,
                asset="ETH",
                quantity=Decimal('10.0'),
                price_per_unit=Decimal('2500'),
                total_value=Decimal('25000'),
                fees=Decimal('50'),
                exchange="Coinbase"
            ),
            
            # BTC sale at loss (potential wash sale scenario)
            Transaction(
                id="btc_sell_1",
                timestamp=base_date + timedelta(days=45),
                transaction_type=TransactionType.SELL,
                asset="BTC",
                quantity=Decimal('0.5'),
                price_per_unit=Decimal('38000'),
                total_value=Decimal('19000'),
                fees=Decimal('25'),
                exchange="Coinbase"
            ),
            
            # BTC repurchase (wash sale trigger)
            Transaction(
                id="btc_buy_2",
                timestamp=base_date + timedelta(days=60),
                transaction_type=TransactionType.BUY,
                asset="BTC",
                quantity=Decimal('0.3'),
                price_per_unit=Decimal('40000'),
                total_value=Decimal('12000'),
                fees=Decimal('30'),
                exchange="Binance"
            ),
            
            # ETH sale at gain (long-term after 1 year)
            Transaction(
                id="eth_sell_1",
                timestamp=base_date + timedelta(days=380),
                transaction_type=TransactionType.SELL,
                asset="ETH",
                quantity=Decimal('3.0'),
                price_per_unit=Decimal('3200'),
                total_value=Decimal('9600'),
                fees=Decimal('20'),
                exchange="Coinbase"
            ),
            
            # Staking rewards
            Transaction(
                id="eth_reward_1",
                timestamp=base_date + timedelta(days=200),
                transaction_type=TransactionType.REWARD,
                asset="ETH",
                quantity=Decimal('0.15'),
                price_per_unit=Decimal('2800'),
                total_value=Decimal('420'),
                fees=Decimal('0'),
                notes="Ethereum staking rewards"
            ),
            
            # DeFi yield
            Transaction(
                id="defi_yield_1",
                timestamp=base_date + timedelta(days=250),
                transaction_type=TransactionType.DeFi_YIELD,
                asset="USDC",
                quantity=Decimal('150'),
                price_per_unit=Decimal('1.0'),
                total_value=Decimal('150'),
                fees=Decimal('5'),
                notes="Compound protocol yield"
            ),
            
            # Additional trading for tax loss harvesting demo
            Transaction(
                id="ada_buy_1",
                timestamp=base_date + timedelta(days=100),
                transaction_type=TransactionType.BUY,
                asset="ADA",
                quantity=Decimal('5000'),
                price_per_unit=Decimal('0.35'),
                total_value=Decimal('1750'),
                fees=Decimal('10'),
                exchange="Kraken"
            ),
            
            # ADA loss (good for harvesting)
            Transaction(
                id="ada_sell_1",
                timestamp=base_date + timedelta(days=300),
                transaction_type=TransactionType.SELL,
                asset="ADA",
                quantity=Decimal('5000'),
                price_per_unit=Decimal('0.28'),
                total_value=Decimal('1400'),
                fees=Decimal('8'),
                exchange="Kraken"
            )
        ]
        
        print(f"✅ Created {len(self.sample_transactions)} sample transactions")
        print(f"   📅 Date Range: {self.sample_transactions[0].timestamp.strftime('%Y-%m-%d')} to {self.sample_transactions[-1].timestamp.strftime('%Y-%m-%d')}")
        print(f"   💰 Assets: {list(set(tx.asset for tx in self.sample_transactions))}")
        print(f"   🏢 Exchanges: {list(set(tx.exchange for tx in self.sample_transactions if tx.exchange))}")
    
    def process_transactions_with_analysis(self):
        """Process transactions and show detailed tax impact analysis."""
        print("\n⚖️  Processing Transactions with Tax Analysis")
        print("-" * 45)
        
        total_tax_impact = Decimal('0')
        significant_events = []
        
        for i, transaction in enumerate(self.sample_transactions, 1):
            print(f"\n📝 Processing Transaction {i}/{len(self.sample_transactions)}: {transaction.id}")
            
            # Process transaction and get impact analysis
            impact = self.tax_system.add_transaction(transaction)
            
            if impact.get('tax_impact', Decimal('0')) != 0:
                total_tax_impact += impact['tax_impact']
                significant_events.append({
                    'transaction': transaction,
                    'impact': impact
                })
                
                print(f"   💸 Tax Impact: ${impact['tax_impact']:,.2f}")
                if impact.get('wash_sale_affected'):
                    print(f"   ⚠️  Wash Sale Detected!")
                if impact.get('holding_period'):
                    print(f"   📅 Holding Period: {impact['holding_period']}")
            else:
                print(f"   ✅ No immediate tax impact")
        
        print(f"\n📊 Transaction Processing Summary:")
        print(f"   💰 Total Tax Impact: ${total_tax_impact:,.2f}")
        print(f"   🎯 Significant Tax Events: {len(significant_events)}")
        
        # Show wash sale analysis
        wash_sale_events = [event for event in significant_events 
                           if event['impact'].get('wash_sale_affected')]
        if wash_sale_events:
            print(f"   ⚠️  Wash Sale Events: {len(wash_sale_events)}")
        
        logger.info(f"Processed {len(self.sample_transactions)} transactions with ${total_tax_impact:,.2f} total tax impact")
    
    def generate_tax_reports(self):
        """Generate comprehensive tax reports."""
        print("\n📄 Generating Comprehensive Tax Reports")
        print("-" * 40)
        
        taxpayer_info = {
            'name': 'John Crypto Trader',
            'ssn': '123-45-6789',
            'address': '123 Blockchain Ave, Crypto City, CC 12345'
        }
        
        # Generate standard reports
        print("📋 Generating standard tax reports...")
        
        try:
            reports = self.tax_system.generate_standard_reports(2024, taxpayer_info)
            
            print(f"✅ Generated {len(reports)} reports:")
            for report_name, result in reports.items():
                if 'error' not in result:
                    print(f"   📄 {report_name}: {result.get('filename', 'Generated successfully')}")
                else:
                    print(f"   ❌ {report_name}: {result['error']}")
            
            # Generate specialized reports
            self.generate_specialized_reports(taxpayer_info)
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            print(f"❌ Report generation failed: {e}")
    
    def generate_specialized_reports(self, taxpayer_info: dict):
        """Generate specialized tax reports."""
        print("\n📊 Generating specialized reports...")
        
        specialized_reports = [
            (ReportType.WASH_SALE_REPORT, ReportFormat.PDF),
            (ReportType.TAX_LOSS_HARVEST, ReportFormat.HTML),
            (ReportType.AUDIT_TRAIL, ReportFormat.CSV)
        ]
        
        for report_type, format_type in specialized_reports:
            try:
                config = ReportConfiguration(
                    report_type=report_type,
                    format=format_type,
                    tax_year=2024,
                    jurisdiction=JurisdictionCompliance.US_FEDERAL,
                    taxpayer_info=taxpayer_info
                )
                
                result = self.tax_system.generate_tax_report(config)
                print(f"   ✅ {report_type.value}: {result.get('filename', 'Generated')}")
                
            except Exception as e:
                print(f"   ❌ {report_type.value}: {e}")
    
    async def run_tax_optimization(self):
        """Run comprehensive tax optimization analysis."""
        print("\n🎯 Running Tax Optimization Analysis")
        print("-" * 40)
        
        try:
            # Run different optimization strategies
            strategies = [
                OptimizationStrategy.TAX_LOSS_HARVEST,
                OptimizationStrategy.LONG_TERM_OPTIMIZATION,
                OptimizationStrategy.WASH_SALE_AVOIDANCE
            ]
            
            all_recommendations = []
            
            for strategy in strategies:
                print(f"\n🔄 Running {strategy.value} optimization...")
                
                recommendations = await self.tax_system.run_tax_optimization(strategy)
                all_recommendations.extend(recommendations)
                
                print(f"   📈 Found {len(recommendations)} opportunities")
                
                # Show top recommendations
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec.action} {rec.quantity} {rec.asset}")
                    print(f"      💰 Expected Savings: ${rec.expected_tax_savings:.2f}")
                    print(f"      📊 Confidence: {rec.confidence:.1%}")
                    print(f"      ⚡ Timeline: {rec.execution_timeline}")
            
            # Show optimization summary
            print(f"\n📊 Optimization Summary:")
            print(f"   🎯 Total Opportunities: {len(all_recommendations)}")
            print(f"   💰 Total Potential Savings: ${sum(rec.expected_tax_savings for rec in all_recommendations):,.2f}")
            
            # Demonstrate recommendation execution (dry run)
            if all_recommendations:
                await self.demonstrate_recommendation_execution(all_recommendations[0])
                
        except Exception as e:
            logger.error(f"Tax optimization failed: {e}")
            print(f"❌ Tax optimization failed: {e}")
    
    async def demonstrate_recommendation_execution(self, recommendation):
        """Demonstrate executing a tax optimization recommendation."""
        print(f"\n🚀 Demonstrating Recommendation Execution")
        print("-" * 45)
        
        print(f"📋 Recommendation: {recommendation.action} {recommendation.quantity} {recommendation.asset}")
        print(f"💰 Expected Savings: ${recommendation.expected_tax_savings:.2f}")
        print(f"📊 Risk Score: {recommendation.risk_score:.2f}")
        
        # Execute in dry-run mode
        result = await self.tax_system.execute_optimization_recommendation(
            recommendation, 
            dry_run=True
        )
        
        print(f"✅ Dry Run Execution Result:")
        print(f"   🎯 Status: {'Success' if result.get('executed') else 'Failed'}")
        print(f"   📝 Details: {result.get('details', 'N/A')}")
        if result.get('warnings'):
            print(f"   ⚠️  Warnings: {result['warnings']}")
    
    def demonstrate_real_time_analysis(self):
        """Demonstrate real-time tax impact analysis."""
        print("\n⚡ Real-Time Tax Impact Analysis")
        print("-" * 35)
        
        # Create hypothetical transactions for analysis
        hypothetical_transactions = [
            Transaction(
                id="hypothetical_btc_sell",
                timestamp=datetime.now(),
                transaction_type=TransactionType.SELL,
                asset="BTC",
                quantity=Decimal('0.5'),
                price_per_unit=Decimal('65000'),
                total_value=Decimal('32500'),
                fees=Decimal('50')
            ),
            Transaction(
                id="hypothetical_eth_buy",
                timestamp=datetime.now(),
                transaction_type=TransactionType.BUY,
                asset="ETH",
                quantity=Decimal('5.0'),
                price_per_unit=Decimal('3500'),
                total_value=Decimal('17500'),
                fees=Decimal('30')
            )
        ]
        
        print("🔮 Analyzing hypothetical transactions:")
        
        for tx in hypothetical_transactions:
            impact = self.tax_system.get_real_time_tax_impact(tx)
            
            print(f"\n📊 Transaction: {tx.transaction_type.value} {tx.quantity} {tx.asset}")
            print(f"   💸 Tax Impact: ${impact.get('tax_impact', 0):,.2f}")
            print(f"   📈 Impact Type: {impact.get('impact_type', 'unknown')}")
            
            if impact.get('wash_sale_risk'):
                risk_level = impact['wash_sale_risk'].get('risk_level', 'UNKNOWN')
                print(f"   ⚠️  Wash Sale Risk: {risk_level}")
            
            if impact.get('optimization_notes'):
                print(f"   💡 Optimization Notes:")
                for note in impact['optimization_notes']:
                    print(f"      • {note}")
    
    def show_advanced_portfolio_analysis(self):
        """Show advanced portfolio analysis with tax implications."""
        print("\n📈 Advanced Portfolio Analysis")
        print("-" * 32)
        
        try:
            analysis = self.tax_system.get_comprehensive_portfolio_analysis()
            
            # Portfolio summary
            portfolio = analysis['portfolio_summary']
            print(f"📊 Portfolio Overview:")
            print(f"   💰 Total Cost Basis: ${portfolio['total_cost_basis']:,.2f}")
            print(f"   🎯 Assets: {len(portfolio['holdings'])}")
            print(f"   📈 Tax Efficiency Score: {analysis['tax_efficiency_score']:.1f}/100")
            
            # Holdings breakdown
            print(f"\n💼 Holdings Breakdown:")
            for asset, holding in portfolio['holdings'].items():
                print(f"   {asset}: {holding['quantity']} units @ avg ${holding['avg_cost_basis']:.2f}")
            
            # Wash sale analysis
            wash_sale = analysis['wash_sale_analysis']
            print(f"\n⚠️  Wash Sale Analysis:")
            print(f"   🔄 Total Wash Sales: {wash_sale['total_wash_sales']}")
            print(f"   💸 Disallowed Loss: ${wash_sale['total_disallowed_loss']:,.2f}")
            print(f"   📊 Exposure Level: {wash_sale['exposure_level']}")
            
            # Holding period analysis
            holding_periods = analysis['holding_period_analysis']
            if 'short_term_percentage' in holding_periods:
                print(f"\n📅 Holding Period Analysis:")
                print(f"   📊 Short-term: {holding_periods['short_term_percentage']:.1f}%")
                print(f"   📈 Long-term: {holding_periods['long_term_percentage']:.1f}%")
                print(f"   🎯 Optimization: {holding_periods['optimization_score']}")
            
            # Optimization opportunities
            opportunities = analysis['optimization_opportunities']
            if opportunities:
                print(f"\n💡 Current Opportunities:")
                for opp in opportunities:
                    print(f"   • {opp}")
            
            # Risk assessment
            risks = analysis['risk_assessment']
            print(f"\n⚠️  Risk Assessment:")
            print(f"   🌊 Wash Sale Risk: {risks['wash_sale_exposure']}")
            print(f"   🎯 Concentration Risk: {risks['concentration_risk']}")
            print(f"   📊 Audit Risk: {risks['audit_risk']:.1%}")
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            print(f"❌ Portfolio analysis failed: {e}")
    
    def show_asset_specific_analysis(self):
        """Show detailed analysis for specific assets."""
        print("\n🔍 Asset-Specific Tax Analysis")
        print("-" * 32)
        
        # Analyze each asset in portfolio
        portfolio = self.tax_system.tax_engine.get_portfolio_summary()
        
        for asset in portfolio['holdings'].keys():
            try:
                analysis = self.tax_system.get_asset_tax_analysis(asset)
                
                print(f"\n📊 {asset} Analysis:")
                print(f"   💰 Current Position: {analysis['current_position']['quantity']} units")
                print(f"   💸 Cost Basis: ${analysis['current_position']['cost_basis']:,.2f}")
                print(f"   📈 Tax Lots: {len(analysis['tax_lots'])}")
                
                # Show wash sale risk
                if 'wash_sale_risk' in analysis:
                    risk = analysis['wash_sale_risk']
                    print(f"   ⚠️  Wash Sale Risk: {risk['risk_level']}")
                
                # Show harvesting potential
                if 'harvesting_potential' in analysis:
                    potential = analysis['harvesting_potential']
                    print(f"   🎯 Harvesting Potential: {potential['potential']}")
                    print(f"   💡 Recommendation: {potential['recommendation']}")
                
            except Exception as e:
                print(f"   ❌ Analysis failed for {asset}: {e}")
    
    def export_tax_data(self):
        """Export comprehensive tax data."""
        print("\n📤 Exporting Tax Data")
        print("-" * 22)
        
        try:
            # Export comprehensive data
            export_data = self.tax_system.export_tax_data(
                format='json', 
                include_sensitive=True
            )
            
            # Save to file
            export_file = Path("comprehensive_tax_export.json")
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Tax data exported to: {export_file}")
            print(f"   📊 Transactions: {len(export_data.get('transactions', []))}")
            print(f"   🎯 Tax Events: {len(export_data.get('tax_events', []))}")
            print(f"   📈 System Stats: {export_data['system_statistics']}")
            
            # Also export CSV for spreadsheet use
            csv_result = self.tax_system.tax_reporter.export_to_csv(
                {'tax_events': self.tax_system.tax_engine.tax_events},
                'tax_events_export.csv'
            )
            
            print(f"✅ CSV export: {csv_result.get('filename', 'tax_events_export.csv')}")
            print(f"   📊 Records: {csv_result.get('records', 0)}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            print(f"❌ Export failed: {e}")
    
    def show_system_status(self):
        """Show comprehensive system status."""
        print("\n⚙️  System Status")
        print("-" * 16)
        
        status = self.tax_system.get_system_status()
        
        print(f"🔧 Configuration:")
        print(f"   📍 Jurisdiction: {status['configuration']['jurisdiction']}")
        print(f"   📋 Accounting: {status['configuration']['accounting_method']}")
        print(f"   🔄 Wash Sale: {status['configuration']['wash_sale_enabled']}")
        
        print(f"\n📊 Statistics:")
        stats = status['statistics']
        print(f"   📝 Transactions: {stats['transactions_processed']}")
        print(f"   🎯 Tax Events: {stats['tax_events_generated']}")
        print(f"   📄 Reports: {stats['reports_created']}")
        print(f"   🔄 Optimization Runs: {stats['optimization_runs']}")
        
        print(f"\n💼 Portfolio:")
        portfolio = status['portfolio_summary']
        print(f"   🎯 Assets: {portfolio['total_assets']}")
        print(f"   📝 Transactions: {portfolio['total_transactions']}")
        print(f"   📊 Tax Events: {portfolio['total_tax_events']}")
        
        if status.get('optimization_performance'):
            perf = status['optimization_performance']
            print(f"\n🎯 Optimization Performance:")
            if 'total_opportunities_identified' in perf:
                print(f"   💡 Opportunities: {perf['total_opportunities_identified']}")
                print(f"   💰 Expected Savings: ${perf['total_expected_tax_savings']:,.2f}")
                print(f"   ✅ Realized Savings: ${perf['realized_tax_savings']:,.2f}")

def main():
    """Main execution function."""
    print("🚀 Starting Phase 8 Tax and Reporting System Example")
    print("=" * 60)
    
    # Create and run comprehensive example
    example = ComprehensiveTaxExample()
    example.run_complete_example()
    
    # Show final system status
    print("\n" + "="*60)
    example.show_system_status()
    
    print("\n🎉 Phase 8 Tax and Reporting System Example Complete!")
    print("📊 The system demonstrates:")
    print("   • Advanced tax calculation with wash sale detection")
    print("   • Professional tax reporting and compliance")  
    print("   • Sophisticated tax optimization and loss harvesting")
    print("   • Real-time tax impact analysis")
    print("   • Comprehensive audit trails and documentation")
    print("   • Multi-jurisdiction support and regulatory compliance")
    
    return example.tax_system

if __name__ == "__main__":
    tax_system = main()