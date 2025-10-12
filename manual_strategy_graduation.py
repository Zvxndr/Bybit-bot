#!/usr/bin/env python3
"""
Manual Strategy Graduation & Retirement Interface
===============================================

Interactive interface for manually managing strategy lifecycle:
- View strategies by phase (backtest, paper, live)
- Manual promotion/graduation controls
- Strategy retirement and analysis
- Performance comparison between strategies

This complements the automated pipeline with manual oversight.
"""

import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

def get_strategies_by_phase() -> Dict[str, List[Dict]]:
    """Get all strategies organized by pipeline phase"""
    
    try:
        conn = sqlite3.connect("data/trading_bot.db")
        cursor = conn.cursor()
        
        # Get all strategies with their current status
        cursor.execute("""
            SELECT strategy_id, strategy_name, current_phase, asset_pair, 
                   backtest_score, paper_pnl, live_pnl, sharpe_ratio, 
                   win_rate, created_at, last_updated
            FROM strategy_pipeline 
            ORDER BY current_phase, last_updated DESC
        """)
        
        strategies = cursor.fetchall()
        conn.close()
        
        # Organize by phase
        phases = {'backtest': [], 'paper': [], 'live': [], 'retired': []}
        
        for strategy in strategies:
            strategy_id, name, phase, asset, backtest_score, paper_pnl, live_pnl, sharpe, win_rate, created, updated = strategy
            
            strategy_data = {
                'strategy_id': strategy_id,
                'name': name or strategy_id,
                'asset_pair': asset,
                'backtest_score': backtest_score,
                'paper_pnl': paper_pnl,
                'live_pnl': live_pnl,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'created_at': created,
                'last_updated': updated,
                'phase': phase
            }
            
            if phase in phases:
                phases[phase].append(strategy_data)
            else:
                phases['retired'].append(strategy_data)
        
        return phases
        
    except Exception as e:
        print(f"❌ Error getting strategies: {e}")
        return {'backtest': [], 'paper': [], 'live': [], 'retired': []}

def display_strategies_by_phase(phases: Dict[str, List[Dict]]):
    """Display strategies organized by pipeline phase"""
    
    print("📊 CURRENT STRATEGY PIPELINE STATUS")
    print("=" * 60)
    
    for phase_name, strategies in phases.items():
        if not strategies:
            continue
            
        phase_emoji = {
            'backtest': '🧪',
            'paper': '📄', 
            'live': '🚀',
            'retired': '🏁'
        }
        
        print(f"\n{phase_emoji.get(phase_name, '📋')} {phase_name.upper()} PHASE ({len(strategies)} strategies)")
        print("-" * 50)
        
        for i, strategy in enumerate(strategies, 1):
            print(f"{i:2d}. {strategy['strategy_id'][:15]:<15} | {strategy['asset_pair']:<8}")
            
            if phase_name == 'backtest':
                score = strategy['backtest_score'] or 0
                sharpe = strategy['sharpe_ratio'] or 0
                print(f"     📈 Score: {score:.1f}% | Sharpe: {sharpe:.2f}")
                
            elif phase_name == 'paper':
                pnl = strategy['paper_pnl'] or 0
                win_rate = strategy['win_rate'] or 0
                print(f"     💰 P&L: ${pnl:.2f} | Win Rate: {win_rate:.1f}%")
                
            elif phase_name == 'live':
                pnl = strategy['live_pnl'] or 0
                print(f"     🎯 Live P&L: ${pnl:.2f}")
            
            print(f"     ⏰ Updated: {strategy['last_updated'][:16] if strategy['last_updated'] else 'N/A'}")
            print()

def manual_promote_strategy(strategy_id: str, from_phase: str, to_phase: str) -> bool:
    """Manually promote a strategy to the next phase"""
    
    try:
        conn = sqlite3.connect("data/trading_bot.db")
        cursor = conn.cursor()
        
        # Update strategy phase
        cursor.execute("""
            UPDATE strategy_pipeline 
            SET current_phase = ?, last_updated = ?
            WHERE strategy_id = ? AND current_phase = ?
        """, (to_phase, datetime.now().isoformat(), strategy_id, from_phase))
        
        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            print(f"✅ {strategy_id} promoted: {from_phase} → {to_phase}")
            return True
        else:
            conn.close()
            print(f"❌ Strategy {strategy_id} not found in {from_phase} phase")
            return False
            
    except Exception as e:
        print(f"❌ Error promoting strategy: {e}")
        return False

def manual_retire_strategy(strategy_id: str, reason: str = "Manual retirement") -> bool:
    """Manually retire a strategy"""
    
    try:
        conn = sqlite3.connect("data/trading_bot.db")
        cursor = conn.cursor()
        
        # Update strategy to retired status
        cursor.execute("""
            UPDATE strategy_pipeline 
            SET current_phase = ?, last_updated = ?, notes = ?
            WHERE strategy_id = ?
        """, ("retired", datetime.now().isoformat(), reason, strategy_id))
        
        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            print(f"🏁 {strategy_id} retired: {reason}")
            return True
        else:
            conn.close()
            print(f"❌ Strategy {strategy_id} not found")
            return False
            
    except Exception as e:
        print(f"❌ Error retiring strategy: {e}")
        return False

def compare_strategies(strategy_ids: List[str]):
    """Compare performance of multiple strategies"""
    
    try:
        conn = sqlite3.connect("data/trading_bot.db")
        cursor = conn.cursor()
        
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Strategy ID':<20} {'Phase':<10} {'Score':<8} {'Sharpe':<8} {'P&L':<10} {'Win Rate':<10}")
        print("-" * 80)
        
        for strategy_id in strategy_ids:
            cursor.execute("""
                SELECT strategy_id, current_phase, backtest_score, sharpe_ratio, 
                       COALESCE(paper_pnl, 0) + COALESCE(live_pnl, 0) as total_pnl,
                       win_rate
                FROM strategy_pipeline 
                WHERE strategy_id = ?
            """, (strategy_id,))
            
            result = cursor.fetchone()
            if result:
                sid, phase, score, sharpe, pnl, win_rate = result
                score_str = f"{score:.1f}%" if score else "N/A"
                sharpe_str = f"{sharpe:.2f}" if sharpe else "N/A"
                pnl_str = f"${pnl:.2f}" if pnl else "N/A"
                wr_str = f"{win_rate:.1f}%" if win_rate else "N/A"
                
                print(f"{sid:<20} {phase:<10} {score_str:<8} {sharpe_str:<8} {pnl_str:<10} {wr_str:<10}")
            else:
                print(f"{strategy_id:<20} {'NOT FOUND':<10}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error comparing strategies: {e}")

def interactive_graduation_interface():
    """Interactive manual graduation interface"""
    
    print("🎓 MANUAL STRATEGY GRADUATION & RETIREMENT INTERFACE")
    print("=" * 60)
    print("Commands:")
    print("  1. view - Show all strategies by phase")
    print("  2. promote <strategy_id> - Promote strategy to next phase")
    print("  3. retire <strategy_id> - Retire a strategy")
    print("  4. compare <id1> <id2> <id3> - Compare strategies")
    print("  5. quit - Exit interface")
    print()
    
    while True:
        try:
            command = input("📝 Enter command: ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print("👋 Goodbye!")
                break
                
            elif command == 'view' or command == '1':
                phases = get_strategies_by_phase()
                display_strategies_by_phase(phases)
                
            elif command.startswith('promote') or command.startswith('2'):
                parts = command.split()
                if len(parts) >= 2:
                    strategy_id = parts[1]
                    
                    # Get current phase
                    conn = sqlite3.connect("data/trading_bot.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT current_phase FROM strategy_pipeline WHERE strategy_id = ?", (strategy_id,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        current_phase = result[0]
                        
                        # Determine next phase
                        phase_progression = {
                            'backtest': 'paper',
                            'paper': 'live'
                        }
                        
                        if current_phase in phase_progression:
                            next_phase = phase_progression[current_phase]
                            manual_promote_strategy(strategy_id, current_phase, next_phase)
                        else:
                            print(f"❌ Cannot promote from {current_phase} phase")
                    else:
                        print(f"❌ Strategy {strategy_id} not found")
                else:
                    print("❌ Usage: promote <strategy_id>")
                    
            elif command.startswith('retire') or command.startswith('3'):
                parts = command.split()
                if len(parts) >= 2:
                    strategy_id = parts[1]
                    reason = " ".join(parts[2:]) if len(parts) > 2 else "Manual retirement"
                    manual_retire_strategy(strategy_id, reason)
                else:
                    print("❌ Usage: retire <strategy_id> [reason]")
                    
            elif command.startswith('compare') or command.startswith('4'):
                parts = command.split()
                if len(parts) >= 2:
                    strategy_ids = parts[1:]
                    compare_strategies(strategy_ids)
                else:
                    print("❌ Usage: compare <strategy_id1> <strategy_id2> [strategy_id3...]")
                    
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    
    print("🤖 AI TRADING SYSTEM - MANUAL STRATEGY MANAGEMENT")
    print()
    
    # Show current status first
    phases = get_strategies_by_phase()
    display_strategies_by_phase(phases)
    
    # Start interactive interface
    print(f"\n🎮 Starting interactive management interface...")
    print(f"   Total strategies: {sum(len(strategies) for strategies in phases.values())}")
    print()
    
    interactive_graduation_interface()

if __name__ == "__main__":
    main()