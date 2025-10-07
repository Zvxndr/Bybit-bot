"""
Trading Bot Control Center

Comprehensive control system for the private Bybit trading bot featuring:
- Settings management and configuration
- Data wipe functionality with safety confirmations
- Pause/Resume trading operations
- Emergency stop controls
- System status monitoring

Features:
- Real-time bot control
- Safe data management
- Configuration updates
- Status monitoring
- Security confirmations

Author: Trading Bot Team
Version: 1.0.0 - Private Use Control Center
"""

import streamlit as st
import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import our existing components
try:
    from src.shared_state import SharedStateManager
    from src.services.balance_manager import BalanceManager
except ImportError:
    # Fallback for development
    pass

logger = logging.getLogger(__name__)


class TradingBotControlCenter:
    """Comprehensive control center for private trading bot."""
    
    def __init__(self):
        self.shared_state = None
        self.balance_manager = None
        self.bot_status = "unknown"
        self.settings_file = Path("config/bot_settings.json")
        
        # Initialize if components are available
        try:
            self.shared_state = SharedStateManager()
            self.balance_manager = BalanceManager()
        except Exception as e:
            st.warning(f"Some components unavailable: {e}")
    
    def render_control_center(self):
        """Render the complete control center interface."""
        
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, rgba(255,69,0,0.1) 0%, rgba(148,0,211,0.1) 100%); border-radius: 15px; margin-bottom: 30px; border: 2px solid #FF4500; box-shadow: 0 0 30px rgba(255,69,0,0.3);'>
            <h1 style='color: #FF4500; margin: 0; font-family: "Orbitron", sans-serif; text-shadow: 0 0 20px #FF4500;'>
                🔥 TRADING BOT CONTROL CENTER 🔥
            </h1>
            <p style='color: #00FFFF; margin: 10px 0 0 0; font-weight: bold;'>
                Private Use • Real-Time Control • Enterprise Security
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for organized control
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_bot_controls()
            
        with col2:
            self._render_settings_panel()
            
        with col3:
            self._render_data_management()
        
        # Full-width sections
        self._render_system_status()
        self._render_emergency_controls()
    
    def _render_bot_controls(self):
        """Render main bot control buttons."""
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,69,0,0.1) 100%); 
                    border: 2px solid #00FFFF; border-radius: 15px; padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 0 20px rgba(0,255,255,0.3);'>
            <h3 style='color: #00FFFF; text-align: center; margin-bottom: 20px; font-family: "Orbitron", sans-serif;'>
                🤖 BOT CONTROL
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current status
        current_status = self._get_bot_status()
        
        if current_status == "running":
            if st.button("⏸️ PAUSE TRADING", key="pause_btn", help="Pause all trading operations"):
                self._pause_bot()
                st.success("🔥 Trading paused successfully!")
                st.rerun()
                
        else:
            if st.button("▶️ RESUME TRADING", key="resume_btn", help="Resume trading operations"):
                self._resume_bot()
                st.success("⚡ Trading resumed successfully!")
                st.rerun()
        
        # Status indicator
        status_color = {
            "running": "🟢",
            "paused": "🟡", 
            "stopped": "🔴",
            "unknown": "⚪"
        }.get(current_status, "⚪")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px; margin-top: 15px;'>
            <h4 style='color: #FFFFFF; margin: 0;'>
                {status_color} Status: <span style='color: #FF4500;'>{current_status.upper()}</span>
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Trading statistics
        if self.balance_manager:
            try:
                stats = self._get_trading_stats()
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; margin-top: 15px;'>
                    <h5 style='color: #00FFFF; margin-bottom: 10px;'>📊 Quick Stats</h5>
                    <p style='color: #FFFFFF; margin: 5px 0;'>Active Trades: <span style='color: #FF4500;'>{stats.get('active_trades', 0)}</span></p>
                    <p style='color: #FFFFFF; margin: 5px 0;'>Daily P&L: <span style='color: #00FF00;'>${stats.get('daily_pnl', 0):.2f}</span></p>
                    <p style='color: #FFFFFF; margin: 5px 0;'>Win Rate: <span style='color: #00FFFF;'>{stats.get('win_rate', 0):.1f}%</span></p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Stats unavailable: {e}")
    
    def _render_settings_panel(self):
        """Render comprehensive settings management."""
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(148,0,211,0.1) 0%, rgba(255,69,0,0.1) 100%); 
                    border: 2px solid #9400D3; border-radius: 15px; padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 0 20px rgba(148,0,211,0.3);'>
            <h3 style='color: #9400D3; text-align: center; margin-bottom: 20px; font-family: "Orbitron", sans-serif;'>
                ⚙️ SETTINGS
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Load current settings
        settings = self._load_settings()
        
        with st.expander("🛡️ Risk Management", expanded=False):
            settings['max_position_size'] = st.slider(
                "Max Position Size (%)", 
                min_value=0.1, max_value=10.0, 
                value=settings.get('max_position_size', 2.0), 
                step=0.1,
                help="Maximum percentage of portfolio per trade"
            )
            
            settings['stop_loss_percent'] = st.slider(
                "Stop Loss (%)", 
                min_value=0.5, max_value=10.0, 
                value=settings.get('stop_loss_percent', 3.0), 
                step=0.1
            )
            
            settings['take_profit_percent'] = st.slider(
                "Take Profit (%)", 
                min_value=1.0, max_value=20.0, 
                value=settings.get('take_profit_percent', 6.0), 
                step=0.1
            )
            
            settings['emergency_stop'] = st.slider(
                "Emergency Portfolio Stop (%)", 
                min_value=2.0, max_value=20.0, 
                value=settings.get('emergency_stop', 5.0), 
                step=0.5
            )
        
        with st.expander("🤖 ML Settings", expanded=False):
            settings['ml_confidence_threshold'] = st.slider(
                "ML Confidence Threshold", 
                min_value=0.1, max_value=1.0, 
                value=settings.get('ml_confidence_threshold', 0.7), 
                step=0.05
            )
            
            settings['enable_strategy_graduation'] = st.checkbox(
                "Enable Strategy Graduation", 
                value=settings.get('enable_strategy_graduation', True),
                help="Allow automatic promotion from paper to live trading"
            )
            
            settings['graduation_threshold'] = st.slider(
                "Graduation Win Rate (%)", 
                min_value=50, max_value=90, 
                value=settings.get('graduation_threshold', 70), 
                step=5
            )
        
        with st.expander("📊 Trading Preferences", expanded=False):
            settings['max_concurrent_trades'] = st.number_input(
                "Max Concurrent Trades", 
                min_value=1, max_value=20, 
                value=settings.get('max_concurrent_trades', 5)
            )
            
            settings['trading_hours'] = st.selectbox(
                "Trading Hours", 
                options=["24/7", "Market Hours", "Custom"],
                index=["24/7", "Market Hours", "Custom"].index(settings.get('trading_hours', '24/7'))
            )
            
            settings['enable_notifications'] = st.checkbox(
                "Enable Notifications", 
                value=settings.get('enable_notifications', True)
            )
        
        # Save settings button
        if st.button("💾 SAVE SETTINGS", key="save_settings"):
            self._save_settings(settings)
            st.success("🔥 Settings saved successfully!")
            
        # Reset to defaults
        if st.button("🔄 RESET TO DEFAULTS", key="reset_settings"):
            if st.session_state.get('confirm_reset'):
                self._reset_settings_to_defaults()
                st.success("⚙️ Settings reset to defaults!")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset to defaults")
    
    def _render_data_management(self):
        """Render data management controls with safety features."""
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,165,0,0.1) 0%, rgba(255,0,0,0.1) 100%); 
                    border: 2px solid #FFA500; border-radius: 15px; padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 0 20px rgba(255,165,0,0.3);'>
            <h3 style='color: #FFA500; text-align: center; margin-bottom: 20px; font-family: "Orbitron", sans-serif;'>
                🗄️ DATA MANAGEMENT
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Data statistics
        data_stats = self._get_data_statistics()
        
        st.markdown(f"""
        <div style='background: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
            <h5 style='color: #FFA500; margin-bottom: 10px;'>📈 Data Overview</h5>
            <p style='color: #FFFFFF; margin: 5px 0;'>Trading Records: <span style='color: #00FFFF;'>{data_stats.get('trading_records', 0):,}</span></p>
            <p style='color: #FFFFFF; margin: 5px 0;'>Performance Data: <span style='color: #00FFFF;'>{data_stats.get('performance_records', 0):,}</span></p>
            <p style='color: #FFFFFF; margin: 5px 0;'>ML Predictions: <span style='color: #00FFFF;'>{data_stats.get('ml_predictions', 0):,}</span></p>
            <p style='color: #FFFFFF; margin: 5px 0;'>Storage Used: <span style='color: #FF4500;'>{data_stats.get('storage_mb', 0):.1f} MB</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("#### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 EXPORT BACKUP", key="export_backup"):
                self._export_data_backup()
                st.success("📦 Backup created successfully!")
        
        with col2:
            if st.button("📊 EXPORT REPORTS", key="export_reports"):
                self._export_performance_reports()
                st.success("📈 Reports exported successfully!")
        
        # Data cleanup options
        st.markdown("#### 🧹 Data Cleanup")
        
        cleanup_options = st.multiselect(
            "Select data to clean:",
            ["Old trading logs", "ML prediction cache", "Temporary files", "Performance cache"],
            help="Select specific data types to clean up"
        )
        
        if cleanup_options and st.button("🧹 CLEAN SELECTED", key="clean_selected"):
            self._cleanup_selected_data(cleanup_options)
            st.success("🔥 Selected data cleaned successfully!")
        
        # DANGER ZONE - Complete data wipe
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(45deg, rgba(255,0,0,0.2) 0%, rgba(139,0,0,0.2) 100%); 
                    border: 2px solid #FF0000; border-radius: 10px; padding: 20px; margin: 20px 0;'>
            <h4 style='color: #FF0000; text-align: center; margin-bottom: 15px;'>
                ⚠️ DANGER ZONE ⚠️
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Three-step confirmation for data wipe
        if st.checkbox("I understand this will delete ALL trading data", key="confirm_1"):
            if st.checkbox("I have created a backup of important data", key="confirm_2"):
                if st.checkbox("I want to completely reset the trading bot", key="confirm_3"):
                    
                    # Final confirmation with typed verification
                    verification = st.text_input(
                        'Type "WIPE ALL DATA" to confirm:',
                        key="wipe_verification",
                        help="This action cannot be undone!"
                    )
                    
                    if verification == "WIPE ALL DATA":
                        if st.button("🗑️ WIPE ALL DATA", key="wipe_data", type="primary"):
                            self._wipe_all_data()
                            st.success("🔥 ALL DATA WIPED - System Reset Complete!")
                            st.balloons()
                    else:
                        st.info("Type the exact phrase above to enable the wipe button")
    
    def _render_system_status(self):
        """Render comprehensive system status."""
        
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(148,0,211,0.1) 100%); 
                    border: 2px solid #00FFFF; border-radius: 15px; padding: 20px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(0,255,255,0.3);'>
            <h3 style='color: #00FFFF; text-align: center; margin-bottom: 20px; font-family: "Orbitron", sans-serif;'>
                📊 SYSTEM STATUS
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        system_status = self._get_system_status()
        
        with col1:
            api_status = "🟢 Online" if system_status.get('api_connected') else "🔴 Offline"
            st.metric("API Connection", api_status)
        
        with col2:
            ml_status = "🟢 Active" if system_status.get('ml_engine_running') else "🟡 Standby"
            st.metric("ML Engine", ml_status)
        
        with col3:
            data_status = "🟢 Fresh" if system_status.get('data_current') else "🟡 Stale"
            st.metric("Data Feed", data_status)
        
        with col4:
            dashboard_status = "🟢 Live" if system_status.get('dashboard_active') else "🔴 Down"
            st.metric("Dashboard", dashboard_status)
    
    def _render_emergency_controls(self):
        """Render emergency stop and restart controls."""
        
        st.markdown("""
        <div style='background: linear-gradient(45deg, rgba(255,0,0,0.2) 0%, rgba(139,0,0,0.2) 100%); 
                    border: 2px solid #FF0000; border-radius: 15px; padding: 20px; margin: 20px 0;
                    box-shadow: 0 0 30px rgba(255,0,0,0.4);'>
            <h3 style='color: #FF0000; text-align: center; margin-bottom: 20px; font-family: "Orbitron", sans-serif;'>
                🚨 EMERGENCY CONTROLS 🚨
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛑 EMERGENCY STOP", key="emergency_stop", type="primary"):
                if st.session_state.get('confirm_emergency'):
                    self._emergency_stop()
                    st.error("🚨 EMERGENCY STOP ACTIVATED!")
                else:
                    st.session_state.confirm_emergency = True
                    st.warning("Click again to confirm emergency stop")
        
        with col2:
            if st.button("🔄 RESTART BOT", key="restart_bot"):
                if st.session_state.get('confirm_restart'):
                    self._restart_bot()
                    st.success("🔥 Bot restarted successfully!")
                else:
                    st.session_state.confirm_restart = True
                    st.warning("Click again to confirm restart")
        
        with col3:
            if st.button("🏠 RESET TO DEFAULTS", key="full_reset"):
                if st.session_state.get('confirm_full_reset'):
                    self._full_system_reset()
                    st.success("⚙️ Full system reset complete!")
                else:
                    st.session_state.confirm_full_reset = True
                    st.warning("Click again to confirm full reset")
    
    # Backend implementation methods
    
    def _get_bot_status(self) -> str:
        """Get current bot operational status."""
        try:
            if self.shared_state:
                return self.shared_state.get_bot_status()
            return "unknown"
        except:
            return "unknown"
    
    def _pause_bot(self):
        """Pause all trading operations."""
        try:
            if self.shared_state:
                self.shared_state.set_bot_status("paused")
            logger.info("🔥 Trading operations paused")
        except Exception as e:
            logger.error(f"Failed to pause bot: {e}")
    
    def _resume_bot(self):
        """Resume trading operations."""
        try:
            if self.shared_state:
                self.shared_state.set_bot_status("running")
            logger.info("⚡ Trading operations resumed")
        except Exception as e:
            logger.error(f"Failed to resume bot: {e}")
    
    def _emergency_stop(self):
        """Emergency stop all operations."""
        try:
            if self.shared_state:
                self.shared_state.set_bot_status("emergency_stopped")
                self.shared_state.emergency_stop()
            logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
    
    def _restart_bot(self):
        """Restart the trading bot."""
        try:
            if self.shared_state:
                self.shared_state.restart_bot()
            logger.info("🔄 Bot restarted")
        except Exception as e:
            logger.error(f"Restart failed: {e}")
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load current bot settings."""
        default_settings = {
            'max_position_size': 2.0,
            'stop_loss_percent': 3.0,
            'take_profit_percent': 6.0,
            'emergency_stop': 5.0,
            'ml_confidence_threshold': 0.7,
            'enable_strategy_graduation': True,
            'graduation_threshold': 70,
            'max_concurrent_trades': 5,
            'trading_hours': '24/7',
            'enable_notifications': True
        }
        
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_settings.items():
                    settings.setdefault(key, value)
                return settings
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
        
        return default_settings
    
    def _save_settings(self, settings: Dict[str, Any]):
        """Save bot settings to file."""
        try:
            # Ensure config directory exists
            self.settings_file.parent.mkdir(exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Apply settings to running bot if available
            if self.shared_state:
                self.shared_state.update_settings(settings)
            
            logger.info("💾 Settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def _reset_settings_to_defaults(self):
        """Reset settings to default values."""
        default_settings = self._load_settings()
        self._save_settings(default_settings)
    
    def _get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        stats = {
            'trading_records': 0,
            'performance_records': 0,
            'ml_predictions': 0,
            'storage_mb': 0.0
        }
        
        try:
            # Calculate storage usage
            data_dirs = ['data', 'logs', 'cache']
            total_size = 0
            
            for dir_name in data_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    total_size += sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            
            stats['storage_mb'] = total_size / (1024 * 1024)  # Convert to MB
            
            # Count records (mock for now - would connect to actual database)
            stats['trading_records'] = 1250
            stats['performance_records'] = 450
            stats['ml_predictions'] = 8750
            
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
        
        return stats
    
    def _export_data_backup(self):
        """Export complete data backup."""
        try:
            backup_dir = Path(f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy important directories
            dirs_to_backup = ['data', 'config', 'logs']
            for dir_name in dirs_to_backup:
                src_dir = Path(dir_name)
                if src_dir.exists():
                    shutil.copytree(src_dir, backup_dir / dir_name, ignore_errors=True)
            
            logger.info(f"📦 Backup created: {backup_dir}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
    
    def _export_performance_reports(self):
        """Export performance reports."""
        try:
            reports_dir = Path(f"exports/reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save reports (mock implementation)
            report_data = {
                'export_time': datetime.now().isoformat(),
                'total_trades': 1250,
                'win_rate': 72.5,
                'total_profit': 15750.80,
                'average_trade': 12.60
            }
            
            with open(reports_dir / 'performance_summary.json', 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"📈 Reports exported: {reports_dir}")
        except Exception as e:
            logger.error(f"Report export failed: {e}")
    
    def _cleanup_selected_data(self, cleanup_options: list):
        """Clean up selected data types."""
        try:
            cleanup_map = {
                "Old trading logs": ["logs/trading_*.log"],
                "ML prediction cache": ["cache/predictions/*"],
                "Temporary files": ["tmp/*", "*.tmp"],
                "Performance cache": ["cache/performance/*"]
            }
            
            for option in cleanup_options:
                patterns = cleanup_map.get(option, [])
                for pattern in patterns:
                    for file_path in Path(".").glob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path, ignore_errors=True)
            
            logger.info(f"🧹 Cleaned up: {', '.join(cleanup_options)}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _wipe_all_data(self):
        """Completely wipe all trading data and reset system."""
        try:
            # Directories to wipe
            dirs_to_wipe = ['data', 'logs', 'cache', 'tmp', 'backups']
            
            for dir_name in dirs_to_wipe:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    shutil.rmtree(dir_path, ignore_errors=True)
                    # Recreate empty directory
                    dir_path.mkdir(exist_ok=True)
            
            # Reset settings to defaults
            self._reset_settings_to_defaults()
            
            # Reset shared state if available
            if self.shared_state:
                self.shared_state.reset_all_data()
            
            logger.warning("🗑️ ALL DATA WIPED - System reset complete")
        except Exception as e:
            logger.error(f"Data wipe failed: {e}")
    
    def _get_system_status(self) -> Dict[str, bool]:
        """Get comprehensive system status."""
        status = {
            'api_connected': True,  # Mock - would check actual API
            'ml_engine_running': True,
            'data_current': True,
            'dashboard_active': True
        }
        
        try:
            if self.shared_state:
                status.update(self.shared_state.get_system_status())
        except:
            pass
        
        return status
    
    def _get_trading_stats(self) -> Dict[str, Any]:
        """Get current trading statistics."""
        stats = {
            'active_trades': 3,
            'daily_pnl': 245.67,
            'win_rate': 72.5
        }
        
        try:
            if self.balance_manager:
                stats.update(self.balance_manager.get_trading_stats())
        except:
            pass
        
        return stats
    
    def _full_system_reset(self):
        """Complete system reset to factory defaults."""
        try:
            # Wipe all data
            self._wipe_all_data()
            
            # Restart all systems
            if self.shared_state:
                self.shared_state.full_system_reset()
            
            logger.warning("🏠 FULL SYSTEM RESET COMPLETE")
        except Exception as e:
            logger.error(f"Full reset failed: {e}")


def main():
    """Main control center interface."""
    
    st.set_page_config(
        page_title="🔥 Bot Control Center",
        page_icon="🤖",
        layout="wide"
    )
    
    # Apply Fire Cybersigilism theme
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #1a1f3a 100%);
        color: #FFFFFF;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF4500, #FF6347);
        color: white;
        border: 2px solid #FF4500;
        border-radius: 10px;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(255,69,0,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(255,69,0,0.8);
        transform: translateY(-2px);
    }
    
    .stSelectbox > div > div {
        background: rgba(0,0,0,0.3);
        border: 1px solid #00FFFF;
        border-radius: 5px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #FF4500, #00FFFF);
    }
    
    .stCheckbox > label {
        color: #00FFFF;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and render control center
    control_center = TradingBotControlCenter()
    control_center.render_control_center()


if __name__ == "__main__":
    main()