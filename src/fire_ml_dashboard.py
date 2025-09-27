"""
Fire Dashboard ML Components - Real-Time AI Display

Enhanced Fire Cybersigilism dashboard components to display ML predictions,
strategy graduation status, and live trading decisions in real-time.

Features:
- Animated ML prediction displays with fire/cyber aesthetics
- Real-time strategy graduation progress bars
- Live trading decision indicators
- Performance attribution visualizations
- Ensemble model confidence meters
- Cyberpunk-styled ML insights

Author: Trading Bot Team
Version: 1.0.0 - Fire Cybersigilism Edition
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional

# Custom Fire Cybersigilism styling
FIRE_COLORS = {
    'primary_fire': '#FF4500',      # OrangeRed
    'secondary_fire': '#FF6347',    # Tomato  
    'cyber_blue': '#00FFFF',        # Cyan
    'cyber_purple': '#9400D3',      # DarkViolet
    'dark_bg': '#0D1117',           # Dark background
    'success_green': '#00FF00',     # Lime
    'warning_orange': '#FFA500',    # Orange
    'danger_red': '#FF0000'         # Red
}


def apply_fire_cybersigilism_theme():
    """Apply Fire Cybersigilism theme to Streamlit components."""
    
    st.markdown("""
    <style>
    /* Fire Cybersigilism Theme */
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #1a1f3a 100%);
    }
    
    .fire-dashboard-container {
        background: linear-gradient(45deg, rgba(255,69,0,0.1) 0%, rgba(148,0,211,0.1) 100%);
        border: 2px solid #FF4500;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 30px rgba(255,69,0,0.3);
        animation: fire-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes fire-glow {
        from { box-shadow: 0 0 30px rgba(255,69,0,0.3); }
        to { box-shadow: 0 0 40px rgba(255,69,0,0.6); }
    }
    
    .ml-prediction-card {
        background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,69,0,0.1) 100%);
        border: 1px solid #00FFFF;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #00FFFF;
        animation: cyber-pulse 1.5s infinite;
    }
    
    @keyframes cyber-pulse {
        0%, 100% { border-color: #00FFFF; }
        50% { border-color: #9400D3; }
    }
    
    .strategy-graduation-bar {
        background: linear-gradient(90deg, #FF4500 0%, #00FF00 100%);
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        animation: fire-flow 3s linear infinite;
    }
    
    @keyframes fire-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .live-trading-indicator {
        color: #00FF00;
        font-weight: bold;
        animation: success-blink 1s infinite;
    }
    
    @keyframes success-blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.7; }
    }
    
    .fire-title {
        color: #FF4500;
        text-shadow: 0 0 10px rgba(255,69,0,0.8);
        font-weight: bold;
        text-align: center;
    }
    
    .cyber-title {
        color: #00FFFF;
        text-shadow: 0 0 10px rgba(0,255,255,0.8);
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


class FireMLDashboard:
    """Fire Cybersigilism ML Dashboard Components"""
    
    def __init__(self):
        self.fire_colors = FIRE_COLORS
        apply_fire_cybersigilism_theme()
    
    def display_ml_predictions(self, predictions_data: Dict[str, Any]):
        """Display real-time ML predictions with fire/cyber styling."""
        
        st.markdown('<div class="fire-dashboard-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="fire-title">🔥 REAL-TIME AI PREDICTIONS 🔥</h2>', unsafe_allow_html=True)
        
        if predictions_data and 'real_time_signals' in predictions_data:
            
            # Create columns for multiple predictions
            cols = st.columns(min(len(predictions_data['real_time_signals']), 3))
            
            for i, signal in enumerate(predictions_data['real_time_signals'][:3]):
                with cols[i % 3]:
                    self._display_prediction_card(signal)
            
            # ML Confidence Chart
            if len(predictions_data['real_time_signals']) > 0:
                st.markdown("### 🎯 Model Confidence & Agreement")
                self._display_confidence_chart(predictions_data['real_time_signals'])
        
        else:
            st.markdown('<div class="ml-prediction-card">', unsafe_allow_html=True)
            st.warning("🤖 ML Predictions initializing... Please wait for Speed Demon data.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _display_prediction_card(self, signal: Dict[str, Any]):
        """Display individual prediction card."""
        
        symbol = signal.get('symbol', 'UNKNOWN')
        prediction = signal.get('prediction', 0.0)
        confidence = signal.get('confidence', 0.0)
        signal_strength = signal.get('signal_strength', 'NEUTRAL')
        
        # Determine colors based on prediction
        if prediction > 0.6:
            card_color = "rgba(0,255,0,0.2)"    # Green for bullish
            text_color = "#00FF00"
        elif prediction < 0.4:
            card_color = "rgba(255,0,0,0.2)"    # Red for bearish  
            text_color = "#FF0000"
        else:
            card_color = "rgba(255,255,0,0.2)"  # Yellow for neutral
            text_color = "#FFFF00"
        
        st.markdown(f"""
        <div style="background: {card_color}; border: 1px solid {text_color}; 
                    border-radius: 10px; padding: 15px; margin: 10px 0;
                    animation: cyber-pulse 1.5s infinite;">
            <h4 style="color: {text_color}; text-align: center;">{symbol}</h4>
            <p style="color: {text_color}; font-size: 18px; text-align: center;">
                Prediction: <strong>{prediction:.3f}</strong>
            </p>
            <p style="color: {text_color}; text-align: center;">
                Confidence: <strong>{confidence:.1%}</strong>
            </p>
            <p style="color: {text_color}; text-align: center;">
                Strength: <strong>{signal_strength}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_confidence_chart(self, signals: List[Dict[str, Any]]):
        """Display ML model confidence chart."""
        
        # Prepare data for chart
        symbols = [s.get('symbol', 'Unknown') for s in signals]
        confidences = [s.get('confidence', 0) for s in signals]
        agreements = [s.get('model_agreement', 0) for s in signals]
        
        # Create dual-axis chart
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Confidence bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=confidences,
                name="Confidence",
                marker_color=self.fire_colors['primary_fire'],
                opacity=0.8
            ),
            secondary_y=False,
        )
        
        # Agreement line
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=agreements,
                mode='lines+markers',
                name="Model Agreement",
                line=dict(color=self.fire_colors['cyber_blue'], width=3),
                marker=dict(size=10)
            ),
            secondary_y=True,
        )
        
        # Styling
        fig.update_layout(
            title="ML Model Performance Metrics",
            plot_bgcolor='rgba(13,17,23,0.9)',
            paper_bgcolor='rgba(13,17,23,0.9)',
            font_color=self.fire_colors['cyber_blue']
        )
        
        fig.update_yaxes(title_text="Confidence Level", secondary_y=False)
        fig.update_yaxes(title_text="Model Agreement", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_strategy_graduation(self, graduation_data: Dict[str, Any]):
        """Display strategy graduation system status."""
        
        st.markdown('<div class="fire-dashboard-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="cyber-title">⚡ STRATEGY GRADUATION SYSTEM ⚡</h2>', unsafe_allow_html=True)
        
        # Graduation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            paper_count = graduation_data.get('paper_trading_count', 0)
            st.metric(
                label="📊 Paper Trading",
                value=paper_count,
                delta=f"+{paper_count-2}" if paper_count > 2 else None
            )
        
        with col2:
            live_ready = graduation_data.get('live_ready_count', 0)
            st.metric(
                label="🚀 Ready for Live",
                value=live_ready,
                delta=f"+{live_ready}" if live_ready > 0 else None
            )
        
        with col3:
            recent_grads = len(graduation_data.get('recent_graduations', []))
            st.metric(
                label="🎓 Recently Graduated",
                value=recent_grads,
                delta=f"+{recent_grads}" if recent_grads > 0 else None
            )
        
        with col4:
            auto_enabled = graduation_data.get('auto_graduation_enabled', False)
            status_color = "🟢" if auto_enabled else "🔴"
            st.metric(
                label="🤖 Auto Graduation",
                value=f"{status_color} {'ON' if auto_enabled else 'OFF'}"
            )
        
        # Graduation progress bars
        if graduation_data.get('recent_graduations'):
            st.markdown("### 📈 Recent Graduations")
            for strategy in graduation_data['recent_graduations'][:3]:
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <p style="color: {self.fire_colors['success_green']};">✅ {strategy}</p>
                    <div class="strategy-graduation-bar"></div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_live_trading_decisions(self, decisions_data: Dict[str, Any]):
        """Display live trading decisions influenced by ML."""
        
        st.markdown('<div class="fire-dashboard-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="fire-title">💰 LIVE TRADING DECISIONS 💰</h2>', unsafe_allow_html=True)
        
        if decisions_data:
            # ML Influence percentage
            ml_influence = decisions_data.get('ml_influence_percentage', 0)
            traditional_weight = decisions_data.get('traditional_strategy_weight', 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 ML Influence")
                self._display_influence_gauge(ml_influence, "ML Models")
            
            with col2:
                st.markdown("### 📊 Traditional Strategies")
                self._display_influence_gauge(traditional_weight, "Technical Analysis")
            
            # Recent decisions
            recent_decisions = decisions_data.get('recent_decisions', [])
            if recent_decisions:
                st.markdown("### 📋 Recent Trading Decisions")
                
                for decision in recent_decisions[-5:]:  # Show last 5 decisions
                    self._display_decision_card(decision)
        
        else:
            st.markdown('<div class="live-trading-indicator">', unsafe_allow_html=True)
            st.info("🤖 Live trading decisions will appear here when Speed Demon strategies are active.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _display_influence_gauge(self, percentage: float, label: str):
        """Display influence percentage as a gauge."""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': label},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': self.fire_colors['primary_fire']},
                'steps': [
                    {'range': [0, 50], 'color': self.fire_colors['cyber_blue']},
                    {'range': [50, 100], 'color': self.fire_colors['cyber_purple']}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(13,17,23,0.9)',
            paper_bgcolor='rgba(13,17,23,0.9)',
            font_color=self.fire_colors['cyber_blue']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_decision_card(self, decision: Dict[str, Any]):
        """Display individual trading decision card."""
        
        symbol = decision.get('symbol', 'UNKNOWN')
        action = decision.get('action', 'HOLD')
        ml_confidence = decision.get('ml_confidence', 0.0)
        position_size = decision.get('position_size', 0.0)
        expected_return = decision.get('expected_return', 0.0)
        timestamp = decision.get('timestamp', '')
        
        # Color coding
        if action == 'BUY':
            action_color = self.fire_colors['success_green']
            border_color = self.fire_colors['success_green']
        elif action == 'SELL':
            action_color = self.fire_colors['danger_red']
            border_color = self.fire_colors['danger_red']
        else:
            action_color = self.fire_colors['warning_orange']
            border_color = self.fire_colors['warning_orange']
        
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, rgba(255,69,0,0.1) 0%, rgba(0,255,255,0.1) 100%);
                    border: 2px solid {border_color}; border-radius: 10px; 
                    padding: 15px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: {action_color}; margin: 0;">{symbol} - {action}</h4>
                <span style="color: {self.fire_colors['cyber_blue']}; font-size: 12px;">{timestamp}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span style="color: {self.fire_colors['cyber_blue']};">
                    ML Confidence: <strong>{ml_confidence:.1%}</strong>
                </span>
                <span style="color: {self.fire_colors['cyber_blue']};">
                    Size: <strong>{position_size:.2%}</strong>
                </span>
                <span style="color: {self.fire_colors['cyber_blue']};">
                    Expected: <strong>{expected_return:.1%}</strong>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_performance_attribution(self, perf_data: Dict[str, Any]):
        """Display ML performance attribution analytics."""
        
        st.markdown('<div class="fire-dashboard-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="cyber-title">📊 PERFORMANCE ATTRIBUTION 📊</h2>', unsafe_allow_html=True)
        
        if perf_data:
            # ML Contribution breakdown
            ml_contrib = perf_data.get('ml_contribution', {})
            
            if ml_contrib:
                st.markdown("### 🤖 ML Components Performance")
                
                labels = list(ml_contrib.keys())
                values = list(ml_contrib.values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=[
                            self.fire_colors['primary_fire'],
                            self.fire_colors['cyber_blue'],
                            self.fire_colors['cyber_purple'],
                            self.fire_colors['success_green']
                        ]
                    )
                ])
                
                fig.update_layout(
                    title="ML Component Contributions",
                    plot_bgcolor='rgba(13,17,23,0.9)',
                    paper_bgcolor='rgba(13,17,23,0.9)',
                    font_color=self.fire_colors['cyber_blue'],
                    yaxis_title="Performance Score"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Strategy Performance comparison
            strategy_perf = perf_data.get('strategy_performance', {})
            
            if strategy_perf:
                st.markdown("### 🎯 Strategy Performance Metrics")
                
                strategy_df = pd.DataFrame(strategy_perf).T
                strategy_df.index.name = 'Strategy'
                strategy_df = strategy_df.round(3)
                
                st.dataframe(
                    strategy_df.style.format({
                        'return': '{:.1%}',
                        'sharpe': '{:.2f}',
                        'drawdown': '{:.1%}'
                    }),
                    use_container_width=True
                )
        
        else:
            st.info("🔄 Performance attribution data will be available after strategies start trading.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_ensemble_insights(self, ensemble_data: Dict[str, Any]):
        """Display ensemble model insights and weights."""
        
        st.markdown('<div class="fire-dashboard-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="fire-title">🧠 ENSEMBLE MODEL INSIGHTS 🧠</h2>', unsafe_allow_html=True)
        
        # Sample ensemble data if not provided
        if not ensemble_data:
            ensemble_data = {
                'model_weights': {
                    'LightGBM': 0.35,
                    'XGBoost': 0.25,
                    'RandomForest': 0.20,
                    'NeuralNet': 0.20
                },
                'model_agreement': 0.78,
                'prediction_distribution': {
                    'bullish': 0.45,
                    'neutral': 0.35,
                    'bearish': 0.20
                }
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model weights pie chart
            if 'model_weights' in ensemble_data:
                st.markdown("### 🎯 Model Weights")
                
                weights = ensemble_data['model_weights']
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(weights.keys()),
                        values=list(weights.values()),
                        marker_colors=[
                            self.fire_colors['primary_fire'],
                            self.fire_colors['cyber_blue'],
                            self.fire_colors['cyber_purple'],
                            self.fire_colors['success_green']
                        ]
                    )
                ])
                
                fig.update_layout(
                    plot_bgcolor='rgba(13,17,23,0.9)',
                    paper_bgcolor='rgba(13,17,23,0.9)',
                    font_color=self.fire_colors['cyber_blue']
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prediction distribution
            if 'prediction_distribution' in ensemble_data:
                st.markdown("### 📈 Prediction Distribution")
                
                dist = ensemble_data['prediction_distribution']
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        marker_color=[
                            self.fire_colors['success_green'],  # bullish
                            self.fire_colors['warning_orange'], # neutral
                            self.fire_colors['danger_red']      # bearish
                        ]
                    )
                ])
                
                fig.update_layout(
                    plot_bgcolor='rgba(13,17,23,0.9)',
                    paper_bgcolor='rgba(13,17,23,0.9)',
                    font_color=self.fire_colors['cyber_blue'],
                    yaxis_title="Probability"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement metric
        agreement = ensemble_data.get('model_agreement', 0.0)
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <h3 style="color: {self.fire_colors['cyber_blue']};">
                🤝 Model Agreement: <span style="color: {self.fire_colors['primary_fire']};">{agreement:.1%}</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# Global Fire ML Dashboard instance
fire_ml_dashboard = FireMLDashboard()