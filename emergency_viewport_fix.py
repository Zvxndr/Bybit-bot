"""
Emergency Viewport Fix - Forces dashboard to fit any screen size
Run this if the dashboard still doesn't fit properly
"""

# CSS Override for Ultra-Compact Display
VIEWPORT_FIT_CSS = """
<style id="emergency-viewport-fix">
/* EMERGENCY VIEWPORT FIT - Override all layouts */
* {
    box-sizing: border-box !important;
}

html, body {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

.debug-banner {
    height: 24px !important;
    padding: 4px 0 !important;
    font-size: 10px !important;
    line-height: 1 !important;
}

.main-container {
    height: calc(100vh - 24px) !important;
    max-height: calc(100vh - 24px) !important;
    margin-top: 24px !important;
    grid-template-columns: 200px 1fr !important;
}

.sidebar {
    padding: 0.5rem 0 !important;
    height: 100% !important;
    overflow-y: auto !important;
}

.sidebar-brand {
    padding: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

.sidebar-brand h1 {
    font-size: 14px !important;
    margin-bottom: 0.25rem !important;
}

.sidebar-brand p {
    font-size: 10px !important;
}

.nav-link {
    padding: 0.5rem 0.75rem !important;
    font-size: 11px !important;
}

.nav-link i {
    font-size: 10px !important;
    margin-right: 0.5rem !important;
}

.main-content {
    padding: 0.5rem !important;
    height: 100% !important;
    overflow-y: auto !important;
}

.content-title {
    font-size: 18px !important;
    margin-bottom: 0.25rem !important;
}

.content-subtitle {
    font-size: 11px !important;
    margin-bottom: 0.5rem !important;
}

.strategy-pipeline-layout {
    height: calc(100vh - 120px) !important;
    max-height: calc(100vh - 120px) !important;
    grid-template-columns: 180px 1fr 200px !important;
    gap: 0.5rem !important;
}

.pipeline-panel {
    max-height: 100% !important;
    overflow: hidden !important;
}

.pipeline-header {
    padding: 0.5rem !important;
}

.pipeline-header h3 {
    font-size: 14px !important;
    margin-bottom: 0.25rem !important;
}

.strategy-list, .backtest-content, .strategy-code-content {
    max-height: calc(100% - 60px) !important;
    overflow-y: auto !important;
}

/* Force mobile layout on very small screens */
@media (max-width: 1200px) {
    .main-container {
        grid-template-columns: 1fr !important;
    }
    
    .sidebar {
        display: none !important;
    }
    
    .strategy-pipeline-layout {
        grid-template-columns: 1fr !important;
        grid-template-rows: auto 1fr auto !important;
        height: calc(100vh - 80px) !important;
    }
}

/* Ultra-compact for very small heights */
@media (max-height: 700px) {
    .debug-banner {
        height: 20px !important;
        padding: 2px 0 !important;
        font-size: 9px !important;
    }
    
    .main-container {
        height: calc(100vh - 20px) !important;
        margin-top: 20px !important;
    }
    
    .strategy-pipeline-layout {
        height: calc(100vh - 80px) !important;
    }
}
</style>
"""

def apply_emergency_viewport_fix():
    """Apply emergency viewport fix by injecting CSS into dashboard"""
    print("🚨 EMERGENCY VIEWPORT FIX")
    print("Copy and paste this CSS into your browser's developer console:")
    print("document.head.insertAdjacentHTML('beforeend', `" + VIEWPORT_FIT_CSS + "`);")
    print("\n📱 Or add this to the dashboard HTML head section")
    print(VIEWPORT_FIT_CSS)

if __name__ == "__main__":
    apply_emergency_viewport_fix()