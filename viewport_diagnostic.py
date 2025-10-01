"""
VIEWPORT DIAGNOSTIC TOOL
Help us identify your exact screen dimensions and viewport constraints
"""

def get_viewport_info():
    """Generate JavaScript to get exact viewport dimensions"""
    
    diagnostic_js = """
// VIEWPORT DIAGNOSTIC - Paste this in browser console (F12)
console.log("=== VIEWPORT DIAGNOSTIC ===");
console.log("Screen Resolution:", screen.width + "x" + screen.height);
console.log("Available Screen:", screen.availWidth + "x" + screen.availHeight);
console.log("Window Outer:", window.outerWidth + "x" + window.outerHeight);
console.log("Window Inner:", window.innerWidth + "x" + window.innerHeight);
console.log("Document Body:", document.body.clientWidth + "x" + document.body.clientHeight);
console.log("Device Pixel Ratio:", window.devicePixelRatio);
console.log("Viewport Meta:", document.querySelector('meta[name="viewport"]')?.content || "Not found");

// Test current layout constraints
const mainContainer = document.querySelector('.main-container');
const pipelineLayout = document.querySelector('.strategy-pipeline-layout');

if (mainContainer) {
    console.log("Main Container:", mainContainer.offsetWidth + "x" + mainContainer.offsetHeight);
    console.log("Main Container Computed:", getComputedStyle(mainContainer).height);
}

if (pipelineLayout) {
    console.log("Pipeline Layout:", pipelineLayout.offsetWidth + "x" + pipelineLayout.offsetHeight);
    console.log("Pipeline Computed:", getComputedStyle(pipelineLayout).height);
    console.log("Pipeline Scroll:", pipelineLayout.scrollHeight > pipelineLayout.clientHeight ? "OVERFLOWING" : "FITS");
}

// Check what's causing overflow
const body = document.body;
console.log("Body Scroll Height:", body.scrollHeight);
console.log("Body Client Height:", body.clientHeight);
console.log("Is Overflowing:", body.scrollHeight > body.clientHeight ? "YES - PROBLEM!" : "No");

console.log("=== END DIAGNOSTIC ===");
"""
    
    return diagnostic_js

def create_quick_fixes():
    """Generate multiple CSS fix attempts"""
    
    fixes = {
        "ultra_compact": """
/* ULTRA COMPACT FIX */
.debug-banner { height: 20px !important; padding: 2px 0 !important; font-size: 10px !important; }
.main-container { height: calc(100vh - 20px) !important; margin-top: 20px !important; }
.sidebar { width: 150px !important; padding: 0.25rem 0 !important; }
.sidebar-brand { padding: 0.25rem !important; }
.sidebar-brand h1 { font-size: 12px !important; }
.nav-link { padding: 0.25rem 0.5rem !important; font-size: 10px !important; }
.main-content { padding: 0.25rem !important; }
.content-title { font-size: 16px !important; margin-bottom: 0.25rem !important; }
.strategy-pipeline-layout { 
    height: calc(100vh - 80px) !important; 
    grid-template-columns: 120px 1fr 140px !important; 
    gap: 0.25rem !important; 
}
.pipeline-header { padding: 0.25rem !important; }
.pipeline-header h3 { font-size: 12px !important; margin-bottom: 0.25rem !important; }
""",
        
        "force_single_column": """
/* FORCE SINGLE COLUMN */
.main-container { grid-template-columns: 1fr !important; }
.sidebar { display: none !important; }
.strategy-pipeline-layout { 
    grid-template-columns: 1fr !important; 
    grid-template-rows: 200px 1fr 200px !important;
    height: calc(100vh - 60px) !important;
    gap: 0.5rem !important;
}
""",
        
        "micro_layout": """
/* MICRO LAYOUT */
* { font-size: 10px !important; }
.debug-banner { display: none !important; }
.main-container { height: 100vh !important; margin-top: 0 !important; grid-template-columns: 120px 1fr !important; }
.strategy-pipeline-layout { 
    height: calc(100vh - 40px) !important; 
    grid-template-columns: 100px 1fr 100px !important;
    gap: 0.25rem !important;
}
.pipeline-panel { border-radius: 4px !important; }
.pipeline-header { padding: 0.25rem !important; }
""",
        
        "scroll_enabled": """
/* ENABLE SCROLLING */
html, body { overflow: auto !important; height: auto !important; min-height: 100vh !important; }
.main-container { height: auto !important; min-height: calc(100vh - 32px) !important; }
.strategy-pipeline-layout { height: auto !important; min-height: 600px !important; }
.pipeline-panel { height: auto !important; max-height: 400px !important; }
"""
    }
    
    return fixes

if __name__ == "__main__":
    print("🔍 VIEWPORT DIAGNOSTIC HELPER")
    print("\n1. Open your dashboard in browser")
    print("2. Press F12 to open Developer Console")
    print("3. Paste this JavaScript code:")
    print("\n" + "="*60)
    print(get_viewport_info())
    print("="*60)
    
    print("\n📋 COPY THE CONSOLE OUTPUT AND SHARE IT")
    print("\nThen try these quick fixes one by one:")
    
    fixes = create_quick_fixes()
    for name, css in fixes.items():
        print(f"\n🔧 FIX #{name.upper()}:")
        print(f"document.head.insertAdjacentHTML('beforeend', `<style>{css}</style>`);")
        print("-" * 50)