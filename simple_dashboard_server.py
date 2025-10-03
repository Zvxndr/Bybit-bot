"""
Simple Dashboard Server
Serves only the professional dashboard without trading processes
Production-ready version for DigitalOcean deployment
Updated: October 1, 2025
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Health check endpoint for DigitalOcean
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "bybit-dashboard"}')
            return
            
        if self.path == '/' or self.path == '/dashboard':
            # Serve the AdminLTE dashboard (single source of truth)
            dashboard_path = Path(__file__).parent / 'src' / 'templates' / 'adminlte_dashboard.html'
            if dashboard_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.send_error(404, "Dashboard not found")
        else:
            # Handle other static files normally
            super().do_GET()

def main():
    # Use environment PORT or default to 8080
    PORT = int(os.environ.get('PORT', 8080))
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    Handler = DashboardHandler
    
    # Bind to 0.0.0.0 for DigitalOcean
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"🌐 Dashboard Server running at http://0.0.0.0:{PORT}")
        print(f"📱 Professional Dashboard: http://0.0.0.0:{PORT}/dashboard")
        print(f"💚 Health Check: http://0.0.0.0:{PORT}/health")
        print(f"🔧 Static Mode - No background trading processes")
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Dashboard server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()