"""
Simple Dashboard Server
Serves only the professional dashboard without trading processes
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            # Serve the professional dashboard
            dashboard_path = Path(__file__).parent / 'src' / 'templates' / 'professional_dashboard.html'
            if dashboard_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.send_error(404, "Dashboard not found")
        else:
            # Handle other static files normally
            super().do_GET()

def main():
    PORT = 8080
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    Handler = DashboardHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Dashboard Server running at http://localhost:{PORT}")
        print(f"📱 Professional Dashboard: http://localhost:{PORT}/dashboard")
        print(f"🔧 Static Mode - No background trading processes")
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Dashboard server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()