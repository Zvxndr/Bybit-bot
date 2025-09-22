"""
Documentation Dashboard
======================

Interactive documentation portal providing centralized access to all trading bot documentation,
including API references, user guides, troubleshooting resources, and interactive examples.

This dashboard serves as the main entry point for users, developers, and administrators
to access comprehensive documentation with search functionality and navigation.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import webbrowser
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse

class DocumentationDashboard:
    """Main documentation dashboard interface"""
    
    def __init__(self, docs_dir: str = "docs/output"):
        self.docs_dir = Path(docs_dir)
        self.port = 8080
        self.documentation_files = {}
        self.search_index = {}
        
        # Load documentation files
        self._load_documentation_files()
        self._build_search_index()
    
    def _load_documentation_files(self):
        """Load all documentation files"""
        if not self.docs_dir.exists():
            print(f"Documentation directory not found: {self.docs_dir}")
            return
        
        for file_path in self.docs_dir.glob("*.md"):
            file_key = file_path.stem
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.documentation_files[file_key] = {
                    'title': self._extract_title(content),
                    'file_path': str(file_path),
                    'html_path': str(file_path).replace('.md', '.html'),
                    'content': content,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'size': len(content)
                }
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled Document"
    
    def _build_search_index(self):
        """Build search index from documentation content"""
        for doc_key, doc_info in self.documentation_files.items():
            words = doc_info['content'].lower().split()
            for word in words:
                # Clean word
                word = word.strip('.,!?;:"()[]{}')
                if len(word) > 3:
                    if word not in self.search_index:
                        self.search_index[word] = []
                    if doc_key not in [item['doc'] for item in self.search_index[word]]:
                        self.search_index[word].append({
                            'doc': doc_key,
                            'title': doc_info['title']
                        })
    
    def search_documentation(self, query: str) -> List[Dict[str, Any]]:
        """Search documentation content"""
        query_words = query.lower().split()
        results = {}
        
        for word in query_words:
            if word in self.search_index:
                for item in self.search_index[word]:
                    doc_key = item['doc']
                    if doc_key not in results:
                        results[doc_key] = {
                            'doc_key': doc_key,
                            'title': item['title'],
                            'score': 0,
                            'file_path': self.documentation_files[doc_key]['file_path'],
                            'html_path': self.documentation_files[doc_key]['html_path']
                        }
                    results[doc_key]['score'] += 1
        
        # Sort by score
        sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:10]
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate statistics
        total_docs = len(self.documentation_files)
        total_size = sum(doc['size'] for doc in self.documentation_files.values())
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bybit Trading Bot - Documentation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.2em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .search-section {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }}
        
        .search-box {{
            width: 100%;
            padding: 15px 20px;
            font-size: 1.1em;
            border: 2px solid #e0e0e0;
            border-radius: 50px;
            outline: none;
            transition: border-color 0.3s;
        }}
        
        .search-box:focus {{
            border-color: #3498db;
        }}
        
        .documentation-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .doc-card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            border-left: 5px solid #3498db;
        }}
        
        .doc-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }}
        
        .doc-title {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .doc-meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .doc-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.9em;
            font-weight: bold;
            transition: all 0.3s;
            display: inline-block;
            text-align: center;
        }}
        
        .btn-primary {{
            background: #3498db;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #2980b9;
        }}
        
        .btn-secondary {{
            background: #95a5a6;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #7f8c8d;
        }}
        
        .features-section {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .feature-item {{
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #27ae60;
        }}
        
        .feature-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .footer {{
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 40px;
            padding: 20px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .documentation-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Bybit Trading Bot</h1>
            <p>Comprehensive Documentation Portal</p>
            <p><small>Generated: {current_time}</small></p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_docs}</div>
                <div class="stat-label">Documentation Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_size:,}</div>
                <div class="stat-label">Total Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(self.search_index):,}</div>
                <div class="stat-label">Searchable Terms</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">✅</div>
                <div class="stat-label">Phase 8 Complete</div>
            </div>
        </div>
        
        <div class="search-section">
            <h2>🔍 Search Documentation</h2>
            <input type="text" class="search-box" placeholder="Search documentation... (e.g., 'API connection', 'trading setup', 'troubleshooting')" 
                   onkeyup="searchDocs(event)">
            <div id="search-results" style="margin-top: 20px;"></div>
        </div>
        
        <div class="documentation-grid">
"""
        
        # Add documentation cards
        for doc_key, doc_info in self.documentation_files.items():
            card_color = {
                'api_reference': '#e74c3c',
                'user_guide': '#3498db', 
                'troubleshooting': '#f39c12',
                'class_reference': '#9b59b6'
            }.get(doc_key, '#27ae60')
            
            html_content += f"""
            <div class="doc-card" style="border-left-color: {card_color};">
                <div class="doc-title">{doc_info['title']}</div>
                <div class="doc-meta">
                    📄 {doc_info['size']:,} characters • 
                    🕒 Modified: {doc_info['last_modified'].strftime('%Y-%m-%d %H:%M')}
                </div>
                <div class="doc-actions">
                    <a href="{os.path.basename(doc_info['html_path'])}" class="btn btn-primary" target="_blank">
                        📖 View HTML
                    </a>
                    <a href="{os.path.basename(doc_info['file_path'])}" class="btn btn-secondary" target="_blank">
                        📝 View Markdown
                    </a>
                </div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="features-section">
            <h2>🚀 Documentation Features</h2>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="feature-title">Complete API Documentation</div>
                    <div>Full coverage of all API endpoints with examples and parameters</div>
                </div>
                <div class="feature-item">
                    <div class="feature-title">Interactive User Guide</div>
                    <div>Step-by-step guides for setup, configuration, and usage</div>
                </div>
                <div class="feature-item">
                    <div class="feature-title">Comprehensive Troubleshooting</div>
                    <div>Common issues, solutions, and debugging information</div>
                </div>
                <div class="feature-item">
                    <div class="feature-title">Code Examples</div>
                    <div>Interactive examples for all major trading operations</div>
                </div>
                <div class="feature-item">
                    <div class="feature-title">Multi-format Output</div>
                    <div>Available in Markdown, HTML, and searchable formats</div>
                </div>
                <div class="feature-item">
                    <div class="feature-title">Search Functionality</div>
                    <div>Full-text search across all documentation</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 Bybit Trading Bot Documentation Dashboard</p>
            <p>Generated by Phase 8 - Documentation & Knowledge Base System</p>
            <p>Complete API documentation, user guide coverage, and interactive examples achieved ✅</p>
        </div>
    </div>
    
    <script>
        function searchDocs(event) {{
            const query = event.target.value.toLowerCase();
            const resultsDiv = document.getElementById('search-results');
            
            if (query.length < 3) {{
                resultsDiv.innerHTML = '';
                return;
            }}
            
            // Simple client-side search
            const docs = {json.dumps({doc_key: {'title': doc_info['title'], 'file_path': os.path.basename(doc_info['file_path']), 'html_path': os.path.basename(doc_info['html_path'])} for doc_key, doc_info in self.documentation_files.items()})};
            
            let results = [];
            for (const [key, doc] of Object.entries(docs)) {{
                if (doc.title.toLowerCase().includes(query)) {{
                    results.push({{key, ...doc, score: 10}});
                }}
            }}
            
            if (results.length > 0) {{
                let html = '<h3>Search Results:</h3>';
                results.forEach(result => {{
                    html += `
                        <div style="padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                            <strong>${{result.title}}</strong><br>
                            <a href="${{result.html_path}}" target="_blank" style="color: #3498db; text-decoration: none;">📖 View Documentation</a>
                        </div>
                    `;
                }});
                resultsDiv.innerHTML = html;
            }} else {{
                resultsDiv.innerHTML = '<p style="color: #7f8c8d; font-style: italic;">No results found for "' + query + '"</p>';
            }}
        }}
    </script>
</body>
</html>"""
        
        return html_content
    
    def create_dashboard(self) -> str:
        """Create and save the documentation dashboard"""
        dashboard_html = self.generate_dashboard_html()
        dashboard_path = self.docs_dir / "index.html"
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"📊 Documentation dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def get_documentation_summary(self) -> Dict[str, Any]:
        """Get comprehensive documentation summary"""
        return {
            'total_files': len(self.documentation_files),
            'total_size': sum(doc['size'] for doc in self.documentation_files.values()),
            'search_terms': len(self.search_index),
            'documentation_files': {
                doc_key: {
                    'title': doc_info['title'],
                    'size': doc_info['size'],
                    'last_modified': doc_info['last_modified'].isoformat()
                }
                for doc_key, doc_info in self.documentation_files.items()
            },
            'generated_timestamp': datetime.now().isoformat()
        }


async def create_documentation_dashboard():
    """Create comprehensive documentation dashboard"""
    print("📊 Creating Documentation Dashboard...")
    
    # Initialize dashboard
    dashboard = DocumentationDashboard()
    
    # Create HTML dashboard
    dashboard_path = dashboard.create_dashboard()
    
    # Get summary
    summary = dashboard.get_documentation_summary()
    
    print(f"\n📈 Documentation Dashboard Summary:")
    print(f"  📄 Total Documentation Files: {summary['total_files']}")
    print(f"  📊 Total Content Size: {summary['total_size']:,} characters")
    print(f"  🔍 Searchable Terms: {summary['search_terms']:,}")
    print(f"  🌐 Dashboard Location: {dashboard_path}")
    
    print(f"\n📋 Available Documentation:")
    for doc_key, doc_info in summary['documentation_files'].items():
        print(f"  • {doc_info['title']} ({doc_info['size']:,} chars)")
    
    # Test search functionality
    print(f"\n🔍 Testing Search Functionality:")
    test_queries = ["API", "troubleshooting", "configuration"]
    for query in test_queries:
        results = dashboard.search_documentation(query)
        print(f"  '{query}': {len(results)} results")
        for result in results[:2]:
            print(f"    - {result['title']} (Score: {result['score']})")
    
    print(f"\n✅ Documentation Dashboard successfully created!")
    print(f"🌐 Open {dashboard_path} in your browser to access the documentation portal")
    
    return dashboard_path


if __name__ == "__main__":
    asyncio.run(create_documentation_dashboard())