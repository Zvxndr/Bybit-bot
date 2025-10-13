#!/bin/bash
# Build React frontend for production deployment on DigitalOcean

echo "🛠️  Building React Frontend for Production"

# Change to frontend directory
cd frontend

# Install dependencies (if package-lock.json exists, Node.js is available)
if command -v npm &> /dev/null; then
    echo "📦 Installing npm dependencies..."
    npm install
    
    echo "🔨 Building React app for production..."
    npm run build
    
    echo "✅ Frontend build complete!"
    echo "📁 Build output in: frontend/dist/"
    
else
    echo "⚠️  Node.js not available - using pre-built static files"
    
    # Create a simple static build directory
    mkdir -p dist
    
    # Create index.html that loads our components
    cat > dist/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Trading Dashboard</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        gray: {
                            750: '#374151',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    </style>
</head>
<body class="bg-gray-900">
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;

        const Dashboard = () => {
            const [strategies, setStrategies] = useState([]);
            const [mlStatus, setMlStatus] = useState(null);
            const [selectedPeriod, setSelectedPeriod] = useState('all');
            const [loading, setLoading] = useState(true);
            const [isAuthenticated, setIsAuthenticated] = useState(false);
            const [loginForm, setLoginForm] = useState({ username: '', password: '' });

            const handleLogin = (e) => {
                e.preventDefault();
                if (loginForm.username === 'admin' && loginForm.password === 'password') {
                    setIsAuthenticated(true);
                    localStorage.setItem('isAuthenticated', 'true');
                }
            };

            const handleLogout = () => {
                setIsAuthenticated(false);
                localStorage.removeItem('isAuthenticated');
            };

            useEffect(() => {
                setIsAuthenticated(localStorage.getItem('isAuthenticated') === 'true');
            }, []);

            useEffect(() => {
                if (!isAuthenticated) return;

                const fetchData = async () => {
                    try {
                        const [strategiesResponse, mlStatusResponse] = await Promise.all([
                            fetch(`/api/strategies/ranking?period=${selectedPeriod}`),
                            fetch('/api/ml/status')
                        ]);

                        const strategiesData = await strategiesResponse.json();
                        const mlStatusData = await mlStatusResponse.json();

                        setStrategies(strategiesData.strategies || []);
                        setMlStatus(mlStatusData);
                        setLoading(false);
                    } catch (error) {
                        console.error('Error fetching data:', error);
                        setLoading(false);
                    }
                };

                fetchData();
                const interval = setInterval(fetchData, 5000);
                return () => clearInterval(interval);
            }, [selectedPeriod, isAuthenticated]);

            if (!isAuthenticated) {
                return (
                    <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4">
                        <div className="max-w-md w-full">
                            <div className="text-center mb-8">
                                <h1 className="text-2xl font-bold text-white mb-2">ML Trading Dashboard</h1>
                                <p className="text-gray-400">Secure access to your autonomous trading system</p>
                            </div>
                            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
                                <form onSubmit={handleLogin} className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">Username</label>
                                        <input
                                            type="text"
                                            value={loginForm.username}
                                            onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                                            placeholder="Enter username"
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
                                        <input
                                            type="password"
                                            value={loginForm.password}
                                            onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                                            placeholder="Enter password"
                                            required
                                        />
                                    </div>
                                    <button
                                        type="submit"
                                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
                                    >
                                        Sign In
                                    </button>
                                </form>
                                <div className="mt-4 p-3 bg-gray-750 rounded-lg border border-gray-600">
                                    <p className="text-xs text-gray-400">Demo: username: admin, password: password</p>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            }

            if (loading) {
                return (
                    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                        <div className="text-white">Loading...</div>
                    </div>
                );
            }

            return (
                <div className="min-h-screen bg-gray-900">
                    <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                        <div className="flex items-center justify-between">
                            <h1 className="text-xl font-bold text-white">ML Trading Dashboard</h1>
                            <button
                                onClick={handleLogout}
                                className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded-lg text-sm transition-colors"
                            >
                                Logout
                            </button>
                        </div>
                    </nav>

                    <main className="p-6 space-y-6">
                        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
                            <div>
                                <h1 className="text-3xl font-bold text-white mb-2">Strategy Dashboard</h1>
                                <p className="text-gray-400">ML-generated autonomous trading strategies ranked by performance</p>
                            </div>
                            
                            <div className="flex space-x-2 mt-4 lg:mt-0">
                                {['all', 'year', 'month', 'week'].map((period) => (
                                    <button
                                        key={period}
                                        onClick={() => setSelectedPeriod(period)}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium uppercase tracking-wide transition-colors ${
                                            selectedPeriod === period
                                                ? 'bg-blue-600 text-white'
                                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                        }`}
                                    >
                                        {period === 'all' ? 'All Time' : period}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {mlStatus && (
                            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
                                <h3 className="text-lg font-semibold text-white mb-4">ML Algorithm Status</h3>
                                <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
                                    <div className="text-center">
                                        <div className="text-2xl font-bold text-purple-400">{mlStatus.status}</div>
                                        <div className="text-xs text-gray-400 uppercase">Status</div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-2xl font-bold text-blue-400">{mlStatus.generation_rate}</div>
                                        <div className="text-xs text-gray-400 uppercase">Strategies/Hour</div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-2xl font-bold text-green-400">{mlStatus.processing}</div>
                                        <div className="text-xs text-gray-400 uppercase">Processing</div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-2xl font-bold text-yellow-400">{mlStatus.queue}</div>
                                        <div className="text-xs text-gray-400 uppercase">In Queue</div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-2xl font-bold text-red-400">{mlStatus.cpu_usage}%</div>
                                        <div className="text-xs text-gray-400 uppercase">CPU Usage</div>
                                    </div>
                                </div>
                                <div>
                                    <h4 className="text-sm font-medium text-gray-300 mb-2">Recent Activity</h4>
                                    <div className="space-y-1 max-h-32 overflow-y-auto">
                                        {mlStatus.recent_activity?.map((activity, index) => (
                                            <div key={index} className="flex space-x-3 text-sm">
                                                <span className="text-gray-500 min-w-12">{activity.timestamp}</span>
                                                <span className="text-gray-300">{activity.message}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-700">
                                <h2 className="text-xl font-semibold text-white">
                                    Strategy Rankings - {selectedPeriod === 'all' ? 'All Time' : selectedPeriod.charAt(0).toUpperCase() + selectedPeriod.slice(1)}
                                </h2>
                                <p className="text-gray-400 text-sm mt-1">{strategies.length} active strategies</p>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead className="bg-gray-750">
                                        <tr className="text-left">
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Rank</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Strategy</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Status</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Return</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Sharpe</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Win Rate</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Max DD</th>
                                            <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase">Trades</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-700">
                                        {strategies.map((strategy) => (
                                            <tr key={strategy.id} className="hover:bg-gray-750">
                                                <td className="px-6 py-4">
                                                    <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                                                        strategy.rank === 1 ? 'bg-yellow-500 text-black' :
                                                        strategy.rank === 2 ? 'bg-gray-400 text-black' :
                                                        strategy.rank === 3 ? 'bg-orange-600 text-white' :
                                                        'bg-gray-600 text-gray-300'
                                                    }`}>
                                                        {strategy.rank}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-white font-medium">{strategy.name}</td>
                                                <td className="px-6 py-4">
                                                    <span className={`px-2 py-1 rounded-full text-xs font-medium uppercase ${
                                                        strategy.status === 'live' ? 'bg-green-900 text-green-300' :
                                                        strategy.status === 'paper' ? 'bg-blue-900 text-blue-300' :
                                                        'bg-yellow-900 text-yellow-300'
                                                    }`}>
                                                        {strategy.status}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className="text-green-400">+{strategy.return_percent.toFixed(1)}%</span>
                                                </td>
                                                <td className="px-6 py-4 text-white">{strategy.sharpe_ratio.toFixed(2)}</td>
                                                <td className="px-6 py-4 text-white">{strategy.win_rate.toFixed(1)}%</td>
                                                <td className="px-6 py-4">
                                                    <span className="text-red-400">-{strategy.max_drawdown.toFixed(1)}%</span>
                                                </td>
                                                <td className="px-6 py-4 text-white">{strategy.total_trades.toLocaleString()}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </main>
                </div>
            );
        };

        ReactDOM.render(<Dashboard />, document.getElementById('root'));
    </script>
</body>
</html>
EOF

    echo "✅ Static build created!"
fi

echo "🎯 Frontend ready for DigitalOcean deployment!"