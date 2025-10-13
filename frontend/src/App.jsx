import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Dashboard from './components/Dashboard'
import Login from './components/Login'
import ApiStatus from './components/ApiStatus'
import AuthProvider, { useAuth } from './components/AuthProvider'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 10, // 10 minutes
      refetchOnWindowFocus: false,
    },
  },
})

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth()
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }
  
  return children
}

// App Layout with Navigation
const AppLayout = ({ children }) => {
  const { logout, user } = useAuth()
  
  const handleLogout = async () => {
    await logout()
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Top Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-white">ML Trading Dashboard</h1>
            <div className="flex space-x-4 ml-8">
              <button 
                onClick={() => window.location.href = '/dashboard'}
                className="text-gray-300 hover:text-white transition-colors"
              >
                Dashboard
              </button>
              <button 
                onClick={() => window.location.href = '/api-status'}
                className="text-gray-300 hover:text-white transition-colors"
              >
                API Status
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-gray-300 text-sm">Welcome, {user?.username}</span>
            <button
              onClick={handleLogout}
              className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded-lg text-sm transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>
      
      {/* Main Content */}
      <main className="p-6">
        {children}
      </main>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <AppLayout>
                    <Dashboard />
                  </AppLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/api-status"
              element={
                <ProtectedRoute>
                  <AppLayout>
                    <ApiStatus />
                  </AppLayout>
                </ProtectedRoute>
              }
            />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Router>
      </AuthProvider>
    </QueryClientProvider>
  )
}

export default App