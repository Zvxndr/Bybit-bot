// Simple auth service for demo purposes
// In production, this would integrate with real authentication backend

class AuthService {
  constructor() {
    this.isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  }

  login(credentials) {
    return new Promise((resolve, reject) => {
      // Mock authentication
      setTimeout(() => {
        if (credentials.username === 'admin' && credentials.password === 'password') {
          this.isAuthenticated = true;
          localStorage.setItem('isAuthenticated', 'true');
          resolve({ success: true, user: { username: 'admin', role: 'admin' } });
        } else {
          reject(new Error('Invalid credentials'));
        }
      }, 1000);
    });
  }

  logout() {
    this.isAuthenticated = false;
    localStorage.removeItem('isAuthenticated');
    return Promise.resolve();
  }

  isLoggedIn() {
    return this.isAuthenticated;
  }

  getUser() {
    if (this.isAuthenticated) {
      return {
        username: 'admin',
        role: 'admin',
        permissions: ['read', 'write', 'admin']
      };
    }
    return null;
  }
}

export const authService = new AuthService();
export default authService;