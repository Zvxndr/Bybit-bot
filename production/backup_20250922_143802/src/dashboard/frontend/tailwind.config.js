/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'crypto-dark': '#0D1421',
        'crypto-darker': '#050C14',
        'crypto-blue': '#1E40AF',
        'crypto-green': '#10B981',
        'crypto-red': '#EF4444',
        'crypto-yellow': '#F59E0B',
        'crypto-gray': '#374151',
        'crypto-light-gray': '#6B7280',
      },
      fontFamily: {
        'mono': ['Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      },
      boxShadow: {
        'crypto': '0 4px 14px 0 rgba(0, 118, 255, 0.39)',
        'crypto-lg': '0 10px 25px 0 rgba(0, 118, 255, 0.3)',
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}