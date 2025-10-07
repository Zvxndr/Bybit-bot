/**
 * Index Page Specific JavaScript
 * Additional functionality for the main dashboard page
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Index page script loaded');
    
    // Page-specific initialization
    initializeIndexPage();
});

function initializeIndexPage() {
    // Add any index page specific functionality here
    console.log('🏠 Index page initialized');
    
    // Example: Add welcome message or specific dashboard features
    updateWelcomeMessage();
    
    // Initialize any index-specific widgets
    initializeQuickActions();
}

function updateWelcomeMessage() {
    const welcomeElement = document.querySelector('.welcome-message');
    if (welcomeElement) {
        const currentHour = new Date().getHours();
        let greeting = 'Good day';
        
        if (currentHour < 12) {
            greeting = 'Good morning';
        } else if (currentHour < 17) {
            greeting = 'Good afternoon';
        } else {
            greeting = 'Good evening';
        }
        
        welcomeElement.textContent = `${greeting}, Trader!`;
    }
}

function initializeQuickActions() {
    // Add quick action buttons functionality
    const quickActionButtons = document.querySelectorAll('.quick-action-btn');
    quickActionButtons.forEach(button => {
        button.addEventListener('click', handleQuickAction);
    });
}

function handleQuickAction(event) {
    const action = event.target.dataset.action;
    console.log('🎯 Quick action triggered:', action);
    
    switch(action) {
        case 'refresh':
            if (window.app) {
                window.app.refreshSystemData();
            }
            break;
        case 'portfolio':
            // Navigate to portfolio page
            window.location.href = '/portfolio.html';
            break;
        case 'settings':
            // Navigate to settings page
            window.location.href = '/settings.html';
            break;
        default:
            console.log('Unknown action:', action);
    }
}

// Export functions for global access
window.IndexPage = {
    initializeIndexPage,
    updateWelcomeMessage,
    initializeQuickActions,
    handleQuickAction
};