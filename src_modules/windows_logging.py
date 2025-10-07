"""
Windows-Compatible Logging Configuration
========================================

Handles emoji and Unicode characters for Windows deployment.
"""

import logging
import sys
import os

def setup_windows_logging():
    """Setup logging that works on Windows with Unicode support"""
    
    # Set UTF-8 encoding for stdout/stderr if possible
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, TypeError):
        # Fallback for older Python versions
        pass
    
    # Windows-compatible log format (no emojis in file logs)
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console format (attempt emoji, fallback to text)
    try:
        # Test if console can handle emojis
        test_output = "🔧 Debug"
        sys.stdout.buffer.write(test_output.encode('utf-8'))
        sys.stdout.buffer.flush()
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    except (UnicodeEncodeError, AttributeError):
        console_format = '%(asctime)s - %(name)s - %(levelname)s - [DEBUG] %(message)s'
    
    return file_format, console_format

def create_windows_safe_message(message):
    """Convert emoji messages to Windows-safe equivalents"""
    emoji_map = {
        '🔧': '[DEBUG]',
        '✅': '[SUCCESS]', 
        '🚨': '[WARNING]',
        '📧': '[API]',
        '🎯': '[TARGET]',
        '🔄': '[PROCESS]',
        '🛡️': '[SAFETY]',
        '⚠️': '[ALERT]',
        '❌': '[ERROR]',
        '🚀': '[START]',
        '🔍': '[SEARCH]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_map.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    return safe_message