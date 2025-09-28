@echo off
echo Creating Windows-safe version of main.py...

python -c "
import re

# Read main.py
with open('src/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define emoji replacements
replacements = {
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
    '🔍': '[SEARCH]',
    '⏸️': '[PAUSED]',
    '🛑': '[STOP]',
    '📡': '[SIGNAL]',
    '🌐': '[WEB]',
    '🧹': '[CLEANUP]'
}

# Replace all emojis
for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

# Save updated file
with open('src/main_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed version saved as src/main_fixed.py')
"

echo Done!