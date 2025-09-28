# PowerShell script to fix all emoji issues
$sourceFile = "src/main.py"
$backupFile = "src/main_backup.py"

# Create backup
Copy-Item $sourceFile $backupFile

# Read the content
$content = Get-Content $sourceFile -Raw -Encoding UTF8

# Replace all emojis with Windows-safe equivalents
$replacements = @{
    "🔧" = "[DEBUG]"
    "✅" = "[SUCCESS]"
    "🚨" = "[WARNING]"
    "📧" = "[API]"
    "🎯" = "[TARGET]"
    "🔄" = "[PROCESS]"
    "🛡️" = "[SAFETY]"
    "⚠️" = "[ALERT]"
    "❌" = "[ERROR]"
    "🚀" = "[START]"
    "🔍" = "[SEARCH]"
    "⏸️" = "[PAUSED]"
    "🛑" = "[STOP]"
    "📡" = "[SIGNAL]"
    "🌐" = "[WEB]"
    "🧹" = "[CLEANUP]"
}

foreach ($emoji in $replacements.Keys) {
    $content = $content.Replace($emoji, $replacements[$emoji])
}

# Save the updated content
$content | Out-File $sourceFile -Encoding UTF8 -NoNewline

Write-Host "✅ Emojis replaced successfully!"
Write-Host "📝 Backup saved as: $backupFile"