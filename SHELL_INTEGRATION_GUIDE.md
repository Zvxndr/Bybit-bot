# VS Code Shell Integration Setup Guide
# =====================================

## Current Status: ✅ ALREADY ENABLED

Your VS Code shell integration is already active! Here's how to optimize it:

## 🔧 Enhanced PowerShell Profile Setup

### 1. Check Your Current Profile
```powershell
# Check if profile exists
Test-Path $PROFILE

# View current profile location
$PROFILE
```

### 2. Create/Update PowerShell Profile
Create or edit your PowerShell profile to enhance VS Code integration:

```powershell
# Create profile if it doesn't exist
if (!(Test-Path -Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force
}

# Edit the profile
notepad $PROFILE
```

### 3. Add to Your PowerShell Profile
Add these lines to your PowerShell profile for better VS Code integration:

```powershell
# VS Code Shell Integration Enhancements
if ($env:TERM_PROGRAM -eq "vscode") {
    # Set better colors for VS Code terminal
    $Host.UI.RawUI.BackgroundColor = "Black"
    $Host.UI.RawUI.ForegroundColor = "White"
    
    # Enable better command prediction
    Set-PSReadLineOption -PredictionSource History
    Set-PSReadLineOption -PredictionViewStyle ListView
    
    # Enhanced key bindings for VS Code
    Set-PSReadLineKeyHandler -Key Tab -Function Complete
    Set-PSReadLineKeyHandler -Key Ctrl+d -Function DeleteChar
    Set-PSReadLineKeyHandler -Key Ctrl+w -Function BackwardDeleteWord
    
    # Better history navigation
    Set-PSReadLineKeyHandler -Key UpArrow -Function HistorySearchBackward
    Set-PSReadLineKeyHandler -Key DownArrow -Function HistorySearchForward
    
    Write-Host "🚀 VS Code PowerShell integration loaded!" -ForegroundColor Green
}

# Useful aliases for development
Set-Alias -Name ll -Value Get-ChildItem
Set-Alias -Name grep -Value Select-String
Set-Alias -Name which -Value Get-Command

# Function to quickly navigate to project root
function cdroot { 
    while (!(Test-Path ".git") -and (Get-Location).Path -ne (Get-Location).Root) {
        Set-Location ..
    }
}

# Function to show git status with colors
function gs { git status --short }
function ga { git add $args }
function gc { git commit -m $args }
function gp { git push }

# Quick Python virtual environment activation
function venv {
    if (Test-Path ".venv/Scripts/Activate.ps1") {
        .\.venv\Scripts\Activate.ps1
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } elseif (Test-Path "venv/Scripts/Activate.ps1") {
        .\venv\Scripts\Activate.ps1
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "❌ No virtual environment found" -ForegroundColor Red
    }
}
```

## 🎯 VS Code Settings for Better Shell Integration

Add these to your VS Code settings.json:

```json
{
    // Terminal settings
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.enablePersistentSessions": true,
    "terminal.integrated.persistentSessionReviveProcess": "onExit",
    
    // Shell integration features
    "terminal.integrated.shellIntegration.enabled": true,
    "terminal.integrated.shellIntegration.showWelcome": false,
    "terminal.integrated.shellIntegration.decorationsEnabled": "both",
    
    // Command history and suggestions
    "terminal.integrated.commandsToSkipShell": [],
    "terminal.integrated.allowChords": false,
    
    // Better terminal appearance
    "terminal.integrated.fontSize": 14,
    "terminal.integrated.fontFamily": "Cascadia Code, Consolas, monospace",
    "terminal.integrated.cursorBlinking": true,
    "terminal.integrated.cursorStyle": "line",
    
    // Right-click behavior
    "terminal.integrated.rightClickBehavior": "default",
    
    // Auto-scroll and selection
    "terminal.integrated.scrollback": 10000,
    "terminal.integrated.fastScrollSensitivity": 5
}
```

## 🚀 Shell Integration Features You Can Use

### 1. Command Decorations
- Success/failure indicators next to commands
- Quick access to command output
- Easy re-running of commands

### 2. Smart Working Directory Detection
- VS Code knows your current directory
- Better file path suggestions
- Context-aware operations

### 3. Enhanced Terminal-to-Editor Integration
- Ctrl+Click on file paths to open them
- Quick problem detection and navigation
- Better IntelliSense based on terminal context

### 4. Command Palette Integration
- Run terminal commands from Command Palette
- Quick terminal focus shortcuts
- Better terminal management

## 🔍 Verify Shell Integration Features

Run these commands to test shell integration:

```powershell
# Test 1: Command with output (should show decorations)
Get-ChildItem

# Test 2: Command that creates files (VS Code should detect changes)
New-Item -ItemType File -Name "test_integration.txt"
Remove-Item "test_integration.txt"

# Test 3: Navigate directories (working directory should update in VS Code)
cd src
cd ..

# Test 4: Git operations (should show in Source Control)
git status
```

## 🎨 Optional: Install Oh My Posh for Better Prompts

For an enhanced terminal experience:

```powershell
# Install Oh My Posh
winget install JanDeDobbeleer.OhMyPosh

# Install a Nerd Font (recommended: CascadiaCode Nerd Font)
# Download from: https://github.com/ryanoasis/nerd-fonts/releases

# Add to your PowerShell profile:
oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH\clean-detailed.omp.json" | Invoke-Expression
```

## ✅ Your Current Status

- ✅ Shell integration is ACTIVE
- ✅ VS Code can communicate with PowerShell
- ✅ Command decorations should be working
- ✅ Working directory sync is enabled

## 🛠️ Troubleshooting

If shell integration isn't working properly:

1. **Restart VS Code** - Sometimes needed after profile changes
2. **Check Terminal Profile** - Ensure PowerShell is the default
3. **Update VS Code** - Latest versions have better shell integration
4. **Check Extensions** - Some extensions might interfere

## 🎯 Next Steps

1. Update your PowerShell profile with the enhancements above
2. Restart VS Code to apply changes
3. Test the integration features
4. Optionally install Oh My Posh for better visuals

Your shell integration is already working - these steps will just make it even better! 🚀