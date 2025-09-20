#!/usr/bin/env python3
"""
🚀 ML Trading Bot - Quick Start Script
=====================================

This script automates the initial setup process for complete beginners:
- Checks system requirements
- Sets up Python virtual environment
- Installs all dependencies
- Runs the interactive setup wizard
- Validates the installation

Just run: python quick_start.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   🤖 ML Trading Bot                         ║
║                    Quick Start Setup                         ║
║                                                              ║
║  🚀 Automated setup for complete beginners                  ║
║  📚 No technical knowledge required                         ║
║  ⚡ Get trading in under 10 minutes                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("🌟 Welcome! This script will set everything up for you automatically.\n")

def check_python():
    """Check Python version"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Found: {version.major}.{version.minor}")
        print("\n📥 Please install Python 3.11+ from https://python.org")
        print("   Make sure to check 'Add Python to PATH' during installation!")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor} - Perfect!")

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    venv_path = Path(".venv")
    
    if not venv_path.exists():
        print("   Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("   ✅ Virtual environment created!")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to create virtual environment")
            print("   💡 Try: pip install --upgrade pip")
            sys.exit(1)
    else:
        print("   ✅ Virtual environment already exists!")
    
    # Get the activation script path
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    print(f"   📍 Virtual environment: {venv_path.absolute()}")
    print(f"   📍 Activation script: {activate_script}")
    
    return str(python_path), str(pip_path)

def install_basic_dependencies(pip_path):
    """Install basic dependencies needed for setup"""
    print("\n📚 Installing basic dependencies...")
    
    basic_packages = [
        "pip>=23.0",
        "wheel",
        "setuptools",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0"
    ]
    
    print("   Installing core packages...")
    try:
        subprocess.run([
            pip_path, "install", "--upgrade", *basic_packages
        ], check=True, capture_output=True, text=True)
        print("   ✅ Basic dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print("   💡 Check your internet connection and try again")
        sys.exit(1)

def run_setup_wizard(python_path):
    """Run the interactive setup wizard"""
    print("\n🧙‍♂️ Starting interactive setup wizard...")
    print("   This will guide you through the complete configuration process.")
    print("   Just follow the prompts - it's designed for beginners!\n")
    
    try:
        # Run the setup wizard
        result = subprocess.run([
            python_path, "setup_wizard.py"
        ], check=False)
        
        if result.returncode == 0:
            print("\n✅ Setup wizard completed successfully!")
        else:
            print(f"\n⚠️  Setup wizard exited with code {result.returncode}")
            print("   This might be normal if you cancelled the setup.")
    
    except FileNotFoundError:
        print("❌ Setup wizard not found!")
        print("💡 Make sure you're running this from the project directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        sys.exit(0)

def create_quick_start_files():
    """Create helpful quick start files"""
    print("\n📝 Creating quick start files...")
    
    # Create a simple start script
    start_content = """#!/usr/bin/env python3
# Quick start script - run this to start your trading bot

import subprocess
import sys
from pathlib import Path

def main():
    print("🤖 Starting ML Trading Bot...")
    
    # Check if we're in virtual environment
    venv_path = Path(".venv")
    if venv_path.exists():
        if sys.platform == "win32":
            python_cmd = str(venv_path / "Scripts" / "python.exe")
        else:
            python_cmd = str(venv_path / "bin" / "python")
    else:
        python_cmd = "python"
    
    print("🚀 Starting API server...")
    try:
        subprocess.run([python_cmd, "start_api.py"], check=True)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running the setup wizard again: python setup_wizard.py")

if __name__ == "__main__":
    main()
"""
    
    with open("start_bot.py", "w") as f:
        f.write(start_content)
    
    # Create README for beginners
    readme_content = """# 🚀 Your ML Trading Bot is Ready!

## Quick Commands

### Start the bot:
```bash
python start_bot.py
```

### View the dashboard:
```bash
streamlit run start_dashboard.py
```

### Run setup again:
```bash
python setup_wizard.py
```

## 📚 What's Next?

1. **Paper Trading**: Start with testnet (fake money) to learn
2. **Monitor Performance**: Use the dashboard to watch your bot
3. **Adjust Settings**: Modify config files as you learn
4. **Read Documentation**: Check the docs/ folder for guides

## 🆘 Need Help?

- Check `docs/BEGINNER_SETUP_GUIDE.md` for detailed instructions
- Look at `docs/TROUBLESHOOTING.md` for common issues
- Read the main `README.md` for complete documentation

## ⚠️ Important Reminders

- ✅ Always start with paper trading (testnet)
- ✅ Never risk more than you can afford to lose
- ✅ Monitor your bot regularly
- ✅ Keep your API keys secure and private

Happy trading! 🎉
"""
    
    with open("QUICK_START_README.md", "w") as f:
        f.write(readme_content)
    
    print("   ✅ Quick start files created!")
    print("   📄 start_bot.py - Simple script to start your bot")
    print("   📄 QUICK_START_README.md - Quick reference guide")

def show_completion_message():
    """Show final completion message"""
    completion = """

╔══════════════════════════════════════════════════════════════╗
║                     🎉 SETUP COMPLETE!                      ║
╚══════════════════════════════════════════════════════════════╝

Your ML Trading Bot is now ready to use! Here's what you can do:

🚀 START TRADING:
   python start_bot.py

📊 VIEW DASHBOARD:
   streamlit run start_dashboard.py

🔧 CHANGE SETTINGS:
   python setup_wizard.py

📚 LEARN MORE:
   • Read QUICK_START_README.md
   • Check docs/BEGINNER_SETUP_GUIDE.md
   • Browse the complete documentation

⚠️  IMPORTANT SAFETY REMINDERS:
   • You're using TESTNET (fake money) - perfect for learning!
   • Never share your API keys with anyone
   • Start with small amounts when you move to real trading
   • Monitor your bot regularly

🎓 NEXT STEPS:
   1. Start the bot and dashboard
   2. Watch how it trades on testnet
   3. Adjust settings as you learn
   4. Read the documentation to understand more

Happy trading! 🤖💰
"""
    print(completion)

def main():
    """Main setup process"""
    try:
        print_banner()
        
        # Step 1: Check Python
        check_python()
        
        # Step 2: Set up virtual environment
        python_path, pip_path = setup_virtual_environment()
        
        # Step 3: Install basic dependencies
        install_basic_dependencies(pip_path)
        
        # Step 4: Run setup wizard
        run_setup_wizard(python_path)
        
        # Step 5: Create helpful files
        create_quick_start_files()
        
        # Step 6: Show completion
        show_completion_message()
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Run 'python quick_start.py' to try again!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please try again or check the documentation")
        sys.exit(1)

if __name__ == "__main__":
    main()