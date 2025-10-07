#!/bin/bash
set -e

echo "🚀 CONTAINER STARTUP SCRIPT"
echo "📁 Working Directory: $(pwd)"
echo "🐍 Python Path: $(which python)"
echo "📂 Python Version: $(python --version)"
echo "📋 Files in /app:"
ls -la /app/

echo "🎯 EXECUTING: python main.py"
exec python main.py "$@"