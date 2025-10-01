#!/bin/bash

# DigitalOcean Frontend Update Script
# Forces fresh deployment with cache busting

echo "🚀 Forcing DigitalOcean frontend update..."

# Add cache-busting timestamp
timestamp=$(date +%s)
echo "📅 Update timestamp: $timestamp"

# Add version bump to force container rebuild
echo "🔄 Bumping app version for cache busting..."

# Create deployment trigger file
echo "DEPLOYMENT_VERSION=$timestamp" > .deploy_version

# Force git push with all recent changes
echo "📤 Pushing changes to DigitalOcean..."
git add .
git commit -m "🔄 FORCE UPDATE: Frontend refresh + cache bust $timestamp" --allow-empty
git push origin main --force-with-lease

echo "✅ Deployment pushed! DigitalOcean should rebuild container in ~2-3 minutes"
echo "🔗 Check your app at: https://auto-wealth-[your-app].ondigitalocean.app"
echo "🔄 If changes don't appear, check DigitalOcean App Platform logs"