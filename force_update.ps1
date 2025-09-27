# DigitalOcean Frontend Update Script - PowerShell Version
# Forces fresh deployment with cache busting

Write-Host "🚀 Forcing DigitalOcean frontend update..." -ForegroundColor Green

# Add cache-busting timestamp
$timestamp = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Write-Host "📅 Update timestamp: $timestamp" -ForegroundColor Yellow

# Add version bump to force container rebuild
Write-Host "🔄 Bumping app version for cache busting..." -ForegroundColor Blue

# Create deployment trigger file
"DEPLOYMENT_VERSION=$timestamp" | Out-File -FilePath ".deploy_version" -Encoding utf8

# Force git push with all recent changes
Write-Host "📤 Pushing changes to DigitalOcean..." -ForegroundColor Cyan
git add .
git commit -m "🔄 FORCE UPDATE: Frontend refresh + cache bust $timestamp" --allow-empty
git push origin main --force-with-lease

Write-Host "✅ Deployment pushed! DigitalOcean should rebuild container in ~2-3 minutes" -ForegroundColor Green
Write-Host "🔗 Check your app at: https://auto-wealth-[your-app].ondigitalocean.app" -ForegroundColor White
Write-Host "🔄 If changes don't appear, check DigitalOcean App Platform logs" -ForegroundColor Yellow