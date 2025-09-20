"""
Environment-specific configuration files for production deployment.

This directory contains configuration files for different deployment environments:
- development.yaml: Development environment settings
- testing.yaml: Testing environment settings  
- staging.yaml: Staging environment settings
- production.yaml: Production environment settings
- secrets.yaml: Encrypted secrets (generated, not committed to version control)

Configuration hierarchy (in priority order):
1. Environment variables
2. Environment-specific YAML files
3. Secrets file
4. Default values in production.py

Security Notes:
- secrets.yaml should never be committed to version control
- Use encrypted values with 'enc:' prefix in secrets.yaml
- Store master encryption key in TRADING_BOT_MASTER_KEY environment variable
- Set appropriate file permissions (600) for secrets files
"""