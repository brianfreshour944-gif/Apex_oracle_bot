import os
import re

# Check actual logging calls that might leak secrets
files_to_check = [
    'src/bot.py',
    'src/risk.py', 
    'src/exchange.py',
    'src/committee/committee.py',
    'src/telegram_alerts.py'
]

# Look for actual logging of sensitive values
log_patterns = [
    (r'logger\.(info|error|debug|warning|critical)\(f?["\'].*\{.*[Aa][Pp][Ii]_?[Kk][Ee][Yy].*\}', 'API key in f-string log'),
    (r'logger\.(info|error|debug|warning|critical)\(f?["\'].*\{.*[Ss][Ee][Cc][Rr][Ee][Tt].*\}', 'Secret in f-string log'),
    (r'logger\.(info|error|debug|warning|critical)\(f?["\'].*\{.*[Tt][Oo][Kk][Ee][Nn].*\}', 'Token in f-string log'),
    (r'send_telegram_alert\(f?["\'].*\{.*[Aa][Pp][Ii]_?[Kk][Ee][Yy].*\}', 'API key in telegram'),
    (r'send_telegram_alert\(f?["\'].*\{.*[Ss][Ee][Cc][Rr][Ee][Tt].*\}', 'Secret in telegram'),
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        for pattern, desc in log_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                print('FILE: ' + filepath)
                for m in matches[:5]:
                    line_num = content[:m.start()].count('\n') + 1
                    line = content[m.start():m.end()]
                    print('  Line ' + str(line_num) + ': ' + desc)
                    print('    ' + line[:200])
                print()

# Also check for hardcoded secrets in config
print()
print("=== Checking for hardcoded secrets in config ===")
config_files = ['src/config.py', '.env.example', '.env']
for cf in config_files:
    if os.path.exists(cf):
        with open(cf, 'r') as f:
            content = f.read()
        # Look for actual secret values (not just field definitions)
        secret_patterns = [
            (r'(API_KEY|SECRET_KEY|TOKEN|PASSWORD)\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded secret value'),
        ]
        for pattern, desc in secret_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                print('FILE: ' + cf)
                for m in matches[:5]:
                    line_num = content[:m.start()].count('\n') + 1
                    line = content[m.start():m.end()]
                    print('  Line ' + str(line_num) + ': ' + desc)
                    print('    ' + line[:200])
                print()