import re
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'\{Colors\.\w+\}', '', content)
content = re.sub(r'f"\{Colors\.\w+\}', 'f"', content)
content = re.sub(r'\[ Colors\.\w+\]', '[ ]', content)

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')