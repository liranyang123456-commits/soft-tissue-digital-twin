import json
import sys

entries = json.load(open(sys.argv[1]))
print(" ".join(e if isinstance(e, str) else e["name"] for e in entries))