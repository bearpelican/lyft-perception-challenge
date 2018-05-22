import sys
import json

file = sys.argv[-1]
with open("/home/workspace/ashaw/results.json", "r") as f:
    print(json.dumps(json.loads(f.read())))
#     print(f.read())
    
#     f.write("hello\n")
# #     f.write(sys.argv[-1])
#     f.write(str(sys.argv))
import time
time.sleep(40)