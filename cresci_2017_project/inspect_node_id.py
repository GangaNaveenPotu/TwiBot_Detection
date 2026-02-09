
import ijson
import json

with open('D:/Capstone/cresci-2017/node.json', 'r', encoding='utf-8') as f:
    items = ijson.items(f, 'item')
    for i, item in enumerate(items):
        if i < 5:
            print(json.dumps(item, indent=2))
        else:
            break
