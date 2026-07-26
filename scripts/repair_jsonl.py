"""More robust repair: tokenize as JSON stream using raw_decode."""
import json, sys
inp, out = sys.argv[1], sys.argv[2]
text = open(inp).read()
json_decoder = json.JSONDecoder()
i = 0
n = 0
with open(out, "w") as g:
    while i < len(text):
        # skip whitespace
        while i < len(text) and text[i] in ' \t\n\r':
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = json_decoder.raw_decode(text, i)
            g.write(json.dumps(obj) + "\n")
            n += 1
            i = end
        except json.JSONDecodeError:
            # find next '{' and skip
            next_brace = text.find('{', i+1)
            if next_brace == -1:
                break
            i = next_brace
print(f"{inp}: out={n}")
