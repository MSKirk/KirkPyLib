from __future__ import division, print_function, absolute_import
import json
import matplotlib


path = '/Users/mskirk/py/lib/ColorMap/newcm2.jscm'

with open(path) as f:
    data = json.loads(f.read())
    name = data["name"]
    colors = data["colors"]
    colors = [colors[i:i + 6] for i in range(0, len(colors), 6)]
    colors = [[int(c[2 * i:2 * i + 2], 16) / 255 for i in range(3)] for c in colors]
cmap = matplotlib.colors.ListedColormap(colors, name)

