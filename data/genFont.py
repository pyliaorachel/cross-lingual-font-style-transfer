# python2

import fontforge
F = fontforge.open("PingFang.ttc")
for name in F:
    filename = "pingfang/" + name + ".png"
    # print name
    F[name].export(filename, 128)
    # F[name].export(filename, 600)     # set height to 600 pixels
