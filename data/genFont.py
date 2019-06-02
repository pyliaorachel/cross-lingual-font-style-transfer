# python2

import fontforge
F = fontforge.open("PingFang.ttc")
for name in F:
    filename = "pingfang/" + name + ".png"
    # print name
    F[name].export(filename, 128)

    # x = F[21121]
    # x = F["Identity.21121"]
    # y = x.unicode # y = 29508
    # F[name].export(filename, 600)     # set height to 600 pixels
