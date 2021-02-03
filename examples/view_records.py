#!/usr/bin/env python
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import yaml

by_recfile = OrderedDict()
by_name = OrderedDict()

# the units used on the x-axis, either "dim" (for dimensions) or "neurons"
x_units = None

benchmarks = set()
for recfile in sys.argv[1:]:
    with open(recfile, "r") as fh:
        records = yaml.load(fh, Loader=yaml.Loader)
        by_recfile[recfile] = records

    for rec in records:
        benchmarks.add(rec["benchmark"])

for recfile, records in by_recfile.items():
    for rec in records:
        rec["filename"] = recfile
        name = rec["name"]
        if len(benchmarks) > 1:
            name = "%s %s" % (rec["benchmark"], name)

        by_name.setdefault(name, []).append(rec)

        if x_units is None:
            x_units = "dim" if "dim" in rec else "neurons"

        if x_units not in rec:
            raise ValueError(
                "Trying to use %r for the x-axis, but one of the records in "
                "%r does not have %r" % (x_units, recfile, x_units)
            )

for name, recs in by_name.items():
    print(name.strip())
    if name.strip() == "Tahiti":
        name = "ATI Radeon HD 7970"
    if name.strip() == "ref":
        name = "NumPy Reference"
    if name.strip().startswith("Tesla"):
        name = "NVidia " + name.strip()
    if "5540" in name.strip():
        name = "Intel Xeon E5540 @ 2.53GHz"
    if "2620" in name:
        name = "Intel Xeon E5-2620 @ 2.00GHz"
    if "Core" in name.strip():
        name = "Intel Core i7-3770 @ 3.40GHz"

    oks = [rec for rec in recs if rec["status"] == "ok"]
    x = [rec[x_units] for rec in oks]
    buildtimes = [rec.get("buildtime", 0) for rec in oks]
    warmtimes = [rec.get("warmtime", 0) for rec in oks]
    runtimes = [rec.get("runtime", 0) for rec in oks]
    tottimes = [sum(t) for t in zip(buildtimes, warmtimes, runtimes)]
    filenames = [rec["filename"] for rec in oks]

    for xx, rt, fname in zip(x, runtimes, filenames):
        print("  %4d, %8.3f, %s" % (xx, rt, fname))

    plt.plot(x, runtimes, ".-", markersize=30, label=name.strip() + " run")
    # plt.plot(x, buildtimes, ".-", markersize=30, label=name.strip() + " build")
    # plt.plot(x, warmtimes, ".-", markersize=30, label=name.strip() + " warm")
    # plt.plot(x, tottimes, ".-", markersize=30, label=name.strip() + " tot")
    # plt.yscale("log")

plt.xlabel("n. dimensions convolved" if x_units == "dim" else "n. neurons")
plt.ylabel("simulation time (seconds)")
# plt.ylim(0, 20)
plt.legend(loc=2)
plt.show()
