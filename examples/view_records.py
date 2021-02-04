#!/usr/bin/env python

import click
import matplotlib.pyplot as plt
import yaml


@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--x-units",
    default=None,
    help="Units to use for the x-axis: 'dim' (dimensions), 'synapses', or 'neurons'. "
    "By default, will choose a value based on the benchmark record. All records must "
    "use the same x-axis units.",
)
def main(files, x_units):
    """View benchmarking results.

    FILES is one or more benchmarking .pkl record files to view.
    """

    by_recfile = load_records(files)

    x_units = get_x_units(by_recfile, x_units)
    by_name = sort_records(by_recfile, x_units)

    for name, recs in by_name.items():
        name = name.strip()

        if name == "Tahiti":
            name = "ATI Radeon HD 7970"
        if name == "ref":
            name = "NumPy Reference"
        if name.startswith("Tesla"):
            name = "NVidia " + name
        if "5540" in name:
            name = "Intel Xeon E5540 @ 2.53GHz"
        if "2620" in name:
            name = "Intel Xeon E5-2620 @ 2.00GHz"
        if "Core" in name:
            name = "Intel Core i7-3770 @ 3.40GHz"

        oks = [rec for rec in recs if rec["status"] == "ok"]
        x = [rec[x_units] for rec in oks]
        # buildtimes = [rec.get("buildtime", 0) for rec in oks]
        # warmtimes = [rec.get("warmtime", 0) for rec in oks]
        runtimes = [rec.get("runtime", 0) for rec in oks]
        # tottimes = [sum(t) for t in zip(buildtimes, warmtimes, runtimes)]
        filenames = [rec["filename"] for rec in oks]

        print(name)
        for xx, rt, fname in zip(x, runtimes, filenames):
            print("  %4d, %8.3f, %s" % (xx, rt, fname))

        plt.plot(x, runtimes, ".-", markersize=30, label=name + " run")
        # plt.plot(x, buildtimes, ".-", markersize=30, label=name + " build")
        # plt.plot(x, warmtimes, ".-", markersize=30, label=name + " warm")
        # plt.plot(x, tottimes, ".-", markersize=30, label=name + " tot")
        # plt.yscale("log")
        # plt.xscale("log")

    plt.xlabel("n. dimensions convolved" if x_units == "dim" else "n. " + x_units)
    plt.ylabel("simulation time (seconds)")
    # plt.ylim(0, 20)
    plt.legend(loc=2)
    plt.show()


def load_records(files):
    by_recfile = {}

    for recfile in files:
        with open(recfile, "r") as fh:
            records = yaml.load(fh, Loader=yaml.Loader)
            by_recfile[recfile] = records

    if sum(len(records) for records in by_recfile.values()) == 0:
        raise ValueError("Could not find any records in provided files")

    return by_recfile


def get_x_units(by_recfile, x_units):
    if x_units is None:
        for records in by_recfile.values():
            for rec in records:
                x_units = (
                    "dim"
                    if "dim" in rec
                    else "synapses"
                    if "synapses" in rec
                    else "neurons"
                )
                break

    assert x_units in ("dim", "synapses", "neurons")

    return x_units


def sort_records(by_recfile, x_units):
    benchmarks = set()
    by_name = {}

    for recfile, records in by_recfile.items():
        for rec in records:
            benchmarks.add(rec["benchmark"])
            rec["filename"] = recfile
            name = rec["name"]
            if len(benchmarks) > 1:
                name = "%s %s" % (rec["benchmark"], name)

            by_name.setdefault(name, []).append(rec)

            if x_units not in rec:
                raise ValueError(
                    "Trying to use %r for the x-axis, but one of the records in "
                    "%r does not have %r" % (x_units, recfile, x_units)
                )

    return by_name


if __name__ == "__main__":
    main()
