#!/usr/bin/env python

import pathlib
import subprocess

import click


@click.command("run")
@click.option(
    "--script", default="benchmark_circconv.py", help="Benchmark script to run"
)
@click.option(
    "--script-args",
    default="",
    help="Semi-colon-separated list of script arguments "
    "(e.g. 'dims=1,2,3;simtime=2.0')",
)
@click.option(
    "--backends",
    default="ref,ocl",
    help="Comma-separated list of backends to benchmark, with colon to specify name "
    "(e.g. 'ref:Reference,ocl:OpenCL')",
)
@click.option("--save-dir", default=None, help="Directory in which to save the results")
def run(script, script_args, backends, save_dir):
    """Run a benchmark script on one or more backends."""

    backends = parse_backend_arg(backends)
    script_args = parse_script_args(script_args)

    backends = unique_backend_names(backends)

    for key, name in backends:
        args = ["python", script, key]
        args.extend(["--save-name", make_filename(name, script)])

        if name is not None:
            args.extend(["--name", name])

        if save_dir is not None:
            args.extend(["--save-dir", save_dir])

        for arg, val in script_args:
            args.extend(["--" + arg, val])

        print(args)

        subprocess.call(args)

    _view(script, backends, save_dir)


@click.command("view")
@click.option(
    "--script",
    default="benchmark_circconv.py",
    help="Benchmark script the saved results are for",
)
@click.option(
    "--backends",
    default="ref,ocl",
    help="Comma-separated list of backends to view results for, "
    "with colon to specify name (e.g. 'ref:Reference,ocl:OpenCL')",
)
@click.option(
    "--save-dir", default=None, help="Directory in which the results are saved"
)
def view(script, backends, save_dir):
    """View benchmark results from the `run` command."""
    _view(script, backends, save_dir)


def _view(script, backends, save_dir):
    backends = parse_backend_arg(backends)
    backends = unique_backend_names(backends)

    args = ["python", "view_records.py"]

    save_dir = pathlib.Path("." if save_dir is None else save_dir)
    for key, name in backends:
        args.append(str(save_dir / make_filename(name, script)))

    subprocess.call(args)


def parse_backend_arg(backend_arg):
    if isinstance(backend_arg, str):
        backend_list = []
        for val in backend_arg.split(","):
            i = val.find(":")
            key, name = (val[:i], val[i + 1 :]) if i >= 0 else (val, None)
            backend_list.append((key, name))

        return backend_list
    else:
        return backend_arg


def parse_script_args(script_args):
    if isinstance(script_args, str):
        script_list = []
        for val in script_args.split(";"):
            i = val.find("=")
            key, name = (val[:i], val[i + 1 :]) if i >= 0 else (val, None)
            if key:
                script_list.append((key, name))

        return script_list
    else:
        return script_args


def make_filename(name, script):
    script = pathlib.Path(script)
    return f"record_{script.stem}_{name}.yml"


def unique_backend_names(backends):
    backends = list(backends)

    unique_names = set()

    for i, (key, name) in enumerate(backends):
        name = key if name is None else name
        if name in unique_names:
            for k in range(10000):
                unique_name = f"{name}{k}"
                if unique_name not in unique_names:
                    break
            else:
                raise ValueError(f"Could not find unique name for {name}")

            name = unique_name

        assert name not in unique_names
        unique_names.add(name)
        backends[i] = (key, name)

    return backends


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(view)


if __name__ == "__main__":
    cli()
