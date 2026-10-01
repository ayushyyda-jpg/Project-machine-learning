"""
Minimal notebook executor (stand-in for `jupyter nbconvert --execute`).

Jupyter could not be installed in the build environment, so this script runs
every code cell of a .ipynb top to bottom in ONE shared namespace (exactly
like a fresh kernel), captures stdout/stderr, rich results (DataFrame HTML)
and matplotlib figures, and writes them back into the notebook as outputs.
It stops at the first error and exits non-zero.

On a machine with Jupyter, the equivalent is:
    jupyter nbconvert --to notebook --execute --inplace housing_price_prediction.ipynb

Usage: python tools/run_notebook.py housing_price_prediction.ipynb
"""
import ast, base64, contextlib, io, json, os, sys, traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rich_output(value):
    data = {"text/plain": repr(value)}
    if hasattr(value, "_repr_html_"):
        html = value._repr_html_()
        if html:
            data["text/html"] = html
    return data


def flush_figures(outputs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                                 "text/plain": f"<Figure size {fig.get_size_inches()[0]*100:.0f}x{fig.get_size_inches()[1]*100:.0f} with {len(fig.axes)} Axes>"}})
    plt.close("all")


def main(path):
    nb = json.load(open(path))
    os.chdir(os.path.dirname(os.path.abspath(path)) or ".")
    sys.path.insert(0, os.getcwd())
    ns = {"__name__": "__main__"}
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []

        class Stream(io.TextIOBase):
            # Writes go into the output list in order, like a Jupyter kernel
            def __init__(self, name): self.name = name
            def write(self, text):
                if not text: return 0
                if self.name == "stderr" and not text.strip(): return len(text)
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == self.name:
                    outputs[-1]["text"] += text
                else:
                    outputs.append({"output_type": "stream", "name": self.name, "text": text})
                return len(text)

        def display(*objs):
            for o in objs:
                outputs.append({"output_type": "display_data", "metadata": {}, "data": rich_output(o)})
        ns["display"] = display
        plt.show = lambda *a, **k: flush_figures(outputs)

        src = "".join(cell["source"])
        out, err = Stream("stdout"), Stream("stderr")
        tree = ast.parse(src)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                result = eval(compile(ast.Expression(last.value), f"<cell {count}>", "eval"), ns) if last else None
        except Exception:
            tb = traceback.format_exc()
            print(f"ERROR in cell {count}:\n{src}\n{tb}", file=sys.stderr)
            sys.exit(1)
        flush_figures(outputs)
        if result is not None and not (last and src.rstrip().endswith(";")):
            outputs.append({"output_type": "execute_result", "execution_count": count,
                            "metadata": {}, "data": rich_output(result)})
        cell["outputs"] = outputs
        cell["execution_count"] = count
        print(f"cell {count} ok", flush=True)
    json.dump(nb, open(path, "w"), indent=1)
    print(f"Executed {count} code cells without errors.")


if __name__ == "__main__":
    main(sys.argv[1])
