import subprocess
import pytest
import os
from datetime import datetime
import json

today = datetime.now().strftime("%Y-%m-%d")

new_notebook = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
new_notebook = json.dumps(new_notebook, indent=4)


notebook_with_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["2.302585092994046\n"],
                }
            ],
            "source": ["import math \n", "print(math.log(10))"],
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
notebook_with_content = json.dumps(notebook_with_content, indent=4)


cell_front_matter = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<!-- ---\n",
        "title: <required-title>\n",
        f"date: {today}\n",
        "downloads: true\n",
        "weight: <required-weight> See: https://github.com/pytorch-ignite/examples/issues/30\n",
        "summary: <use either this or the `<!--more-->` tag below to provide summary for this notebook, "
        "and delete the other>\n"
        "tags:\n",
        "  - <required-tag>\n",
        "--- -->\n",
        "\n",
        "# title-placeholder\n",
        "\n",
        "<If you are not using the `summary` variable above, use this space to "
        "provide a summary for this notebook.>\n",
        "<Otherwise, delete the `<!--more-->` below.>",
        "\n",
        "<!--more-->",
    ],
}


cwd = os.getcwd()


@pytest.mark.parametrize("name", ["dummy_notebook", "dummy_notebook.ipynb"])
def test_new_notebook_creation(name):
    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")

    if not name.endswith(".ipynb"):
        name = name + ".ipynb"

    assert output == f"Generated {os.path.join(cwd, name)}\n"
    subprocess.call(["rm", "-rf", name])


def test_existing_blank_notebook():
    name = "dummy_notebook_empty.ipynb"
    with open(name, "w") as f:
        f.write(new_notebook)

    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")
    assert output == f"Added frontmatter to {os.path.join(cwd, name)}\n"
    subprocess.call(["rm", "-rf", name])


def test_existing_non_empty_notebook():
    name = "dummy_notebook_with_content.ipynb"
    with open(name, "w") as f:
        f.write(notebook_with_content)

    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")

    assert output == f"Added frontmatter to {os.path.join(cwd, name)}\n"

    # Check to make sure its added as the first cell
    with open(name) as fp:
        content = json.load(fp)
    assert content["cells"][0] == cell_front_matter
    subprocess.call(["rm", "-rf", name])


@pytest.mark.parametrize("name", ["dummy_notebook_empty", "dummy_notebook_empty.ipynb"])
def test_front_matter_multiple_times(name):
    # This will create a notebook with frontmatter
    _ = subprocess.check_output(["python", "../generate.py", name])
    # Second call should not add frontmatter again
    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")

    if not name.endswith(".ipynb"):
        name = name + ".ipynb"

    assert (
        output
        == f"Frontmatter cell already exists in {os.path.join(cwd, name)}. Exiting\n"
    )

    # Check to make sure only added once.
    with open(name) as fp:
        content = json.load(fp)

    if len(content["cells"]) > 1:
        assert content["cells"][0] != content["cells"][1]
    subprocess.call(["rm", "-rf", name])