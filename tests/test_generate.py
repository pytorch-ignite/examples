import subprocess
import pytest
import os
from datetime import datetime
import json
today = datetime.now().strftime('%Y-%m-%d')

cell_front_matter ={
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


@pytest.mark.parametrize("name", ["test_notebooks/dummy_notebook", "test_notebooks/dummy_notebook.ipynb"])
def test_new_notebook_creation(name):
    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")

    if not name.endswith(".ipynb"):
        name = name + ".ipynb"

    assert output == f"Generated {os.path.join(cwd, name)}\n"
    subprocess.call(["rm", "-rf", name])


def test_existing_blank_notebook():
    name = "test_notebooks/dummy_notebook_empty.ipynb"
    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")
    assert output == f"Added frontmatter to {os.path.join(cwd, name)}\n"


def test_existing_non_empty_notebook():
    name = "test_notebooks/dummy_notebook_with_content.ipynb"
    output = subprocess.check_output(["python", "../generate.py", name]).decode("utf-8")

    assert output == f"Added frontmatter to {os.path.join(cwd, name)}\n"

    # Check to make sure its added as the first cell
    with open(name) as fp:
        content = json.load(fp)
    assert content["cells"][0] == cell_front_matter


@pytest.mark.parametrize("name", ["test_notebooks/dummy_notebook_empty", "test_notebooks/dummy_notebook_empty.ipynb"])
def test_front_matter_multiple_times(name):

    _ = subprocess.check_output(["python", "../generate.py", name])
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
