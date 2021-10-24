# Generate plain notebooks with the required frontmatter defined

# Usage:
#   $ python generate.py <notebook-names>... [-h]
#
#   Generate plain notebooks with the required frontmatter defined.
#
# Positional arguments:
#   notebook_names        Notebooks to generate
#
# Options:
#   -h, --help            show this help message and exit
#
# Example:
#   python generate.py {data-iterator,fastai-lr-finder,gradient-accumulation,installation}

import json
import os
from argparse import ArgumentParser
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')

notebook = {
  'nbformat': 4,
  'nbformat_minor': 0,
  'metadata': {
    'kernelspec': {
      'display_name': 'Python 3',
      'name': 'python3',
    },
    'accelerator': 'GPU',
  },
  'cells': [
    {
      'cell_type': 'markdown',
      'metadata': {},
      'source': [
        '<!-- ---\n',
        'title: <required-title>\n',
        f'date: {today}\n',
        'downloads: true\n',
        'sidebar: true\n',
        'weight: <required-weight> See: https://github.com/pytorch-ignite/examples/issues/30\n',
        'summary: <use either this or the `<!--more-->` tag below to provide summary for this notebook, '
        'and delete the other>\n'
        'tags:\n',
        '  - <required-tag>\n',
        '--- -->\n',
        '\n',
        '# title-placeholder\n',
        '\n',
        '<If you are not using the `summary` variable above, use this space to '
        'provide a summary for this notebook.>\n',
        '\n',
        '<!--more -->',
      ]
    }
  ]
}

if __name__ == '__main__':
  cwd = os.getcwd()
  parser = ArgumentParser(
    'generate',
    '$ python generate.py <notebook-names>... [-h]',
    'Generate plain notebooks with the required frontmatter defined.'
  )
  parser.add_argument(
    'notebook_names',
    help='Notebooks to generate',
    nargs='+',
  )
  args = parser.parse_args()
  for name in args.notebook_names:
    name = name + ".ipynb"
    with open(name, 'w') as fp:
      json.dump(notebook, fp, indent=2)
      print(f'Generated {os.path.join(cwd, name)}')
