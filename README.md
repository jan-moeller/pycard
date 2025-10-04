# pycard

A python playing cards template engine.

[![Testing](https://github.com/jan-moeller/pycard/actions/workflows/.testing.yml/badge.svg)](https://github.com/jan-moeller/pycard/actions/workflows/.testing.yml)

![](https://github.com/jan-moeller/pycard/raw/main/example/out/0.svg)
![](https://github.com/jan-moeller/pycard/raw/main/example/out/1.svg)

## Usage

To use pycard, you need an SVG template file, and a yaml card definitions file. See the example/
directory to see both files used to generate the above cards.

### SVG Templates

The SVG template simply uses SVG syntax with [PyForma](https://pypi.org/project/pyforma/)
template syntax. It defines the general look of the cards, where texts are, and so on.

In addition to the PyForma default functions, the following functions are available:

- `read_text`: Accepts a file path and reads the content of the file into a string.
- `read_bytes`: Accepts a file path and reads the content of the file into a bytestring.
- `mimetype`: Accepts a file path and returns the mimetype for it.
- `b64encode`: Accepts a bytestring and turns it into a base64-encoded string.
- `b64decode`: Accepts a base64-encoded string and turns it into a bytestring.
- `bytes2hex`: Accepts a bytestring and turns it into a hexadecimal string representation.
- `hex2bytes`: Accepts a hexadecimal string representation and turns it into a bytestring.

### Card Definitions

The cards are defined using a simple, composable yaml format. In its simplest form, the yaml
can contain a list of dictionaries, mapping variable names to the value this variable should
take for that card.

#### Categories

To not have to repeat every variable for each card, it is possible to create a category of
cards wherever a card could be placed in the yaml, by giving that list entry the `!category`
tag. Categories may have two keys, `variables` and `cards`. The variables key contains a
dictionary of variable names to their values, just like individual cards do, but they are
applied by default to all cards listed in the `cards` section. Categories can be nested, i.e.
the `cards` section of a category can contain further categories.

#### Variables

Variables defined with the `!template` tag can refer to other variables. This is useful if, e.g.
you want to refer to the card title in its text box.

There are a number of variables with pre-assigned meaning:

- `template`: Must be set to the path of the svg template file, relative to the CLI-provided
  template directory.
- `out_path`: Automatically set to the path to the output directory, as provided via the CLI.
- `assets_path`: Automatically set to the path to the assets directory, as provided via the CLI.
- `ordinal`: Automatically set to the zero-based number of the processed card, as counted from
  top to bottom in the card definitions file.
- `num_cards`: Automatically set to the total number of cards.

### CLI

```
Usage: pycard.py [OPTIONS] CARDS_FILE

  Main application entry point

Options:
  -o, --out-path PATH             Path to the output folder.
  -t, --templates-path PATH       Path to the template folder.
  -a, --assets-path PATH          Path to the assets folder.
  --png [none|firefox|chromium|webkit]
                                  Render as PNG using the provided method.
  --help                          Show this message and exit.
```