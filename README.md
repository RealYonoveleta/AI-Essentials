# AI-Essentials

A collection of AI essential utilities and tools built with Python.

## Requirements

- Python 3.10+

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

Run the CLI entry point:

```bash
ai-essentials
```

Or run directly:

```bash
python -m ai_essentials.main
```

## Development

### Running Tests

```bash
pytest
```

### Project Structure

```
AI-Essentials/
├── src/
│   └── ai_essentials/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── pyproject.toml
└── README.md
```