"""Main entry point for AI Essentials."""


def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}! Welcome to AI Essentials."


def main() -> None:
    """Run the AI Essentials CLI."""
    print(greet("World"))


if __name__ == "__main__":
    main()
