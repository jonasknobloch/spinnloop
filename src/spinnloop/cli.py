import typer

from .trace import trace
from .tilings import tilings


app = typer.Typer()

app.command()(trace)
app.command()(tilings)


if __name__ == "__main__":
    app()

