import typer

from .trace import trace
from .tilings import tilings
from .model import model


app = typer.Typer()

app.command()(trace)
app.command()(tilings)
app.command()(model)

if __name__ == "__main__":
    app()

