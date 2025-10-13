import typer

from .trace import trace


app = typer.Typer()
app.command()(trace)


if __name__ == "__main__":
    app()

