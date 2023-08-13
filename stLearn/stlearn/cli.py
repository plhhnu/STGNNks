from stlearn.interactive.app import app
import click
from flask.cli import FlaskGroup


app.run(host="0.0.0.0", port=9999, debug=True)
