import click

from . import graphit


@click.command()
@click.option('-n', '--name',
              default=None,
              type=str,
              help='The graphs will be saved in graphs/name if specified')
@click.option('-c', '--classes',
              default=['Connectome', 'NonLazyConnectomeOriginal'],
              type=str,
              multiple=True,
              help='Classes to time and graph')
def cli_graphit(name, classes):
    graphit(name, classes)


if __name__ == '__main__':
    cli_graphit()