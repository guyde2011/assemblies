"""
Utility for measuring connectome performance.
Can measure Connectome classes (which implement ABCConnectome),
by running: (for example)
python -m benchmark -c Connectome -c OtherConnectome
"""

from .time_graphs import graphit