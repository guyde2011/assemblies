"""
Utility for measuring connectome performance.
Can measure connectome classes (which implement ABCConnectome),
by running: (for example)
python -m benchmark -c connectome -c OtherConnectome
"""

from .time_graphs import graphit