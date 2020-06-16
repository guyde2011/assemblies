import resource
import psutil


def protecc_ram(pctg: float = 0.5):
    """Limit RAM usage to at most some percentage of available RAM"""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (psutil.virtual_memory().available * 1024 * pctg, hard))
