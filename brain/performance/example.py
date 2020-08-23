
# TODO: remove this file
@multithreaded(count=4)
def do_sum(area1, area2):
    
    def multi_sum(block_start, block_end):
        ...\
            
@multithreaded(params=['mat'], after=lambda x: sum(x))
def do_sum(first, last, mat):
    return np.sum(mat[first:last])

@do_sum.after
def after_sum(x):
    return sum(x)


@multithreaded(params={'rg': get_random_generators})
def filler(rg: Generator, first: int, last: int):
    out[first:last] = rg.binomial(1, p, out[first:last].shape)

@filter.param('rg')
def filter_param_rg(count):
    return ...
