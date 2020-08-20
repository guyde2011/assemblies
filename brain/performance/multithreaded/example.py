from multithreaded import multithreaded

# TODO: remove code examples from source directories
# TODO: convert examples into tests. examples that are intended to teach usage, should be in a manual or readme file

@multithreaded(threads=6)
def sum_list(list_chunk):
    return sum(list_chunk)


@sum_list.params
def sum_list_params(thread_count, lst):
    return [((lst[2 * i:2 * i + 2],), {}) for i in range(thread_count)]


@sum_list.after
def sum_list_after(sums):
    return sum([s or 0 for s in sums])


if __name__ == '__main__':
    print(sum_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
