import collections
from multiprocessing import Pool, cpu_count


class EmbarrassingParallelisation:
    """
    This class provides a context manager for embarrassingly parallel
    environments that can be used by invoking a 'with' statement.

    """

    def __init__(self, number_of_processes, number_of_ops, verbosity=0):
        # Default to using all cores if either too many or "-1" are
        # specified.
        if number_of_processes > cpu_count() or number_of_processes == -1:
            number_of_processes = cpu_count()
        if verbosity >= 3:
            print(
                "Starting parallel processing pool with up"
                " to {} cores.".format(number_of_processes))
        self.number_of_processes = number_of_processes

        self.number_of_ops = number_of_ops
        chunk = collections.namedtuple('chunk', 'start finish')
        chunks = []
        step_size = number_of_ops // number_of_processes
        if step_size > 1:
            begin = -1
            for i in range(number_of_processes):
                end = begin + step_size
                if number_of_ops - step_size < end:
                    end = number_of_ops
                chunks.append(chunk(start=begin + 1, finish=end))
                begin += step_size
        else:
            chunks.append(chunk(start=0, finish=number_of_ops))
            chunks.append(chunk(start=0, finish=number_of_ops))
        self.chunks = chunks
        self.pool = Pool(processes=self.number_of_processes)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.pool.terminate()


if __name__ == '__main__':
    pass
