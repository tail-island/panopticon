import numpy

from itertools import chain, count, islice, repeat, starmap, tee


data_files_sets = ((('./data/train/working.txt',),
                    ('./data/train/dancing.txt',)),
                   (('./data/test/working.txt',),
                    ('./data/test/dancing.txt',)))
#                  (('./data/train/working.txt',),
#                   ('./data/train/dancing.txt',)))


class Dataset:
    def _shuffle(self):
        indice = numpy.arange(len(self.inputs))
        numpy.random.shuffle(indice)

        self.inputs = self.inputs[indice]
        self.labels = self.labels[indice]
    
    def __init__(self, inputs_set):
        self.inputs = numpy.asarray(tuple(chain(*inputs_set)))
        self.labels = numpy.asarray(tuple(chain(*starmap(lambda i, inputs: repeat(i, len(inputs)), enumerate(inputs_set)))))
        self._batch_index = 0

        self._shuffle()

    def next_batch(self, batch_size):
        if self._batch_index + batch_size > len(self.inputs):
            self._shuffle()
            self._batch_index = 0

        start = self._batch_index
        end = self._batch_index = self._batch_index + batch_size

        return self.inputs[start:end], self.labels[start:end]


def load():
    def inputs_in_data_files(data_files):
        def inputs_in_data_file(data_file):
            poses = map(tuple, map(lambda s: map(float, s.split()), open(data_file)))
            actions = zip(*starmap(lambda i, it: islice(it, i, None), enumerate(tee(poses, 8))))
            
            return map(tuple, starmap(chain, actions))

        return tuple(chain(*map(inputs_in_data_file, data_files)))
    
    for data_files_set in data_files_sets:
        yield Dataset(tuple(map(inputs_in_data_files, data_files_set)))
