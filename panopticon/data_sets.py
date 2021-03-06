import numpy

from itertools import chain, islice, repeat, starmap, tee


channel_size = 10
history_size = 25

# data_files_lists = ((('./data/kob-0.txt', './data/kur-0.txt', './data/nak-0.txt'),
#                      ('./data/kob-1.txt', './data/kur-1.txt', './data/nak-1.txt')),
#                     (('./data/ike-0.txt',),
#                      ('./data/ike-1.txt',)))

# data_files_lists = ((('./data/ike-0.txt', './data/kur-0.txt', './data/nak-0.txt'),
#                      ('./data/ike-1.txt', './data/kur-1.txt', './data/nak-1.txt')),
#                     (('./data/kob-0.txt',),
#                      ('./data/kob-1.txt',)))

# data_files_lists = ((('./data/ike-0.txt', './data/kob-0.txt', './data/nak-0.txt'),
#                      ('./data/ike-1.txt', './data/kob-1.txt', './data/nak-1.txt')),
#                     (('./data/kur-0.txt',),
#                      ('./data/kur-1.txt',)))

data_files_lists = ((('./data/ike-0.txt', './data/kob-0.txt', './data/kur-0.txt'),
                     ('./data/ike-1.txt', './data/kob-1.txt', './data/kur-1.txt')),
                    (('./data/nak-0.txt',),
                     ('./data/nak-1.txt',)))


class DataSet:
    def _shuffle(self):
        indice = numpy.arange(len(self.inputs))
        numpy.random.shuffle(indice)

        self.inputs = self.inputs[indice]
        self.labels = self.labels[indice]

    def __init__(self, inputs_list):
        self.inputs = numpy.asarray(tuple(chain(*inputs_list)))
        self.labels = numpy.asarray(tuple(chain(*starmap(lambda i, inputs: repeat(i, len(inputs)), enumerate(inputs_list)))))
        self._batch_index = 0

        self._shuffle()

    def next_batch(self, batch_size):
        if self._batch_index + batch_size > len(self.inputs):
            self._shuffle()
            self._batch_index = 0

        start = self._batch_index
        end = self._batch_index = self._batch_index + batch_size

        return self.inputs[start:end], self.labels[start:end]


def actions(poses):
    return map(tuple, starmap(chain, zip(*starmap(lambda i, it: islice(it, i, None), enumerate(tee(poses, history_size))))))
    

def load():
    def inputs_in_data_files(data_files):
        def inputs_in_data_file(data_file):
            return actions(map(tuple, map(lambda s: map(float, s.split()), open(data_file))))
        
        return tuple(chain(*map(inputs_in_data_file, data_files)))

    for data_files_list in data_files_lists:
        yield DataSet(tuple(map(inputs_in_data_files, data_files_list)))
