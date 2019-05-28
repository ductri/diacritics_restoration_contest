from data_for_train import docs

COLUMN_NAME = 'tgt'


def bootstrap():
    docs.bootstrap()


def get_dl_train(batch_size, size=None):
    return docs.create_data_loader(docs.ROOT + 'main/data_for_train/output/my_train.csv', COLUMN_NAME, batch_size,
                                   docs.NUM_WORKERS, size=size, shuffle=True)


def get_dl_test(batch_size):
    return docs.create_data_loader(docs.ROOT + 'main/data_for_train/output/my_test.csv', COLUMN_NAME, batch_size,
                                   docs.NUM_WORKERS, shuffle=False)


def get_dl_eval(batch_size, size=None):
    return docs.create_data_loader(docs.ROOT + 'main/data_for_train/output/my_eval.csv', COLUMN_NAME, batch_size,
                                   docs.NUM_WORKERS, shuffle=False, size=size)
