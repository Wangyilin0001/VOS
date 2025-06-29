import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt,line,epoch,two = 1):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    l1,l2,best_ls,d_ls = data_loader.initialize(opt,line,epoch,two)
    return data_loader,l1,l2,best_ls,d_ls


def CreateDataset(opt,line,epoch,two = 1):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from .aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    if epoch == 1 and two == 1:
        dataset.one()
    l1,l2,best_ls_x,d_ls = dataset.initialize(opt,line,epoch,two)
    return dataset,l1,l2,best_ls_x,d_ls


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, line, epoch,two = 1):
        BaseDataLoader.initialize(self, opt)
        self.dataset, l1, l2,best_ls_x,d_ls = CreateDataset(opt, line, epoch,two)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            pin_memory = True
           )
        return l1, l2 , best_ls_x,d_ls

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
