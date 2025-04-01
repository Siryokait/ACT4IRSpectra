from data_provider.data_loader import Dataset_tablet
from torch.utils.data import DataLoader

data_dict = {"tablet": Dataset_tablet}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        if args.data == 'melamine' and args.train_d == 'R568':
            drop_last = True

    data_set = Data(
        args=args,
        flag=flag
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    return data_set, data_loader
