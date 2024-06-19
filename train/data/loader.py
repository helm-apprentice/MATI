import torch
import torch.utils.data.dataloader
import importlib
import collections
from torch._six import string_classes
from lib.utils import TensorDict, TensorList
import warnings

if float(torch.__version__[:3]) >= 1.9 or len('.'.join((torch.__version__).split('.')[0:2])) > 3:
    int_classes = int
else:
    from torch._six import int_classes


def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module('torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None


def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    
    if isinstance(batch[0], torch.Tensor):
        #print(f'batch[0] is torch.Tensor: {batch[0]}' )
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.stack(batch, 1, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        print(f'batch[0] 2: {batch[0]}' )
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        #print(f'batch[0] is int_classes: {batch[0]}' )
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        #print(f'batch[0] is float: {batch[0]}' )
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        #print(f'batch[0] is string_classes: {batch[0]}' )
        return batch
    elif isinstance(batch[0], TensorDict):
        #print(f'batch[0] is TensorDict: {batch[0]}' )
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        #print(f'batch[0] is collections.Mapping: {batch[0]}' )
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        #print(f'batch[0] is Tensorlist: {batch[0]}' )
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        #print(f'batch[0] is collections.Sequence: {batch[0]}' )
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        #print(f'batch[0] is None {batch[0]}' )
        return batch
    warnings.resetwarnings()
    raise TypeError((error_msg.format(type(batch[0]))))


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Note: The only difference with default pytorch DataLoader is that an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        stack_dim (int): Dimension along which to stack to form the batch. (default: 0)
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraries
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim


'''
让我们通过一个详细的例子来说明如何使用修改后的 `ltr_collate_stack1` 函数处理包含嵌套字典的 `TensorDict`。

假设我们有一批数据（batch），每个数据项是一个 `TensorDict`，其中包含嵌套字典，用于表示多模态数据（比如 `visible` 和 `infrared` 图像）。
我们将展示如何使用 `ltr_collate_stack1` 函数来将这个批次中的数据合适地组合起来。

### 示例数据

```python
batch = [
    TensorDict({
        'template_images': {
            'visible': torch.tensor([1, 2]),
            'infrared': torch.tensor([3, 4])
        },
        'search_images': torch.tensor([5, 6])
    }),
    TensorDict({
        'template_images': {
            'visible': torch.tensor([7, 8]),
            'infrared': torch.tensor([9, 10])
        },
        'search_images': torch.tensor([11, 12])
    })
]
```

这里，每个 `TensorDict` 代表一个数据样本，包含两种模态的 `template_images`（可见光和红外），以及单模态的 `search_images`。

### `ltr_collate_stack1` 函数

函数 `ltr_collate_stack1` 被用来处理批次中的每个数据样本，并且它能够递归地处理嵌套字典。

```python
def ltr_collate_stack1(batch):
    # ...（其他类型的处理）

    elif isinstance(batch[0], TensorDict):
        # 对于 TensorDict 类型，递归地调用 ltr_collate_stack1
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})

    # ...（其他类型的处理）
```

### 应用 `ltr_collate_stack1` 函数

当我们对示例数据应用 `ltr_collate_stack1` 函数时，如下所示：

```python
collated_batch = ltr_collate_stack1(batch)
```

函数将执行以下步骤：

1. 检查 `batch` 中的第一个元素类型，发现它是一个 `TensorDict`。
2. 对于 `TensorDict` 中的每个键（比如 `template_images` 和 `search_images`），
    函数将从批次中的所有数据样本中收集相应的值，并递归地调用 `ltr_collate_stack1`。
3. 对于 `template_images` 中的每个模态（`visible` 和 `infrared`），函数将堆叠批次中相应的张量。
4. 最终，所有这些堆叠的张量将被组合成一个新的 `TensorDict`，它是批次处理后的结果。

### 结果

经过处理后，`collated_batch` 将是这样的：

```python
TensorDict({
    'template_images': {
        'visible': torch.tensor([[1, 2], [7, 8]]),  # 堆叠 visible 图像
        'infrared': torch.tensor([[3, 4], [9, 10]])  # 堆叠 infrared 图像
    },
    'search_images': torch.tensor([[5, 6], [11, 12]])  # 堆叠 search 图像
})
```

在这个结果中，`'template_images'` 下的 `'visible'` 和 `'infrared'` 图像以及 `'search_images'` 都被适当地堆叠在一起，形成了适合批量处理的结构。
这样的结构对于后续在神经网络中的使用是非常方便的。
'''