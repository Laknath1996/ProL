import torch
import torch.utils
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from copy import deepcopy

# get pattern
def get_cycle(N: int) -> list:
    """Get the primary cycle

    Parameters
    ----------
    N : int
        time between two task switches

    Returns
    -------
    list
        primary cycle
    """
    return [1] * N + [0] * N

def get_multi_cycle(N, num_tasks):
    """Get the primary cycle (for multi-task problems)

    Parameters
    ----------
    N : int
        time between two task switches
    num_tasks : int
        number of tasks

    Returns
    -------
    list
        primary cycle
    """
    flags = np.arange(num_tasks)
    unit = []
    for flag in flags:
        unit += [flag] * N
    return unit

# synthetic data
def draw_synthetic_samples(flip):
    y = np.random.binomial(1, 0.5)
    l = 1-y if flip else y
    x = l * np.random.uniform(-2, -1) + (1-l) * np.random.uniform(1, 2)
    return x, y

def get_synthetic_data(N, total_time_steps, seed=1996):
    """Get synthetic data sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    seed : random seed

    Returns
    -------
    index sequence
    """
    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("bool")
    data = np.zeros((total_time_steps, 2)).astype('float')
    np.random.seed(seed)
    data[pattern] = np.array([draw_synthetic_samples(True) for _ in range(sum(pattern))])
    data[~pattern] = np.array([draw_synthetic_samples(False) for _ in range(sum(~pattern))])
    return torch.from_numpy(data[:, 0]).float(), torch.from_numpy(data[:, 1]).long()

# vision data
def get_torch_dataset(root, name='mnist'):
    """Get the original torch datase

    Returns
    -------
    _type_
        torch dataset
    """
    if name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True
        )
        # normalize
        tmp = dataset.data.float() / 255.0
        tmp = (tmp - 0.5)/0.5
        dataset.data = tmp[..., None]

    elif name == 'cifar-10':
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True
        )
        # normalize
        tmp = torch.from_numpy(dataset.data).float() / 255.0
        tmp = (tmp - 0.5)/0.5
        dataset.data = tmp
        tmp = dataset.targets
        dataset.targets = torch.Tensor(tmp).long()
        
    assert dataset.data.max() == 1.0
    assert dataset.data.min() == -1.0
    return dataset

# vision (label swap and covariate shift)
def get_task_indicies_and_map(tasks: list, y: np.ndarray, type='covariate-shift'):
    """Get the indices for each task + the label mapping

    Parameters
    ----------
    tasks : list
        task specification e.g. [[0, 1], [2, 3]]
    y : np.ndarray
        dataset targets
    """
    tasklib = {}
    for i, task in enumerate(tasks):
        tasklib[i] = []
        for lab in task:
            tasklib[i].extend(
                np.where(y == lab)[0].tolist()
            )
    if type == 'covariate-shift':
        mapdict = {}
        for task in tasks:
            for i, lab in enumerate(task):
                mapdict[lab] = i
        maplab = lambda lab : mapdict[lab]
        return tasklib, maplab
    elif type == 'label-swap':
        assert len(tasklib) == 1
        assert len(tasks) == 1
        assert len(tasks[0]) == 2
        
        taskids = np.array(tasklib[0])
        np.random.seed(1)
        taskids = taskids[np.random.permutation(len(taskids))]
        tasklib_ = {}
        tasklib_[0] = taskids[:len(taskids)//2].tolist()
        tasklib_[1] = taskids[len(taskids)//2:].tolist()
        
        tasks_ = [tasks[0], [10, 11]]
        mapdict = {}
        for task in tasks_:
            for i, lab in enumerate(task):
                mapdict[lab] = i
        mapdict[10] = 1
        mapdict[11] = 0
        maplab = lambda lab : mapdict[lab] 

        dummydict = {
            tasks[0][0]: 10, 
            tasks[0][1]: 11
        }
        dummylab = lambda lab : dummydict[lab]
        return tasklib_, maplab, dummylab
    else:
        raise NotImplementedError

def get_sequence_indices(N, total_time_steps, tasklib, seed=1996, remove_train_samples=False):
    """Get indices for a sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    tasklib : original task indices
    seed : random seed

    Returns
    -------
    index sequence
    """
    tasklib = deepcopy(tasklib)
    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("bool")
    seqInd = np.zeros((total_time_steps,)).astype('int')
    np.random.seed(seed)
    seqInd[pattern] = np.random.choice(tasklib[0], sum(pattern), replace=False)
    seqInd[~pattern] = np.random.choice(tasklib[1], sum(~pattern), replace=False)

    if remove_train_samples:
        tasklib[0] = list(
            set(tasklib[0]) - set(tasklib[0]).intersection(seqInd)
        )
        tasklib[1] = list(
            set(tasklib[1]) - set(tasklib[1]).intersection(seqInd)
        )
        return seqInd, tasklib
    else:
        return seqInd

# vision (multi-class and multi-task)
def flatten(tasks):
    return [x for task in tasks for x in task]

def get_multi_indices_and_map(tasks, dataset):
    """Get the indices for each task + the label mapping

    Parameters
    ----------
    tasks : list
        task specification e.g. [[0, 1], [2, 3]]
    dataset : torch Dataset
        torch dataset 
    """
    y = dataset.targets.numpy()

    # get the number of unique classes in the task spec
    labels = np.unique(flatten(tasks))

    # get the indices for each unique class in the task spec
    classlib = {}
    for lab in labels:
        classlib[lab] = []
        classlib[lab].extend(
            np.where(y == lab)[0].tolist()
        )

    # get the total number of appearance of each class in the task spec
    label_usage = []
    for lab in labels:
        count = 0
        for task in tasks:
            if lab in task:
                count += 1
        label_usage.append(count)

    # split the indices of each class according to class usage above
    for lab, count in zip(labels, label_usage):
        tmp = np.array(classlib[lab])
        tmp = np.array_split(tmp, count)
        classlib[lab] = tmp

    # get the indices corresponding to each task
    tasklib = {}
    for i, task in enumerate(tasks):
        tasklib[i] = []
        for lab in task:
            tasklib[i].extend(
                classlib[lab].pop()
            )

    # get the label mapper
    task_labels = [np.arange(len(task)).tolist() for task in tasks]
    dummy_labels = deepcopy(task_labels)
    k = 10
    for i, _ in enumerate(task_labels):
        for j, _ in enumerate(task_labels[i]):
            dummy_labels[i][j] = k
            k += 1
    mapdict = dict(zip(flatten(dummy_labels), flatten(task_labels)))
    maplab = lambda lab : mapdict[lab]

    # assign dummy labels to the original labels
    for i, _ in enumerate(tasks):
        tmp = y[tasklib[i]]
        for trulab, dumlab in zip(tasks[i], dummy_labels[i]):
            tmp[tmp == trulab] = dumlab
        y[tasklib[i]] = tmp

    dataset.targets = torch.from_numpy(y).long()

    return tasklib, maplab, dataset

def get_multi_sequence_indices(N, total_time_steps, tasklib, seed=1996, remove_train_samples=False):
    """Get indices for a sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    tasklib : original task indices
    seed : random seed

    Returns
    -------
    index sequence
    """
    tasklib = deepcopy(tasklib)
    num_tasks = len(tasklib)
    unit = get_multi_cycle(N, num_tasks)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("int")
    seqInd = np.zeros((total_time_steps,)).astype('int')

    np.random.seed(seed)
    for taskid in range(num_tasks):
        seqInd[pattern==taskid] = np.random.choice(tasklib[taskid], sum(pattern==taskid), replace=False)

    if remove_train_samples:
        for taskid in range(num_tasks):
            tasklib[taskid] = list(
                set(tasklib[taskid]) - set(tasklib[taskid]).intersection(seqInd)
            ) 
        return seqInd, tasklib
    else:
        return seqInd

if __name__ == "__main__":
    x, y = get_synthetic_data(20, 100)
    print(x.shape)




