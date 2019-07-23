"""Semeval cQA data loader."""

import typing
import os
from pathlib import Path

import keras
import pandas as pd

import matchzoo

_url = "http://alt.qcri.org/semeval2016/task3/data/uploads/" \
    "semeval2016-task3-cqa-arabic-md-train-v1.3.zip"


def load_data(
    stage: str = 'train',
    task: str = 'ranking',
    filtered: bool = False,
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = Path('../../data/semeval')
    file_path = data_root.joinpath(f'{stage}.json')
    data_pack = _read_data(file_path)
    if filtered and stage in ('dev', 'test'):
        ref_path = data_root.joinpath(f'{stage}.ref')
        filter_ref_path = data_root.joinpath(f'{stage}-filtered.ref')
        with open(filter_ref_path, mode='r') as f:
            filtered_ids = set([line.split()[0] for line in f])
        filtered_lines = []
        with open(ref_path, mode='r') as f:
            for idx, line in enumerate(f.readlines()):
                if line.split()[0] in filtered_ids:
                    filtered_lines.append(idx)
        data_pack = data_pack[filtered_lines]

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'semeval', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='semeval'
    )
    return Path(ref_path).parent.joinpath('CQA')


def _read_data(path):
    table = pd.read_json(path, lines=True)
    df = pd.DataFrame({
        'text_left': table['question'].str.join(' '),
        'text_right': table['qaquestion'].str.join(' '),
        'id_left': table['qid'],
        'id_right': table['qaid'],
        'label': table['qarel'].astype(float)
    })

    print(df)
    return matchzoo.pack(df)
