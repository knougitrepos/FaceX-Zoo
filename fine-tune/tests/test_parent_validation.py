import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.parent_validation import select_cohort, recognition_metrics, compare_parents


def test_fixed_cohort_excludes_training_ids_and_is_repeatable(tmp_path):
    meta = tmp_path/'identities.csv'
    meta.write_text('Class_ID,Name,Sample_Num,Flag\ntrain,T,4,1\na,A,4,0\nb,B,4,0\nc,C,4,0\n')
    images = tmp_path/'test.txt'
    names = [f'{identity}/{i}.jpg' for identity in 'abc' for i in range(4)]
    images.write_text('\n'.join(names))
    options = dict(identity_count=2,images_per_identity=3,gallery_per_identity=1)
    first = select_cohort(meta,images,**options)
    images.write_text('\n'.join(reversed(names)))
    assert select_cohort(meta,images,**options) == first
    assert len(first) == 6 and sum(r['role']=='gallery' for r in first) == 2
    images.write_text('\n'.join(names+['train/1.jpg']))
    with pytest.raises(ValueError,match='training identity'):
        select_cohort(meta,images,**options)


def test_identification_uses_identity_pooling_and_conservative_ties():
    rows = [dict(identity=i,role=role) for i,role in [('a','gallery'),('a','gallery'),('b','gallery'),('a','probe'),('b','probe')]]
    values = np.asarray([[1,0],[1,0],[0,1],[1,0],[1,0]],dtype=np.float32)
    metrics,ranks = recognition_metrics(values,rows)
    assert ranks.tolist() == [1,2]
    assert metrics['rank1_correct'] == 1 and metrics['rank5_correct'] == 2
    assert metrics['identities'] == 2 and metrics['gallery_images'] == 3
    tied = np.ones((len(rows),2),dtype=np.float32)
    assert recognition_metrics(tied,rows)[1].tolist() == [2,2]


def test_pair_comparison_rejects_different_epoch_or_cohort(tmp_path):
    a,b = tmp_path/'a.json',tmp_path/'b.json'
    a.write_text(json.dumps(dict(protocol_sha256='a',epoch=1)))
    b.write_text(json.dumps(dict(protocol_sha256='a',epoch=2)))
    with pytest.raises(ValueError,match='same cohort and epoch'):
        compare_parents(a,b,tmp_path/'comparison.json')
