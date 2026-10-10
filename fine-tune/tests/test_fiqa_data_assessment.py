import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.fiqa_data_assessment import assess_survface_fit_pool


def test_capacity_is_proposal_only_and_anonymous_images_never_fit(tmp_path):
    folder=tmp_path/'Face_Identification_Test_Set/unmated_probe';folder.mkdir(parents=True)
    for native in range(1,101):
        for image in range(2):(folder/f'{native}_cam1_{image}.jpg').write_bytes(b'count-only fixture')
    for i in range(3):(folder/f'distractors_{i}.jpg').write_bytes(b'count-only fixture')
    result=assess_survface_fit_pool(tmp_path)
    assert result['status']=='proposal_only_not_applied' and result['official_evaluation_equivalent'] is False
    assert result['anonymous_test_only_images']==3
    assert sum(row['images'] for row in result['groups'].values())==200
    assert sum(row['identities'] for row in result['groups'].values())==100
    assert result==assess_survface_fit_pool(tmp_path)
    assert len(list(folder.iterdir()))==203
