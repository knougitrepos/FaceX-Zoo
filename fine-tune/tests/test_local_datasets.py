"""Real MAT/file contracts on small independent fixtures, with no raw mutations."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules import local_datasets as local
from modules import finetune_protocol as protocol


@pytest.fixture
def release(tmp_path,monkeypatch):
    monkeypatch.setitem(local.GROUP_COUNTS,'tinyface',{'training':4,'gallery':2,'registered_probe':2,'gallery_distractor':3})
    monkeypatch.setitem(local.GALLERY_ID_COUNTS,'tinyface',2)
    monkeypatch.setitem(protocol.PAPER_SPLITS,'tinyface',(2,4))
    root=tmp_path/'tinyface'
    paths=['Training_Set/1/1_1.jpg','Training_Set/1/1_2.jpg','Training_Set/2/2_1.jpg','Training_Set/2/2_2.jpg',
           'Testing_Set/Gallery_Match/3_1.jpg','Testing_Set/Gallery_Match/4_1.jpg',
           'Testing_Set/Probe/3_2.jpg','Testing_Set/Probe/4_2.jpg',
           'Testing_Set/Gallery_Distractor/scene_1.jpg','Testing_Set/Gallery_Distractor/scene_2.jpg','Testing_Set/Gallery_Distractor/scene_3.jpg']
    for i,relative in enumerate(paths):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(p),np.full((12+i,15,3),50+i,dtype=np.uint8))
    evaluation=root/'Face_Identification_Evaluation';evaluation.mkdir()
    (evaluation/'test_face_identification.m').write_text('% fixture protocol')
    (root/'readme.txt').write_text('Fixture official layout')
    for filename,key,names in [('gallery_match_img_ID_pairs.mat','gallery',['3_1.jpg','4_1.jpg']),('probe_img_ID_pairs.mat','probe',['3_2.jpg','4_2.jpg'])]:
        payload={key+'_set':np.array(names,dtype=object),key+'_ids':np.array([1,2])}
        savemat(evaluation/filename,payload)
        (root/'Testing_Set'/filename).write_bytes((evaluation/filename).read_bytes())
    profile=protocol.default_profiles(tmp_path)['tinyface']
    profile.update(raw_root=str(root),num_classes=2,expected_images=4)
    return profile,root


def test_full_decode_and_native_identity_manifest(release):
    profile,root=release
    before=(root/'Testing_Set/Probe/3_2.jpg').read_bytes()
    audit=local.prepare_local_release(profile,workers=2)
    assert audit['all_images_decode'] is True and audit['groups']['training']['images']==4
    record=protocol.prepare_protocol(profile)
    assert record['images']==4 and record['evaluation_images_without_native_identity']==3
    assert record['identity_disjoint'] and record['evaluation_identities']==2
    protocol.verify_protocol(profile)
    assert local.prepare_local_release(profile,workers=1)==audit
    assert (root/'Testing_Set/Probe/3_2.jpg').read_bytes()==before
    assert len(Path(profile['train_list']).read_text().splitlines())==4


def test_mat_membership_catches_replaced_file_even_when_count_matches(release):
    profile,root=release
    (root/'Testing_Set/Probe/3_2.jpg').rename(root/'Testing_Set/Probe/3_9.jpg')
    with pytest.raises(ValueError,match='MAT'):
        local.inventory(profile)


def test_training_identity_overlap_is_rejected(release):
    profile,root=release
    folder=root/'Training_Set/1';folder.rename(root/'Training_Set/3')
    for file in (root/'Training_Set/3').iterdir():file.rename(file.with_name(file.name.replace('1_','3_',1)))
    with pytest.raises(ValueError,match='overlap'):
        local.inventory(profile)


def test_source_mutation_invalidates_completed_audit(release):
    profile,root=release
    local.prepare_local_release(profile,workers=1)
    p=root/'Testing_Set/Probe/3_2.jpg';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed since audit'):
        local.prepare_local_release(profile)


def test_decode_failure_never_registers_completed_dataset(release):
    profile,root=release
    (root/'Testing_Set/Probe/3_2.jpg').write_bytes(b'invalid JPEG')
    with pytest.raises(ValueError,match='decode failed'):
        local.prepare_local_release(profile)
    assert not Path(profile['local_audit']).exists()
    assert not Path(profile['training_manifest']).exists()


def test_scface_remains_disabled():
    root=Path(__file__).resolve().parents[2]
    from modules.notebook_workflow import resolve_settings,merge_settings,plan_matrix
    settings=resolve_settings(root)
    assert len(plan_matrix(settings,'pretrain'))==1
    assert len(plan_matrix(settings,'finetune'))==40
    assert set(row['dataset'] for row in plan_matrix(settings,'export'))=={'tinyface','survface'}
    with pytest.raises(ValueError,match='disabled'):
        plan_matrix(merge_settings(settings,{'campaign':{'finetune_datasets':['scface']}}),'finetune')


def test_survface_pair_folder_may_contain_unused_images(tmp_path,monkeypatch):
    monkeypatch.setitem(local.GROUP_COUNTS,'survface',{'training':4,'gallery':2,'registered_probe':2,'unmated_probe':2,'verification':3})
    monkeypatch.setitem(local.GALLERY_ID_COUNTS,'survface',2)
    monkeypatch.setitem(protocol.PAPER_SPLITS,'survface',(2,4))
    profile=protocol.default_profiles(tmp_path)['survface'];profile.update(num_classes=2,expected_images=4,legacy_fiqa_manifest=None)
    root=Path(profile['raw_root'])
    layout={'training_set':['1/1_cam1_1.jpg','1/1_cam1_2.jpg','2/2_cam1_1.jpg','2/2_cam1_2.jpg'],
            'Face_Identification_Test_Set/gallery':['3_cam1_1.jpg','4_cam1_1.jpg'],
            'Face_Identification_Test_Set/mated_probe':['3_cam2_1.jpg','4_cam2_1.jpg'],
            'Face_Identification_Test_Set/unmated_probe':['5_cam1_1.jpg','distractors_1.jpg'],
            'Face_Verification_Test_Set/verification_images':['3_cam1_1.jpg','4_cam1_1.jpg','unused_cam1_1.jpg']}
    for folder,names in layout.items():
        for name in names:
            p=root/folder/name;p.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(p),np.full((16,16,3),80,dtype=np.uint8))
    evaluation=root/'Face_Identification_Evaluation';evaluation.mkdir()
    (evaluation/'test_open_set_face_identification.m').write_text('% fixture')
    (root/'readme.txt').write_text('fixture')
    for role,key,file in [('gallery','gallery','gallery_img_ID_pairs.mat'),('mated_probe','mated_probe','mated_probe_img_ID_pairs.mat')]:
        payload={key+'_set':np.array(layout['Face_Identification_Test_Set/'+role],dtype=object),key+'_ids':np.array([1,2])}
        savemat(evaluation/file,payload);(root/'Face_Identification_Test_Set'/file).write_bytes((evaluation/file).read_bytes())
    for key in ('positive_pairs_names','negative_pairs_names'):
        savemat(root/'Face_Verification_Test_Set'/f'{key}.mat',{key:np.tile(np.array(['3_cam1_1.jpg','4_cam1_1.jpg'],dtype=object),(5320,1))})
    record=local.prepare_local_release(profile,workers=2)
    assert record['protocol_metadata_sha256']['verification_unreferenced_images']==1
    provenance=protocol.prepare_protocol(profile)
    assert provenance['identity_disjoint'] and provenance['evaluation_images_without_native_identity']==1
    assert provenance['evaluation_identities']==3
    protocol.verify_protocol(profile)
