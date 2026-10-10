"""Inspect official local QMUL releases without altering their files or splits."""
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from .checkpoint_manager import compute_sha256

GROUP_COUNTS = {
    "tinyface": {"training":7804,"gallery":4443,"registered_probe":3728,"gallery_distractor":153428},
    "survface": {"training":220888,"gallery":60294,"registered_probe":60423,"unmated_probe":121736,"verification":10051},
}
GALLERY_ID_COUNTS = {"tinyface":2569,"survface":3000}
GROUP_FOLDERS = {
    "tinyface": {"training":"Training_Set","gallery":"Testing_Set/Gallery_Match",
                 "registered_probe":"Testing_Set/Probe","gallery_distractor":"Testing_Set/Gallery_Distractor"},
    "survface": {"training":"training_set","gallery":"Face_Identification_Test_Set/gallery",
                 "registered_probe":"Face_Identification_Test_Set/mated_probe",
                 "unmated_probe":"Face_Identification_Test_Set/unmated_probe",
                 "verification":"Face_Verification_Test_Set/verification_images"},
}


def _pairs(path, names_key, ids_key):
    value = loadmat(path, squeeze_me=True)
    names = np.atleast_1d(value[names_key]).reshape(-1).tolist()
    ids = np.atleast_1d(value[ids_key]).reshape(-1).tolist()
    if len(names) != len(ids) or len(set(names)) != len(names):
        raise ValueError(f"Invalid/duplicate MAT entries: {path}")
    result = []
    for filename, identity in zip(names,ids):
        filename = str(filename)
        if Path(filename).name != filename or not filename.lower().endswith('.jpg'):
            raise ValueError(f"Unsafe protocol filename: {filename}")
        if int(identity) != identity or int(identity) < 1:
            raise ValueError("Invalid official identity ID")
        result.append((filename,int(identity)))
    return result


def inventory(profile):
    name, root = profile['name'], Path(profile['raw_root']).resolve()
    groups, metadata = {}, {}
    for role, folder in GROUP_FOLDERS[name].items():
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset folder missing: {directory}")
        groups[role] = sorted(p for p in directory.rglob('*.jpg') if p.is_file())
        if len(groups[role]) != GROUP_COUNTS[name][role]:
            raise ValueError(f"{name}/{role} count mismatch: {len(groups[role])} != {GROUP_COUNTS[name][role]}")
    training_ids = {p.parent.name for p in groups['training']}
    if len(training_ids) != profile['num_classes']:
        raise ValueError("Training identity count differs from paper")
    for p in groups['training']:
        if p.stem.split('_',1)[0] != p.parent.name:
            raise ValueError(f"Training filename/native identity mismatch: {p}")
    definition = (("gallery", "gallery_match_img_ID_pairs.mat" if name=='tinyface' else "gallery_img_ID_pairs.mat", "gallery_set","gallery_ids"),
                  ("registered_probe", "probe_img_ID_pairs.mat" if name=='tinyface' else "mated_probe_img_ID_pairs.mat", "probe_set" if name=='tinyface' else "mated_probe_set", "probe_ids" if name=='tinyface' else "mated_probe_ids"))
    source_dir = root / ("Testing_Set" if name=='tinyface' else "Face_Identification_Test_Set")
    pairs = {}
    native_mapping, reverse_mapping = {}, {}
    for role, filename, names_key, ids_key in definition:
        original, duplicate = source_dir/filename, root/'Face_Identification_Evaluation'/filename
        pairs[role] = _pairs(original,names_key,ids_key)
        if pairs[role] != _pairs(duplicate,names_key,ids_key):
            raise ValueError(f"Evaluation/test MAT copies differ: {filename}")
        if {p.name for p in groups[role]} != {n for n,_ in pairs[role]}:
            raise ValueError(f"Missing/extra images relative to MAT: {name}/{role}")
        for image, official_id in pairs[role]:
            native = Path(image).stem.split('_',1)[0]
            if native in training_ids or not native.isdigit():
                raise ValueError("Training/test native identity overlap or invalid native ID")
            if native_mapping.get(native,official_id) != official_id or reverse_mapping.get(official_id,native) != native:
                raise ValueError("Inconsistent native-to-official identity mapping")
            native_mapping[native], reverse_mapping[official_id] = official_id,native
        for file in (original,duplicate):metadata[str(file.relative_to(root))] = compute_sha256(str(file))
    if not {i for _,i in pairs['registered_probe']} <= {i for _,i in pairs['gallery']}:
        raise ValueError("Probe identity missing from gallery")
    if len({i for _,i in pairs['gallery']}) != GALLERY_ID_COUNTS[name]:
        raise ValueError("Registered gallery identity count differs from protocol")
    # Verify every named 1:1 pair too, although identification is the FR target.
    if name=='survface':
        files={p.name for p in groups['verification']}
        referenced=set()
        for key in ('positive_pairs_names','negative_pairs_names'):
            p=root/'Face_Verification_Test_Set'/f'{key}.mat'
            matrix=loadmat(p,squeeze_me=True)[key]
            if matrix.shape != (5320,2):raise ValueError("Verification pair count differs")
            referenced.update(str(v) for v in matrix.flatten())
            metadata[str(p.relative_to(root))]=compute_sha256(str(p))
        if not referenced <= files:raise ValueError("Verification pair MAT references missing images")
        # The released verification folder also contains unused images. Keep
        # and audit them; they do not define extra evaluation pairs.
        metadata['verification_unreferenced_images']=len(files-referenced)
        for p in groups['unmated_probe']:
            native=p.stem.split('_',1)[0]
            if native.isdigit():
                if native in training_ids or native in native_mapping:
                    raise ValueError("SurvFace unmated identity overlaps training/gallery")
            elif not p.stem.startswith('distractors_') or not p.stem.removeprefix('distractors_').isdigit():
                raise ValueError("Unexpected unlabeled SurvFace distractor filename")
    for relative in ('readme.txt','Face_Identification_Evaluation/'+('test_face_identification.m' if name=='tinyface' else 'test_open_set_face_identification.m')):
        p=root/relative
        if not p.is_file():raise FileNotFoundError(str(p))
        metadata[relative]=compute_sha256(str(p))
    digest=hashlib.sha256()
    for role, files in groups.items():
        for p in files:
            stat=p.stat()
            if stat.st_size==0:raise ValueError(f"Empty image: {p}")
            digest.update(f"{p.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    digest.update(json.dumps(metadata,sort_keys=True).encode())
    return root,groups,pairs,metadata,digest.hexdigest()


def _immutable(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise FileExistsError(f"Existing provenance differs; use a new protocol directory: {path}")
    else:
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(content);os.replace(temporary,path)


def _write_manifest(path, rows):
    import io
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=['image_path','source_identity_id','official_identity_id','protocol_role','split','protocol_index'])
    writer.writeheader();writer.writerows(rows)
    _immutable(path,out.getvalue().encode('utf-8'))


def _decode(path):
    content=path.read_bytes();image=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.size==0:raise ValueError(f"Image decode failed: {path}")
    if content.startswith(b'\xff\xd8') and b'\xff\xd9' not in content:
        raise ValueError(f"JPEG end marker missing: {path}")
    return path,len(content),image.shape[:2],hashlib.sha256(content).hexdigest()


def prepare_local_release(profile, workers=4, decode_all=True):
    """Read-only raw audit; persist derived manifests/hashes under fine-tune/data."""
    root,groups,pairs,metadata,snapshot=inventory(profile)
    audit_file=Path(profile['local_audit'])
    if audit_file.exists():
        record=json.loads(audit_file.read_text(encoding='utf-8'))
        if record['source_snapshot']!=snapshot:raise ValueError("Dataset changed since audit; select a new protocol directory")
        if record.get('all_images_decode') is not True or not record.get('completed'):
            raise ValueError("Existing full image audit is incomplete")
        for key, hash_key in (('training_manifest','training_manifest_sha256'),('evaluation_manifest','evaluation_manifest_sha256'),('image_hashes','image_hashes_sha256')):
            if compute_sha256(record[key])!=record[hash_key]:raise ValueError(f"Audit artifact changed: {key}")
        return record
    if not decode_all:raise ValueError("A full decode audit is required before registering a local release")
    if not 1<=workers<=8:raise ValueError("AUDIT_WORKERS must be in [1,8]")
    cv2.setNumThreads(1)
    audit_file.parent.mkdir(parents=True,exist_ok=True)
    hash_file=audit_file.parent/'image_hashes.csv';temp=hash_file.with_suffix('.tmp')
    stats={};total=sum(map(len,groups.values()));completed=0
    try:
        with temp.open('w',newline='',encoding='utf-8') as out, ThreadPoolExecutor(max_workers=workers) as pool:
            writer=csv.writer(out);writer.writerow(['image_path','bytes','height','width','sha256'])
            for role, files in groups.items():
                heights,widths=[],[]
                iterator=iter(files)
                while batch:=list(islice(iterator,1024)):
                    for p,size,(h,w),checksum in pool.map(_decode,batch):
                        writer.writerow([p.relative_to(root).as_posix(),size,h,w,checksum]);heights.append(h);widths.append(w)
                    completed+=len(batch)
                    if completed//20000!=(completed-len(batch))//20000:print(f"{profile['name']} decode {completed:,}/{total:,}",flush=True)
                stats[role]={'images':len(files),'height_min':min(heights),'height_max':max(heights),'width_min':min(widths),'width_max':max(widths)}
        if hash_file.exists():raise FileExistsError(str(hash_file))
        os.replace(temp,hash_file)
    finally:temp.unlink(missing_ok=True)
    if inventory(profile)[-1] != snapshot:
        raise ValueError("Dataset changed during full decode audit")
    train_rows=[dict(image_path=p.relative_to(root).as_posix(),source_identity_id=p.parent.name,official_identity_id='',protocol_role='training',split='training',protocol_index=i) for i,p in enumerate(groups['training'])]
    evaluation=[]
    for role, table in pairs.items():
        folder=GROUP_FOLDERS[profile['name']][role]
        for i,(filename,official_id) in enumerate(table):
            evaluation.append(dict(image_path=f'{folder}/{filename}',source_identity_id=Path(filename).stem.split('_',1)[0],official_identity_id=official_id,protocol_role=role,split='test',protocol_index=i))
    extra_role='unmated_probe' if profile['name']=='survface' else 'gallery_distractor'
    for i,p in enumerate(groups[extra_role]):
        native=p.stem.split('_',1)[0]
        known=profile['name']=='survface' and native.isdigit()
        role=('unknown_known_probe' if known else 'unknown_unknown_probe') if profile['name']=='survface' else 'gallery_distractor'
        evaluation.append(dict(image_path=p.relative_to(root).as_posix(),source_identity_id=native if known else '',official_identity_id='',protocol_role=role,split='test',protocol_index=i))
    _write_manifest(profile['training_manifest'],train_rows);_write_manifest(profile['evaluation_manifest'],evaluation)
    gallery_ids={i for _,i in pairs['gallery']};probe_ids={i for _,i in pairs['registered_probe']}
    notes=[]
    if profile['name']=='survface':notes.append('Official website training 220890 differs from UCFace/local release 220888; website total 463507 differs from local identification total 463341. No official archive hash available.')
    else:notes.append('Official README/Table IX claim 2569 probe IDs; supplied complete MAT has 2563 unique probe IDs, 3728 images. Preserve MAT instead of inventing missing probes.')
    record={'name':profile['name'],'raw_root':str(root),'completed':True,'all_images_decode':True,'image_sha256_computed':True,
            'source_snapshot':snapshot,'groups':stats,'training_identities':profile['num_classes'],
            'registered_gallery_identities':len(gallery_ids),'registered_probe_identities':len(probe_ids),
            'protocol_metadata_sha256':metadata,'notes':notes,'release_identity':'local MAT/count/decode verified; byte identity to author archive unconfirmed',
            'image_hashes':str(hash_file.resolve()),'image_hashes_sha256':compute_sha256(str(hash_file)),
            'training_manifest':str(Path(profile['training_manifest']).resolve()),'training_manifest_sha256':compute_sha256(profile['training_manifest']),
            'evaluation_manifest':str(Path(profile['evaluation_manifest']).resolve()),'evaluation_manifest_sha256':compute_sha256(profile['evaluation_manifest'])}
    _immutable(audit_file,(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
    return record
