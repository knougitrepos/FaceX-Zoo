"""Read-only capacity proposal, not a committed FIQA split or fitted model."""
import hashlib
import os
from collections import Counter
from pathlib import Path


def assess_survface_fit_pool(root, seed=8972, development_fraction=.4, safety_fraction=.3):
    """Keep the official gallery; reserve labeled unknown identities only in a proposal."""
    if not 0<development_fraction<1 or not 0<safety_fraction<1:
        raise ValueError("Fractions must be in (0,1)")
    directory=Path(root)/'Face_Identification_Test_Set/unmated_probe'
    counts=Counter();anonymous=0
    for entry in os.scandir(directory):
        if not entry.is_file() or not entry.name.endswith('.jpg'):continue
        native=entry.name.split('_',1)[0]
        if native.isdigit():counts[native]+=1
        else:anonymous+=1
    groups={role:[] for role in ('fit','safety','test_known_unknown')}
    for identity in sorted(counts):
        digest=hashlib.sha256(f'fiqa-development-v1:{seed}:{identity}'.encode()).digest()
        development=int.from_bytes(digest[:8],'big')/2**64<development_fraction
        # Same fit/safety hash rule as the current ronbun continuous module,
        # using a true native ID rather than a per-image pseudo-identity.
        value=hashlib.sha256(f'{seed}:survface:native:{identity}'.encode()).digest()
        role=('safety' if int.from_bytes(value[:8],'big')/2**64<safety_fraction else 'fit') if development else 'test_known_unknown'
        groups[role].append(identity)
    return {'status':'proposal_only_not_applied','seed':seed,'development_fraction':development_fraction,
            'safety_fraction':safety_fraction,'native_unknown_identities':len(counts),
            'native_unknown_images':sum(counts.values()),'anonymous_test_only_images':anonymous,
            'groups':{role:{'identities':len(ids),'images':sum(counts[x] for x in ids)} for role,ids in groups.items()},
            'fit_requirements':{'fit_non_mated_minimum_in_current_code':20,'safety_non_mated_minimum_in_current_code':20,
                                'expected_tail_images_at_fpir_0_01':{role:sum(counts[x] for x in groups[role])*.01 for role in ('fit','safety')}},
            'gallery':'unchanged official 3000 registered IDs',
            'calibration_source':'official non-mated evaluation pool; reserve before score inspection and exclude from final FIQA test',
            'training_split':'unchanged complete official FR training set',
            'official_evaluation_equivalent':False,
            'limitations':['Proposal changes non-mated test denominator; cannot claim exact UCFace evaluation reproduction',
                          'Fit sees known-unknown native IDs; anonymous distractors kept test-only create a possible mixture/domain shift',
                          'Repeated images per identity are correlated; image counts are not independent sample counts',
                          'No quality-bin coverage, embeddings, score fitting, PQ training, thresholds, or formal FPIR guarantee verified',
                          'Current ronbun per-image unknown labels need true-native identity grouping before fit/safety use']}
