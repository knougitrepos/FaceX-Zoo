import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.trainer import AverageMeter, DeferredLossMeter, Trainer, set_seed, TrainingStopped
from modules import notebook_workflow as flow
from modules.checkpoint_manager import CheckpointManager
from modules.model_builder import build_model
from test_notebook_workflow import cfg


def test_deferred_meter_preserves_ordered_scalar_summation():
    old = AverageMeter()
    deferred = DeferredLossMeter(3)
    for value, count in [(1e10,64),(0.0001,32),(-1e10,64),(7.1,4)]:
        loss = torch.tensor(value,requires_grad=True)
        old.update(loss.item(),count)
        deferred.update_tensor(loss,count)
    deferred.flush()
    assert (deferred.sum,deferred.count,deferred.avg)==(old.sum,old.count,old.avg)
    assert not deferred.pending


@pytest.mark.parametrize('head',['arcface','adaface','magface'])
def test_deferred_sync_does_not_change_weights_history_or_recovery(cfg,tmp_path,head):
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    cfg=flow.merge_settings(cfg,{'training':{'epoches':2,'print_freq':100}})
    def create(sync,path):
        settings=flow.merge_settings(cfg,{'training':{'loss_sync_batches':sync}})
        set_seed(8972)
        model=build_model(settings['backbone'],settings['heads'][head])
        train,val=flow.make_loaders(settings)
        return Trainer(model,train,val,settings,CheckpointManager(str(path)),torch.device('cpu'),head_type=head,stage='pretrain')
    try:
        reference=create(1,tmp_path/'reference')
        reference.train(model_prefix='reference')
        deferred=create(20,tmp_path/'deferred')
        deferred.train(model_prefix='deferred')
        assert deferred.history==reference.history
        for key,value in reference.model.state_dict().items():
            assert torch.equal(value,deferred.model.state_dict()[key]),key
        # Trigger a stop with queued scalars, not at a periodic logging boundary.
        broken=create(20,tmp_path/'broken')
        stop=tmp_path/'stop.request'
        broken.config['runtime']={'stop_file':str(stop)}
        original_step=broken.optimizer.step
        count=0
        def step(*args,**kwargs):
            nonlocal count
            result=original_step(*args,**kwargs)
            count+=1
            if count==2:stop.touch()
            return result
        broken.optimizer.step=step
        with pytest.raises(TrainingStopped):broken.train(model_prefix='broken')
        stop.unlink()
        resumed=create(20,tmp_path/'resumed')
        start=resumed.resume_from_checkpoint(broken.last_saved_ckpt)
        resumed.train(start_epoch=start,model_prefix='resumed')
        assert resumed.history==reference.history
        for key,value in reference.model.state_dict().items():
            assert torch.equal(value,resumed.model.state_dict()[key]),key
    finally:
        torch.set_num_threads(threads)
