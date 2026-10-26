import io,unittest,tempfile,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .loss import temperature_kl
from .adapter import compute_loss,forward_eval,on_training_complete,state_hash


class DummyAdapter:
    @staticmethod
    def forward_eval(bundle,batch):return bundle['net'](batch['input'])

class DistillationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(6)

    def bundle(self,mode='kd',weight=.3,threshold=.7):
        teacher=nn.Sequential(nn.Linear(5,6),nn.BatchNorm1d(6),nn.ReLU(),nn.Dropout(.5),nn.Linear(6,3)).eval().requires_grad_(False)
        return {'net':nn.Linear(5,3),'teacher':teacher,'teacher_adapter':DummyAdapter,'mode':mode,'weight':weight,'temperature':4.,'epsilon':.1,'settings':{'confidence_threshold':threshold},'teacher_initial_state_sha256':state_hash(teacher),'teacher_checkpoint_sha256':'test-only','steps':0,'statistics':{k:torch.zeros((),dtype=torch.float64) for k in ('ce_sum','kd_sum','total_sum','samples','accepted','teacher_correct')}}

    def test_kl_direction_batch_reduction_and_temperature(self):
        student=torch.randn(4,3,requires_grad=True);teacher=torch.randn(4,3,requires_grad=True)
        value,mask=temperature_kl(student,teacher,4.)
        expected=F.kl_div(F.log_softmax(student/4,1),F.softmax(teacher.detach()/4,1),reduction='batchmean')*16
        torch.testing.assert_close(value,expected);self.assertTrue(mask.all())
        value.backward();self.assertIsNone(teacher.grad);self.assertIsNotNone(student.grad)

    def test_identical_distributions_have_zero_kl(self):
        logits=torch.randn(4,15);loss,_=temperature_kl(logits,logits)
        self.assertLess(abs(loss.item()),2e-6)

    def test_all_rejected_is_ce_only_and_finite(self):
        bundle=self.bundle('confidence',threshold=1.0)
        batch={'input':torch.randn(4,5),'label':torch.tensor([0,1,2,0])}
        logits=bundle['net'](batch['input']);ce=nn.CrossEntropyLoss()
        loss=compute_loss(bundle,logits,batch,ce)
        torch.testing.assert_close(loss,ce(logits,batch['label']),rtol=0,atol=0)
        loss.backward();self.assertEqual(bundle['statistics']['accepted'].item(),0)

    def test_zero_weight_is_exact_original_ce(self):
        bundle=self.bundle(weight=0);bundle.pop('teacher');bundle.pop('teacher_adapter')
        batch={'input':torch.randn(4,5),'label':torch.tensor([0,1,2,0])};logits=bundle['net'](batch['input']);ce=nn.CrossEntropyLoss()
        torch.testing.assert_close(compute_loss(bundle,logits,batch,ce),ce(logits,batch['label']),rtol=0,atol=0)

    def test_teacher_weights_buffers_and_gradients_unchanged(self):
        bundle=self.bundle();before=state_hash(bundle['teacher']);optimizer=torch.optim.AdamW(bundle['net'].parameters(),lr=1e-4)
        for _ in range(2):
            batch={'input':torch.randn(4,5),'label':torch.tensor([0,1,2,0])};optimizer.zero_grad()
            compute_loss(bundle,bundle['net'](batch['input']),batch,nn.CrossEntropyLoss()).backward();optimizer.step()
        self.assertEqual(before,state_hash(bundle['teacher']))
        self.assertTrue(all(p.grad is None for p in bundle['teacher'].parameters()));self.assertFalse(bundle['teacher'].training)
        on_training_complete(bundle)

    def test_diagnostics_do_not_retain_autograd_graph(self):
        bundle=self.bundle('label_smoothing')
        for _ in range(2):
            batch={'input':torch.randn(4,5),'label':torch.tensor([0,1,2,0])}
            compute_loss(bundle,bundle['net'](batch['input']),batch,nn.CrossEntropyLoss()).backward()
        self.assertTrue(all(v.grad_fn is None and not v.requires_grad for v in bundle['statistics'].values()))

    def test_invalid_temperature_and_shape_fail(self):
        with self.assertRaises(ValueError):temperature_kl(torch.randn(2,3),torch.randn(3,3))
        with self.assertRaises(ValueError):temperature_kl(torch.randn(2,3),torch.randn(2,3),0)

    def test_saved_student_has_no_teacher_and_original_eval_interface(self):
        from ..transmixer_fusion.model import FusionTransMixerNet
        net=FusionTransMixerNet('F4').eval();stream=io.BytesIO();torch.save(net,stream);stream.seek(0)
        saved=torch.load(stream,weights_only=False)
        batch={'hsi_pca':torch.randn(1,1,30,11,11),'aux':torch.randn(1,1,11,11)}
        with torch.no_grad():torch.testing.assert_close(forward_eval({'net':saved},batch),net(batch['hsi_pca'],batch['aux']))
        self.assertFalse(any('teacher' in name for name in saved.state_dict()))

    def test_native_config_and_registry_keep_student_initialization(self):
        from src.config import ExperimentConfig,ConfigManager
        if 'distillation' not in ExperimentConfig.__dataclass_fields__:self.skipTest('Core hook integration deliberately deferred until existing training ends')
        from src.model_registry import get_model_adapter
        from ..cnn.model import CNNet
        from ..transmixer_fusion.model import FusionTransMixerNet
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'untrained_test_teacher.pth';torch.save(CNNet(30,1,128,15),path)
            settings={'teacher_checkpoint':str(path),'teacher_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weight':.3,'temperature':4}
            torch.manual_seed(6);reference=FusionTransMixerNet('F4');expected_rng=torch.get_rng_state()
            torch.manual_seed(6);cfg=ExperimentConfig(model_name='TransMixer_KD_F4_CNN',dataset_type=0,distillation=settings)
            adapter=get_model_adapter(cfg.model_name);bundle=adapter.build_model(ConfigManager(cfg),0,torch.device('cpu'))
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v,bundle['net'].state_dict()[k]) for k,v in reference.state_dict().items()))
            batch={'hsi_pca':torch.randn(2,1,30,11,11),'aux':torch.randn(2,1,11,11),'label':torch.tensor([0,1])}
            logits=adapter.forward_train(bundle,batch);adapter.compute_loss(bundle,logits,batch,nn.CrossEntropyLoss()).backward()
            on_training_complete(bundle)

    def test_native_config_round_trip_retains_kd_settings(self):
        from src.config import ExperimentConfig,ConfigManager
        if 'distillation' not in ExperimentConfig.__dataclass_fields__:self.skipTest('Awaiting core hook integration')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json';cfg=ExperimentConfig(model_name='TransMixer_Control_F4_LabelSmooth',dataset_type=0,distillation={'label_smoothing':.1})
            ConfigManager(cfg).save_config(str(path));manager=ConfigManager();manager.load_config(str(path))
            self.assertEqual(manager.config.distillation,{'label_smoothing':.1})
        self.assertIsNone(ExperimentConfig().distillation)

if __name__=='__main__':unittest.main()
