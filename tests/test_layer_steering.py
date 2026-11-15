import unittest
import pandas as pd
import torch
from scripts.layer_steering import (
    VARIANTS, LayerConfig, geometry, intervention_vectors, steering_wrapper, validation_subset,
)
from tests import test_global_refusal as fixtures


class LayerSteeringTest(unittest.TestCase):
    def setUp(self):
        self.raw = {0:torch.tensor([3.,0.]),1:torch.tensor([0.,5.])}

    def test_cosines_and_budget_invariance(self):
        cosine, norms = geometry(self.raw)
        torch.testing.assert_close(torch.tensor(cosine.values),torch.eye(2))
        self.assertEqual(norms.raw_norm.tolist(),[3.,5.])
        for variant in VARIANTS:
            vectors = intervention_vectors(self.raw,0,variant,6.)
            self.assertAlmostEqual(sum(float(v.norm()) for v in vectors.values()),6.)
        self.assertTrue(torch.equal(intervention_vectors(self.raw,0,'shared_direction',6.)[1],torch.tensor([3.,0.])))
        self.assertTrue(torch.equal(intervention_vectors(self.raw,0,'layer_specific',6.)[1],torch.tensor([0.,3.])))

    def test_subset_uses_validation_only(self):
        data = pd.DataFrame({'prompt':[f'p{i}' for i in range(16)],
                             'label':['safe','harmful']*8,
                             'split':['train']*4+['validation']*8+['test']*4})
        cfg = LayerConfig(validation_per_class=2)
        subset = validation_subset(data,cfg)
        self.assertEqual(subset.label.value_counts().to_dict(),{'safe':2,'harmful':2})
        self.assertTrue(subset.split.eq('validation').all())
        pd.testing.assert_frame_equal(subset,validation_subset(data,cfg))

    def test_actual_safetune_hooks_and_restoration(self):
        fixture = fixtures.GlobalRefusalTest(); fixture.setUp(); model = fixture.model
        original = {k:v.clone() for k,v in model.state_dict().items()}
        raw = {0:torch.randn(16),1:torch.randn(16)}
        inputs = {'input_ids':torch.tensor([[2,4,7]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
        for variant in VARIANTS:
            actual = {}; handles = []
            vectors = intervention_vectors(raw,0,variant,4.)
            for layer in vectors:
                def before(module,args,output,li=layer): actual[li] = output.detach().clone()
                handles.append(model.model.layers[layer].register_forward_hook(before))
            try:
                with self.assertRaisesRegex(RuntimeError,'interrupt'):
                    with steering_wrapper(model,raw,0,variant,4.):
                        for layer in vectors:
                            def after(module,args,output,li=layer):
                                torch.testing.assert_close(output-actual[li],
                                    vectors[li].view(1,1,-1).expand_as(output),atol=1e-6,rtol=1e-5)
                            handles.append(model.model.layers[layer].register_forward_hook(after))
                        model(**inputs)
                        raise RuntimeError('interrupt')
            finally:
                for handle in handles: handle.remove()
            self.assertFalse(any(m._forward_hooks for m in model.modules()))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,original[k]))


if __name__ == '__main__':unittest.main()
