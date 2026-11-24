"""Independent mathematical checks, CPU FP64 only; these do not qualify CUDA implementations."""
import math
import unittest
import torch
from kernel_agent.operators import references as ref


class ReferenceTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(771)

    def test_softmax_extremes_masks_and_translation(self):
        x=torch.tensor([[10000.,9999.,-10000.],[0.,0.,0.]],dtype=torch.float64)
        p=ref.safe_softmax(x);torch.testing.assert_close(p,torch.softmax(x,-1));torch.testing.assert_close(p,ref.safe_softmax(x+12345))
        mask=torch.tensor([[True,False,True],[False,False,False]])
        actual=ref.safe_softmax(x,mask);self.assertTrue(torch.isfinite(actual).all());torch.testing.assert_close(actual[1],torch.zeros(3,dtype=torch.float64));self.assertEqual(actual[0,1],0.)

    def test_online_partition_invariance_and_empty_identity(self):
        scores=torch.randn(137,dtype=torch.float64)*20;values=torch.randn(137,19,dtype=torch.float64)
        scores[3:19]=-torch.inf;expected=torch.softmax(scores,-1)@values;lse=torch.logsumexp(scores,0)
        for tile in [1,7,32,64,500]:
            out,actual_lse=ref.online_attention(scores,values,tile);torch.testing.assert_close(out,expected,rtol=2e-13,atol=2e-13);torch.testing.assert_close(actual_lse,lse)
        for x,v in [(scores[:0],values[:0]),(torch.full_like(scores,-torch.inf),values)]:
            out,actual_lse=ref.online_attention(x,v,17);self.assertTrue(torch.isfinite(out).all());self.assertTrue((out==0).all());self.assertEqual(float(actual_lse),-math.inf)

    def test_attention_bottom_right_and_mask(self):
        q=torch.randn(2,3,4,8,dtype=torch.float64);k=torch.randn(2,3,7,8,dtype=torch.float64);v=torch.randn(2,3,7,5,dtype=torch.float64)
        actual=ref.attention(q,k,v,causal=True)
        for b in range(2):
            for h in range(3):
                for s in range(4):
                    visible=s+7-4+1;logits=k[b,h,:visible]@q[b,h,s]/math.sqrt(8);expected=torch.softmax(logits,0)@v[b,h,:visible];torch.testing.assert_close(actual[b,h,s],expected)
        masked=ref.attention(q,k,v,mask=torch.zeros(4,7,dtype=torch.bool));self.assertTrue((masked==0).all())

    def test_mla_absorption_and_latent_core(self):
        b,h,l,r,d,dr,dv=3,4,17,11,7,6,9
        q=torch.randn(b,h,d,dtype=torch.float64);qr=torch.randn(b,h,dr,dtype=torch.float64);c=torch.randn(b,l,r,dtype=torch.float64);kr=torch.randn(b,l,dr,dtype=torch.float64)
        uk=torch.randn(h,r,d,dtype=torch.float64)*.1;uv=torch.randn(h,r,dv,dtype=torch.float64)*.1;lengths=[0,9,17];scale=(d+dr)**-.5
        explicit=ref.mla_layer_explicit(q,qr,c,kr,uk,uv,lengths,scale);absorbed,lse=ref.mla_layer_absorbed(q,qr,c,kr,uk,uv,lengths,scale)
        torch.testing.assert_close(explicit,absorbed,rtol=2e-13,atol=2e-13);self.assertTrue(torch.isneginf(lse[0]).all());self.assertTrue((absorbed[0]==0).all())

    def test_rope_positions_norm_and_inner_product(self):
        x=torch.randn(2,5,8,dtype=torch.float64);positions=torch.tensor([0,17])
        actual=ref.rotary(x,positions);torch.testing.assert_close(actual.square().sum(-1),x.square().sum(-1));torch.testing.assert_close(actual[0],x[0])
        y=torch.randn_like(x);rotated_y=ref.rotary(y,positions);torch.testing.assert_close((actual*rotated_y).sum(-1),(x*y).sum(-1))


if __name__=='__main__':unittest.main()
