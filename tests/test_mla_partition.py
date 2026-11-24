import random,unittest
from kernel_agent.operators.partition import split_page_table


class PartitionTests(unittest.TestCase):
    def test_physical_token_order_and_no_overlap_across_empty_ragged_tails(self):
        rng=random.Random(771)
        for page_size in [1,16,64]:
            lengths=[0,1,15,16,17,127,257,513];ptr=[0]
            for n in lengths:ptr.append(ptr[-1]+(n+page_size-1)//page_size)
            indices=list(range(ptr[-1]));rng.shuffle(indices)
            expected=[[indices[ptr[b]+t//page_size]*page_size+t%page_size for t in range(n)] for b,n in enumerate(lengths)]
            for splits in [1,2,4]:
                actual=[[] for _ in lengths]
                for part in split_page_table(lengths,ptr,indices,page_size,splits):
                    for b,n in enumerate(part['lengths']):actual[b].extend(part['indices'][part['indptr'][b]+t//page_size]*page_size+t%page_size for t in range(n))
                self.assertEqual(actual,expected)
    def test_invalid_configuration_and_page_count_rejected(self):
        for page_size,splits,ptr in [(0,1,[0,1]),(16,0,[0,1]),(16,True,[0,1]),(16,2,[0,0])]:
            with self.subTest(page_size=page_size,splits=splits),self.assertRaises(ValueError):split_page_table([1],ptr,[0],page_size,splits)


if __name__=='__main__':unittest.main()
