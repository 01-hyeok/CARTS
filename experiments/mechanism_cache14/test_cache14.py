import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cache_store import ExactBlockCache, array_info, digest, exact_equal, file_sha, scope
from common_ops import CommonOperations
from scoped_gate import assess, fingerprint


def proof(operation='oracle100'):
    return scope(cell='ETTh1_96', split='test', query_ids_sha256='q',
                 channel_ids_sha256='c', candidate_ids_sha256='n', pool_ids_sha256='p',
                 mask_sha256='m', data_sha256='d', normalization_sha256='norm',
                 code_sha256='code', runtime={'engine':'numpy CPU', 'dtype':'float32'},
                 seed=0, batch_shape=[2,7,128], operation=operation)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent));self.root=Path(self.tmp.name)
        self.cache=ExactBlockCache(self.root)
        self.p=proof();self.a={'ids':np.arange(1400,dtype=np.int64).reshape(2,7,100),
                              'values':np.arange(1400,dtype=np.float32).reshape(2,7,100)}
    def tearDown(self):
        assert Path(self.tmp.name).resolve().parent == Path(__file__).resolve().parent
        self.tmp.cleanup()
    def test_exact_roundtrip(self):
        self.assertEqual(self.cache.put(self.p,self.a),'STORED')
        got=self.cache.load(self.p,expected=self.a)
        self.assertTrue(all(exact_equal(got[k],a) for k,a in self.a.items()))
    def test_nan_signed_zero(self):
        a={'value':np.array([0.,-0.,np.nan],np.float32)}
        self.cache.put(self.p,a);self.assertTrue(exact_equal(a['value'],self.cache.load(self.p)['value']))
    def test_reordered_ids_rejected(self):
        self.cache.put(self.p,self.a);e=copy.deepcopy(self.a);e['ids']=e['ids'][...,::-1]
        self.assertIsNone(self.cache.load(self.p,expected=e))
    def test_changed_value_rejected(self):
        self.cache.put(self.p,self.a);e=copy.deepcopy(self.a);e['values'][0,0,0]=1
        self.assertIsNone(self.cache.load(self.p,expected=e))
    def test_changed_dtype_rejected(self):
        self.cache.put(self.p,self.a);e=copy.deepcopy(self.a);e['values']=e['values'].astype('float64')
        self.assertIsNone(self.cache.load(self.p,expected=e))
    def test_changed_provenance(self):
        self.cache.put(self.p,self.a)
        for k in ['cell','query_ids_sha256','channel_ids_sha256','candidate_ids_sha256','pool_ids_sha256','mask_sha256','data_sha256','normalization_sha256','code_sha256']:
            p=dict(self.p);p[k]+='_changed';self.assertIsNone(self.cache.load(p))
    def test_runtime_isolation(self):
        self.cache.put(self.p,self.a);p=dict(self.p,runtime={'engine':'torch1'});self.assertIsNone(self.cache.load(p))
    def test_split_isolation(self):
        self.cache.put(self.p,self.a);self.assertIsNone(self.cache.load(dict(self.p,split='val')))
    def test_shape_isolation(self):
        self.cache.put(self.p,self.a);self.assertIsNone(self.cache.load(dict(self.p,batch_shape=[1,7,128])))
    def test_truncated_payload_recomputes_preserves(self):
        self.cache.put(self.p,self.a);f=next(self.root.glob('*/arrays.npz'));f.write_bytes(b'broken')
        got,state=self.cache.get_or_compute(self.p,lambda:self.a)
        self.assertEqual(state,'STORED');self.assertEqual(f.read_bytes(),b'broken')
        self.assertIsNotNone(self.cache.load(self.p,expected=self.a))
    def test_manifest_corruption(self):
        self.cache.put(self.p,self.a);next(self.root.glob('*/manifest.json')).write_text('{}')
        self.assertIsNone(self.cache.load(self.p))
    def test_unpublished_generation_ignored(self):
        (self.root/'unpublished').mkdir();self.assertIsNone(self.cache.load(self.p))
    def test_unsafe_pointer(self):
        (self.root/(digest(self.p)+'.index.json')).write_text(json.dumps({'generation':'../x'}))
        self.assertIsNone(self.cache.load(self.p))
    def test_dense_optin(self):
        a={'d':np.zeros((2,7,128),np.float32)}
        self.assertEqual(self.cache.put(proof('future_mse'),a),'BYPASS_DENSE_DISABLED')
    def test_candidate_future_tensor_forbidden(self):
        self.cache.dense=True
        with self.assertRaises(ValueError):self.cache.put(proof('future_mse'),{'d':np.zeros((2,7,128,96))})
    def test_dense_bound(self):
        self.cache.dense=True;self.cache.max_entry_bytes=100
        self.assertEqual(self.cache.put(proof('future_mse'),{'d':np.zeros((2,7,128))}),'BYPASS_CAPACITY_RECOMPUTE')
    def test_total_bound(self):
        self.cache.max_total_bytes=1;self.assertEqual(self.cache.put(self.p,self.a),'BYPASS_CAPACITY_RECOMPUTE')
    def test_busy_never_waits_or_deletes(self):
        p=self.root/'cache.writer.lock';p.write_text('existing writer')
        self.assertEqual(self.cache.put(self.p,self.a),'BYPASS_WRITER_BUSY');self.assertTrue(p.exists())
    def test_hit_does_not_call_compute(self):
        self.cache.put(self.p,self.a)
        got,state=self.cache.get_or_compute(self.p,lambda:(_ for _ in ()).throw(AssertionError('called')))
        self.assertEqual(state,'HIT')
    def test_object_forbidden(self):
        with self.assertRaises(ValueError):self.cache.put(self.p,{'a':np.array([object()])})
    def test_missing_provenance(self):
        p=dict(self.p);p.pop('mask_sha256')
        with self.assertRaises(ValueError):self.cache.load(p)
    def test_model_dependent_scope_forbidden(self):
        with self.assertRaises(ValueError):self.cache.load(dict(self.p,checkpoint='weights'))
    def test_oracle_shared_operation(self):
        ops=CommonOperations(self.cache);d=np.arange(128,dtype=np.float32)[None,None].repeat(2,0).repeat(7,1)
        mask=np.ones_like(d,dtype=bool);ids=np.broadcast_to(np.arange(128),d.shape)
        oracle=np.broadcast_to(np.arange(100,dtype=np.int64),(2,7,100))
        got,_=ops.oracle(self.p,lambda:oracle,d,mask,ids,expected=oracle)
        got,state=ops.oracle(self.p,lambda:(_ for _ in ()).throw(AssertionError('called')),d,mask,ids)
        self.assertEqual(state,'HIT');self.assertTrue(exact_equal(got,oracle))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent));r=Path(self.tmp.name);self.old=r/'old';self.new=r/'new'
        self.old.mkdir();self.new.mkdir();self.ckpt=r/'ckpt';self.ckpt.write_bytes(b'weights')
        src='X=1\ndef reproduce():\n return X\ndef analyze():\n return 10\n'
        for root in [self.old,self.new]:(root/'worker.py').write_text(src)
        self.identity={'checkpoint_sha256':file_sha(self.ckpt),'data':{'counts':{'val':2,'test':3},'channels':7},
                       'pool_hashes':{'test':'pool'},'source_hashes':{'worker.py':file_sha(self.old/'worker.py')}}
        self.gate={'status':'PASS','identity':copy.deepcopy(self.identity),'rows':[]}
        for split,n in [('val',2),('test',3)]:
            self.gate['rows'].append(dict(status='PASS',split=split,queries=n,channels=7,units=n*7,ordered_mismatch_count=0,
                ordered_id_sha256='abc',canonical_ordered_id_sha256='abc',source_checkpoint_sha256=file_sha(self.ckpt)))
        self.gp=r/'gate.json';self.gp.write_text(json.dumps(self.gate));self.runtime={
            'torch_version':'2.4.1','cuda_runtime':'12.1','matmul_tf32':False,
            'device_uuid':'GPU-test','dtype':'float32','external_dependencies_sha256':'deps','blas_libraries_sha256':'blas'}
    def tearDown(self):
        assert Path(self.tmp.name).resolve().parent == Path(__file__).resolve().parent
        self.tmp.cleanup()
    def run_assess(self):
        self.identity['source_hashes']['worker.py']=file_sha(self.new/'worker.py')
        return assess(self.gate,self.identity,self.old,self.new,self.runtime,self.runtime,self.ckpt,self.gp)
    def test_unchanged_eligible_not_activated(self):
        x=self.run_assess();self.assertTrue(x['status'].startswith('ELIGIBLE'));self.assertFalse(x['production_activation'])
    def test_analysis_only_reuse(self):
        p=self.new/'worker.py';p.write_text(p.read_text().replace('return 10','return 20'))
        self.assertTrue(self.run_assess()['status'].startswith('ELIGIBLE'))
    def test_reproduce_change_rejected(self):
        p=self.new/'worker.py';p.write_text(p.read_text().replace('return X','return X+1'))
        self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_global_change_rejected(self):
        p=self.new/'worker.py';p.write_text(p.read_text().replace('X=1','X=2'))
        self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_default_side_effect_rejected(self):
        p=self.new/'worker.py';p.write_text(p.read_text().replace('analyze()','analyze(x=print(1))'))
        self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_checkpoint_tamper(self):
        self.ckpt.write_bytes(b'new');self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_missing_population(self):
        self.gate['rows'].pop();self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_wrong_units(self):
        self.gate['rows'][0]['units']=1;self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_wrong_ids(self):
        self.gate['rows'][0]['canonical_ordered_id_sha256']='different';self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_missing_source(self):
        (self.old/'worker.py').unlink();self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_protocol_or_pool_change(self):
        self.identity['pool_hashes']['test']='changed';self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_unknown_runtime(self):
        self.runtime={'torch':'UNKNOWN'};self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_runtime_change(self):
        r=assess(self.gate,self.identity,self.old,self.new,self.runtime,{'torch':'1.12'},self.ckpt,self.gp)
        self.assertEqual(r['status'],'RECOMPUTE_REQUIRED')
    def test_runtime_inventory_missing(self):
        self.runtime.pop('blas_libraries_sha256');self.assertEqual(self.run_assess()['status'],'RECOMPUTE_REQUIRED')
    def test_unlisted_adapter_change(self):
        self.assertNotEqual(fingerprint('def x():return 1','adapter.py'),fingerprint('def x():return 2','adapter.py'))
    def test_current_source_tamper(self):
        self.identity['source_hashes']['worker.py']='wrong'
        r=assess(self.gate,self.identity,self.old,self.new,self.runtime,self.runtime,self.ckpt,self.gp)
        self.assertEqual(r['status'],'RECOMPUTE_REQUIRED')


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        p=Path(self.tmp.name)/'source';p.write_bytes(b'source')
        self.kw=dict(cell='ETTh1_96',split='test',query_ids=np.arange(2),channel_ids=np.arange(7),expected_channels=7,
            candidate_ids=np.arange(128),pool_ids=np.broadcast_to(np.arange(100),(2,7,100)).copy(),
            mask=np.ones((2,7,128),bool),x=np.ones((2,7,96),np.float32),y=np.ones((2,7,96),np.float32),
            normalization={'train_mean':[0.],'train_std':[1.]},data_files={'dataset':p},dependency_files={'code':p},
            runtime={'engine':'CPU test'},operation='future_mse')
    def tearDown(self):
        assert Path(self.tmp.name).resolve().parent==Path(__file__).resolve().parent
        self.tmp.cleanup()
    def get(self):
        from provenance import block_proof
        return block_proof(**self.kw)
    def test_actual_values_affect_key(self):
        a=self.get();self.kw['y'][0,0,0]+=1;self.assertNotEqual(a,self.get())
    def test_channel_reduction_rejected(self):
        self.kw['expected_channels']=21
        with self.assertRaises(AssertionError):self.get()
    def test_duplicate_pool_rejected(self):
        self.kw['pool_ids'][0,0,0]=1
        with self.assertRaises(AssertionError):self.get()
    def test_invalid_mask_rejected(self):
        self.kw['mask'][0,0,0]=False
        with self.assertRaises(AssertionError):self.get()
    def test_pool_order_is_key(self):
        a=self.get();self.kw['pool_ids']=self.kw['pool_ids'][...,::-1]
        self.assertNotEqual(a,self.get())
    def test_changed_data_file_is_key(self):
        a=self.get();self.kw['data_files']['dataset'].write_bytes(b'changed');self.assertNotEqual(a,self.get())
    def test_wrong_bank_id_rejected(self):
        self.kw['candidate_ids']=np.arange(128)[::-1]
        with self.assertRaises(AssertionError):self.get()
    def test_query_alignment_is_key(self):
        a=self.get();self.kw['query_ids']=np.arange(2)[::-1];self.assertNotEqual(a,self.get())


if __name__=='__main__':unittest.main(verbosity=2)
