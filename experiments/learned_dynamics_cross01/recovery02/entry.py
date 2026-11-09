from recovery_runtime import *
import traceback,runpy
if __name__=='__main__':
    try:
        seal_check();protect()
        assert read(OUT/'PREFLIGHT.json')['status']=='PASS'
        assert read(OUT/'PREFLIGHT.json')['source_seal_sha256']==sha(OUT/'SOURCE_SEAL.json')
        lock=hold(ORIGINAL/'GPU.lock');gpu_ready()
        import compat;compat.install()
        mode=sys.argv[1];args=sys.argv[2:]
        if mode=='resume':
            import resume_weather;resume_weather.run()
        elif mode=='native':
            import run_experiment as r
            assert len(args)==2 and args[1]=='FULL' and args[0] in ('Weather_192','Weather_336','Weather_720','ETTh1_192','ETTh1_336')
            r.Experiment(*args).run()
        elif mode in ('pool','cross'):
            sys.path.insert(0,str(CODE/'core/supplemental'))
            import run_supplement as s
            cell,training=args[:2];pool=args[2] if len(args)>2 else None
            s.Supplement(cell,mode,training,pool).run()
        else:raise ValueError(mode)
    except BaseException as e:
        dump(OUT/('FAILURE_'+datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),at=now()));raise
