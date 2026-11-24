"""Drain two subprocess pipes with byte/time limits and cleanup on host I/O failures."""
import os,selectors,subprocess,time
from pathlib import Path


def capture(cmd,directory,*,timeout,stdout_limit,stderr_limit,cleanup):
    if min(timeout,stdout_limit,stderr_limit)<=0:raise ValueError('positive capture limits required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();reason=None;error_text=None;counts={'stdout':0,'stderr':0};observed={'stdout':0,'stderr':0}
    selector=selectors.DefaultSelector();files={};limits={'stdout':stdout_limit,'stderr':stderr_limit}
    try:
        # No client/worker exists until all trusted host log files are available.
        for name in counts:files[name]=(directory/(name+'.bin')).open('wb')
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    except BaseException:
        selector.close()
        for file in files.values():
            try:file.close()
            except OSError:pass
        raise
    try:
        for name,pipe in [('stdout',proc.stdout),('stderr',proc.stderr)]:
            os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,name)
        while selector.get_map():
            if time.monotonic()-started>=timeout:reason='timeout';break
            for key,_ in selector.select(timeout=min(.1,max(.001,timeout-(time.monotonic()-started)))):
                data=os.read(key.fileobj.fileno(),65536);name=key.data
                if not data:selector.unregister(key.fileobj);continue
                observed[name]+=len(data);remaining=limits[name]-counts[name]
                accepted=data[:remaining];files[name].write(accepted);counts[name]+=len(accepted)
                if len(data)>remaining:reason=name+'_limit';break
            if reason:break
        if not reason:
            try:proc.wait(timeout=max(.001,timeout-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:reason='timeout'
    except (OSError,ValueError) as error:
        reason='host_io_error';error_text=str(error)[:512]
    finally:
        # Break attach backpressure before asking Docker to remove the owned container.
        try:
            selector.close()
            for pipe in [proc.stdout,proc.stderr]:
                try:pipe.close()
                except OSError as error:reason=reason or 'host_io_error';error_text=str(error)[:512]
            for file in files.values():
                try:file.close()
                except OSError as error:reason=reason or 'host_io_error';error_text=str(error)[:512]
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:reason=reason or 'client_stop_timeout'
        finally:
            try:cleanup_result=cleanup(reason)
            except Exception as error:
                # Never lose an unknown cleanup state and accidentally start fallback.
                cleanup_result={'removed':False,'error_type':type(error).__name__,'error':str(error)[:512]}
        for name in counts:
            try:counts[name]=(directory/(name+'.bin')).stat().st_size
            except OSError:pass
    return {'exit':proc.returncode,'reason':reason,'host_io_error':error_text,'stored_bytes':counts,'observed_bytes':observed,
            'limits':limits,'timeout_seconds':timeout,'wall_seconds':time.monotonic()-started,'cleanup':cleanup_result}


def confirmed_removed(result):
    message=result.get('stderr','').lower()
    return result.get('exit')==1 and ('no such object' in message or 'no such container' in message)
