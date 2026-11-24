"""Small adapter to existing Linux rlimits and libseccomp; fixed trusted workers only."""
import ctypes,errno,os,resource,socket


def restrict():
    status=open('/proc/self/status').read()
    fields={k:next(line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith(k+':')) for k in ['CapEff','CapBnd','NoNewPrivs']}
    if os.getuid()!=65534 or os.getgid()!=65534 or os.getgroups() or fields!={'CapEff':'0000000000000000','CapBnd':'0000000000000000','NoNewPrivs':'1'}:raise RuntimeError('worker privilege boundary invalid')
    for kind,limit in [(resource.RLIMIT_CORE,0),(resource.RLIMIT_NOFILE,256),(resource.RLIMIT_FSIZE,16*1024*1024),(resource.RLIMIT_NPROC,128)]:resource.setrlimit(kind,(limit,limit))
    resource.setrlimit(resource.RLIMIT_CPU,(35,40))
    lib=ctypes.CDLL('libseccomp.so.2');lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    class Compare(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Compare)]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x7fff0000)
    if not ctx:raise RuntimeError('seccomp context unavailable')
    try:
        # SCMP_CMP_NE: keep local Unix IPC used by the CUDA driver, deny every other family.
        number=lib.seccomp_syscall_resolve_name(b'socket');comparison=Compare(0,1,socket.AF_UNIX,0)
        if number<0 or lib.seccomp_rule_add_array(ctx,0x00050000|errno.EPERM,number,1,ctypes.byref(comparison))!=0:raise RuntimeError('seccomp network rule unavailable')
        if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp network boundary unavailable')
    finally:lib.seccomp_release(ctx)
    for family in [socket.AF_INET,socket.AF_INET6]:
        try:socket.socket(family,socket.SOCK_STREAM);raise RuntimeError('IP socket boundary not enforced')
        except OSError as error:
            if error.errno!=errno.EPERM:raise
    local=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);local.close()
    return {'uid':os.getuid(),'gid':os.getgid(),'capabilities_zero':True,'no_new_privs':True,'ipv4_socket_denied':True,'ipv6_socket_denied':True,'unix_ipc_available':True,'file_limit_bytes':16*1024*1024,'open_files_limit':256,'process_limit':128,'cpu_soft_seconds':35,'cpu_hard_seconds':40}
