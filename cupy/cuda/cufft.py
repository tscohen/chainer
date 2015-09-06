"""Thin wrapper of CUFFT."""
import ctypes
import sys

from cupy.cuda import internal

if 'win32' == sys.platform:
    _cufft = internal.load_library(
        internal.get_windows_cuda_library_names('cufft'))
else:
    _cufft = internal.load_library('cufft')

_I = ctypes.c_int
_P = ctypes.c_void_p

CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

CUFFT_COMPATIBILITY_NATIVE = 0x00
CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
CUFFT_COMPATIBILITY_FFTW_ALL = 0x03

Plan = _P
Stream = _P

###############################################################################
# Error handling
###############################################################################

STATUS = {
    0: 'CUFFT_SUCCESS',
    1: 'CUFFT_INVALID_PLAN',
    2: 'CUFFT_ALLOC_FAILED',
    3: 'CUFFT_INVALID_TYPE',
    4: 'CUFFT_INVALID_VALUE',
    5: 'CUFFT_INTERNAL_ERROR',
    6: 'CUFFT_EXEC_FAILED',
    7: 'CUFFT_SETUP_FAILED',
    8: 'CUFFT_INVALID_SIZE',
    9: 'CUFFT_UNALIGNED_DATA',
    10: 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    11: 'CUFFT_INVALID_DEVICE',
    12: 'CUFFT_PARSE_ERROR',
    13: 'CUFFT_NO_WORKSPACE',
}


class CUFFTError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUFFTError, self).__init__(STATUS[status])


def check_status(status):
    if status != 0:
        raise CUFFTError(status)


###############################################################################
# Plan
###############################################################################

_cufft.cufftPlan1d.argtypes = (_P, _I, _I, _I)


def cufftPlan1d(nx, fft_type, batch=1):

    plan = Plan()
    status = _cufft.cufftPlan1d(ctypes.byref(plan), nx, fft_type, batch)
    check_status(status)
    return plan


_cufft.cufftPlan2d.argtype = (_P, _I, _I, _I)


def cufftPlan2d(nx, ny, fft_type):

    plan = Plan()
    status = _cufft.cufftPlan2d(ctypes.byref(plan), nx, ny, fft_type)
    check_status(status)
    return plan


_cufft.cufftPlan3d.argtype = (_P, _I, _I, _I, _I)


def cufftPlan3d(nx, ny, nz, fft_type):

    plan = Plan()
    status = _cufft.cufftPlan3d(ctypes.byref(plan), nx, ny, nz, fft_type)
    check_status(status)
    return plan


_cufft.cufftPlanMany.argtype = (_P, _I, _P, _P, _I, _I, _P, _I, _I, _I, _I)


def cufftPlanMany(rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, fft_type, batch):

    plan = Plan()
    status = _cufft.cufftPlanMany(ctypes.byref(plan), rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, batch)
    check_status(status)
    return plan


_cufft.cufftDestroy.argtype = (Plan,)


def cufftDestroy(plan):
    status = _cufft.cufftDestroy(plan)
    check_status(status)


_cufft.cufftSetCompatibilityMode.argtypes = (Plan, _I)


def cufftSetCompatibilityMode(plan, mode):
    status = _cufft.cufftSetCompatibilityMode(plan, mode)
    check_status(status)


###############################################################################
# Execute
###############################################################################

_cufft.cufftExecC2C.argtypes = (Plan, _P, _P, _I)


def cufftExecC2C(plan, idata, odata, direction):
    status = _cufft.cufftExecC2C(plan, idata, odata, direction)
    check_status(status)


_cufft.cufftExecR2C.argtypes = (Plan, _P, _P)


def cufftExecR2C(plan, idata, odata):
    status = _cufft.cufftExecR2C(plan, idata, odata)
    check_status(status)


_cufft.cufftExecC2R.argtypes = (Plan, _P, _P)


def cufftExecC2R(plan, idata, odata):
    status = _cufft.cufftExecC2R(plan, idata, odata)
    check_status(status)


_cufft.cufftExecZ2Z.argtypes = (Plan, _P, _P, _I)


def cufftExecZ2Z(plan, idata, odata, direction):
    status = _cufft.cufftExecZ2Z(plan, idata, odata, direction)
    check_status(status)


_cufft.cufftExecD2Z.argtypes = (Plan, _P, _P)


def cufftExecD2Z(plan, idata, odata):
    status = _cufft.cufftExecD2Z(plan, idata, odata)
    check_status(status)


_cufft.cufftExecZ2D.argtypes = (Plan, _P, _P)


def cufftExecZ2D(plan, idata, odata):
    status = _cufft.cufftExecZ2D(plan, idata, odata)
    check_status(status)


###############################################################################
# Stream
###############################################################################

_cufft.cufftSetStream.argtypes = (Plan, Stream)


def cufftSetStream(plan, stream):
    status = _cufft.cufftSetStream(plan, stream)
    check_status(status)
