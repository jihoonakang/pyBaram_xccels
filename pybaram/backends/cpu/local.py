from numba import types
from numba.extending import intrinsic
from numba.core import cgutils


@intrinsic
def stack_empty_impl(typingctx,size,dtype):
    # Ref : https://github.com/numba/numba/issues/5084

    def impl(context, builder, signature, args):
        ty=context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty,size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64,dtype)
    return sig, impl
