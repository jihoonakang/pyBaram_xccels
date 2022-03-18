# -*- coding: utf-8 -*-
import numba as nb
import math


def make_parallel_loop1d(ne, func, n0=0):
    # Get number of threads
    n = ne - n0
    num_threads = nb.get_num_threads()
    num_per_thread = int(math.ceil(n / num_threads))

    # Compile func
    _func = nb.jit(nopython=True, fastmath=True)(func)
    
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def loop(*args):        
        for index_thread in nb.prange(num_threads):
            i_begin = n0 + index_thread * num_per_thread
            i_end = n0 + min(n, (index_thread + 1) * num_per_thread)
            
            _func(i_begin, i_end, *args)               
                
    return loop
            
            
def make_serial_loop1d(ne, func, n0=0):
    # Compile func
    _func = nb.jit(nopython=True, fastmath=True)(func)

    @nb.jit(nopython=True, fastmath=True)
    def loop(*args):
        _func(n0, ne, *args)
            
    return loop
