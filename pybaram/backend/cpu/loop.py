# -*- coding: utf-8 -*-
import numba as nb
import math


def make_parallel_loop1d(n, func):
    # Get number of threads
    num_threads = nb.get_num_threads()
    num_per_thread = int(math.ceil(n / num_threads))
    
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def loop(*args):       
        
        for index_thread in nb.prange(num_threads):
            i_begin = index_thread * num_per_thread
            i_end = min(n, (index_thread + 1) * num_per_thread)
            
            func(i_begin, i_end, *args)               
                
    return loop
            
            
def make_serial_loop1d(n, func):

    @nb.jit(nopython=True, fastmath=True)
    def loop(*args):
        func(0, n, *args)
            
    return loop
