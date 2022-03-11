# -*- coding: utf-8 -*-
import os
import sys
import time


class Progressbar:
    def __init__(self, t0, te, fmt='{:.2f}'):
        self._rintv = 1/(te-t0)
        self._fmt = fmt

        # Get column size
        try:
            self._col = int(os.popen('stty size', 'r').read().split()[1])
        except:
            self._col = 80

        # Mark initial time
        self._tinit = time.time()

    def __call__(self, t):
        # progress
        prog = t*self._rintv

        # Mark time
        elapsed = time.time() - self._tinit
        eta_txt = "ETA: {0:05g}/{1:05g}".format(elapsed, 1 / prog * elapsed)

        # Status
        status_txt = self._fmt.format(t)

        # Calculate length
        length = self._col - 5 - len(eta_txt) - len(status_txt)

        # Bar
        bcur = max(int(round(length*prog)) - 1, 0)
        brem = max(length - bcur - 1, 0)
        bartxt = "="*bcur + ">" + " "*brem
        bar = "\r[{0}] ".format(bartxt)

        # Text
        text = ' '.join([bar, status_txt, eta_txt])
        sys.stdout.write(text)
        sys.stdout.flush()
