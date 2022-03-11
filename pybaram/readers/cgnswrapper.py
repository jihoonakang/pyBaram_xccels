# -*- coding: utf-8 -*-
from ctypes import POINTER, create_string_buffer, c_char_p, c_int, c_void_p

from pybaram.utils.ctypes import load_lib


# Possible CGNS exception types
CGNSError = type('CGNSError', (Exception,), {})
CGNSNodeNotFound = type('CGNSNodeNotFound', (CGNSError,), {})
CGNSIncorrectPath = type('CGNSIncorrectPath', (CGNSError,), {})
CGNSNoIndexDim = type('CGNSNoIndexDim', (CGNSError,), {})


class CGNSWrapper(object):
    # Possible return codes
    _statuses = {
        -1: CGNSError,
        -2: CGNSNodeNotFound,
        -3: CGNSIncorrectPath,
        -4: CGNSNoIndexDim
    }

    def __init__(self):
        # Load CGNS 3.3+
        # Previous version may yield undefined symbols from HDF5.
        # Develop branch after the commit (e0faea6) fixes this issue.
        self.lib = lib = load_lib('cgns')

        # Constants (from cgnslib.h)
        self.CG_MODE_READ = 0
        self.RealDouble = 4
        self.Unstructured = 3
        self.PointRange, self.ElementRange = 4, 6
        self.PointList, self.ElementList = 2, 7

        # cg_open
        lib.cg_open.argtypes = [c_char_p, c_int, POINTER(c_int)]
        lib.cg_open.errcheck = self._errcheck

        # cg_close
        lib.cg_close.argtypes = [c_int]
        lib.cg_close.errcheck = self._errcheck

        # cg_base_read
        lib.cg_base_read.argtypes = [c_int, c_int, c_char_p, POINTER(c_int),
                                     POINTER(c_int)]
        lib.cg_base_read.errcheck = self._errcheck

        # cg_nzones
        lib.cg_nzones.argtypes = [c_int, c_int, POINTER(c_int)]
        lib.cg_nzones.errcheck = self._errcheck

        # cg_zone_read
        lib.cg_zone_read.argtypes = [c_int, c_int, c_int, c_char_p,
                                     POINTER(c_int)]
        lib.cg_zone_read.errcheck = self._errcheck

        # cg_zone_type
        lib.cg_zone_type.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_zone_type.errcheck = self._errcheck

        # cg_coord_read
        lib.cg_coord_read.argtypes = [
            c_int, c_int, c_int, c_char_p, c_int, POINTER(c_int),
            POINTER(c_int), c_void_p
        ]
        lib.cg_coord_read.errcheck = self._errcheck

        # cg_nbocos
        lib.cg_nbocos.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_nbocos.errcheck = self._errcheck

        # cg_boco_info
        lib.cg_boco_info.argtypes = [
            c_int, c_int, c_int, c_int, c_char_p, POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]
        lib.cg_boco_info.errcheck = self._errcheck

        # cg_boco_read
        lib.cg_boco_read.argtypes = [c_int, c_int, c_int, c_int,
                                     POINTER(c_int), c_void_p]
        lib.cg_boco_read.errcheck = self._errcheck

        # cg_nsections
        lib.cg_nsections.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_nsections.errcheck = self._errcheck

        # cg_section_read
        lib.cg_section_read.argtypes = [
            c_int, c_int, c_int, c_int, c_char_p, POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]
        lib.cg_section_read.errcheck = self._errcheck

        # cg_ElementDataSize
        lib.cg_ElementDataSize.argtypes = [c_int, c_int, c_int, c_int,
                                           POINTER(c_int)]
        lib.cg_ElementDataSize.errcheck = self._errcheck

        # cg_elements_read
        lib.cg_elements_read.argtypes = [c_int, c_int, c_int, c_int,
                                         c_void_p, c_void_p]
        lib.cg_elements_read.errcheck = self._errcheck

    def _errcheck(self, status, fn, arg):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CGNSError

    def open(self, name):
        file = c_int()
        self.lib.cg_open(bytes(name, 'utf-8'), self.CG_MODE_READ, file)
        return file

    def close(self, file):
        self.lib.cg_close(file)

    def base_read(self, file, idx):
        celldim, physdim = c_int(), c_int()
        name = create_string_buffer(32)

        self.lib.cg_base_read(file, idx + 1, name, celldim, physdim)

        return {'file': file, 'idx': idx + 1,
                'name': name.value.decode('utf-8'),
                'CellDim': celldim.value, 'PhysDim': physdim.value}

    def nzones(self, base):
        n = c_int()
        self.lib.cg_nzones(base['file'], base['idx'], n)
        return n.value

    def zone_read(self, base, idx):
        zonetype = c_int()
        name = create_string_buffer(32)
        size = (c_int * 3)()

        self.lib.cg_zone_read(base['file'], base['idx'], idx + 1, name, size)

        # Check zone type
        self.lib.cg_zone_type(base['file'], base['idx'], idx + 1, zonetype)
        if zonetype.value != self.Unstructured:
            raise RuntimeError('ReadCGNS_read: Incorrect zone type for file')

        return {'base': base, 'idx': idx + 1,
                'name': name.value.decode('utf-8'),
                'size': list(size)}

    def coord_read(self, zone, name, x):
        i = c_int(1)
        j = c_int(zone['size'][0])

        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        # The data type does not need to be the same as the one in which the
        # coordinates are stored in the file
        # http://cgns.github.io/CGNS_docs_current/midlevel/grid.html
        datatype = self.RealDouble

        self.lib.cg_coord_read(file, base, zone, bytes(name, 'utf-8'),
                               datatype, i, j, x.ctypes.data)

    def nbocos(self, zone):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']
        n = c_int()

        self.lib.cg_goto(file, base, b'Zone_t', 1, b'ZoneBC_t', 1, b'end')
        self.lib.cg_nbocos(file, base, zone, n)
        return n.value

    def boco_read(self, zone, idx):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        name = create_string_buffer(32)
        bocotype = c_int()
        ptset_type = c_int()
        npnts = c_int()
        normalindex = (c_int * 3)()
        normallistsize = c_int()
        normaldatatype = c_int()
        ndataset = c_int()

        self.lib.cg_boco_info(
            file, base, zone, idx + 1, name, bocotype, ptset_type, npnts,
            normalindex, normallistsize, normaldatatype, ndataset
        )

        val = (c_int * npnts.value)()
        self.lib.cg_boco_read(file, base, zone, idx + 1, val, None)

        if ptset_type.value in [self.PointRange, self.ElementRange]:
            bc_range = tuple(val)
            bc_list = list(range(bc_range[0], bc_range[1]+1))
        elif ptset_type.value in [self.PointList, self.ElementList]:
            bc_list = list(val)
            bc_range = min(bc_list), max(bc_list)
        else:
            raise RuntimeError('Only range/list BC is supported')

        return {'name': name.value.decode('utf-8'),
                'range': bc_range, 'list': bc_list}

    def nsections(self, zone):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        n = c_int()
        self.lib.cg_nsections(file, base, zone, n)

        return n.value

    def section_read(self, zone, idx):
        file = zone['base']['file']
        base = zone['base']['idx']
        zidx = zone['idx']

        name = create_string_buffer(32)
        etype, start, end, nbdry = c_int(), c_int(), c_int(), c_int()
        pflag, cdim = c_int(), c_int()

        self.lib.cg_section_read(
            file, base, zidx, idx + 1, name, etype, start, end, nbdry, pflag
        )

        self.lib.cg_ElementDataSize(file, base, zidx, idx + 1, cdim)

        return {'zone': zone, 'idx': idx + 1, 'dim': cdim.value,
                'etype': etype.value, 'range': (start.value, end.value)}

    def elements_read(self, sect, conn):
        file = sect['zone']['base']['file']
        base = sect['zone']['base']['idx']
        zone = sect['zone']['idx']
        idx = sect['idx']

        self.lib.cg_elements_read(file, base, zone, idx,
                                  conn.ctypes.data, None)
