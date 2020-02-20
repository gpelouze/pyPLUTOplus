#!/usr/bin/env python3

''' Helper tool to generate hybrid grids containing uniform and stretched
segments.

Example
=======

Generate a grid with 3 uniform segments of fixed cell size joined by
stretched grid segments. In each stretched grid segment, cell size varies
from the cell size of one uniform segment to the other.

Table: Characteristics of the 5 segments in this example.

Position [Mm]   Type        Resolution [km]
--------------- ----------- --------------
0.0-3.0         uniform     300
3.0-3.5         stretched   300 to 30
3.5-4.5         uniform     300
4.5-5.0         stretched   30 to 300
5.0-100.0       uniform     300


>>> g = Grid((
...     0.0, 'u', 300e-3,
...     3.0, 's',
...     3.5, 'u', 30e-3,
...     4.5, 's',
...     5.0, 'u', 300e-3,
...     100,
...     ))
>>> print('Pluto definition:', g.pluto_definition)
    Pluto definition: Xi-grid 5  0.0 10 u  3.0 5 s  3.5 33 u  4.5 5 s  5.0 317 u  100.0
>>> print('Number of points:', g.N)
    Number of points: 370

Then add the contents of `g.pluto_definition` to the `[Grid]` block
of your `pluto.ini`, after specifying the appropriate dimension number.
Eg. `Xi-grid ...` → `X1-grid ...`.


Additional verifications
========================

Additional verifications are performed to ensure that the grid can be used by
PLUTO (v4.3). Warnings are displayed when the following criteria are not met.

- Ensure that the grid definitions is shorter than 128 characters.
  Longer definitions are truncated by PLUTO.
- For stretch segments, ensure that r <= 1.2. If r > 1.2, PLUTO will display
  "WARNING:  alpha=%12.6e > 1.05" (sic! see `Src/set_grid.c:521`).
  This tool displays a warning if r > 1.2 and if r < 1/1.2.

'''

import warnings

import numpy as np


class UniformGridSegment():
    def __init__(self, xL, xR, Δx):
        self.xL = xL
        self.xR = xR
        self.Δx = Δx

    @property
    def N(self):
        return int(np.round((self.xR - self.xL) / self.Δx))

    @property
    def Δx_arr(self):
        return np.full(self.N, self.Δx)

    @property
    def x(self):
        return self.xL - self.Δx / 2 + np.cumsum(self.Δx_arr)

    @property
    def pluto_definition(self):
        return f'{self.xL:0.1f} {self.N} u'

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.xL}, '
                f'{self.xR}, '
                f'{self.Δx})')


class StretchedGridSegment():
    def __init__(self, xL, xR, ΔxL, ΔxR):
        self.xL = xL
        self.xR = xR
        self.ΔxL = ΔxL
        self.ΔxR = ΔxR

        r_limit = 1.2
        if self.r > r_limit:
            warnings.warn(f'r = {self.r:.2f} > {r_limit:.1f}')
        if self.r < 1 / r_limit:
            warnings.warn(f'r = {self.r:.2f} < {1 / r_limit:.1f}')

    @property
    def r(self):
        return ((self.xR - self.xL + self.ΔxR) /
                (self.xR - self.xL + self.ΔxL))

    @property
    def C(self):
        return (self.xR - self.xL) / self.ΔxL

    @property
    def N(self):
        return int(np.round(
            np.log(self.r + self.C*self.r - self.C) / np.log(self.r) - 1))

    @property
    def Δx_arr(self):
        Δx_values = np.array([self.ΔxL * self.r**(i) for i in range(self.N+1)])
        return (Δx_values[1:] + Δx_values[:-1]) / 2

    @property
    def x(self):
        x = self.xL - self.ΔxL / 2 + np.cumsum(self.Δx_arr)
        # return (x[1:] + x[:-1]) / 2
        return x

    @property
    def pluto_definition(self):
        return f'{self.xL:0.1f} {self.N} s'

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.xL}, '
                f'{self.xR}, '
                f'{self.ΔxL}, '
                f'{self.ΔxR})')


class Grid():
    def __init__(self, segments_def):
        '''
        segments_def : tuple
            tuple containing grid segment definitions in the format:
                xL (u Δx|s)   [...]   x_end
        '''
        self.segments = self._parse_segments_def(segments_def)
        self._check_segments()

    def _parse_segments_def(self, segments_def):
        segments = []
        segments_def = list(segments_def)[::-1]
        Δx = None
        xR = segments_def.pop()
        segment_type = None
        while segments_def:
            ΔxL = Δx
            Δx = None
            xL = xR
            xR = None
            segment_type_L = segment_type
            segment_type = segments_def.pop()
            if segment_type == 'u':
                Δx = segments_def.pop()
                try:
                    xR = segments_def.pop()
                except IndexError:
                    ''' this happens when segments_def ends with:
                            ..., 'u', <float>) '''
                    raise ValueError('invalid format for uniform grid')
                segment = UniformGridSegment(xL, xR, Δx)
            elif segment_type == 's':
                xR = segments_def.pop()
                try:
                    segment_type_R = segments_def[-1]
                except IndexError:
                    raise ValueError('stretch segments must lay between '
                                     'two uniform segments')
                if isinstance(segment_type_R, (int, float, np.number)):
                    raise ValueError('invalid format for stretched grid')
                if (segment_type_L != 'u') or (segment_type_R != 'u'):
                    raise ValueError('stretched segments must lay between '
                                     'two uniform segments')
                ΔxR = segments_def[-2]
                segment = StretchedGridSegment(xL, xR, ΔxL, ΔxR)
            else:
                raise ValueError(f'invalid segment type: {segment_type}')
            segments.append(segment)
        return segments

    def _check_segments(self):
        xL = self.segments[0].xL
        for segment in self.segments:
            xR = segment.xR
            if xL >= xR:
                raise ValueError('segment boundaries must be increasing')
            xL = xR

    @property
    def N(self):
        return sum([s.N for s in self.segments])

    @property
    def Δx_arr(self):
        return np.hstack([s.Δx_arr for s in self.segments])

    @property
    def x(self):
        return np.hstack([s.x for s in self.segments])

    @property
    def pluto_definition(self):
        segments_defs = '  '.join([s.pluto_definition for s in self.segments])
        segments_count = len(self.segments)
        grid_xR = self.segments[-1].xR
        pluto_def = f'Xi-grid {segments_count} {segments_defs} {grid_xR:0.1f}'
        if len(pluto_def) > 127:
            warnings.warn('grid definition is longer than 127 characters. '
                          'This is not supported by PLUTO<=4.3.')
        return pluto_def

    def __repr__(self):
        segments_repr = ', '.join([repr(s) for s in self.segments])
        return f'{self.__class__.__name__}({segments_repr})'
