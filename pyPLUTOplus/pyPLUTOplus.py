#!/usr/bin/env python3

from collections import OrderedDict
import configparser
import contextlib
import datetime
import os
import re

import numpy as np
import pyPLUTO as pp
import scipy.interpolate as sint
import tqdm


class pload(pp.pload):
    ''' extend pyPLUTO.pload '''
    def get_var(self, varname):
        ''' Programmatically access a variable '''
        return self.__getattribute__(varname)


class CONST():
    ''' PLUTO constants: CONST_* values as defined in pluto.h '''
    AH     = 1.008              # Atomic weight of Hydrogen
    AHe    = 4.004              # Atomic weight of Helium
    AZ     = 30.0               # Mean atomic weight of heavy elements
    amu    = 1.66053886e-24     # Atomic mass unit
    au     = 1.49597892e13      # Astronomical unit
    c      = 2.99792458e10      # Speed of Light
    e      = 4.80320425e-10     # Elementary (proton) charge
    eV     = 1.602176463158e-12 # Electron Volt in erg
    G      = 6.6726e-8          # Gravitational Constant
    h      = 6.62606876e-27     # Planck Constant
    kB     = 1.3806505e-16      # Boltzmann constant
    ly     = 0.9461e18          # Light year
    mp     = 1.67262171e-24     # Proton mass
    mn     = 1.67492728e-24     # Neutron mass
    me     = 9.1093826e-28      # Electron mass
    mH     = 1.6733e-24         # Hydrogen atom mass
    Msun   = 2.e33              # Solar Mass
    Mearth = 5.9736e27          # Earth Mass
    NA     = 6.0221367e23       # Avogadro Contant
    pc     = 3.0856775807e18    # Parsec
    PI     = 3.14159265358979   # \f$ \pi \f$
    Rearth = 6.378136e8         # Earth Radius
    Rgas   = 8.3144598e7        # Perfect gas constant
    Rsun   = 6.96e10            # Solar Radius
    sigma  = 5.67051e-5         # Stephan Boltmann constant
    sigmaT = 6.6524e-25         # Thomson Cross section


class PlutoGridDimension():
    ''' Represent the grid along a given dimension (eg. X1-grid) '''
    def __init__(self, xL, xR, n_ghosts=0):
        ''' Initialize the grid dimension

        Parameters
        ==========
        xL : 1D array of shape (n,)
            Coordinates of the left side of each cell.
        xR : 1D array of shape (n,)
            Coordinates of the left side of each cell.
        n_ghosts : int (default: 0)
            Number of ghost points.
        '''
        if len(xL) != len(xR):
            raise ValueError('xL and xR must have the same length')
        self.N = len(xL)
        self.i = np.arange(self.N)
        self.xL = xL
        self.xR = xR
        self.n_ghosts = n_ghosts

    @property
    def x(self):
        ''' Coordinates of the center of each cell. '''
        return (self.xL + self.xR) / 2

    def iter_cells(self):
        ''' Returns an iterator which yields the indice and the left, center,
        and right coordinate of each grid cell. '''
        return zip(self.i, self.xL, self.x, self.xR)

    def __repr__(self):
        return f'PlutoGridDimension({self.x})'


class _PlutoGridDimensionBuilder():
    ''' Helper class to build a PlutoGridDimension.

    This is a dependency of `PlutoGridReader` that shouldn't be used directly.
    '''
    def __init__(self, n_points):
        self.N = n_points
        self.i = []
        self.xL = []
        self.xR = []

    def append(self, i, xL, xR):
        self.i.append(i)
        self.xL.append(xL)
        self.xR.append(xR)

    def check_consistency(self):
        if not (len(self.i) == len(self.xL) == len(self.xR) == self.N):
            raise ValueError('unconsistent grid sizes')
        if not np.all(np.array(self.i) == np.arange(self.N) + 1):
            raise ValueError('uneven grid indices')

    def to_PlutoGridDimension(self):
        self.check_consistency()
        return PlutoGridDimension(np.array(self.xL), np.array(self.xR))


class PlutoGrid():
    ''' Represent the grid used for a simulation. '''

    _supported_dimensions = (1, 2, 3)
    _supported_geometries = ('cartesian', 'cylindrical', 'spherical', 'polar')

    def __init__(self, n_dimensions, geometry, x1, x2, x3):
        ''' Create a new PlutoGrid

        Parameters
        ==========
        n_dimensions : int
            The number of dimensions (1, 2, or 3).
        geometry : str
            The geometry of the simulation ('cartesian', 'cylindrical',
            'spherical', or 'polar').
        x1, x2, x2 : PlutoGridDimension, 1D array, or None
            The coordinates along each dimension.
            - If passed a PlutoGridDimension, it is used directly.
            - If passed an array, it is assumed to contain the coordinate of
              the center of each grid points.
            - Pass None for inactive dimensions.
        '''
        if n_dimensions not in self._supported_dimensions:
            raise ValueError(f'unsupported n_dimension value: {n_dimensions}')
        geometry = geometry.lower()
        if geometry not in self._supported_geometries:
            raise ValueError(f'unsupported geometry: {geometry}')
        self.n_dimensions = n_dimensions
        self.geometry = geometry
        self.x1 = self._init_dimension(x1)
        self.x2 = self._init_dimension(x2)
        self.x3 = self._init_dimension(x3)

    @property
    def active_dimensions(self):
        if self.n_dimensions == 1:
            return (self.x1, )
        if self.n_dimensions == 2:
            return (self.x1, self.x2)
        if self.n_dimensions == 3:
            return (self.x1, self.x2, self.x3)

    @property
    def all_dimensions(self):
        return self.x1, self.x2, self.x3

    def _init_dimension(self, x):
        if isinstance(x, PlutoGridDimension):
            return x
        else:
            if x is None:
                # assume one cell of size 1 centered on 0.5
                xL = np.array([0])
                xR = np.array([1])
            elif len(x) == 1:
                # assume a cell of size 1 centered on x[0]
                xL = np.array([x[0] - .5])
                xR = np.array([x[0] + .5])
            else:
                x_face = (x[1:] + x[:-1]) / 2
                dx = x[1:] - x[:-1]
                xL = np.full_like(x, np.nan)
                xL[1:] = x_face
                xL[0] = x_face[0] - dx[0]
                xR = np.full_like(x, np.nan)
                xR[:-1] = x_face
                xR[-1] = x_face[-1] + dx[-1]
                assert np.all(np.isfinite(xL))
                assert np.all(np.isfinite(xR))
            return PlutoGridDimension(xL, xR)

    def write_to(self, filename):
        ''' Write the grid to a file (usually `grid.out`) using
        `PlutoGridWriter.write_to()`. '''
        writer = PlutoGridWriter()
        writer.write_to(self, filename)

    def __repr__(self):
        repr_string = 'PLUTO grid:\n'
        repr_string += f'# DIMENSIONS: {self.n_dimensions:d}\n'
        repr_string += f'# GEOMETRY: {self.geometry}\n'
        for i, dim in enumerate(self.active_dimensions):
            repr_string += (f'# X{i+1}: [ {dim.xL[0]:.6f},  {dim.xR[-1]:.6f}], '
                            f'{dim.N:d} point(s), {dim.n_ghosts:d} ghosts')
        return repr_string


class PlutoGridReader():
    ''' Read a PLUTO grid.out file. '''

    def __init__(self, data_dir):
        ''' Parse and store the grid.

        Parameters
        ==========
        data_dir : str
            The PLUTO data output directory, containing `grid.out`.
        '''
        self.data_dir = data_dir
        self._grid_out_fname = os.path.join(self.data_dir, 'grid.out')

    def _raise_parse_line_error(self, line):
        msg = 'unexpected line in grid file: {}'.format(line)
        raise ValueError(msg)

    def read(self):
        ''' Read the `{data_dir}/grid.out` file and return a PlutoGrid object.
        '''
        dimension_builders = []
        current_dimension_builder = None
        with open(self._grid_out_fname, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#'):
                    ''' comment '''
                    line = line.strip('#').strip()
                    if line.lower().startswith('dimensions'):
                        ''' likely 'DIMENSIONS: (int)' '''
                        try:
                            _, ndim = line.split(':')
                            n_dimensions = int(ndim)
                        except (TypeError, ValueError):
                            self._raise_parse_line_error(line)
                    if line.lower().startswith('geometry'):
                        ''' likely 'GEOMETRY: (str)' '''
                        try:
                            _, geometry = line.split(':')
                            geometry_name = geometry.lower().strip()
                        except (TypeError, ValueError):
                            self._raise_parse_line_error(line)
                elif len(line.split()) == 1:
                    ''' Likely the start of a new dimension. Line should
                    contain a single int indicating the number of points in
                    this dimension. '''
                    try:
                        n_points = int(line)
                    except ValueError:
                        self._raise_parse_line_error(line)
                    current_dimension_builder = _PlutoGridDimensionBuilder(n_points)
                    dimension_builders.append(current_dimension_builder)
                elif len(line.split()) == 3:
                    ''' Likely (i, xR, xL) triplet for the current dimension
                    '''
                    i, xL, xR = line.split()
                    try:
                        i = int(i)
                        xL = float(xL)
                        xR = float(xR)
                    except ValueError:
                        self._raise_parse_line_error(line)
                    current_dimension_builder.append(i, xL, xR)
                else:
                    self._raise_parse_line_error(line)
        if len(dimension_builders) != 3:
            raise ValueError('invalid number of dimensions')
        for builder in dimension_builders:
            builder.to_PlutoGridDimension()
        dimensions = [builder.to_PlutoGridDimension()
                      for builder in dimension_builders]
        grid = PlutoGrid(n_dimensions, geometry_name, *dimensions)
        return grid


class PlutoGridWriter():
    ''' Write a grid.out file '''
    def to_string(self, grid):
        ''' Convert a PlutoGrid to a string that can later be written to a
        file (usually `grid.out`). '''
        now = datetime.datetime.now()
        string = ('# ******************************************************\n'
                  '# PLUTO 4.3 Grid File\n'
                  f'# Generated on  {now:%a %b %d %H:%M:%S %Y}\n'
                  '# \n'
                  )
        string += f'# DIMENSIONS: {grid.n_dimensions:d}\n'
        string += f'# GEOMETRY:   {grid.geometry.upper()}\n'
        for i, dim in enumerate(grid.active_dimensions):
            string += (f'# X{i+1}: [ {dim.xL[0]:.6f},  {dim.xR[-1]:.6f}], '
                       f'{dim.N:d} point(s), {dim.n_ghosts:d} ghosts\n')
        string += '# ******************************************************\n'
        for dim in grid.all_dimensions:
            string += f'{dim.N:d} \n'
            for i, xL, x, xR in dim.iter_cells():
                string += f' {i+1:d}   {xL:.12e}    {xR:.12e}\n'
        return string

    def write_to(self, grid, filename):
        ''' Write a PlutoGrid to a file (usually `grid.out`). '''
        with open(filename, 'w') as f:
            f.write(self.to_string(grid))


class PlutoIni(configparser.ConfigParser):
    ''' Represent `pluto.ini` file. '''
    def __init__(self, ini_dir):
        ''' Parse and store `pluto.ini` file

        Parameters
        ==========
        ini_dir : str
            Directory containing a valid `pluto.ini`.
        '''
        self.ini_dir = ini_dir
        self.pluto_ini = os.path.join(self.ini_dir, 'pluto.ini')
        super().__init__(delimiters=(' ',))
        self.read(self.pluto_ini)


class PlutoDefinitions(dict):
    ''' Represent PLUTO `definitions.h` file. '''
    def __init__(self, ini_dir):
        ''' Parse and store PLUTO `definitions.h` file

        Parameters
        ==========
        ini_dir : str
            Directory containing a valid PLUTO `definitions.h`.
        '''
        self.ini_dir = ini_dir
        self.definitions_h = os.path.join(self.ini_dir, 'definitions.h')
        self.define_statements = self.load_define_statements()
        define_values = self._eval_define_statements(self.define_statements)
        super().__init__(define_values)
        self._set_defaults()

    def load_define_statements(self):
        ''' Load the `#define` statements in `definitions.h`. '''
        with open(self.definitions_h, 'r') as f:
            definitions_str = f.read()
        definitions_str = self._remove_comments(definitions_str)
        return self._get_define_statements(definitions_str)

    def _remove_comments(self, string):
        ''' Remove .h comments from string '''
        string, _ = re.subn(r'//.*?\n', '\n', string)
        string, _ = re.subn(r'/\*(.|\n)*?\*/', '', string)
        return string

    def _get_define_statements(self, string):
        ''' Parse `#define` statements from a .h string. '''
        define_statements = []
        _define_re = re.compile(r'^\s*#define\s*(\w+)\s*(.*)')
        for line in string.split('\n'):
            m = _define_re.match(line)
            if m:
                define_statements.append(m.groups())
        return define_statements

    def _eval_define_statements(self, define_statements):
        ''' Convert `#define` statements values from string to Python types.
        '''
        define_values = []
        for k, v in define_statements:
            v = self._eval_define_statement_value(v)
            define_values.append((k, v))
        return define_values

    def _eval_define_statement_value(self, value):
        ''' Convert a `#define` statements value into Python types using regex
        substitution and `eval()`. '''
        context_vars = dict(
            CONST=CONST,
            sqrt=np.sqrt,
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            )
        try:
            value, _ = re.subn(r'CONST_', 'CONST.', value)
            value, _ = re.subn(r'\^', '**.', value)
            value, _ = re.subn(r'YES', 'True', value)
            value, _ = re.subn(r'NO', 'False', value)
            return eval(value, context_vars, dict())
        except Exception:
            return value

    def _get_given_type(self, varname, types):
        ''' Get the value of a `#define` statement if it has any of the listed
        `types`, or raise `ValueError` if it has another type.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        types : tuple
            A tuple containing acceptable dtypes.
        '''
        val = self[varname]
        if isinstance(val, types):
            return val
        else:
            raise ValueError("'{}' is not a number".format(val))

    def _set_defaults(self):
        if 'H_MASS_FRAC' not in self:
            self['H_MASS_FRAC'] = 0.7110
        if 'He_MASS_FRAC' not in self:
            if (not self['COOLING']) and (self['EOS'] == 'PVTE_LAW'):
                self['He_MASS_FRAC'] = 1 - self['H_MASS_FRAC']
            else:
                self['He_MASS_FRAC'] = 0.2741
        if 'Z_MASS_FRAC' not in self:
            self['Z_MASS_FRAC'] = 1 - self['H_MASS_FRAC'] - self['He_MASS_FRAC']
        return self

    def get_number(self, varname):
        ''' Get a variable it is a number, or raise ValueError.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        '''
        return self._get_given_type(varname, (int, float))

    def get_int(self, varname):
        ''' Get a variable it is an integer, or raise ValueError.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        '''
        return self._get_given_type(varname, (int, ))

    def get_float(self, varname):
        ''' Get a variable it is an float, or raise ValueError.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        '''
        return self._get_given_type(varname, (float, ))

    def get_bool(self, varname):
        ''' Get a variable it is an boolean, or raise ValueError.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        '''
        return self._get_given_type(varname, (bool, ))

    def get_str(self, varname):
        ''' Get a variable it is an string, or raise ValueError.

        Parameters
        ==========
        varname : str
            Name of a variable defined the `definitions.h`.
        '''
        return self._get_given_type(varname, (str, ))

    def __repr__(self):
        return 'PlutoDefinitions({})'.format(super().__repr__())


class PlutoUnits():
    ''' Code units as defined in `definitions.h`.

    Contains the PLUTO base units (length, velocity, and density), as well as
    derived units such as time or mass.

    Example
    =======

    >>> p = PlutoUnits('./my_sim/')
    >>> print(p.length) # prints the value of UNIT_LENGTH in cgs
    >>> print(p.time) # prints UNIT_LENGTH / UNIT_VELOCITY in cgs
    '''

    def __init__(self, ini_dir):
        ''' Parse and store the code units defined in `{ini_dir}/definitions.h`.

        Parameters
        ==========
        ini_dir : str
            Directory containing a valid PLUTO `definitions.h`.
        '''
        self.ini_dir = ini_dir
        self._defs = PlutoDefinitions(self.ini_dir)
        self.UNIT_DENSITY = self._defs.get_number('UNIT_DENSITY') # g cm-3
        self.UNIT_LENGTH = self._defs.get_number('UNIT_LENGTH') # cm
        self.UNIT_VELOCITY = self._defs.get_number('UNIT_VELOCITY') # cm s-1

    @property
    def _MU(self):
        ''' Mean molecular weight computed as in PLUTO's MeanMolecularWeight
        function (Src/mean_mol_weight.c).

        Only COOLING == NO or TABULATED are supported.
        '''
        cooling = self._defs.get('COOLING')
        if cooling in (False, 'TABULATED'):
            H_MASS_FRAC = self._defs.get_number('H_MASS_FRAC')
            He_MASS_FRAC = self._defs.get_number('He_MASS_FRAC')
            Z_MASS_FRAC = self._defs.get_number('Z_MASS_FRAC')
            FRAC_He = (He_MASS_FRAC/CONST.AHe*CONST.AH/H_MASS_FRAC)
            FRAC_Z = (Z_MASS_FRAC /CONST.AZ *CONST.AH/H_MASS_FRAC)
            return ((CONST.AH + FRAC_He*CONST.AHe + FRAC_Z*CONST.AZ) /
                    (2.0 + FRAC_He + FRAC_Z*(1.0 + CONST.AZ*0.5)))

        else:
            raise ValueError(f"Can't compute mu for {cooling} cooling.")


    @property
    def density(self):
        ''' density in g cm-3 '''
        return self.UNIT_DENSITY

    @property
    def length(self):
        ''' length in cm '''
        return self.UNIT_LENGTH

    @property
    def velocity(self):
        ''' velocity in cm s-1 '''
        return self.UNIT_VELOCITY

    @property
    def energy_density(self):
        ''' energy density in erg cm-2 '''
        return self.UNIT_DENSITY * self.UNIT_VELOCITY**2

    @property
    def number_density(self):
        ''' number density in cm-3 '''
        return self.UNIT_DENSITY / CONST.amu

    @property
    def magnetic_field(self):
        ''' magnetic field in G '''
        return np.sqrt(4*CONST.PI*self.UNIT_DENSITY*self.UNIT_VELOCITY**2)

    @property
    def mass(self):
        ''' mass in g '''
        return self.UNIT_DENSITY * self.UNIT_LENGTH**3

    @property
    def power_density(self):
        ''' power density in erg s-1 cm-2 '''
        return self.UNIT_DENSITY * self.UNIT_VELOCITY**3 / self.UNIT_LENGTH

    @property
    def pressure(self):
        ''' pressure in dyn cm-2 '''
        return self.UNIT_DENSITY * self.UNIT_VELOCITY**2

    @property
    def temperature(self):
        ''' temperature in K '''
        return self.UNIT_VELOCITY**2*CONST.amu/CONST.kB

    @property
    def time(self):
        ''' time in s '''
        return self.UNIT_LENGTH / self.UNIT_VELOCITY

    def __repr__(self):
        s = ('PlutoUnits('
             'length={length:.3g}, '
             'velocity={velocity:.3g}, '
             'density={density:.3g})')
        s = s.format(
            length=self.length,
            velocity=self.velocity,
            density=self.density)
        return s


class PlutoDataset():
    def __init__(self, ini_dir, data_dir=None, datatype=None, level=0,
                 x1range=None, x2range=None, x3range=None,
                 ns_values=None, last_ns=None, load_data=True):
        ''' Time series of PLUTO data

        Parameters
        ==========
        ini_dir : str
            Directory containing the pluto.ini and definition.h files.
            (This is not necessarily the directory containing the output data,
            i.e. output_dir defined in pluto.ini.)
        data_dir : str or None
            Directory containing PLUTO out and dbl files.
            If None, it is read from pluto.ini.
        datatype, level, x1range, x2range, x3range :
            Passed to pyPLUTO.pload and pyPLUTO.nlast_info.
            See pyPLUTO.pload documentation for more info.
        ns_values : list of int, or None
            Step numbers of the data files to include in the dataset.
            If None, load all data files up to last_ns (see next).
        last_ns : int or None (default: None)
            Step number of the last data file to include in the movie.
            If None, all data files are loaded (last_ns is then determined by
            pyPLUTO.nlast_info).
        load_data : bool (default: True)
            If True, load the data during init
        '''

        self.ini_dir = ini_dir
        self.data_dir = data_dir
        self.datatype = datatype
        self.level = level
        self.x1range = x1range
        self.x2range = x2range
        self.x3range = x3range

        if not os.path.isdir(self.ini_dir):
            raise NotADirectoryError(
                f'Ini dir is not a directory: {self.ini_dir}')

        self.ini = PlutoIni(self.ini_dir)
        self.units = PlutoUnits(self.ini_dir)
        self.definitions = PlutoDefinitions(self.ini_dir)

        # get data_dir from ini file
        if self.data_dir is None:
            self.data_dir = self.ini['Static Grid Output'].get('output_dir', '.')
        # Handle relative data dir definitions, eg.
        # /foo/bar -> /foo/bar/
        # ./foo/bar -> {self.ini_dir}/foo/bar/
        self.data_dir = re.sub(
            r'^\.(/|$)',
            self.ini_dir + '/',
            self.data_dir,
            )
        # pyPLUTO crashes if passed w_dir option without trailing slash
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError(
                f'Data dir is not a directory: {self.data_dir}')

        # load nlast_info
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.nlast_info = pp.nlast_info(
                w_dir=self.data_dir,
                datatype=self.datatype)

        # determine last_ns and ns_values
        if (ns_values is not None) and (last_ns is not None):
            raise ValueError('cannot set both ns_values and last_ns')
        if ns_values is not None:
            self.last_ns = None
            self.ns_values = ns_values
            # replace negative values
            sim_last_ns = self.nlast_info['nlast']
            self.ns_values = [
                sim_last_ns + ns + 1 if ns < 0
                else ns
                for ns in self.ns_values]
        else:
            if last_ns is None:
                last_ns = self.nlast_info['nlast']
            self.last_ns = last_ns
            self.ns_values = np.arange(0, self.last_ns+1)

        self._step_data= []
        if load_data:
            self.load_data()

    def load_data(self):
        ''' Load datafiles for all steps '''
        for ns in self.ns_values:
            self._step_data.append(self.load_step_data(ns))
        self._update_vars()

    def load_step_data(self, ns):
        ''' Load datafile from step ns '''
        return pload(
            ns,
            w_dir=self.data_dir,
            datatype=self.datatype,
            level=self.level,
            x1range=self.x1range,
            x2range=self.x2range,
            x3range=self.x3range,
            )

    def _update_vars(self):
        ''' Update variable names '''
        self.vars = self._step_data[0].vars
        # All step files should have the same variables list.
        # Use the first one and verify that it is the case.
        for this_step_data in self._step_data:
            if this_step_data.vars != self.vars:
                raise ValueError('inconsistent vars at different steps')

    def get_step(self, i):
        ''' Get pyPLUTO pload object for step i '''
        return self._step_data[i]

    def get_var(self, varname):
        ''' Get array of a variable at all time steps '''
        return np.array([sd.get_var(varname) for sd in self._step_data])

    def save_dbl(self, data_dir):
        ''' Save as dbl files.

        **kwargs are passed to DblWriter.write_to()
        '''
        writer = DblWriter()
        writer.write_to(self, data_dir, **kwargs)

    def _reshape_step(self, sd, new_coordinates):
        ''' Reshape pyPLUTO.pload object (used by .reshape()) '''

        new_x1, new_x2, new_x3 = new_coordinates
        old_x1, old_x2, old_x3 = sd.x1, sd.x2, sd.x3

        sd.x1 = new_x1
        sd.x2 = new_x2
        sd.x3 = new_x3

        sd.n1 = len(new_x1)
        sd.n2 = len(new_x2)
        sd.n3 = len(new_x3)

        sd.irange = list(range(sd.n1))
        sd.jrange = list(range(sd.n2))
        sd.krange = list(range(sd.n3))

        if self.ndim == 1:
            sd.nshp = (sd.n1, )
            self.x1r = None
        elif self.ndim == 2:
            sd.nshp = (sd.n1, sd.n2)
            self.x1r = None
            self.x2r = None
        elif self.ndim == 3:
            sd.nshp = (sd.n1, sd.n2, sd.n3)
            self.x1r = None
            self.x2r = None
            self.x3r = None
        else:
            raise ValueError(f'invalid ndim: {self.ndim}')

        for varname in self.vars:
            data = sd.__getattribute__(varname)
            if self.ndim == 1:
                new_data = sint.interp1d(old_x1, data)(new_x1)
            elif self.ndim == 2:
                new_data = sint.interp2d(old_x2, old_x1, data)(new_x2, new_x1)
            elif self.ndim == 3:
                new_points = np.array(np.meshgrid(new_x1, new_x2, new_x3)).T
                new_data = sint.interpn((old_x1, old_x2, old_x3), data, new_points)
            else:
                raise ValueError(f'invalid ndim: {self.ndim}')
            sd.__setattr__(varname, new_data)

    def reshape(self, new_shape):
        ''' Reshape dataset spatial coordinates

        Parameters
        ==========
        new_shape : tuple of int
            New shape for each axis of the dataset.
            This tuple must contain as many elements as there are dimensions in
            the dataset (as set in self.definitions['DIMENSIONS']).

        Example
        =======
        Reshape a 2D dataset such that x1 has 32 cells and x2 has 64:
        >>> self.reshape((32, 64))

        '''
        if len(new_shape) != self.ndim:
            raise ValueError(f'invalid new_shape size '
                             f'(expected {self.ndim}, got {len(new_shape)})')

        new_coordinates = [self.x1, self.x2, self.x3]
        for i, new_n in enumerate(new_shape):
            new_coordinates[i] = np.linspace(
                new_coordinates[i].min(),
                new_coordinates[i].max(),
                new_n,
                )

        for sd in tqdm.tqdm(self._step_data, desc='Reshaping dataset'):
            self._reshape_step(sd, new_coordinates)

    @property
    def Dt(self):
        return np.array([sd.Dt for sd in self._step_data])

    @property
    def t(self):
        return np.array([sd.SimTime for sd in self._step_data])

    @property
    def t_cgs(self):
        return self.t * self.units.time

    @property
    def x1(self):
        return self.get_step(0).x1

    @property
    def x2(self):
        return self.get_step(0).x2

    @property
    def x3(self):
        return self.get_step(0).x3

    @property
    def n1(self):
        return self.get_step(0).n1

    @property
    def n2(self):
        return self.get_step(0).n2

    @property
    def n3(self):
        return self.get_step(0).n3

    @property
    def ndim(self):
        return self.definitions['DIMENSIONS']

    def __getitem__(self, item):
        ''' Access step or var

        item : str or int
            str -> self.get_var(item)
            int -> self.get_step(item)
        '''
        if isinstance(item, str):
            return self.get_var(item)
        elif isinstance(item, int):
            return self.get_step(item)
        else:
            raise ValueError('invalid item type')


class DblDefinitions(dict):
    ''' Simplified clone of PlutoDefinitions to be used by DblDataset. '''
    def __init__(self, dimensions, geometry):
        define_values = {
            'DIMENSIONS': dimensions,
            'GEOMETRY': geometry.upper(),
            }
        super().__init__(define_values)


class DblStepData():
    ''' Time step of a DblDataset '''
    def __init__(self, data, coordinates, SimTime, Dt=np.nan):
        ''' Create a new dbl time step

        Parameters
        ===========
        data : dict of array
            Dictionary where keys are the variable names and values are arrays
            of the same shape containing the values of the corresponding
            variable in the simulation domain for the given snapshot.
        coordinates : tuple of 1D arrays
            A tuple of 1D arrays giving the coordinates along each active
            dimension.
        SimTime : float
            Simulation time for the snapshot.
        Dt : float or None
            Integration time step for the snapshot.
        '''
        self.Dt = Dt
        self.SimTime = SimTime
        # Append inactive dimensions to coordinates until the total number of
        # spatial coordinates is 3.
        # Inactive dimension coordinates are a single cell of size 1 centered
        # on 0.5.
        inactive_dimension_coordinate = np.array([0.5])
        coordinates = list(coordinates)  # convert from tuple
        while len(coordinates) < 3:
            coordinates.append(inactive_dimension_coordinate)
        self.x1, self.x2, self.x3 = coordinates

        for var_name, arr in data.items():
            self.__setattr__(var_name, arr)

        self.vars = list(data.keys())

    def get_var(self, varname):
        ''' Programmatically access a variable '''
        return self.__getattribute__(varname)


class DblDataset(PlutoDataset):
    ''' Dataset that can be written to a dbl file '''
    def __init__(self, data, coordinates, n_dimensions, geometry, Dt=None):
        ''' Create a new dbl dataset

        Parameters
        ==========
        data : dict of arrays
            Dictionary where keys are the variable names and values are arrays
            of dimension (1 + n_dimensions), where the first axis is time and
            the following are spatial dimensions. All arrays must have the same
            shape.
        coordinates : tuple of 1D arrays
            A tuple of 1D arrays giving time and the coordinates along each
            active dimension.
            Must contain (1 + n_dimensions) arrays.
        n_dimensions : int (1, 2, or 3)
            Number of dimensions in the simulation.
        geometry : str
            Geometry of the simulation ('cartesian', 'cylindrical',
            'spherical', or 'polar').
        Dt : 1D array or None (default: None)
            Integration time step at each snapshot. Same shape as
            coordinates[0] (time).
            If None, set to NaN values.
        '''

        self._ensure_consistency(data, coordinates, n_dimensions)

        sample_arr = list(data.values())[0]
        nt, *spatial_shape = sample_arr.shape
        self.ns_values = np.arange(nt)
        self.last_ns = self.ns_values[-1]

        self.definitions = DblDefinitions(n_dimensions, geometry)

        self._step_data = []
        time_cordinates = coordinates[0]
        spatial_coordinates = coordinates[1:]
        if Dt is None:
            Dt = np.full_like(time_cordinates, np.nan)
        for ns in self.ns_values:
            this_data = OrderedDict([
                (var_name, arr[ns]) for var_name, arr in data.items()])
            step_data = DblStepData(
                this_data,
                spatial_coordinates,
                time_cordinates[ns],
                Dt=Dt[ns],
                )
            self._step_data.append(step_data)

        self._update_vars()  # sets self.vars from self._step_data

    def _ensure_consistency(self, data, coordinates, n_dimensions):
        ''' Raise ValueError if n_dimensions, the number and shape of
        coordinates, and the shape of data are inconsistent. '''
        # ensure there are as many coordinates as dimensions and time
        if len(coordinates) != n_dimensions + 1:
            msg = (f'invalid number of coordinates: '
                   f'got {len(coordinates)}, but expected {n_dimensions + 1} '
                   f'(one for time, and one for each dimension)')
            raise ValueError(msg)
        # ensure that the shape of each array in data matches that of the
        # coordinates
        expected_shape = tuple(len(coord) for coord in coordinates)
        for var_name, arr in data.items():
            if arr.shape != expected_shape:
                msg = (f'invalid shape for {var_name}: '
                       f'expected {expected_shape}, got {arr.shape}')
                raise ValueError(msg)

    def create_step_data(self):
        step_data = DblStepData()
        return step_data

    def save_dbl(self, data_dir, **kwargs):
        ''' Save as dbl files.

        **kwargs are passed to DblWriter.write_to()
        '''
        writer = DblWriter()
        writer.write_to(self, data_dir, **kwargs)


class DblVarFile():
    ''' Represent a `dbl.out` file. '''
    def __init__(self, t, Dt, ns_values, var_names,
                 file_type='single_file',
                 endianness='little'):
        ''' Initialize a dbl var file

        Parameters
        ==========
        t : array of shape (n,)
            Time of each snapshot.
        Dt : array of shape (n,)
            Timestep of each snapshot.
        ns_values : array of shape (n,)
            Integration step of each snapshot.
        var_names : list of str
            Ordered list of the variable names.
        file_type : str (default: 'single_file')
            Whether variables are written to a single file ('single_file') or
            to multiple files ('multiple_files').
        endianness : str (default: 'little')
            Endianness of the dbl file ('little' or 'big').
        '''
        if not (len(t) == len(Dt) == len(ns_values)):
            raise ValueError('t, Dt, and ns_values must have the same shape')

        self.t = t
        self.Dt = Dt
        self.ns_values = ns_values
        self.var_names = var_names
        self.file_type = file_type
        self.endianness = endianness

    def to_string(self):
        ''' Convert to a string that can later be written to a file (usually
        `dbl.out`). '''
        var_names_str = ' '.join(self.var_names)
        string = ''
        for i, (t, Dt, ns) in enumerate(zip(self.t, self.Dt, self.ns_values)):
            string += (
                f'{i:d} {t:.6e} {Dt:.6e} {ns:d} '
                f'{self.file_type} {self.endianness} {var_names_str} \n'
                )
        return string

    def write_to(self, filename):
        ''' Write to a file (usually `dbl.out`). '''
        with open(filename, 'w') as f:
            f.write(self.to_string())


class DblWriter():
    ''' Rudimentary .dbl files writer '''

    def write_to(self, dataset, data_dir,
                 file_type='single_file', endianness='little'):
        ''' Write a PLUTO dataset to dbl files.

        Parameters
        ==========
        dataset : PlutoDataset
            The dataset to write
        data_dir : str
            The directory to which the dbl files will be written.
            If it does not exist, it will be created.
        file_type : str (default: single_file)
            Each snapshot can be saved either as a single file containing all
            variables (`single_file`), or as one file per variable
            (`multiple_files`).
        endianness : str (default: 'little')
            Endianness of the dbl file ('little' or 'big').

        This function will create the following files:
            - `{data_dir}/dbl.out`
            - `{data_dir}/grid.out`
            - `{data_dir}/data.{i}.dbl` if passed `single_file`,
              or `{data_dir}/{var_name}.{i}.dbl` if passed `multiple_files`,
              where `i` is the id of each snapshot, and `var_name` the name of
              each variable in each snapshot.
        '''

        if file_type not in ('single_file', 'multiple_files'):
            raise ValueError(f'unsupported file_type value: {file_type}')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        var_file = DblVarFile(
            dataset.t,
            dataset.Dt,
            dataset.ns_values,
            dataset.vars,
            file_type=file_type,
            endianness=endianness,
            )
        filename = f'{data_dir}/dbl.out'
        print('Writing var file: ', filename)
        var_file.write_to(filename)

        grid = PlutoGrid(
            dataset.definitions['DIMENSIONS'],
            dataset.definitions['GEOMETRY'],
            dataset.x1,
            dataset.x2,
            dataset.x3,
            )
        filename = f'{data_dir}/grid.out'
        print('Writing grid file:', filename)
        grid.write_to(filename)

        for ns in dataset.ns_values:
            data = self.get_step_data(dataset, ns)
            if file_type == 'single_file':
                filename = f'{data_dir}/data.{ns:04d}.dbl'
                data = np.array(data)
                self.write_array(data, filename, endianness)
            elif file_type == 'multiple_files':
                for nv, varname in enumerate(dataset.vars):
                    filename = f'{data_dir}/{varname}.{ns:04d}.dbl'
                    var_data = data[nv]
                    self.write_array(var_data, filename, endianness)

    def get_step_data(self, dataset, ns):
        ''' Get the data for a given step of the dataset.

        Parameters
        ==========
        dataset : PlutoDataset
            The dataset
        ns : int
            The index of the step from which to retrieve the data.
        '''
        step_dataset = dataset.get_step(ns)
        data = [step_dataset.get_var(var) for var in step_dataset.vars]
        return data

    def write_array(self, array, filename, endianness):
        ''' Write a numpy array to a binary file.

        Parameters
        ==========
        array : ndarray
            The array to write
        filename : str
            Name of the file where to write the data
        '''
        if endianness == 'little':
            endian = '<'
        elif endianness == 'big':
            endian = '>'
        else:
            raise ValueError(f'unknown endianness: {self.endianness}')
        # cast to little-endian float64, ie. double precision
        array = array.T
        array = array.astype(f'{endian}f8')
        print('Writing data file:', filename)
        array.tofile(filename)
