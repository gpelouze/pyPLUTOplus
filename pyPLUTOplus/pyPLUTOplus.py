#!/usr/bin/env python3

import configparser
import datetime
import os
import re

import numpy as np
import pyPLUTO as pp


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
                xL = np.array([0])
                xR = np.array([1])
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
    ''' Represent PLUTO `definitions.h` file.
    '''
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
                 last_ns=None, load_data=True):
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
        last_ns : int or None (default: None)
            Step number of the last data file to include in the movie.
            If None, this is determined by pyPLUTO.nlast_info.
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

        self.last_ns = last_ns
        if self.last_ns is None:
            nlast_info = pp.nlast_info(
                w_dir=self.data_dir,
                datatype=self.datatype)
            self.last_ns = nlast_info['nlast']
        self.ns_values = np.arange(0, self.last_ns+1)

        self.units = PlutoUnits(self.ini_dir)

        self._pload_datasets = []
        if load_data:
            self.load_data()

    def load_data(self):
        ''' Load datafiles for all steps '''
        for ns in self.ns_values:
            self._pload_datasets.append(self.load_step_data(ns))
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
        ''' Update variable names

        The list of var names is the union of the var names from all step files
        '''
        self.vars = set()
        for pds in self._pload_datasets:
            for v in pds.vars:
                self.vars.add(v)
        self.vars = list(self.vars)

    def get_step(self, i):
        ''' Get pyPLUTO pload object for step i '''
        return self._pload_datasets[i]

    def get_var(self, varname):
        ''' Get array of a variable at all time steps '''
        return np.array([self.get_step(i).get_var(varname)
                         for i in self.ns_values])

    @property
    def Dt(self):
        return np.array([pds.Dt for pds in self._pload_datasets])

    @property
    def t(self):
        return np.array([pds.SimTime for pds in self._pload_datasets])

    @property
    def t_cgs(self):
        return self.t * self.units.time

    @property
    def x1(self):
        return self.get_step(0).get_var('x1')

    @property
    def x2(self):
        return self.get_step(0).get_var('x2')

    @property
    def x3(self):
        return self.get_step(0).get_var('x3')

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
