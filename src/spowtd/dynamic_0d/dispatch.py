# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Dispatch script for dynamic 0d solvers"""

import collections
import json

import yaml

import spowtd.dynamic_0d.dynamic_solve as dynamic_mod
import spowtd.elapsed_time as elapsed_time_mod
import spowtd.functions.recharge as recharge_mod

from spowtd.functions.peat_properties import PeatProperties


DEFAULT_SOLVER = 'fixed-step-storage'


def dispatch(cli_args):
    """Dispatch script for dynamic 0d solvers"""
    peat_properties = PeatProperties.from_file(cli_args.peat_properties)
    recharge = recharge_mod.create_recharge_function(
        json.load(cli_args.net_precipitation)
    )
    curvature = cli_args.curvature
    solver = dynamic_mod.instantiate_solver(
        solver_name=(cli_args.solver or DEFAULT_SOLVER),
        numerical_parameters=json_or_empty_dict(cli_args.numerical_parameters),
        peat_properties=peat_properties,
        recharge=recharge,
        curvature=curvature,
        solver_trajectory_file=cli_args.record_solver_trajectory,
    )
    trajectory = dynamic_mod.solve_0d_surface(
        solver=solver,
        time_grid=list(
            elapsed_time_mod.ElapsedDaysIterator(**yaml.safe_load(cli_args.time_grid))
        ),
        initial_conditions=InitialConditions(
            water_level=cli_args.initial_water_level, surface=cli_args.initial_surface
        ),
    )
    dynamic_mod.postprocess_trajectory(
        peat_properties.transmissivity,
        peat_properties.specific_yield,
        curvature,
        recharge,
        trajectory,
    )
    json.dump(trajectory, cli_args.output, indent=2)


InitialConditions = collections.namedtuple(
    'InitialConditions', ['water_level', 'surface']
)


def json_or_empty_dict(infile):
    """Load JSON from file if it is not None"""
    return json.load(infile) if infile else {}
