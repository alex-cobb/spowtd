# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Peat properties"""

import json

from dataclasses import dataclass

import yaml

from spowtd.functions._transmissivity import Transmissivity
from spowtd.functions._specific_yield import SpecificYield
from spowtd.functions._peat_growth import PeatGrowth
from spowtd.functions.transmissivity import create_transmissivity_function
from spowtd.functions.peat_growth import create_peat_growth_function
from spowtd.functions.specific_yield import create_specific_yield_function


@dataclass
class PeatProperties:
    """Peat properties

    Transmissivity, specific yield and peat growth functions

    """

    transmissivity: Transmissivity
    specific_yield: SpecificYield
    peat_growth: PeatGrowth

    @classmethod
    def from_file(cls, json_or_yaml_file):
        """Instantiate peat properties from a JSON or YAML file"""
        return cls.from_parameters(**json_or_yaml(json_or_yaml_file))

    @classmethod
    def from_parameters(cls, transmissivity, specific_yield, peat_growth=None):
        """Instantiate peat properties from parameters

        Parameters are mappings of parameters for transmissivity,
        specific yield, and (optionally) peat growth

        """
        return cls(
            transmissivity=create_transmissivity_function(**transmissivity),
            specific_yield=create_specific_yield_function(**specific_yield),
            peat_growth=peat_growth and create_peat_growth_function(**peat_growth),
        )


def json_or_yaml(infile):
    """Load JSON from file, or failing that, YAML"""
    try:
        return json.load(infile)
    except json.decoder.JSONDecodeError:
        infile.seek(0)
        return yaml.safe_load(infile)
