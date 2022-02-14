#!/bin/env python

#
# Unit information which may be missing from lightcones
#
UNIT_CONV_NO_UNITS = {"U_"+base_unit : 0 for base_unit in "MLtIT"}
UNIT_CONV_PHOTON_FLUX_PER_UNIT_SURFACE = dict(UNIT_CONV_NO_UNITS, U_L=-2.0, U_t=-1.0)
UNIT_CONV_ENERGY_FLUX_PER_UNIT_SURFACE = dict(UNIT_CONV_NO_UNITS, U_M=1.0, U_t=-3.0)
missing_units = {
    "XrayErositaLowIntrinsicPhotons"   : UNIT_CONV_PHOTON_FLUX_PER_UNIT_SURFACE,
    "XrayErositaHighIntrinsicPhotons"  : UNIT_CONV_PHOTON_FLUX_PER_UNIT_SURFACE,
    "XrayROSATIntrinsicPhotons"        : UNIT_CONV_PHOTON_FLUX_PER_UNIT_SURFACE,
    "XrayErositaLowIntrinsicEnergies"  : UNIT_CONV_ENERGY_FLUX_PER_UNIT_SURFACE,
    "XrayErositaHighIntrinsicEnergies" : UNIT_CONV_ENERGY_FLUX_PER_UNIT_SURFACE,
    "XrayROSATIntrinsicEnergies"       : UNIT_CONV_ENERGY_FLUX_PER_UNIT_SURFACE,
}

# Name of attributes in InternalCodeUnits and Units groups
unit_cgs_name = {
    "U_I" : "Unit current in cgs (U_I)",
    "U_L" : "Unit length in cgs (U_L)",
    "U_M" : "Unit mass in cgs (U_M)",
    "U_T" : "Unit temperature in cgs (U_T)",
    "U_t" : "Unit time in cgs (U_t)",
}
cgs_unit = {
    "U_I" : "A",
    "U_L" : "cm",
    "U_M" : "g",
    "U_T" : "K",
    "U_t" : "s",
}

def read_cgs_units(infile):
    """
    Read CGS unit definitions from the Units group in a snapshot or lightcone
    """
    units_cgs = {}
    for name in unit_cgs_name:
        units_cgs[name] = infile["Units"].attrs[unit_cgs_name[name]][0]
    return units_cgs
    
def compute_cgs_factor(exponents, units_cgs):
    """
    Compute conversion to CGS using the Units group and the dimensions of a dataset
    """
    cgs_factor = 1.0
    for base_unit in units_cgs:
        exponent = exponents[base_unit]
        cgs_factor *= (units_cgs[base_unit]**exponent)
    return cgs_factor

def cgs_expression(exponents, units_cgs):

    if all([e==0.0 for e in exponents.values()]):
        return "[ - ]"

    expression = ""
    for base_unit, power in exponents.items():
        if power == 0:
            pass
        elif power == 1.0:
            expression += base_unit + " "
        elif power % 1.0 == 0.0:
            expression += ("%s^%d " % (base_unit, power))
        else:
            expression += ("%s^%7.4f " % (base_unit, power))

    expression += "[ "
    for base_unit, power in exponents.items():
        if power == 0:
            pass
        elif power == 1.0:
            expression += cgs_unit[base_unit]
        elif power % 1.0 == 0.0:
            expression += ("%s^%d " % (cgs_unit[base_unit], power))
        else:
            expression += ("%s^%7.4f " % (cgs_unit[base_unit], power))
    expression += "]"
    
    return expression
    
def correct_units(dset, name, units_cgs):
    """
    Return corrected unit attributes for the specified dataset as a dict,
    which will be empty if there's nothing to do
    """

    if dset.attrs["a-scale exponent"] != 0 or dset.attrs["h-scale exponent"]:
        raise NotImplementedError("Only implemented for quantities with no a/h factors")

    corrections = {}
    if name in missing_units:

        # Find correct exponents
        for base_unit in units_cgs:
            expected_exponent = missing_units[name][base_unit]
            found_exponent = dset.attrs[base_unit+" exponent"][0]
            if found_exponent != expected_exponent:
                corrections[base_unit+" exponent"] = [expected_exponent,]
        
        # If any exponents changed, need to update CGS factor and expression
        if len(corrections) > 0:

            # Find new CGS conversion
            cgs_conversion = compute_cgs_factor(missing_units[name], units_cgs)
            corrections["Conversion factor to CGS (not including cosmological corrections)"] = [cgs_conversion,]
            corrections["Expression for physical CGS units"] = cgs_expression(missing_units[name], units_cgs)

    return corrections
