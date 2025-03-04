

def get_model_version_str_for_sd1_sd2(v2, v_parameterization):
    # only for reference
    version_str = "sd"
    if v2:
        version_str += "_v2"
    else:
        version_str += "_v1"
    if v_parameterization:
        version_str += "_v"
    return version_str
