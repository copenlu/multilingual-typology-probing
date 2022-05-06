import json
from uncertainties import ufloat, core


class ResultsEncoder(json.JSONEncoder):
    """
    Adds some serialization functionality to handle uncertain floats.
    """
    def default(self, obj):
        if isinstance(obj, core.AffineScalarFunc):
            return {
                "type": "ufloat",
                "nominal_value": obj.nominal_value,
                "std_dev": obj.std_dev
            }

        return json.JSONEncoder.default(self, obj)


class ResultsDecoder(json.JSONDecoder):
    """
    Adds some deserialization functionality to handle uncertain floats.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj

        type = obj["_type"]
        if type == "ufloat":
            return ufloat(obj["nominal_value"], obj["std_dev"])

        return obj
