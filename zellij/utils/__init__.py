# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from .converters import (
    DoNothing,
    IntMinMax,
    IntBinning,
    FloatMinMax,
    FloatBinning,
    CatToFloat,
    CatToInt,
    ArrayDefaultC,
)
from .neighborhoods import (
    IntInterval,
    FloatInterval,
    CatRandom,
    ArrayDefaultN,
    PermutationRandom,
)

from .distances import Euclidean, Manhattan, Mixed
