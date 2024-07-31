# This file is used to define the class of Building  # noqa: INP001, D100
# Developed by GUAN, XINGQUAN @ UCLA in June 2018
# Updated in Sept. 2018


# #########################################################################
#                     Define a class including steel material property    #
# #########################################################################


class SteelMaterial:
    """This class is used to define the steel material.
    It includes the following physical quantities:
    (1) Yield stress (Fy)
    (2) Ultimate stress (Fu)
    (3) Young's modulus (E)
    (4) Ry value
    """  # noqa: D205, D400, D404, D415

    def __init__(  # noqa: ANN204
        self,
        yield_stress=50,  # noqa: ANN001
        ultimate_stress=65,  # noqa: ANN001
        elastic_modulus=29000,  # noqa: ANN001
        Ry_value=1.1,  # noqa: ANN001, N803
    ):
        """:param yield_stress: Fy of steel material, default value is 50 ksi
        :param elastic_modulus: E of steel material, default value is 29000 ksi
        """  # noqa: D205
        self.Fy = yield_stress
        self.Fu = ultimate_stress
        self.E = elastic_modulus
        self.Ry = Ry_value
