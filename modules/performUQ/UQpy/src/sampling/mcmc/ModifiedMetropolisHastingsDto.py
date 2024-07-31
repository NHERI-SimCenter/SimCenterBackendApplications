from typing import Literal  # noqa: INP001, D100

from pydantic import Field
from src.UQpyDTO import UQpyDTO


class ModifiedMetropolisHastingsDto(UQpyDTO):  # noqa: D101
    method: Literal['Modified Metropolis Hastings'] = 'Modified Metropolis Hastings'
    burn_length: int = Field(default=0, alias='burn-in', ge=0)
    jump: int = Field(default=1, ge=0)
    # dimension: int = Field(..., gt=0)  # noqa: ERA001
    # n_chains: int = Field(default=1, alias='numChains', ge=1)  # noqa: ERA001
    random_state: int = Field(..., alias='randomState')
    save_log_pdf: bool = False
    concatenate_chains = True
    proposal_is_symmetric = False

    def init_to_text(self):  # noqa: ANN201, D102
        from UQpy.sampling.mcmc.ModifiedMetropolisHastings import (
            ModifiedMetropolisHastings,
        )

        c = ModifiedMetropolisHastings

        class_name = c.__module__.split('.')[-1]
        import_statement = 'from ' + c.__module__ + ' import ' + class_name + '\n'

        stretch_parameters = self.dict()
        stretch_parameters.pop('method')
        stretch_parameters['log_pdf_target'] = 'marginals.log_pdf'
        stretch_parameters['seed'] = 'list(marginals.rvs(numRV,))'
        # stretch_parameters["seed"] = f"list(marginals.rvs({self.n_chains},))"  # noqa: ERA001
        str_parameters = ''
        for key in stretch_parameters:
            if stretch_parameters[key] is None:
                continue
            str_parameters += key + '=' + str(stretch_parameters[key]) + ', '

        prerequisite_str = import_statement
        prerequisite_str += 'sampling = ' + class_name + '(' + str_parameters + ')'
        sampling_str = 'sampling'

        return (prerequisite_str, sampling_str)
