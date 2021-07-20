####################################################################
# LICENSING INFORMATION
####################################################################
"""
	LICENSE INFORMATION:
	
	Copyright (c) 2020-2030, The Regents of the University of California (Regents).

	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, this 
		list of conditions and the following disclaimer.
	2. Redistributions in binary form must reproduce the above copyright notice,
		this list of conditions and the following disclaimer in the documentation
		and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.

	REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
	
"""
####################################################################
# AUTHOR INFORMATION
####################################################################
# 2020 - 2021: Ajay B Harish (ajaybh@berkeley.edu)

####################################################################
# Import all necessary modules
####################################################################
# Standard python modules
import os

# Other custom modules
from hydroUtils import hydroUtils

####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Solve():
	"""
	This class includes the methods related to
	solver for openfoam7.

	Methods
	--------
		fvSchemetext: Get all the text for the fvSchemes
	"""

	#############################################################
	def solverheader(self,fileobjec):
		'''
		Creates the text for the header

		Variable
		-----------
			header: Header for the solver-files
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"system";\n\tobject\t"""+fileobjec+""";\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

		# Return the header for U file
		return header

	#############################################################
	def fvSchemetext(self,data):
		'''
		Creates the necessary text for fvSchemes for openfoam7

		Arguments
		-----------
			data: all the JSON data
		'''

		# Get the header text for the U-file
		fvSchemetext = self.solverheader("fvSchemes")

		# Add all other items
		# ddt
		fvSchemetext = fvSchemetext + 'ddtSchemes\n{\n\tdefault\tEuler;\n}\n\n'

		# grad 
		fvSchemetext = fvSchemetext + 'gradSchemes\n{\n\tdefault\tGauss linear;\n}\n'

        # div 
		fvSchemetext = fvSchemetext + '\ndivSchemes\n{\n\t'
		fvSchemetext = fvSchemetext + 'div(rhoPhi,U)\tGauss limitedLinearV 1;\n\t'
		fvSchemetext = fvSchemetext + 'div(U)\tGauss linear;\n\t'
		fvSchemetext = fvSchemetext + 'div((rhoPhi|interpolate(porosity)),U)\tGauss limitedLinearV 1;\n\t'
		fvSchemetext = fvSchemetext + 'div(rhoPhiPor,UPor)\tGauss limitedLinearV 1;\n\t'
		fvSchemetext = fvSchemetext + 'div(rhoPhi,UPor)\tGauss limitedLinearV 1;\n\t'
		fvSchemetext = fvSchemetext + 'div(rhoPhiPor,U)\tGauss limitedLinearV 1;\n\t'
		fvSchemetext = fvSchemetext + 'div(phi,alpha)\tGauss vanLeer;\n\t'
		fvSchemetext = fvSchemetext + 'div(phirb,alpha)\tGauss interfaceCompression;\n\t'
		fvSchemetext = fvSchemetext + 'div((muEff*dev(T(grad(U)))))\tGauss linear;\n\t'
		fvSchemetext = fvSchemetext + 'div(phi,k)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div(phi,epsilon)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi|interpolate(porosity)),k)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi*interpolate(rho)),k)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi|interpolate(porosity)),epsilon)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div(phi,omega)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi|interpolate(porosity)),omega)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi*interpolate(rho)),omega)\tGauss upwind;\n\t'
		fvSchemetext = fvSchemetext + 'div((phi*interpolate(rho)),epsilon)\tGauss upwind;\n'
		fvSchemetext = fvSchemetext + '}\n\n'

		# Laplacian
		fvSchemetext = fvSchemetext + 'laplacianSchemes\n{\n\tdefault\tGauss linear corrected;\n}\n\n'

		# interpolation
		fvSchemetext = fvSchemetext + 'interpolationSchemes\n{\n\tdefault\tlinear;\n}\n\n'

		# snGrad
		fvSchemetext = fvSchemetext + 'snGradSchemes\n{\n\tdefault\tcorrected;\n}\n\n'

		# flux
		fvSchemetext = fvSchemetext + 'fluxRequired\n{\n\t'
		fvSchemetext = fvSchemetext + 'default\tno;\n\t'
		fvSchemetext = fvSchemetext + 'p_rgh;\n\t'
		fvSchemetext = fvSchemetext + 'pcorr;\n\t'
		fvSchemetext = fvSchemetext + 'alpha.water;\n'
		fvSchemetext = fvSchemetext + '}\n'

		return fvSchemetext

	