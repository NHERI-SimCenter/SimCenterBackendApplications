def infowriter(info, meshdir):  # noqa: INP001, D100
    """
    Write the information dictionary to a tcl file.

    Parameters
    ----------
    info : dict
        Information dictionary.
    meshdir : str
        Directory where the tcl file will be written

    Returns
    -------
    None

    """
    # ============================================================================
    # Create a tcl file
    # ============================================================================
    f = open(f'{meshdir}/Modelinfo.tcl', 'w')  # noqa: SIM115, PTH123
    f.write(f'wipe\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Cores Information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set regcores       {info['regcores']}\n")
    f.write(f"set pmlcores       {info['pmlcores']}\n")
    f.write(f"set drmcores       {info['drmcores']}\n")
    f.write(f"set structurecores {info['structurecores']}\n")
    f.write(f"set AnalysisType   {info['AnalysisType']}\n")

    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Model Information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set StructureType     \"{info['StructureType']}\"\n")
    f.write(f"set NStory            {info['NStory']}\n")
    f.write(f"set NBay              {info['NBay']}\n")
    f.write(f"set NBayZ             {info['NBayZ']}\n")
    f.write(f"set StartNodeX       {info['StartNodeX']}\n")
    f.write(f"set StartNodeY       {info['StartNodeY']}\n")
    f.write(f"set StartNodeZ       {info['StartNodeZ']}\n")
    f.write(f"set LCol              {info['LCol']}\n")
    f.write(f"set LBeam             {info['LBeam']}\n")
    f.write(f"set LGird             {info['LGird']}\n")
    f.write(f"set SectionType       {info['SectionType']}\n")
    f.write(f"set HaveStructure     \"{info['HaveStructure']}\"\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Soil Information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set dx                {info['dx']}\n")
    f.write(f"set dy                {info['dy']}\n")
    f.write(f"set dz                {info['dz']}\n")
    f.write(f"set llx               {info['llx']}\n")
    f.write(f"set lly               {info['lly']}\n")
    f.write(f"set llz               {info['llz']}\n")
    f.write(f"set drmthicknessx     {info['drmthicknessx']}\n")
    f.write(f"set drmthicknessy     {info['drmthicknessy']}\n")
    f.write(f"set drmthicknessz     {info['drmthicknessz']}\n")
    f.write(f"set numdrmlayers      {info['numdrmlayers']}\n")
    f.write(f"set lx                {info['lx']}\n")
    f.write(f"set ly                {info['ly']}\n")
    f.write(f"set lz                {info['lz']}\n")
    f.write(f"set nx                {info['nx']}\n")
    f.write(f"set ny                {info['ny']}\n")
    f.write(f"set nz                {info['nz']}\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# PML information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set AbsorbingElements \"{info['AbsorbingElements']}\"\n")
    f.write(f"set numpmllayers      {info['numpmllayers']}\n")
    f.write(f"set pmlthicknessx     {info['pmlthicknessx']}\n")
    f.write(f"set pmlthicknessy     {info['pmlthicknessy']}\n")
    f.write(f"set pmlthicknessz     {info['pmlthicknessz']}\n")
    f.write(f"set pmltotalthickness {info['pmltotalthickness']}\n")
    f.write(f"set HaveAbsorbingElements \"{info['HaveAbsorbingElements']}\"\n")
    f.write(f"set Absorbing_rayleigh_alpha {info['Absorbing_rayleigh_alpha']}\n")
    f.write(f"set Absorbing_rayleigh_beta {info['Absorbing_rayleigh_beta']}\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# General information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set meshdir           \"{info['meshdir']}\"\n")
    f.write(f"set outputdir         \"{info['outputdir']}\"\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Embedding foundation\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set EmbeddingFoundation \"{info['EmbeddingFoundation']}\"\n")
    info2 = info['EmbeddedFoundation']
    f.write(
        f"set EmbeddedFoundation [dict create xmax {info2['xmax']} xmin {info2['xmin']} ymax {info2['ymax']} ymin {info2['ymin']} zmax {info2['zmax']} zmin {info2['zmin']}]\n"
    )
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Fondation information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set HaveFoundation \"{info['HaveFoundation']}\"\n")
    f.write('set foundationBlocks {}\n')
    for i, block in enumerate(info['foundationBlocks']):  # noqa: B007
        f.write(
            f"lappend foundationBlocks [dict create matTag {block['matTag']} xmax {block['xmax']} xmin {block['xmin']} ymax {block['ymax']} ymin {block['ymin']} zmax {block['zmax']} zmin {block['zmin']} Xmeshsize {block['Xmeshsize']} Ymeshsize {block['Ymeshsize']} Zmeshsize {block['Zmeshsize']}]\n"
        )
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Piles information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set HavePiles \"{info['HavePiles']}\"\n")
    f.write('set pilelist {}\n')
    for i, pile in enumerate(info['pilelist']):  # noqa: B007
        f.write(
            f"lappend pilelist [dict create xtop {pile['xtop']} ytop {pile['ytop']} ztop {pile['ztop']} xbottom {pile['xbottom']} ybottom {pile['ybottom']} zbottom {pile['zbottom']} numberofElements {pile['numberofElements']}]\n"
        )
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# cells and nodes information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set soilfoundation_num_cells {info['soilfoundation_num_cells']}\n")
    f.write(f"set soilfoundation_num_nodes {info['soilfoundation_num_points']}\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# DRM information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set DRMFile           \"{info['DRMFile']}\"\n")
    f.write(
        f"set DRM_Provider_Software \"{info['DRMinformation']['DRM_Provider_Software']}\"\n"
    )
    f.write(f"set DRM_factor \"{info['DRMinformation']['factor']}\"\n")
    f.write(f"set crd_scale \"{info['DRMinformation']['crd_scale']}\"\n")
    f.write(
        f"set distance_tolerance \"{info['DRMinformation']['distance_tolerance']}\"\n"
    )
    f.write(
        f"set do_coordinate_transformation \"{info['DRMinformation']['do_coordinate_transformation']}\"\n"
    )
    f.write(f"set T00 \"{info['DRMinformation']['T00']}\"\n")
    f.write(f"set T01 \"{info['DRMinformation']['T01']}\"\n")
    f.write(f"set T02 \"{info['DRMinformation']['T02']}\"\n")
    f.write(f"set T10 \"{info['DRMinformation']['T10']}\"\n")
    f.write(f"set T11 \"{info['DRMinformation']['T11']}\"\n")
    f.write(f"set T12 \"{info['DRMinformation']['T12']}\"\n")
    f.write(f"set T20 \"{info['DRMinformation']['T20']}\"\n")
    f.write(f"set T21 \"{info['DRMinformation']['T21']}\"\n")
    f.write(f"set T22 \"{info['DRMinformation']['T22']}\"\n")
    f.write(f"set originX \"{info['DRMinformation']['originX']}\"\n")
    f.write(f"set originY \"{info['DRMinformation']['originY']}\"\n")
    f.write(f"set originZ \"{info['DRMinformation']['originZ']}\"\n")
    f.write(
        f"set DRM_Location \"{info['DRMinformation']['DRM_Location'].lower()}\"\n"
    )
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# Analysis information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set Analysis_dt {info['AnalysisInfo']['dt']}\n")
    f.write(f"set Analysis_duration {info['AnalysisInfo']['t_final']}\n")
    f.write(f"set Analysis_record_dt {info['AnalysisInfo']['recording_dt']}\n")
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f'# material information\n')  # noqa: F541
    f.write(
        f'# ============================================================================\n'  # noqa: F541
    )
    f.write(f"set Vs {info['MaterialInfo']['Vs']}\n")
    f.write(f"set rho {info['MaterialInfo']['rho']}\n")
    f.write(f"set nu {info['MaterialInfo']['nu']}\n")
    f.close()

    return None  # noqa: RET501, PLR1711
