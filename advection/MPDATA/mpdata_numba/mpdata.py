
from fvms.mpdata.upwind import upwind
from fvms.mpdata.centred import centred
from fvms.mpdata.update_fluxdiv import update_fluxdiv
from fvms.mpdata.fluxdiv import fluxdiv
from fvms.mpdata.density_factor import density_factor
from fvms.mpdata.pseudoflux import pseudo_flux
from fvms.mpdata.local_minmax import local_minmax
from fvms.mpdata.nonoscoefficients import nonoscoefficients
from fvms.mpdata.limit_pseudovel import limit_pseudovel
from fvms.mpdata.mass_flux import mass_flux_accumulated, mass_flux_upwind


def mpdata_advection(config, fields, advect_density=False, mpdata_order=2, mpdata_gauge=True):

  psi, grg, grgr, grgd, vxh, vyh, vzh, fxh, fyh, fzh, txh, tyh, tzh, \
    div, cnh, cnl, psimin, psimax = (
        fields.tracer,
        fields.tmp1,
        fields.tmp2,
        fields.tmp3,
        fields.velx_face,
        fields.vely_face,
        fields.velz_face,
        fields.flux_face,
        fields.fluy_face,
        fields.fluz_face,
        fields.tmpx_face,
        fields.tmpy_face,
        fields.tmpz_face,
        fields.tmp4,
        fields.tmp5,
        fields.tmp6,
        fields.tmp7,
        fields.tmp8
    )

  nonos = config.nonos

  if mpdata_order < 2: nonos = False

  if nonos:
    local_minmax(config, psi, psimin, psimax, linit_minmax=True)

  upwind(config, psi, vxh, vyh, vzh, fxh, fyh, fzh)
  update_fluxdiv(config, psi, grg, fxh, fyh, fzh)
  density_factor(config, psi, grgr)

  if mpdata_order < 2:
    if advect_density:
      mass_flux_upwind(config, fxh, fyh, fzh, vxh, vyh, vzh)
    return 

  if nonos:
    local_minmax(config, psi, psimin, psimax)

  centred(config, psi, vxh, vyh, vzh, txh, tyh, tzh)
  fluxdiv(config, div, txh, tyh, tzh)

  pseudo_flux(config, psi, grg, div, vxh, vyh, vzh, txh, tyh, tzh)

  if nonos: 

    nonoscoefficients(config, psimin, psimax, txh, tyh, tzh, grg, psi, cnh, cnl)
 
    limit_pseudovel(config, cnh, cnl, txh, tyh, tzh, vxh, vyh, vzh)
 
    density_factor(config, psi, grgr, inverse=True)
    update_fluxdiv(config, psi, grg, vxh, vyh, vzh)

    if advect_density:
      mass_flux_accumulated(config, fxh, fyh, fzh, vxh, vyh, vzh, vxh, vyh, vzh)

  else:
     
    density_factor(config, psi, grgr, inverse=True)
    update_fluxdiv(config, psi, grg, txh, tyh, tzh)

    if advect_density:
      mass_flux_accumulated(config, fxh, fyh, fzh, txh, tyh, tzh, vxh, vyh, vzh)
  
  density_factor(config, psi, grgr)