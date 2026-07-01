#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "June 28, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid

class RunFastSlipPy(FastSlipPy):
    """
    This can be customized for specific runs.
    """
    def run(self):
        super().run()

        self.grid.plot_grid()
        self.grid.plot_mesh()

if __name__ == "__main__":
    # Customise parameters here or leave all defaults

    params = ModelParameters(
        case_type = "groningen",
        alpha = 60.0,
        xsize = 320e3,
        ysize = 80e3,
        Nx = 641, Ny = 161,
        #Nx = 141, Ny = 31,
        Nt = 100,
        output_interval = 10,
        checkpoint_interval = 100,
        rho = 2670.0,
        rhof = 1150.0,
        rhog = 200.0,
        Vp = 0.0,
        cs = 3464,
        mu0 = 0.6,
        nu = 0.25,
        #E=0.55e10, #according to k_critical = sigam * (b-a) / d_c, E = 1e10
        V0 = 1e-6,
        a0 = 0.01,
        b0 = 0.015,
        L = 0.008,
        dt_init = 1.0,
        dt_max = 1e6,
        Vw = 1e90,
        Vi = 1e-30
    )

    yr = 365 * 24 * 3600.0
    params.loading.tload = 0.0 * yr
    params.loading.dPdt_pre = 0.0
    params.loading.dPdt_post = 0.0

    params.bc.left.set_fixed()
    params.bc.right.set_fixed()
    params.bc.bottom.set_fixed()
    params.bc.top.set_free()

    params.layers.set_groningen()

    model = RunFastSlipPy(params=params, output_dir="output")
    model.run()
