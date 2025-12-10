import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb
from healpy.sphtfunc import smoothing

def GenMap(cosmo_params, nside=1024, lmax=1024*3, dx=200.0, z_source=(0.0, 1.0), random_seed=24, Plot=False, Savepath=None, random_type='gaussian', smooth_angle=None, smooth_type='pre'):
    """
    Generate shear and convergence maps using GLASS and CAMB cosmology.

    Parameters:
    - cosmo_params: dict with keys 'h', 'Oc', 'Ob', 'w' for cosmological parameters.
    - nside: HEALPix nside parameter.
    - lmax: Maximum angular mode number.
    - dx: Shell spacing in comoving distance (Mpc).
    - z_source: Tuple with min and max redshift for source distribution.

    Returns:
    - kappa_bar: Convergence map.
    - gamm1_bar: Shear component 1 map.
    - gamm2_bar: Shear component 2 map.
    - phi_bar: Shear orientation map.
    """
    # import time
    # start_time = time.time()
    # Extract cosmological parameters
    h = cosmo_params.get('h', 0.7)
    Oc = cosmo_params.get('Oc', 0.25)
    Ob = cosmo_params.get('Ob', 0.05)
    w = cosmo_params.get('w', -1.0)

    Om = Oc + Ob
    if 'sigma8' in cosmo_params:
        sigma8 = cosmo_params.get('sigma8', 0.8)
    elif 'S8' in cosmo_params:
        S8 = cosmo_params.get('S8', 0.8)
        sigma8 = S8 / (Om / 0.3)**0.5

    # Create a random number generator
    rng = np.random.default_rng(seed=random_seed)

    # Set up CAMB parameters
    pars = camb.CAMBparams(DoLensing=False)
    pars.set_cosmology(H0=h * 100, ombh2=Ob * h**2, omch2=Oc * h**2)
    pars.set_dark_energy(w=w)
    pars.InitPower.set_params(ns=0.965, As=2.1e-9)
    pars.set_matter_power(kmax=10.0, nonlinear=False)
    result = camb.get_results(pars)
    sigma8_0 = result.get_sigma8_0()
    As_corrected = pars.InitPower.As * (sigma8 / sigma8_0)**2
    pars.InitPower.set_params(As=As_corrected)
    results = camb.get_background(pars)

    # Get the cosmology from CAMB
    cosmo = Cosmology(results)

    # Shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=dx)

    # Linear radial window functions
    shells = glass.linear_windows(zb)

    # Compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)

    # Apply discretisation to the full set of spectra
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)


    # Set up lognormal fields for simulation
    if random_type == 'lognormal':
        fields = glass.lognormal_fields(shells)
    elif random_type == 'gaussian':
        fields = glass.gaussian_fields(shells)
    else:
        raise ValueError("random_type must be either 'lognormal' or 'gaussian'")

    # Compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)
    if smooth_angle is not None and smooth_type == 'pre':
        for i in range(len(gls)):
            if gls[i].shape[0] > 1:
                gls[i] = gls[i] * np.exp(-np.arange(gls[i].shape[0]) * (np.arange(gls[i].shape[0]) + 1) * smooth_angle**2 / 2)

    # Generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)
    # This will compute the convergence field iteratively
    convergence = glass.MultiPlaneConvergence(cosmo)

    # Localised redshift distribution
    z = np.array([1])
    dndz = np.ones_like(z)  # np.exp(-((z - 0.5) ** 2) / (0.1) ** 2)

    # Distribute dN/dz over the radial window functions
    ngal = glass.partition(z, dndz, shells)

    # The integrated convergence and shear field over the redshift distribution
    kappa_bar = np.zeros(12 * nside**2)
    gamm1_bar = np.zeros(12 * nside**2)
    gamm2_bar = np.zeros(12 * nside**2)
    kappa_bar_unsmoothed = np.zeros(12 * nside**2)

    # Main loop to simulate the matter fields iteratively
    
    for i, delta_i in enumerate(matter):
        #print(f"Processing shell {i+1}/{len(shells)}")
        # Add lensing plane from the window function of this shell
        convergence.add_window(delta_i, shells[i])

        # Get convergence field
        kappa_i = convergence.kappa
        kappa_i_unsmoothed = kappa_i.copy()
        if smooth_angle is not None and smooth_type == 'post':
            kappa_i = smoothing(kappa_i, sigma=smooth_angle, lmax=lmax)
        gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i, lmax=lmax)
        # Add to mean fields using the galaxy number density as weight
        kappa_bar += ngal[i] * kappa_i
        gamm1_bar += ngal[i] * gamm1_i
        gamm2_bar += ngal[i] * gamm2_i

        kappa_bar_unsmoothed += ngal[i] * kappa_i_unsmoothed



    # Normalize mean fields by the total galaxy number density
    kappa_bar /= ngal.sum()
    gamm1_bar /= ngal.sum()
    gamm2_bar /= ngal.sum()
    kappa_bar_unsmoothed /= ngal.sum()
    # gamm1_bar, gamm2_bar = glass.shear_from_convergence(kappa_bar, lmax=lmax)
    phi_bar = 0.5 * np.arctan2(gamm2_bar, gamm1_bar)    
    # end_time = time.time()
    # print(f"Map generation completed in {end_time - start_time:.2f} seconds.")

    if Plot:
        # Visualize a square patch of the shear and convergence field
        field_size_arcmin = 400
        field_size_rad = np.deg2rad(field_size_arcmin / 60.0)  # transform to radians

        # Choose the center of the patch (can be adjusted)
        center_theta = np.pi/2  # zenith
        center_phi = 0.0

        # Create a denser version
        fig, ax = plt.subplots(figsize=(12, 12))

        # Higher resolution grid
        n_grid_hires = 256
        x_range_hires = np.linspace(-field_size_rad/2, field_size_rad/2, n_grid_hires)
        y_range_hires = np.linspace(-field_size_rad/2, field_size_rad/2, n_grid_hires)
        X_hires, Y_hires = np.meshgrid(x_range_hires, y_range_hires)

        theta_grid_hires = center_theta + Y_hires
        phi_grid_hires = center_phi + X_hires / np.sin(center_theta)
        theta_grid_hires = np.clip(theta_grid_hires, 0, np.pi)
        phi_grid_hires = phi_grid_hires % (2*np.pi)

        pixels_hires = hp.ang2pix(nside, theta_grid_hires, phi_grid_hires)
        kappa_patch_hires = kappa_bar[pixels_hires]
        gamm1_patch_hires = gamm1_bar[pixels_hires]
        gamm2_patch_hires = gamm2_bar[pixels_hires]
        phi_patch_hires = 0.5 * np.arctan2(gamm2_patch_hires, gamm1_patch_hires)
        shear_amplitude_hires = np.sqrt(gamm1_patch_hires**2 + gamm2_patch_hires**2)

        # Background
        im = ax.imshow(kappa_patch_hires, extent=[-field_size_arcmin/2, field_size_arcmin/2, 
                                                 -field_size_arcmin/2, field_size_arcmin/2],
                       cmap='RdYlBu_r', origin='lower')

        # Denser vectors
        vector_step_dense = 2
        x_sample_dense = X_hires[::vector_step_dense, ::vector_step_dense] * 180/np.pi * 60
        y_sample_dense = Y_hires[::vector_step_dense, ::vector_step_dense] * 180/np.pi * 60
        phi_sample_dense = phi_patch_hires[::vector_step_dense, ::vector_step_dense]
        shear_amp_dense = 1 
        u_sample_dense = np.cos(phi_sample_dense)
        v_sample_dense = np.sin(phi_sample_dense)
        # Vector length scaling
        scale_factor_dense = shear_amp_dense / np.max(shear_amp_dense) * 6
        u_sample_dense *= scale_factor_dense
        v_sample_dense *= scale_factor_dense    
        # Plot vectors
        ax.quiver(x_sample_dense, y_sample_dense, u_sample_dense, v_sample_dense,
                  angles='xy', scale_units='xy',
                  color='white', width=0.001,
                  headwidth=0, headlength=0, headaxislength=0)
        ax.set_xlabel('RA offset [arcmin]', fontsize=14)
        ax.set_ylabel('Dec offset [arcmin]', fontsize=14)
        ax.set_title(f'High-Resolution {field_size_arcmin}x{field_size_arcmin} arcmin² Shear Field', fontsize=16)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Convergence κ', fontsize=14)
        ax.set_aspect('equal')
        plt.tight_layout()
        if Savepath is not None:
            plt.savefig(Savepath, dpi=300, bbox_inches='tight')
        else:
            plt.savefig("outfiles/shear_field_patch_hires.png", dpi=300, bbox_inches='tight')

        # plot full-sky kappa only map        
        fig2 = plt.figure(figsize=(10, 8))
        hp.mollview(kappa_bar, title='Full-Sky Convergence Map κ', cmap='RdYlBu_r',
                    min=np.min(kappa_bar), max=np.max(kappa_bar))
        plt.savefig("outfiles/fullsky_convergence_map.png", dpi=300, bbox_inches='tight')

    return kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar



if __name__ == "__main__":
    # example usage

    cosmo_params = {'h': 0.7,
                    'Oc': 0.25,
                    'Ob': 0.05,
                    'S8': 0.8,
                    'w': -1.0}

    kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=512, lmax=512*3, Plot=True, Savepath="outfiles/shear_field_patch_hires.png", random_seed=12, smooth_angle=np.radians(90/60), smooth_type='post')

    # save data
    np.savez("outfiles/ShearConvergenceMapData.npz",
             kappa_bar_unsmoothed=kappa_bar_unsmoothed,
            kappa_bar=kappa_bar,
            gamm1_bar=gamm1_bar,
            gamm2_bar=gamm2_bar,
            phi_bar=phi_bar)
    # how to load:
    # data = np.load("outfiles/ShearConvergenceMapData.npz")
    # kappa_bar = data['kappa_bar']
    # gamm1_bar = data['gamm1_bar']
    # gamm2_bar = data['gamm2_bar']
    # phi_bar = data['phi_bar']

