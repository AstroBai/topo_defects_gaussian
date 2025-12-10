import pyccl as ccl
import numpy as np


def DefectDensity(cosmo, dndz, lmax, filter=None, filter_angle=None):
    """Compute the angular power spectrum of the projected density of
    topological defects.

    Args:
        cosmo (:obj:`ccl.Cosmology`): a Cosmology object.
        dndz (:obj:`tuple`): tuple (z, dN/dz) defining the redshift
            distribution of defects.
        lmax (:obj:`int`): maximum multipole to compute.
        filter (:obj:`str`, optional): type of filter to apply to the
            projected density field. Options are 'gaussian' or None.
            Default is None.
        filter_angle (:obj:`float`, optional): scale of the filter in
            radians. Only used if filter is not None.
    """ 
    ells = np.arange(lmax + 1)
    tracer = ccl.WeakLensingTracer(cosmo, dndz=dndz)
    cl_kappa = ccl.angular_cl(cosmo, tracer, tracer, ells)
    cl_psi = 4 / (ells[1:] * (ells[1:] + 1))**2 * cl_kappa[1:]
    cl_psi = np.insert(cl_psi, 0, 0.0)
    def legendre_derivative(ell, n):
        import math
        m = int(n/2)
        if ell <= m:
            return 0
        else:
            # math.factorial(ell+m)/math.factorial(ell-m)
            if m == 2:
                return (-1)**m * math.factorial(2*m)/math.factorial(m)**2 * (ell-1)*ell*(ell+1)*(ell+2)/4**m
            elif m == 3:
                return (-1)**m * math.factorial(2*m)/math.factorial(m)**2 * (ell-2)*(ell-1)*ell*(ell+1)*(ell+2)*(ell+3)/4**m
            
    lgd_factorials_6 = np.array([legendre_derivative(l, 6) for l in ells])
    lgd_factorials_4 = np.array([legendre_derivative(l, 4) for l in ells])
    
    if filter == 'gaussian' and filter_angle is not None:
        gauss_filter = gaussian_filter(ells, filter_angle)
        cl_psi *= gauss_filter

    m6 = np.sum((2*ells+1)/(4*np.pi) * cl_psi * lgd_factorials_6)
    m4 = np.sum((2*ells+1)/(4*np.pi) * cl_psi * lgd_factorials_4)

    density = np.abs(m6 / m4) * 3 / (10 * np.pi)
    return density

def gaussian_filter(ells, angle):
    """Compute a Gaussian filter in multipole space.

    Args:
        ells (:obj:`array`): array of multipoles.
        angle (:obj:`float`): scale of the filter in radians.
    """
    return np.exp(-ells * (ells + 1) * angle**2 / 2)
    

def DefectCF(cosmo, dndz, lmax, theta, filter=None, filter_angle=None):
    """Compute the defect two-point correlation function.

    Args:
        cosmo (:obj:`ccl.Cosmology`): a Cosmology object.
        dndz (:obj:`tuple`): tuple (z, dN/dz) defining the redshift
            distribution of defects.
        lmax (:obj:`int`): maximum multipole to compute.
        theta (:obj:`array`): angular separations in radians.
        filter (:obj:`str`, optional): type of filter to apply to the
            projected density field. Options are 'gaussian' or None.
            Default is None.
        filter_angle (:obj:`float`, optional): scale of the filter in
            radians. Only used if filter is not None.
    """
    


if __name__ == "__main__":
    # Example usage
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0, 3, 100)
    dndz = (z, z**2 * np.exp(-z))
    lmaxs = np.logspace(1, 5, 10).astype(int)
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    densities = []
    for lmax in tqdm(lmaxs):
        density = DefectDensity(cosmo, dndz, lmax)
        densities.append(density)
    densities = np.array(densities)
    plt.figure(figsize=(8,6))
    plt.loglog(lmaxs, densities, marker='o')
    plt.xlabel(r"$\ell_{\mathrm{max}}$")
    plt.ylabel("Defect Density")
    plt.title("Topological Defect Density vs Max Multipole")
    plt.grid()
    plt.savefig("outfiles/defect_density_vs_lmax.png")

    lmaxs = 2**np.linspace(5, 12, 8).astype(int)
    plt.figure(figsize=(8,6))
    for lmax in tqdm(lmaxs):
        theta_arcmin = np.logspace(-1, 2, 20)
        theta_rad = theta_arcmin * (np.pi / 180) / 60
        densities_filter = []
        for theta in theta_rad:
            density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
            densities_filter.append(density)
        densities_filter = np.array(densities_filter)
        plt.loglog(theta_arcmin, densities_filter, label=f"lmax={lmax}")
    plt.legend()
    plt.xlabel("Filter Scale [arcmin]")
    plt.ylabel("Defect Density")
    plt.title("Topological Defect Density vs Filter Scale")
    plt.grid()
    plt.savefig("outfiles/defect_density_vs_filter_scale.png")

    lmax = 256*3
    Oms = np.linspace(0.1, 0.5, 10)
    plt.figure(figsize=(8,6))
    for Om in Oms:
        cosmo = ccl.Cosmology(Omega_c=Om-0.05, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
        theta_arcmin = np.logspace(0, 3, 20)
        theta_rad = theta_arcmin * (np.pi / 180) / 60
        densities_om = []
        for theta in theta_rad:
            density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
            densities_om.append(density)
        densities_om = np.array(densities_om)
        plt.loglog(theta_arcmin, densities_om, label=f"Om={Om:.2f}")
    plt.legend()
    plt.xlabel("Filter Scale [arcmin]")
    plt.ylabel("Defect Density")
    plt.title("Topological Defect Density vs Filter Scale for Different Omega_m")
    plt.grid()
    plt.savefig("outfiles/defect_density_vs_filter_scale_omega_m.png")

    sigma8s = np.linspace(0.6, 1.0, 10)
    plt.figure(figsize=(8,6))
    for sigma8 in sigma8s:
        cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=sigma8, n_s=0.96)
        theta_arcmin = np.logspace(1.5, 3, 20)
        theta_rad = theta_arcmin * (np.pi / 180) / 60
        densities_s8 = []
        for theta in theta_rad:
            density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
            densities_s8.append(density)
        densities_s8 = np.array(densities_s8)
        plt.semilogx(theta_arcmin, densities_s8, label=f"sigma8={sigma8:.2f}")
    plt.legend()
    plt.xlabel("Filter Scale [arcmin]")
    plt.ylabel("Defect Density")
    plt.title("Topological Defect Density vs Filter Scale for Different Sigma8")
    plt.grid()
    plt.savefig("outfiles/defect_density_vs_filter_scale_sigma8.png")