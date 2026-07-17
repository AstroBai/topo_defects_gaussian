import pyccl as ccl
import numpy as np
import healpy as hp
from scipy.special import j0, jvp


def _cl_psi_from_kappa(cl_kappa):
    ells = np.arange(len(cl_kappa))
    cl_psi = np.zeros_like(cl_kappa)
    valid = ells >= 2
    denom = (ells[valid] ** 2) * ((ells[valid] + 1) ** 2)
    cl_psi[valid] = 4.0 * cl_kappa[valid] / denom
    return cl_psi


def _corr_theta_derivative_from_cl_bessel(cl_psi, theta, order):
    n = int(order)
    if n < 0:
        raise ValueError("order must be non-negative")

    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi
    x = np.outer(ells, theta)
    kernel = jvp(0, x, n=n)
    return np.sum((pref * (ells ** n))[:, None] * kernel, axis=0)


def _compute_c0_moments(cl_psi):
    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi

    c4_0 = np.sum(pref * (3.0 / 8.0) * ells**4)
    c6_0 = np.sum(pref * (-5.0 / 16.0) * ells**6)
    return c4_0, c6_0


def _compute_c0_from_moment(cl_field):
    ells = np.arange(len(cl_field), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_field
    return np.sum(pref)


def _compute_h_u_and_derivative_analytic(theta, c1, c2, c3, c4, c5, c6, c4_0):
    r = theta

    l0 = c4_0 / 3.0
    l = c3 / r - 2.0 * c2 / (r**2) + 2.0 * c1 / (r**3)
    m = c4
    n = 3.0 * c2 / (r**2) - 3.0 * c1 / (r**3)
    p = -c5
    q = -c4 / r + 3.0 * c3 / (r**2) - 6.0 * c2 / (r**3) + 6.0 * c1 / (r**4)
    rr = -3.0 * c3 / (r**2) + 9.0 * c2 / (r**3) - 9.0 * c1 / (r**4)

    scomb = m + n - 2.0 * l
    a = l0**2 - (scomb**2) / 16.0
    b = l0**2 - l**2
    detk = a * b

    l1 = c4 / r - 3.0 * c3 / (r**2) + 6.0 * c2 / (r**3) - 6.0 * c1 / (r**4)
    l2 = c5 / r - 4.0 * c4 / (r**2) + 12.0 * c3 / (r**3) - 24.0 * c2 / (r**4) + 24.0 * c1 / (r**5)
    m1 = c5
    m2 = c6
    n1 = 3.0 * c3 / (r**2) - 9.0 * c2 / (r**3) + 9.0 * c1 / (r**4)
    n2 = 3.0 * c4 / (r**2) - 15.0 * c3 / (r**3) + 36.0 * c2 / (r**4) - 36.0 * c1 / (r**5)
    p1 = -c6
    q1 = -c5 / r + 4.0 * c4 / (r**2) - 12.0 * c3 / (r**3) + 24.0 * c2 / (r**4) - 24.0 * c1 / (r**5)
    q2 = -c6 / r + 5.0 * c5 / (r**2) - 20.0 * c4 / (r**3) + 60.0 * c3 / (r**4) - 120.0 * c2 / (r**5) + 120.0 * c1 / (r**6)
    rr1 = -3.0 * c4 / (r**2) + 15.0 * c3 / (r**3) - 36.0 * c2 / (r**4) + 36.0 * c1 / (r**5)
    rr2 = -3.0 * c5 / (r**2) + 21.0 * c4 / (r**3) - 81.0 * c3 / (r**4) + 180.0 * c2 / (r**5) - 180.0 * c1 / (r**6)

    s1 = m1 + n1 - 2.0 * l1
    s2 = m2 + n2 - 2.0 * l2
    a1 = -0.125 * scomb * s1
    a2 = -0.125 * (s1**2 + scomb * s2)
    b1 = -2.0 * l * l1
    b2 = -2.0 * (l1**2 + l * l2)
    detk1 = a1 * b + a * b1
    detk2 = a2 * b + 2.0 * a1 * b1 + a * b2

    sqrt_detk = np.sqrt(detk)

    s = sqrt_detk
    invs = 1.0 / s
    s1 = detk1 / (2.0 * s)
    s2 = detk2 / (2.0 * s) - (detk1**2) / (4.0 * s**3)
    invs1 = -s1 * invs**2
    invs2 = -s2 * invs**2 + 2.0 * (s1**2) * invs**3

    t = q - rr
    t1 = q1 - rr1
    t2 = q2 - rr2

    g = t**2
    g1 = 2.0 * t * t1
    g2 = 2.0 * (t1**2 + t * t2)

    aa1 = 0.25 * (g * invs + r * g1 * invs + r * g * invs1)
    aa2 = 0.25 * (2.0 * g1 * invs + 2.0 * g * invs1 + r * g2 * invs + 2.0 * r * g1 * invs1 + r * g * invs2)

    mm = q * (p + rr - 2.0 * q)
    mm1 = q1 * (p + rr - 2.0 * q) + q * (p1 + rr1 - 2.0 * q1)
    bb = 0.25 * mm * invs
    bb1 = 0.25 * (mm1 * invs + mm * invs1)

    h_u = aa1 + bb
    dh_u = aa2 + bb1

    return h_u, dh_u


def DefectDensity(cosmo, dndz, lmax, filter=None, filter_angle=None,
                  include_pixel_window=False, nside=None):
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
        include_pixel_window (:obj:`bool`, optional): if True, include
            HEALPix pixel window suppression in harmonic space.
            Default is False.
        nside (:obj:`int`, optional): HEALPix nside for pixel window.
            Required if include_pixel_window is True.
    """ 
    ells = np.arange(lmax + 1)
    tracer = ccl.WeakLensingTracer(cosmo, dndz=dndz)
    cl_kappa = ccl.angular_cl(cosmo, tracer, tracer, ells)

    if include_pixel_window:
        if nside is None:
            raise ValueError("nside must be provided when include_pixel_window=True")
        # Map-domain pixelization corresponds to multiplying Cl by W_ell^2.
        pixwin = hp.pixwin(nside=int(nside), lmax=lmax, pol=False)
        cl_kappa = cl_kappa * pixwin**2

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
        cl_psi *= gauss_filter**2

    m6 = np.sum((2*ells+1)/(4*np.pi) * cl_psi * lgd_factorials_6)
    m4 = np.sum((2*ells+1)/(4*np.pi) * cl_psi * lgd_factorials_4)

    density = np.abs(m6 / m4) * 3 / (10 * np.pi) + 1/np.pi 
    return density

def gaussian_filter(ells, angle):
    """Compute a Gaussian filter in multipole space.

    Args:
        ells (:obj:`array`): array of multipoles.
        angle (:obj:`float`): scale of the filter in radians.
    """
    return np.exp(-ells * (ells + 1) * angle**2 / 2)
    

def DefectCF(cosmo, dndz, lmax, theta, filter=None, filter_angle=None, N_side=None):
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
        N_side (:obj:`int`, optional): HEALPix nside for pixel window.
            Required if include_pixel_window is True.

    Returns:
        :obj:`array` or :obj:`float`: defect two-point correlation
        function :math:`g_u(\theta)`.
    """
    theta_in = np.asarray(theta, dtype=np.float64)
    theta_arr = np.atleast_1d(theta_in)
    if np.any(theta_arr <= 0):
        raise ValueError("theta must be strictly positive in radians")

    ells = np.arange(lmax + 1)
    tracer = ccl.WeakLensingTracer(cosmo, dndz=dndz)
    cl_kappa = ccl.angular_cl(cosmo, tracer, tracer, ells)

    cl_psi = _cl_psi_from_kappa(cl_kappa)

    if N_side is not None:
        pixel_window = hp.pixwin(nside=N_side, lmax=lmax)
        cl_psi *= pixel_window**2

    if filter == 'gaussian' and filter_angle is not None:
        gauss_filter = gaussian_filter(ells, filter_angle)
        cl_psi *= gauss_filter**2

    # Normalize the working field so C(0)=1 as done in the full-sky script.
    c0_raw = _compute_c0_from_moment(cl_psi)
    if not np.isfinite(c0_raw) or np.abs(c0_raw) < np.finfo(np.float64).tiny:
        raise ValueError("Computed C(0) is zero or non-finite; cannot normalize")
    cl_psi /= c0_raw

    c1 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 1)
    c2 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 2)
    c3 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 3)
    c4 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 4)
    c5 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 5)
    c6 = _corr_theta_derivative_from_cl_bessel(cl_psi, theta_arr, 6)

    c4_0, c6_0 = _compute_c0_moments(cl_psi)
    _, dh_u = _compute_h_u_and_derivative_analytic(theta_arr, c1, c2, c3, c4, c5, c6, c4_0)

    d_u = np.abs(3.0 * c6_0 / (10.0 * np.pi * c4_0))
    if not np.isfinite(d_u) or np.abs(d_u) < np.finfo(np.float64).tiny:
        raise ValueError("Computed defect density normalization is zero or non-finite")

    g_u = dh_u / (4.0 * np.pi**2 * d_u**2 * theta_arr)

    if np.ndim(theta_in) == 0:
        return float(g_u[0])
    return g_u


if __name__ == "__main__":
    # Example usage
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0, 3, 100)
    dndz = (z, z**2 * np.exp(-z))
    lmaxs = np.logspace(1, 5, 10).astype(int)
    # from tqdm import tqdm
    # import matplotlib.pyplot as plt
    # densities = []
    # for lmax in tqdm(lmaxs):
    #     density = DefectDensity(cosmo, dndz, lmax)
    #     densities.append(density)
    # densities = np.array(densities)
    # plt.figure(figsize=(8,6))
    # plt.loglog(lmaxs, densities, marker='o')
    # plt.xlabel(r"$\ell_{\mathrm{max}}$")
    # plt.ylabel("Defect Density")
    # plt.title("Topological Defect Density vs Max Multipole")
    # plt.grid()
    # plt.savefig("outfiles/defect_density_vs_lmax.png")

    # lmaxs = 2**np.linspace(5, 12, 8).astype(int)
    # plt.figure(figsize=(8,6))
    # for lmax in tqdm(lmaxs):
    #     theta_arcmin = np.logspace(-1, 2, 20)
    #     theta_rad = theta_arcmin * (np.pi / 180) / 60
    #     densities_filter = []
    #     for theta in theta_rad:
    #         density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
    #         densities_filter.append(density)
    #     densities_filter = np.array(densities_filter)
    #     plt.loglog(theta_arcmin, densities_filter, label=f"lmax={lmax}")
    # plt.legend()
    # plt.xlabel("Filter Scale [arcmin]")
    # plt.ylabel("Defect Density")
    # plt.title("Topological Defect Density vs Filter Scale")
    # plt.grid()
    # plt.savefig("outfiles/defect_density_vs_filter_scale.png")

    # lmax = 256*3
    # Oms = np.linspace(0.1, 0.5, 10)
    # plt.figure(figsize=(8,6))
    # for Om in Oms:
    #     cosmo = ccl.Cosmology(Omega_c=Om-0.05, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    #     theta_arcmin = np.logspace(0, 3, 20)
    #     theta_rad = theta_arcmin * (np.pi / 180) / 60
    #     densities_om = []
    #     for theta in theta_rad:
    #         density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
    #         densities_om.append(density)
    #     densities_om = np.array(densities_om)
    #     plt.loglog(theta_arcmin, densities_om, label=f"Om={Om:.2f}")
    # plt.legend()
    # plt.xlabel("Filter Scale [arcmin]")
    # plt.ylabel("Defect Density")
    # plt.title("Topological Defect Density vs Filter Scale for Different Omega_m")
    # plt.grid()
    # plt.savefig("outfiles/defect_density_vs_filter_scale_omega_m.png")

    # sigma8s = np.linspace(0.6, 1.0, 10)
    # plt.figure(figsize=(8,6))
    # for sigma8 in sigma8s:
    #     cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=sigma8, n_s=0.96)
    #     theta_arcmin = np.logspace(1.5, 3, 20)
    #     theta_rad = theta_arcmin * (np.pi / 180) / 60
    #     densities_s8 = []
    #     for theta in theta_rad:
    #         density = DefectDensity(cosmo, dndz, lmax, filter='gaussian', filter_angle=theta)
    #         densities_s8.append(density)
    #     densities_s8 = np.array(densities_s8)
    #     plt.semilogx(theta_arcmin, densities_s8, label=f"sigma8={sigma8:.2f}")
    # plt.legend()
    # plt.xlabel("Filter Scale [arcmin]")
    # plt.ylabel("Defect Density")
    # plt.title("Topological Defect Density vs Filter Scale for Different Sigma8")
    # plt.grid()
    # plt.savefig("outfiles/defect_density_vs_filter_scale_sigma8.png")

    # 3rd order derivative w.r.t Omega_m at fiducial, w/o noise, at lmax=300 only for Om, use finite difference
    def finite_diff_3rd_order(func, params, ell_max, h=1e-2):
        params = np.array(params)
        f_minus_2h = func(params - 2*h, ell_max)
        f_minus_h = func(params - h, ell_max)
        f_plus_h = func(params + h, ell_max)
        f_plus_2h = func(params + 2*h, ell_max)
        return (f_minus_2h - 2*f_minus_h + 2*f_plus_h - f_plus_2h) / (2 * h**3)
    
    params_fid = np.array([0.3, 0.8])
    ell_max_test = 512
    third_derivative_om = finite_diff_3rd_order(lambda p, ell: DefectDensity(ccl.Cosmology(Omega_c=p[0]-0.05, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96), dndz, ell), params_fid, ell_max_test)
    print(f"3rd order derivative w.r.t Omega_m at params={params_fid}, ell_max={ell_max_test}: {third_derivative_om:.5e}")