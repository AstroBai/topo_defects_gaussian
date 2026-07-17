import argparse
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from scipy.special import jvp

class DefectCF:

    def __init__(self, cosmo_params, ell_max, theta_arcmin, smooth_arcmin):
        self.cosmo_params = cosmo_params
        self.ell_max = ell_max
        self.theta_arcmin = theta_arcmin
        self.smooth_arcmin = smooth_arcmin
        self.ell = jnp.arange(ell_max + 1)
        self.kernels = self.make_bessel_derivative_kernels(self.ell, jnp.deg2rad(theta_arcmin / 60.0), max_order=6)

    
    def make_bessel_derivative_kernels(self, ell, theta_rad, max_order=6):
        x = np.outer(np.asarray(ell, dtype=np.float64), np.asarray(theta_rad, dtype=np.float64))
        kernels = {}
        for n in range(1, max_order + 1):
            kernels[n] = jnp.asarray(jvp(0, x, n=n), dtype=jnp.float64)
        return kernels


    def cl_psi_from_kappa(self, cl_kappa, ell):
        cl_psi = jnp.zeros_like(cl_kappa)
        valid = ell >= 2
        denom = (ell**2) * (ell + 1.0) ** 2
        cl_psi = jnp.where(valid, 4.0 * cl_kappa / denom, 0.0)
        return cl_psi


    def gaussian_filter_ell(self, ell, sigma_rad):
        return jnp.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2)


    def corr_theta_derivative_from_cl_kernel(self, cl_psi, ell, kernel, order):
        pref = (ell / (2.0 * jnp.pi)) * cl_psi * (ell ** order)
        return jnp.sum(pref[:, None] * kernel, axis=0)


    def compute_c0_moments(self, cl_psi, ell):
        pref = (ell / (2.0 * jnp.pi)) * cl_psi
        c4_0 = jnp.sum(pref * (3.0 / 8.0) * ell**4)
        c6_0 = jnp.sum(pref * (-5.0 / 16.0) * ell**6)
        return c4_0, c6_0


    def compute_h_u_and_derivative_analytic(self, theta, c1, c2, c3, c4, c5, c6, c4_0):
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

        s = jnp.sqrt(jnp.maximum(detk, 1e-300))
        invs = 1.0 / s
        s1p = detk1 / (2.0 * s)
        s2p = detk2 / (2.0 * s) - (detk1**2) / (4.0 * s**3)
        invs1 = -s1p * invs**2
        invs2 = -s2p * invs**2 + 2.0 * (s1p**2) * invs**3

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


    def g_u_from_cl_kappa(self, cl_kappa, ell, theta_arcmin, smooth_arcmin, kernels):
        theta = jnp.deg2rad(theta_arcmin / 60.0)
        sigma_rad = jnp.deg2rad(smooth_arcmin / 60.0)

        cl_psi = self.cl_psi_from_kappa(cl_kappa, ell)
        cl_psi = cl_psi * self.gaussian_filter_ell(ell, sigma_rad) ** 2

        c1 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[1], 1)
        c2 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[2], 2)
        c3 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[3], 3)
        c4 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[4], 4)
        c5 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[5], 5)
        c6 = self.corr_theta_derivative_from_cl_kernel(cl_psi, ell, kernels[6], 6)

        c4_0, c6_0 = self.compute_c0_moments(cl_psi, ell)
        _, dh_u = self.compute_h_u_and_derivative_analytic(theta, c1, c2, c3, c4, c5, c6, c4_0)

        d_u = jnp.abs(3.0 * c6_0 / (10.0 * jnp.pi * c4_0)) + 1/np.pi
        g_u = dh_u / (4.0 * jnp.pi**2 * d_u**2 * theta)
        return g_u


    def cl_kappa_from_params(self, pvec, ell, fixed):
        omega_c, sigma8, hubble, w0 = pvec
        cosmo = jc.Cosmology(
            Omega_c=omega_c,
            Omega_b=fixed["Omega_b"],
            h=hubble,
            n_s=fixed["n_s"],
            sigma8=sigma8,
            Omega_k=0.0,
            w0=w0,
            wa=0.0,
        )

        nz = jc.redshift.delta_nz(1.0)
        wl = jc.probes.WeakLensing([nz])
        cl = jc.angular_cl.angular_cl(cosmo, ell, [wl])[0]
        return cl
    
    def get_gu(self):
        cl_kappa = self.cl_kappa_from_params(
            pvec=[self.cosmo_params["Omega_c"], self.cosmo_params["sigma8"], self.cosmo_params["h"], self.cosmo_params["w"]],
            ell=self.ell,
            fixed={"Omega_b": 0.049, "n_s": 0.96},
        )
        g_u = self.g_u_from_cl_kappa(cl_kappa, self.ell, self.theta_arcmin, self.smooth_arcmin, self.kernels)
        return g_u
    
    def get_fisher_derivative(self, param_name):
        pvec = [
            self.cosmo_params["Omega_c"],
            self.cosmo_params["sigma8"],
            self.cosmo_params["h"],
            self.cosmo_params["w"],
        ]
        idx = {"Omega_c": 0, "sigma8": 1, "h": 2, "w": 3}[param_name]
        # autodiff with jax
        def g_u_func(pvec):
            pvec = jnp.asarray(pvec, dtype=jnp.float64)
            cl_kappa = self.cl_kappa_from_params(
                pvec=pvec,
                ell=self.ell,
                fixed={"Omega_b": 0.049, "n_s": 0.96},
            )
            g_u = self.g_u_from_cl_kappa(cl_kappa, self.ell, self.theta_arcmin, self.smooth_arcmin, self.kernels)
            return g_u
        jac = jax.jacfwd(g_u_func)(jnp.asarray(pvec, dtype=jnp.float64))
        return np.asarray(jac[:, idx])








