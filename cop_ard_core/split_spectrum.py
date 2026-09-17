"""
Split-spectrum phase decomposition: joint unwrapping of low- and
high-band sub-bands and separation of dispersive and non-dispersive phase.
"""

import numpy as np
import xarray as xr

from cop_ard_core.detrend import fit_2d_poly, to_polygrid2d
from cop_ard_core.logger import timeit
from cop_ard_core.unwrap import unwrap_phase


def joint_unwrap(
    mosaic_l_coherence: xr.DataArray,
    mosaic_h_coherence: xr.DataArray,
    mosaic_l_phase: xr.DataArray,
    mosaic_h_phase: xr.DataArray,
    cohe_th: float = 0.3,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Jointly unwrap low-band and high-band interferometric phases using a
    common phase reference.

    This function computes a common mean phase from the low-frequency and
    high-frequency sub-band interferograms and unwraps it first. The
    unwrapped mean phase is then used as a reference to remove the common
    phase contribution from each sub-band. The residual phases are
    independently unwrapped and the common phase is subsequently added back.

    This approach can improve phase-unwrapping robustness by reducing the
    phase gradients that must be handled in each sub-band independently.

    Parameters
    ----------
    mosaic_l_coherence
        Coherence of the low-frequency sub-band interferogram.
    mosaic_h_coherence
        Coherence of the high-frequency sub-band interferogram.
    mosaic_l_phase
        Wrapped interferometric phase of the low-frequency sub-band [rad].
    mosaic_h_phase
        Wrapped interferometric phase of the high-frequency sub-band [rad].
    cohe_th
        Coherence threshold passed to the phase-unwrapping algorithm.
        Pixels with coherence below this threshold are typically excluded
        from the unwrapping process. Default is 0.3.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        A tuple containing:

        - **ph_l_unw**: Unwrapped low-frequency sub-band phase.
        - **ph_h_unw**: Unwrapped high-frequency sub-band phase.
        - **cohe_mask**: Geometric mean coherence used as the unwrapping
          quality mask.

    Notes
    -----
    The coherence mask is computed as

    .. math::

        \\gamma = \\sqrt{\\gamma_h \\gamma_l}

    where :math:`\\gamma_h` and :math:`\\gamma_l` are the coherences of the
    high-frequency and low-frequency sub-bands.

    The common phase reference is estimated from the angle of the complex sum

    .. math::

        \\phi_{mean} =
        \\arg\\left(
        e^{j\\phi_h} + e^{j\\phi_l}
        \\right)

    After unwrapping :math:`\\phi_{mean}`, the residual phases

    .. math::

        \\phi_l - \\phi_{mean}

    and

    .. math::

        \\phi_h - \\phi_{mean}

    are independently unwrapped. The unwrapped mean phase is then added back
    to recover the final low-band and high-band unwrapped phases.
    """
    # extract the average coherence between the two sub-bands
    cohe_mask = xr.ufuncs.sqrt(mosaic_h_coherence * mosaic_l_coherence)

    # compute the average phase between the two sub-bands
    ph_mean = xr.ufuncs.angle(np.exp(1j * mosaic_h_phase) + np.exp(1j * mosaic_l_phase))
    ph_mean_unw = unwrap_phase(
        ph_mean,
        cohe_mask,
        threshold=cohe_th,
    )

    # unwrap sub-band residues and add back the average phase
    ph_l_unw = (
        unwrap_phase(
            mosaic_l_phase - ph_mean_unw,
            cohe_mask,
            threshold=cohe_th,
        )
        + ph_mean_unw
    )

    ph_h_unw = (
        unwrap_phase(
            mosaic_h_phase - ph_mean_unw,
            cohe_mask,
            threshold=cohe_th,
        )
        + ph_mean_unw
    )

    return ph_l_unw, ph_h_unw, cohe_mask


def joint_unwrap_delta_mean(
    mosaic_l_coherence: xr.DataArray,
    mosaic_h_coherence: xr.DataArray,
    mosaic_l_phase: xr.DataArray,
    mosaic_h_phase: xr.DataArray,
    cohe_th: float = 0.3,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Jointly unwrap the mean and differential phases derived from low-band
    and high-band interferograms.

    This function computes the geometric mean coherence from the two
    split-spectrum sub-bands and uses it to drive phase unwrapping.
    Two wrapped phase products are generated:

    * the mean phase, obtained from the complex average of the two
      sub-band interferograms;
    * the differential phase, obtained from the phase difference
      between the high-band and low-band interferograms.

    Both phase products are independently unwrapped using the same
    coherence mask and threshold.

    Parameters
    ----------
    mosaic_l_coherence
        Coherence of the low-frequency sub-band interferogram.
    mosaic_h_coherence
        Coherence of the high-frequency sub-band interferogram.
    mosaic_l_phase
        Wrapped interferometric phase of the low-frequency sub-band [rad].
    mosaic_h_phase
        Wrapped interferometric phase of the high-frequency sub-band [rad].
    cohe_th
        Coherence threshold passed to the phase-unwrapping algorithm.
        Pixels with coherence below this threshold are typically excluded
        from the unwrapping process. Default is 0.3.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        A tuple containing:

        - **phase_delta_unw**: Unwrapped differential phase
          (:math:`\\phi_h - \\phi_l`).
        - **phase_mean_unw**: Unwrapped mean phase derived from the
          complex average of the two sub-bands.
        - **cohe_mask**: Geometric mean coherence used as the
          unwrapping quality mask.

    Notes
    -----
    The coherence mask is computed as

    .. math::

        \\gamma = \\sqrt{\\gamma_h \\gamma_l}

    where :math:`\\gamma_h` and :math:`\\gamma_l` are the coherences of
    the high-frequency and low-frequency sub-bands.

    The mean phase is estimated from the angle of the complex sum

    .. math::

        \\phi_{mean} =
        \\arg\\left(
        e^{j\\phi_h} + e^{j\\phi_l}
        \\right)

    while the differential phase is

    .. math::

        \\phi_{delta} = \\phi_h - \\phi_l

    Both quantities are subsequently phase-unwrapped using the same
    coherence mask.
    """
    # extract the average coherence between the two sub-bands
    cohe_mask = xr.ufuncs.sqrt(mosaic_h_coherence * mosaic_l_coherence)

    # compute the average phase between the two sub-bands and unwrap
    phase_mean = xr.ufuncs.angle(np.exp(1j * mosaic_h_phase) + np.exp(1j * mosaic_l_phase))
    phase_mean_unw = unwrap_phase(
        phase_mean,
        cohe_mask,
        threshold=cohe_th,
    )

    # compute the delta phase and unwrap
    phase_delta = mosaic_h_phase - mosaic_l_phase
    phase_delta_unw = unwrap_phase(
        phase_delta,
        cohe_mask,
        threshold=cohe_th,
    )

    return phase_delta_unw, phase_mean_unw, cohe_mask


def split_spectrum_core(
    phase_unw_l: xr.DataArray,
    phase_unw_h: xr.DataArray,
    f0: float,
    f_l_tot: float,
    f_h_tot: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Estimate dispersive and non-dispersive phase components using the
    split-spectrum method from low-band and high-band unwrapped phases.

    This function decomposes the interferometric phase into its dispersive
    (ionospheric) and non-dispersive components using the unwrapped phases
    obtained from two spectral sub-bands. The method exploits the different
    frequency dependence of dispersive and non-dispersive phase terms to
    separate the two contributions.

    Parameters
    ----------
    phase_unw_l
        Unwrapped interferometric phase of the low-frequency sub-band [rad].
    phase_unw_h
        Unwrapped interferometric phase of the high-frequency sub-band [rad].
    f0
        Central frequency of the full SAR signal [Hz].
    f_l_tot
        Central frequency of the low-frequency sub-band [Hz].
    f_h_tot
        Central frequency of the high-frequency sub-band [Hz].

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing:

        - **phase_iono**: Estimated dispersive (ionospheric) phase component.
        - **phase_nd**: Estimated non-dispersive phase component.

    Notes
    -----
    The dispersive component is computed as

    .. math::

        \\phi_{iono} =
        \\left(
        \\phi_l \\frac{f_h}{f_0}
        -
        \\phi_h \\frac{f_l}{f_0}
        \\right)
        \\frac{f_l f_h}{f_h^2 - f_l^2}

    while the non-dispersive component is computed as

    .. math::

        \\phi_{nd} =
        \\left(
        \\phi_h \\frac{f_h}{f_0}
        -
        \\phi_l \\frac{f_l}{f_0}
        \\right)
        \\frac{f_0}{2 (f_h - f_l)}

    where :math:`\\phi_l` and :math:`\\phi_h` are the unwrapped phases of the
    low-frequency and high-frequency sub-bands, respectively.

    References
    ----------
    The formulation follows the standard split-spectrum approach for
    ionospheric phase estimation in SAR interferometry.
    """
    # dispersive component
    phase_iono = (
        (phase_unw_l * (f_h_tot / f0) - phase_unw_h * (f_l_tot / f0)) * (f_l_tot * f_h_tot) / (f_h_tot**2 - f_l_tot**2)
    )

    # non-dispersive component
    phase_nd = (phase_unw_h * (f_h_tot / f0) - phase_unw_l * (f_l_tot / f0)) * f0 / (2 * (f_h_tot - f_l_tot))

    return phase_iono, phase_nd


def split_spectrum_core_delta_mean(
    delta_phase: xr.DataArray,
    mean_phase: xr.DataArray,
    f0: float,
    f_l_tot: float,
    f_h_tot: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Estimate dispersive and non-dispersive phase components using the
    split-spectrum method.

    This function decomposes the interferometric phase into its dispersive
    (ionospheric) and non-dispersive components from the phase difference
    and phase average of two sub-band interferograms. The formulation follows
    the standard split-spectrum approach, where the ionospheric phase scales
    inversely with frequency while the non-dispersive phase scales linearly
    with frequency.

    Parameters
    ----------
    delta_phase
        Difference between the high-band and low-band interferometric phases.
        Typically defined as :math:`\\phi_h - \\phi_l`.
    mean_phase
        Mean phase computed from the high-band and low-band interferometric
        phases. Typically defined as
        :math:`(\\phi_h + \\phi_l) / 2`.
    f0
        Central frequency of the full SAR signal [Hz].
    f_l_tot
        Central frequency of the low-frequency sub-band [Hz].
    f_h_tot
        Central frequency of the high-frequency sub-band [Hz].

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing:

        - **phase_iono**: Estimated dispersive (ionospheric) phase component.
        - **phase_nd**: Estimated non-dispersive phase component.

    Notes
    -----
    The dispersive component is computed as

    .. math::

        \\phi_{iono} =
        -\\frac{1}{2}
        \\frac{f_l f_h}{f_0 (f_h - f_l)}
        \\Delta\\phi
        +
        \\frac{f_l f_h}{f_0 (f_h + f_l)}
        \\bar{\\phi}

    while the non-dispersive component is computed as

    .. math::

        \\phi_{nd} =
        \\frac{1}{4}
        \\frac{f_h + f_l}{f_h - f_l}
        \\Delta\\phi
        +
        \\frac{1}{2}
        \\bar{\\phi}

    where :math:`\\Delta\\phi` is the phase difference and
    :math:`\\bar{\\phi}` is the phase mean of the two sub-bands.
    """
    # dispersive component
    alpha = -0.5 * (f_l_tot * f_h_tot) / (f0 * (f_h_tot - f_l_tot))
    beta = (f_l_tot * f_h_tot) / (f0 * (f_h_tot + f_l_tot))
    phase_iono = alpha * delta_phase + beta * mean_phase

    # non-dispersive component
    alpha = 0.25 * (f_l_tot + f_h_tot) / (f_h_tot - f_l_tot)
    beta = 0.5
    phase_nd = alpha * delta_phase + beta * mean_phase

    return phase_iono, phase_nd


def scale_azimuth_time(
    azimuth_time_raw: xr.DataArray,
    azimuth_time: xr.DataArray,
    H_orb: xr.DataArray | float,
    H_target: xr.DataArray | float,
) -> xr.DataArray:
    """
    Rescale azimuth acquisition times from orbital altitude to a target
    altitude assuming a linear geometric relationship.

    This function computes the azimuth time offset between the acquisition
    geometry at orbital altitude and the corresponding ground-referenced
    azimuth time. The offset is then scaled according to the ratio between
    a target height and the orbital height, allowing the azimuth timing to
    be projected to a different elevation layer, such as an ionospheric
    shell. The resulting offset is finally added back to the reference
    azimuth time to obtain the projected acquisition time.

    Parameters
    ----------
    azimuth_time_raw
        Original azimuth acquisition time associated with the orbital
        geometry.
    azimuth_time
        Reference azimuth time, typically corresponding to the
        ground-projected acquisition geometry.
    H_orb
        Orbital height above the reference surface. Can be either a scalar
        value or an xarray object matching the input dimensions.
    H_target
        Target height to which the azimuth time offset is projected. Can be
        either a scalar value or an xarray object matching the input
        dimensions.

    Returns
    -------
    xr.DataArray
        Azimuth acquisition time projected to the target height.

    Notes
    -----
    The azimuth time difference between orbital and ground geometries is
    first computed as

    .. math::

        \\Delta t = t_{raw} - t_{ground}

    The time offset is then scaled according to

    .. math::

        \\Delta t_{target}
        =
        \\Delta t
        \\frac{H_{target}}{H_{orb}}

    Finally, the projected absolute acquisition time is obtained as

    .. math::

        t_{target}
        =
        t_{ground}
        +
        \\Delta t_{target}

    This approximation assumes a linear relationship between azimuth-time
    displacement and altitude.
    """
    # compute difference between azimuth time on the ground and at orbit level
    az_time = (azimuth_time_raw - azimuth_time).astype(float)

    # rescale with the elevation of ionosphere
    az_time = (az_time * (H_target / H_orb)).astype("timedelta64[ns]")

    # compute the absolute time
    az_time = az_time + azimuth_time

    return az_time


@timeit
def fit_ionosphere(
    phase_iono: xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    H_iono: float,
    H_orb: float,
    degree: int = 1,
    norm: int = 1,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Fit a two-dimensional polynomial model to the ionospheric phase screen.

    This function projects the ionospheric phase observations onto the
    ionospheric shell, constructs a two-dimensional coordinate system in
    azimuth and slant-range, and performs a polynomial least-squares fit.
    The resulting polynomial coefficients are returned as a coefficient grid
    together with the scaling and centering parameters required to evaluate
    the model on the original coordinates.

    Parameters
    ----------
    phase_iono
        Estimated ionospheric phase screen to be fitted. The input is expected
        to contain the dimensions ``azimuth_time`` and
        ``slant_range_time``.
    azimuth_time_raw
        Original azimuth-time coordinates prior to ionospheric height
        projection.
    H_iono
        Effective ionospheric shell height [m].
    H_orb
        Satellite orbital height above the reference surface [m].
    degree
        Degree of the two-dimensional polynomial fit. Default is 1.
    norm
        Polynomial normalization strategy passed to ``fit_2d_poly``.
        Default is 1.
    weights
        Optional weighting factors to be used during the fitting process.
        When provided, it must have the same spatial geometry as
        ``phase_iono``.

    Returns
    -------
    xr.DataArray
        Two-dimensional grid of polynomial coefficients with dimensions
        ``("i", "j")``. The array attributes contain the scaling and
        centering parameters required to reconstruct the fitted model:

        - **deg**: polynomial degree.
        - **offset**: coordinate offset applied during fitting.
        - **scale_fact**: coordinate scaling factor.
        - **off_y**: azimuth-coordinate offset applied after projection.

    Notes
    -----
    The fitting procedure consists of the following steps:

    1. Project azimuth coordinates from the satellite acquisition geometry
       to the ionospheric shell height.
    2. Flatten the phase screen and corresponding coordinates into a
       one-dimensional point cloud.
    3. Optionally flatten the weighting array.
    4. Fit a two-dimensional polynomial surface of the requested degree.
    5. Convert the fitted coefficient vector into a two-dimensional
       polynomial coefficient matrix using ``to_polygrid2d``.

    The returned coefficient grid can subsequently be evaluated using the
    stored scaling and offset parameters to reconstruct the fitted
    ionospheric phase model.
    """
    y_ = scale_azimuth_time(
        azimuth_time_raw,
        phase_iono.azimuth_time,
        H_orb,
        H_iono,
    ).values

    # Compute the fitting coordinates in slant-range and projected azimuth.
    x_ = np.tile(
        phase_iono.slant_range_time.values,
        phase_iono.azimuth_time.size,
    ).ravel()

    off_y = y_[0, 0]
    y_ = (y_ - off_y).astype(float).ravel()

    # Flatten the ionospheric phase screen into a point cloud.
    data = phase_iono.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values

    # Flatten optional weights using the same ordering as the data.
    w_stack = (
        weights.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values if weights is not None else None
    )

    # Compute polynomial-fit coefficients.
    coeffs, x_sc, x_0 = fit_2d_poly(
        x_,
        y_,
        data,
        degree=degree,
        norm=norm,
        weights=w_stack,
    )

    attrs = {
        "deg": degree,
        "offset": x_0,
        "scale_fact": x_sc,
        "off_y": str(off_y),
    }

    # Store the coefficients as a 2-D polynomial coefficient grid together
    # with the metadata required for coordinate rescaling and centering.
    poly_corr = xr.DataArray(
        data=to_polygrid2d(coeffs.x, degree),
        dims=["i", "j"],
        name="poly_corr",
        coords={key: np.arange(degree + 1) for key in ["i", "j"]},
        attrs=attrs,
    )

    return poly_corr


def reconstruct_iono_surface(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    poly_corr: xr.DataArray,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
) -> xr.DataArray:
    """
    Reconstruct an ionospheric correction surface from a fitted 2-D polynomial
    model.

    This function evaluates a previously estimated polynomial correction
    surface on a target azimuth/slant-range grid. The azimuth coordinates are
    first projected to the ionospheric layer using the orbital and ionospheric
    heights, then normalized using the scaling and centering parameters stored
    in the polynomial model. Finally, the polynomial is evaluated to obtain
    the reconstructed ionospheric correction surface.

    Parameters
    ----------
    azimuth_time
        Target azimuth-time coordinates on which the correction surface is
        reconstructed.
    slant_range_time
        Target slant-range-time coordinates on which the correction surface is
        reconstructed.
    azimuth_time_raw
        Original azimuth-time coordinates before projection to the
        ionospheric layer.
    poly_corr
        Polynomial coefficients describing the correction surface.
        The DataArray is expected to contain the following attributes:

        - ``off_y``: azimuth coordinate offset used during fitting.
        - ``offset``: coordinate centering vector.
        - ``scale_fact``: coordinate scaling factors.

    H_iono
        Effective ionospheric shell height used for azimuth-time projection.
    H_orb
        Satellite orbital height used for azimuth-time projection.

    Returns
    -------
    xr.DataArray
        Reconstructed ionospheric correction surface evaluated on the
        ``(azimuth_time, slant_range_time)`` grid.

    Notes
    -----
    The reconstruction proceeds through the following steps:

    1. Project the azimuth-time coordinates to the ionospheric layer.
    2. Build the coordinate pairs corresponding to every
       azimuth/slant-range sample.
    3. Apply the centering and scaling parameters used during polynomial
       estimation.
    4. Evaluate the 2-D polynomial surface using
       :func:`numpy.polynomial.polynomial.polyval2d`.
    5. Reshape the result back to the original grid and return it as an
       :class:`xarray.DataArray`.

    The reconstructed surface is evaluated as

    .. math::

        z = P(y, x)

    where :math:`P` is the fitted two-dimensional polynomial stored in
    ``poly_corr`` and :math:`(y, x)` are the normalized azimuth and
    slant-range coordinates.
    """
    y_ = scale_azimuth_time(
        azimuth_time_raw,
        azimuth_time,
        H_orb,
        H_iono,
    ).values

    # compute minimization axis
    x_ = np.tile(slant_range_time.values, azimuth_time.size).ravel()
    off_y = np.datetime64(poly_corr.attrs["off_y"], "ns")
    y_ = (y_ - off_y).astype(float).ravel()

    x_full = np.vstack([y_, x_]).T

    x_full = (x_full - poly_corr.attrs["offset"]) / poly_corr.attrs["scale_fact"]

    # reconstruct the surface in correspondence of the coordinates x_full
    surf = np.polynomial.polynomial.polyval2d(
        x_full[:, 0],
        x_full[:, 1],
        poly_corr.values,
    )

    surf = surf.reshape(
        azimuth_time.size,
        slant_range_time.size,
    )

    surf = xr.DataArray(
        surf,
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "azimuth_time": azimuth_time.values,
            "slant_range_time": slant_range_time.values,
        },
    )

    return surf
