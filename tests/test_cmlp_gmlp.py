"""Unit tests for cop_ard_core.cmlp and cop_ard_core.gmlp."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.cmlp import (
    calculate_stable_dates,
    get_current_reference,
    update_drift,
    update_recursive_reference,
)
from cop_ard_core.gmlp import get_radar_spacing, warp_cmlp


class TestCalculateStableDates:
    """Tests for calculate_stable_dates."""

    def test_calculate_stable_dates_basic(self):
        """Should select dates at stable_reference_step intervals."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-13",
                "2020-01-25",
                "2020-02-06",
                "2020-02-18",
                "2020-03-01",
            ],
            dtype="datetime64[ns]",
        )
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=2)
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1

    def test_calculate_stable_dates_step_one(self):
        """Step of 1 should select every date after start."""
        dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1)
        assert len(result) >= 1

    def test_calculate_stable_dates_with_starting_date(self):
        """Starting date should affect selection."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-13",
                "2020-01-25",
                "2020-02-06",
                "2020-02-18",
            ],
            dtype="datetime64[ns]",
        )
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1, starting_date="2020-01-13")
        assert isinstance(result, np.ndarray)

    def test_calculate_stable_dates_with_stable_reference(self):
        """Should extend from existing stable reference."""
        stable_ref = xr.DataArray(
            [np.datetime64("2020-01-01", "ns"), np.datetime64("2020-01-25", "ns")],
            dims=["acquisition_date"],
        )
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-13",
                "2020-01-25",
                "2020-02-06",
                "2020-02-18",
            ],
            dtype="datetime64[ns]",
        )
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1, stable_reference=stable_ref)
        assert isinstance(result, np.ndarray)

    def test_calculate_stable_dates_empty(self):
        """Empty dates array should return empty result."""
        dates = np.array([], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        with pytest.raises(ValueError):
            calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1)


class TestGetCurrentReference:
    """Unit tests for get_current_reference."""

    @pytest.fixture
    def cstack_3_dates(self):
        """3-date cstack with string acquisition_date coordinates."""
        return xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )

    @pytest.fixture
    def cstack_5_dates(self):
        """5-date cstack with regularly spaced dates."""
        return xr.DataArray(
            np.arange(60, dtype=np.float64).reshape(5, 3, 4),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]},
        )

    @pytest.fixture
    def last_reference(self):
        """A last_reference DataArray with a date before cstack's start.

        Uses scalar np.datetime64 for acquisition_date coordinate, which is
        required by the function's date_l parsing logic.
        """
        return xr.DataArray(
            np.ones((1, 4, 4)) * 99.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-15")},
        )

    @pytest.fixture
    def stable_reference(self):
        """A stable_reference DataArray with a date before cstack's start.

        Uses 1D string array for acquisition_date coordinate, which is
        required by the function's date_s parsing and .sel() logic.
        """
        return xr.DataArray(
            np.ones((1, 4, 4)) * 77.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-20"])},
        )

    def test_get_current_reference_no_last_or_stable(self):
        """With no last_reference or stable_reference, should use cstack's earliest date."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert isinstance(current, xr.DataArray)
        # Values should match cstack's earliest date slice (excluding starting_date attr)
        np.testing.assert_allclose(current.values, cstack.sel(acquisition_date="2020-01-01").values)

    def test_get_current_reference_no_last_or_stable_returns_dates(self):
        """Dates should be the acquisition_date coordinate values of the current reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        # dates from string-coord cstack is a string array
        np.testing.assert_array_equal(dates, np.array(["2020-01-01"]))

    def test_get_current_reference_no_last_or_stable_type(self):
        """Output types: current is DataArray, dates is array-like."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert isinstance(current, xr.DataArray)
        assert hasattr(dates, "__iter__") or isinstance(dates, np.datetime64)

    def test_get_current_reference_no_last_or_stable_shape(self, cstack_3_dates):
        """Shape should match the cstack's earliest date slice."""
        current, _ = get_current_reference(cstack_3_dates, None, None)
        assert current.shape == cstack_3_dates.shape[1:]

    def test_get_current_reference_no_last_or_stable_shape_dims(self, cstack_3_dates):
        """Dimensions should match the cstack's earliest date slice."""
        current, _ = get_current_reference(cstack_3_dates, None, None)
        assert current.dims == ("az", "rg")

    def test_get_current_reference_no_last_or_stable_starting_date_attr(self, cstack_3_dates):
        """starting_date attribute should be set when last_reference is None."""
        current, _ = get_current_reference(cstack_3_dates, None, None)
        assert "starting_date" in current.attrs
        assert current.attrs["starting_date"] == "2020-01-01"

    def test_get_current_reference_no_last_or_stable_starting_date_not_set_with_last(self, cstack_3_dates, last_reference):
        """starting_date should NOT be set when last_reference is provided."""
        current, _ = get_current_reference(cstack_3_dates, last_reference, None)
        assert "starting_date" not in current.attrs

    def test_get_current_reference_with_last_reference(self):
        """Should use last_reference when it predates cstack and no stable_reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 42.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        xr.testing.assert_identical(current, last_ref)
        assert dates == np.datetime64("2019-12-01")

    def test_get_current_reference_with_last_reference_uses_last_values(self):
        """Current reference values should come from last_reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.full((1, 4, 4), 99.5),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-11-01")},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        np.testing.assert_allclose(current.values, 99.5)

    def test_get_current_reference_with_last_reference_dims(self):
        """Current reference should preserve last_reference's dims when used."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        assert current.dims == ("acquisition_date", "az", "rg")

    def test_get_current_reference_with_stable_reference(self):
        """Should select stable_reference when older than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-20"])},
        )
        current, dates = get_current_reference(cstack, None, stable_ref)
        assert isinstance(current, xr.DataArray)
        # dates from string-coord stable_ref is a string array
        np.testing.assert_array_equal(dates, np.array(["2019-12-20"]))

    def test_get_current_reference_with_stable_reference_uses_stable_values(self):
        """Current reference values should come from stable_reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.full((1, 4, 4), 77.0),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-10-01"])},
        )
        current, _ = get_current_reference(cstack, None, stable_ref)
        np.testing.assert_allclose(current.values, 77.0)

    def test_get_current_reference_with_stable_reference_drops_dim(self):
        """Stable reference should lose acquisition_date dim after .sel()."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-20"])},
        )
        current, _ = get_current_reference(cstack, None, stable_ref)
        # .sel() drops the acquisition_date dimension
        assert current.dims == ("az", "rg")
        assert current.shape == (4, 4)

    def test_get_current_reference_with_stable_reference_and_last(self):
        """With both references: when last < stable, should use stable."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 11.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-11-01")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 22.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-01"])},
        )
        current, dates = get_current_reference(cstack, last_ref, stable_ref)
        np.testing.assert_allclose(current.values, 22.0)
        # dates from stable_ref is a string array
        np.testing.assert_array_equal(dates, np.array(["2019-12-01"]))

    def test_get_current_reference_with_both_refs_last_older(self):
        """With both references: when last < stable, stable is chosen."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 1.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-01-01")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 2.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-06-01"])},
        )
        current, _ = get_current_reference(cstack, last_ref, stable_ref)
        np.testing.assert_allclose(current.values, 2.0)

    def test_get_current_reference_with_both_refs_last_newer(self):
        """With both references: when last >= stable, last is chosen."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 10.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-25")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 20.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-20"])},
        )
        current, dates = get_current_reference(cstack, last_ref, stable_ref)
        np.testing.assert_allclose(current.values, 10.0)
        assert dates == np.datetime64("2019-12-25")

    def test_get_current_reference_with_both_refs_same_date(self):
        """With both references: when last == stable dates, last is chosen (last >= stable)."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        same_date = np.datetime64("2019-12-01")
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 10.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": same_date},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 20.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-01"])},
        )
        current, dates = get_current_reference(cstack, last_ref, stable_ref)
        np.testing.assert_allclose(current.values, 10.0)
        assert dates == np.datetime64("2019-12-01")

    def test_get_current_reference_raises_if_last_too_recent(self, cstack_3_dates, last_reference):
        """Should raise if last_reference is more recent than cstack."""
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2020-03-01")},
        )
        with pytest.raises(Exception, match="Last current reference"):
            get_current_reference(cstack_3_dates, last_ref, None)

    def test_get_current_reference_raises_last_more_recent_than_cstack_with_stable(self):
        """Should raise if last >= stable and last >= cstack start."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2020-02-01")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-01"])},
        )
        with pytest.raises(Exception, match="Last current reference"):
            get_current_reference(cstack, last_ref, stable_ref)

    def test_get_current_reference_raises_stable_too_recent(self):
        """Should raise if stable_reference is more recent than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2020-02-01"])},
        )
        with pytest.raises(Exception, match="Last stable reference"):
            get_current_reference(cstack, None, stable_ref)

    def test_get_current_reference_raises_both_too_recent(self):
        """Should raise if both references are more recent than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2020-03-01")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2020-02-01"])},
        )
        with pytest.raises(Exception, match="Last current reference"):
            get_current_reference(cstack, last_ref, stable_ref)

    def test_get_current_reference_raises_stable_not_yet_reached(self):
        """Should raise if stable date has not yet been reached by cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2020-01-01"])},
        )
        with pytest.raises(Exception, match="Last stable reference"):
            get_current_reference(cstack, None, stable_ref)

    def test_get_current_reference_earliest_date_selected(self):
        """Should always use the minimum date from cstack as the reference point."""
        cstack = xr.DataArray(
            np.arange(60, dtype=np.float64).reshape(5, 3, 4),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        np.testing.assert_array_equal(dates, np.array(["2020-01-01"]))

    def test_get_current_reference_uses_cstack_earliest_when_no_refs(self):
        """When no references, the current should be cstack sliced at the earliest date."""
        cstack = xr.DataArray(
            np.arange(60, dtype=np.float64).reshape(5, 3, 4),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]},
        )
        current, _ = get_current_reference(cstack, None, None)
        expected = cstack.sel(acquisition_date="2020-01-01")
        np.testing.assert_allclose(current.values, expected.values)

    def test_get_current_reference_single_date_cstack(self):
        """Single-date cstack should work when no references are provided."""
        cstack = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert current.shape == (4, 4)
        np.testing.assert_array_equal(dates, np.array(["2020-01-01"]))

    def test_get_current_reference_single_date_cstack_with_last(self):
        """Single-date cstack with a valid last_reference should return last."""
        cstack = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 5.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        np.testing.assert_allclose(current.values, 5.0)
        assert dates == np.datetime64("2019-12-01")

    def test_get_current_reference_datetime_coords(self):
        """Should handle datetime64 coordinates on cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert isinstance(current, xr.DataArray)
        assert isinstance(dates, np.datetime64)

    def test_get_current_reference_datetime_coords_with_last(self):
        """Should handle datetime64 coordinates with last_reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 33.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        np.testing.assert_allclose(current.values, 33.0)
        assert dates == np.datetime64("2019-12-01")

    def test_get_current_reference_complex_data(self):
        """Should preserve complex data types in current reference."""
        data = np.arange(60, dtype=np.complex128) * (1 + 0.5j)
        data = data.reshape(5, 3, 4)
        cstack = xr.DataArray(
            data,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 3, 4), dtype=np.complex128) * (2.0 + 1.0j),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        assert np.iscomplexobj(current.values)
        np.testing.assert_allclose(current.values.real, 2.0)
        np.testing.assert_allclose(current.values.imag, 1.0)

    def test_get_current_reference_complex_stable(self):
        """Should preserve complex data from stable_reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4), dtype=np.complex128) * (3.0 + 4.0j),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-11-01"])},
        )
        current, _ = get_current_reference(cstack, None, stable_ref)
        assert np.iscomplexobj(current.values)
        np.testing.assert_allclose(current.values.real, 3.0)
        np.testing.assert_allclose(current.values.imag, 4.0)

    def test_get_current_reference_last_with_nan_values(self):
        """Should return last_reference even if it contains NaN values."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.full((1, 4, 4), np.nan),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        assert np.all(np.isnan(current.values))

    def test_get_current_reference_last_preserves_attrs(self):
        """Current reference should preserve last_reference's attrs."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
            attrs={"some_meta": "value", "processing": "coregistered"},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        assert current.attrs.get("some_meta") == "value"
        assert current.attrs.get("processing") == "coregistered"

    def test_get_current_reference_last_equals_cstack_date(self):
        """When last_reference date equals cstack min date, should raise."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2020-01-01")},
        )
        with pytest.raises(Exception, match="Last current reference"):
            get_current_reference(cstack, last_ref, None)

    def test_get_current_reference_last_less_than_stable_uses_stable(self):
        """When last < stable dates: stable is chosen since date_s < date_l."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 11.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-11-01")},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 22.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-01"])},
        )
        current, _ = get_current_reference(cstack, last_ref, stable_ref)
        np.testing.assert_allclose(current.values, 22.0)

    def test_get_current_reference_returns_two_values(self):
        """Function should return exactly two values."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        result = get_current_reference(cstack, None, None)
        assert len(result) == 2
        current, dates = result

    @pytest.mark.parametrize(
        ("cstack_dates", "last_date", "stable_date", "expected_value", "expected_date"),
        [
            pytest.param(
                ["2020-01-01", "2020-01-13", "2020-01-25"],
                None,
                None,
                "cstack_earliest",
                "2020-01-01",
                id="no_refs_cstack",
            ),
            pytest.param(
                ["2020-01-01", "2020-01-13", "2020-01-25"],
                "2019-12-01",
                None,
                99.0,
                np.datetime64("2019-12-01"),
                id="last_only",
            ),
            pytest.param(
                ["2020-01-01", "2020-01-13", "2020-01-25"],
                None,
                "2019-11-01",
                77.0,
                np.array(["2019-11-01"]),
                id="stable_only",
            ),
            pytest.param(
                ["2020-01-01", "2020-01-13", "2020-01-25"],
                "2019-11-01",
                "2019-12-01",
                22.0,
                np.array(["2019-12-01"]),
                id="both_refs_stable_newer",
            ),
            pytest.param(
                ["2020-01-01", "2020-01-13", "2020-01-25"],
                "2019-12-25",
                "2019-12-20",
                10.0,
                np.datetime64("2019-12-25"),
                id="both_refs_last_newer",
            ),
        ],
    )
    def test_get_current_reference_parametrized_scenarios(
        self,
        cstack_dates,
        last_date,
        stable_date,
        expected_value,
        expected_date,
    ):
        """Parametrized happy-path scenarios for get_current_reference."""
        cstack = xr.DataArray(
            np.ones((len(cstack_dates), 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": cstack_dates},
        )

        if last_date is not None:
            last_val = 10.0 if expected_value == 10.0 else expected_value
            last_ref = xr.DataArray(
                np.ones((1, 4, 4)) * last_val,
                dims=["acquisition_date", "az", "rg"],
                coords={"acquisition_date": np.datetime64(last_date)},
            )
        else:
            last_ref = None

        if stable_date is not None:
            stable_ref = xr.DataArray(
                np.ones((1, 4, 4)) * 22.0 if expected_value == 22.0 else 77.0,
                dims=["acquisition_date", "az", "rg"],
                coords={"acquisition_date": np.array([stable_date])},
            )
        else:
            stable_ref = None

        if expected_value == "cstack_earliest":
            current, dates = get_current_reference(cstack, last_ref, stable_ref)
            np.testing.assert_array_equal(dates, np.array([expected_date]))
            np.testing.assert_allclose(current.values, 1.0)
        else:
            current, dates = get_current_reference(cstack, last_ref, stable_ref)
            np.testing.assert_allclose(current.values, expected_value)

    def test_get_current_reference_different_shapes_last(self):
        """Should work when last_reference has different spatial shape than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 8, 8)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)) * 55.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, _ = get_current_reference(cstack, last_ref, None)
        assert current.shape == (1, 4, 4)
        np.testing.assert_allclose(current.values, 55.0)

    def test_get_current_reference_different_shapes_stable(self):
        """Should work when stable_reference has different spatial shape than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 8, 8)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        stable_ref = xr.DataArray(
            np.ones((1, 6, 6)) * 88.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-01"])},
        )
        current, _ = get_current_reference(cstack, None, stable_ref)
        assert current.shape == (6, 6)
        np.testing.assert_allclose(current.values, 88.0)

    def test_get_current_reference_multiband_cstack(self):
        """Should handle cstack with extra dimensions (e.g. multiband)."""
        cstack = xr.DataArray(
            np.ones((2, 3, 4, 4)),
            dims=["band", "acquisition_date", "az", "rg"],
            coords={"band": ["Lband", "Cband"], "acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert current.shape == (2, 4, 4)
        np.testing.assert_array_equal(dates, np.array(["2020-01-01"]))

    def test_get_current_reference_last_multiband(self):
        """Should handle last_reference with multiple bands."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 2, 4, 4)) * 15.0,
            dims=["acquisition_date", "band", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01"), "band": ["Lband", "Cband"]},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        assert current.shape == (1, 2, 4, 4)
        np.testing.assert_allclose(current.values, 15.0)

    def test_get_current_reference_noncontiguous_dates(self):
        """Should handle non-contiguous date spacing."""
        cstack = xr.DataArray(
            np.ones((4, 3, 3)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-06-15", "2020-08-20", "2021-03-01"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        np.testing.assert_array_equal(dates, np.array(["2020-01-01"]))

    def test_get_current_reference_noncontiguous_dates_with_last(self):
        """Non-contiguous dates should still correctly pick last_reference."""
        cstack = xr.DataArray(
            np.ones((4, 3, 3)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-06-15", "2020-08-20", "2021-03-01"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 3, 3)) * 42.0,
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-25")},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        np.testing.assert_allclose(current.values, 42.0)
        assert dates == np.datetime64("2019-12-25")

    def test_get_current_reference_starting_date_format(self, cstack_3_dates):
        """starting_date attribute should be a string representation of the date."""
        current, _ = get_current_reference(cstack_3_dates, None, None)
        assert isinstance(current.attrs["starting_date"], str)
        assert current.attrs["starting_date"] == "2020-01-01"

    def test_get_current_reference_ascending_date_order(self):
        """Should handle cstack with dates in ascending order (standard)."""
        dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
        cstack = xr.DataArray(
            np.ones((4, 3, 3)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": dates},
        )
        current, result_dates = get_current_reference(cstack, None, None)
        np.testing.assert_array_equal(result_dates, np.array(["2020-01-01"]))

    def test_get_current_reference_returns_correct_date_for_each_scenario(self):
        """Verify dates match the acquisition_date coordinate of the returned reference."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2019-12-01")},
        )
        current, dates = get_current_reference(cstack, last_ref, None)
        assert dates == current.acquisition_date.values


class TestUpdateDrift:
    """Tests for update_drift."""

    def test_update_drift_basic(self):
        """Should return coherence DataArray."""
        drift_interf = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        reference = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        stable_ref = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        result = update_drift(
            drift_interf,
            reference,
            stable_ref,
            multilook={"azimuth_time": 3, "slant_range_time": 3},
            coherence_threshold=0.3,
        )
        assert isinstance(result, np.ndarray)

    def test_update_drift_no_nan(self):
        """Should handle drift_interf with no NaN values."""
        drift_interf = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        reference = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        stable_ref = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        result = update_drift(
            drift_interf,
            reference,
            stable_ref,
            multilook={},
            coherence_threshold=0.3,
        )
        assert isinstance(result, np.ndarray)


class TestUpdateRecursiveReference:
    """Tests for update_recursive_reference."""

    def test_update_recursive_reference_basic(self):
        """Basic update should return correct shape."""
        slc = xr.DataArray(
            np.ones((1, 6, 6), dtype=np.complex128),
            dims=["n_bands", "az", "rg"],
        )
        current_ref = xr.DataArray(np.ones((1, 6, 6), dtype=np.complex128), dims=["n_bands", "az", "rg"])
        normalized_interf = xr.DataArray(
            np.ones((1, 2, 2), dtype=np.complex128) * 0.8,
            dims=["n_bands", "coarse_az", "coarse_rg"],
        )
        result = update_recursive_reference(slc, current_ref, normalized_interf, memory_param=0.5)
        assert result.shape == slc.shape

    def test_update_recursive_reference_nan_handling(self):
        """NaN values should be replaced with 0."""
        slc = xr.DataArray(
            np.ones((1, 6, 6), dtype=np.complex128),
            dims=["n_bands", "az", "rg"],
        )
        current_ref = xr.DataArray(np.ones((1, 6, 6), dtype=np.complex128), dims=["n_bands", "az", "rg"])
        normalized_interf = xr.DataArray(
            np.array([[[np.nan, 0.5], [0.3, 0.2]]]),
            dims=["n_bands", "coarse_az", "coarse_rg"],
        )
        result = update_recursive_reference(slc, current_ref, normalized_interf, memory_param=0.5)
        assert not np.any(np.isnan(result.values))


class TestGetRadarSpacing:
    """Tests for get_radar_spacing."""

    def test_get_radar_spacing_basic(self):
        """Should return azimuth and range steps."""
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        rg = xr.DataArray(np.array([0.0, 0.5, 1.0]), dims=["slant_range_time"])
        az_step, rg_step = get_radar_spacing(az, rg)
        assert az_step is not None
        assert rg_step is not None


class TestWarpCmlp:
    """Tests for warp_cmlp."""

    def test_warp_cmlp_phase(self):
        """Should warp phase data correctly."""
        source = np.ones((4, 4), dtype=np.complex128)
        target = np.zeros((1, 4, 4), dtype=np.complex128)
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target,
            target_indexes,
            source,
            x,
            y,
            kernel,
            is_phase=True,
            is_coherence=False,
        )
        assert target.shape == (1, 4, 4)

    def test_warp_cmlp_coherence_clipped(self):
        """Coherence output should be clipped to [0, 1]."""
        source = np.ones((4, 4), dtype=np.float64)
        target = np.zeros((1, 4, 4))
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target,
            target_indexes,
            source,
            x,
            y,
            kernel,
            is_phase=False,
            is_coherence=True,
        )
        assert np.all(target >= 0.0)
        assert np.all(target <= 1.0)

    def test_warp_cmlp_no_error_standard(self):
        """Standard warp should not raise."""
        source = np.random.rand(4, 4)
        target = np.zeros((1, 4, 4))
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target,
            target_indexes,
            source,
            x,
            y,
            kernel,
            is_phase=False,
            is_coherence=False,
        )
        assert target.shape == (1, 4, 4)
