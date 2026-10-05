"""Offline unit tests for cop_ard_core.iono_height."""

from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
import retry.api

from cop_ard_core import iono_height


@pytest.fixture(autouse=True)
def offline_dependencies(monkeypatch):
    """Reject unexpected remote CSV reads and real PyIRI model evaluations."""
    read_csv = pd.read_csv

    def read_local_csv(source, *args, **kwargs):
        if isinstance(source, str) and "://" in source:
            pytest.fail(f"Unexpected remote CSV read: {source}")
        return read_csv(source, *args, **kwargs)

    monkeypatch.setattr(iono_height.pd, "read_csv", read_local_csv)
    model = Mock(side_effect=AssertionError("Unexpected PyIRI model evaluation"))
    monkeypatch.setattr(iono_height.ml, "IRI_density_1day", model)
    return model


@pytest.fixture
def weather_table():
    """Small unindexed space-weather table with distinct daily fluxes."""
    return pd.DataFrame({"DATE": ["2024-02-28", "2024-02-29"], "F10.7_OBS": [99.1, 143.5]})


@pytest.fixture
def weather_db(weather_table):
    """Expected database indexed by midnight timestamps."""
    return weather_table.assign(DATE=pd.to_datetime(weather_table["DATE"])).set_index("DATE")


@pytest.fixture
def dbfile(tmp_path):
    """Local CSV fixture, independent of network or external data files."""
    path = tmp_path / "space_weather.csv"
    path.write_text("DATE,F10.7_OBS\n2024-02-28,99.1\n2024-02-29,143.5\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def retry_sleep(monkeypatch):
    """Replace the sleep used by the public retry-decorated initializer."""
    sleep = Mock()
    monkeypatch.setattr(retry.api.time, "sleep", sleep)
    return sleep


@pytest.mark.unit
class TestConvertEcefToLlh:
    """Tests for convert_ecef_to_llh using independent WGS84 geometry."""

    @pytest.mark.parametrize(
        "ecef,expected",
        [
            pytest.param([6378137.0, 0.0, 0.0], [0.0, 0.0, 0.0], id="equatorial-origin"),
            pytest.param([0.0, 6379387.0, 0.0], [90.0, 0.0, 1250.0], id="east-with-height"),
            pytest.param([0.0, -6378137.0, 0.0], [-90.0, 0.0, 0.0], id="west"),
            pytest.param([0.0, 0.0, 6357002.314245179], [0.0, 90.0, 250.0], id="north-pole"),
        ],
    )
    def test_known_geometric_points(self, ecef, expected):
        """Return longitude before latitude, degrees, and altitude in meters."""
        result = iono_height.convert_ecef_to_llh(np.array(ecef))

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)

    def test_oblique_point_from_ellipsoid_formula(self):
        """Recover distinct nonzero angles without using pyproj as the oracle."""
        longitude, latitude, height = -135.0, -30.0, 430.0
        lon, lat = np.deg2rad([longitude, latitude])
        flattening = 1 / 298.257223563
        eccentricity_squared = flattening * (2 - flattening)
        radius = 6378137.0 / np.sqrt(1 - eccentricity_squared * np.sin(lat) ** 2)
        ecef = np.array(
            [
                (radius + height) * np.cos(lat) * np.cos(lon),
                (radius + height) * np.cos(lat) * np.sin(lon),
                (radius * (1 - eccentricity_squared) + height) * np.sin(lat),
            ]
        )

        np.testing.assert_allclose(
            iono_height.convert_ecef_to_llh(ecef), [longitude, latitude, height], rtol=0, atol=1e-6
        )


@pytest.mark.unit
class TestGetF107:
    """Tests for get_f107."""

    @pytest.mark.parametrize("day,expected", [(28, 99.1), (29, 143.5)])
    def test_date_lookup_converts_to_float(self, weather_db, day, expected):
        """Select the requested date and convert string flux values to float."""
        weather_db["F10.7_OBS"] = weather_db["F10.7_OBS"].astype(str)

        result = iono_height.get_f107(weather_db, 2024, 2, day)

        assert isinstance(result, float)
        assert result == expected

    @pytest.mark.parametrize("missing", ["date", "column"])
    def test_missing_data_preserves_key_error(self, weather_db, missing):
        """Both absent dates and absent flux columns get a dated ValueError."""
        if missing == "date":
            weather_db = weather_db.iloc[:1]
        else:
            weather_db = weather_db.drop(columns="F10.7_OBS")

        with pytest.raises(ValueError, match="F10.7 solar flux data missing.*2024-02-29") as exc_info:
            iono_height.get_f107(weather_db, 2024, 2, 29)

        assert isinstance(exc_info.value.__cause__, KeyError)


@pytest.mark.unit
class TestInitIonoDb:
    """Tests for the public retry-decorated init_iono_db."""

    def test_existing_csv_is_parsed_and_indexed(self, dbfile, weather_db):
        """Read a real local CSV and replace DATE with a DatetimeIndex."""
        result = iono_height.init_iono_db(dbfile=dbfile)

        pd.testing.assert_frame_equal(result, weather_db)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "DATE" not in result.columns

    @pytest.mark.parametrize("missing_file", [False, True], ids=["no-dbfile", "missing-dbfile"])
    @pytest.mark.parametrize("custom_url", [False, True], ids=["default-url", "custom-url"])
    def test_download_source_selection(
        self, monkeypatch, tmp_path, weather_table, weather_db, missing_file, custom_url
    ):
        """Use the selected URL for both None and nonexistent local paths."""
        read_csv = Mock(return_value=weather_table.copy())
        monkeypatch.setattr(iono_height.pd, "read_csv", read_csv)
        dbfile = str(tmp_path / "missing.csv") if missing_file else None
        url = "https://example.invalid/weather.csv" if custom_url else "https://celestrak.org/SpaceData/SW-All.csv"
        kwargs = {"url": url} if custom_url else {}

        result = iono_height.init_iono_db(dbfile=dbfile, **kwargs)

        read_csv.assert_called_once_with(url)
        pd.testing.assert_frame_equal(result, weather_db)
        if dbfile is not None:
            assert not (tmp_path / "missing.csv").exists()

    @pytest.mark.parametrize("stage", ["read", "parse"])
    def test_failures_are_retried_and_wrapped(self, monkeypatch, dbfile, weather_table, retry_sleep, stage):
        """Exhaust all attempts without waiting, retaining the original cause."""
        cause = OSError("CSV unavailable") if stage == "read" else ValueError("invalid DATE values")
        read_csv = Mock(side_effect=cause) if stage == "read" else Mock(return_value=weather_table.copy())
        monkeypatch.setattr(iono_height.pd, "read_csv", read_csv)
        if stage == "parse":
            monkeypatch.setattr(iono_height.pd, "to_datetime", Mock(side_effect=cause))

        with pytest.raises(RuntimeError, match=f"Could not initialize ionospheric database: {cause}") as exc_info:
            iono_height.init_iono_db(dbfile=dbfile)

        assert exc_info.value.__cause__ is cause
        assert read_csv.call_args_list == [call(dbfile)] * 5
        assert retry_sleep.call_count == 4

    @pytest.mark.parametrize(
        "table,cause_type",
        [
            pytest.param(pd.DataFrame({"DATE": ["not-a-date"], "F10.7_OBS": [100]}), ValueError, id="invalid-date"),
            pytest.param(pd.DataFrame({"F10.7_OBS": [100]}), KeyError, id="missing-date-column"),
        ],
    )
    def test_malformed_tables_are_wrapped(self, monkeypatch, dbfile, retry_sleep, table, cause_type):
        """Actual pandas parsing/schema failures are wrapped by the public API."""
        read_csv = Mock(side_effect=lambda *args: table.copy())
        monkeypatch.setattr(iono_height.pd, "read_csv", read_csv)

        with pytest.raises(RuntimeError, match="Could not initialize ionospheric database") as exc_info:
            iono_height.init_iono_db(dbfile=dbfile)

        assert isinstance(exc_info.value.__cause__, cause_type)
        assert read_csv.call_count == 5
        assert retry_sleep.call_count == 4

    def test_transient_read_failure_retries_successfully(
        self, monkeypatch, dbfile, weather_table, weather_db, retry_sleep
    ):
        """A failed first read can recover through the decorated public function."""
        read_csv = Mock(side_effect=[OSError("temporary failure"), weather_table.copy()])
        monkeypatch.setattr(iono_height.pd, "read_csv", read_csv)

        result = iono_height.init_iono_db(dbfile=dbfile)

        pd.testing.assert_frame_equal(result, weather_db)
        assert read_csv.call_args_list == [call(dbfile), call(dbfile)]
        retry_sleep.assert_called_once_with(1)


@pytest.mark.unit
class TestReadIonoDb:
    """Tests for read_iono_db."""

    def test_local_csv_lookup(self, dbfile):
        """Resolve a daily flux through real local initialization and lookup."""
        assert iono_height.read_iono_db(2024, 2, 29, dbfile=dbfile) == 143.5

    @pytest.mark.parametrize("dbfile", [None, "custom-weather.csv"])
    def test_delegates_database_and_date(self, monkeypatch, weather_db, dbfile):
        """Pass dbfile to initialization and the resulting frame/date to lookup."""
        initialize = Mock(return_value=weather_db)
        lookup = Mock(return_value=123.4)
        monkeypatch.setattr(iono_height, "init_iono_db", initialize)
        monkeypatch.setattr(iono_height, "get_f107", lookup)

        assert iono_height.read_iono_db(2024, 2, 29, dbfile=dbfile) == 123.4

        initialize.assert_called_once_with(dbfile=dbfile)
        lookup.assert_called_once_with(weather_db, 2024, 2, 29)

    @pytest.mark.parametrize("stage", ["initialize", "lookup"])
    def test_dependency_errors_propagate(self, monkeypatch, weather_db, stage):
        """Do not wrap or suppress errors from either delegated operation."""
        cause = RuntimeError("database unavailable") if stage == "initialize" else ValueError("missing flux")
        initialize = Mock(side_effect=cause) if stage == "initialize" else Mock(return_value=weather_db)
        lookup = Mock(side_effect=cause)
        monkeypatch.setattr(iono_height, "init_iono_db", initialize)
        monkeypatch.setattr(iono_height, "get_f107", lookup)

        with pytest.raises(type(cause)) as exc_info:
            iono_height.read_iono_db(2024, 2, 29, dbfile="weather.csv")

        assert exc_info.value is cause
        if stage == "initialize":
            lookup.assert_not_called()


@pytest.mark.unit
class TestGetIonoHeight:
    """Tests for get_iono_height with a mocked PyIRI model."""

    @pytest.mark.parametrize("as_timestamp", [False, True], ids=["string", "timestamp"])
    def test_model_arguments_and_peak_height(self, monkeypatch, offline_dependencies, dbfile, tmp_path, as_timestamp):
        """Forward date/time, ordered coordinates, grid, flux and coefficients."""
        acquisition_date = "2024-02-29T14:23:45"
        if as_timestamp:
            acquisition_date = pd.Timestamp(acquisition_date)
        convert = Mock(return_value=np.array([-123.25, 48.75, 875.0]))
        monkeypatch.setattr(iono_height, "convert_ecef_to_llh", convert)
        monkeypatch.setattr(iono_height.PyIRI, "coeff_dir", "/mock/iri/coefficients")
        read_database = Mock(wraps=iono_height.read_iono_db)
        monkeypatch.setattr(iono_height, "read_iono_db", read_database)
        model = offline_dependencies
        model.side_effect = None
        model.return_value = ({"hm": np.array([[312.75, 999.0], [444.0, 555.0]])}, {}, {}, {}, {}, {}, np.zeros(1))
        # Deliberately different insertion order from the required ECEF XYZ order.
        scp = {"Z": 300.0, "X": 100.0, "Y": 200.0}

        result = iono_height.get_iono_height(acquisition_date, scp, str(tmp_path), dbfile=dbfile)

        assert isinstance(result, float)
        assert result == 312750.0
        convert.assert_called_once()
        np.testing.assert_array_equal(convert.call_args.args[0], [100.0, 200.0, 300.0])
        read_database.assert_called_once_with(2024, 2, 29, dbfile=dbfile)
        model.assert_called_once()
        args = model.call_args.args
        assert len(args) == 9
        assert model.call_args.kwargs == {}
        assert args[:3] == (2024, 2, 29)
        np.testing.assert_allclose(args[3], [14 + 23 / 60 + 45 / 3600], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(args[4], [-123.25])
        np.testing.assert_array_equal(args[5], [48.75])
        np.testing.assert_array_equal(args[6], np.arange(90, 1001, 5))
        assert args[7:] == (143.5, "/mock/iri/coefficients")

    def test_model_failure_preserves_cause(self, offline_dependencies, dbfile, tmp_path):
        """Wrap model errors in a descriptive RuntimeError with the same cause."""
        cause = ValueError("model coefficients invalid")
        offline_dependencies.side_effect = cause

        with pytest.raises(RuntimeError, match="Error computing ionospheric density via PyIRI") as exc_info:
            iono_height.get_iono_height(
                "2024-02-29T12:00:00", {"X": 6378137.0, "Y": 0.0, "Z": 0.0}, str(tmp_path), dbfile=dbfile
            )

        assert exc_info.value.__cause__ is cause
        assert str(cause) in str(exc_info.value)
        offline_dependencies.assert_called_once()

    def test_missing_solar_data_propagates_before_model(self, offline_dependencies, dbfile, tmp_path):
        """Missing daily flux remains a ValueError and never invokes PyIRI."""
        with pytest.raises(ValueError, match="F10.7 solar flux data missing.*2024-03-01") as exc_info:
            iono_height.get_iono_height(
                "2024-03-01T12:00:00", {"X": 6378137.0, "Y": 0.0, "Z": 0.0}, str(tmp_path), dbfile=dbfile
            )

        assert isinstance(exc_info.value.__cause__, KeyError)
        offline_dependencies.assert_not_called()
