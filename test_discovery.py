"""Tests for coin universe discovery, get_symbols, and load_data with symbols param."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from prepare import (
    discover_coins,
    get_symbols,
    save_universe,
    load_data,
    _select_training_subset,
    _validate_symbols,
    _default_universe,
    DEFAULT_SYMBOLS,
    UNIVERSE_CACHE,
    CAP_TIERS,
    DATA_DIR,
    PIPELINE_SIZE,
    TRAINING_SIZE,
    STABLECOINS,
    MIN_DAILY_VOLUME_USD,
)


class TestDefaultUniverse(unittest.TestCase):
    def test_returns_default_symbols(self):
        u = _default_universe()
        assert u["pipeline"] == list(DEFAULT_SYMBOLS)
        assert u["training"] == list(DEFAULT_SYMBOLS)
        assert u["version"] == "v0-default"


class TestSelectTrainingSubset(unittest.TestCase):
    def test_basic_selection(self):
        pipeline = ["BTC", "ETH", "SOL", "DOGE", "XRP", "HYPE", "kPEPE",
                     "FARTCOIN", "PUMP", "ASTER", "ZRO", "SUI", "ENA", "ADA"]
        result = _select_training_subset(pipeline)
        assert len(result) == TRAINING_SIZE
        assert "BTC" in result
        assert "ETH" in result

    def test_tier_diversity(self):
        pipeline = ["BTC", "ETH", "SOL", "DOGE", "XRP", "HYPE", "kPEPE",
                     "FARTCOIN", "PUMP", "ASTER", "ZRO", "SUI", "ENA", "ADA"]
        result = _select_training_subset(pipeline)
        tier1 = [s for s in result if CAP_TIERS.get(s, 3) == 1]
        tier2 = [s for s in result if CAP_TIERS.get(s, 3) == 2]
        assert len(tier1) >= 2, f"Expected >= 2 tier-1 coins, got {tier1}"
        assert len(tier2) >= 3, f"Expected >= 3 tier-2 coins, got {tier2}"

    def test_small_pipeline(self):
        """If pipeline has fewer than TRAINING_SIZE coins, return all."""
        pipeline = ["BTC", "ETH", "SOL"]
        result = _select_training_subset(pipeline)
        assert result == ["BTC", "ETH", "SOL"]

    def test_btc_eth_always_first(self):
        pipeline = ["HYPE", "ETH", "BTC", "SOL", "XRP", "DOGE",
                     "kPEPE", "FARTCOIN", "PUMP", "ASTER", "ZRO"]
        result = _select_training_subset(pipeline)
        assert result[0] == "BTC"
        assert result[1] == "ETH"


class TestDiscoverCoins(unittest.TestCase):
    @patch("prepare.requests.post")
    def test_api_failure_returns_default(self, mock_post):
        import requests as req
        mock_post.side_effect = req.RequestException("timeout")
        result = discover_coins()
        assert result["pipeline"] == list(DEFAULT_SYMBOLS)
        assert result["version"] == "v0-default"

    @patch("prepare.requests.post")
    def test_successful_discovery(self, mock_post):
        # Mock HL metaAndAssetCtxs response
        meta = {"universe": [{"name": f"COIN{i}"} for i in range(30)]}
        asset_ctxs = [
            {"dayNtlVlm": str(10_000_000 - i * 100_000), "openInterest": str(5_000_000 - i * 50_000)}
            for i in range(30)
        ]
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[meta, asset_ctxs]),
        )
        mock_post.return_value.raise_for_status = MagicMock()
        result = discover_coins(top_n=10)
        assert len(result["pipeline"]) == 10
        assert len(result["training"]) <= TRAINING_SIZE
        assert "version" in result

    @patch("prepare.requests.post")
    def test_all_coins_filtered_returns_default(self, mock_post):
        meta = {"universe": [{"name": "USDT"}, {"name": "USDC"}]}
        asset_ctxs = [
            {"dayNtlVlm": "100000000", "openInterest": "50000000"},
            {"dayNtlVlm": "100000000", "openInterest": "50000000"},
        ]
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[meta, asset_ctxs]),
        )
        mock_post.return_value.raise_for_status = MagicMock()
        result = discover_coins()
        assert result["pipeline"] == list(DEFAULT_SYMBOLS)


class TestGetSymbols(unittest.TestCase):
    def test_no_cache_returns_default(self):
        with patch("prepare.UNIVERSE_CACHE", "/tmp/nonexistent_cache.json"):
            result = get_symbols("training")
            assert result == list(DEFAULT_SYMBOLS)

    def test_reads_from_cache(self):
        universe = {
            "pipeline": ["BTC", "ETH", "SOL", "HYPE"],
            "training": ["BTC", "ETH", "SOL"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(universe, f)
            f.flush()
            # Mock _validate_symbols so test doesn't depend on local parquet files
            with patch("prepare.UNIVERSE_CACHE", f.name), \
                 patch("prepare._validate_symbols", return_value=["BTC", "ETH", "SOL"]):
                result = get_symbols("training")
                assert result == ["BTC", "ETH", "SOL"]
                result_pipe = get_symbols("pipeline")
                assert result_pipe == ["BTC", "ETH", "SOL", "HYPE"]
        os.unlink(f.name)

    def test_corrupt_cache_returns_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            f.flush()
            with patch("prepare.UNIVERSE_CACHE", f.name):
                result = get_symbols("training")
                assert result == list(DEFAULT_SYMBOLS)
        os.unlink(f.name)


class TestValidateSymbols(unittest.TestCase):
    def test_valid_symbols_pass(self):
        """Symbols with good data should be returned."""
        result = _validate_symbols(["BTC", "ETH"], "val")
        assert "BTC" in result
        assert "ETH" in result

    def test_missing_data_file_excluded(self):
        """Symbols with no parquet file should be excluded."""
        result = _validate_symbols(["BTC", "NONEXISTENT_COIN_XYZ"], "val")
        assert "NONEXISTENT_COIN_XYZ" not in result
        assert "BTC" in result

    def test_extreme_price_ratio_excluded(self):
        """Symbols with >1000x price ratio should be excluded."""
        # HYPE has a 23M x price ratio in val split
        result = _validate_symbols(["BTC", "HYPE"], "val")
        assert "HYPE" not in result
        assert "BTC" in result

    def test_insufficient_bars_excluded(self):
        """Symbols with < 1000 bars should be excluded."""
        # kPEPE has < 1000 bars in val split
        result = _validate_symbols(["BTC", "kPEPE"], "val")
        assert "kPEPE" not in result

    def test_all_invalid_returns_default(self):
        """If all symbols fail validation, return DEFAULT_SYMBOLS."""
        result = _validate_symbols(["NONEXISTENT_1", "NONEXISTENT_2"], "val")
        assert result == list(DEFAULT_SYMBOLS)

    def test_invalid_split_returns_all(self):
        """Invalid split name should return symbols unfiltered."""
        result = _validate_symbols(["BTC", "ETH", "FAKE"], "invalid_split")
        assert result == ["BTC", "ETH", "FAKE"]


class TestLoadDataFilters(unittest.TestCase):
    def test_load_data_accepts_symbols_param(self):
        """load_data with explicit symbols should only load those symbols."""
        data = load_data("val", symbols=["BTC", "ETH"])
        assert all(s in ["BTC", "ETH"] for s in data.keys())

    def test_load_data_default_uses_training(self):
        """load_data without symbols arg uses get_symbols('training')."""
        with patch("prepare.get_symbols", return_value=["BTC", "ETH"]):
            data = load_data("val")
            assert all(s in ["BTC", "ETH"] for s in data.keys())

    def test_zero_price_bars_filtered(self):
        """Bars with close=0 should not appear in loaded data."""
        data = load_data("val", symbols=["BTC"])
        if "BTC" in data:
            assert (data["BTC"]["close"] > 0).all()

    def test_extreme_price_ratio_excluded(self):
        """Coins with extreme price ratios should be excluded from load."""
        data = load_data("val", symbols=["HYPE"])
        assert "HYPE" not in data


if __name__ == "__main__":
    unittest.main()
