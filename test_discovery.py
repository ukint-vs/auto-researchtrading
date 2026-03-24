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
    _default_universe,
    DEFAULT_SYMBOLS,
    UNIVERSE_CACHE,
    CAP_TIERS,
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
        mock_post.side_effect = Exception("timeout")
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
            with patch("prepare.UNIVERSE_CACHE", f.name):
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


class TestLoadDataSymbols(unittest.TestCase):
    def test_load_data_accepts_symbols_param(self):
        """load_data with explicit symbols should only load those symbols."""
        data = load_data("val", symbols=["BTC", "ETH"])
        assert all(s in ["BTC", "ETH"] for s in data.keys())

    def test_load_data_default_uses_training(self):
        """load_data without symbols arg uses get_symbols('training')."""
        with patch("prepare.get_symbols", return_value=["BTC", "ETH"]):
            data = load_data("val")
            assert all(s in ["BTC", "ETH"] for s in data.keys())


if __name__ == "__main__":
    unittest.main()
