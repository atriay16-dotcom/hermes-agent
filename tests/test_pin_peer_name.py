"""Tests for the ``pinPeerName`` / ``pinUserPeer`` config flag.

Under a gateway (Telegram, Discord, Slack, ...) Hermes passes the
platform-native user ID as ``runtime_user_peer_name`` into
``HonchoSessionManager``.  By default that ID wins over any configured
``peer_name`` so multi-user bots scope memory per user.

For single-user deployments connecting over multiple platforms,
``pinUserPeer: true`` pins the user peer to ``peer_name`` so memory stays
unified across platforms.

Tests cover config parsing (``client.py::from_global_config``) and resolver
order (``session.py::get_or_create``), stubbing Honcho API calls so the
chosen ``user_peer_id`` can be asserted without touching the network.
"""

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.session import HonchoSessionManager
from hermes_plugin_honcho.session_peers import HonchoPeerUnresolvedError


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestPinPeerNameConfigParsing:
    def test_default_is_false(self):
        """Default preserves existing behaviour — multi-user bots unaffected."""
        config = HonchoClientConfig()
        assert config.pin_peer_name is False

    def test_root_level_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": True,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True
        assert config.peer_name == "Igor"

    def test_host_block_true(self, tmp_path, monkeypatch):
        """Host-level flag works the same as root-level."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "hosts": {
                "hermes": {"pinPeerName": True},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True


    def test_explicit_false_parses(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": False,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is False


class TestRuntimePeerMappingConfigParsing:
    def test_defaults_are_empty(self):
        config = HonchoClientConfig()
        assert config.user_peer_aliases == {}
        assert config.runtime_peer_prefix == ""


    def test_malformed_alias_config_is_ignored(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": ["not", "a", "map"],
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {}


# ---------------------------------------------------------------------------
# Peer resolution (the actual bug fix)
# ---------------------------------------------------------------------------


def _patch_manager_for_resolution_test(mgr: HonchoSessionManager) -> None:
    """Stub out the Honcho client so ``get_or_create`` doesn't try to talk
    to the network — we only care about the user_peer_id chosen before
    those calls happen.
    """
    fake_peer = MagicMock()
    mgr._get_or_create_peer = MagicMock(return_value=fake_peer)
    mgr._get_or_create_honcho_session = MagicMock(
        return_value=(MagicMock(), [], None)
    )


class TestPeerResolutionOrder:
    """Matrix of (runtime_id, pin_peer_name, peer_name) → expected user_peer_id."""

    def _config(
        self,
        *,
        peer_name: str | None,
        pin_peer_name: bool,
        user_peer_aliases: dict[str, str] | None = None,
        runtime_peer_prefix: str = "",
        session_peer_prefix: bool = False,
    ) -> HonchoClientConfig:
        # The test doesn't need auth / Honcho — disable the provider so
        # the manager doesn't try to open a real client.
        return HonchoClientConfig(
            api_key="test-key",
            peer_name=peer_name,
            pin_peer_name=pin_peer_name,
            user_peer_aliases=user_peer_aliases or {},
            runtime_peer_prefix=runtime_peer_prefix,
            session_peer_prefix=session_peer_prefix,
            enabled=False,
            write_frequency="turn",  # avoid spawning the async writer thread
        )

    def test_runtime_wins_when_pin_is_false(self):
        """Regression guard: default behaviour must stay unchanged.
        Multi-user bots rely on the platform-native ID winning."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=False),
            runtime_user_peer_name="7654321",  # e.g. Telegram UID
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "7654321", (
            "pin_peer_name=False is the multi-user default — the gateway's "
            "platform-native user ID must win so each user gets their own "
            "peer scope.  If this regresses, every Telegram/Discord/Slack "
            "bot immediately merges memory across users."
        )

    def test_alias_wins_for_known_runtime_id(self):
        """Known platform IDs can preserve an existing stable Honcho peer."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                user_peer_aliases={"7654321": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor"

    def test_unknown_runtime_id_uses_prefix(self):
        """Unknown gateway users stay isolated but become platform-scoped."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "telegram_7654321"

    def test_prefixed_runtime_id_hashes_when_sanitization_is_lossy(self):
        """Generated prefixed IDs avoid merges caused by lossy sanitization."""
        raw_peer_id = "telegram_user:42"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == f"telegram_user-42-{expected_hash}"

    def test_prefixed_runtime_id_hashes_when_it_collides_with_peer_name(self):
        """Unknown generated peers should not silently merge into peerName."""
        raw_peer_id = "telegram_7654321"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="telegram_7654321",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == f"telegram_7654321-{expected_hash}"


    def test_alias_value_is_sanitized_after_selection(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"7654321": "Alice Smith!"},
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Alice-Smith-"

    def test_alias_keys_match_raw_runtime_id_before_sanitization(self):
        """Alias selection is exact on platform IDs before Honcho ID cleanup."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={
                    "user:42": "raw-match",
                    "user-42": "sanitized-match",
                },
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == "raw-match"

    def test_session_peer_prefix_is_orthogonal_to_runtime_peer_prefix(self):
        """sessionPeerPrefix scopes session IDs; runtimePeerPrefix scopes user peers."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
                session_peer_prefix=True,
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "telegram_7654321"
        assert session.honcho_session_id == "telegram-7654321"

    def test_config_wins_when_pin_is_true(self):
        """With pin enabled, configured peer_name beats runtime ID."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=True,
                user_peer_aliases={"7654321": "Alias"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",  # Telegram pushes this in
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor", (
            "With pinPeerName=true the user's configured peer_name must "
            "beat the platform-native runtime ID so memory stays unified "
            "across Telegram/Discord/Slack for the same person."
        )

    def test_pin_noop_when_peer_name_missing(self):
        """Safety: pinPeerName alone (no peer_name) must not silently drop
        the runtime identity.  Without a configured peer_name there's
        nothing to pin to — fall through to runtime mapping."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=True,
                user_peer_aliases={"7654321": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor"


    def test_alt_runtime_id_can_match_alias_without_changing_raw_fallback(self):
        """Stable alternate IDs can map known users while primary ID fallback stays unchanged."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"union-user": "Igor"},
                runtime_peer_prefix="feishu_",
            ),
            runtime_user_peer_name="open-id",
            runtime_user_peer_name_alt="union-user",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:chat")
        assert session.user_peer_id == "Igor"


    @pytest.mark.parametrize("key", ["telegram:123", "rheijo5"])
    def test_everything_missing_refuses_to_mint_a_peer(self, key):
        """No runtime identity, no peer_name, no pin: the resolver used to derive
        ``user-telegram-123`` / ``user-default-rheijo5`` from the session key, which
        put desktop sessions on a phantom peer (#93326). It must refuse instead."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name=None, pin_peer_name=False),
            runtime_user_peer_name=None,
        )
        _patch_manager_for_resolution_test(mgr)

        with pytest.raises(HonchoPeerUnresolvedError, match="peerName"):
            mgr.get_or_create(key)
        assert mgr._get_or_create_peer.call_count == 0
        assert mgr._get_or_create_honcho_session.call_count == 0
        assert key not in mgr._cache

    def test_peer_name_alone_is_the_declared_owner(self):
        """No runtime identity: the configured peerName is the peer, unpinned or not."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=False),
            runtime_user_peer_name=None,
        )
        _patch_manager_for_resolution_test(mgr)

        assert mgr.get_or_create("rheijo5").user_peer_id == "Igor"


class TestCrossPlatformMemoryUnification:
    """The same physical user talking to Hermes via Telegram AND Discord
    lands on ONE peer when ``pinPeerName`` is opted in.
    """

    def _config_pinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def test_telegram_and_discord_collapse_to_one_peer_when_pinned(self):
        """Single-user deployment: Telegram UID and Discord snowflake
        both resolve to the same configured peer_name."""
        # Telegram turn
        mgr_telegram = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_telegram)
        telegram_session = mgr_telegram.get_or_create("telegram:7654321")

        # Discord turn (separate manager instance — simulates a fresh
        # platform-adapter invocation)
        mgr_discord = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="1348750102029926454",
        )
        _patch_manager_for_resolution_test(mgr_discord)
        discord_session = mgr_discord.get_or_create("discord:1348750102029926454")

        assert telegram_session.user_peer_id == "Igor"
        assert discord_session.user_peer_id == "Igor"
        assert telegram_session.user_peer_id == discord_session.user_peer_id, (
            "cross-platform memory unification is the whole point of "
            "pinPeerName — both platforms must land on the same Honcho peer"
        )

    def test_multiuser_default_keeps_platforms_separate(self):
        """Negative control: with pinPeerName=false (the default), two
        different platform IDs must produce two different peers so
        multi-user bots don't merge users."""
        cfg = HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=False,
            enabled=False,
            write_frequency="turn",
        )
        mgr_a = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_a",
        )
        mgr_b = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_b",
        )
        _patch_manager_for_resolution_test(mgr_a)
        _patch_manager_for_resolution_test(mgr_b)

        sess_a = mgr_a.get_or_create("telegram:a")
        sess_b = mgr_b.get_or_create("telegram:b")

        assert sess_a.user_peer_id == "user_a"
        assert sess_b.user_peer_id == "user_b"
        assert sess_a.user_peer_id != sess_b.user_peer_id, (
            "multi-user default MUST keep users separate — a regression "
            "here would silently merge unrelated users' memory"
        )


class TestPinUserPeerAlias:
    """``pinUserPeer`` and ``pinPeerName`` both resolve to the same internal
    ``pin_peer_name`` field.  Precedence when both appear: host pinUserPeer →
    host pinPeerName → root pinUserPeer → root pinPeerName → default.
    """


    def test_pinPeerName_still_works_unchanged(self, tmp_path):
        from hermes_plugin_honcho.client import HonchoClientConfig
        import json
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "peerName": "eri",
            "hosts": {"hermes": {"pinPeerName": True}},
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True


class TestPinTransition:
    """Behavior when honcho.json flips ``pinPeerName`` true → false.

    Covers two contracts:
      1. A freshly-built manager picks up the flipped config and resolves
         the same runtime ID to a new peer (no resolver staleness).
      2. The gateway's agent-cache signature reflects honcho identity-mapping
         changes, so a config edit busts the cached AIAgent on the next turn.
    """

    def _pinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def _unpinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=False,
            enabled=False,
            write_frequency="turn",
        )

    def test_fresh_manager_after_flip_resolves_to_runtime(self):
        pinned_mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(pinned_mgr)
        before = pinned_mgr.get_or_create("telegram:7654321")
        assert before.user_peer_id == "Igor"

        unpinned_mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._unpinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(unpinned_mgr)
        after = unpinned_mgr.get_or_create("telegram:7654321")
        assert after.user_peer_id == "7654321", (
            "After flipping pinPeerName off, the same runtime ID must resolve "
            "to its own peer — otherwise multi-user mode silently merges users."
        )

    def test_cached_session_survives_config_flip_in_same_manager(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)
        first = mgr.get_or_create("telegram:7654321")
        assert first.user_peer_id == "Igor"

        mgr._config = self._unpinned()
        second = mgr.get_or_create("telegram:7654321")
        assert second.user_peer_id == "Igor", (
            "The per-key session cache is keyed by session-key, not by "
            "resolved peer.  In-process flips don't invalidate it — the "
            "gateway cache must bust the whole manager instead."
        )

    def test_identity_signature_reflects_both_session_prefixes(self, tmp_path, monkeypatch):
        """Flipping either session prefix mid-flight must invalidate the cached agent: both feed the
        ``resolve_session_name`` output frozen into the provider's ``_session_key`` at construction."""
        from hermes_plugin_honcho import HonchoMemoryProvider

        cfg_path = tmp_path / "honcho.json"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base = {"apiKey": "k", "peerName": "Igor", "aiPeer": "hermes"}
        provider = HonchoMemoryProvider()

        cfg_path.write_text(json.dumps({**base, "sessionPeerPrefix": True, "sessionAiPeerPrefix": False}))
        sig_user_only = provider.identity_signature()["session_prefixing"]
        cfg_path.write_text(json.dumps({**base, "sessionPeerPrefix": True, "sessionAiPeerPrefix": True}))
        sig_both = provider.identity_signature()["session_prefixing"]

        assert sig_user_only != sig_both

    def test_identity_signature_reflects_a_repointed_host_workspace(self, tmp_path, monkeypatch):
        """``hermes honcho peers map`` can repoint a host block's workspace; the cached agent's manager
        is bound to the old one until the signature changes."""
        from hermes_plugin_honcho import HonchoMemoryProvider

        cfg_path = tmp_path / "honcho.json"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base = {"apiKey": "k", "peerName": "Igor", "aiPeer": "hermes"}
        provider = HonchoMemoryProvider()

        cfg_path.write_text(json.dumps({**base, "hosts": {"hermes": {"workspace": "old"}}}))
        sig_old = provider.identity_signature()["workspace"]
        cfg_path.write_text(json.dumps({**base, "hosts": {"hermes": {"workspace": "new"}}}))
        sig_new = provider.identity_signature()["workspace"]

        assert (sig_old, sig_new) == ("old", "new")


class TestProfilePeerUniqueness:
    """Each Hermes profile can pin to its own unique peerName.

    Profile cloning copies host blocks, but operators routinely diverge them
    afterwards (e.g. `hermes -p partner` pinned to a different person's peer).
    The resolver must honor host-level ``peerName`` so two profiles in the
    same workspace stay scoped to different Honcho peers.
    """

    def _pinned_to(self, name: str) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name=name,
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def test_two_profiles_pinned_to_different_peer_names_resolve_distinctly(self):
        mgr_a = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned_to("alice"),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_a)
        sess_a = mgr_a.get_or_create("telegram:7654321")

        mgr_b = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned_to("bob"),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_b)
        sess_b = mgr_b.get_or_create("telegram:7654321")

        assert sess_a.user_peer_id == "alice"
        assert sess_b.user_peer_id == "bob"
        assert sess_a.user_peer_id != sess_b.user_peer_id, (
            "Profiles pinned to distinct peer names must not collapse to "
            "the same Honcho peer — otherwise profile isolation is fictional."
        )


class TestIdentitySignatureMetadataCollision:
    """``identity_signature()`` follows honcho.json's content, not just its stat.

    Linux stamps mtime from a coarse clock (one tick, ~1 ms at HZ=1000), so two rewrites inside one
    tick keep ``(st_mtime_ns, st_size)`` whenever the payloads are the same length, as "old" and
    "new" are. These tests pin the stat with ``os.utime`` instead of relying on timing, so the
    collision is deterministic. The gateway holds one provider instance for the whole process, so
    host and profile contexts that share it must not share a memo entry either.
    """

    _BASE = {"apiKey": "k", "peerName": "Igor", "aiPeer": "hermes"}

    @staticmethod
    def _stat_key(path):
        st = path.stat()
        return st.st_mtime_ns, st.st_size

    @staticmethod
    def _pin_mtime(path, mtime_ns):
        import os

        os.utime(path, ns=(mtime_ns, mtime_ns))

    @staticmethod
    def _provider(home, monkeypatch):
        from hermes_plugin_honcho import HonchoMemoryProvider

        monkeypatch.setenv("HERMES_HOME", str(home))
        return HonchoMemoryProvider()

    def _workspace_cfg(self, workspace, **extra):
        return json.dumps({**self._BASE, **extra, "hosts": {"hermes": {"workspace": workspace}}})

    def test_same_size_rewrite_with_identical_stat_returns_fresh_workspace(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        cfg_path.write_text(self._workspace_cfg("old"))
        before = self._stat_key(cfg_path)
        assert provider.identity_signature()["workspace"] == "old"

        cfg_path.write_text(self._workspace_cfg("new"))
        self._pin_mtime(cfg_path, before[0])
        assert self._stat_key(cfg_path) == before, "precondition: the rewrite must collide on (mtime_ns, size)"

        assert provider.identity_signature()["workspace"] == "new"

    def test_same_path_replacement_with_identical_stat_returns_fresh_workspace(self, tmp_path, monkeypatch):
        import os

        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        cfg_path.write_text(self._workspace_cfg("old"))
        before = self._stat_key(cfg_path)
        assert provider.identity_signature()["workspace"] == "old"

        staged = tmp_path / "honcho.json.staged"
        staged.write_text(self._workspace_cfg("new"))
        self._pin_mtime(staged, before[0])
        os.replace(staged, cfg_path)
        assert self._stat_key(cfg_path) == before, "precondition: the replacement must collide on (mtime_ns, size)"

        assert provider.identity_signature()["workspace"] == "new"

    def test_unchanged_content_returns_an_equal_signature_without_reparsing(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        parses = []
        original = HonchoClientConfig.from_global_config

        def counting(cls, *args, **kwargs):
            parses.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(HonchoClientConfig, "from_global_config", classmethod(counting))
        cfg_path.write_text(self._workspace_cfg("same"))
        first = provider.identity_signature()
        assert provider.identity_signature() == first
        assert provider.identity_signature() == first
        assert len(parses) == 1, "an unchanged file must be served from the memo"

        before = self._stat_key(cfg_path)
        cfg_path.write_text(self._workspace_cfg("same"))
        self._pin_mtime(cfg_path, before[0] + 1_000_000_000)
        assert provider.identity_signature() == first

        for workspace in ("w1", "w2", "w3"):
            cfg_path.write_text(self._workspace_cfg(workspace))
            self._pin_mtime(cfg_path, before[0])
            assert provider.identity_signature()["workspace"] == workspace
        assert len(provider._identity_signature_memo) <= 1, "the memo must stay bounded to one entry"

    def test_host_contexts_sharing_one_file_stay_isolated(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        cfg_path.write_text(json.dumps({**self._BASE, "hosts": {"hermes_a": {"workspace": "ws-a"},
                                                                "hermes_b": {"workspace": "ws-b"}}}))
        seen = []
        for host in ("hermes_a", "hermes_b", "hermes_a"):
            monkeypatch.setenv("HERMES_HONCHO_HOST", host)
            seen.append(provider.identity_signature()["workspace"])

        assert seen == ["ws-a", "ws-b", "ws-a"]

    def test_profiles_with_identical_stat_stay_isolated(self, tmp_path, monkeypatch):
        homes = [tmp_path / "alpha", tmp_path / "bravo"]
        for home in homes:
            home.mkdir()
            (home / "honcho.json").write_text(self._workspace_cfg(home.name))
        pinned = self._stat_key(homes[0] / "honcho.json")[0]
        for home in homes:
            self._pin_mtime(home / "honcho.json", pinned)
        assert self._stat_key(homes[0] / "honcho.json") == self._stat_key(homes[1] / "honcho.json")

        provider = self._provider(homes[0], monkeypatch)
        seen = []
        for home in (homes[0], homes[1], homes[0]):
            monkeypatch.setenv("HERMES_HOME", str(home))
            seen.append(provider.identity_signature()["workspace"])

        assert seen == ["alpha", "bravo", "alpha"]

    def test_missing_and_malformed_fall_back_then_a_valid_rewrite_with_identical_stat_is_seen(
            self, tmp_path, monkeypatch, caplog):
        import logging

        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        missing = provider.identity_signature()
        assert missing and missing["workspace"] == "hermes", "a missing file keeps the env fallback, not {}"

        valid = self._workspace_cfg("new")
        cfg_path.write_text(valid[:-1] + "!")
        before = self._stat_key(cfg_path)
        with caplog.at_level(logging.WARNING):
            malformed = provider.identity_signature()
            assert provider.identity_signature() == malformed
        assert malformed == missing, "unparseable JSON falls back to env exactly like a missing file"
        assert sum("Failed to read" in r.getMessage() for r in caplog.records) == 1, (
            "an unchanged malformed file must not re-warn on every message")

        cfg_path.write_text(valid)
        self._pin_mtime(cfg_path, before[0])
        assert self._stat_key(cfg_path) == before, "precondition: the fix must collide on (mtime_ns, size)"

        assert provider.identity_signature()["workspace"] == "new"

    def _race_parse(self, monkeypatch, cfg_path, pinned, *, restore):
        """Rewrite the file to "new" between the hashed snapshot and the parse, once; with ``restore``
        put the "old" bytes back before returning, so the bytes read after the parse match again."""
        original, raced = HonchoClientConfig.from_global_config, []

        def racing(cls, *args, **kwargs):
            if raced:
                return original(*args, **kwargs)
            raced.append(1)
            cfg_path.write_text(self._workspace_cfg("new"))
            self._pin_mtime(cfg_path, pinned)
            cfg = original(*args, **kwargs)
            if restore:
                cfg_path.write_text(self._workspace_cfg("old"))
                self._pin_mtime(cfg_path, pinned)
            return cfg

        monkeypatch.setattr(HonchoClientConfig, "from_global_config", classmethod(racing))

    @pytest.mark.parametrize("restore", [False, True], ids=["write-lands", "write-reverted"])
    def test_a_write_racing_the_parse_is_not_memoized_under_the_hashed_bytes(self, tmp_path, monkeypatch, restore):
        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        cfg_path.write_text(self._workspace_cfg("old"))
        pinned = cfg_path.stat().st_mtime_ns
        self._race_parse(monkeypatch, cfg_path, pinned, restore=restore)

        assert provider.identity_signature()["workspace"] == "new", "the parse saw the racing write"
        cfg_path.write_text(self._workspace_cfg("old"))
        self._pin_mtime(cfg_path, pinned)

        assert provider.identity_signature()["workspace"] == "old"

    def test_signature_and_logs_carry_no_secret_or_content_fingerprint(self, tmp_path, monkeypatch, caplog):
        import logging

        secret = "sk-honcho-SENTINEL-0123456789abcdef"
        cfg_path = tmp_path / "honcho.json"
        provider = self._provider(tmp_path, monkeypatch)
        caplog.set_level(logging.DEBUG)
        signatures, digests, pinned = [], [], None
        for workspace in ("old", "new"):
            cfg_path.write_text(self._workspace_cfg(workspace, apiKey=secret))
            pinned = pinned if pinned is not None else cfg_path.stat().st_mtime_ns
            self._pin_mtime(cfg_path, pinned)
            content = cfg_path.read_bytes()
            digests += [hashlib.sha256(content).hexdigest(), hashlib.blake2b(content).hexdigest()]
            signatures.append(provider.identity_signature())

        exposed = repr(signatures) + caplog.text
        assert secret not in exposed
        assert not any(d[:16] in exposed for d in digests)
        assert [s["workspace"] for s in signatures] == ["old", "new"]
