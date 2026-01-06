# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the ForgeJobBackend module."""

import pytest
from unittest.mock import MagicMock, patch

from forge.controller.job_backend import (
    ForgeJobBackend,
    JobConfig,
    JobScheduler,
    MastJobConfig,
    MeshSpec,
    SlurmJobConfig,
)


class TestJobConfig:
    """Test JobConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JobConfig()
        assert config.scheduler == JobScheduler.LOCAL
        assert isinstance(config.slurm, SlurmJobConfig)
        assert isinstance(config.mast, MastJobConfig)

    def test_scheduler_from_string(self):
        """Test scheduler can be set from string."""
        config = JobConfig(scheduler="slurm")
        assert config.scheduler == JobScheduler.SLURM

        config = JobConfig(scheduler="mast")
        assert config.scheduler == JobScheduler.MAST

        config = JobConfig(scheduler="local")
        assert config.scheduler == JobScheduler.LOCAL

    def test_slurm_config(self):
        """Test SLURM configuration."""
        slurm_cfg = SlurmJobConfig(
            partition="gpu",
            time_limit="24:00:00",
            gpus_per_node=4,
            job_name="test_job",
        )
        config = JobConfig(scheduler=JobScheduler.SLURM, slurm=slurm_cfg)
        assert config.slurm.partition == "gpu"
        assert config.slurm.time_limit == "24:00:00"
        assert config.slurm.gpus_per_node == 4
        assert config.slurm.job_name == "test_job"

    def test_mast_config(self):
        """Test MAST configuration."""
        mast_cfg = MastJobConfig(
            hpc_identity="test_identity",
            hpc_job_oncall="test_oncall",
            rm_attribution="test_attribution",
        )
        config = JobConfig(scheduler=JobScheduler.MAST, mast=mast_cfg)
        assert config.mast.hpc_identity == "test_identity"
        assert config.mast.hpc_job_oncall == "test_oncall"
        assert config.mast.rm_attribution == "test_attribution"


class TestMeshSpec:
    """Test MeshSpec dataclass."""

    def test_mesh_spec_creation(self):
        """Test MeshSpec creation with all fields."""
        spec = MeshSpec(
            name="trainers",
            num_hosts=4,
            host_type="gpu.large",
            gpus_per_host=8,
        )
        assert spec.name == "trainers"
        assert spec.num_hosts == 4
        assert spec.host_type == "gpu.large"
        assert spec.gpus_per_host == 8

    def test_mesh_spec_defaults(self):
        """Test MeshSpec default values."""
        spec = MeshSpec(name="workers", num_hosts=2)
        assert spec.host_type == "gtt_any"
        assert spec.gpus_per_host is None


class TestForgeJobBackend:
    """Test ForgeJobBackend class."""

    def test_init(self):
        """Test backend initialization."""
        config = JobConfig()
        backend = ForgeJobBackend(config)
        assert backend._config == config
        assert backend._job is None
        assert backend._state is None
        assert backend._pending_meshes == {}
        assert not backend._applied

    def test_request_mesh(self):
        """Test mesh request registration."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)
        assert "trainers" in backend._pending_meshes
        assert backend._pending_meshes["trainers"].num_hosts == 4

        backend.request_mesh("evaluators", num_hosts=2, host_type="gpu.small")
        assert "evaluators" in backend._pending_meshes
        assert backend._pending_meshes["evaluators"].host_type == "gpu.small"

    def test_request_mesh_updates_existing(self):
        """Test that requesting same mesh name updates the spec."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)
        backend.request_mesh("trainers", num_hosts=8)
        assert backend._pending_meshes["trainers"].num_hosts == 8

    def test_has_mesh(self):
        """Test has_mesh check."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        assert not backend.has_mesh("trainers")
        backend.request_mesh("trainers", num_hosts=4)
        assert backend.has_mesh("trainers")

    def test_mesh_names(self):
        """Test mesh_names property."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)
        backend.request_mesh("evaluators", num_hosts=2)

        assert set(backend.mesh_names) == {"trainers", "evaluators"}

    def test_is_applied_initially_false(self):
        """Test is_applied is initially false."""
        config = JobConfig()
        backend = ForgeJobBackend(config)
        assert not backend.is_applied

    def test_apply_without_meshes_raises(self):
        """Test apply raises if no meshes registered."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        with pytest.raises(RuntimeError, match="No meshes requested"):
            backend.apply()

    def test_request_mesh_after_apply_raises(self):
        """Test requesting mesh after apply raises error."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        # Mock the job creation and application
        with patch.object(backend, "_create_job") as mock_create:
            mock_job = MagicMock()
            mock_job.state.return_value = MagicMock()
            mock_create.return_value = mock_job

            backend.request_mesh("trainers", num_hosts=4)
            backend.apply()

            with pytest.raises(RuntimeError, match="Cannot request mesh.*after apply"):
                backend.request_mesh("new_mesh", num_hosts=2)

    def test_get_host_mesh_before_apply_raises(self):
        """Test get_host_mesh raises if not applied."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)

        with pytest.raises(RuntimeError, match="before apply"):
            backend.get_host_mesh("trainers")

    def test_get_host_mesh_unknown_name_raises(self):
        """Test get_host_mesh raises for unknown mesh name."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        # Mock apply
        with patch.object(backend, "_create_job") as mock_create:
            mock_job = MagicMock()
            mock_job.state.return_value = MagicMock()
            mock_create.return_value = mock_job

            backend.request_mesh("trainers", num_hosts=4)
            backend.apply()

            with pytest.raises(KeyError, match="unknown.*not registered"):
                backend.get_host_mesh("unknown")

    def test_apply_is_idempotent(self):
        """Test that multiple apply calls don't re-apply."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        with patch.object(backend, "_create_job") as mock_create:
            mock_job = MagicMock()
            mock_job.state.return_value = MagicMock()
            mock_create.return_value = mock_job

            backend.request_mesh("trainers", num_hosts=4)

            backend.apply()
            backend.apply()
            backend.apply()

            # Should only create job once
            mock_create.assert_called_once()

    def test_shutdown_clears_state(self):
        """Test shutdown clears all state."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        with patch.object(backend, "_create_job") as mock_create:
            mock_job = MagicMock()
            mock_job.state.return_value = MagicMock()
            mock_create.return_value = mock_job

            backend.request_mesh("trainers", num_hosts=4)
            backend.apply()

            backend.shutdown()

            assert backend._job is None
            assert backend._state is None
            assert not backend._applied
            assert len(backend._pending_meshes) == 0
            mock_job.kill.assert_called_once()

    def test_shutdown_without_job_is_safe(self):
        """Test shutdown without job doesn't raise."""
        config = JobConfig()
        backend = ForgeJobBackend(config)

        # Should not raise
        backend.shutdown()


class TestForgeJobBackendLocalJob:
    """Test ForgeJobBackend with LocalJob."""

    def test_create_local_job(self):
        """Test LocalJob creation."""
        config = JobConfig(scheduler=JobScheduler.LOCAL)
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)
        backend.request_mesh("evaluators", num_hosts=2)

        with patch("monarch._src.job.job.LocalJob") as mock_local_job:
            mock_job = MagicMock()
            mock_local_job.return_value = mock_job
            mock_job.state.return_value = MagicMock()

            backend.apply()

            # LocalJob should be created with mesh names as tuple
            mock_local_job.assert_called_once()
            call_kwargs = mock_local_job.call_args
            assert "trainers" in call_kwargs.kwargs["hosts"]
            assert "evaluators" in call_kwargs.kwargs["hosts"]


class TestForgeJobBackendSlurmJob:
    """Test ForgeJobBackend with SlurmJob."""

    def test_create_slurm_job(self):
        """Test SlurmJob creation."""
        slurm_cfg = SlurmJobConfig(
            partition="gpu",
            time_limit="24:00:00",
            gpus_per_node=8,
        )
        config = JobConfig(scheduler=JobScheduler.SLURM, slurm=slurm_cfg)
        backend = ForgeJobBackend(config)

        backend.request_mesh("trainers", num_hosts=4)

        with patch("monarch._src.job.slurm.SlurmJob") as mock_slurm_job:
            mock_job = MagicMock()
            mock_slurm_job.return_value = mock_job
            mock_job.state.return_value = MagicMock()

            backend.apply()

            mock_slurm_job.assert_called_once()
            call_kwargs = mock_slurm_job.call_args.kwargs
            assert call_kwargs["meshes"] == {"trainers": 4}
            assert call_kwargs["partition"] == "gpu"
            assert call_kwargs["time_limit"] == "24:00:00"
            assert call_kwargs["gpus_per_node"] == 8
