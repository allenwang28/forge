# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch Jobs API integration for Forge resource provisioning.

This module provides a bridge between Forge's imperative resource allocation
model and Monarch's declarative Jobs API. It allows Forge to use MASTJob,
SlurmJob, or LocalJob for host provisioning while maintaining backward
compatibility with the existing Provisioner interface.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monarch.actor import HostMesh, ProcMesh
    from monarch._src.job.job import JobState, JobTrait

logger = logging.getLogger(__name__)


def _mount_mnt_directory(mount_dst: str) -> None:
    """Mounts the MAST remote directory to the specified destination.

    This function mounts a remote workspace directory that contains huggingface models
    and other shared resources needed for training.

    Args:
        mount_dst: Destination path where the directory should be mounted (e.g., "/mnt/wsfuse")
    """
    # Sanity check of the mounted directory
    sanity_path = os.path.join(mount_dst, "huggingface_models/")
    if os.path.exists(sanity_path):
        return

    # Otherwise, mount the directory
    if not os.path.exists(mount_dst):
        os.makedirs(mount_dst, exist_ok=True)

    # Store original LD_LIBRARY_PATH to restore after mounting
    original_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    try:
        clean_env = os.environ.copy()
        if "LD_LIBRARY_PATH" in clean_env:
            del clean_env["LD_LIBRARY_PATH"]

        subprocess.run(
            [
                "/packages/oil.oilfs/oilfs-wrapper",
                "ws://ws.ai.pci0ai/genai_fair_llm",
                mount_dst,
            ],
            capture_output=True,
            text=True,
            check=True,
            env=clean_env,
        )
        logger.info("Done mounting %s", mount_dst)
    except subprocess.CalledProcessError as e:
        logger.warning("Error during mounting %s: %s, Stderr: %s", mount_dst, e, e.stderr)
    except FileNotFoundError:
        # oilfs-wrapper not available (e.g., not on MAST)
        logger.debug("oilfs-wrapper not found, skipping mount")
    finally:
        # Restore original LD_LIBRARY_PATH
        if original_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = original_ld_library_path
        elif "LD_LIBRARY_PATH" in os.environ:
            del os.environ["LD_LIBRARY_PATH"]


class _MastSetupActor:
    """Actor for MAST-specific remote setup (mounting directories).

    This is lazily imported to avoid import errors when monarch.actor is not available.
    """

    _actor_class = None

    @classmethod
    def get_actor_class(cls):
        """Get the actual Actor class, creating it on first access."""
        if cls._actor_class is None:
            from monarch.actor import Actor, endpoint
            from monarch._src.actor.actor_mesh import current_rank

            class MastSetupActor(Actor):
                @endpoint
                def mount(self, mount_dst: str):
                    point = current_rank()
                    # The last dimension is the local proc count.
                    last_label = point.extent.labels[-1]
                    proc_count = point.size(last_label)
                    if current_rank().rank % proc_count != 0:
                        # Only use one rank per host to mount the directory
                        return
                    _mount_mnt_directory(mount_dst)

            cls._actor_class = MastSetupActor
        return cls._actor_class


class JobScheduler(Enum):
    """Supported job schedulers."""

    LOCAL = "local"
    SLURM = "slurm"
    MAST = "mast"


@dataclass
class MeshSpec:
    """Specification for a named mesh within a job.

    Args:
        name: Unique name for this mesh (e.g., "trainers", "evaluators")
        num_hosts: Number of hosts to allocate
        host_type: Host type for MAST (e.g., "gtt_any", "gpu.small")
        gpus_per_host: Number of GPUs per host (for resource requests)
    """

    name: str
    num_hosts: int
    host_type: str = "gtt_any"
    gpus_per_host: int | None = None


@dataclass
class SlurmJobConfig:
    """Configuration for SLURM jobs.

    Args:
        partition: SLURM partition to submit to
        time_limit: Maximum runtime in HH:MM:SS format
        python_exe: Python executable for worker processes
        monarch_port: Port for Monarch communication
        job_name: Name prefix for the SLURM job
        gpus_per_node: Number of GPUs per node
        exclusive: Whether to request exclusive node access
        slurm_args: Additional SLURM arguments
    """

    partition: str | None = None
    time_limit: str | None = None
    python_exe: str = "python"
    monarch_port: int = 22222
    job_name: str = "forge_job"
    gpus_per_node: int | None = 8
    exclusive: bool = True
    slurm_args: tuple[str, ...] = ()
    job_start_timeout: int | None = None


@dataclass
class MastJobConfig:
    """Configuration for MAST jobs.

    Args:
        hpc_identity: HPC identity for job submission
        hpc_job_oncall: Oncall for the job
        rm_attribution: Resource manager attribution
        hpc_cluster_uuid: Cluster UUID (default: MastProdCluster)
        packages: Python packages to include
        timeout_sec: Job timeout in seconds
        env: Environment variables for the job
        mount_dst: Destination path for MAST workspace mount (e.g., "/mnt/wsfuse")
            Set to None to disable mounting.
    """

    hpc_identity: str = ""
    hpc_job_oncall: str = ""
    rm_attribution: str = ""
    hpc_cluster_uuid: str = "MastProdCluster"
    packages: tuple[str, ...] = ()
    timeout_sec: int = 3600
    env: dict[str, str] = field(default_factory=dict)
    mount_dst: str | None = "/mnt/wsfuse"


@dataclass
class JobConfig:
    """Configuration for Forge job-based provisioning.

    Args:
        scheduler: Which job scheduler to use
        slurm: SLURM-specific configuration
        mast: MAST-specific configuration
    """

    scheduler: JobScheduler = JobScheduler.LOCAL
    slurm: SlurmJobConfig = field(default_factory=SlurmJobConfig)
    mast: MastJobConfig = field(default_factory=MastJobConfig)

    def __post_init__(self):
        if isinstance(self.scheduler, str):
            self.scheduler = JobScheduler(self.scheduler)


class ForgeJobBackend:
    """Manages a Monarch Job for Forge resource allocation.

    This class bridges Forge's imperative allocation model with Monarch's
    declarative Jobs API. It buffers mesh requests and applies them lazily
    when HostMeshes are first accessed.

    Example usage:
        backend = ForgeJobBackend(config)
        backend.request_mesh("trainers", num_hosts=4)
        backend.request_mesh("evaluators", num_hosts=2)
        backend.apply()

        trainers_mesh = backend.get_host_mesh("trainers")
        evaluators_mesh = backend.get_host_mesh("evaluators")
    """

    def __init__(self, config: JobConfig):
        self._config = config
        self._job: JobTrait | None = None
        self._state: JobState | None = None
        self._pending_meshes: dict[str, MeshSpec] = {}
        self._applied = False
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the job backend with scheduler-specific configuration.

        This should be called once before any other operations. It configures
        the transport layer based on the scheduler type:
        - MAST: Uses MetaTlsWithHostname transport
        - SLURM: Uses TcpWithHostname transport (handled by SlurmJob)
        - LOCAL: No special transport configuration needed
        """
        if self._initialized:
            return

        if self._config.scheduler == JobScheduler.MAST:
            # MAST requires explicit transport configuration
            try:
                from monarch._rust_bindings.monarch_hyperactor.channel import (
                    ChannelTransport,
                )
                from monarch._rust_bindings.monarch_hyperactor.config import configure

                configure(default_transport=ChannelTransport.MetaTlsWithHostname)
                logger.info("Configured MAST transport: MetaTlsWithHostname")
            except ImportError as e:
                logger.warning(
                    "Could not configure MAST transport (internal packages not available): %s",
                    e,
                )
        elif self._config.scheduler == JobScheduler.SLURM:
            # SlurmJob handles transport configuration internally
            logger.debug("SLURM transport configuration handled by SlurmJob")
        else:
            # LOCAL scheduler doesn't need special transport
            logger.debug("LOCAL scheduler, no transport configuration needed")

        self._initialized = True

    async def remote_setup(self, procs: "ProcMesh") -> None:
        """Perform scheduler-specific remote setup on a ProcMesh.

        This should be called after spawning procs to perform any
        scheduler-specific initialization:
        - MAST: Mounts the shared workspace directory (/mnt/wsfuse)
        - SLURM/LOCAL: No additional setup needed

        Args:
            procs: The ProcMesh to set up
        """
        if self._config.scheduler != JobScheduler.MAST:
            return

        mount_dst = self._config.mast.mount_dst
        if mount_dst is None:
            logger.debug("MAST mount disabled (mount_dst=None)")
            return

        logger.debug("Running MAST remote setup (mounting %s)", mount_dst)

        try:
            MastSetupActor = _MastSetupActor.get_actor_class()
            setup = procs.spawn("_mast_setup", MastSetupActor)
            await setup.mount.call(mount_dst)
            logger.info("MAST remote setup completed (mounted %s)", mount_dst)
        except Exception as e:
            logger.warning("MAST remote setup failed: %s", e)

    def _create_job(self) -> "JobTrait":
        """Create the appropriate Job instance based on configuration."""
        if self._config.scheduler == JobScheduler.LOCAL:
            from monarch._src.job.job import LocalJob

            # LocalJob takes a tuple of mesh names
            mesh_names = tuple(self._pending_meshes.keys())
            return LocalJob(hosts=mesh_names)

        elif self._config.scheduler == JobScheduler.SLURM:
            from monarch._src.job.slurm import SlurmJob

            slurm_cfg = self._config.slurm
            # SlurmJob takes a dict of mesh_name -> num_hosts
            meshes = {
                spec.name: spec.num_hosts for spec in self._pending_meshes.values()
            }
            return SlurmJob(
                meshes=meshes,
                python_exe=slurm_cfg.python_exe,
                slurm_args=slurm_cfg.slurm_args,
                monarch_port=slurm_cfg.monarch_port,
                job_name=slurm_cfg.job_name,
                time_limit=slurm_cfg.time_limit,
                partition=slurm_cfg.partition,
                exclusive=slurm_cfg.exclusive,
                gpus_per_node=slurm_cfg.gpus_per_node,
                job_start_timeout=slurm_cfg.job_start_timeout,
            )

        elif self._config.scheduler == JobScheduler.MAST:
            # Import MAST job from meta module
            try:
                from monarch._src.job.meta import MASTJob
            except ImportError as e:
                raise ImportError(
                    "MASTJob requires internal Meta packages. "
                    "Use 'slurm' or 'local' scheduler for open-source usage."
                ) from e

            mast_cfg = self._config.mast
            job = MASTJob(
                hpcIdentity=mast_cfg.hpc_identity,
                hpcJobOncall=mast_cfg.hpc_job_oncall,
                rmAttribution=mast_cfg.rm_attribution,
                hpcClusterUuid=mast_cfg.hpc_cluster_uuid,
                packages=mast_cfg.packages,
                timeout_sec=mast_cfg.timeout_sec,
                env=mast_cfg.env,
            )

            # Add meshes with host type
            for spec in self._pending_meshes.values():
                job.add_mesh(spec.name, spec.num_hosts, spec.host_type)

            return job

        else:
            raise ValueError(f"Unknown scheduler: {self._config.scheduler}")

    def request_mesh(
        self,
        name: str,
        num_hosts: int,
        host_type: str = "gtt_any",
        gpus_per_host: int | None = None,
    ) -> None:
        """Register a mesh requirement (before apply).

        Args:
            name: Unique name for this mesh
            num_hosts: Number of hosts to allocate
            host_type: Host type for MAST scheduling
            gpus_per_host: GPUs per host (for resource hints)

        Raises:
            RuntimeError: If apply() has already been called
        """
        if self._applied:
            raise RuntimeError(
                f"Cannot request mesh '{name}' after apply() has been called. "
                "Jobs API requires all meshes to be declared upfront."
            )

        if name in self._pending_meshes:
            logger.warning(
                f"Mesh '{name}' already requested, updating with new spec "
                f"(hosts: {num_hosts}, type: {host_type})"
            )

        self._pending_meshes[name] = MeshSpec(
            name=name,
            num_hosts=num_hosts,
            host_type=host_type,
            gpus_per_host=gpus_per_host,
        )
        logger.debug(f"Registered mesh request: {name} ({num_hosts} hosts)")

    def apply(self) -> None:
        """Submit job to scheduler.

        This creates the job with all registered mesh requirements and
        submits it to the scheduler. After this call, get_host_mesh()
        can be used to retrieve allocated HostMeshes.

        Raises:
            RuntimeError: If no meshes have been requested
        """
        if self._applied:
            logger.debug("Job already applied, skipping")
            return

        if not self._pending_meshes:
            raise RuntimeError("No meshes requested. Call request_mesh() before apply().")

        logger.info(
            f"Applying job with {len(self._pending_meshes)} mesh(es): "
            f"{list(self._pending_meshes.keys())}"
        )

        self._job = self._create_job()
        self._job.apply()
        self._state = self._job.state()
        self._applied = True

        logger.info("Job applied successfully")

    def get_host_mesh(self, name: str) -> "HostMesh":
        """Get a HostMesh by name after apply.

        Args:
            name: Name of the mesh (as registered with request_mesh)

        Returns:
            The allocated HostMesh

        Raises:
            RuntimeError: If apply() hasn't been called
            AttributeError: If the mesh name doesn't exist
        """
        if not self._applied or self._state is None:
            raise RuntimeError(
                f"Cannot get mesh '{name}' before apply(). "
                "Call apply() after registering all meshes."
            )

        if name not in self._pending_meshes:
            raise KeyError(
                f"Mesh '{name}' was not registered. "
                f"Available meshes: {list(self._pending_meshes.keys())}"
            )

        return getattr(self._state, name)

    def has_mesh(self, name: str) -> bool:
        """Check if a mesh with the given name has been requested."""
        return name in self._pending_meshes

    @property
    def is_applied(self) -> bool:
        """Whether the job has been applied."""
        return self._applied

    @property
    def mesh_names(self) -> list[str]:
        """List of all registered mesh names."""
        return list(self._pending_meshes.keys())

    def shutdown(self) -> None:
        """Kill the job and release all resources."""
        if self._job is not None:
            logger.info("Shutting down job backend...")
            try:
                self._job.kill()
                logger.info("Job killed successfully")
            except Exception as e:
                logger.warning(f"Error killing job: {e}")
            finally:
                self._job = None
                self._state = None
                self._applied = False
                self._pending_meshes.clear()
