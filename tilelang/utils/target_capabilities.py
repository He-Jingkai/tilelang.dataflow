"""Immutable target capability snapshots resolved from the CUDA driver.

This module deliberately does not depend on PyTorch.  Device selection and
hardware attributes come from the CUDA Driver API so compilation, codegen and
launch can share one target identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from tilelang.contrib import nvcc


class TargetCapabilityResolutionError(RuntimeError):
    """Raised when a complete target capability snapshot cannot be built."""


_ARCH_PATTERN = re.compile(r"(?:^|\s)-arch(?:=|\s+)(sm_[0-9]+[af]?)\b")
_KNOWN_TCGEN05_FAMILIES = frozenset((10, 11, 12))
_CUDA_TMA_MAX_NONINNERMOST_BOX_EXTENT = 256
_TARGET_CAPABILITY_SCHEMA_VERSION = 4
_LEGACY_TARGET_CAPABILITY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class TargetCapabilitySnapshot:
    """One immutable CUDA compilation and launch capability identity."""

    backend: str
    device_ordinal: int | None
    compute_capability: tuple[int, int]
    target: str
    arch: str
    supports_cluster_launch: bool
    max_cluster_size: int | None
    supports_tma: bool
    max_tma_noninnermost_box_extent: int | None
    supports_tensor_core_mma: bool
    supports_wgmma: bool
    supports_tcgen05: bool
    max_dynamic_shared_memory: int | None
    compiler_version: tuple[int, ...] | None
    driver_version: tuple[int, int] | None
    multiprocessor_count: int | None = None
    max_threads_per_block: int | None = None
    max_registers_per_block: int | None = None
    device_name: str | None = None
    resolution_source: str = "cuda-driver"
    schema_version: int = _TARGET_CAPABILITY_SCHEMA_VERSION
    canonical_json: str = ""
    fingerprint: str = ""
    compatibility_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version not in {
            _LEGACY_TARGET_CAPABILITY_SCHEMA_VERSION,
            _TARGET_CAPABILITY_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported target capability schema version {self.schema_version}")
        if self.backend != "cuda":
            raise ValueError(f"unsupported target capability backend {self.backend!r}")
        major, minor = self.compute_capability
        if major <= 0 or minor < 0:
            raise ValueError(
                f"compute_capability must contain positive major and non-negative minor values, got {self.compute_capability!r}"
            )
        if self.supports_tma:
            if self.max_tma_noninnermost_box_extent is None or self.max_tma_noninnermost_box_extent <= 0:
                raise ValueError("TMA-capable targets require a positive non-innermost box extent limit")
        elif self.max_tma_noninnermost_box_extent is not None:
            raise ValueError("non-TMA targets cannot declare a TMA box extent limit")
        for name in (
            "multiprocessor_count",
            "max_threads_per_block",
            "max_registers_per_block",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided, got {value}")
        expected_arch = compute_capability_from_arch(self.arch)
        if expected_arch != self.compute_capability:
            raise ValueError(f"target arch {self.arch!r} identifies compute capability {expected_arch!r}, not {self.compute_capability!r}")
        payload = self.identity_payload()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        compatibility_json = json.dumps(
            self.compatibility_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "canonical_json", canonical_json)
        object.__setattr__(self, "fingerprint", sha256(canonical_json))
        object.__setattr__(
            self,
            "compatibility_fingerprint",
            sha256(compatibility_json),
        )

    @classmethod
    def for_cuda(
        cls,
        compute_capability: tuple[int, int],
        *,
        arch: str | None = None,
        target: str | None = None,
        device_ordinal: int | None = None,
        supports_cluster_launch: bool | None = None,
        supports_tma: bool | None = None,
        supports_tensor_core_mma: bool | None = None,
        supports_wgmma: bool | None = None,
        supports_tcgen05: bool | None = None,
        max_cluster_size: int | None = None,
        max_dynamic_shared_memory: int | None = None,
        multiprocessor_count: int | None = None,
        max_threads_per_block: int | None = 1024,
        max_registers_per_block: int | None = 65536,
        compiler_version: tuple[int, ...] | None = None,
        driver_version: tuple[int, int] | None = None,
        device_name: str | None = None,
        resolution_source: str = "override",
    ) -> TargetCapabilitySnapshot:
        """Build an explicit snapshot for cross compilation and tests."""

        compute_capability = (int(compute_capability[0]), int(compute_capability[1]))
        resolved_arch = arch or cuda_arch_for_compute_capability(compute_capability)
        resolved_target = target or f"cuda -arch={resolved_arch}"
        policy = cuda_capability_policy(compute_capability)
        resolved_supports_tma = policy["supports_tma"] if supports_tma is None else bool(supports_tma)
        return cls(
            backend="cuda",
            device_ordinal=None if device_ordinal is None else int(device_ordinal),
            compute_capability=compute_capability,
            target=str(resolved_target),
            arch=str(resolved_arch),
            supports_cluster_launch=(
                policy["supports_cluster_launch"] if supports_cluster_launch is None else bool(supports_cluster_launch)
            ),
            max_cluster_size=(None if max_cluster_size is None else int(max_cluster_size)),
            supports_tma=resolved_supports_tma,
            max_tma_noninnermost_box_extent=(_CUDA_TMA_MAX_NONINNERMOST_BOX_EXTENT if resolved_supports_tma else None),
            supports_tensor_core_mma=(
                policy["supports_tensor_core_mma"] if supports_tensor_core_mma is None else bool(supports_tensor_core_mma)
            ),
            supports_wgmma=(policy["supports_wgmma"] if supports_wgmma is None else bool(supports_wgmma)),
            supports_tcgen05=(policy["supports_tcgen05"] if supports_tcgen05 is None else bool(supports_tcgen05)),
            max_dynamic_shared_memory=(None if max_dynamic_shared_memory is None else int(max_dynamic_shared_memory)),
            multiprocessor_count=(None if multiprocessor_count is None else int(multiprocessor_count)),
            max_threads_per_block=(None if max_threads_per_block is None else int(max_threads_per_block)),
            max_registers_per_block=(None if max_registers_per_block is None else int(max_registers_per_block)),
            compiler_version=(None if compiler_version is None else tuple(int(item) for item in compiler_version)),
            driver_version=(None if driver_version is None else tuple(int(item) for item in driver_version)),
            device_name=device_name,
            resolution_source=str(resolution_source),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TargetCapabilitySnapshot:
        """Restore a recorded capability identity and verify its fingerprints."""

        if not isinstance(value, dict):
            raise TypeError("target capability snapshot must be a dictionary")
        snapshot = cls(
            backend=str(value["backend"]),
            device_ordinal=(None if value.get("device_ordinal") is None else int(value["device_ordinal"])),
            compute_capability=tuple(int(item) for item in value["compute_capability"]),
            target=str(value["target"]),
            arch=str(value["arch"]),
            supports_cluster_launch=bool(value["supports_cluster_launch"]),
            max_cluster_size=(None if value.get("max_cluster_size") is None else int(value["max_cluster_size"])),
            supports_tma=bool(value["supports_tma"]),
            max_tma_noninnermost_box_extent=(
                None if value.get("max_tma_noninnermost_box_extent") is None else int(value["max_tma_noninnermost_box_extent"])
            ),
            supports_tensor_core_mma=bool(value["supports_tensor_core_mma"]),
            supports_wgmma=bool(value["supports_wgmma"]),
            supports_tcgen05=bool(value["supports_tcgen05"]),
            max_dynamic_shared_memory=(None if value.get("max_dynamic_shared_memory") is None else int(value["max_dynamic_shared_memory"])),
            multiprocessor_count=(None if value.get("multiprocessor_count") is None else int(value["multiprocessor_count"])),
            max_threads_per_block=(None if value.get("max_threads_per_block") is None else int(value["max_threads_per_block"])),
            max_registers_per_block=(None if value.get("max_registers_per_block") is None else int(value["max_registers_per_block"])),
            compiler_version=(None if value.get("compiler_version") is None else tuple(int(item) for item in value["compiler_version"])),
            driver_version=(None if value.get("driver_version") is None else tuple(int(item) for item in value["driver_version"])),
            device_name=value.get("device_name"),
            resolution_source=str(value.get("resolution_source", "recorded")),
            schema_version=int(
                value.get(
                    "schema_version",
                    _LEGACY_TARGET_CAPABILITY_SCHEMA_VERSION,
                )
            ),
        )
        for name in ("fingerprint", "compatibility_fingerprint"):
            recorded = value.get(name)
            if recorded is not None and recorded != getattr(snapshot, name):
                raise ValueError(f"target capability {name} does not match its payload")
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            **self.identity_payload(),
        }

    def compatibility_mismatches(
        self,
        actual: TargetCapabilitySnapshot,
    ) -> tuple[str, ...]:
        """Return launch-relevant differences between an artifact and a device."""

        mismatches: list[str] = []
        if self.backend != actual.backend:
            mismatches.append(f"backend artifact={self.backend} device={actual.backend}")
        if self.compute_capability != actual.compute_capability:
            mismatches.append(f"compute_capability artifact={self.compute_capability} device={actual.compute_capability}")
        required_capabilities = (
            "supports_cluster_launch",
            "supports_tma",
            "supports_tensor_core_mma",
            "supports_wgmma",
            "supports_tcgen05",
        )
        for name in required_capabilities:
            if bool(getattr(self, name)) and not bool(getattr(actual, name)):
                mismatches.append(f"{name} required by artifact but unavailable on device")
        return tuple(mismatches)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "device_ordinal": self.device_ordinal,
            "compute_capability": list(self.compute_capability),
            "target": self.target,
            "arch": self.arch,
            "supports_cluster_launch": self.supports_cluster_launch,
            "max_cluster_size": self.max_cluster_size,
            "supports_tma": self.supports_tma,
            "max_tma_noninnermost_box_extent": (self.max_tma_noninnermost_box_extent),
            "supports_tensor_core_mma": self.supports_tensor_core_mma,
            "supports_wgmma": self.supports_wgmma,
            "supports_tcgen05": self.supports_tcgen05,
            "max_dynamic_shared_memory": self.max_dynamic_shared_memory,
            "compiler_version": (None if self.compiler_version is None else list(self.compiler_version)),
            "driver_version": (None if self.driver_version is None else list(self.driver_version)),
            "device_name": self.device_name,
            "resolution_source": self.resolution_source,
        }
        if self.schema_version >= _TARGET_CAPABILITY_SCHEMA_VERSION:
            payload.update(
                {
                    "multiprocessor_count": self.multiprocessor_count,
                    "max_threads_per_block": self.max_threads_per_block,
                    "max_registers_per_block": self.max_registers_per_block,
                }
            )
        return payload

    def compatibility_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "compute_capability": list(self.compute_capability),
            "supports_cluster_launch": self.supports_cluster_launch,
            "supports_tma": self.supports_tma,
            "max_tma_noninnermost_box_extent": (self.max_tma_noninnermost_box_extent),
            "supports_tensor_core_mma": self.supports_tensor_core_mma,
            "supports_wgmma": self.supports_wgmma,
            "supports_tcgen05": self.supports_tcgen05,
        }


def resolve_target_capabilities(
    device_ordinal: int | None = None,
) -> TargetCapabilitySnapshot:
    """Resolve the selected CUDA device without creating a CUDA context."""

    try:
        from cuda.bindings import driver
    except Exception as err:
        raise TargetCapabilityResolutionError(f"CUDA driver bindings are required to resolve target capabilities: {err}") from err

    try:
        result = driver.cuInit(0)[0]
        check_cuda_result(driver, result, "cuInit")
        selected_ordinal, device = select_cuda_device(driver, device_ordinal)
        compiler_version = query_cuda_compiler_version()
        return resolve_target_capabilities_from_device(
            driver,
            device,
            device_ordinal=selected_ordinal,
            compiler_version=compiler_version,
        )
    except TargetCapabilityResolutionError:
        raise
    except Exception as err:
        requested = "current context or device 0" if device_ordinal is None else f"device {device_ordinal}"
        raise TargetCapabilityResolutionError(f"failed to resolve CUDA target capabilities for {requested}: {err}") from err


def resolve_target_capabilities_from_device(
    driver: Any,
    device: Any,
    *,
    device_ordinal: int,
    compiler_version: tuple[int, ...] | None,
) -> TargetCapabilitySnapshot:
    """Resolve a snapshot from an already selected CUDA driver device."""

    major = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
        required=True,
    )
    minor = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
        required=True,
    )
    compute_capability = (major, minor)
    arch = cuda_arch_for_compute_capability(compute_capability)
    cluster_launch = bool(
        query_device_attribute(
            driver,
            device,
            "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
            required=False,
            default=0,
        )
    )
    max_dynamic_shared_memory = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        required=False,
        default=None,
    )
    multiprocessor_count = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
        required=False,
        default=None,
    )
    max_threads_per_block = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
        required=False,
        default=None,
    )
    max_registers_per_block = query_device_attribute(
        driver,
        device,
        "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
        required=False,
        default=None,
    )
    supports_tma = bool(
        query_device_attribute(
            driver,
            device,
            "CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED",
            required=False,
            default=0,
        )
    )
    policy = cuda_capability_policy(compute_capability)
    return TargetCapabilitySnapshot(
        backend="cuda",
        device_ordinal=int(device_ordinal),
        compute_capability=compute_capability,
        target=f"cuda -arch={arch}",
        arch=arch,
        supports_cluster_launch=cluster_launch,
        max_cluster_size=None,
        supports_tma=supports_tma,
        max_tma_noninnermost_box_extent=(_CUDA_TMA_MAX_NONINNERMOST_BOX_EXTENT if supports_tma else None),
        supports_tensor_core_mma=policy["supports_tensor_core_mma"],
        supports_wgmma=policy["supports_wgmma"],
        supports_tcgen05=policy["supports_tcgen05"],
        max_dynamic_shared_memory=max_dynamic_shared_memory,
        multiprocessor_count=multiprocessor_count,
        max_threads_per_block=max_threads_per_block,
        max_registers_per_block=max_registers_per_block,
        compiler_version=compiler_version,
        driver_version=query_driver_version(driver),
        device_name=query_device_name(driver, device),
        resolution_source="cuda-driver",
    )


def target_capability_override(
    target: Any,
    *,
    arch: str | None = None,
    compiler_version: tuple[int, ...] | None = None,
) -> TargetCapabilitySnapshot:
    """Normalize an explicit CUDA target used for cross compilation."""

    target_text = str(target).strip()
    target_arch = arch or resolve_target_arch(target, target_text)
    if not target_arch:
        raise TargetCapabilityResolutionError("cross-compile target override must include an explicit CUDA arch")
    if not target_arch.startswith("sm_"):
        raise TargetCapabilityResolutionError(f"unsupported CUDA target override arch {target_arch!r}; expected sm_<version>")
    if not target_text or target_text == "cuda":
        target_text = f"cuda -arch={target_arch}"
    elif _ARCH_PATTERN.search(target_text) is None:
        target_text = f"{target_text} -arch={target_arch}"
    return TargetCapabilitySnapshot.for_cuda(
        compute_capability_from_arch(target_arch),
        arch=target_arch,
        target=target_text,
        compiler_version=compiler_version or try_query_cuda_compiler_version(),
        resolution_source="override",
    )


def select_cuda_device(driver: Any, device_ordinal: int | None) -> tuple[int, Any]:
    result, count = driver.cuDeviceGetCount()
    check_cuda_result(driver, result, "cuDeviceGetCount")
    if count <= 0:
        raise TargetCapabilityResolutionError("CUDA driver reports no devices")

    if device_ordinal is None:
        selected = current_context_device(driver)
        if selected is not None:
            return selected
        device_ordinal = 0
    device_ordinal = int(device_ordinal)
    if device_ordinal < 0 or device_ordinal >= count:
        raise TargetCapabilityResolutionError(f"CUDA device ordinal {device_ordinal} is outside [0, {count})")
    result, device = driver.cuDeviceGet(device_ordinal)
    check_cuda_result(driver, result, f"cuDeviceGet({device_ordinal})")
    return device_ordinal, device


def current_context_device(driver: Any) -> tuple[int, Any] | None:
    get_current = getattr(driver, "cuCtxGetCurrent", None)
    get_device = getattr(driver, "cuCtxGetDevice", None)
    if get_current is None or get_device is None:
        return None
    result, context = get_current()
    check_cuda_result(driver, result, "cuCtxGetCurrent")
    if int(context) == 0:
        return None
    result, device = get_device()
    check_cuda_result(driver, result, "cuCtxGetDevice")
    return int(device), device


def query_device_attribute(
    driver: Any,
    device: Any,
    name: str,
    *,
    required: bool,
    default: int | None = None,
) -> int | None:
    attribute = getattr(driver.CUdevice_attribute, name, None)
    if attribute is None:
        if required:
            raise TargetCapabilityResolutionError(f"CUDA driver bindings do not expose required attribute {name}")
        return default
    result, value = driver.cuDeviceGetAttribute(attribute, device)
    if result != driver.CUresult.CUDA_SUCCESS:
        if required:
            check_cuda_result(driver, result, f"cuDeviceGetAttribute({name})")
        return default
    return int(value)


def query_driver_version(driver: Any) -> tuple[int, int] | None:
    try:
        result, version = driver.cuDriverGetVersion()
        check_cuda_result(driver, result, "cuDriverGetVersion")
        version = int(version)
        return version // 1000, (version % 1000) // 10
    except Exception:
        return None


def query_device_name(driver: Any, device: Any) -> str | None:
    try:
        result, name = driver.cuDeviceGetName(256, device)
        check_cuda_result(driver, result, "cuDeviceGetName")
        if isinstance(name, bytes):
            return name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return str(name)
    except Exception:
        return None


def query_cuda_compiler_version() -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in nvcc.get_cuda_version())
    except Exception as err:
        raise TargetCapabilityResolutionError(f"failed to query CUDA compiler version from NVCC: {err}") from err


def try_query_cuda_compiler_version() -> tuple[int, ...] | None:
    try:
        return query_cuda_compiler_version()
    except TargetCapabilityResolutionError:
        return None


def resolve_target_arch(target: Any, target_text: str) -> str | None:
    arch = getattr(target, "arch", None)
    if arch:
        return str(arch)
    attrs = getattr(target, "attrs", None)
    if attrs is not None and attrs.get("arch"):
        return str(attrs["arch"])
    match = _ARCH_PATTERN.search(target_text)
    return None if match is None else match.group(1)


def compute_capability_from_arch(arch: str) -> tuple[int, int]:
    normalized = str(arch).strip().lower()
    if not normalized.startswith("sm_"):
        raise ValueError(f"CUDA arch must start with 'sm_', got {arch!r}")
    digits = normalized[3:].rstrip("af")
    if not digits.isdigit() or len(digits) not in (2, 3):
        raise ValueError(f"unsupported CUDA arch {arch!r}")
    numeric = int(digits)
    return numeric // 10, numeric % 10


def cuda_arch_for_compute_capability(compute_capability: tuple[int, int]) -> str:
    major, minor = compute_capability
    suffix = "a" if major in {9, 10, 11, 12} else ""
    return f"sm_{major * 10 + minor}{suffix}"


def cuda_capability_policy(compute_capability: tuple[int, int]) -> dict[str, bool]:
    major, minor = compute_capability
    return {
        # Cross-compilation has no device attributes to query.  These defaults
        # model documented ISA families and can be overridden explicitly by
        # TargetCapabilitySnapshot.for_cuda().  Runtime snapshots query the
        # CUDA Driver API for cluster and tensor-map support instead.
        "supports_cluster_launch": major >= 9,
        "supports_tma": major >= 9,
        # Warp-level MMA is a separate fallback capability from Hopper WGMMA
        # and Blackwell TCGEN05.  Lowering consumers use this feature bit and
        # never reinterpret the architecture string themselves.
        "supports_tensor_core_mma": major >= 7,
        # WGMMA is the Hopper SM90a instruction family, not a generic sm>=90 feature.
        "supports_wgmma": (major, minor) == (9, 0),
        # TCGEN05 is a Blackwell family capability.  Backend implementation
        # selection remains the responsibility of the lowering registry.
        "supports_tcgen05": major in _KNOWN_TCGEN05_FAMILIES,
    }


def check_cuda_result(driver: Any, result: Any, action: str) -> None:
    if result == driver.CUresult.CUDA_SUCCESS:
        return
    raise TargetCapabilityResolutionError(f"{action} failed: {result}")


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
