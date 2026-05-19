"""Shared JVM state and helpers for the regional ground-motion module."""  # noqa: INP001

import os
import platform
import subprocess
import sys

# JPype permits only one JVM per Python process. The entry-point scripts
# (FetchOpenSHA.py, HazardSimulation.py, HazardSimulationEQ.py,
# ScenarioForecast.py) check this flag before calling jpype.startJVM().
JVM_started = False


def _find_jvm_macos():
    """Locate an arch-matched JVM on macOS via /usr/libexec/java_home -a."""
    py_arch = platform.machine()
    if py_arch not in ('arm64', 'x86_64'):
        return None

    try:
        result = subprocess.run(  # noqa: S603
            ['/usr/libexec/java_home', '-a', py_arch],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    jvm_home = result.stdout.strip()
    if not jvm_home:
        return None

    libjli = os.path.join(jvm_home, 'lib', 'libjli.dylib')  # noqa: PTH118
    return libjli if os.path.exists(libjli) else None  # noqa: PTH110


def _find_jvm_windows():
    """Locate a JPype-compatible JVM on Windows.

    Honors ``JAVA_HOME`` first (which the Adoptium / Temurin installer sets
    automatically). Falls back to scanning the standard install folders
    under ``Program Files``; when several JDKs are installed, the one with
    the highest-sorting directory name wins (so JDK 17 beats JDK 16, and
    JDK 21 beats JDK 17). Returns None if nothing usable is found, letting
    JPype's default lookup take over.
    """

    def _jvm_dll(jdk_home):
        candidate = os.path.join(jdk_home, 'bin', 'server', 'jvm.dll')  # noqa: PTH118
        return candidate if os.path.exists(candidate) else None  # noqa: PTH110

    # 1. JAVA_HOME (set by every modern Windows JDK installer).
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        dll = _jvm_dll(java_home)
        if dll:
            return dll

    # 2. Scan the conventional install roots.
    is_64bit = sys.maxsize > 2**32
    program_files = os.environ.get(
        'ProgramFiles' if is_64bit else 'ProgramFiles(x86)',
        r'C:\Program Files',
    )

    found = []
    for parent in (
        os.path.join(program_files, 'Eclipse Adoptium'),  # noqa: PTH118
        os.path.join(program_files, 'Java'),  # noqa: PTH118
        os.path.join(program_files, 'Microsoft', 'jdk'),  # noqa: PTH118
    ):
        if not os.path.isdir(parent):  # noqa: PTH112
            continue
        try:
            entries = os.listdir(parent)
        except OSError:
            continue
        for entry in entries:
            dll = _jvm_dll(os.path.join(parent, entry))  # noqa: PTH118
            if dll:
                found.append((entry, dll))

    if not found:
        return None

    # Sort by directory name descending; modern Adoptium directory names
    # embed the version (e.g., "jdk-17.0.10.7-hotspot"), so this lands on
    # the newest JDK in nearly all cases.
    found.sort(reverse=True)
    return found[0][1]


def find_compatible_jvm_path():
    """Return a JVM library path matching the running Python interpreter.

    On macOS, multiple JDKs of different architectures can be installed
    side-by-side; JPype's default lookup may pick one whose architecture
    does not match Python, which then fails to load. This helper uses
    ``/usr/libexec/java_home -a <arch>`` to find a JDK whose architecture
    matches the current Python.

    On Windows, the equivalent risk is less severe (32-bit and 64-bit
    JDKs live in separate ``Program Files`` trees) but stale or
    misconfigured ``JAVA_HOME`` values, and multiple JDK versions
    coexisting, can still derail JPype's default lookup. This helper
    validates ``JAVA_HOME`` and, failing that, scans the standard install
    folders, preferring the highest version found.

    On Linux and other platforms, returns None and lets JPype's default
    lookup handle things. Returning None is also the safe fallback when
    the platform-specific lookup cannot find anything usable; the caller
    should pass the result through to ``jpype.startJVM(jvmpath=...)``
    unconditionally.
    """
    if sys.platform == 'darwin':
        return _find_jvm_macos()
    if sys.platform == 'win32':
        return _find_jvm_windows()
    return None
