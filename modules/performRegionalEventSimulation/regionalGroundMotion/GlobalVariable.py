"""Shared JVM state and helpers for the regional ground-motion module."""  # noqa: INP001

import os
import platform
import subprocess
import sys

# JPype allows only one JVM per Python process; entry-point scripts check
# this flag before calling jpype.startJVM().
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

    Honors ``JAVA_HOME`` first, then scans the standard install folders
    under ``Program Files``. When several JDKs are installed, the one with
    the highest-sorting directory name wins (so JDK 17 beats JDK 16).
    Returns None if nothing usable is found.
    """

    def _jvm_dll(jdk_home):
        candidate = os.path.join(jdk_home, 'bin', 'server', 'jvm.dll')  # noqa: PTH118
        return candidate if os.path.exists(candidate) else None  # noqa: PTH110

    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        dll = _jvm_dll(java_home)
        if dll:
            return dll

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

    # Modern Adoptium directory names embed the version
    # (e.g. "jdk-17.0.10.7-hotspot"), so reverse-sorted alphabetically
    # is also reverse-sorted by version.
    found.sort(reverse=True)
    return found[0][1]


def find_compatible_jvm_path():
    """Return a JVM library path matching the running Python interpreter.

    Guards against JPype's default lookup picking a JVM whose architecture
    does not match Python (common on macOS when both arm64 and x86_64 JDKs
    are installed) or a stale JAVA_HOME on Windows. Returns None on Linux
    and other platforms, and as a safe fallback when no usable JVM is
    found; callers should pass the result through to
    ``jpype.startJVM(jvmpath=...)`` unconditionally.
    """
    if sys.platform == 'darwin':
        return _find_jvm_macos()
    if sys.platform == 'win32':
        return _find_jvm_windows()
    return None
