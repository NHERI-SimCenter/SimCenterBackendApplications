import os  # noqa: D100

from conan import ConanFile
from conan.tools.cmake import CMake
from conan.tools.files import copy


class simCenterBackendApps(ConanFile):  # noqa: D101
    name = "SimCenterBackendApplications"
    version = "1.2.2"
    description = "Backend applications for SimCenter software"
    license = "BSD 3-Clause"
    author = "Michael Gardner mhgardner@berkeley.edu"
    url = "https://github.com/NHERI-SimCenter/SimCenterBackendApplications"
    settings = "os", "compiler", "build_type", "arch"
    package_type = "application"
    short_paths = True

    default_options = {  # noqa: RUF012
        "libcurl/*:with_ssl": "openssl",
        "boost/*:without_fiber": True,
    }

    generators = "CMakeToolchain", "CMakeDeps"
    build_policy = "missing"

    requires = [  # noqa: RUF012
        "jansson/2.14",
        "zlib/1.3.1",
        "libcurl/8.12.1",
        "eigen/3.4.0",
        "clara/1.1.5",
        "nanoflann/1.6.0",
        "nlopt/2.10.0",
        "kissfft/131.1.0",
        "boost/1.88.0",
        "nlohmann_json/3.11.3",
    ]

    def build(self):  # noqa: D102
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):  # noqa: D102
        copy(self, "LICENSE", self.source_folder,
             os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()
        copy(self, "*", os.path.join(self.source_folder, "applications"),
             os.path.join(self.package_folder, "bin"))

    def package_info(self):  # noqa: D102
        self.cpp_info.bindirs = ["bin"]
