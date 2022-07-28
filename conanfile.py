from conans import ConanFile, CMake, tools
import os

class simCenterBackendApps(ConanFile):
    name = "SimCenterBackendApplications"
    version = "1.2.2"
    description = "Backend applications for SimCenter software"
    license = "BSD 3-Clause"
    author = "Michael Gardner mhgardner@berkeley.edu"
    url = "https://github.com/NHERI-SimCenter/SimCenterBackendApplications"
    settings = {"os": None, "build_type": None, "compiler": None, "arch": ["x86_64"]}
    options = {"shared": [True, False]}
    default_options = {"mkl-static:threaded": False, "ipp-static:simcenter_backend": True, "libcurl:with_ssl":"openssl"}    
    generators = "cmake"
    build_policy = "missing"
    requires = "jansson/2.13.1", \
               "zlib/1.2.11", \
               "libcurl/7.72.0", \
               "eigen/3.3.7", \
               "clara/1.1.5", \
               "jsonformoderncpp/3.7.0", \
               "smelt/1.2.0@simcenter/stable", \
               "mkl-static/2019.4@simcenter/stable", \
               "ipp-static/2019.4@simcenter/stable", \
               "nanoflann/1.3.2", \
               "nlopt/2.6.2",\
                   
    # Custom attributes for Bincrafters recipe conventions
    _source_subfolder = "source_subfolder"
    _build_subfolder = "build_subfolder"
    # Set short paths for Windows
    short_paths = True    
    scm = {
        "type": "git",  # Use "type": "svn", if local repo is managed using SVN
        "subfolder": _source_subfolder,
        "url": "auto",
        "revision": "auto"
    }


    def configure(self):
        self.options.shared = False

        if self.settings.os == "Windows":
            self.options["libcurl"].with_winssl = True
            self.options["libcurl"].with_openssl = False


    def configure_cmake(self):
        cmake = CMake(self)
        cmake.configure(source_folder=self._source_subfolder)
        return cmake
    
    def build(self):
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        self.copy(pattern="LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self.configure_cmake()
        cmake.install()
        self.copy("*", dst="bin", src=self._source_subfolder + "/applications")

    def package_info(self):
        self.env_info.PATH.append(os.path.join(self.package_folder, "bin"))