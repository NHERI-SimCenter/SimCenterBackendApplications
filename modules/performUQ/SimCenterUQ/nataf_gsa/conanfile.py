import os  # noqa: CPY001, D100, INP001

from conans import CMake, ConanFile


class simCenterBackendApps(ConanFile):  # noqa: D101
    name = 'nataf_gsa_cpp_mpi'
    version = '1.0.0'
    description = 'Software for creating nataf_gsa'
    license = 'BSD 2-Clause'
    settings = {'os': None, 'build_type': None, 'compiler': None, 'arch': ['x86_64']}  # noqa: RUF012
    options = {'shared': [True, False]}  # noqa: RUF012
    default_options = {  # noqa: RUF012
        'mkl-static:threaded': False,
        'ipp-static:simcenter_backend': True,
    }
    generators = 'cmake'
    build_policy = 'missing'
    requires = (
        'eigen/3.3.7',
        'jsonformoderncpp/3.7.0',
        'mkl-static/2019.4@simcenter/stable',
        'boost/1.74.0',
        'nlopt/2.6.2',
    )
    # Custom attributes for Bincrafters recipe conventions
    _source_subfolder = 'source_subfolder'
    _build_subfolder = 'build_subfolder'
    # Set short paths for Windows
    short_paths = True
    scm = {  # noqa: RUF012
        'type': 'git',  # Use "type": "svn", if local repo is managed using SVN
        'subfolder': _source_subfolder,
        'url': 'auto',
        'revision': 'auto',
    }

    def configure(self):  # noqa: D102
        self.options.shared = False

        if self.settings.os == 'Windows':
            self.options['libcurl'].with_winssl = True
            self.options['libcurl'].with_openssl = False

    def configure_cmake(self):  # noqa: D102
        cmake = CMake(self)
        cmake.configure(source_folder=self._source_subfolder)
        return cmake

    def build(self):  # noqa: D102
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):  # noqa: D102
        self.copy(pattern='LICENSE', dst='licenses', src=self._source_subfolder)
        cmake = self.configure_cmake()
        cmake.install()
        self.copy('*', dst='bin', src=self._source_subfolder + '/applications')

    def package_info(self):  # noqa: D102
        self.env_info.PATH.append(os.path.join(self.package_folder, 'bin'))  # noqa: PTH118
