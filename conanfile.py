import os  # noqa: D100

from conans import CMake, ConanFile


class simCenterBackendApps(ConanFile):  # noqa: D101
    name = 'SimCenterBackendApplications'
    version = '1.2.2'
    description = 'Backend applications for SimCenter software'
    license = 'BSD 3-Clause'
    author = 'Michael Gardner mhgardner@berkeley.edu'
    url = 'https://github.com/NHERI-SimCenter/SimCenterBackendApplications'
    settings = {  # noqa: RUF012
        'os': None,
        'build_type': None,
        'compiler': None,
        'arch': ['x86_64', 'armv8'],
    }
    options = {'shared': [True, False]}  # noqa: RUF012
    default_options = {  # noqa: RUF012
        'libcurl:with_ssl': 'openssl',
        'boost:without_fiber': True,
    }
    generators = 'cmake'
    build_policy = 'missing'
    requires = [  # noqa: RUF012
        'jansson/2.13.1',
        'zlib/1.3.1',
        'libcurl/8.4.0',
        'eigen/3.3.7',
        'clara/1.1.5',
        'jsonformoderncpp/3.7.0',
        'nanoflann/1.3.2',
        'nlopt/2.7.1',
        'kissfft/131.1.0',
        'boost/1.84.0'        
    ]


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
