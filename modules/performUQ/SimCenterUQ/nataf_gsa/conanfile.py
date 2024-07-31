from conans import ConanFile, CMake, tools
import os


class simCenterBackendApps(ConanFile):
    name = 'nataf_gsa_cpp_mpi'
    version = '1.0.0'
    description = 'Software for creating nataf_gsa'
    license = 'BSD 2-Clause'
    settings = {'os': None, 'build_type': None, 'compiler': None, 'arch': ['x86_64']}
    options = {'shared': [True, False]}
    default_options = {
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
    scm = {
        'type': 'git',  # Use "type": "svn", if local repo is managed using SVN
        'subfolder': _source_subfolder,
        'url': 'auto',
        'revision': 'auto',
    }

    def configure(self):
        self.options.shared = False

        if self.settings.os == 'Windows':
            self.options['libcurl'].with_winssl = True
            self.options['libcurl'].with_openssl = False

    def configure_cmake(self):
        cmake = CMake(self)
        cmake.configure(source_folder=self._source_subfolder)
        return cmake

    def build(self):
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        self.copy(pattern='LICENSE', dst='licenses', src=self._source_subfolder)
        cmake = self.configure_cmake()
        cmake.install()
        self.copy('*', dst='bin', src=self._source_subfolder + '/applications')

    def package_info(self):
        self.env_info.PATH.append(os.path.join(self.package_folder, 'bin'))
