REM .bat file to build backend applications

REM Activate Python virtual environment
call c:\Users\fmcke\python_env\python3.12-conan\Scripts\activate
echo Python virtual environment activated.

REM Call Intel oneAPI setvars.bat (adjust the path as needed)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
echo Intel environment set.

echo Running Conan install...
conan install . --output-folder=build --build=missing
if errorlevel 1 (
    echo Conan install failed.
    exit /b 1
)

echo Building with CMake...
cmake --build build --config Release
if errorlevel 1 (
    echo CMake build failed.
    exit /b 1
)

echo Installing with CMake...
cmake --install build --config Release --prefix applications
if errorlevel 1 (
    echo CMake install failed.
    exit /b 1
)



