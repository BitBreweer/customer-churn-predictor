@echo off

echo ======================================
echo Creating Virtual Environment...
echo ======================================

python -m venv venv

echo.
echo ======================================
echo Activating Environment...
echo ======================================

call venv\Scripts\activate

echo.
echo ======================================
echo Upgrading pip...
echo ======================================

python -m pip install --upgrade pip

echo.
echo ======================================
echo Installing Dependencies...
echo ======================================

pip install -r requirements.txt

echo.
echo ======================================
echo Environment Setup Complete!
echo ======================================
echo.

pause