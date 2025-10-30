@echo off
REM Скрипт сборки Geant4 Parser в Windows EXE

echo ========================================
echo Geant4 Log Parser - Сборка EXE
echo ========================================
echo.

REM Проверка установки PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ОШИБКА] PyInstaller не установлен!
    echo Установите: pip install pyinstaller
    pause
    exit /b 1
)

echo [1/4] Очистка старых сборок...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist geant4_parser.spec del geant4_parser.spec

echo [2/4] Создание спецификации...
pyinstaller --name=Geant4Parser ^
    --onefile ^
    --console ^
    --icon=NONE ^
    --add-data "requirements.txt;." ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=matplotlib ^
    --hidden-import=seaborn ^
    --hidden-import=openpyxl ^
    --collect-all=matplotlib ^
    --collect-all=seaborn ^
    geant4_parser_GUI.py

if errorlevel 1 (
    echo [ОШИБКА] Ошибка при создании спецификации!
    pause
    exit /b 1
)

echo [3/4] Сборка исполняемого файла...
REM Файл уже собран на предыдущем шаге

echo [4/4] Проверка результата...
if exist "dist\Geant4Parser.exe" (
    echo.
    echo ========================================
    echo СБОРКА УСПЕШНО ЗАВЕРШЕНА!
    echo ========================================
    echo.
    echo Исполняемый файл: dist\Geant4Parser.exe
    echo Размер:
    dir "dist\Geant4Parser.exe" | find "Geant4Parser.exe"
    echo.
    echo Использование:
    echo   Geant4Parser.exe -i your_log_file.log
    echo.
) else (
    echo [ОШИБКА] Не удалось создать EXE файл!
)

echo Очистка временных файлов...
if exist build rmdir /s /q build
if exist geant4_parser.spec del geant4_parser.spec

echo.
echo Готово!
pause
