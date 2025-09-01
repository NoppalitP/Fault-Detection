@echo off

REM เปิดใช้งาน environment
call myenv\Scripts\activate.bat

REM ติดตั้ง LightGBM (จะติดตั้งใน env ที่เปิดแล้ว)
pip install lightgbm

REM รันโปรแกรมหลักโดยตรงจาก python.exe ของ env
myenv\python.exe -m app.main

pause
