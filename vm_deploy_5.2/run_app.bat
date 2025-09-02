@echo off

REM เปลี่ยนโฟลเดอร์ทำงานไปที่ C:\Saturn\Acoustic
cd /d C:\Saturn\Acoustic

REM เปิดใช้งาน environment
call myenv\Scripts\activate.bat

REM รันโปรแกรมหลักโดยตรงจาก python.exe ของ env
myenv\python.exe -m app.main

pause
