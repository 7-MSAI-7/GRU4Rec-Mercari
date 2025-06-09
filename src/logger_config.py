# -*- coding: utf-8 -*-
"""
이 파일은 프로젝트의 로그(log) 기록 방식을 설정하는 역할을 합니다.
로그는 프로그램이 실행되면서 발생하는 다양한 이벤트(정보, 경고, 오류 등)를
시간 순서대로 기록한 것으로, 프로그램의 동작을 이해하고 문제를 진단하는 데 매우 중요합니다.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
import src.settings as config


def setup_logger():
    """
    프로젝트 전반에 걸쳐 사용할 로거(logger)를 설정합니다.
    이 함수를 호출하면, 프로그램의 실행 기록이 콘솔(화면)과 파일에 동시에 출력됩니다.
    - 파일: 'logs/training.log' 경로에 저장됩니다.
    - 콘솔: 터미널 화면에 직접 표시됩니다.
    """
    # 루트 로거를 가져옵니다. 로거는 로그 메시지를 보내는 객체입니다.
    logger = logging.getLogger()
    # 로거가 처리할 최소 로그 레벨을 INFO로 설정합니다.
    # (DEBUG, INFO, WARNING, ERROR, CRITICAL 순으로 심각도가 높아집니다.)
    logger.setLevel(logging.INFO)

    # 이전에 설정된 핸들러(로그를 어디로 보낼지 결정하는 객체)가 있다면,
    # 중복으로 로그가 찍히는 것을 방지하기 위해 모두 제거합니다.
    if logger.hasHandlers():
        logger.handlers.clear()

    # 설정 파일(config.py)에서 로그 디렉터리 경로를 가져와 전체 파일 경로를 만듭니다.
    log_file_path = os.path.join(config.LOG_FILE_DIR, "training.log")
    # 로그 파일이 저장될 디렉터리가 존재하지 않으면 새로 생성합니다.
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 로그 메시지의 형식을 정의합니다.
    # 예: "2023-10-27 10:00:00 - root - INFO - 로거가 설정되었습니다."
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 시간 - 로거이름 - 로그레벨 - 메시지
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- 파일 핸들러 설정 ---
    # 로그를 파일에 기록하는 핸들러입니다.
    # RotatingFileHandler는 파일 크기가 너무 커지는 것을 방지합니다.
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,  # 파일 크기가 5MB에 도달하면
        backupCount=5,             # 최대 5개의 백업 파일을 만들며 로그 파일을 교체합니다.
        encoding="utf-8"           # 한글이 깨지지 않도록 인코딩을 설정합니다.
    )
    file_handler.setFormatter(formatter)  # 위에서 정의한 포맷을 핸들러에 적용합니다.
    logger.addHandler(file_handler)       # 로거에 파일 핸들러를 추가합니다.

    # --- 콘솔 핸들러 설정 ---
    # 로그를 콘솔(터미널 화면)에 출력하는 핸들러입니다.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter) # 동일한 포맷을 적용합니다.
    logger.addHandler(console_handler)      # 로거에 콘솔 핸들러를 추가합니다.

    logging.info("로거가 성공적으로 설정되었습니다.")
