import serial
import time
import sys

# 설정
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SCAN_RANGE = range(1, 254) # 1~254번까지 스캔

# LX-16A 모든 읽기 명령 코드
CMDS = {
    "ID": 14,
    "각도 제한": 21,
    "전압 제한": 23,
    "온도 제한": 25,
    "현재 온도": 26,
    "현재 전압": 27,
    "현재 위치": 28,
    "구동 모드": 30,
    "토크 스위치": 32,
    "LED 상태": 34,
    "LED 알람": 36
}

class LX16AFullReader:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
    
    def _send_cmd(self, s_id, cmd):
        packet = [0x55, 0x55, s_id, 3, cmd]
        packet.append((~(sum(packet[2:]) & 0xff)) & 0xff)
        self.ser.reset_input_buffer()
        self.ser.write(bytearray(packet))
        time.sleep(0.015)

    def _recv(self):
        start = time.time()
        while (time.time() - start) < 0.1:
            if self.ser.in_waiting >= 6:
                header = self.ser.read(2)
                if header != b'\x55\x55': continue
                s_id, length, cmd = self.ser.read(3)
                data = self.ser.read(length - 3)
                self.ser.read(1) # Checksum
                return data
        return None

    def read_all(self, s_id):
        results = {}
        # 첫 번째 명령(위치)으로 서보 존재 확인
        self._send_cmd(s_id, CMDS["현재 위치"])
        first_res = self._recv()
        if first_res is None: return None
        
        results["현재 위치"] = first_res
        # 존재가 확인되면 나머지 정보 모두 읽기
        for name, code in CMDS.items():
            if name == "현재 위치": continue
            self._send_cmd(s_id, code)
            results[name] = self._recv()
        return results

    def display_info(self, s_id, data):
        print(f"\n{'#'*60}")
        print(f"### [SERVO ID {s_id}] 전체 데이터 리포트 ###")
        print(f"{'#'*60}")

        # 1. 실시간 데이터
        pos = data["현재 위치"][0] + (data["현재 위치"][1] << 8)
        if pos > 32767: pos -= 65536
        temp = data["현재 온도"][0] if data["현재 온도"] else "N/A"
        vin = (data["현재 전압"][0] + (data["현재 전압"][1] << 8)) / 1000.0 if data["현재 전압"] else 0
        
        print(f"\n[1. 실시간 상태]")
        print(f" - 위치(Position): {pos} ({pos*0.24:.2f}도)")
        print(f" - 온도(Temp)    : {temp}°C")
        print(f" - 전압(Voltage) : {vin:.2f}V")

        # 2. 구동 모드 및 토크
        mode_raw = data["구동 모드"]
        if mode_raw:
            mode = "모터 (회전)" if mode_raw[0] == 1 else "서보 (각도)"
            speed = mode_raw[2] + (mode_raw[3] << 8)
            if speed > 32767: speed -= 65536
            load = "ON (잠김)" if data["토크 스위치"] and data["토크 스위치"][0] == 1 else "OFF (풀림)"
            
            print(f"\n[2. 구동 설정]")
            print(f" - 작동 모드     : {mode}")
            if mode_raw[0] == 1: print(f" - 회전 속도     : {speed}")
            print(f" - 토크 상태     : {load}")

        # 3. 안전 제한 값
        ang = data["각도 제한"]
        v_lim = data["전압 제한"]
        t_lim = data["온도 제한"]
        
        print(f"\n[3. 안전 제한 설정]")
        if ang: print(f" - 각도 범위     : {ang[0]+(ang[1]<<8)} ~ {ang[2]+(ang[3]<<8)}")
        # 전압 제한: 2바이트씩 mV 단위로 해석 (Low, High, Low, High)
        if v_lim and len(v_lim) >= 4:
            v_min = (v_lim[0] + (v_lim[1] << 8)) / 1000.0
            v_max = (v_lim[2] + (v_lim[3] << 8)) / 1000.0
            print(f" - 전압 범위     : {v_min:.2f}V ~ {v_max:.2f}V")
        if t_lim: print(f" - 온도 임계치   : {t_lim[0]}°C")

        # 4. LED 설정
        led = "N/A"
        if data["LED 상태"]: led = "OFF" if data["LED 상태"][0] == 1 else "ON"
        alarm = bin(data["LED 알람"][0]) if data["LED 알람"] else "N/A"
        
        print(f"\n[4. 기타 설정]")
        print(f" - LED 기본상태  : {led}")
        print(f" - LED 알람비트  : {alarm}")
        print(f"\n{'='*60}")

    def close(self):
        self.ser.close()

def main():
    reader = LX16AFullReader(SERIAL_PORT, BAUD_RATE)
    print(f"\n[작업 시작] 총 {len(SCAN_RANGE)}개의 서보를 스캔합니다...")
    
    found_count = 0
    for i in SCAN_RANGE:
        # 현재 작업 상태 표시 (한 줄 업데이트)
        sys.stdout.write(f"\r>> 검색 중: ID {i:2} ... ")
        sys.stdout.flush()
        
        data = reader.read_all(i)
        
        if data:
            # 서보를 찾으면 진행 메시지를 지우고 리포트 출력
            sys.stdout.write("\r" + " " * 40 + "\r") 
            reader.display_info(i, data)
            found_count += 1
            
    # 모든 스캔 종료 후
    sys.stdout.write(f"\r스캔이 완료되었습니다. (발견된 서보: {found_count}개)\n")
    reader.close()

if __name__ == "__main__":
    main()
