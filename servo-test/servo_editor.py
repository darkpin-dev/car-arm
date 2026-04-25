import serial
import time
import sys

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200

# 영구 저장 관련 명령 코드
CMD_ID_WRITE = 13
CMD_ANGLE_OFFSET_WRITE = 17
CMD_ANGLE_OFFSET_READ = 18
CMD_ANGLE_LIMIT_WRITE = 20
CMD_ANGLE_LIMIT_READ = 21
CMD_VIN_LIMIT_WRITE = 22
CMD_VIN_LIMIT_READ = 23
CMD_TEMP_LIMIT_WRITE = 24
CMD_TEMP_LIMIT_READ = 25
CMD_SERVO_MODE_WRITE = 29
CMD_SERVO_MODE_READ = 30
CMD_LED_CTRL_WRITE = 33
CMD_LED_CTRL_READ = 34
CMD_LED_ERROR_WRITE = 35
CMD_LED_ERROR_READ = 36

class LX16AEditor:
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
        except Exception as e:
            print(f"포트 열기 실패: {e}")
            sys.exit(1)

    def _send_cmd(self, s_id, cmd, params=[]):
        length = len(params) + 3
        packet = [0x55, 0x55, s_id, length, cmd] + params
        packet.append((~(sum(packet[2:]) & 0xff)) & 0xff)
        self.ser.reset_input_buffer()
        self.ser.write(bytearray(packet))
        time.sleep(0.05)

    def _read_res(self):
        start = time.time()
        while (time.time() - start) < 0.2:
            if self.ser.in_waiting >= 6:
                header = self.ser.read(2)
                if header != b'\x55\x55': continue
                s_id, length, cmd = self.ser.read(3)
                data = list(self.ser.read(length - 3))
                self.ser.read(1)
                return data
        return None

    def get_input(self, prompt, current_val):
        val = input(f"{prompt} [현재: {current_val}]: ").strip()
        return val if val else None

    def start_edit(self, s_id):
        print(f"\n--- 서보 ID {s_id} 설정 편집 ---")
        
        # 1. ID 변경
        val = self.get_input("새 서보 ID", s_id)
        if val is not None:
            new_id = int(val)
            self._send_cmd(s_id, CMD_ID_WRITE, [new_id])
            s_id = new_id

        # 2. 각도 편차
        self._send_cmd(s_id, CMD_ANGLE_OFFSET_READ)
        res = self._read_res()
        if res:
            offset = res[0]
            if offset > 127: offset -= 256
            val = self.get_input("각도 편차", offset)
            if val is not None:
                n = int(val)
                self._send_cmd(s_id, CMD_ANGLE_OFFSET_WRITE, [n if n >= 0 else n + 256])

        # 3. 각도 제한
        self._send_cmd(s_id, CMD_ANGLE_LIMIT_READ)
        res = self._read_res()
        if res:
            curr_min, curr_max = res[0]+(res[1]<<8), res[2]+(res[3]<<8)
            v_min = self.get_input("최소 각도", curr_min)
            v_max = self.get_input("최대 각도", curr_max)
            if v_min or v_max:
                n_min, n_max = int(v_min or curr_min), int(v_max or curr_max)
                self._send_cmd(s_id, CMD_ANGLE_LIMIT_WRITE, [n_min&0xff, n_min>>8, n_max&0xff, n_max>>8])

        # 4. 구동 모드
        self._send_cmd(s_id, CMD_SERVO_MODE_READ)
        res = self._read_res()
        if res:
            val = self.get_input("모드 (0:서보, 1:모터)", "모터" if res[0] == 1 else "서보")
            if val is not None:
                n_mode, n_speed = int(val), 0
                if n_mode == 1: n_speed = int(input("회전 속도: ") or 0)
                speed_p = [n_speed&0xff, n_speed>>8] if n_speed >= 0 else [(n_speed+65536)&0xff, (n_speed+65536)>>8]
                self._send_cmd(s_id, CMD_SERVO_MODE_WRITE, [n_mode, 0] + speed_p)

        # 5. 전압 제한
        self._send_cmd(s_id, CMD_VIN_LIMIT_READ)
        res = self._read_res()
        if res and len(res) >= 4:
            v_min_curr = res[0] + (res[1] << 8)
            v_max_curr = res[2] + (res[3] << 8)
            v_min = self.get_input("최소 전압(mV)", v_min_curr)
            v_max = self.get_input("최대 전압(mV)", v_max_curr)
            if v_min or v_max:
                n_min, n_max = int(v_min or v_min_curr), int(v_max or v_max_curr)
                self._send_cmd(s_id, CMD_VIN_LIMIT_WRITE, [n_min&0xff, n_min>>8, n_max&0xff, n_max>>8])

        # 6. 온도 제한
        self._send_cmd(s_id, CMD_TEMP_LIMIT_READ)
        res = self._read_res()
        if res:
            val = self.get_input("최대 온도 제한", res[0])
            if val is not None: self._send_cmd(s_id, CMD_TEMP_LIMIT_WRITE, [int(val)])

        # 7. LED 기본 상태
        self._send_cmd(s_id, CMD_LED_CTRL_READ)
        res = self._read_res()
        if res:
            val = self.get_input("LED 기본 상태 (0:ON, 1:OFF)", "OFF" if res[0] == 1 else "ON")
            if val is not None: self._send_cmd(s_id, CMD_LED_CTRL_WRITE, [int(val)])

        # 8. LED 알람 비트
        self._send_cmd(s_id, CMD_LED_ERROR_READ)
        res = self._read_res()
        if res:
            val = self.get_input("LED 에러 알람 비트", res[0])
            if val is not None: self._send_cmd(s_id, CMD_LED_ERROR_WRITE, [int(val)])

        print("\n--- 설정 편집 완료 ---")

    def close(self):
        self.ser.close()

if __name__ == "__main__":
    editor = LX16AEditor(SERIAL_PORT, BAUD_RATE)
    try:
        s_id_input = input("편집할 서보 ID: ").strip()
        if s_id_input: editor.start_edit(int(s_id_input))
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
    finally:
        editor.close()
